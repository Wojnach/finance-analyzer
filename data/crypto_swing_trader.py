"""Crypto swing trader — autonomous BUY/SELL logic for BTC + ETH warrants.

Mirrors the structure of `data/metals_swing_trader.py` but compact: only the
core swing-decision logic is implemented here. Avanza-side execution is
delegated to an injected ``executor`` (None in DRY_RUN, real implementation
plugged in once XBT-TRACKER / ETH-TRACKER metadata has been probed live).

Public API:
    trader = CryptoSwingTrader(page=None, executor=None)  # DRY_RUN
    trader.evaluate_and_execute(prices, signal_data)

Where:
    prices       — dict mapping "BTC-USD" / "ETH-USD" -> latest underlying USD
    signal_data  — dict from Layer 1 signal_engine, keyed by ticker

The trader is intentionally side-effect-light when DRY_RUN=True: it logs
decisions to ``data/crypto_swing_decisions.jsonl`` and updates state, but
never touches Avanza. This is the same pattern metals_loop used for its
first ~3 weeks before being flipped live.

Design notes:
- Entry gates mirror metals: MIN_BUY_VOTERS=3, MIN_BUY_CONFIDENCE=0.60,
  RSI in [35, 68], MACD-improving≥2 cycles, regime-confirmed, signal-
  persistence≥2, MACD-decay-ratio, RSI-slope, stale-signal rejection.
- Exit gates mirror metals: take-profit 4% underlying / 8% warrant, hard
  stop -3% underlying, signal-reversal exit on SELL consensus, momentum
  exit after 5min hold, max-hold 72h safety net.
- Crypto is 24/7 — there is NO EOD forced exit (metals has one tied to US
  session close).
"""
from __future__ import annotations

import datetime
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from data import crypto_swing_config as cfg
from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json

logger = logging.getLogger("crypto_swing_trader")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "positions": {},          # pos_id -> position dict
        "last_buy_ts": {},        # ticker -> ISO timestamp
        "consecutive_losses": 0,
        "session_started": _now_iso(),
        "last_cycle_ts": "",
        "cash_sek": cfg.INITIAL_BUDGET_SEK,
        "cycles_completed": 0,
    }


def _load_state() -> dict[str, Any]:
    state = load_json(cfg.STATE_FILE) or {}
    if not state:
        return _default_state()
    # Backfill any missing keys (forward-compat with new state fields)
    default = _default_state()
    for k, v in default.items():
        state.setdefault(k, v)
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["last_cycle_ts"] = _now_iso()
    try:
        atomic_write_json(cfg.STATE_FILE, state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_state failed: %s", exc)


def _log_decision(decision: dict[str, Any]) -> None:
    decision.setdefault("ts", _now_iso())
    try:
        atomic_append_jsonl(cfg.DECISIONS_LOG, decision)
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_decision failed: %s", exc)


def _log_trade(trade: dict[str, Any]) -> None:
    trade.setdefault("ts", _now_iso())
    try:
        atomic_append_jsonl(cfg.TRADES_LOG, trade)
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_trade failed: %s", exc)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _signal_age_seconds(signal_data: dict[str, Any]) -> float | None:
    """Return age in seconds based on signal_data['timestamp'].

    None if absent/unparseable. Caller decides how to handle.
    """
    ts_str = signal_data.get("timestamp") or signal_data.get("ts")
    if not ts_str:
        return None
    try:
        ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return (_now_utc() - ts).total_seconds()


def _extract_action(sig: dict[str, Any]) -> str:
    """Read a per-ticker signal entry's recommended action."""
    if not sig:
        return "HOLD"
    for k in ("recommendation", "action", "consensus", "signal"):
        v = sig.get(k)
        if isinstance(v, str) and v.upper() in ("BUY", "SELL", "HOLD"):
            return v.upper()
    return "HOLD"


def _extract_confidence(sig: dict[str, Any]) -> float:
    if not sig:
        return 0.0
    for k in ("calibrated_confidence", "confidence"):
        v = sig.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _extract_voters(sig: dict[str, Any]) -> tuple[int, int]:
    """Return (buy_voters, sell_voters) from the signal entry."""
    buy = sig.get("buy_voters") or sig.get("buy_count") or 0
    sell = sig.get("sell_voters") or sig.get("sell_count") or 0
    try:
        return int(buy), int(sell)
    except (TypeError, ValueError):
        return 0, 0


def _extract_indicator(sig: dict[str, Any], name: str) -> float | None:
    """Read an indicator value (RSI, MACD) — accept several common keys."""
    if not sig:
        return None
    indicators = sig.get("indicators") or {}
    if name in indicators and indicators[name] is not None:
        try:
            return float(indicators[name])
        except (TypeError, ValueError):
            pass
    if name in sig and sig[name] is not None:
        try:
            return float(sig[name])
        except (TypeError, ValueError):
            pass
    return None


def _extract_regime(sig: dict[str, Any]) -> str:
    return (sig.get("regime") or sig.get("market_regime") or "unknown").lower()


# ---------------------------------------------------------------------------
# Core trader class
# ---------------------------------------------------------------------------
class CryptoSwingTrader:
    """Compact swing-trader engine for BTC + ETH warrants.

    Constructor:
        page     — Playwright page (None in DRY_RUN; required for live trading)
        executor — optional callable(action, **kwargs) for live order placement.
                   Signature: executor("buy", warrant=..., units=..., price=...) etc.
                   When None, all order-placement is logged-only.
    """

    def __init__(self, page=None, executor: Callable | None = None) -> None:
        self.page = page
        self.executor = executor
        self.state = _load_state()
        self.warrant_catalog: dict[str, dict] = {}
        self._refresh_warrant_catalog()

        # In-memory rolling histories (per-ticker, used by entry gates)
        # Persisted to state when written.
        self._confidence_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=cfg.CONFIDENCE_HISTORY_MAX))
        self._rsi_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=cfg.RSI_HISTORY_MAX))
        self._macd_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=cfg.MACD_DECAY_PEAK_LOOKBACK))
        self._regime_history: dict[str, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=10))
        self._fast_tick_ref: dict[str, dict] = {}  # ticker -> {ref_price, ref_ts}

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------
    def _refresh_warrant_catalog(self) -> None:
        try:
            from data.crypto_warrant_refresh import load_catalog_or_fetch
            self.warrant_catalog = load_catalog_or_fetch(self.page) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("warrant catalog refresh failed: %s — using fallback", exc)
            self.warrant_catalog = dict(cfg.WARRANT_CATALOG_FALLBACK)

    def _select_warrant(self, underlying: str,
                        direction: str) -> dict | None:
        """Pick the best warrant matching (underlying, direction).

        Selection priority:
          1. Spread <= MIN_SPREAD_PCT, leverage closest to TARGET_LEVERAGE.
          2. Else: fallback tracker (1x).
          3. Else: None — caller must skip the trade.
        """
        candidates = [
            w for w in self.warrant_catalog.values()
            if w.get("underlying") == underlying
               and w.get("direction") == direction
        ]
        if not candidates:
            return None

        # Hard filters
        viable = []
        for w in candidates:
            spread = w.get("spread_pct")
            if spread is not None and spread > cfg.MIN_SPREAD_PCT:
                continue
            lev = float(w.get("leverage") or 0)
            if lev < cfg.MIN_ACCEPTABLE_LEVERAGE:
                continue
            viable.append(w)
        if not viable:
            # Fallback to any 1x tracker if the leverage filter excluded everything
            viable = [w for w in candidates if (w.get("leverage") or 1.0) <= 1.0]
        if not viable:
            return None

        # Rank by closeness to target leverage
        viable.sort(key=lambda w: abs(float(w.get("leverage") or 1.0)
                                       - cfg.TARGET_LEVERAGE))
        return viable[0]

    # ------------------------------------------------------------------
    # Entry gates
    # ------------------------------------------------------------------
    def _update_history(self, ticker: str, sig: dict[str, Any]) -> None:
        c = _extract_confidence(sig)
        if c is not None:
            self._confidence_history[ticker].append(c)
        rsi = _extract_indicator(sig, "rsi")
        if rsi is not None:
            self._rsi_history[ticker].append(rsi)
        macd = _extract_indicator(sig, "macd")
        if macd is not None:
            self._macd_history[ticker].append(macd)
        regime = _extract_regime(sig)
        action = _extract_action(sig)
        self._regime_history[ticker].append((regime, action))

    def _entry_gate_persistence(self, ticker: str) -> tuple[bool, str]:
        hist = self._confidence_history[ticker]
        if len(hist) < cfg.SIGNAL_PERSISTENCE_CHECKS:
            return False, "insufficient persistence history"
        recent = list(hist)[-cfg.SIGNAL_PERSISTENCE_CHECKS:]
        if all(c >= cfg.MIN_BUY_CONFIDENCE for c in recent):
            return True, "ok"
        return False, f"confidence not persistent (recent={recent})"

    def _entry_gate_macd_decay(self, ticker: str) -> tuple[bool, str]:
        hist = list(self._macd_history[ticker])
        if not hist:
            return True, "macd history empty (skip gate)"
        peak = max(abs(m) for m in hist) or 0
        current = abs(hist[-1])
        if peak <= 0:
            return True, "macd peak zero"
        ratio = current / peak
        if ratio >= cfg.MACD_DECAY_MIN_RATIO:
            return True, f"macd ratio {ratio:.2f} >= {cfg.MACD_DECAY_MIN_RATIO}"
        return False, f"macd decayed: ratio {ratio:.2f} < {cfg.MACD_DECAY_MIN_RATIO}"

    def _entry_gate_rsi_slope(self, ticker: str) -> tuple[bool, str]:
        hist = list(self._rsi_history[ticker])
        if len(hist) < 2:
            return True, "rsi history too short (skip gate)"
        # Accept if RSI not falling over RSI_SLOPE_LOOKBACK_CHECKS
        slope_n = min(cfg.RSI_SLOPE_LOOKBACK_CHECKS, len(hist) - 1)
        if hist[-1] >= hist[-1 - slope_n]:
            return True, f"rsi rising (slope {slope_n}c)"
        # Or if RSI dipped below threshold in last RSI_DIP_LOOKBACK_CHECKS
        dip_n = min(cfg.RSI_DIP_LOOKBACK_CHECKS, len(hist))
        if min(hist[-dip_n:]) <= cfg.RSI_DIP_BELOW_LEVEL:
            return True, f"rsi dipped <= {cfg.RSI_DIP_BELOW_LEVEL} recently"
        return False, "rsi falling and no recent dip"

    def _entry_gate_regime_confirm(self, ticker: str, action: str,
                                   regime: str) -> tuple[bool, str]:
        hist = list(self._regime_history[ticker])
        if len(hist) < cfg.REGIME_CONFIRM_CHECKS:
            return False, "regime history too short"
        recent = hist[-cfg.REGIME_CONFIRM_CHECKS:]
        if all(r == regime and a == action for (r, a) in recent):
            return True, "regime confirmed"
        return False, f"regime not confirmed (recent={recent})"

    def _evaluate_entry(self, ticker: str, sig: dict[str, Any],
                        is_momentum: bool) -> tuple[bool, str, dict[str, Any]]:
        """Returns (allow, reason, ctx).

        Tightening gates that mirror the post-mortem chain in metals:
        2026-04-17 / 18 / 21. Momentum-override path skips persistence/MACD-
        decay/RSI-slope (those are designed for slow grind setups, not bursts).
        """
        ctx: dict[str, Any] = {"ticker": ticker, "is_momentum": is_momentum}

        # Stale-signal gate (always runs first; same in std + momentum paths)
        age = _signal_age_seconds(sig)
        if age is not None and age > cfg.MAX_SIGNAL_AGE_SEC:
            return False, f"stale signal age={age:.0f}s", ctx

        action = _extract_action(sig)
        if action != "BUY":
            return False, f"action != BUY (got {action})", ctx

        conf = _extract_confidence(sig)
        buy_voters, _ = _extract_voters(sig)
        rsi = _extract_indicator(sig, "rsi")
        regime = _extract_regime(sig)
        ctx.update({"confidence": conf, "buy_voters": buy_voters, "rsi": rsi,
                    "regime": regime})

        # Choose thresholds based on momentum override
        min_conf = (cfg.MOMENTUM_MIN_BUY_CONFIDENCE if is_momentum
                    else cfg.MIN_BUY_CONFIDENCE)
        min_voters = (cfg.MOMENTUM_MIN_BUY_VOTERS if is_momentum
                      else cfg.MIN_BUY_VOTERS)

        if conf < min_conf:
            return False, f"confidence {conf:.2f} < {min_conf}", ctx
        if buy_voters < min_voters:
            return False, f"buy_voters {buy_voters} < {min_voters}", ctx

        # RSI bounds (always)
        if rsi is not None and not (cfg.RSI_ENTRY_LOW <= rsi <= cfg.RSI_ENTRY_HIGH):
            return False, f"rsi {rsi:.1f} outside [{cfg.RSI_ENTRY_LOW},{cfg.RSI_ENTRY_HIGH}]", ctx

        if is_momentum:
            return True, "momentum override (relaxed gates)", ctx

        # Standard-path gates
        ok, why = self._entry_gate_persistence(ticker)
        if not ok:
            return False, f"persistence: {why}", ctx
        ok, why = self._entry_gate_macd_decay(ticker)
        if not ok:
            return False, f"macd_decay: {why}", ctx
        ok, why = self._entry_gate_rsi_slope(ticker)
        if not ok:
            return False, f"rsi_slope: {why}", ctx
        ok, why = self._entry_gate_regime_confirm(ticker, "BUY", regime)
        if not ok:
            return False, f"regime: {why}", ctx

        return True, "ok", ctx

    # ------------------------------------------------------------------
    # Exit gates
    # ------------------------------------------------------------------
    def _evaluate_exit(self, pos: dict[str, Any], current_underlying: float,
                      current_warrant_bid: float | None,
                      sig: dict[str, Any]) -> tuple[bool, str]:
        """Returns (should_sell, reason)."""
        entry_underlying = pos.get("entry_underlying_price") or 0.0
        if entry_underlying <= 0:
            return False, "no entry underlying price"
        direction = pos.get("direction", "LONG")
        sign = 1 if direction == "LONG" else -1
        underlying_pct = sign * (current_underlying / entry_underlying - 1.0) * 100.0
        pos["last_underlying_pct"] = round(underlying_pct, 3)

        # 1. Hard stop
        if underlying_pct <= -cfg.HARD_STOP_UNDERLYING_PCT:
            return True, f"HARD_STOP underlying {underlying_pct:.2f}%"

        # 2. Take profit (underlying-based)
        if underlying_pct >= cfg.TAKE_PROFIT_UNDERLYING_PCT:
            return True, f"TAKE_PROFIT underlying {underlying_pct:.2f}%"

        # 3. Warrant-side TP
        entry_warrant = pos.get("entry_warrant_bid")
        if entry_warrant and current_warrant_bid:
            warrant_pct = (current_warrant_bid / entry_warrant - 1.0) * 100.0
            pos["last_warrant_pct"] = round(warrant_pct, 3)
            if warrant_pct >= cfg.WARRANT_TAKE_PROFIT_PCT:
                return True, f"WARRANT_TP {warrant_pct:.2f}%"

            # Trailing on warrant peak
            peak = pos.get("peak_warrant_bid", entry_warrant)
            if current_warrant_bid > peak:
                pos["peak_warrant_bid"] = current_warrant_bid
                peak = current_warrant_bid
            peak_pct_from_entry = (peak / entry_warrant - 1.0) * 100.0
            if peak_pct_from_entry >= cfg.WARRANT_TRAILING_START_PCT:
                drawdown = (peak - current_warrant_bid) / peak * 100.0
                if drawdown >= cfg.WARRANT_TRAILING_DISTANCE_PCT:
                    return True, f"WARRANT_TRAIL drawdown {drawdown:.2f}%"

        # 4. Trailing on underlying peak
        peak_und = pos.get("peak_underlying_price", entry_underlying)
        if direction == "LONG":
            if current_underlying > peak_und:
                pos["peak_underlying_price"] = current_underlying
                peak_und = current_underlying
        else:
            if current_underlying < peak_und:
                pos["peak_underlying_price"] = current_underlying
                peak_und = current_underlying

        peak_pct_from_entry = sign * (peak_und / entry_underlying - 1.0) * 100.0
        if peak_pct_from_entry >= cfg.TRAILING_START_PCT:
            drawdown = sign * (peak_und - current_underlying) / peak_und * 100.0
            if drawdown >= cfg.TRAILING_DISTANCE_PCT:
                return True, f"TRAIL_UND drawdown {drawdown:.2f}%"

        # 5. Signal reversal
        if cfg.SIGNAL_REVERSAL_EXIT:
            action = _extract_action(sig)
            opposite = "SELL" if direction == "LONG" else "BUY"
            buy_voters, sell_voters = _extract_voters(sig)
            opposing_voters = sell_voters if direction == "LONG" else buy_voters
            if action == opposite and opposing_voters >= cfg.MIN_BUY_VOTERS:
                return True, f"SIGNAL_REVERSAL {action} voters={opposing_voters}"

        # 6. Max hold safety net
        entry_ts = pos.get("entry_ts")
        if entry_ts:
            try:
                ts = datetime.datetime.fromisoformat(
                    entry_ts.replace("Z", "+00:00"))
                age_h = (_now_utc() - ts).total_seconds() / 3600.0
                if age_h >= cfg.MAX_HOLD_HOURS:
                    return True, f"MAX_HOLD {age_h:.1f}h"
            except (ValueError, AttributeError, TypeError):
                pass

        return False, "no exit"

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    def _cooldown_cleared(self, ticker: str) -> bool:
        last = self.state.get("last_buy_ts", {}).get(ticker)
        if not last:
            return True
        try:
            ts = datetime.datetime.fromisoformat(last.replace("Z", "+00:00"))
            age_min = (_now_utc() - ts).total_seconds() / 60.0
        except (ValueError, AttributeError, TypeError):
            return True
        consec = self.state.get("consecutive_losses", 0)
        mult = cfg.LOSS_ESCALATION.get(min(consec, 3), 8)
        return age_min >= cfg.BUY_COOLDOWN_MINUTES * mult

    # ------------------------------------------------------------------
    # Order placement (DRY_RUN-aware)
    # ------------------------------------------------------------------
    def _place_buy(self, ticker: str, warrant: dict, signal_ctx: dict,
                   underlying_price: float) -> dict:
        # Position sizing
        cash = float(self.state.get("cash_sek") or cfg.INITIAL_BUDGET_SEK)
        budget = cash * (cfg.POSITION_SIZE_PCT / 100.0)
        if budget < cfg.MIN_TRADE_SEK:
            return {"executed": False, "reason": f"budget {budget:.0f} < min {cfg.MIN_TRADE_SEK}"}

        warrant_ask = warrant.get("ask") or warrant.get("last") or 1.0
        units = max(int(budget / warrant_ask), 0)
        if units <= 0:
            return {"executed": False, "reason": "units=0 after sizing"}

        pos_id = f"{ticker}_{int(time.time())}"
        pos = {
            "pos_id": pos_id,
            "ticker": ticker,
            "warrant_key": warrant.get("name"),
            "ob_id": warrant.get("ob_id"),
            "direction": warrant.get("direction", "LONG"),
            "leverage": float(warrant.get("leverage") or 1.0),
            "units": units,
            "entry_warrant_bid": warrant_ask,
            "entry_underlying_price": underlying_price,
            "peak_underlying_price": underlying_price,
            "peak_warrant_bid": warrant_ask,
            "entry_ts": _now_iso(),
            "signal_context": signal_ctx,
        }

        if cfg.DRY_RUN or self.executor is None:
            decision = {"action": "BUY_DRY_RUN", "pos": pos, "warrant": warrant,
                        "underlying_price": underlying_price}
            _log_decision(decision)
            self.state["positions"][pos_id] = pos
            self.state.setdefault("last_buy_ts", {})[ticker] = pos["entry_ts"]
            return {"executed": True, "dry_run": True, "pos_id": pos_id}

        # Live path
        try:
            res = self.executor("buy", warrant=warrant, units=units,
                                price=warrant_ask, pos=pos)
            if res and res.get("ok"):
                self.state["positions"][pos_id] = pos
                self.state.setdefault("last_buy_ts", {})[ticker] = pos["entry_ts"]
                _log_trade({"action": "BUY", "pos": pos, "result": res})
                return {"executed": True, "dry_run": False, "pos_id": pos_id, "result": res}
            return {"executed": False, "reason": "executor returned not-ok",
                    "result": res}
        except Exception as exc:  # noqa: BLE001
            return {"executed": False, "reason": f"executor exception: {exc}"}

    def _place_sell(self, pos_id: str, pos: dict[str, Any],
                    current_underlying: float, current_warrant_bid: float,
                    reason: str) -> dict:
        if cfg.DRY_RUN or self.executor is None:
            decision = {"action": "SELL_DRY_RUN", "pos_id": pos_id,
                        "underlying_price": current_underlying,
                        "warrant_bid": current_warrant_bid, "reason": reason}
            _log_decision(decision)
            entry = pos.get("entry_underlying_price") or 0
            pnl_pct = (current_underlying / entry - 1.0) * 100.0 if entry else 0
            sign = 1 if pos.get("direction") == "LONG" else -1
            pnl_pct *= sign
            _log_trade({"action": "SELL", "pos_id": pos_id, "reason": reason,
                        "underlying_pct": round(pnl_pct, 3),
                        "dry_run": True, "exit_underlying": current_underlying,
                        "exit_warrant_bid": current_warrant_bid})
            self.state["positions"].pop(pos_id, None)
            if pnl_pct < 0:
                self.state["consecutive_losses"] = self.state.get(
                    "consecutive_losses", 0) + 1
            else:
                self.state["consecutive_losses"] = 0
            return {"executed": True, "dry_run": True, "pnl_pct": pnl_pct}

        try:
            res = self.executor("sell", pos=pos, price=current_warrant_bid,
                                reason=reason)
            if res and res.get("ok"):
                _log_trade({"action": "SELL", "pos_id": pos_id, "reason": reason,
                            "result": res})
                self.state["positions"].pop(pos_id, None)
                return {"executed": True, "dry_run": False, "result": res}
            return {"executed": False, "reason": "executor returned not-ok",
                    "result": res}
        except Exception as exc:  # noqa: BLE001
            return {"executed": False, "reason": f"executor exception: {exc}"}

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------
    def evaluate_and_execute(self, prices: dict[str, float],
                             signal_data: dict[str, Any]) -> dict[str, Any]:
        """One cycle of swing-trader logic.

        Args:
            prices: {"BTC-USD": 105000.0, "ETH-USD": 3500.0}
            signal_data: full signal_engine output dict (per-ticker entries
                live under signal_data["per_ticker"][ticker] OR
                signal_data["tickers"][ticker]).

        Returns:
            Summary dict with counts of decisions taken.
        """
        per_ticker = (signal_data.get("per_ticker")
                      or signal_data.get("tickers")
                      or {})
        per_ticker = {k: v for k, v in per_ticker.items()
                      if k in cfg.INSTRUMENTS}

        # 1. Update histories from this cycle's signals
        for ticker, sig in per_ticker.items():
            self._update_history(ticker, sig)

        actions: list[dict] = []

        # 2. Evaluate exits FIRST (don't enter on a ticker we should be exiting)
        positions = list(self.state.get("positions", {}).items())
        for pos_id, pos in positions:
            ticker = pos.get("ticker")
            current_und = prices.get(ticker)
            if current_und is None:
                continue
            sig = per_ticker.get(ticker, {})
            warrant_bid = (sig.get("warrant_bid")
                           or pos.get("entry_warrant_bid"))
            should_exit, reason = self._evaluate_exit(
                pos, float(current_und), float(warrant_bid) if warrant_bid else None, sig)
            if should_exit:
                res = self._place_sell(pos_id, pos, float(current_und),
                                       float(warrant_bid or 0), reason)
                actions.append({"type": "exit", "pos_id": pos_id, "reason": reason,
                                "result": res})

        # 3. Evaluate entries
        active_tickers = {p.get("ticker") for p in self.state.get("positions", {}).values()}
        max_concurrent = cfg.MAX_CONCURRENT
        room = max_concurrent - len(active_tickers)

        for ticker, sig in per_ticker.items():
            if room <= 0:
                break
            if ticker in active_tickers:
                continue
            if not self._cooldown_cleared(ticker):
                continue

            # Momentum-override check
            momentum = self._has_fresh_momentum_candidate(ticker)
            allow, reason, ctx = self._evaluate_entry(ticker, sig,
                                                       is_momentum=momentum)
            if not allow:
                continue

            warrant = self._select_warrant(
                underlying=ticker,
                direction="LONG",  # SHORT side disabled in v1 (momentum doesn't write SHORT)
            )
            if not warrant:
                actions.append({"type": "skip", "ticker": ticker,
                                "reason": "no warrant available"})
                continue

            res = self._place_buy(ticker, warrant, ctx,
                                  float(prices.get(ticker, 0)))
            actions.append({"type": "entry", "ticker": ticker,
                            "warrant": warrant.get("name"),
                            "ctx": ctx, "result": res})
            if res.get("executed"):
                room -= 1
                active_tickers.add(ticker)

        self.state["cycles_completed"] = self.state.get("cycles_completed", 0) + 1
        _save_state(self.state)
        return {"actions": actions,
                "n_positions": len(self.state.get("positions", {}))}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _has_fresh_momentum_candidate(self, ticker: str) -> bool:
        """Read MOMENTUM_STATE_FILE for a fresh candidate matching ticker."""
        if not cfg.MOMENTUM_ENTRY_ENABLED:
            return False
        state = load_json(cfg.MOMENTUM_STATE_FILE) or {}
        cand = state.get(ticker)
        if not cand:
            return False
        ts_str = cand.get("ts")
        if not ts_str:
            return False
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (_now_utc() - ts).total_seconds()
        except (ValueError, AttributeError, TypeError):
            return False
        return age <= cfg.MOMENTUM_CANDIDATE_TTL_SEC


__all__ = [
    "CryptoSwingTrader",
    "_load_state", "_save_state", "_default_state",
    "_extract_action", "_extract_confidence", "_extract_voters",
    "_extract_indicator", "_extract_regime", "_signal_age_seconds",
]
