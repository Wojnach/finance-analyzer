"""Autonomous metals swing trader — rule-based warrant BUY/SELL on Avanza.

Integrates into metals_loop.py. Called every price cycle. Shares Playwright session.
No LLM dependency — decisions in <1s based on signal consensus + price rules.

Usage from metals_loop.py:
    from metals_swing_trader import SwingTrader
    trader = SwingTrader(page)
    trader.evaluate_and_execute(prices, signal_data)
"""

import datetime
import json
import time

import requests
from metals_swing_config import (
    ACCOUNT_ID,
    BUY_COOLDOWN_MINUTES,
    DECISIONS_LOG,
    DRY_RUN,
    EOD_EXIT_MINUTES_BEFORE,
    HARD_STOP_UNDERLYING_PCT,
    INITIAL_BUDGET_SEK,
    LOSS_ESCALATION,
    MACD_IMPROVING_CHECKS,
    MAX_CONCURRENT,
    MAX_HOLD_HOURS,
    MIN_ACCEPTABLE_LEVERAGE,
    MIN_BARRIER_DISTANCE_PCT,
    MIN_BUY_CONFIDENCE,
    MIN_BUY_TF_RATIO,
    MIN_BUY_VOTERS,
    MIN_SPREAD_PCT,
    MIN_TRADE_SEK,
    POSITION_SIZE_PCT,
    REGIME_CONFIRM_CHECKS,
    RSI_ENTRY_HIGH,
    RSI_ENTRY_LOW,
    SIGNAL_REVERSAL_EXIT,
    STATE_FILE,
    STOP_LOSS_UNDERLYING_PCT,
    STOP_LOSS_VALID_DAYS,
    TAKE_PROFIT_UNDERLYING_PCT,
    TARGET_LEVERAGE,
    TELEGRAM_SUMMARY_INTERVAL,
    TRADES_LOG,
    TRAILING_DISTANCE_PCT,
    TRAILING_START_PCT,
)
from metals_swing_config import (
    WARRANT_CATALOG as STATIC_WARRANT_CATALOG,
)

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json

# Dynamic warrant catalog refresher — replaces the stale hardcoded list.
try:
    from metals_warrant_refresh import load_catalog_or_fetch
    _REFRESHER_AVAILABLE = True
except ImportError:
    _REFRESHER_AVAILABLE = False

from portfolio.avanza_control import (
    delete_stop_loss,
    fetch_account_cash,
    fetch_price,
    place_order,
    place_stop_loss,
)

SEND_PERIODIC_SUMMARY = False


def _log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [SWING] {msg}", flush=True)


def _now_utc():
    return datetime.datetime.now(datetime.UTC)


def _cet_hour():
    """Get current CET hour as float. Uses zoneinfo (DST-safe)."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo("Europe/Stockholm"))
        return now.hour + now.minute / 60
    except ImportError:
        now = datetime.datetime.now(datetime.UTC)
        return ((now.hour + 1) % 24) + now.minute / 60


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state():
    """Load swing trader state from disk."""
    result = load_json(STATE_FILE)
    if result is not None:
        return result
    return _default_state()


def _default_state():
    return {
        "cash_sek": 0,
        "positions": {},
        "consecutive_losses": 0,
        "last_buy_ts": None,
        "total_trades": 0,
        "total_pnl_sek": 0,
        "session_trades": 0,
        "macd_history": {},
    }


def _save_state(state):
    try:
        atomic_write_json(STATE_FILE, state, indent=2, ensure_ascii=False)
    except Exception as e:
        _log(f"State save error: {e}")


def _delete_stop_loss(page, stop_id):
    """Delete a stop-loss order by ID."""
    success, _ = delete_stop_loss(page, ACCOUNT_ID, stop_id)
    return success


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

_tg_config = None

def _send_telegram(msg):
    global _tg_config
    if _tg_config is None:
        try:
            with open("config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            _tg_config = {
                "token": cfg["telegram"]["token"],
                "chat_id": cfg["telegram"]["chat_id"],
            }
        except Exception:
            _tg_config = {}

    if not _tg_config.get("token"):
        _log("Telegram not configured")
        return

    # Check mute_all from config
    try:
        with open("config.json", encoding="utf-8") as f:
            _mute = json.load(f).get("telegram", {}).get("mute_all", False)
        if _mute:
            _log(f"[TG muted] {msg[:80]}")
            return
    except Exception:
        pass

    try:
        requests.post(
            f"https://api.telegram.org/bot{_tg_config['token']}/sendMessage",
            json={"chat_id": _tg_config["chat_id"], "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        _log(f"Telegram error: {e}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_decision(decision):
    try:
        atomic_append_jsonl(DECISIONS_LOG, decision)
    except Exception as e:
        _log(f"Decision log error: {e}")


def _log_trade(trade):
    try:
        atomic_append_jsonl(TRADES_LOG, trade)
    except Exception as e:
        _log(f"Trade log error: {e}")


# ---------------------------------------------------------------------------
# SwingTrader class
# ---------------------------------------------------------------------------

class SwingTrader:
    """Autonomous rule-based swing trader for metals warrants on Avanza."""

    def __init__(self, page):
        self.page = page
        self.state = _load_state()
        self.check_count = 0

        # Track consecutive signal history per ticker (action, regime) — used
        # to reject single-check regime flips from trending-down → ranging BUY.
        # Keyed by ticker; value is a list of (action, regime) tuples.
        self.regime_history: dict[str, list[tuple[str, str]]] = {}

        # Load dynamic warrant catalog (live refresh from Avanza). Falls back
        # to the static hardcoded catalog if the refresh fails entirely.
        self.warrant_catalog = self._load_warrant_catalog()

        # Sync cash from Avanza on init
        self._sync_cash()
        _log(f"SwingTrader init: cash={self.state['cash_sek']:.0f} SEK, "
             f"positions={len(self.state['positions'])}, "
             f"catalog={len(self.warrant_catalog)} warrants, "
             f"DRY_RUN={DRY_RUN}")

    def _load_warrant_catalog(self) -> dict:
        """Load dynamic warrant catalog, falling back to static config.

        The refresher fetches live data from Avanza and filters dead/knocked
        out warrants. If it returns an empty dict (total network failure),
        we fall back to the static config catalog so the trader can still
        operate (with whatever stale entries it contains).
        """
        if not _REFRESHER_AVAILABLE:
            _log("Refresher unavailable — using static catalog")
            return dict(STATIC_WARRANT_CATALOG)
        try:
            catalog = load_catalog_or_fetch()
        except Exception as e:  # noqa: BLE001
            _log(f"Catalog refresh raised {type(e).__name__}: {e} — using static")
            return dict(STATIC_WARRANT_CATALOG)
        if not catalog:
            _log("Refresher returned empty — using static catalog")
            return dict(STATIC_WARRANT_CATALOG)
        return catalog

    def _sync_cash(self):
        """Fetch real ISK buying power from Avanza and update state.

        Falls back to INITIAL_BUDGET_SEK when the API fails and no saved
        balance exists (e.g. first startup with empty state file).
        """
        acc = fetch_account_cash(self.page, ACCOUNT_ID)
        if acc and acc.get("buying_power") is not None:
            self.state["cash_sek"] = float(acc["buying_power"])
            _save_state(self.state)
            _log(f"Cash synced: {self.state['cash_sek']:.0f} SEK")
        else:
            if self.state["cash_sek"] == 0:
                self.state["cash_sek"] = float(INITIAL_BUDGET_SEK)
                _save_state(self.state)
                _log(f"Cash sync failed, using configured budget: {self.state['cash_sek']:.0f} SEK")
            else:
                _log(f"Cash sync failed, using saved: {self.state['cash_sek']:.0f} SEK")

    def evaluate_and_execute(self, prices, signal_data):
        """Main entry point — called every loop cycle during market hours.

        Args:
            prices: dict from metals_loop price fetch (keyed by position name)
            signal_data: dict from read_signal_data() with XAG-USD/XAU-USD signals
        """
        self.check_count += 1

        # Re-sync cash from Avanza every 30 checks (~30 min) to catch manual deposits
        # and recover from initial API failures.
        if self.check_count % 30 == 0:
            self._sync_cash()

        # Periodic catalog refresh (every 360 checks ≈ 6h with 60s loop)
        if self.check_count % 360 == 0:
            try:
                fresh = self._load_warrant_catalog()
                if fresh:
                    self.warrant_catalog = fresh
                    _log(f"Catalog auto-refreshed: {len(self.warrant_catalog)} warrants")
            except Exception as e:  # noqa: BLE001
                _log(f"Catalog auto-refresh failed: {e}")

        # Update regime history BEFORE entry checks so the gate sees the latest state.
        self._update_regime_history(signal_data)

        # Check exits first (protect capital)
        self._check_exits(prices, signal_data)

        # Then check entries
        self._check_entries(prices, signal_data)

        # Track MACD history for improving-checks requirement
        self._update_macd_history(signal_data)

        # Periodic Telegram summary (when SEND_PERIODIC_SUMMARY enabled).
        # Restored after f6b491c accidentally dropped this call site.
        if SEND_PERIODIC_SUMMARY and self.check_count % TELEGRAM_SUMMARY_INTERVAL == 0:
            self._send_summary(signal_data)

    def _update_regime_history(self, signal_data):
        """Append (action, regime) snapshot per ticker, capped at last 10 entries."""
        if not signal_data:
            return
        for ticker in ("XAG-USD", "XAU-USD"):
            sig = signal_data.get(ticker)
            if not sig:
                continue
            entry = (sig.get("action", "HOLD"), sig.get("regime", "unknown"))
            hist = self.regime_history.setdefault(ticker, [])
            hist.append(entry)
            # Keep last 10 to bound memory
            if len(hist) > 10:
                self.regime_history[ticker] = hist[-10:]

    def _regime_confirmed(self, ticker: str, action: str, regime: str) -> bool:
        """Return True if the (action, regime) pair held for REGIME_CONFIRM_CHECKS in a row.

        Rejects single-check flips like trending-down → ranging BUY in one tick.
        """
        hist = self.regime_history.get(ticker, [])
        if len(hist) < REGIME_CONFIRM_CHECKS:
            return False
        recent = hist[-REGIME_CONFIRM_CHECKS:]
        return all(a == action and r == regime for a, r in recent)

    # -------------------------------------------------------------------
    # Entry logic
    # -------------------------------------------------------------------

    def _check_entries(self, prices, signal_data):
        """Scan for BUY opportunities on XAG-USD and XAU-USD."""
        if not signal_data:
            return

        active_count = len(self.state["positions"])
        if active_count >= MAX_CONCURRENT:
            return

        # Check cooldown
        if not self._cooldown_cleared():
            return

        for underlying_ticker in ["XAG-USD", "XAU-USD"]:
            sig = signal_data.get(underlying_ticker)
            if not sig:
                continue

            # Already have a position in this underlying?
            if self._has_position(underlying_ticker):
                continue

            # Evaluate entry criteria
            entry_ok, reason = self._evaluate_entry(sig, underlying_ticker)
            if not entry_ok:
                decision = {
                    "ts": _now_utc().isoformat(),
                    "check": self.check_count,
                    "underlying": underlying_ticker,
                    "action": "SKIP_BUY",
                    "reason": reason,
                    "signal": _compact_signal(sig),
                }
                _log_decision(decision)
                continue

            # Select best warrant
            direction = "SHORT" if sig.get("action") == "SELL" else "LONG"
            warrant = self._select_warrant(underlying_ticker, direction)
            if not warrant:
                _log(f"No valid warrant for {underlying_ticker} {direction}")
                continue

            # Calculate position size
            cash = self.state["cash_sek"]
            alloc = cash * POSITION_SIZE_PCT / 100
            if alloc < MIN_TRADE_SEK:
                _log(f"Insufficient cash: {cash:.0f} SEK (need {MIN_TRADE_SEK})")
                continue

            ask_price = warrant["live_ask"]
            if ask_price <= 0:
                continue

            units = int(alloc / ask_price)
            if units < 1:
                continue

            total_cost = units * ask_price

            # Execute BUY
            self._execute_buy(warrant, units, ask_price, underlying_ticker, sig, total_cost)

    def _evaluate_entry(self, sig, ticker):
        """Check if signal data meets entry criteria. Returns (ok, reason).

        Gates today (post-SG-incident hardening):
        - action == BUY (SHORT entries gated until _check_exits is direction-aware)
        - confidence >= MIN_BUY_CONFIDENCE (user rule: no sub-60% trades)
        - majority_count > minority_count (not just majority_count >= MIN)
        - RSI in entry zone
        - MACD improving (LONG) or declining (SHORT) for N checks
        - regime stable for N consecutive checks (no single-check flips)

        TODO(short-side): SHORT entries are accepted by the catalog (115 warrants
        include SHORT MINIs) and direction-aware checks below are wired, but
        _check_exits still uses LONG-only math (positive und_change = profit,
        peak_underlying tracks max, etc). Enable SELL→SHORT here only after
        _check_exits, peak/trough tracking, and exit_optimizer all flip on the
        position's "direction" field. See codex review of f6b491c.
        """
        buy_count = sig.get("buy_count", 0)
        sell_count = sig.get("sell_count", 0)
        rsi = sig.get("rsi", 50)
        action = sig.get("action", "HOLD")
        confidence = float(sig.get("confidence", 0) or 0)

        # Direction check — LONG needs BUY. SHORT entries blocked at gate (see TODO above).
        if action == "BUY":
            majority = buy_count
            minority = sell_count
            direction = "LONG"
        elif action == "SELL":
            return False, f"action=SELL — SHORT entries gated (TODO: direction-aware exits)"
        else:
            return False, f"action={action} (need BUY consensus)"

        # User rule: no signal trades below 60% calibrated confidence
        if confidence < MIN_BUY_CONFIDENCE:
            return False, f"confidence {confidence:.2f} < {MIN_BUY_CONFIDENCE}"

        # Require strict majority, not just minimum voters. The 2026-04-09
        # incident fired BUY with buy=3 sell=4 because only MIN_BUY_VOTERS
        # was checked. Now we also require majority > minority.
        if majority <= minority:
            return False, f"no strict majority: {direction}={majority} vs other={minority}"

        # Minimum voter threshold (belt-and-suspenders alongside majority check)
        if majority < MIN_BUY_VOTERS:
            return False, f"{direction}_count={majority} < {MIN_BUY_VOTERS}"

        # Timeframe alignment — count timeframes voting in our direction.
        tf = sig.get("timeframes", {})
        if tf:
            target_label = "BUY" if direction == "LONG" else "SELL"
            aligned_tfs = sum(1 for v in tf.values() if v == target_label)
            total_tfs = len(tf)
            if total_tfs > 0 and aligned_tfs / total_tfs < MIN_BUY_TF_RATIO:
                return False, f"TF alignment {aligned_tfs}/{total_tfs} {target_label} < {MIN_BUY_TF_RATIO:.0%}"

        # RSI zone (LONG: not oversold, not overbought; SHORT: mirrored)
        if direction == "LONG":
            if rsi < RSI_ENTRY_LOW:
                return False, f"RSI {rsi:.1f} < {RSI_ENTRY_LOW} (oversold, wait for bounce)"
            if rsi > RSI_ENTRY_HIGH:
                return False, f"RSI {rsi:.1f} > {RSI_ENTRY_HIGH} (overbought)"
        else:  # SHORT — mirror the zone (high RSI = overbought = good short entry)
            if rsi > 100 - RSI_ENTRY_LOW:
                return False, f"RSI {rsi:.1f} > {100 - RSI_ENTRY_LOW} (overbought reversal, wait)"
            if rsi < 100 - RSI_ENTRY_HIGH:
                return False, f"RSI {rsi:.1f} < {100 - RSI_ENTRY_HIGH} (already oversold)"

        # MACD trend — LONG needs MACD rising, SHORT needs MACD falling.
        macd_hist = self.state.get("macd_history", {}).get(ticker, [])
        if len(macd_hist) >= MACD_IMPROVING_CHECKS:
            recent = macd_hist[-MACD_IMPROVING_CHECKS:]
            if direction == "LONG":
                trending = all(recent[i] > recent[i - 1] for i in range(1, len(recent)))
                if not trending:
                    return False, f"MACD not improving for {MACD_IMPROVING_CHECKS} checks"
            else:  # SHORT
                trending = all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
                if not trending:
                    return False, f"MACD not declining for {MACD_IMPROVING_CHECKS} checks"
        # If not enough MACD history yet, skip this check (allow entry)

        # Regime confirmation — require N consecutive (action, regime) checks.
        # The 2026-04-09 incident fired BUY in one cycle after 20+ trending-down
        # checks; the regime flipped to "ranging" + action flipped to "BUY" in
        # a single tick. Reject these single-check flips.
        regime = sig.get("regime", "unknown")
        if not self._regime_confirmed(ticker, action, regime):
            return False, f"regime not confirmed: need {REGIME_CONFIRM_CHECKS}x ({action},{regime})"

        # EOD check — don't buy near close
        h = _cet_hour()
        close_cet = 21.0 + 55 / 60  # 21:55
        minutes_to_close = (close_cet - h) * 60
        if minutes_to_close < EOD_EXIT_MINUTES_BEFORE + 60:
            return False, f"Too close to EOD ({minutes_to_close:.0f}min left)"

        return True, "entry criteria met"

    def _select_warrant(self, underlying, direction):
        """Pick best warrant by leverage/barrier/spread/issuer.

        Uses the dynamic catalog (refreshed from Avanza at startup). Candidates
        are scored on four factors:
        - leverage proximity to TARGET_LEVERAGE (weight 0.35)
        - barrier distance (weight 0.30)
        - bid-ask spread (weight 0.15)
        - issuer fee preference: AVA products get 1.0, others 0.5 (weight 0.20)

        AVA-issued warrants have 0 SEK courtage on Avanza while SG/VT/BNP
        charge regular courtage — the fee_score penalizes non-AVA products.

        Fails closed: returns None if the best candidate's leverage is below
        MIN_ACCEPTABLE_LEVERAGE, so the trader SKIPs instead of falling back
        to a 1.5x tracker (which was the 2026-04-09 SG incident).
        """
        candidates = []
        for key, w in self.warrant_catalog.items():
            if w.get("underlying") != underlying or w.get("direction") != direction:
                continue

            data = fetch_price(self.page, w["ob_id"], w["api_type"])
            if not data or not data.get("bid") or not data.get("ask"):
                continue

            bid = data["bid"]
            ask = data["ask"]
            if bid <= 0 or ask <= 0:
                continue

            spread_pct = (ask - bid) / bid * 100
            live_leverage = data.get("leverage") or w.get("leverage")
            live_barrier = data.get("barrier") or w.get("barrier")

            # Barrier distance — LONG: price above barrier, SHORT: price below barrier.
            barrier_dist = 999
            if live_barrier and live_barrier > 0:
                underlying_price = data.get("underlying", 0)
                if underlying_price > 0:
                    if direction == "LONG":
                        barrier_dist = (underlying_price - live_barrier) / underlying_price * 100
                    else:  # SHORT
                        barrier_dist = (live_barrier - underlying_price) / underlying_price * 100

            # Filter: minimum thresholds
            if barrier_dist < MIN_BARRIER_DISTANCE_PCT:
                _log(f"  {w['name']}: barrier too close ({barrier_dist:.1f}% < {MIN_BARRIER_DISTANCE_PCT}%)")
                continue
            if spread_pct > MIN_SPREAD_PCT:
                _log(f"  {w['name']}: spread too wide ({spread_pct:.1f}% > {MIN_SPREAD_PCT}%)")
                continue

            # Score: prefer leverage near target, far from barrier, tight spread, AVA issuer
            leverage_score = 1.0 / (1 + abs(live_leverage - TARGET_LEVERAGE))
            barrier_score = min(barrier_dist / 30, 1.0)
            spread_score = 1.0 / (1 + spread_pct)
            # isAza=True → 0 courtage on Avanza; other issuers pay regular fees
            is_aza = bool(w.get("isAza", False))
            fee_score = 1.0 if is_aza else 0.5
            score = (
                leverage_score * 0.35
                + barrier_score * 0.30
                + spread_score * 0.15
                + fee_score * 0.20
            )

            candidates.append({
                **w,
                "key": key,
                "live_bid": bid,
                "live_ask": ask,
                "live_leverage": live_leverage,
                "live_barrier": live_barrier,
                "barrier_dist": barrier_dist,
                "spread_pct": spread_pct,
                "underlying_price": data.get("underlying", 0),
                "is_aza": is_aza,
                "score": score,
            })

        if not candidates:
            _log(f"  No valid {direction} candidates for {underlying} in catalog "
                 f"(catalog size: {len(self.warrant_catalog)})")
            return None

        best = max(candidates, key=lambda w: w["score"])

        # Fail-closed: if the best candidate is under-leveraged, SKIP the trade
        # entirely rather than silently buy a low-leverage tracker. The 2026-04-09
        # SG incident bought 1.75x because it was the only candidate that passed
        # the barrier gate — now we explicitly reject that degenerate case.
        if best["live_leverage"] < MIN_ACCEPTABLE_LEVERAGE:
            _log(f"  SKIP_BUY: best candidate {best['name']} has lev "
                 f"{best['live_leverage']:.2f}x < MIN_ACCEPTABLE_LEVERAGE "
                 f"({MIN_ACCEPTABLE_LEVERAGE}). Not falling back to a tracker.")
            return None

        _log(f"  Selected: {best['name']} lev={best['live_leverage']:.2f}x "
             f"barrier={best['barrier_dist']:.1f}% spread={best['spread_pct']:.2f}% "
             f"AVA={best['is_aza']} score={best['score']:.3f}")
        return best

    def _execute_buy(self, warrant, units, ask_price, underlying_ticker, sig, total_cost):
        """Execute a BUY order and set stop-loss."""
        pos_id = f"pos_{int(time.time())}"

        _log(f"BUY {warrant['name']}: {units}u @ {ask_price} = {total_cost:.0f} SEK "
             f"(underlying: {underlying_ticker}, lev: {warrant['live_leverage']:.1f}x)")

        trade_record = {
            "ts": _now_utc().isoformat(),
            "action": "BUY",
            "pos_id": pos_id,
            "warrant_key": warrant["key"],
            "warrant_name": warrant["name"],
            "underlying": underlying_ticker,
            "units": units,
            "price": ask_price,
            "total_sek": round(total_cost, 2),
            "underlying_price": warrant.get("underlying_price", 0),
            "leverage": warrant["live_leverage"],
            "signal": _compact_signal(sig),
            "dry_run": DRY_RUN,
        }

        if DRY_RUN:
            _log(f"  [DRY RUN] Would place BUY order: {units}u @ {ask_price}")
            trade_record["result"] = "DRY_RUN"
        else:
            success, result = place_order(self.page, ACCOUNT_ID, warrant["ob_id"], "BUY", ask_price, units)
            trade_record["result"] = result
            if not success:
                _log(f"  BUY FAILED: {result}")
                _log_trade(trade_record)
                _send_telegram(f"*SWING BUY FAILED* {warrant['name']}\n{result.get('parsed', {}).get('message', str(result)[:100])}")
                return

        _log_trade(trade_record)

        # Update state
        underlying_price = warrant.get("underlying_price", 0)
        self.state["positions"][pos_id] = {
            "warrant_key": warrant["key"],
            "warrant_name": warrant["name"],
            "ob_id": warrant["ob_id"],
            "api_type": warrant["api_type"],
            "underlying": underlying_ticker,
            "units": units,
            "entry_price": ask_price,
            "entry_underlying": underlying_price,
            "entry_ts": _now_utc().isoformat(),
            "peak_underlying": underlying_price,
            "trailing_active": False,
            "stop_order_id": None,
            "leverage": warrant["live_leverage"],
        }
        self.state["cash_sek"] -= total_cost
        self.state["last_buy_ts"] = _now_utc().isoformat()
        self.state["total_trades"] += 1
        self.state["session_trades"] += 1
        _save_state(self.state)

        # Place hardware stop-loss
        self._set_stop_loss(pos_id)

        # Telegram
        msg = (f"{'*[DRY] ' if DRY_RUN else '*'}SWING BUY* {warrant['name']}\n"
               f"`{units}u @ {ask_price} = {total_cost:.0f} SEK`\n"
               f"`Lev: {warrant['live_leverage']:.1f}x | Underlying: {underlying_price:.2f}`\n"
               f"`Signals: {sig.get('buy_count', 0)}B/{sig.get('sell_count', 0)}S | RSI {sig.get('rsi', 0):.0f}`\n"
               f"_TP: +{TAKE_PROFIT_UNDERLYING_PCT}% und | Stop: -{HARD_STOP_UNDERLYING_PCT}% und_")
        _send_telegram(msg)

        decision = {
            "ts": _now_utc().isoformat(),
            "check": self.check_count,
            "underlying": underlying_ticker,
            "action": "BUY",
            "warrant": warrant["name"],
            "units": units,
            "price": ask_price,
            "total_sek": round(total_cost, 2),
            "signal": _compact_signal(sig),
            "dry_run": DRY_RUN,
        }
        _log_decision(decision)

    def _set_stop_loss(self, pos_id):
        """Place hardware stop-loss for a position."""
        pos = self.state["positions"].get(pos_id)
        if not pos:
            return

        entry_und = pos.get("entry_underlying", 0)
        leverage = pos.get("leverage", 5.0)
        entry_price = pos["entry_price"]

        if entry_und <= 0 or entry_price <= 0:
            _log("  Cannot set stop: no entry underlying price")
            return

        # Stop at -STOP_LOSS_UNDERLYING_PCT% on underlying, translated to warrant price
        und_drop_pct = STOP_LOSS_UNDERLYING_PCT / 100
        warrant_drop_pct = und_drop_pct * leverage
        trigger_price = round(entry_price * (1 - warrant_drop_pct), 2)
        sell_price = round(trigger_price * 0.99, 2)  # sell 1% below trigger for fill

        if trigger_price <= 0:
            _log("  Stop price would be <=0, skipping")
            return

        _log(f"  Setting stop-loss: trigger={trigger_price} sell={sell_price} "
             f"(und -{STOP_LOSS_UNDERLYING_PCT}% * {leverage:.1f}x lev)")

        if DRY_RUN:
            _log(f"  [DRY RUN] Would place stop-loss @ {trigger_price}")
            pos["stop_order_id"] = "DRY_RUN"
            _save_state(self.state)
            return

        success, stop_id = place_stop_loss(
            self.page, ACCOUNT_ID, pos["ob_id"], trigger_price, sell_price,
            pos["units"], valid_days=STOP_LOSS_VALID_DAYS,
        )
        if success:
            pos["stop_order_id"] = stop_id
            _save_state(self.state)
            _log(f"  Stop-loss placed: {stop_id}")
        else:
            _log("  Stop-loss FAILED")
            _send_telegram(f"_SWING: stop-loss failed for {pos['warrant_name']}_")

    # -------------------------------------------------------------------
    # Exit logic
    # -------------------------------------------------------------------

    def _check_exits(self, prices, signal_data):
        """Check exit conditions on all open positions."""
        now = _now_utc()
        h = _cet_hour()
        close_cet = 21.0 + 55 / 60  # 21:55 CET
        minutes_to_close = (close_cet - h) * 60

        to_remove = []

        for pos_id, pos in list(self.state["positions"].items()):
            # Get current underlying price
            underlying_price = self._get_underlying_price(pos, prices)
            if underlying_price <= 0:
                continue

            entry_und = pos.get("entry_underlying", 0)
            if entry_und <= 0:
                continue

            # Track peak
            if underlying_price > pos.get("peak_underlying", 0):
                pos["peak_underlying"] = underlying_price

            und_change_pct = (underlying_price - entry_und) / entry_und * 100
            peak_und = pos.get("peak_underlying", entry_und)
            from_peak_pct = (underlying_price - peak_und) / peak_und * 100 if peak_und > 0 else 0

            # Get current warrant price for P&L
            warrant_data = fetch_price(self.page, pos["ob_id"], pos["api_type"])
            current_bid = warrant_data.get("bid", 0) if warrant_data else 0

            exit_reason = None

            # --- Exit optimizer: probabilistic exit assessment ---
            try:
                from portfolio.cost_model import get_cost_model
                from portfolio.exit_optimizer import MarketSnapshot, Position, compute_exit_plan
                from portfolio.session_calendar import get_session_info
                sess = get_session_info("warrant", underlying=pos.get("underlying"))
                if sess.is_open and sess.remaining_minutes >= 2:
                    opt_pos = Position(
                        symbol=pos.get("underlying", ""),
                        qty=pos.get("units", 0),
                        entry_price_sek=pos.get("entry_price", 0),
                        entry_underlying_usd=entry_und,
                        entry_ts=datetime.datetime.fromisoformat(pos["entry_ts"]) if pos.get("entry_ts") else _now_utc(),
                        instrument_type="warrant",
                        leverage=pos.get("leverage", 5.0),
                        financing_level=pos.get("financing_level"),
                    )
                    # NOTE: MarketSnapshot.bid is documented as the underlying's
                    # USD bid (not the warrant SEK bid). Passing the warrant bid
                    # here was the source of the 2026-04-09 -2430 SEK fake-loss
                    # bug — _compute_pnl_sek treated warrant_bid_sek as if it
                    # were a silver USD price and computed a -46% "move".
                    # Pass None so the optimizer uses market.price (the correct
                    # underlying USD reference) for the market-exit estimate.
                    opt_market = MarketSnapshot(
                        asof_ts=_now_utc(),
                        price=underlying_price,
                        bid=None,
                        usdsek=10.85,
                    )
                    exit_plan = compute_exit_plan(
                        opt_pos, opt_market, sess.session_end,
                        costs=get_cost_model("warrant"), n_paths=2000,
                    )
                    pos["_exit_plan"] = {
                        "recommended": exit_plan.recommended.action,
                        "rec_price": exit_plan.recommended.price_usd,
                        "rec_ev": exit_plan.recommended.ev_sek,
                        "rec_fill_prob": exit_plan.recommended.fill_prob,
                        "stop_hit_prob": exit_plan.stop_hit_prob,
                        "risk_flags": list(exit_plan.recommended.risk_flags),
                        "market_exit_pnl": exit_plan.market_exit.pnl_sek,
                    }
                    # Override: if optimizer says market exit due to risk override
                    if (exit_plan.recommended.action == "market"
                            and any(f in exit_plan.recommended.risk_flags
                                    for f in ("KNOCKOUT_DANGER", "SESSION_END_IMMINENT"))):
                        exit_reason = f"EXIT_OPTIMIZER: {', '.join(exit_plan.recommended.risk_flags)} (EV {exit_plan.recommended.ev_sek:+,.0f} SEK)"
                    # Override: if stop hit probability is very high
                    elif exit_plan.stop_hit_prob > 0.30:
                        exit_reason = f"EXIT_OPTIMIZER: stop hit prob {exit_plan.stop_hit_prob:.0%} > 30% (EV {exit_plan.recommended.ev_sek:+,.0f} SEK)"
            except Exception as e:
                _log(f"Exit optimizer error: {e}")

            # 1. Take profit
            if not exit_reason and und_change_pct >= TAKE_PROFIT_UNDERLYING_PCT:
                exit_reason = f"TAKE_PROFIT: underlying +{und_change_pct:.2f}% >= +{TAKE_PROFIT_UNDERLYING_PCT}%"

            # 2. Trailing stop
            if not exit_reason and und_change_pct >= TRAILING_START_PCT:
                pos["trailing_active"] = True
                if from_peak_pct <= -TRAILING_DISTANCE_PCT:
                    exit_reason = f"TRAILING_STOP: {from_peak_pct:.2f}% from peak (trail={TRAILING_DISTANCE_PCT}%)"

            # 3. Hard stop
            if not exit_reason and und_change_pct <= -HARD_STOP_UNDERLYING_PCT:
                exit_reason = f"HARD_STOP: underlying {und_change_pct:.2f}% <= -{HARD_STOP_UNDERLYING_PCT}%"

            # 4. Signal reversal
            if not exit_reason and SIGNAL_REVERSAL_EXIT and signal_data:
                sig = signal_data.get(pos["underlying"], {})
                sell_count = sig.get("sell_count", 0)
                tf = sig.get("timeframes", {})
                sell_tfs = sum(1 for v in tf.values() if v == "SELL") if tf else 0
                total_tfs = len(tf) if tf else 7
                if sell_count >= MIN_BUY_VOTERS and total_tfs > 0 and sell_tfs / total_tfs >= MIN_BUY_TF_RATIO:
                    exit_reason = f"SIGNAL_REVERSAL: {sell_count}S, {sell_tfs}/{total_tfs} TFs SELL"

            # 5. Time limit
            if not exit_reason:
                entry_ts = datetime.datetime.fromisoformat(pos["entry_ts"])
                held_hours = (now - entry_ts).total_seconds() / 3600
                if held_hours >= MAX_HOLD_HOURS:
                    exit_reason = f"TIME_LIMIT: held {held_hours:.1f}h >= {MAX_HOLD_HOURS}h"

            # 6. EOD exit
            if not exit_reason and minutes_to_close <= EOD_EXIT_MINUTES_BEFORE:
                exit_reason = f"EOD_EXIT: {minutes_to_close:.0f}min to close"

            # 7. Momentum exit — 3 consecutive declining underlying prices
            if not exit_reason and len(self.state.get("_und_history", {}).get(pos["underlying"], [])) >= 3:
                hist = self.state["_und_history"][pos["underlying"]][-3:]
                if all(hist[i] < hist[i - 1] for i in range(1, len(hist))):
                    decline_rate = (hist[-1] - hist[0]) / hist[0] * 100
                    if decline_rate < -0.3:  # at least -0.3% over 3 checks
                        exit_reason = f"MOMENTUM_EXIT: 3 declining checks ({decline_rate:.2f}%)"

            if exit_reason:
                self._execute_sell(pos_id, pos, current_bid, underlying_price, exit_reason)
                to_remove.append(pos_id)

        # Update underlying price history for momentum tracking
        if signal_data:
            if "_und_history" not in self.state:
                self.state["_und_history"] = {}
            for ticker in ["XAG-USD", "XAU-USD"]:
                und_price = self._get_ticker_underlying_price(ticker, prices)
                if und_price > 0:
                    hist = self.state["_und_history"].setdefault(ticker, [])
                    hist.append(und_price)
                    if len(hist) > 10:
                        hist.pop(0)

        if to_remove:
            _save_state(self.state)

    def _execute_sell(self, pos_id, pos, current_bid, underlying_price, reason):
        """Execute a SELL order for a position."""
        units = pos["units"]
        entry_price = pos["entry_price"]
        warrant_pnl_pct = ((current_bid / entry_price) - 1) * 100 if entry_price > 0 else 0
        proceeds = units * current_bid

        _log(f"SELL {pos['warrant_name']}: {units}u @ {current_bid} = {proceeds:.0f} SEK "
             f"(PnL: {warrant_pnl_pct:+.1f}%) — {reason}")

        trade_record = {
            "ts": _now_utc().isoformat(),
            "action": "SELL",
            "pos_id": pos_id,
            "warrant_key": pos["warrant_key"],
            "warrant_name": pos["warrant_name"],
            "underlying": pos["underlying"],
            "units": units,
            "price": current_bid,
            "total_sek": round(proceeds, 2),
            "entry_price": entry_price,
            "pnl_pct": round(warrant_pnl_pct, 2),
            "pnl_sek": round(proceeds - (units * entry_price), 2),
            "underlying_price": underlying_price,
            "entry_underlying": pos.get("entry_underlying", 0),
            "reason": reason,
            "held_hours": round((_now_utc() - datetime.datetime.fromisoformat(pos["entry_ts"])).total_seconds() / 3600, 2),
            "dry_run": DRY_RUN,
        }

        if DRY_RUN:
            _log(f"  [DRY RUN] Would place SELL order: {units}u @ {current_bid}")
            trade_record["result"] = "DRY_RUN"
        else:
            success, result = place_order(self.page, ACCOUNT_ID, pos["ob_id"], "SELL", current_bid, units)
            trade_record["result"] = result
            if not success:
                _log(f"  SELL FAILED: {result}")
                _log_trade(trade_record)
                _send_telegram(f"*SWING SELL FAILED* {pos['warrant_name']}\n{result.get('parsed', {}).get('message', str(result)[:100])}")
                return

        _log_trade(trade_record)

        # Cancel hardware stop-loss
        if pos.get("stop_order_id") and pos["stop_order_id"] != "DRY_RUN":
            if not DRY_RUN:
                ok = _delete_stop_loss(self.page, pos["stop_order_id"])
                _log(f"  Stop-loss cancelled: {ok}")

        # Update state
        pnl_sek = proceeds - (units * entry_price)
        self.state["cash_sek"] += proceeds
        self.state["total_pnl_sek"] += pnl_sek
        self.state["total_trades"] += 1
        self.state["session_trades"] += 1

        if pnl_sek < 0:
            self.state["consecutive_losses"] = self.state.get("consecutive_losses", 0) + 1
        else:
            self.state["consecutive_losses"] = 0

        # Remove position
        del self.state["positions"][pos_id]
        _save_state(self.state)

        # Telegram
        pnl_emoji = "+" if pnl_sek >= 0 else ""
        msg = (f"{'*[DRY] ' if DRY_RUN else '*'}SWING SELL* {pos['warrant_name']}\n"
               f"`{units}u @ {current_bid} = {proceeds:.0f} SEK`\n"
               f"`PnL: {pnl_emoji}{warrant_pnl_pct:.1f}% ({pnl_emoji}{pnl_sek:.0f} SEK)`\n"
               f"`Reason: {reason}`\n"
               f"_Cash: {self.state['cash_sek']:.0f} SEK | Total PnL: {self.state['total_pnl_sek']:+.0f} SEK_")
        _send_telegram(msg)

        decision = {
            "ts": _now_utc().isoformat(),
            "check": self.check_count,
            "underlying": pos["underlying"],
            "action": "SELL",
            "warrant": pos["warrant_name"],
            "units": units,
            "price": current_bid,
            "pnl_pct": round(warrant_pnl_pct, 2),
            "pnl_sek": round(pnl_sek, 2),
            "reason": reason,
            "dry_run": DRY_RUN,
        }
        _log_decision(decision)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _has_position(self, underlying_ticker):
        """Check if we already have a position in this underlying."""
        for pos in self.state["positions"].values():
            if pos["underlying"] == underlying_ticker:
                return True
        return False

    def _cooldown_cleared(self):
        """Check if BUY cooldown has elapsed (with loss escalation)."""
        last_ts = self.state.get("last_buy_ts")
        if not last_ts:
            return True

        try:
            last_dt = datetime.datetime.fromisoformat(last_ts)
            elapsed_min = (_now_utc() - last_dt).total_seconds() / 60

            losses = self.state.get("consecutive_losses", 0)
            multiplier = LOSS_ESCALATION.get(min(losses, max(LOSS_ESCALATION.keys())), 1)
            required_min = BUY_COOLDOWN_MINUTES * multiplier

            if elapsed_min < required_min:
                return False
        except Exception:
            pass

        return True

    def _get_underlying_price(self, pos, prices):
        """Get current underlying price from loop's price data or fetch directly."""
        # Try from loop's price data (keyed by position name, not ticker)
        for key, p in prices.items():
            if isinstance(p, dict) and p.get("underlying"):
                ticker = "XAG-USD" if "silver" in key.lower() else "XAU-USD"
                if ticker == pos.get("underlying"):
                    return p["underlying"]

        # Fallback: fetch warrant price and extract underlying
        data = fetch_price(self.page, pos["ob_id"], pos["api_type"])
        if data and data.get("underlying"):
            return data["underlying"]

        return 0

    def _get_ticker_underlying_price(self, ticker, prices):
        """Get underlying price for a ticker from loop price data."""
        for key, p in prices.items():
            if isinstance(p, dict) and p.get("underlying"):
                mapped = "XAG-USD" if "silver" in key.lower() else "XAU-USD"
                if mapped == ticker:
                    return p["underlying"]
        return 0

    def _update_macd_history(self, signal_data):
        """Track MACD histogram values across checks."""
        if not signal_data:
            return

        if "macd_history" not in self.state:
            self.state["macd_history"] = {}

        for ticker in ["XAG-USD", "XAU-USD"]:
            sig = signal_data.get(ticker)
            if not sig:
                continue

            macd_hist = sig.get("macd_hist")
            if macd_hist is None:
                continue

            history = self.state["macd_history"].setdefault(ticker, [])
            history.append(macd_hist)
            if len(history) > 20:
                history.pop(0)

        _save_state(self.state)

    def _send_summary(self, signal_data):
        """Send periodic Telegram summary."""
        positions = self.state["positions"]
        cash = self.state["cash_sek"]

        if not positions:
            return

        lines = [f"*SWING #{self.check_count}* {len(positions)} position(s)"]
        for _pid, pos in positions.items():
            data = fetch_price(self.page, pos["ob_id"], pos["api_type"])
            bid = data.get("bid", 0) if data else 0
            pnl = ((bid / pos["entry_price"]) - 1) * 100 if pos["entry_price"] > 0 else 0
            held = (_now_utc() - datetime.datetime.fromisoformat(pos["entry_ts"])).total_seconds() / 3600
            trail = " TRAIL" if pos.get("trailing_active") else ""
            lines.append(f"`{pos['warrant_name']}: {bid} ({pnl:+.1f}%) {held:.1f}h{trail}`")
        lines.append(f"`Cash: {cash:.0f} | Trades: {self.state['total_trades']} | PnL: {self.state['total_pnl_sek']:+.0f}`")
        if DRY_RUN:
            lines.append("_DRY RUN mode_")
        _send_telegram("\n".join(lines))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _compact_signal(sig):
    """Create compact signal summary for logging."""
    if not sig:
        return {}
    return {
        "action": sig.get("action"),
        "buy": sig.get("buy_count", 0),
        "sell": sig.get("sell_count", 0),
        "rsi": round(sig.get("rsi", 0), 1),
        "macd": sig.get("macd_hist"),
        "regime": sig.get("regime"),
        "confidence": sig.get("confidence", 0),
    }
