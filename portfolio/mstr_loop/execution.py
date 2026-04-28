"""Execution layer for MSTR Loop — gated by PHASE.

- shadow: decisions append to SHADOW_LOG; positions tracked in-memory only.
- paper:  decisions deduct/credit state.cash_sek; positions persisted.
- live:   decisions place real Avanza orders via portfolio.avanza_session.

All three paths share the same Decision → Position lifecycle so migration
between phases is a config flag flip, not a code change.

Design choices:
- Shadow and paper do not call Avanza at all — the module must import
  cleanly even if avanza_session/avanza_control are unavailable.
- Live path lazy-imports avanza_session so shadow/paper users don't pay
  Playwright startup.
- Every fill (shadow or paper or live) writes to TRADES_LOG. Shadow
  additionally writes to SHADOW_LOG with pre-fill decision metadata
  (for scoring against outcomes via scripts/mstr_loop_scorecard.py).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from portfolio.file_utils import atomic_append_jsonl
from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position
from portfolio.mstr_loop.strategies.base import Decision

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _approx_cert_price_from_underlying(
    entry_cert_price: float,
    entry_underlying_price: float,
    current_underlying_price: float,
    leverage: float,
    direction: str,
) -> float:
    """Approximate the cert price from an underlying move, for shadow/paper P&L.

    Simple linear-leverage model: `cert_pct ≈ underlying_pct × leverage × dir_sign`.
    Daily-reset compounding is ignored (we're intraday-only; the error is
    bounded). When Phase D runs live, this function is NOT used — we read
    the real cert price from the broker quote.
    """
    if entry_underlying_price <= 0 or entry_cert_price <= 0:
        return entry_cert_price
    under_pct = (
        (current_underlying_price - entry_underlying_price) /
        entry_underlying_price
    )
    dir_sign = 1.0 if direction == "LONG" else -1.0
    cert_pct = under_pct * leverage * dir_sign
    return max(0.01, entry_cert_price * (1.0 + cert_pct))


def _compute_shadow_cert_price(bundle: MstrBundle, direction: str) -> float:
    """Cert price used in shadow/paper modes.

    Preference order (v2 Tier 1):
      1. Try live Avanza quote — gives shadow P&L the same scale as Phase D
         would see. Avoids the shadow→paper P&L discontinuity that would
         otherwise fire on phase transitions.
      2. Fall back to synthetic 100.0 SEK if Avanza session unavailable
         (tests, offline env, or pre-auth cold start).
    """
    try:
        ob_id = (
            config.BULL_MSTR_OB_ID if direction == "LONG" else config.BEAR_MSTR_OB_ID
        )
        if ob_id:
            from portfolio.avanza_session import get_quote
            q = get_quote(ob_id)
            # Prefer ask ("sell" at Avanza) for a BUY simulation
            ask = float(q.get("sell") or 0) if q else 0.0
            if ask > 0:
                return ask
    except Exception:
        logger.debug("execution: live quote unavailable, using synthetic", exc_info=True)
    return 100.0


def _notional_for_entry(state: BotState) -> float:
    """Return the target SEK notional for a new entry, phase-aware."""
    if config.PHASE == "shadow":
        return float(config.SHADOW_NOTIONAL_SEK)
    if config.PHASE == "paper":
        cash = state.cash_sek
        raw = cash * (config.POSITION_SIZE_PCT / 100.0)
        alloc = max(raw, float(config.MIN_TRADE_SEK))
        alloc = min(alloc, cash * 0.95)
        if alloc < config.MIN_TRADE_SEK:
            return 0.0  # infeasible — caller should skip
        return alloc
    # live
    cash = state.cash_sek
    raw = cash * (config.POSITION_SIZE_PCT / 100.0)
    alloc = max(raw, float(config.MIN_TRADE_SEK))
    alloc = min(alloc, cash * 0.95)
    if alloc < config.MIN_TRADE_SEK:
        return 0.0
    return alloc


def execute(decision: Decision, bundle: MstrBundle, state: BotState) -> bool:
    """Route a decision to the correct execution path.

    Returns True on success (order placed / shadow logged), False if we
    refused to act (e.g. insufficient cash in paper mode).
    """
    if decision.action == "BUY":
        return _handle_buy(decision, bundle, state)
    if decision.action == "SELL":
        return _handle_sell(decision, bundle, state)
    logger.warning("execution: unknown action %r — ignored", decision.action)
    return False


# ----------------------------------------------------------------------
# BUY
# ----------------------------------------------------------------------
def _handle_buy(decision: Decision, bundle: MstrBundle, state: BotState) -> bool:
    notional = decision.notional_sek_hint or _notional_for_entry(state)
    if notional <= 0:
        logger.info("execution: %s BUY skipped — insufficient cash for MIN_TRADE_SEK",
                    decision.strategy_key)
        return False

    cert_ask = _estimate_cert_ask(bundle, decision.direction)
    if cert_ask <= 0:
        logger.info("execution: %s BUY skipped — cert ask unavailable",
                    decision.strategy_key)
        return False

    units = int(notional / cert_ask)
    if units < 1:
        logger.info("execution: %s BUY skipped — notional %.0f / ask %.2f < 1 unit",
                    decision.strategy_key, notional, cert_ask)
        return False

    total_cost = units * cert_ask

    # Per-phase execution
    if config.PHASE == "shadow":
        _record_shadow("SHADOW_BUY", decision, bundle, cert_ask, units, total_cost)
    elif config.PHASE == "paper":
        if total_cost > state.cash_sek * 0.95:
            logger.info("execution: %s BUY refused — total_cost %.0f > 95%% of cash",
                        decision.strategy_key, total_cost)
            return False
        state.cash_sek -= total_cost
        _record_trade("BUY", decision, bundle, cert_ask, units, total_cost)
    elif config.PHASE == "live":
        ok = _live_place_buy(decision, cert_ask, units)
        if not ok:
            return False
        state.cash_sek -= total_cost  # live cash will re-sync next cycle
        _record_trade("BUY", decision, bundle, cert_ask, units, total_cost)
    else:
        logger.error("execution: unknown PHASE %r", config.PHASE)
        return False

    # Create the position in state — same for all phases. Shadow mode
    # keeps the Position in-memory only; the session.py EOD-flatten or
    # strategy exit rule drives the close.
    pos = Position(
        strategy_key=decision.strategy_key,
        direction=decision.direction,
        cert_ob_id=decision.cert_ob_id,
        entry_underlying_price=bundle.price_usd,
        entry_cert_price=cert_ask,
        units=units,
        entry_units=units,           # immutable baseline for partial-exit tranche math
        entry_ts=_now_iso(),
        trail_active=False,
        peak_underlying_price=bundle.price_usd,
        stop_price_cert=None,
        rationale=decision.rationale,
    )
    state.add_position(pos)

    # Per-trade Telegram alert (fire-and-forget, non-fatal).
    try:
        from portfolio.mstr_loop import telegram_report
        telegram_report.send_trade_alert(
            "BUY",
            decision,
            {"underlying_price": bundle.price_usd, "units": units,
             "cert_price": cert_ask},
        )
    except Exception:
        logger.debug("execution: telegram BUY alert failed", exc_info=True)
    return True


# ----------------------------------------------------------------------
# SELL
# ----------------------------------------------------------------------
def _handle_sell(decision: Decision, bundle: MstrBundle, state: BotState) -> bool:
    pos = state.get_position(decision.strategy_key)
    if pos is None:
        logger.info("execution: %s SELL skipped — no open position",
                    decision.strategy_key)
        return False

    cert_bid = _estimate_cert_bid(bundle, pos)
    # Sell ALL remaining units for full exits. Partial-exit tranches use
    # _handle_partial_sell below.
    sell_units = pos.units
    proceeds = sell_units * cert_bid
    pnl_sek = proceeds - (sell_units * pos.entry_cert_price)

    if config.PHASE == "shadow":
        _record_shadow("SHADOW_SELL", decision, bundle, cert_bid, sell_units, proceeds,
                       extra={"pnl_sek": pnl_sek, "exit_reason": decision.exit_reason,
                              "entry_cert_price": pos.entry_cert_price,
                              "entry_underlying_price": pos.entry_underlying_price})
    elif config.PHASE == "paper":
        state.cash_sek += proceeds
        _record_trade("SELL", decision, bundle, cert_bid, sell_units, proceeds,
                      extra={"pnl_sek": pnl_sek})
    elif config.PHASE == "live":
        ok = _live_place_sell(decision, cert_bid, sell_units)
        if not ok:
            return False
        state.cash_sek += proceeds
        _record_trade("SELL", decision, bundle, cert_bid, sell_units, proceeds,
                      extra={"pnl_sek": pnl_sek})
    else:
        logger.error("execution: unknown PHASE %r", config.PHASE)
        return False

    # Update aggregate stats + cooldown marker.
    state.total_trades += 1
    state.total_pnl_sek += pnl_sek
    if pnl_sek > 0:
        state.wins += 1
    else:
        state.losses += 1
    state.last_exit_ts[decision.strategy_key] = _now_iso()
    state.remove_position(decision.strategy_key)

    # Fire-and-forget Telegram + auto-scorecard + trade alert.
    try:
        from portfolio.mstr_loop import telegram_report
        telegram_report.send_trade_alert(
            "SELL", pos,
            {"underlying_price": bundle.price_usd, "pnl_sek": pnl_sek,
             "exit_reason": decision.exit_reason},
        )
    except Exception:
        logger.debug("execution: telegram SELL alert failed", exc_info=True)
    try:
        if config.SCORECARD_UPDATE_ENABLED:
            _update_scorecard_file()
    except Exception:
        logger.debug("execution: scorecard update failed", exc_info=True)
    return True


def _handle_partial_sell(
    pos: Position,
    units_to_sell: int,
    tranche_pct: float,
    bundle: MstrBundle,
    state: BotState,
) -> bool:
    """Sell a fraction of an open position (tranche exit, keep position open).

    Caller (v2 partial-exit ladder in update_trail_state) computes which
    tranche tripped and how many units to sell. This helper keeps the
    position on-book but decrements units + accumulates units_sold.
    """
    if units_to_sell <= 0 or units_to_sell > pos.units:
        return False
    cert_bid = _estimate_cert_bid(bundle, pos)
    proceeds = units_to_sell * cert_bid
    pnl_sek = proceeds - (units_to_sell * pos.entry_cert_price)

    # Fake a Decision object for logging; partial exits don't originate
    # from a strategy.step() return but we want a consistent journal.
    from portfolio.mstr_loop.strategies.base import Decision
    d = Decision(
        strategy_key=pos.strategy_key, action="SELL", direction=pos.direction,
        cert_ob_id=pos.cert_ob_id,
        rationale=f"partial_exit_tranche_at_+{tranche_pct:.1f}%",
        exit_reason=f"tranche_{tranche_pct:.1f}",
    )

    if config.PHASE == "shadow":
        _record_shadow("SHADOW_PARTIAL_SELL", d, bundle, cert_bid, units_to_sell, proceeds,
                       extra={"pnl_sek": pnl_sek, "tranche_pct": tranche_pct,
                              "entry_cert_price": pos.entry_cert_price,
                              "entry_underlying_price": pos.entry_underlying_price})
    elif config.PHASE == "paper":
        state.cash_sek += proceeds
        _record_trade("PARTIAL_SELL", d, bundle, cert_bid, units_to_sell, proceeds,
                      extra={"pnl_sek": pnl_sek, "tranche_pct": tranche_pct})
    elif config.PHASE == "live":
        ok = _live_place_sell(d, cert_bid, units_to_sell)
        if not ok:
            return False
        state.cash_sek += proceeds
        _record_trade("PARTIAL_SELL", d, bundle, cert_bid, units_to_sell, proceeds,
                      extra={"pnl_sek": pnl_sek, "tranche_pct": tranche_pct})

    pos.units -= units_to_sell
    pos.units_sold += units_to_sell
    pos.tranches_hit.append(tranche_pct)
    state.total_pnl_sek += pnl_sek  # partials count toward running P&L
    return True


def _update_scorecard_file() -> None:
    """Recompute scorecard JSON from the shadow/trades logs and persist.

    Kept lightweight — reads the jsonl files we already write and calls
    into the scorecard module as a library. Non-fatal on any failure so a
    trade is never blocked by a scorecard write.
    """
    import pathlib
    # Dynamically import the scorecard script as a module (no package).
    # The script was written self-contained on purpose.
    script_path = pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "mstr_loop_scorecard.py"
    if not script_path.exists():
        return
    # Use importlib to avoid stomping sys.modules with a script name.
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mstr_loop_scorecard_helper", script_path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.main()  # writes data/mstr_loop_scorecard.json as side-effect
    except SystemExit:
        pass  # scorecard main() calls sys.exit, swallow
    except Exception:
        logger.debug("execution: scorecard module run failed", exc_info=True)


# ----------------------------------------------------------------------
# Peak tracking — called once per cycle by loop.py for live positions
# so the trailing-stop math in strategies has up-to-date peak data.
# ----------------------------------------------------------------------
def update_trail_state(state: BotState, bundle: MstrBundle) -> None:
    """Refresh trail_active + peak_underlying_price + partial-exit tranches.

    Called once per cycle by loop.py BEFORE strategy.step() so strategies
    see up-to-date peak data and partial tranches have already executed.

    v2 additions:
    - Partial-exit ladder: if config.PARTIAL_EXIT_LADDER_ENABLED, fire each
      (profit_pct, fraction) tranche the first time price crosses it. The
      tranche fraction is applied to `entry_units` (immutable baseline) so
      prior partial exits don't re-trigger on the next threshold.
    - ATR-adaptive trail: the trail distance exposed to strategies depends
      on current ATR% when ATR_ADAPTIVE_TRAIL_ENABLED is True. The strategy
      still owns the comparison; this function just tracks peak/activation.
    """
    positions_snapshot = list(state.positions.values())  # iterate a copy — _handle_partial_sell may not remove, but defensive
    for pos in positions_snapshot:
        if bundle.price_usd <= 0:
            continue
        # Update peak (direction-aware)
        if pos.direction == "LONG":
            if bundle.price_usd > pos.peak_underlying_price:
                pos.peak_underlying_price = bundle.price_usd
        else:  # SHORT
            if (
                pos.peak_underlying_price == 0
                or bundle.price_usd < pos.peak_underlying_price
            ):
                pos.peak_underlying_price = bundle.price_usd

        pnl_pct = pos.unrealized_underlying_pct(bundle.price_usd)

        # Trail activation — uses same threshold for both strategies (config
        # knobs for momentum_rider and mean_reversion happen to match).
        if not pos.trail_active and pnl_pct >= config.MOMENTUM_RIDER_TRAIL_ACTIVATION_PCT:
            pos.trail_active = True

        # Partial-exit ladder — sell tranches as price crosses thresholds.
        # Round pnl to 4 decimals so float-precision noise (e.g. 1.99999...)
        # doesn't under-trigger a +2% threshold.
        pnl_for_tranche = round(pnl_pct, 4)
        if config.PARTIAL_EXIT_LADDER_ENABLED and pos.entry_units > 0:
            for tranche_pct, fraction in config.PARTIAL_EXIT_TRANCHES:
                if tranche_pct in pos.tranches_hit:
                    continue
                if pnl_for_tranche < tranche_pct:
                    continue
                # Compute how many units to sell for this tranche. Use
                # round() not int() — float precision on e.g. 9×(1/3) can
                # land at 2.9999... which int() would truncate to 2.
                units_to_sell = round(pos.entry_units * fraction)
                if units_to_sell <= 0 or units_to_sell > pos.units:
                    # Either the fraction rounds to zero or we've somehow
                    # already sold too much — mark tranche as hit to avoid
                    # re-processing and move on.
                    pos.tranches_hit.append(tranche_pct)
                    continue
                _handle_partial_sell(pos, units_to_sell, tranche_pct, bundle, state)


# ----------------------------------------------------------------------
# Cert price estimators (shadow + paper only; live uses real quote)
# ----------------------------------------------------------------------
def _estimate_cert_ask(bundle: MstrBundle, direction: str) -> float:
    if config.PHASE == "live":
        return _live_fetch_cert_ask(direction)
    # shadow + paper: synthetic baseline
    return _compute_shadow_cert_price(bundle, direction)


def _estimate_cert_bid(bundle: MstrBundle, pos: Position) -> float:
    """Estimate the bid we could sell at for an open position.

    Phase D: live Avanza quote. Phase B/C: try live quote first (fidelity),
    fall back to leverage-derived synthetic. Uses the position's actual
    leverage (5x for BULL, 3x for BEAR) rather than a global constant.
    """
    if config.PHASE == "live":
        return _live_fetch_cert_bid(pos.cert_ob_id)
    # Try live quote for fidelity in shadow/paper modes too.
    try:
        from portfolio.avanza_session import get_quote
        q = get_quote(pos.cert_ob_id)
        bid = float(q.get("buy") or 0) if q else 0.0
        if bid > 0:
            return bid
    except Exception:
        logger.debug("execution: live bid unavailable, using synthetic", exc_info=True)
    # Synthetic fallback — per-direction leverage
    lev = (
        config.BULL_MSTR_LEVERAGE if pos.direction == "LONG"
        else config.BEAR_MSTR_LEVERAGE
    )
    return _approx_cert_price_from_underlying(
        entry_cert_price=pos.entry_cert_price,
        entry_underlying_price=pos.entry_underlying_price,
        current_underlying_price=bundle.price_usd,
        leverage=lev,
        direction=pos.direction,
    )


# ----------------------------------------------------------------------
# Live-mode Avanza API — lazy import so shadow/paper has no Playwright dep
# ----------------------------------------------------------------------
def _live_fetch_cert_ask(direction: str) -> float:
    try:
        from portfolio.avanza_session import get_quote
    except ImportError:
        logger.exception("execution: avanza_session unavailable in live mode")
        return 0.0
    ob_id = config.BULL_MSTR_OB_ID  # v1 LONG-only
    try:
        q = get_quote(ob_id)
        return float(q.get("sell") or 0)  # "sell" = ask at Avanza
    except Exception:
        logger.exception("execution: live cert ask fetch failed")
        return 0.0


def _live_fetch_cert_bid(ob_id: str) -> float:
    try:
        from portfolio.avanza_session import get_quote
    except ImportError:
        logger.exception("execution: avanza_session unavailable in live mode")
        return 0.0
    try:
        q = get_quote(ob_id)
        return float(q.get("buy") or 0)  # "buy" = bid at Avanza
    except Exception:
        logger.exception("execution: live cert bid fetch failed")
        return 0.0


def _live_place_buy(decision: Decision, price: float, units: int) -> bool:
    try:
        from portfolio.avanza_session import place_buy_order
    except ImportError:
        logger.exception("execution: avanza_session unavailable in live mode")
        return False
    try:
        result = place_buy_order(decision.cert_ob_id, price=price, volume=units)
        ok = result.get("orderRequestStatus") == "SUCCESS"
        if not ok:
            logger.error("execution: live BUY rejected — %r", result)
        return ok
    except Exception:
        logger.exception("execution: live BUY exception")
        return False


def _live_place_sell(decision: Decision, price: float, units: int) -> bool:
    try:
        from portfolio.avanza_session import place_sell_order
    except ImportError:
        logger.exception("execution: avanza_session unavailable in live mode")
        return False
    try:
        result = place_sell_order(decision.cert_ob_id, price=price, volume=units)
        ok = result.get("orderRequestStatus") == "SUCCESS"
        if not ok:
            logger.error("execution: live SELL rejected — %r", result)
        return ok
    except Exception:
        logger.exception("execution: live SELL exception")
        return False


# ----------------------------------------------------------------------
# Journaling helpers
# ----------------------------------------------------------------------
def _record_shadow(event: str, decision: Decision, bundle: MstrBundle,
                   cert_price: float, units: int, total_sek: float,
                   extra: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {
        "ts": _now_iso(),
        "event": event,
        "phase": "shadow",
        "strategy_key": decision.strategy_key,
        "direction": decision.direction,
        "cert_ob_id": decision.cert_ob_id,
        "cert_price": cert_price,
        "units": units,
        "notional_sek": total_sek,
        "rationale": decision.rationale,
        "confidence": decision.confidence,
        "underlying_price": bundle.price_usd,
        "weighted_score_long": bundle.weighted_score_long,
        "weighted_score_short": bundle.weighted_score_short,
        "rsi": bundle.rsi,
        "regime": bundle.regime,
        "p_up_1d": bundle.p_up_1d,
    }
    if extra:
        record.update(extra)
    try:
        atomic_append_jsonl(config.SHADOW_LOG, record)
    except Exception:
        logger.exception("execution: shadow log write failed")


def _record_trade(action: str, decision: Decision, bundle: MstrBundle,
                  cert_price: float, units: int, total_sek: float,
                  extra: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {
        "ts": _now_iso(),
        "phase": config.PHASE,
        "action": action,
        "strategy_key": decision.strategy_key,
        "direction": decision.direction,
        "cert_ob_id": decision.cert_ob_id,
        "cert_price": cert_price,
        "units": units,
        "notional_sek": total_sek,
        "rationale": decision.rationale,
        "exit_reason": decision.exit_reason,
        "underlying_price": bundle.price_usd,
    }
    if extra:
        record.update(extra)
    try:
        atomic_append_jsonl(config.TRADES_LOG, record)
    except Exception:
        logger.exception("execution: trade log write failed")
