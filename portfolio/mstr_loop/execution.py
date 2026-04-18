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
    """Baseline cert price for Phase B (no live cert quote available).

    We don't have a live Avanza quote in shadow mode — use a placeholder
    price of 100 SEK for P&L math. The scorecard works in underlying %
    terms anyway, so this placeholder doesn't affect win-rate metrics.
    """
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
        entry_ts=_now_iso(),
        trail_active=False,
        peak_underlying_price=bundle.price_usd,
        stop_price_cert=None,
        rationale=decision.rationale,
    )
    state.add_position(pos)
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
    proceeds = pos.units * cert_bid
    pnl_sek = proceeds - (pos.units * pos.entry_cert_price)

    if config.PHASE == "shadow":
        _record_shadow("SHADOW_SELL", decision, bundle, cert_bid, pos.units, proceeds,
                       extra={"pnl_sek": pnl_sek, "exit_reason": decision.exit_reason,
                              "entry_cert_price": pos.entry_cert_price,
                              "entry_underlying_price": pos.entry_underlying_price})
    elif config.PHASE == "paper":
        state.cash_sek += proceeds
        _record_trade("SELL", decision, bundle, cert_bid, pos.units, proceeds,
                      extra={"pnl_sek": pnl_sek})
    elif config.PHASE == "live":
        ok = _live_place_sell(decision, cert_bid, pos.units)
        if not ok:
            return False
        state.cash_sek += proceeds
        _record_trade("SELL", decision, bundle, cert_bid, pos.units, proceeds,
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
    return True


# ----------------------------------------------------------------------
# Peak tracking — called once per cycle by loop.py for live positions
# so the trailing-stop math in strategies has up-to-date peak data.
# ----------------------------------------------------------------------
def update_trail_state(state: BotState, bundle: MstrBundle) -> None:
    """Refresh trail_active and peak_underlying_price for all open positions."""
    for pos in state.positions.values():
        if bundle.price_usd <= 0:
            continue
        # Update peak
        if pos.direction == "LONG":
            if bundle.price_usd > pos.peak_underlying_price:
                pos.peak_underlying_price = bundle.price_usd
        else:  # SHORT
            # For SHORT, peak = most favourable = lowest underlying
            if (
                pos.peak_underlying_price == 0
                or bundle.price_usd < pos.peak_underlying_price
            ):
                pos.peak_underlying_price = bundle.price_usd

        # Activate trail once profit threshold reached
        if not pos.trail_active:
            pnl_pct = pos.unrealized_underlying_pct(bundle.price_usd)
            if pnl_pct >= config.MOMENTUM_RIDER_TRAIL_ACTIVATION_PCT:
                pos.trail_active = True


# ----------------------------------------------------------------------
# Cert price estimators (shadow + paper only; live uses real quote)
# ----------------------------------------------------------------------
def _estimate_cert_ask(bundle: MstrBundle, direction: str) -> float:
    if config.PHASE == "live":
        return _live_fetch_cert_ask(direction)
    # shadow + paper: synthetic baseline
    return _compute_shadow_cert_price(bundle, direction)


def _estimate_cert_bid(bundle: MstrBundle, pos: Position) -> float:
    if config.PHASE == "live":
        return _live_fetch_cert_bid(pos.cert_ob_id)
    # shadow + paper: derive from underlying move
    return _approx_cert_price_from_underlying(
        entry_cert_price=pos.entry_cert_price,
        entry_underlying_price=pos.entry_underlying_price,
        current_underlying_price=bundle.price_usd,
        leverage=config.BULL_MSTR_LEVERAGE,
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
