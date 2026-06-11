"""MSTR Loop main orchestrator — 60s cycle, calls each enabled strategy.

Mirrors the metals_loop design. Each cycle:
  1. Check kill-switch + session window.
  2. Build the MstrBundle from agent_summary_compact.json.
  3. Update trail-state for open positions (so trail_active flips on time).
  4. Iterate enabled strategies; execute each returned Decision.
  5. Persist state.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any

from portfolio.file_utils import atomic_append_jsonl
from portfolio.mstr_loop import config, execution, risk, session, state, telegram_report
from portfolio.mstr_loop.data_provider import build_bundle
from portfolio.mstr_loop.strategies import load_enabled_strategies

logger = logging.getLogger(__name__)


def _write_heartbeat(bot_state: state.BotState, cycle_count: int) -> None:
    """Write the loop_health watchdog heartbeat after a successful cycle.

    2026-05-04: thin shim over `portfolio.loop_health.write_heartbeat`
    — schema lives there. Phase rides along as operator-facing context.
    Failure path swallows like before.
    """
    try:
        from portfolio.loop_health import write_heartbeat as _shared
        _shared(
            config.HEARTBEAT_FILE,
            cycle=cycle_count,
            ok=True,
            n_positions=len(bot_state.positions or {}),
            extra={"phase": config.PHASE},
        )
    except Exception:
        logger.debug("loop: heartbeat dispatch failed", exc_info=True)


def _log_poll(bundle_or_none, reason: str, cycle_count: int) -> None:
    """Append a per-cycle snapshot to POLL_LOG for post-hoc debugging."""
    record: dict[str, Any] = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "cycle": cycle_count,
        "phase": config.PHASE,
        "reason": reason,
    }
    if bundle_or_none is not None:
        record.update({
            "underlying_price": bundle_or_none.price_usd,
            "rsi": bundle_or_none.rsi,
            "regime": bundle_or_none.regime,
            "raw_action": bundle_or_none.raw_action,
            "weighted_long": bundle_or_none.weighted_score_long,
            "weighted_short": bundle_or_none.weighted_score_short,
            "stale": bundle_or_none.stale,
            "source_age_sec": bundle_or_none.source_stale_seconds,
        })
    try:
        atomic_append_jsonl(config.POLL_LOG, record)
    except Exception:
        logger.exception("loop: poll log write failed")


def _degraded_bundle_for(pos: "state.Position") -> Any:
    """Build a minimal MstrBundle from a position's last-known entry data.

    2026-06-11 (audit B8 fix 3): the EOD-flatten backstop must run even when
    build_bundle() returns None or an unusable (stale) bundle — otherwise a
    >5-min data-loop lag inside the 21:45-22:00 CET window carries a
    5x-leveraged position overnight. We can't get a fresh quote, so we
    synthesize a degraded bundle off the position's entry underlying price.
    execution._estimate_cert_bid falls back to the leverage-derived
    synthetic when no live quote is available, so the journaled exit price
    is approximate-but-bounded rather than zero. The bundle is flagged
    stale so nothing downstream mistakes it for live data.
    """
    from portfolio.mstr_loop.data_provider import MstrBundle
    return MstrBundle(
        ts=datetime.datetime.now(datetime.UTC).isoformat(),
        source_stale_seconds=float("inf"),
        price_usd=pos.entry_underlying_price,
        raw_action="HOLD",
        raw_weighted_confidence=0.0,
        rsi=50.0,
        macd_hist=0.0,
        bb_position="",
        regime="unknown",
        atr_pct=0.0,
        buy_count=0,
        sell_count=0,
        total_voters=0,
        votes={},
        p_up_1d=0.0,
        exp_return_1d_pct=0.0,
        exp_return_3d_pct=0.0,
        heatmap=[],
        stale=True,
        weighted_score_long=0.0,
        weighted_score_short=0.0,
    )


def _eod_flatten_backstop(
    bot_state: state.BotState, strategies: list, cycle_count: int,
    bundle: Any = None,
) -> None:
    """Force-exit every open position during the EOD window.

    2026-06-11 (audit B8 fix 3): loop-level backstop independent of
    bundle.is_usable(). The strategy-level EOD exit lives in
    _evaluate_exit, which is unreachable when momentum_rider/mean_reversion
    return None on stale data. This runs regardless: if `bundle` is usable
    we exit at its price; otherwise we synthesize a degraded bundle per
    position so the intraday-only contract holds even on a data outage.
    """
    if not bot_state.positions:
        return
    from portfolio.mstr_loop.strategies.base import Decision
    for pos in list(bot_state.positions.values()):
        use_bundle = bundle
        degraded = bundle is None or not bundle.is_usable()
        if degraded:
            use_bundle = _degraded_bundle_for(pos)
        if degraded:
            logger.warning(
                "loop: EOD backstop flattening %s on DEGRADED data "
                "(bundle unusable/missing) — exit price is synthetic",
                pos.strategy_key,
            )
        exit_dec = Decision(
            strategy_key=pos.strategy_key,
            action="SELL",
            direction=pos.direction,
            cert_ob_id=pos.cert_ob_id,
            rationale="eod_flatten_backstop",
            exit_reason="eod_backstop_degraded" if degraded else "eod_backstop",
        )
        try:
            execution.execute(exit_dec, use_bundle, bot_state)
        except Exception:
            logger.exception("loop: EOD backstop execute() raised for %s",
                             pos.strategy_key)


def run_cycle(bot_state: state.BotState, strategies: list, cycle_count: int) -> None:
    """Execute one 60s cycle."""
    bot_state.last_cycle_ts = datetime.datetime.now(datetime.UTC).isoformat()

    # Kill switch
    if session.kill_switch_active():
        _log_poll(None, "kill_switch_active", cycle_count)
        # Flatten any open positions — defensive: if the operator hit the
        # kill switch with positions open, we still exit them rather than
        # leave them dangling.
        bundle = build_bundle()
        if bundle is not None:
            for strat in strategies:
                pos = bot_state.get_position(strat.key)
                if pos is None:
                    continue
                from portfolio.mstr_loop.strategies.base import Decision
                exit_dec = Decision(
                    strategy_key=strat.key,
                    action="SELL",
                    direction=pos.direction,
                    cert_ob_id=pos.cert_ob_id,
                    rationale="kill_switch_flatten",
                    exit_reason="kill_switch",
                )
                execution.execute(exit_dec, bundle, bot_state)
        state.save_state(bot_state)
        return

    # Session window — no new entries outside hours, but we STILL run exits
    # in the EOD flatten window.
    in_window = session.in_session_window()
    in_eod = session.in_eod_flatten_window()
    if not in_window and not in_eod:
        _log_poll(None, "outside_session_window", cycle_count)
        state.save_state(bot_state)
        return

    bundle = build_bundle()

    # 2026-06-11 (audit B8 fix 3): loop-level EOD-flatten backstop. Runs
    # regardless of bundle usability so a stale/missing bundle in the EOD
    # window cannot silently carry a leveraged position overnight. Fires
    # before the bundle-unavailable / unusable early returns below.
    if in_eod:
        _eod_flatten_backstop(bot_state, strategies, cycle_count, bundle=bundle)
        if bundle is None or not bundle.is_usable():
            _log_poll(bundle, "eod_backstop_degraded", cycle_count)
            state.save_state(bot_state)
            return

    if bundle is None:
        _log_poll(None, "bundle_unavailable", cycle_count)
        state.save_state(bot_state)
        return

    # Update trail state before strategies evaluate exits — they rely on
    # accurate peak_underlying_price.
    execution.update_trail_state(bot_state, bundle)

    # Drawdown bookkeeping — must run every cycle so daily/weekly baselines
    # roll at calendar boundaries. Gate checks happen inside risk.any_*.
    if config.DRAWDOWN_CHECK_ENABLED:
        try:
            risk.update_drawdown_peaks(bot_state)
        except Exception:
            logger.exception("loop: drawdown peak update failed")

    _log_poll(bundle, "ok", cycle_count)

    # Cash budget across strategies — each strategy sees state.cash_sek at
    # start of cycle; _handle_buy mutates it atomically via execution.py.
    # Multi-strategy cash contention (one strategy over-allocating) is
    # prevented by the natural iteration order + the 95% cap in sizing.
    for strat in strategies:
        try:
            decision = strat.step(bundle, bot_state)
        except Exception:
            logger.exception("loop: strategy %s raised — skipping this cycle", strat.key)
            continue
        if decision is None:
            continue
        try:
            execution.execute(decision, bundle, bot_state)
        except Exception:
            logger.exception("loop: execute() raised for %s", strat.key)

    # Hourly Telegram status report (throttled inside telegram_report).
    try:
        telegram_report.maybe_send_hourly(bot_state)
    except Exception:
        logger.debug("loop: telegram hourly send failed", exc_info=True)

    state.save_state(bot_state)


def run_forever() -> None:
    """Entry point — infinite cycle loop."""
    logger.info("mstr_loop starting: PHASE=%s", config.PHASE)
    strategies = load_enabled_strategies()
    logger.info("mstr_loop strategies loaded: %s", [s.key for s in strategies])

    bot_state = state.load_state()
    # Phase-specific cash init: if first-ever run in paper mode, the file
    # doesn't exist and state.load_state returns a default_state with
    # INITIAL_PAPER_CASH_SEK already populated. Nothing more to do here.

    cycle_count = 0
    while True:
        cycle_started = time.monotonic()
        cycle_count += 1
        cycle_ok = False
        try:
            run_cycle(bot_state, strategies, cycle_count)
            cycle_ok = True
        except Exception:
            logger.exception("mstr_loop: run_cycle raised — continuing")

        # Heartbeat for the loop_health watchdog. Skipped on exception so
        # the watchdog sees staleness and pages. Mirrors crypto_loop.
        if cycle_ok:
            _write_heartbeat(bot_state, cycle_count)

        # Drift-free cadence
        elapsed = time.monotonic() - cycle_started
        remaining = config.CYCLE_INTERVAL_SEC - elapsed
        if remaining > 0:
            time.sleep(remaining)
