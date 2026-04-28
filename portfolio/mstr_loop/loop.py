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
    if not in_window and not session.in_eod_flatten_window():
        _log_poll(None, "outside_session_window", cycle_count)
        state.save_state(bot_state)
        return

    bundle = build_bundle()
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
        try:
            run_cycle(bot_state, strategies, cycle_count)
        except Exception:
            logger.exception("mstr_loop: run_cycle raised — continuing")

        # Drift-free cadence
        elapsed = time.monotonic() - cycle_started
        remaining = config.CYCLE_INTERVAL_SEC - elapsed
        if remaining > 0:
            time.sleep(remaining)
