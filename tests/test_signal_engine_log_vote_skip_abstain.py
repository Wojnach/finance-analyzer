"""Tests that signal_engine skips log_vote on abstain/scaffold results.

Companion guard to `test_shadow_cycle_throttle.test_throttled_signal_does_not_emit_log_vote_row`
(2026-05-17). This extends the same protection to:

1. `confidence <= 0` — canonical abstain marker from
   `portfolio/signals/{finance_llama,cryptotrader_lm,meta_trader}.py`
   `_abstain()` helper.
2. `indicators.feature_unavailable is True` — explicit scaffold marker
   set when `_FEATURE_AVAILABLE=False` returns the no-op shape.

Without these guards every scaffold dispatch poisons
`llm_probability_log.jsonl` with HOLD/conf=0 rows; combined with the
~64% HOLD-base-rate in outcome backfill, the auto-promotion gate
trivially passes broken shadows at fake "64% accuracy" (the bug fixed
in `.worktrees/shadow-gate-lora-20260518/docs/PLAN.md` 2026-05-18).
"""
from __future__ import annotations

import pathlib


def test_signal_engine_skips_log_vote_for_conf_zero():
    """Static check — the abstain guard exists in the log_vote loop and
    short-circuits before the log_vote() call. We assert on source text
    so a refactor that moves the guard to the wrong scope (e.g. above
    the for loop, skipping all signals; or after log_vote, doing
    nothing) trips the test."""
    from portfolio import signal_engine
    src = pathlib.Path(signal_engine.__file__).read_text()

    log_vote_loop_idx = src.find("for sig_name in llm_signals():")
    assert log_vote_loop_idx > 0, "log_vote loop missing"

    # Find the log_vote( call site for the actual write.
    write_call_idx = src.find("log_vote(\n", log_vote_loop_idx)
    assert write_call_idx > 0, "log_vote call missing"

    # The abstain guard MUST appear between the for-loop header and the
    # log_vote write call.
    abstain_marker = "abstain_conf_zero"
    abstain_idx = src.find(abstain_marker, log_vote_loop_idx)
    assert abstain_idx > 0, "abstain_conf_zero guard missing"
    assert log_vote_loop_idx < abstain_idx < write_call_idx, (
        "abstain guard must sit inside the log_vote loop, "
        "before the write call"
    )

    feature_marker = "feature_unavailable"
    feature_idx = src.find(feature_marker, log_vote_loop_idx)
    assert feature_idx > 0, "feature_unavailable guard missing"
    assert log_vote_loop_idx < feature_idx < write_call_idx, (
        "feature_unavailable guard must sit inside the log_vote loop"
    )


def test_signal_engine_skip_logs_reason():
    """The skip path must emit a single `[log_vote_skipped]` info line per
    skipped row. This is the observability hook from premortem N4 — without
    it, a silent skip on a legitimate prediction would be invisible until
    the daily cron failed to find new rows."""
    from portfolio import signal_engine
    src = pathlib.Path(signal_engine.__file__).read_text()
    log_vote_idx = src.find("for sig_name in llm_signals():")
    skip_log_idx = src.find("[log_vote_skipped]", log_vote_idx)
    assert skip_log_idx > 0, "[log_vote_skipped] info log line missing"
    # Confirm it's at logger.info level (vs debug) so it's surfaced by
    # the default scheduled-task log capture.
    snippet = src[max(0, skip_log_idx - 100):skip_log_idx]
    assert "logger.info" in snippet, "skip-log must be at INFO level for visibility"
