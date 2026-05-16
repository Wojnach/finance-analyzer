"""Tests for the 2026-05-15 cycle-modulo shadow throttle.

shadow_registry.should_run_this_cycle() and cycle_count_now() implement
a stateless modulo gate that keeps expensive shadow LLMs from blowing
the per-cycle budget when re-enabled.

These tests pin the contract:

* modulo=1 always runs.
* modulo=N runs once every N values, at the declared phase.
* Unknown / unregistered signals are not throttled (fall-through True).
* Malformed entries do not silently silence a signal.
* The UTC epoch-minute counter is monotonic and stateless.
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest

from portfolio import shadow_registry as sr


@pytest.fixture
def registry(tmp_path) -> pathlib.Path:
    p = tmp_path / "shadow_registry.json"
    p.write_text(
        json.dumps(
            {
                "shadows": {
                    "cheap": {
                        "entered_shadow_ts": "2026-05-15T00:00:00+00:00",
                        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55},
                        "status": "shadow",
                        "last_reviewed_ts": "2026-05-15T00:00:00+00:00",
                        "cycle_modulo": 1,
                        "cycle_phase": 0,
                    },
                    "every_3rd": {
                        "entered_shadow_ts": "2026-05-15T00:00:00+00:00",
                        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55},
                        "status": "shadow",
                        "last_reviewed_ts": "2026-05-15T00:00:00+00:00",
                        "cycle_modulo": 3,
                        "cycle_phase": 0,
                    },
                    "every_5th_phase_2": {
                        "entered_shadow_ts": "2026-05-15T00:00:00+00:00",
                        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55},
                        "status": "shadow",
                        "last_reviewed_ts": "2026-05-15T00:00:00+00:00",
                        "cycle_modulo": 5,
                        "cycle_phase": 2,
                    },
                    "broken_modulo": {
                        "entered_shadow_ts": "2026-05-15T00:00:00+00:00",
                        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55},
                        "status": "shadow",
                        "last_reviewed_ts": "2026-05-15T00:00:00+00:00",
                        "cycle_modulo": "not_a_number",
                        "cycle_phase": 0,
                    },
                    "zero_modulo": {
                        "entered_shadow_ts": "2026-05-15T00:00:00+00:00",
                        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55},
                        "status": "shadow",
                        "last_reviewed_ts": "2026-05-15T00:00:00+00:00",
                        "cycle_modulo": 0,
                        "cycle_phase": 0,
                    },
                    "phase_out_of_range": {
                        "entered_shadow_ts": "2026-05-15T00:00:00+00:00",
                        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55},
                        "status": "shadow",
                        "last_reviewed_ts": "2026-05-15T00:00:00+00:00",
                        "cycle_modulo": 3,
                        "cycle_phase": 99,
                    },
                }
            }
        )
    )
    return p


def test_modulo_1_always_runs(registry):
    for cycle in range(10):
        assert sr.should_run_this_cycle("cheap", cycle, path=registry) is True


def test_modulo_3_runs_every_third_cycle(registry):
    hits = [
        c for c in range(12)
        if sr.should_run_this_cycle("every_3rd", c, path=registry)
    ]
    assert hits == [0, 3, 6, 9]


def test_phase_offsets_skip_pattern(registry):
    """phase=2, modulo=5 means cycles 2, 7, 12, ... run; everything else
    is throttled."""
    hits = [
        c for c in range(15)
        if sr.should_run_this_cycle("every_5th_phase_2", c, path=registry)
    ]
    assert hits == [2, 7, 12]


def test_unknown_signal_is_not_throttled(registry):
    """Signals with no registry entry default to True so the throttle
    cannot accidentally silence a non-shadow signal."""
    assert sr.should_run_this_cycle("not_registered", 0, path=registry) is True
    assert sr.should_run_this_cycle("not_registered", 999, path=registry) is True


def test_malformed_modulo_falls_back_to_true(registry):
    """A corrupt cycle_modulo value must NOT silently silence a signal
    forever — we'd rather over-run than under-measure."""
    for cycle in range(10):
        assert sr.should_run_this_cycle("broken_modulo", cycle, path=registry) is True


def test_zero_modulo_is_treated_as_no_throttle(registry):
    """Modulo <= 0 is degenerate; falls back to no-throttle so the row
    count anomaly is visible in telemetry instead of being hidden."""
    for cycle in range(10):
        assert sr.should_run_this_cycle("zero_modulo", cycle, path=registry) is True


def test_phase_out_of_range_falls_back_to_true(registry):
    """phase=99 with modulo=3 is misconfigured; the helper must not
    silence the signal — surface the bug through row counts."""
    for cycle in range(10):
        assert sr.should_run_this_cycle("phase_out_of_range", cycle, path=registry) is True


def test_negative_cycle_count_is_safe(registry):
    """Defensive: a caller bug that passes negative cycle counts must
    not flip the modulo gate weirdly."""
    assert sr.should_run_this_cycle("every_3rd", -1, path=registry) is True
    assert sr.should_run_this_cycle("every_3rd", -100, path=registry) is True


def test_cycle_count_now_is_monotonic():
    """cycle_count_now() returns the current UTC epoch minute. It must
    not go backwards between calls."""
    a = sr.cycle_count_now()
    # Sleep less than a minute to avoid flaking the test
    time.sleep(0.05)
    b = sr.cycle_count_now()
    assert b >= a


def test_get_status_for_registered_signal(registry):
    assert sr.get_status("cheap", path=registry) == "shadow"


def test_get_status_for_unknown_signal_is_none(registry):
    assert sr.get_status("not_registered", path=registry) is None


def test_throttled_signal_does_not_emit_log_vote_row():
    """Reproduces 2026-05-17 data-quality regression: finance_llama
    throttled cycles were producing {BUY:0.25, HOLD:0.5, SELL:0.25}
    rows with conf=0.0 in llm_probability_log.jsonl because
    signal_engine.log_vote ran regardless of throttle skip. Calibration
    Brier scores were artificially inflated by these silent abstentions.

    Pin the behaviour: when extra_info[f"{sig}_throttled"]=True is set
    by the dispatch throttle, the corresponding log_vote() call must
    be skipped. The next phase-aligned cycle is the real measurement.
    """
    from portfolio import signal_engine
    src = pathlib.Path(signal_engine.__file__).read_text()
    # Verify the guard is present at the log_vote site
    assert "_throttled" in src and "continue" in src, "throttle guard missing in signal_engine"
    # Find the guard pattern proximate to the log_vote loop
    log_vote_idx = src.find("for sig_name in llm_signals():")
    assert log_vote_idx > 0
    guard_idx = src.find('extra_info.get(f"{sig_name}_throttled")', log_vote_idx)
    assert guard_idx > 0, "throttled-guard must sit inside the log_vote loop"
    # Guard must be before any conf= assignment so we never compute conf
    # for a throttled signal (cheap, but also documents intent)
    conf_idx = src.find("conf = extra_info.get", log_vote_idx)
    assert guard_idx < conf_idx, "throttled-guard must short-circuit before conf read"
