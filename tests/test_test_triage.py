"""Tests for `scripts/test_triage.py`.

The suite reports ~72 failures on the Deck against a `docs/TESTING.md` baseline
that says 24, written when the suite had 7,730 tests instead of 11,666. Nobody
can tell a regression from noise in that state, and re-deriving it by hand costs
a 3.5-minute run plus a pile of one-off greps every time.

This tests the parsing and classification, which is all the logic worth testing
— running pytest and shelling out are thin wrappers over subprocess.

The decisive output is the flake split: a failure that fails under `-n auto` but
passes when its file runs alone is an xdist isolation artifact, not a bug. Only
what fails BOTH ways is real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import test_triage as tt  # noqa: E402

_SAMPLE = """\
FAILED tests/test_bert_sentiment.py::TestLoad::test_model_loads - RuntimeError
FAILED tests/test_llm_prewarmer.py::test_prewarm_after_qwen3_targets_fingpt
FAILED tests/test_consensus.py::TestStockConsensus::test_stock_buy_with_3_voters
FAILED tests/test_metals.py::TestMetalsSignalConfig::test_crypto_total_applicable
FAILED tests/test_widget.py::test_something_new - AssertionError: nope
ERROR tests/test_broken.py::test_import - ImportError
72 failed, 11636 passed, 30 skipped, 3 warnings in 208.11s (0:03:28)
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_every_failed_and_error_nodeid():
    got = tt.parse_failures(_SAMPLE)
    assert "tests/test_bert_sentiment.py::TestLoad::test_model_loads" in got
    assert "tests/test_broken.py::test_import" in got
    assert len(got) == 6


def test_parse_failures_strips_the_trailing_reason_text():
    got = tt.parse_failures("FAILED tests/a.py::test_b - AssertionError: x != y\n")
    assert got == ["tests/a.py::test_b"]


def test_parse_failures_is_empty_on_a_clean_run():
    assert tt.parse_failures("1234 passed in 10s\n") == []


# ---------------------------------------------------------------------------
# 2026-08-21 regression: a captured LOG line at ERROR level was parsed as a
# node id, fed to pytest as a path, pytest died with "no tests ran", the serial
# confirm came back with zero failures, and every real failure was therefore
# classified as "passes serially" — REAL=0 on a suite with 75 failures.
# One bad regex match inverted the entire verdict into a false all-clear.
# ---------------------------------------------------------------------------

_LOG_NOISE = """\
ERROR    portfolio.http_retry:http_retry.py:103 HTTP 429 from https://x/v1/mvrv/last after 0 retries
ERROR: file or directory not found: portfolio.http_retry:http_retry.py:103
ERROR    root:module.py:12 something exploded
FAILED tests/real.py::test_actually_failed - AssertionError
"""


def test_captured_log_lines_are_not_mistaken_for_node_ids():
    got = tt.parse_failures(_LOG_NOISE)
    assert got == ["tests/real.py::test_actually_failed"]


def test_pytest_usage_error_line_is_not_a_node_id():
    assert tt.parse_failures("ERROR: file or directory not found: foo.py:103\n") == []


def test_collection_error_without_a_test_name_is_still_captured():
    """`ERROR tests/foo.py - ImportError` is a real collection failure."""
    assert tt.parse_failures("ERROR tests/foo.py - ImportError: no module\n") == [
        "tests/foo.py"
    ]


def test_every_parsed_nodeid_points_at_a_python_file():
    for nodeid in tt.parse_failures(_SAMPLE + _LOG_NOISE):
        assert ".py" in nodeid, nodeid
        assert ":" not in nodeid.split("::", 1)[0], f"path half has a colon: {nodeid}"


# ---------------------------------------------------------------------------
# An unrunnable serial confirm must never be read as "everything passed"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "no tests ran in 0.11s\nERROR: file or directory not found: bogus\n",
        "ERROR: file or directory not found: bogus\n",
        "",
        "INTERNALERROR> something\n",
    ],
)
def test_serial_run_ok_rejects_unusable_output(text):
    assert tt.serial_run_ok(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "5 failed, 20 passed in 30s\n",
        "50 passed in 12s\n",
        "1 failed, 1 passed, 2 skipped in 3.00s\n",
    ],
)
def test_serial_run_ok_accepts_a_real_run(text):
    assert tt.serial_run_ok(text) is True


def test_split_flakes_refuses_to_clear_failures_when_serial_never_ran():
    """The exact 2026-08-21 shape: real failures, empty serial output."""
    parallel = ["tests/a.py::t1", "tests/b.py::t2"]
    split = tt.split_flakes(parallel, [], serial_ok=False)
    assert split["real"] == parallel, "unrunnable confirm must not clear anything"
    assert split["xdist_flake"] == []
    assert split["confirm_failed"] is True


def test_split_flakes_marks_confirm_succeeded_when_serial_ran():
    split = tt.split_flakes(["tests/a.py::t1"], ["tests/a.py::t1"], serial_ok=True)
    assert split["confirm_failed"] is False
    assert split["real"] == ["tests/a.py::t1"]


def test_parses_the_summary_counts():
    s = tt.parse_summary(_SAMPLE)
    assert s["failed"] == 72
    assert s["passed"] == 11636
    assert s["skipped"] == 30
    assert s["duration_s"] == pytest.approx(208.11)


def test_parse_summary_handles_an_all_green_run():
    s = tt.parse_summary("11700 passed, 30 skipped in 190.00s\n")
    assert s["failed"] == 0 and s["passed"] == 11700


def test_parse_summary_returns_none_fields_when_there_is_no_summary_line():
    assert tt.parse_summary("nothing useful here") == {}


# ---------------------------------------------------------------------------
# Classification — the point is that `unknown` is the bucket that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nodeid,bucket",
    [
        ("tests/test_bert_sentiment.py::T::t", "llm-infra"),
        ("tests/test_llm_prewarmer.py::t", "llm-infra"),
        ("tests/test_llama_server.py::t", "llm-infra"),
        ("tests/test_model_upgrades.py::TestQwen3Signal::t", "llm-infra"),
        ("tests/test_forecast_timeout.py::t", "llm-infra"),
        ("tests/test_chronos_gpu_gate.py::t", "llm-infra"),
        ("tests/test_metals_loop_autonomous.py::t", "metals-loop"),
        ("tests/test_metals_swing_trader.py::t", "metals-loop"),
        ("tests/test_metals.py::T::test_crypto_total_applicable", "applicable-count"),
        (
            "tests/test_signal_pipeline.py::T::test_stock_vote_counts",
            "applicable-count",
        ),
        ("tests/test_widget.py::test_something_new", "unknown"),
    ],
)
def test_classify_assigns_the_expected_bucket(nodeid, bucket):
    assert tt.classify(nodeid) == bucket


def test_applicable_count_wins_over_the_file_level_metals_rule():
    """A metals file can still hold an applicable-count test; the specific
    signal must beat the general one or the interesting case gets buried."""
    assert (
        tt.classify("tests/test_metals.py::T::test_stocks_total_applicable")
        == "applicable-count"
    )
    assert tt.classify("tests/test_metals.py::T::test_entry_logic") == "metals-loop"


def test_bucketize_groups_and_counts():
    got = tt.bucketize(tt.parse_failures(_SAMPLE))
    assert got["llm-infra"] == [
        "tests/test_bert_sentiment.py::TestLoad::test_model_loads",
        "tests/test_llm_prewarmer.py::test_prewarm_after_qwen3_targets_fingpt",
    ]
    assert got["applicable-count"] == [
        "tests/test_metals.py::TestMetalsSignalConfig::test_crypto_total_applicable"
    ]
    assert len(got["unknown"]) == 3  # consensus, widget, broken


# ---------------------------------------------------------------------------
# The flake split — the decisive output
# ---------------------------------------------------------------------------


def test_real_failures_are_those_that_fail_both_ways():
    parallel = ["tests/a.py::t1", "tests/a.py::t2", "tests/b.py::t3"]
    serial = ["tests/a.py::t2"]
    split = tt.split_flakes(parallel, serial)
    assert split["real"] == ["tests/a.py::t2"]
    assert split["xdist_flake"] == ["tests/a.py::t1", "tests/b.py::t3"]


def test_a_test_failing_only_serially_is_reported_not_dropped():
    """Order-dependent the other way round — rare but must not vanish."""
    split = tt.split_flakes(["tests/a.py::t1"], ["tests/a.py::t1", "tests/a.py::t9"])
    assert split["real"] == ["tests/a.py::t1"]
    assert split["serial_only"] == ["tests/a.py::t9"]


def test_split_is_empty_when_nothing_failed():
    split = tt.split_flakes([], [])
    assert split["real"] == [] and split["xdist_flake"] == []


def test_files_to_recheck_dedupes_and_keeps_order():
    got = tt.files_to_recheck(["tests/b.py::t1", "tests/a.py::t2", "tests/b.py::t3"])
    assert got == ["tests/b.py", "tests/a.py"]


# ---------------------------------------------------------------------------
# Baseline block rendering — what lands in docs/TESTING.md
# ---------------------------------------------------------------------------


def test_baseline_block_carries_the_counts_and_the_host():
    block = tt.render_baseline(
        summary={"passed": 11636, "failed": 72, "skipped": 30, "duration_s": 208.11},
        split={
            "real": ["tests/a.py::t1"],
            "xdist_flake": ["tests/b.py::t2"],
            "serial_only": [],
        },
        buckets={"llm-infra": ["tests/x.py::t"], "unknown": []},
        host="steamdeck",
    )
    assert "11636" in block and "72" in block
    assert "steamdeck" in block
    assert tt.BASELINE_START in block and tt.BASELINE_END in block


def test_replace_baseline_swaps_an_existing_block_rather_than_appending():
    doc = (
        f"# Doc\n\nintro\n\n{tt.BASELINE_START}\nold junk\n{tt.BASELINE_END}\n\ntail\n"
    )
    out = tt.replace_baseline(doc, "NEWBLOCK")
    assert "old junk" not in out
    assert "NEWBLOCK" in out
    assert out.count(tt.BASELINE_START) == 0 or "NEWBLOCK" in out
    assert out.startswith("# Doc") and out.rstrip().endswith("tail")


def test_replace_baseline_inserts_when_no_block_exists_yet():
    out = tt.replace_baseline("# Doc\n\nbody\n", "NEWBLOCK")
    assert "NEWBLOCK" in out
    assert out.startswith("# Doc")
