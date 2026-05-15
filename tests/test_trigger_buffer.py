"""Tests for portfolio/trigger_buffer.py — Item 8 of docs/PLAN.md."""

from __future__ import annotations

import pytest

from portfolio import trigger_buffer as tb


# --- parse_reason ---------------------------------------------------------


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("BTC-USD consensus BUY (54%)", ("BTC-USD", "consensus")),
        ("XAU-USD flipped BUY->HOLD (sustained)", ("XAU-USD", "flipped")),
        ("MSTR moved 2.7% down", ("MSTR", "moved")),
        ("XAG-USD sentiment positive->negative (sustained)", ("XAG-USD", "sentiment")),
        ("startup", ("", "startup")),
        ("first_of_day", ("", "first_of_day")),
        ("periodic_review", ("", "periodic_review")),
        ("F&G crossed below 20", ("", "F&G_extreme")),
        ("post-trade Patient BTC-USD", ("BTC-USD", "post_trade")),
        ("some random thing", ("", "other")),
    ],
)
def test_parse_reason(reason, expected):
    assert tb.parse_reason(reason) == expected


# --- add + flush_due ------------------------------------------------------


def _path(tmp_path):
    return tmp_path / "buf.json"


def test_add_then_flush_within_window_returns_nothing(tmp_path):
    p = _path(tmp_path)
    tb.add(["BTC-USD consensus BUY (54%)"], now_ts=1000.0, state_path=p)
    assert tb.flush_due(now_ts=1100.0, state_path=p) == []
    # state preserved
    assert tb.buffer_size(state_path=p) == 1


def test_add_then_flush_after_window_returns_merged(tmp_path):
    p = _path(tmp_path)
    tb.add(["BTC-USD consensus BUY (54%)"], now_ts=1000.0, state_path=p)
    out = tb.flush_due(now_ts=1301.0, state_path=p)
    assert out == ["BTC-USD consensus BUY (54%)"]
    assert tb.buffer_size(state_path=p) == 0


def test_duplicate_reasons_collapse_with_count(tmp_path):
    p = _path(tmp_path)
    r = "BTC-USD consensus BUY (54%)"
    tb.add([r, r, r], now_ts=1000.0, state_path=p)
    out = tb.flush_due(now_ts=1301.0, state_path=p)
    assert out == [f"{r} (x3)"]


def test_distinct_reasons_same_bucket_merge_single_entry(tmp_path):
    p = _path(tmp_path)
    tb.add(
        [
            "BTC-USD consensus BUY (54%)",
            "BTC-USD consensus BUY (60%)",
        ],
        now_ts=1000.0,
        state_path=p,
    )
    out = tb.flush_due(now_ts=1301.0, state_path=p)
    assert len(out) == 1
    # Representative is shortest; total count 2 → "(x2)"
    assert "(x2)" in out[0]
    assert "BTC-USD consensus BUY" in out[0]


def test_escalation_force_flush(tmp_path):
    p = _path(tmp_path)
    tb.add(
        ["BTC-USD consensus BUY (54%)", "XAU-USD moved 1.1% up"],
        now_ts=1000.0,
        state_path=p,
    )
    # Mid-window — would not flush normally.
    assert tb.flush_due(now_ts=1100.0, state_path=p) == []
    # Add escalation.
    tb.add(["first_of_day"], now_ts=1110.0, state_path=p)
    out = tb.flush_due(now_ts=1115.0, state_path=p)
    assert len(out) == 3  # 3 distinct (ticker, reason_type) buckets
    assert tb.buffer_size(state_path=p) == 0


def test_disk_roundtrip_preserves_state(tmp_path):
    p = _path(tmp_path)
    tb.add(["BTC-USD consensus BUY (54%)"], now_ts=1000.0, state_path=p)
    # Simulate new process — fresh load.
    assert tb.buffer_size(state_path=p) == 1
    tb.add(["MSTR moved 2.7% down"], now_ts=1050.0, state_path=p)
    assert tb.buffer_size(state_path=p) == 2
    out = tb.flush_due(now_ts=2000.0, state_path=p)
    assert len(out) == 2


def test_empty_buffer_flush_returns_empty(tmp_path):
    p = _path(tmp_path)
    assert tb.flush_due(now_ts=1000.0, state_path=p) == []
    assert tb.buffer_size(state_path=p) == 0


def test_partial_flush_keeps_unexpired(tmp_path):
    p = _path(tmp_path)
    tb.add(["BTC-USD consensus BUY (54%)"], now_ts=1000.0, state_path=p)
    tb.add(["MSTR moved 2.7% down"], now_ts=1200.0, state_path=p)
    # At t=1305: first entry (age 305) flushes, second (age 105) stays.
    out = tb.flush_due(now_ts=1305.0, state_path=p)
    assert out == ["BTC-USD consensus BUY (54%)"]
    assert tb.buffer_size(state_path=p) == 1
