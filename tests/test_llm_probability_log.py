"""Tests for portfolio.llm_probability_log."""
from __future__ import annotations

import json

import pytest

from portfolio import llm_probability_log as mod


@pytest.fixture
def tmp_log(tmp_path):
    """Fresh log path per test — never touch the production file."""
    return tmp_path / "llm_probability_log.jsonl"


def _valid_probs():
    return {"BUY": 0.2, "HOLD": 0.5, "SELL": 0.3}


def test_happy_path_writes_row(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD", _valid_probs(),
        horizon="1d", chosen="HOLD", confidence=0.5, log_path=tmp_log,
    )
    assert ok is True
    rows = [json.loads(line) for line in tmp_log.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["signal"] == "ministral"
    assert row["ticker"] == "BTC-USD"
    assert row["horizon"] == "1d"
    assert row["probs"] == _valid_probs()
    assert row["chosen"] == "HOLD"
    assert row["confidence"] == 0.5
    assert row["tier"] is None
    assert "ts" in row


def test_chosen_defaults_to_argmax(tmp_log):
    mod.log_vote(
        "ministral", "BTC-USD",
        {"BUY": 0.6, "HOLD": 0.3, "SELL": 0.1},
        log_path=tmp_log,
    )
    row = json.loads(tmp_log.read_text().splitlines()[0])
    assert row["chosen"] == "BUY"


def test_non_llm_signal_is_noop(tmp_log):
    ok = mod.log_vote("rsi", "BTC-USD", _valid_probs(), log_path=tmp_log)
    assert ok is False
    assert not tmp_log.exists()


def test_rejects_probs_not_summing_to_one(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD",
        {"BUY": 0.2, "HOLD": 0.2, "SELL": 0.2},  # sums to 0.6
        log_path=tmp_log,
    )
    assert ok is False
    assert not tmp_log.exists()


def test_rejects_missing_keys(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD",
        {"BUY": 0.5, "SELL": 0.5},  # missing HOLD
        log_path=tmp_log,
    )
    assert ok is False


def test_rejects_negative_probability(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD",
        {"BUY": -0.1, "HOLD": 0.6, "SELL": 0.5},
        log_path=tmp_log,
    )
    assert ok is False


def test_rejects_non_numeric(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD",
        {"BUY": "high", "HOLD": 0.5, "SELL": 0.5},  # type: ignore[dict-item]
        log_path=tmp_log,
    )
    assert ok is False


def test_rejects_bad_chosen(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD", _valid_probs(),
        chosen="MAYBE", log_path=tmp_log,
    )
    assert ok is False


def test_rejects_confidence_out_of_range(tmp_log):
    ok = mod.log_vote(
        "ministral", "BTC-USD", _valid_probs(),
        confidence=1.5, log_path=tmp_log,
    )
    assert ok is False
    ok2 = mod.log_vote(
        "ministral", "BTC-USD", _valid_probs(),
        confidence=-0.01, log_path=tmp_log,
    )
    assert ok2 is False


def test_tier_field_round_trips_for_cascade(tmp_log):
    mod.log_vote(
        "claude_fundamental", "MSTR", _valid_probs(),
        tier="opus", log_path=tmp_log,
    )
    row = json.loads(tmp_log.read_text().splitlines()[0])
    assert row["tier"] == "opus"


def test_multiple_rows_append(tmp_log):
    for _ in range(3):
        mod.log_vote("ministral", "BTC-USD", _valid_probs(), log_path=tmp_log)
    lines = tmp_log.read_text().splitlines()
    assert len(lines) == 3


def test_is_llm_signal_public_helper():
    assert mod.is_llm_signal("ministral") is True
    assert mod.is_llm_signal("claude_fundamental") is True
    assert mod.is_llm_signal("rsi") is False
    assert mod.is_llm_signal("macd") is False


def test_llm_signals_snapshot_is_immutable():
    snapshot = mod.llm_signals()
    assert isinstance(snapshot, frozenset)
    assert "ministral" in snapshot
    with pytest.raises(AttributeError):
        snapshot.add("new_signal")  # type: ignore[attr-defined]


def test_never_raises_on_io_failure(tmp_log, monkeypatch):
    """If the append fails, log_vote must return False, not raise."""
    def _boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(mod, "atomic_append_jsonl", _boom)
    ok = mod.log_vote("ministral", "BTC-USD", _valid_probs(), log_path=tmp_log)
    assert ok is False
