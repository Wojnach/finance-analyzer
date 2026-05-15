"""Tests for portfolio.escalation_gate (Batch E)."""

from __future__ import annotations

import json

import pytest

from portfolio import escalation_gate


def _signals_fixture():
    return {
        "BTC-USD": {
            "enhanced_signals": {
                "rsi": {"action": "BUY"},
                "macd": {"action": "BUY"},
                "ema": {"action": "HOLD"},
                "bb": {"action": "SELL"},
                "volume": {"action": "BUY"},
                "trend": {"action": "BUY"},
            },
            "extra": {"funding_action": "BUY"},
        }
    }


def _make_runner(payload):
    """Return a runner returning a fixed string payload."""
    def _r(_prompt):
        return payload
    return _r


def _make_raising_runner(exc):
    def _r(_prompt):
        raise exc
    return _r


def test_runner_says_not_escalate_high_conf(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(json.dumps({
        "escalate": False, "confidence": 0.8, "why": "ranging tape"
    }))
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD rsi flipped BUY->HOLD"],
        tier=1,
        signals=_signals_fixture(),
        prices={"BTC-USD": 60000.0},
        held_positions={"patient": [], "bold": []},
        runner=runner,
        log_path=log,
    )
    assert esc is False
    assert conf == pytest.approx(0.8)
    assert why == "ranging tape"


def test_runner_says_not_escalate_low_conf(tmp_path):
    """Module returns the LLM's vote+confidence as-is. The min_score
    threshold check is the caller's responsibility (main.py)."""
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(json.dumps({
        "escalate": False, "confidence": 0.3, "why": "weak"
    }))
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD rsi flipped"],
        tier=1,
        signals=_signals_fixture(),
        prices={},
        held_positions={"patient": [], "bold": []},
        runner=runner,
        log_path=log,
    )
    assert esc is False
    assert conf == pytest.approx(0.3)
    assert why == "weak"


def test_runner_says_escalate(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(json.dumps({
        "escalate": True, "confidence": 0.9, "why": "conflicting signals"
    }))
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD top5 split"],
        tier=2,
        signals=_signals_fixture(),
        prices={},
        held_positions={"patient": ["BTC-USD"], "bold": []},
        runner=runner,
        log_path=log,
    )
    assert esc is True
    assert conf == pytest.approx(0.9)


def test_malformed_json_fails_open(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner("this is not json at all")
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD foo"],
        tier=1,
        signals={},
        prices={},
        held_positions={},
        runner=runner,
        log_path=log,
    )
    assert esc is True
    assert conf == 0.0
    assert why == "ministral_unavailable"


def test_runner_raises_fails_open(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_raising_runner(RuntimeError("server down"))
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD foo"],
        tier=1,
        signals={},
        prices={},
        held_positions={},
        runner=runner,
        log_path=log,
    )
    assert esc is True
    assert conf == 0.0
    assert why == "ministral_unavailable"


def test_empty_reasons_fails_open(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(json.dumps({"escalate": False, "confidence": 0.9, "why": "x"}))
    esc, conf, why = escalation_gate.should_escalate(
        [],
        tier=1,
        signals={},
        prices={},
        held_positions={},
        runner=runner,
        log_path=log,
    )
    assert esc is True
    assert conf == 0.0
    assert why == "ministral_unavailable"


def test_log_appends_one_row_per_call(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(json.dumps({"escalate": False, "confidence": 0.7, "why": "ok"}))
    for _ in range(3):
        escalation_gate.should_escalate(
            ["BTC-USD foo"],
            tier=1,
            signals=_signals_fixture(),
            prices={},
            held_positions={"patient": [], "bold": []},
            runner=runner,
            log_path=log,
        )
    with open(log) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 3
    for row in rows:
        assert row["escalate"] is False
        assert row["confidence"] == pytest.approx(0.7)
        assert row["why"] == "ok"
        assert row["tier"] == 1
        assert row["tickers"] == ["BTC-USD"]
        assert "ts" in row


def test_json_embedded_in_text_is_parsed(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(
        'Here is my answer:\n{"escalate": false, "confidence": 0.6, "why": "calm"}\nthanks'
    )
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD foo"],
        tier=1,
        signals={},
        prices={},
        held_positions={},
        runner=runner,
        log_path=log,
    )
    assert esc is False
    assert conf == pytest.approx(0.6)
    assert why == "calm"


def test_missing_escalate_key_fails_open(tmp_path):
    log = str(tmp_path / "gate.jsonl")
    runner = _make_runner(json.dumps({"confidence": 0.9, "why": "x"}))
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD foo"],
        tier=1,
        signals={},
        prices={},
        held_positions={},
        runner=runner,
        log_path=log,
    )
    assert esc is True
    assert conf == 0.0
    assert why == "ministral_unavailable"


def test_runner_timeout_fails_open(tmp_path):
    """2026-05-15: hanging runner triggers 10s timeout, gate fails open."""
    import time as _t
    log = str(tmp_path / "gate.jsonl")

    def hang(_prompt):
        _t.sleep(30)
        return "{}"

    t0 = _t.monotonic()
    esc, conf, why = escalation_gate.should_escalate(
        ["BTC-USD foo"],
        tier=1,
        signals={},
        prices={},
        held_positions={},
        runner=hang,
        log_path=log,
    )
    elapsed = _t.monotonic() - t0
    assert esc is True
    assert conf == 0.0
    assert why == "ministral_unavailable"
    assert elapsed < 12, f"gate did not time out promptly: {elapsed:.1f}s"
