"""Tests for Big Bet Layer 2 evaluation."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from portfolio.bigbet import (
    _build_eval_prompt,
    _format_alert,
    _parse_eval_response,
    invoke_layer2_eval,
)


# --- Fixtures ---

def _make_signals(ticker="BTC-USD"):
    return {
        ticker: {
            "indicators": {
                "rsi": 22.0,
                "macd_hist": -5.0,
                "price_vs_bb": "below_lower",
                "atr_pct": 3.2,
            },
            "extra": {
                "_buy_count": 4,
                "_sell_count": 1,
                "_total_applicable": 11,
                "fear_greed": 8,
                "volume_ratio": 2.5,
            },
        }
    }


def _make_tf_data(ticker="BTC-USD"):
    labels = ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]
    actions = ["SELL", "HOLD", "SELL", "HOLD", "SELL", "SELL", "HOLD"]
    return {ticker: [(l, {"action": a}) for l, a in zip(labels, actions)]}


PRICES = {"BTC-USD": 65000.0}
CONFIG = {"telegram": {"token": "fake", "chat_id": "123"}}
CONDITIONS = ["RSI 22 (oversold) on 15m", "Below lower BB on Now, 12h", "F&G: 8 (Extreme Fear)"]


# --- _parse_eval_response tests ---

def test_eval_returns_probability():
    output = "PROBABILITY: 7/10\nREASONING: Good setup with volume confirmation and extreme fear."
    prob, reason = _parse_eval_response(output)
    assert prob == 7
    assert "Good setup" in reason


def test_eval_parse_failure():
    output = "I think this is a great trade opportunity!"
    prob, reason = _parse_eval_response(output)
    assert prob is None
    assert reason == ""


def test_eval_parse_clamps_range():
    output = "PROBABILITY: 15/10\nREASONING: Off the charts!"
    prob, reason = _parse_eval_response(output)
    assert prob == 10  # clamped to max

    output2 = "PROBABILITY: 0/10\nREASONING: Terrible."
    prob2, reason2 = _parse_eval_response(output2)
    assert prob2 == 1  # clamped to min


# --- invoke_layer2_eval tests ---

@patch("portfolio.bigbet.subprocess.run")
@patch.dict("os.environ", {}, clear=False)
def test_invoke_eval_success(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr("portfolio.bigbet.EVAL_LOG_FILE", tmp_path / "log.jsonl")
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="PROBABILITY: 7/10\nREASONING: Strong capitulation setup.",
    )
    prob, reason = invoke_layer2_eval(
        "BTC-USD", "BULL", CONDITIONS, _make_signals(), _make_tf_data(), PRICES, CONFIG
    )
    assert prob == 7
    assert "capitulation" in reason
    mock_run.assert_called_once()
    # Verify log was written
    assert (tmp_path / "log.jsonl").exists()


@patch("portfolio.bigbet.subprocess.run")
@patch.dict("os.environ", {}, clear=False)
def test_invoke_eval_timeout_fallback(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr("portfolio.bigbet.EVAL_LOG_FILE", tmp_path / "log.jsonl")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
    prob, reason = invoke_layer2_eval(
        "BTC-USD", "BULL", CONDITIONS, _make_signals(), _make_tf_data(), PRICES, CONFIG
    )
    assert prob is None
    assert reason == ""


# --- _format_alert tests ---

def test_format_alert_with_evaluation():
    msg = _format_alert(
        "BTC-USD", "BULL", CONDITIONS, PRICES, 10.5, {"fg": 8},
        probability=7, l2_reasoning="Strong capitulation setup.",
    )
    assert "Claude: 7/10" in msg
    assert "Strong capitulation setup" in msg


def test_format_alert_without_evaluation():
    msg = _format_alert(
        "BTC-USD", "BULL", CONDITIONS, PRICES, 10.5, {"fg": 8},
    )
    assert "Claude:" not in msg
    # Still has all the normal content
    assert "BIG BET: BULL BTC-USD" in msg
    assert "3/6 conditions met" in msg


# --- _build_eval_prompt tests ---

def test_build_eval_prompt_content():
    prompt = _build_eval_prompt(
        "BTC-USD", "BULL", CONDITIONS, _make_signals(), _make_tf_data(), PRICES
    )
    assert "BTC-USD" in prompt
    assert "BULL" in prompt
    assert "4B/1S/6H" in prompt
    assert "PROBABILITY:" in prompt
    assert "RSI 22" in prompt
