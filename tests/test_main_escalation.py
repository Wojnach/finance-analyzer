"""Tests for portfolio/escalation_router.py — Batch D / item 6 of docs/PLAN.md."""

from __future__ import annotations

import json
from unittest.mock import patch

from portfolio import escalation_router as er


# ---------- helpers ----------

def _votes(action_map: dict[str, str]) -> dict:
    """Build a signals dict for one ticker from {signal_name: action}."""
    enh = {name: {"action": a, "confidence": 0.7} for name, a in action_map.items()}
    return {"BTC-USD": {"enhanced_signals": enh, "extra": {}}}


def _acc_cache(names_desc: list[str]) -> dict:
    """Synthesise an accuracy_cache with given signal names ranked by accuracy."""
    one_d = {}
    for i, n in enumerate(names_desc):
        one_d[n] = {
            "accuracy": 0.8 - i * 0.01,
            "samples": 1000,
            "total": 1000,
            "correct": 800,
        }
    return {"1d": one_d}


def _state(tmp_path, drawdown_patient=0.0, drawdown_bold=0.0):
    p = tmp_path / "router_state.json"
    p.write_text(json.dumps({
        "drawdown_patient": drawdown_patient,
        "drawdown_bold": drawdown_bold,
    }))
    return str(p)


# ---------- tier 3 ----------

def test_tier3_escalates(tmp_path):
    state = _state(tmp_path)
    esc, why = er.should_escalate_to_claude(
        ["F&G crossed below 20"], tier=3, signals={}, state_path=state,
    )
    assert esc is True
    assert why == "tier3"


# ---------- healthy baseline ----------

def test_tier1_balanced_no_position_no_escalation(tmp_path):
    state = _state(tmp_path, drawdown_patient=3.0, drawdown_bold=3.0)
    acc = _acc_cache(["rsi", "macd", "ema", "bb", "fear_greed"])
    signals = _votes({"rsi": "BUY", "macd": "BUY", "ema": "BUY",
                      "bb": "BUY", "fear_greed": "BUY"})
    with patch.object(er, "_ticker_held", return_value=False):
        esc, why = er.should_escalate_to_claude(
            ["BTC-USD consensus BUY (54%)"], tier=1, signals=signals,
            accuracy_cache=acc, state_path=state,
            current_drawdown_patient=3.5, current_drawdown_bold=3.5,
        )
    assert esc is False
    assert why == "autonomous"


# ---------- drawdown delta ----------

def test_drawdown_delta_above_threshold_escalates(tmp_path):
    state = _state(tmp_path, drawdown_patient=2.0, drawdown_bold=2.0)
    acc = _acc_cache(["rsi"])
    with patch.object(er, "_ticker_held", return_value=False):
        esc, why = er.should_escalate_to_claude(
            ["BTC-USD moved 1.2% up"], tier=1, signals={},
            accuracy_cache=acc, state_path=state,
            escalate_drawdown_pct=5.0,
            current_drawdown_patient=8.5, current_drawdown_bold=2.0,
        )
    assert esc is True
    assert why.startswith("drawdown_")


def test_drawdown_same_no_escalation(tmp_path):
    state = _state(tmp_path, drawdown_patient=4.0, drawdown_bold=3.0)
    er.record_decision_snapshot(4.0, 3.0, state_path=state)
    with patch.object(er, "_ticker_held", return_value=False):
        esc, why = er.should_escalate_to_claude(
            ["BTC-USD moved 1.0% up"], tier=1, signals={},
            state_path=state,
            current_drawdown_patient=4.0, current_drawdown_bold=3.0,
        )
    assert esc is False


# ---------- top-5 split ----------

def test_top5_split_escalates(tmp_path):
    state = _state(tmp_path)
    acc = _acc_cache(["rsi", "macd", "ema", "bb", "fear_greed"])
    # 3 BUY, 2 SELL among top-5
    signals = _votes({"rsi": "BUY", "macd": "BUY", "ema": "BUY",
                      "bb": "SELL", "fear_greed": "SELL"})
    with patch.object(er, "_ticker_held", return_value=False):
        esc, why = er.should_escalate_to_claude(
            ["BTC-USD consensus BUY (54%)"], tier=2, signals=signals,
            accuracy_cache=acc, state_path=state,
            current_drawdown_patient=0.0, current_drawdown_bold=0.0,
        )
    assert esc is True
    assert "top5_split_BTC-USD" in why


# ---------- held + sell flip ----------

def test_held_position_sell_flip_escalates(tmp_path):
    state = _state(tmp_path)
    signals = _votes({"rsi": "HOLD"})
    acc = _acc_cache(["rsi"])
    with patch.object(er, "_ticker_held", return_value=True):
        esc, why = er.should_escalate_to_claude(
            ["XAG-USD flipped BUY->HOLD (sustained)"],
            tier=1, signals={"XAG-USD": signals["BTC-USD"]},
            accuracy_cache=acc, state_path=state,
            current_drawdown_patient=0.0, current_drawdown_bold=0.0,
        )
    assert esc is True
    assert why == "held_sell_flip_XAG-USD"


def test_not_held_sell_flip_no_escalation(tmp_path):
    state = _state(tmp_path)
    signals = _votes({"rsi": "HOLD"})
    acc = _acc_cache(["rsi"])
    with patch.object(er, "_ticker_held", return_value=False):
        esc, why = er.should_escalate_to_claude(
            ["XAG-USD flipped BUY->HOLD (sustained)"],
            tier=1, signals={"XAG-USD": signals["BTC-USD"]},
            accuracy_cache=acc, state_path=state,
            current_drawdown_patient=0.0, current_drawdown_bold=0.0,
        )
    assert esc is False


# ---------- post-trade ----------

def test_post_trade_reason_escalates(tmp_path):
    state = _state(tmp_path)
    esc, why = er.should_escalate_to_claude(
        ["post-trade Patient BTC-USD"],
        tier=1, signals={}, state_path=state,
        current_drawdown_patient=0.0, current_drawdown_bold=0.0,
    )
    assert esc is True
    assert why.startswith("post_trade")


# ---------- snapshot persistence ----------

def test_record_then_no_escalate_on_same_drawdown(tmp_path):
    state = str(tmp_path / "snap.json")
    er.record_decision_snapshot(4.0, 3.0, state_path=state)
    with patch.object(er, "_ticker_held", return_value=False):
        esc, why = er.should_escalate_to_claude(
            ["BTC-USD moved 0.5% up"], tier=1, signals={},
            state_path=state,
            current_drawdown_patient=4.0, current_drawdown_bold=3.0,
        )
    assert esc is False
    assert why == "autonomous"


# ---------- _parse_ticker (2026-05-15: full-string scan) ----------

class TestParseTicker:
    def test_first_token_hyphenated(self):
        assert er._parse_ticker("BTC-USD moved 2%") == "BTC-USD"

    def test_post_trade_prefix(self):
        # "post-trade Patient BTC-USD reason" -> head=post-trade (blocklisted shape)
        assert er._parse_ticker("post-trade Patient BTC-USD reason") == "BTC-USD"

    def test_sentiment_xau(self):
        assert er._parse_ticker("sentiment XAU-USD positive") == "XAU-USD"

    def test_plain_symbol(self):
        assert er._parse_ticker("MSTR consensus flipped") == "MSTR"

    def test_blocklist_skipped(self):
        # BUY/SELL/HOLD shouldn't be returned as ticker.
        assert er._parse_ticker("BUY signal on ETH-USD") == "ETH-USD"

    def test_prefer_hyphenated_over_plain(self):
        # MSTR appears first but BTC-USD is the actual ticker
        assert er._parse_ticker("MSTR vs BTC-USD comparison") in ("MSTR", "BTC-USD")

    def test_empty_string(self):
        assert er._parse_ticker("") == ""

    def test_no_ticker(self):
        assert er._parse_ticker("first of day") == ""

    def test_non_string(self):
        assert er._parse_ticker(None) == ""  # type: ignore[arg-type]
