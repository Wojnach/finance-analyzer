"""Tests for scripts/oil_loop_scorecard.py.

Verifies the score/pair logic without requiring real data files.
"""
from __future__ import annotations

import datetime
import os
import sys

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import oil_loop_scorecard as ols


def _ts_iso(offset_days: float = 0) -> str:
    now = datetime.datetime.now(datetime.UTC)
    delta = datetime.timedelta(days=offset_days)
    return (now + delta).isoformat()


def test_score_empty_returns_zero():
    assert ols.score([]) == {"n_trades": 0}


def test_score_single_winning_trade():
    trades = [{"pnl_sek": 100}]
    s = ols.score(trades)
    assert s["n_trades"] == 1
    assert s["wins"] == 1
    assert s["losses"] == 0
    assert s["win_rate_pct"] == 100.0
    assert s["total_pnl_sek"] == 100.0


def test_score_winning_and_losing_trades():
    trades = [{"pnl_sek": 200}, {"pnl_sek": -50}, {"pnl_sek": 100}]
    s = ols.score(trades)
    assert s["n_trades"] == 3
    assert s["wins"] == 2
    assert s["losses"] == 1
    assert s["win_rate_pct"] == round(2 / 3 * 100, 2)


def test_score_max_drawdown():
    # Sequence: 100 → 200 → 50 (peak 200, drawdown 150)
    trades = [{"pnl_sek": 100}, {"pnl_sek": 100}, {"pnl_sek": -150}]
    s = ols.score(trades)
    assert s["max_drawdown_sek"] == 150.0


def test_pair_trades_basic():
    decisions = [
        {"ts": _ts_iso(-2), "ticker": "OIL-USD", "action": "BUY",
         "underlying_price": 75.0, "confidence": 0.7},
    ]
    trades = [
        {"ts": _ts_iso(-1), "ticker": "OIL-USD", "action": "SELL",
         "underlying_price": 77.0, "pnl_sek": 200, "exit_reason": "tp"},
    ]
    paired = ols.pair_trades(decisions, trades)
    assert "OIL-USD" in paired
    assert len(paired["OIL-USD"]) == 1
    rt = paired["OIL-USD"][0]
    assert rt["entry_price"] == 75.0
    assert rt["exit_price"] == 77.0
    assert rt["pnl_sek"] == 200
    assert rt["exit_reason"] == "tp"


def test_pair_trades_unmatched_sell_skipped():
    """A SELL with no preceding BUY shouldn't blow up — just skip it."""
    decisions = []
    trades = [
        {"ts": _ts_iso(-1), "ticker": "OIL-USD", "action": "SELL",
         "underlying_price": 77.0},
    ]
    paired = ols.pair_trades(decisions, trades)
    assert paired == {}


def test_observation_window_empty_returns_zero_days():
    w = ols.compute_observation_window([])
    assert w["days_observed"] == 0
    assert w["days_remaining"] == ols.LIVE_MIN_DAYS


def test_observation_window_single_event_recent():
    events = [{"ts": _ts_iso(-5)}]  # 5 days ago
    w = ols.compute_observation_window(events)
    assert w["days_observed"] >= 4.9
    assert w["days_observed"] <= 5.1
    assert w["days_remaining"] == ols.LIVE_MIN_DAYS - round(w["days_observed"], 2)


def test_observation_window_clears_after_min_days():
    """When days_observed exceeds LIVE_MIN_DAYS, days_remaining floors at 0."""
    events = [{"ts": _ts_iso(-50)}]  # 50 days ago, > LIVE_MIN_DAYS=30
    w = ols.compute_observation_window(events)
    assert w["days_observed"] >= ols.LIVE_MIN_DAYS
    assert w["days_remaining"] == 0


def test_observation_window_skips_unparseable_ts():
    events = [{"ts": "garbage"}, {"ts": _ts_iso(-3)}]
    w = ols.compute_observation_window(events)
    # Should still find the valid one
    assert w["days_observed"] >= 2.9
