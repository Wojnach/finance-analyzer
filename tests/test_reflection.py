"""Tests for the periodic trade reflection module."""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from portfolio.reflection import (
    should_reflect,
    compute_reflection,
    save_reflection,
    maybe_reflect,
    load_latest_reflection,
    _count_trades,
    _compute_strategy_metrics,
    _generate_insights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(enabled=True, interval=10, max_age=7):
    return {
        "reflection": {
            "enabled": enabled,
            "trade_interval": interval,
            "max_age_days": max_age,
        }
    }


def _portfolio(transactions=None, cash=500000, initial=500000, holdings=None):
    return {
        "cash_sek": cash,
        "initial_value_sek": initial,
        "transactions": transactions or [],
        "holdings": holdings or {},
    }


def _tx(ticker="BTC-USD", action="BUY", shares=1.0, price_sek=100000):
    return {
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price_sek": price_sek,
    }


def _reflection(trade_count=7, ts=None):
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    return {
        "ts": ts,
        "trade_count_total": trade_count,
        "patient": {"trades": 0},
        "bold": {"trades": 7},
        "insights": [],
    }


# ---------------------------------------------------------------------------
# _count_trades
# ---------------------------------------------------------------------------

class TestCountTrades:
    def test_empty_portfolio(self):
        assert _count_trades({}) == 0

    def test_with_transactions(self):
        pf = _portfolio(transactions=[_tx(), _tx(), _tx()])
        assert _count_trades(pf) == 3


# ---------------------------------------------------------------------------
# _compute_strategy_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_no_trades(self):
        m = _compute_strategy_metrics(_portfolio())
        assert m["trades"] == 0
        assert m["win_rate"] is None
        assert m["total_pnl_pct"] == 0.0

    def test_winning_trade(self):
        pf = _portfolio(
            transactions=[
                _tx("BTC-USD", "BUY", 1.0, 100000),
                _tx("BTC-USD", "SELL", 1.0, 120000),
            ],
            cash=520000,
            initial=500000,
        )
        m = _compute_strategy_metrics(pf)
        assert m["trades"] == 2
        assert m["win_rate"] == 1.0
        assert m["avg_pnl_pct"] == 20.0

    def test_losing_trade(self):
        pf = _portfolio(
            transactions=[
                _tx("ETH-USD", "BUY", 10.0, 20000),
                _tx("ETH-USD", "SELL", 10.0, 18000),
            ],
            cash=480000,
            initial=500000,
        )
        m = _compute_strategy_metrics(pf)
        assert m["win_rate"] == 0.0
        assert m["avg_pnl_pct"] == -10.0

    def test_holdings_listed(self):
        pf = _portfolio(
            holdings={"BTC-USD": {"shares": 0.5}, "ETH-USD": {"shares": 0}},
        )
        m = _compute_strategy_metrics(pf)
        assert m["holdings"] == ["BTC-USD"]


# ---------------------------------------------------------------------------
# _generate_insights
# ---------------------------------------------------------------------------

class TestGenerateInsights:
    def test_no_trades_insight(self):
        patient = {"trades": 0}
        bold = {"trades": 0}
        insights = _generate_insights(patient, bold)
        assert any("no trades" in i for i in insights)

    def test_all_losses_insight(self):
        bold = {"trades": 7, "win_rate": 0.0, "total_pnl_pct": -7.0}
        insights = _generate_insights({"trades": 0}, bold)
        assert any("losses" in i for i in insights)
        assert any("down" in i for i in insights)

    def test_strong_win_rate(self):
        patient = {"trades": 10, "win_rate": 0.8, "total_pnl_pct": 8.0}
        insights = _generate_insights(patient, {"trades": 0})
        assert any("strong" in i for i in insights)


# ---------------------------------------------------------------------------
# should_reflect
# ---------------------------------------------------------------------------

class TestShouldReflect:
    def test_disabled(self):
        assert should_reflect(_cfg(enabled=False)) is False

    @patch("portfolio.reflection.load_json", return_value={})
    @patch("portfolio.reflection.load_jsonl", return_value=[])
    def test_first_reflection_below_interval(self, mock_jsonl, mock_json):
        # 0 trades, interval=10 → not due
        assert should_reflect(_cfg(interval=10)) is False

    @patch("portfolio.reflection.load_json")
    @patch("portfolio.reflection.load_jsonl", return_value=[])
    def test_first_reflection_at_interval(self, mock_jsonl, mock_json):
        # 10 trades total, no prior reflection → due
        pf = _portfolio(transactions=[_tx()] * 5)
        mock_json.side_effect = [pf, pf]
        assert should_reflect(_cfg(interval=10)) is True

    @patch("portfolio.reflection.load_json")
    @patch("portfolio.reflection.load_jsonl")
    def test_trade_interval_triggers(self, mock_jsonl, mock_json):
        pf = _portfolio(transactions=[_tx()] * 10)
        mock_json.side_effect = [pf, pf]
        mock_jsonl.return_value = [_reflection(trade_count=5)]
        # 20 trades now, last reflection at 5 → diff=15 >= 10
        assert should_reflect(_cfg(interval=10)) is True

    @patch("portfolio.reflection.load_json", return_value=_portfolio())
    @patch("portfolio.reflection.load_jsonl")
    def test_stale_reflection_triggers(self, mock_jsonl, mock_json):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        mock_jsonl.return_value = [_reflection(trade_count=0, ts=old_ts)]
        assert should_reflect(_cfg(max_age=7)) is True


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        ref = _reflection(trade_count=7)
        ref_file = tmp_path / "reflections.jsonl"

        with patch("portfolio.reflection.REFLECTIONS_FILE", ref_file):
            save_reflection(ref)
            loaded = load_latest_reflection()
            # load_latest_reflection reads from the real file, so mock it
        # Direct file check
        lines = ref_file.read_text().strip().split("\n")
        assert len(lines) == 1
        saved = json.loads(lines[0])
        assert saved["trade_count_total"] == 7

    def test_load_latest_returns_none_when_empty(self):
        with patch("portfolio.reflection.load_jsonl", return_value=[]):
            assert load_latest_reflection() is None

    def test_load_latest_returns_none_when_stale(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        with patch("portfolio.reflection.load_jsonl",
                    return_value=[_reflection(ts=old_ts)]):
            assert load_latest_reflection(max_age_days=7) is None

    def test_load_latest_returns_fresh(self):
        now_ts = datetime.now(timezone.utc).isoformat()
        with patch("portfolio.reflection.load_jsonl",
                    return_value=[_reflection(ts=now_ts)]):
            ref = load_latest_reflection(max_age_days=7)
            assert ref is not None
            assert ref["trade_count_total"] == 7


# ---------------------------------------------------------------------------
# maybe_reflect
# ---------------------------------------------------------------------------

class TestMaybeReflect:
    @patch("portfolio.reflection.should_reflect", return_value=False)
    def test_not_due(self, mock_sr):
        assert maybe_reflect(_cfg()) is False

    @patch("portfolio.reflection.save_reflection")
    @patch("portfolio.reflection.compute_reflection", return_value=_reflection())
    @patch("portfolio.reflection.should_reflect", return_value=True)
    def test_triggers_when_due(self, mock_sr, mock_cr, mock_save):
        assert maybe_reflect(_cfg()) is True
        mock_save.assert_called_once()

    @patch("portfolio.reflection.should_reflect", side_effect=Exception("boom"))
    def test_error_returns_false(self, mock_sr):
        assert maybe_reflect(_cfg()) is False
