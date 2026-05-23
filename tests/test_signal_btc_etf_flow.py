"""Tests for BTC ETF flow signal module (scaffold)."""
import pytest


class TestBtcEtfFlowSignal:

    def test_non_btc_ticker_returns_hold(self):
        from portfolio.signals.btc_etf_flow import compute
        result = compute("XAG-USD", {})
        assert result["action"] == "HOLD"

    def test_btc_with_no_data_returns_hold(self, tmp_path):
        from portfolio.signals import btc_etf_flow
        from unittest.mock import patch
        with patch.object(btc_etf_flow, "FLOW_CACHE_FILE", tmp_path / "missing.json"):
            result = btc_etf_flow.compute("BTC-USD", {})
        assert result["action"] == "HOLD"
        assert result["indicators"].get("error") == "no_flow_data"

    def test_strong_inflow_returns_buy(self):
        from portfolio.signals.btc_etf_flow import _daily_flow_signal
        vote, ind = _daily_flow_signal({"net_flow_usd": 300_000_000})
        assert vote == "BUY"
        assert ind["strength"] == "strong_inflow"

    def test_strong_outflow_returns_sell(self):
        from portfolio.signals.btc_etf_flow import _daily_flow_signal
        vote, ind = _daily_flow_signal({"net_flow_usd": -250_000_000})
        assert vote == "SELL"
        assert ind["strength"] == "strong_outflow"

    def test_neutral_flow_returns_hold(self):
        from portfolio.signals.btc_etf_flow import _daily_flow_signal
        vote, ind = _daily_flow_signal({"net_flow_usd": 10_000_000})
        assert vote == "HOLD"

    def test_streak_buy(self):
        from portfolio.signals.btc_etf_flow import _streak_signal
        vote, _ = _streak_signal({"consecutive_inflow_days": 5})
        assert vote == "BUY"

    def test_streak_sell(self):
        from portfolio.signals.btc_etf_flow import _streak_signal
        vote, _ = _streak_signal({"consecutive_inflow_days": -4})
        assert vote == "SELL"

    def test_divergence_accumulation(self):
        from portfolio.signals.btc_etf_flow import _divergence_signal
        vote, ind = _divergence_signal(
            {"net_flow_usd": 100_000_000},
            {"price_change_1d": -2.5},
        )
        assert vote == "BUY"
        assert ind["divergence"] == "accumulation"

    def test_applicable_tickers(self):
        from portfolio.signals.btc_etf_flow import APPLICABLE_TICKERS
        assert "BTC-USD" in APPLICABLE_TICKERS
        assert "ETH-USD" not in APPLICABLE_TICKERS
