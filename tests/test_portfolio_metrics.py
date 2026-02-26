"""Tests for enhanced portfolio metrics (per-trade metrics in equity_curve.py)."""

import pytest
from datetime import datetime, timezone, timedelta

from portfolio.equity_curve import _pair_round_trips, compute_trade_metrics


# --- Helpers ---

def _make_tx(ticker, action, shares, price_sek, total_sek, fee_sek=0,
             ts_offset_hours=0, base_ts=None):
    """Create a transaction dict."""
    if base_ts is None:
        base_ts = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
    ts = base_ts + timedelta(hours=ts_offset_hours)
    return {
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price_usd": price_sek / 10,  # Assume fx_rate=10
        "price_sek": price_sek,
        "total_sek": total_sek,
        "fee_sek": fee_sek,
        "fx_rate": 10.0,
        "reason": "test",
    }


def _sample_transactions():
    """Create a sample set of BUY/SELL transactions for testing."""
    return [
        # BUY BTC, SELL BTC (win)
        _make_tx("BTC-USD", "BUY", 0.1, 600000, 60000, fee_sek=30, ts_offset_hours=0),
        _make_tx("BTC-USD", "SELL", 0.1, 660000, 66000, fee_sek=33, ts_offset_hours=48),
        # BUY ETH, SELL ETH (loss)
        _make_tx("ETH-USD", "BUY", 5, 20000, 100000, fee_sek=50, ts_offset_hours=24),
        _make_tx("ETH-USD", "SELL", 5, 18000, 90000, fee_sek=45, ts_offset_hours=72),
        # BUY NVDA, SELL NVDA (win)
        _make_tx("NVDA", "BUY", 100, 1800, 180000, fee_sek=180, ts_offset_hours=48),
        _make_tx("NVDA", "SELL", 100, 1900, 190000, fee_sek=190, ts_offset_hours=96),
    ]


# --- _pair_round_trips ---

class TestPairRoundTrips:
    def test_simple_pair(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, ts_offset_hours=24),
        ]
        trips = _pair_round_trips(txns)
        assert len(trips) == 1
        assert trips[0]["ticker"] == "BTC-USD"
        assert trips[0]["pnl_pct"] == pytest.approx(10.0, abs=0.1)
        assert trips[0]["hold_hours"] == pytest.approx(24.0, abs=0.5)

    def test_partial_sell(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 2.0, 600000, 1200000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, ts_offset_hours=24),
        ]
        trips = _pair_round_trips(txns)
        assert len(trips) == 1
        assert trips[0]["shares"] == pytest.approx(1.0)

    def test_multiple_buys_fifo(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 500000, 500000, ts_offset_hours=0),
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, ts_offset_hours=12),
            _make_tx("BTC-USD", "SELL", 1.5, 660000, 990000, ts_offset_hours=24),
        ]
        trips = _pair_round_trips(txns)
        # First trip: 1.0 shares from first BUY @ 500K
        # Second trip: 0.5 shares from second BUY @ 600K
        assert len(trips) == 2
        assert trips[0]["shares"] == pytest.approx(1.0)
        assert trips[1]["shares"] == pytest.approx(0.5)
        assert trips[0]["buy_price_sek"] == pytest.approx(500000)
        assert trips[1]["buy_price_sek"] == pytest.approx(600000)

    def test_empty_transactions(self):
        trips = _pair_round_trips([])
        assert trips == []

    def test_only_buys_no_trips(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000),
        ]
        trips = _pair_round_trips(txns)
        assert trips == []

    def test_multiple_tickers(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, ts_offset_hours=0),
            _make_tx("ETH-USD", "BUY", 10, 20000, 200000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, ts_offset_hours=24),
            _make_tx("ETH-USD", "SELL", 10, 18000, 180000, ts_offset_hours=24),
        ]
        trips = _pair_round_trips(txns)
        assert len(trips) == 2
        btc_trip = [t for t in trips if t["ticker"] == "BTC-USD"][0]
        eth_trip = [t for t in trips if t["ticker"] == "ETH-USD"][0]
        assert btc_trip["pnl_pct"] > 0  # win
        assert eth_trip["pnl_pct"] < 0  # loss

    def test_fee_tracking(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, fee_sek=300, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, fee_sek=330, ts_offset_hours=24),
        ]
        trips = _pair_round_trips(txns)
        assert trips[0]["fee_sek"] == pytest.approx(630, abs=1)


# --- compute_trade_metrics ---

class TestComputeTradeMetrics:
    def test_basic_metrics(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        assert m["round_trips"] == 3
        assert m["max_consecutive_wins"] >= 1
        assert m["max_consecutive_losses"] >= 1

    def test_profit_factor(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        # 2 wins, 1 loss — profit factor should be > 0
        assert m["profit_factor"] is not None
        assert m["profit_factor"] > 0

    def test_avg_hold_hours(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        assert m["avg_hold_hours"] > 0

    def test_trade_frequency(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        assert m["trade_frequency_per_week"] > 0

    def test_win_loss_ratio(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        assert m["win_loss_ratio"] is not None
        assert m["win_loss_ratio"] > 0

    def test_expectancy(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        # With 2 wins and 1 loss, expectancy should be calculable
        assert isinstance(m["expectancy_pct"], float)

    def test_streaks(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        assert m["max_consecutive_wins"] >= 1
        assert m["max_consecutive_losses"] >= 1

    def test_total_pnl(self):
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        assert isinstance(m["total_pnl_sek"], float)

    def test_empty_transactions(self):
        m = compute_trade_metrics([])
        assert m["round_trips"] == 0
        assert m["profit_factor"] is None
        assert m["avg_hold_hours"] == 0

    def test_only_buys(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000),
        ]
        m = compute_trade_metrics(txns)
        assert m["round_trips"] == 0

    def test_all_wins(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 500000, 500000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 600000, 600000, ts_offset_hours=24),
            _make_tx("ETH-USD", "BUY", 10, 15000, 150000, ts_offset_hours=48),
            _make_tx("ETH-USD", "SELL", 10, 18000, 180000, ts_offset_hours=72),
        ]
        m = compute_trade_metrics(txns)
        assert m["profit_factor"] is None  # No losses, so profit_factor undefined
        assert m["max_consecutive_wins"] == 2
        assert m["max_consecutive_losses"] == 0

    def test_all_losses(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 500000, 500000, ts_offset_hours=24),
            _make_tx("ETH-USD", "BUY", 10, 20000, 200000, ts_offset_hours=48),
            _make_tx("ETH-USD", "SELL", 10, 18000, 180000, ts_offset_hours=72),
        ]
        m = compute_trade_metrics(txns)
        assert m["profit_factor"] == pytest.approx(0.0)
        assert m["max_consecutive_losses"] == 2
        assert m["max_consecutive_wins"] == 0

    def test_calmar_ratio_calculated(self):
        # Need enough trades with a time span
        txns = _sample_transactions()
        m = compute_trade_metrics(txns)
        # Calmar might or might not be computable depending on data
        # Just verify it doesn't crash
        assert m["calmar_ratio"] is None or isinstance(m["calmar_ratio"], float)

    def test_single_round_trip(self):
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, ts_offset_hours=24),
        ]
        m = compute_trade_metrics(txns)
        assert m["round_trips"] == 1
        assert m["max_consecutive_wins"] == 1
