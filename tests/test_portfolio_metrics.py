"""Tests for enhanced portfolio metrics (per-trade metrics in equity_curve.py)."""

from datetime import UTC, datetime, timedelta

import pytest

from portfolio.equity_curve import _pair_round_trips, compute_trade_metrics

# --- Helpers ---

def _make_tx(ticker, action, shares, price_sek, total_sek, fee_sek=0,
             ts_offset_hours=0, base_ts=None):
    """Create a transaction dict."""
    if base_ts is None:
        base_ts = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)
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


# ===================================================================
# P0-6 cascade: profit_factor / total_pnl_sek / calmar use NET pnl_sek
# ===================================================================
#
# `pnl_sek` was changed to be net of fees (was gross). Downstream
# metrics in compute_trade_metrics() that consume pnl_sek must reflect
# the net values. These tests pin the post-fix contract.

class TestProfitFactorNetOfFees:
    """P0-6 cascade: downstream metrics use net pnl_sek."""

    def test_profit_factor_uses_net_pnl(self):
        """Controlled fixture: gross profit_factor != net profit_factor.

        2 winning trades (gross +60000 each, fees 1000 per side = -2000 net per win),
        1 losing trade (gross -40000, fees 1000 per side = -2000 worse).

        Pre-fix (gross): PF = (60000 + 60000) / 40000 = 3.00
        Post-fix (net):  PF = (58000 + 58000) / 42000 = 116000/42000 ~= 2.762
        """
        txns = [
            # Win 1
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, fee_sek=1000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, fee_sek=1000, ts_offset_hours=24),
            # Win 2
            _make_tx("ETH-USD", "BUY", 1.0, 200000, 200000, fee_sek=1000, ts_offset_hours=48),
            _make_tx("ETH-USD", "SELL", 1.0, 260000, 260000, fee_sek=1000, ts_offset_hours=72),
            # Loss 1
            _make_tx("NVDA", "BUY", 1.0, 400000, 400000, fee_sek=1000, ts_offset_hours=96),
            _make_tx("NVDA", "SELL", 1.0, 360000, 360000, fee_sek=1000, ts_offset_hours=120),
        ]
        m = compute_trade_metrics(txns)
        # Net PF = 116000 / 42000 ~= 2.762, not 3.0
        assert m["profit_factor"] == pytest.approx(2.762, abs=0.01)
        # Sanity: not the gross 3.0
        assert m["profit_factor"] < 3.0

    def test_total_pnl_sek_is_net(self):
        """compute_trade_metrics.total_pnl_sek == sum of NET round-trip pnl."""
        txns = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, fee_sek=500, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, fee_sek=550, ts_offset_hours=24),
        ]
        m = compute_trade_metrics(txns)
        # Gross: 60000. Net: 60000 - 500 - 550 = 58950.
        assert m["total_pnl_sek"] == pytest.approx(58950.0, abs=0.01)

    def test_expectancy_pct_uses_pnl_pct_not_pnl_sek(self):
        """expectancy_pct contract: computed from pnl_pct (gross %), unaffected by P0-6.

        This test pins the contract that expectancy is the price-based
        expected-return %, NOT the SEK-based net expectancy. If you want
        a SEK-net expectancy, that's a separate metric.
        """
        # Same trade with very different fee structures should have
        # identical expectancy_pct (pct doesn't change with fees).
        txns_low_fee = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, fee_sek=1, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, fee_sek=1, ts_offset_hours=24),
        ]
        txns_high_fee = [
            _make_tx("BTC-USD", "BUY", 1.0, 600000, 600000, fee_sek=10000, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 660000, 660000, fee_sek=10000, ts_offset_hours=24),
        ]
        m_low = compute_trade_metrics(txns_low_fee)
        m_high = compute_trade_metrics(txns_high_fee)
        # Both round trips: pnl_pct = (660000-600000)/600000 * 100 = 10.0%
        # Expectancy = win_rate(1.0) * avg_win(10) - loss_rate(0) * 0 = 10.0%
        assert m_low["expectancy_pct"] == pytest.approx(m_high["expectancy_pct"], abs=0.001)
        # And total_pnl_sek differs because that one IS net.
        assert m_low["total_pnl_sek"] != m_high["total_pnl_sek"]

    def test_calmar_uses_net_pnl(self):
        """Calmar's mini equity curve seeds from t['pnl_sek'] which is now net.

        Verify total_pnl_sek (the easy-to-compute proxy) reflects net
        across a mixed win+loss sequence. Calmar itself depends on a
        time-span computation; just verify it doesn't crash and is
        either None or float.
        """
        # Build trades where fees materially change the equity curve.
        txns = [
            # Win
            _make_tx("BTC-USD", "BUY", 1.0, 500000, 500000, fee_sek=500, ts_offset_hours=0),
            _make_tx("BTC-USD", "SELL", 1.0, 600000, 600000, fee_sek=500, ts_offset_hours=24),
            # Loss
            _make_tx("ETH-USD", "BUY", 1.0, 200000, 200000, fee_sek=500, ts_offset_hours=48),
            _make_tx("ETH-USD", "SELL", 1.0, 180000, 180000, fee_sek=500, ts_offset_hours=72),
        ]
        m = compute_trade_metrics(txns, initial_value=500000)
        assert m["calmar_ratio"] is None or isinstance(m["calmar_ratio"], float)
        # total_pnl_sek must be net: (100000 - 1000) + (-20000 - 1000) = 78000
        assert m["total_pnl_sek"] == pytest.approx(78000.0, abs=0.01)
