"""Tests for FIFO round-trip matching in equity_curve._pair_round_trips.

Covers basic matching, fee proportionality (including BUG-37 regression),
edge cases (empty/zero-share/no-matching-buy), hold time, and multi-ticker
interleaving.
"""

import pytest

from portfolio.equity_curve import _pair_round_trips

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy(ticker, shares, total_sek, fee_sek, ts="2026-01-01T00:00:00+00:00"):
    return {
        "action": "BUY",
        "ticker": ticker,
        "shares": shares,
        "total_sek": total_sek,
        "fee_sek": fee_sek,
        "timestamp": ts,
    }


def _sell(ticker, shares, total_sek, fee_sek, ts="2026-01-02T00:00:00+00:00"):
    return {
        "action": "SELL",
        "ticker": ticker,
        "shares": shares,
        "total_sek": total_sek,
        "fee_sek": fee_sek,
        "timestamp": ts,
    }


# ===================================================================
# 1. BUG-37 regression: fee double-counting on multi-partial-sell
# ===================================================================

class TestBug37FeeDoubleCount:
    """BUG-37 regression: fee double-counting on multi-partial-sell.

    One BUY of 100 shares with 10 SEK fee, two SELLs of 50 shares each.
    The buy fee must be split proportionally based on the *original* buy
    size, yielding 5 + 5 = 10 SEK total buy-side fee across both round trips.

    Before the fix, remaining_shares was used as denominator instead of
    original_shares, so the second sell's buy_fee_share was computed against
    50 remaining instead of 100 original: 10 * 50/50 = 10 instead of 5.
    Total was 5 + 10 = 15 -- a 50% overcount.
    """

    @pytest.fixture()
    def trips(self):
        txs = [
            _buy("BTC-USD", 100, 10000, 10, "2026-01-01T00:00:00+00:00"),
            _sell("BTC-USD", 50, 6000, 6, "2026-01-02T00:00:00+00:00"),
            _sell("BTC-USD", 50, 6000, 6, "2026-01-03T00:00:00+00:00"),
        ]
        return _pair_round_trips(txs)

    def test_two_round_trips_created(self, trips):
        assert len(trips) == 2

    def test_first_sell_buy_fee_is_5(self, trips):
        """First 50 of 100 shares: buy fee should be 10 * 50/100 = 5."""
        # sell fee share for 50/50 = 6
        # total fee = buy_fee(5) + sell_fee(6) = 11
        assert trips[0]["fee_sek"] == pytest.approx(11.0, abs=0.01)

    def test_second_sell_buy_fee_is_5_not_10(self, trips):
        """Second 50 of 100 shares: buy fee should also be 10 * 50/100 = 5.

        Regression for BUG-37: must use original_shares (100) not
        remaining_shares (50) as denominator.
        """
        # buy_fee=5 + sell_fee=6 = 11
        assert trips[1]["fee_sek"] == pytest.approx(11.0, abs=0.01)

    def test_total_buy_fee_equals_original(self, trips):
        """Sum of buy-side fees across all round trips must equal original buy fee (10)."""
        # buy fees: 5 + 5 = 10, sell fees: 6 + 6 = 12, total = 22
        total_fee = sum(t["fee_sek"] for t in trips)
        assert total_fee == pytest.approx(22.0, abs=0.01)


# ===================================================================
# 2. Single BUY + single SELL (full match)
# ===================================================================

class TestSingleFullMatch:
    def test_one_round_trip(self):
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        trips = _pair_round_trips(txs)
        assert len(trips) == 1

    def test_prices_correct(self):
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        assert t["buy_price_sek"] == pytest.approx(2000.0)
        assert t["sell_price_sek"] == pytest.approx(2500.0)

    def test_shares_match(self):
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        assert t["shares"] == pytest.approx(10.0)

    def test_pnl_pct(self):
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # (2500 - 2000) / 2000 * 100 = 25%
        assert t["pnl_pct"] == pytest.approx(25.0)

    def test_pnl_sek(self):
        """pnl_sek is net of fees (P0-6, 2026-05-02).

        Pre-2026-05-02 behaviour returned the gross price-difference (5000).
        Now returns gross minus buy_fee_share + sell_fee_share = 5000 - 45 = 4955.
        See TestPnlSekNetOfFees below for the full contract.
        """
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # Gross: (2500 - 2000) * 10 = 5000. Net: 5000 - 20 - 25 = 4955.
        assert t["pnl_sek"] == pytest.approx(4955.0)

    def test_fee_total(self):
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # Full match: buy_fee=20, sell_fee=25
        assert t["fee_sek"] == pytest.approx(45.0)

    def test_ticker(self):
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        assert t["ticker"] == "ETH-USD"


# ===================================================================
# 3. One BUY + two partial SELLs (fee proportionality)
# ===================================================================

class TestPartialSells:
    """One BUY of 100 shares, two SELLs of 30 and 70 shares."""

    @pytest.fixture()
    def trips(self):
        txs = [
            _buy("XAG-USD", 100, 10000, 100, "2026-01-01T00:00:00+00:00"),
            _sell("XAG-USD", 30, 3600, 3.6, "2026-01-02T00:00:00+00:00"),
            _sell("XAG-USD", 70, 9100, 9.1, "2026-01-03T00:00:00+00:00"),
        ]
        return _pair_round_trips(txs)

    def test_two_round_trips(self, trips):
        assert len(trips) == 2

    def test_shares_30_and_70(self, trips):
        assert trips[0]["shares"] == pytest.approx(30.0)
        assert trips[1]["shares"] == pytest.approx(70.0)

    def test_first_sell_buy_fee_proportional(self, trips):
        """30/100 of buy fee = 30 SEK. sell fee share = 3.6 (full). Total = 33.6."""
        assert trips[0]["fee_sek"] == pytest.approx(33.6, abs=0.01)

    def test_second_sell_buy_fee_proportional(self, trips):
        """70/100 of buy fee = 70 SEK. sell fee = 9.1. Total = 79.1.

        Regression for BUG-37: must use original_shares (100) not
        remaining_shares (70) as denominator.
        """
        assert trips[1]["fee_sek"] == pytest.approx(79.1, abs=0.01)


# ===================================================================
# 4. Two BUYs + one large SELL (FIFO ordering)
# ===================================================================

class TestFIFOOrdering:
    """Two BUYs at different prices, one SELL spanning both. FIFO = first buy first."""

    @pytest.fixture()
    def trips(self):
        txs = [
            _buy("NVDA", 50, 5000, 5, "2026-01-01T00:00:00+00:00"),   # 100/sh
            _buy("NVDA", 50, 7500, 7.5, "2026-01-02T00:00:00+00:00"), # 150/sh
            _sell("NVDA", 80, 12000, 12, "2026-01-03T00:00:00+00:00"),  # 150/sh
        ]
        return _pair_round_trips(txs)

    def test_two_round_trips(self, trips):
        """80 shares sold: 50 from first BUY + 30 from second BUY."""
        assert len(trips) == 2

    def test_first_trip_uses_first_buy(self, trips):
        """First 50 shares matched against BUY at 100/sh."""
        assert trips[0]["shares"] == pytest.approx(50.0)
        assert trips[0]["buy_price_sek"] == pytest.approx(100.0)
        assert trips[0]["buy_ts"] == "2026-01-01T00:00:00+00:00"

    def test_second_trip_uses_second_buy(self, trips):
        """Remaining 30 shares matched against BUY at 150/sh."""
        assert trips[1]["shares"] == pytest.approx(30.0)
        assert trips[1]["buy_price_sek"] == pytest.approx(150.0)
        assert trips[1]["buy_ts"] == "2026-01-02T00:00:00+00:00"

    def test_sell_price_same_for_both(self, trips):
        """Both round trips use the same sell price (150/sh from the single SELL)."""
        assert trips[0]["sell_price_sek"] == pytest.approx(150.0)
        assert trips[1]["sell_price_sek"] == pytest.approx(150.0)

    def test_pnl_first_trip(self, trips):
        """First trip: (150-100)/100 = 50%."""
        assert trips[0]["pnl_pct"] == pytest.approx(50.0)

    def test_pnl_second_trip(self, trips):
        """Second trip: (150-150)/150 = 0%."""
        assert trips[1]["pnl_pct"] == pytest.approx(0.0)

    def test_leftover_buy_shares_not_matched(self, trips):
        """Second BUY had 50 shares; only 30 matched. 20 remain unmatched (no trip)."""
        assert len(trips) == 2
        total_sold = sum(t["shares"] for t in trips)
        assert total_sold == pytest.approx(80.0)


# ===================================================================
# 5. SELL with no matching BUY (should be skipped)
# ===================================================================

def test_sell_without_buy_skipped():
    txs = [
        _sell("BTC-USD", 10, 12000, 12, "2026-01-02T00:00:00+00:00"),
    ]
    trips = _pair_round_trips(txs)
    assert trips == []


def test_sell_for_wrong_ticker_skipped():
    txs = [
        _buy("ETH-USD", 10, 20000, 20, "2026-01-01T00:00:00+00:00"),
        _sell("BTC-USD", 10, 12000, 12, "2026-01-02T00:00:00+00:00"),
    ]
    trips = _pair_round_trips(txs)
    assert trips == []


# ===================================================================
# 6. Zero-share BUY (should be skipped)
# ===================================================================

def test_zero_share_buy_skipped():
    txs = [
        _buy("BTC-USD", 0, 0, 0, "2026-01-01T00:00:00+00:00"),
        _sell("BTC-USD", 10, 12000, 12, "2026-01-02T00:00:00+00:00"),
    ]
    trips = _pair_round_trips(txs)
    assert trips == []


def test_zero_share_sell_skipped():
    txs = [
        _buy("BTC-USD", 10, 10000, 10, "2026-01-01T00:00:00+00:00"),
        _sell("BTC-USD", 0, 0, 0, "2026-01-02T00:00:00+00:00"),
    ]
    trips = _pair_round_trips(txs)
    assert trips == []


# ===================================================================
# 7. Empty transaction list
# ===================================================================

def test_empty_transactions():
    assert _pair_round_trips([]) == []


def test_only_buys_no_sells():
    txs = [
        _buy("BTC-USD", 10, 10000, 10),
        _buy("ETH-USD", 5, 10000, 10),
    ]
    assert _pair_round_trips(txs) == []


# ===================================================================
# 8. Hold time computation
# ===================================================================

class TestHoldTime:
    def test_24h_hold(self):
        txs = [
            _buy("BTC-USD", 1, 1000, 1, "2026-01-10T08:00:00+00:00"),
            _sell("BTC-USD", 1, 1200, 1, "2026-01-11T08:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        assert t["hold_hours"] == pytest.approx(24.0)

    def test_short_hold_90min(self):
        txs = [
            _buy("XAG-USD", 50, 5000, 5, "2026-02-15T10:00:00+00:00"),
            _sell("XAG-USD", 50, 5500, 5, "2026-02-15T11:30:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        assert t["hold_hours"] == pytest.approx(1.5)

    def test_multi_day_hold(self):
        txs = [
            _buy("PLTR", 20, 2000, 2, "2026-01-01T00:00:00+00:00"),
            _sell("PLTR", 20, 3000, 3, "2026-01-08T00:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        assert t["hold_hours"] == pytest.approx(168.0)  # 7 days


# ===================================================================
# 9. Multiple tickers interleaved
# ===================================================================

class TestMultipleTickers:
    @pytest.fixture()
    def trips(self):
        txs = [
            _buy("BTC-USD", 2, 2000, 2, "2026-01-01T00:00:00+00:00"),
            _buy("ETH-USD", 10, 5000, 5, "2026-01-01T06:00:00+00:00"),
            _sell("ETH-USD", 10, 6000, 6, "2026-01-02T06:00:00+00:00"),
            _sell("BTC-USD", 2, 3000, 3, "2026-01-03T00:00:00+00:00"),
        ]
        return _pair_round_trips(txs)

    def test_two_round_trips(self, trips):
        assert len(trips) == 2

    def test_tickers_correct(self, trips):
        tickers = {t["ticker"] for t in trips}
        assert tickers == {"BTC-USD", "ETH-USD"}

    def test_eth_matched_correctly(self, trips):
        eth = [t for t in trips if t["ticker"] == "ETH-USD"][0]
        assert eth["shares"] == pytest.approx(10.0)
        assert eth["buy_price_sek"] == pytest.approx(500.0)
        assert eth["sell_price_sek"] == pytest.approx(600.0)

    def test_btc_matched_correctly(self, trips):
        btc = [t for t in trips if t["ticker"] == "BTC-USD"][0]
        assert btc["shares"] == pytest.approx(2.0)
        assert btc["buy_price_sek"] == pytest.approx(1000.0)
        assert btc["sell_price_sek"] == pytest.approx(1500.0)

    def test_hold_times_independent(self, trips):
        eth = [t for t in trips if t["ticker"] == "ETH-USD"][0]
        btc = [t for t in trips if t["ticker"] == "BTC-USD"][0]
        assert eth["hold_hours"] == pytest.approx(24.0)
        assert btc["hold_hours"] == pytest.approx(48.0)


# ===================================================================
# Additional edge cases
# ===================================================================

def test_none_fee_treated_as_zero():
    """fee_sek=None should not crash; treated as 0."""
    txs = [
        {"action": "BUY", "ticker": "BTC-USD", "shares": 10, "total_sek": 10000,
         "fee_sek": None, "timestamp": "2026-01-01T00:00:00+00:00"},
        {"action": "SELL", "ticker": "BTC-USD", "shares": 10, "total_sek": 12000,
         "fee_sek": None, "timestamp": "2026-01-02T00:00:00+00:00"},
    ]
    trips = _pair_round_trips(txs)
    assert len(trips) == 1
    assert trips[0]["fee_sek"] == pytest.approx(0.0)


def test_sell_larger_than_all_buys_partial_match():
    """SELL 100 but only 40 bought -- should match 40, discard remaining 60."""
    txs = [
        _buy("MU", 40, 4000, 4, "2026-01-01T00:00:00+00:00"),
        _sell("MU", 100, 12000, 12, "2026-01-02T00:00:00+00:00"),
    ]
    trips = _pair_round_trips(txs)
    assert len(trips) == 1
    assert trips[0]["shares"] == pytest.approx(40.0)


# ===================================================================
# 10. P0-6 regression: pnl_sek MUST be net of fees (2026-05-02)
# ===================================================================
#
# Prior bug: `pnl_sek` reported the gross price-difference times shares,
# while `fee_sek` was reported separately. Downstream consumers
# (compute_trade_metrics → profit_factor, total_pnl_sek, calmar) assumed
# pnl_sek was the realised SEK return after costs. profit_factor was
# overstated by ~8% on a typical mix.
#
# Fix: subtract `buy_fee_share + sell_fee_share` from `pnl_sek` at
# round-trip construction time. `pnl_pct` is left as the gross price-%
# (price-move only), and `fee_sek` continues to report total fees.

class TestPnlSekNetOfFees:
    """P0-6 regression: pnl_sek must be net of buy and sell fees."""

    def test_simple_full_match_pnl_net_of_fees(self):
        """Single full-match BUY/SELL: pnl_sek = gross - buy_fee - sell_fee."""
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # Gross: (2500 - 2000) * 10 = 5000. Net: 5000 - 20 - 25 = 4955.
        assert t["pnl_sek"] == pytest.approx(4955.0)
        # fee_sek field unchanged: 20 + 25 = 45.
        assert t["fee_sek"] == pytest.approx(45.0)

    def test_partial_sell_pnl_net_of_proportional_fees(self):
        """BUY 100, SELL 30: pnl_sek subtracts 0.3 * buy_fee + full sell_fee."""
        txs = [
            _buy("XAG-USD", 100, 10000, 100, "2026-01-01T00:00:00+00:00"),
            _sell("XAG-USD", 30, 3600, 3.6, "2026-01-02T00:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # Buy price 100/sh, sell price 120/sh, matched 30.
        # Gross: (120 - 100) * 30 = 600.
        # Buy fee share: 100 * 30/100 = 30. Sell fee share: 3.6 * 30/30 = 3.6.
        # Net: 600 - 30 - 3.6 = 566.4.
        assert t["pnl_sek"] == pytest.approx(566.4, abs=0.01)

    def test_multi_partial_sell_total_pnl_net(self):
        """Sum of pnl_sek across both partial trips equals gross_total - all fees."""
        txs = [
            _buy("BTC-USD", 100, 10000, 10, "2026-01-01T00:00:00+00:00"),
            _sell("BTC-USD", 50, 6000, 6, "2026-01-02T00:00:00+00:00"),
            _sell("BTC-USD", 50, 6000, 6, "2026-01-03T00:00:00+00:00"),
        ]
        trips = _pair_round_trips(txs)
        assert len(trips) == 2
        # Gross per trip: (120 - 100) * 50 = 1000. Total gross: 2000.
        # Buy fee per trip: 10 * 50/100 = 5. Sell fee per trip: 6 * 50/50 = 6.
        # Net per trip: 1000 - 5 - 6 = 989. Total net: 1978.
        assert trips[0]["pnl_sek"] == pytest.approx(989.0, abs=0.01)
        assert trips[1]["pnl_sek"] == pytest.approx(989.0, abs=0.01)
        total_net = sum(t["pnl_sek"] for t in trips)
        assert total_net == pytest.approx(1978.0, abs=0.01)

    def test_zero_fee_pnl_unchanged(self):
        """fee_sek=0 -> pnl_sek matches old gross behavior (regression baseline)."""
        txs = [
            _buy("BTC-USD", 1, 600000, 0, "2026-01-01T00:00:00+00:00"),
            _sell("BTC-USD", 1, 660000, 0, "2026-01-02T00:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # Gross == net when fees are zero: (660000 - 600000) * 1 = 60000.
        assert t["pnl_sek"] == pytest.approx(60000.0)

    def test_pnl_pct_unaffected_by_fee_fix(self):
        """pnl_pct contract: gross price-% only, NOT net-of-fees."""
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # Price went from 2000 to 2500: +25% regardless of fees.
        assert t["pnl_pct"] == pytest.approx(25.0)

    def test_fee_sek_field_unchanged_after_pnl_fix(self):
        """fee_sek field continues to report total round-trip fees."""
        txs = [
            _buy("ETH-USD", 10, 20000, 20, "2026-01-01T12:00:00+00:00"),
            _sell("ETH-USD", 10, 25000, 25, "2026-01-05T12:00:00+00:00"),
        ]
        t = _pair_round_trips(txs)[0]
        # fee_sek must remain the sum of buy_fee_share + sell_fee_share.
        assert t["fee_sek"] == pytest.approx(45.0)
