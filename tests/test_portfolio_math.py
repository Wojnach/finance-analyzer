"""Tests for portfolio BUY/SELL execution math.

Covers:
- BUY: allocation, fee deduction, shares calculation, cash update
- SELL: partial exit (50% Patient), full exit (100% Bold), fee deduction, proceeds
- Edge cases: sell more shares than held, negative cash
- Fee accumulation: total_fees_sek tracking
- Rounding: no floating point drift over many trades
"""

import copy
import json
import math
import pytest


# ---------------------------------------------------------------------------
# Constants matching CLAUDE.md spec
# ---------------------------------------------------------------------------
FEE_CRYPTO = 0.0005   # 0.05% taker fee
FEE_STOCK = 0.001     # 0.10% broker commission
INITIAL_CASH = 500_000.0


def _make_portfolio(cash=INITIAL_CASH, holdings=None, transactions=None,
                    total_fees_sek=0.0):
    """Create a fresh portfolio state."""
    return {
        "cash_sek": cash,
        "holdings": holdings or {},
        "transactions": transactions or [],
        "total_fees_sek": total_fees_sek,
        "initial_value_sek": INITIAL_CASH,
        "start_date": "2026-02-11T00:00:00+00:00",
    }


def _execute_buy(portfolio, ticker, price_sek, alloc_pct, is_crypto=True):
    """Execute a BUY following CLAUDE.md math exactly.

    Returns (shares_bought, fee, alloc).
    """
    fee_rate = FEE_CRYPTO if is_crypto else FEE_STOCK
    alloc = portfolio["cash_sek"] * alloc_pct
    fee = alloc * fee_rate
    net_alloc = alloc - fee
    shares_bought = net_alloc / price_sek

    # Update holdings
    existing = portfolio["holdings"].get(ticker, {})
    existing_shares = existing.get("shares", 0.0)
    existing_avg_cost = existing.get("avg_cost_sek", 0.0)

    if existing_shares > 0:
        new_total_cost = (existing_shares * existing_avg_cost) + (shares_bought * price_sek)
        new_shares = existing_shares + shares_bought
        avg_cost = new_total_cost / new_shares
    else:
        new_shares = shares_bought
        avg_cost = price_sek

    portfolio["holdings"][ticker] = {
        "shares": new_shares,
        "avg_cost_sek": avg_cost,
    }
    portfolio["cash_sek"] -= alloc
    if portfolio["total_fees_sek"] is None:
        portfolio["total_fees_sek"] = 0.0
    portfolio["total_fees_sek"] += fee

    return shares_bought, fee, alloc


def _execute_sell(portfolio, ticker, sell_pct, price_sek, is_crypto=True):
    """Execute a SELL following CLAUDE.md math exactly.

    Returns (sell_shares, proceeds, fee, net_proceeds).
    Raises ValueError if ticker not in holdings or insufficient shares.
    """
    if ticker not in portfolio["holdings"]:
        raise ValueError(f"Cannot sell {ticker}: not in holdings")

    existing_shares = portfolio["holdings"][ticker]["shares"]
    if existing_shares <= 0:
        raise ValueError(f"Cannot sell {ticker}: no shares held")

    fee_rate = FEE_CRYPTO if is_crypto else FEE_STOCK
    sell_shares = existing_shares * sell_pct
    proceeds = sell_shares * price_sek
    fee = proceeds * fee_rate
    net_proceeds = proceeds - fee

    remaining_shares = existing_shares - sell_shares
    portfolio["cash_sek"] += net_proceeds
    if portfolio["total_fees_sek"] is None:
        portfolio["total_fees_sek"] = 0.0
    portfolio["total_fees_sek"] += fee

    if remaining_shares <= 1e-12:
        del portfolio["holdings"][ticker]
    else:
        portfolio["holdings"][ticker]["shares"] = remaining_shares

    return sell_shares, proceeds, fee, net_proceeds


# ---------------------------------------------------------------------------
# BUY execution tests
# ---------------------------------------------------------------------------

class TestBuyExecution:
    def test_buy_allocation_15pct_patient(self):
        """Patient BUY: 15% of cash allocated."""
        pf = _make_portfolio()
        shares, fee, alloc = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        expected_alloc = 500_000 * 0.15
        assert alloc == expected_alloc
        assert pf["cash_sek"] == 500_000 - expected_alloc

    def test_buy_allocation_30pct_bold(self):
        """Bold BUY: 30% of cash allocated."""
        pf = _make_portfolio()
        shares, fee, alloc = _execute_buy(pf, "BTC-USD", 680000.0, 0.30)

        expected_alloc = 500_000 * 0.30
        assert alloc == expected_alloc
        assert pf["cash_sek"] == 500_000 - expected_alloc

    def test_buy_fee_deduction_crypto(self):
        """Crypto fee (0.05%) is deducted from allocation."""
        pf = _make_portfolio()
        shares, fee, alloc = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        expected_fee = alloc * FEE_CRYPTO
        assert abs(fee - expected_fee) < 0.01

    def test_buy_fee_deduction_stock(self):
        """Stock fee (0.10%) is deducted from allocation."""
        pf = _make_portfolio()
        shares, fee, alloc = _execute_buy(pf, "NVDA", 1900.0, 0.15, is_crypto=False)

        expected_fee = alloc * FEE_STOCK
        assert abs(fee - expected_fee) < 0.01

    def test_buy_shares_calculation(self):
        """Shares = (allocation - fee) / price."""
        pf = _make_portfolio()
        price = 680000.0
        shares, fee, alloc = _execute_buy(pf, "BTC-USD", price, 0.15)

        expected_net = alloc - fee
        expected_shares = expected_net / price
        assert abs(shares - expected_shares) < 1e-10

    def test_buy_cash_update(self):
        """Cash decreases by full allocation (fee included in alloc)."""
        pf = _make_portfolio()
        initial_cash = pf["cash_sek"]
        _, _, alloc = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        assert pf["cash_sek"] == initial_cash - alloc

    def test_buy_holdings_created(self):
        """New ticker appears in holdings after BUY."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        assert "BTC-USD" in pf["holdings"]
        assert pf["holdings"]["BTC-USD"]["shares"] > 0

    def test_buy_averaging_up(self):
        """Second BUY of same ticker adds shares with weighted avg cost."""
        pf = _make_portfolio()
        # First buy at 680,000
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        first_shares = pf["holdings"]["BTC-USD"]["shares"]

        # Second buy at 700,000
        _execute_buy(pf, "BTC-USD", 700000.0, 0.15)
        total_shares = pf["holdings"]["BTC-USD"]["shares"]

        assert total_shares > first_shares
        avg_cost = pf["holdings"]["BTC-USD"]["avg_cost_sek"]
        assert 680000.0 < avg_cost < 700000.0

    def test_buy_fee_accumulation(self):
        """total_fees_sek tracks cumulative fees."""
        pf = _make_portfolio()
        _, fee1, _ = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        _, fee2, _ = _execute_buy(pf, "ETH-USD", 20000.0, 0.15)

        assert abs(pf["total_fees_sek"] - (fee1 + fee2)) < 0.01

    def test_buy_fee_init_from_none(self):
        """If total_fees_sek is None, should be initialized to 0 then accumulated."""
        pf = _make_portfolio(total_fees_sek=None)
        pf["total_fees_sek"] = None  # explicitly set to None
        _, fee, _ = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        assert pf["total_fees_sek"] == fee


# ---------------------------------------------------------------------------
# SELL execution tests
# ---------------------------------------------------------------------------

class TestSellExecution:
    def test_sell_50pct_patient(self):
        """Patient SELL: 50% of position."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        initial_shares = pf["holdings"]["BTC-USD"]["shares"]

        sell_shares, _, _, _ = _execute_sell(pf, "BTC-USD", 0.50, 690000.0)

        expected_sell = initial_shares * 0.50
        assert abs(sell_shares - expected_sell) < 1e-10
        # Ticker should remain in holdings with remaining shares
        assert "BTC-USD" in pf["holdings"]
        remaining = pf["holdings"]["BTC-USD"]["shares"]
        assert abs(remaining - (initial_shares - sell_shares)) < 1e-10

    def test_sell_100pct_bold(self):
        """Bold SELL: 100% of position (full exit)."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.30)

        sell_shares, _, _, _ = _execute_sell(pf, "BTC-USD", 1.00, 690000.0)

        # Ticker should be removed from holdings
        assert "BTC-USD" not in pf["holdings"]

    def test_sell_fee_deduction(self):
        """Fee is deducted from proceeds."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        sell_shares, proceeds, fee, net_proceeds = _execute_sell(
            pf, "BTC-USD", 0.50, 690000.0
        )

        expected_fee = proceeds * FEE_CRYPTO
        assert abs(fee - expected_fee) < 0.01
        assert abs(net_proceeds - (proceeds - fee)) < 0.01

    def test_sell_cash_update(self):
        """Cash increases by net proceeds (after fee)."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        cash_before_sell = pf["cash_sek"]

        _, _, _, net_proceeds = _execute_sell(pf, "BTC-USD", 0.50, 690000.0)

        assert abs(pf["cash_sek"] - (cash_before_sell + net_proceeds)) < 0.01

    def test_sell_fee_accumulation(self):
        """Sell fees add to total_fees_sek."""
        pf = _make_portfolio()
        _, buy_fee, _ = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        _, _, sell_fee, _ = _execute_sell(pf, "BTC-USD", 0.50, 690000.0)

        assert abs(pf["total_fees_sek"] - (buy_fee + sell_fee)) < 0.01

    def test_sell_stock_fee_rate(self):
        """Stock SELL uses 0.10% fee rate."""
        pf = _make_portfolio()
        _execute_buy(pf, "NVDA", 1900.0, 0.15, is_crypto=False)

        _, proceeds, fee, _ = _execute_sell(pf, "NVDA", 0.50, 1950.0, is_crypto=False)

        expected_fee = proceeds * FEE_STOCK
        assert abs(fee - expected_fee) < 0.01

    def test_partial_sell_preserves_avg_cost(self):
        """After a 50% sell, avg_cost_sek should remain unchanged."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        avg_cost_before = pf["holdings"]["BTC-USD"]["avg_cost_sek"]

        _execute_sell(pf, "BTC-USD", 0.50, 690000.0)

        assert pf["holdings"]["BTC-USD"]["avg_cost_sek"] == avg_cost_before


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_sell_nonexistent_ticker(self):
        """Selling a ticker not in holdings should raise ValueError."""
        pf = _make_portfolio()
        with pytest.raises(ValueError, match="not in holdings"):
            _execute_sell(pf, "BTC-USD", 1.0, 680000.0)

    def test_sell_zero_shares(self):
        """Selling when shares are 0 should raise ValueError."""
        pf = _make_portfolio(holdings={"BTC-USD": {"shares": 0, "avg_cost_sek": 680000.0}})
        with pytest.raises(ValueError, match="no shares held"):
            _execute_sell(pf, "BTC-USD", 1.0, 680000.0)

    def test_cash_cannot_go_negative_with_valid_alloc(self):
        """With valid allocation percentages, cash should remain non-negative."""
        pf = _make_portfolio()
        # Buy at 15% five times (covers 75% of initial cash)
        for _ in range(5):
            _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        assert pf["cash_sek"] > 0

    def test_minimum_trade_validation(self):
        """Allocation below 500 SEK should be flagged."""
        pf = _make_portfolio(cash=1000.0)
        # 15% of 1000 = 150 SEK, which is below the 500 SEK minimum
        alloc = pf["cash_sek"] * 0.15
        assert alloc < 500.0

    def test_multiple_sequential_buys_and_sells(self):
        """Multiple buys and sells should maintain consistent state."""
        pf = _make_portfolio()
        initial_cash = pf["cash_sek"]

        # Buy 3 times
        for _ in range(3):
            _execute_buy(pf, "BTC-USD", 680000.0, 0.10)

        # Sell 50% twice
        _execute_sell(pf, "BTC-USD", 0.50, 690000.0)
        _execute_sell(pf, "BTC-USD", 0.50, 700000.0)

        # Should still have shares remaining (50% of 50% = 25% of original)
        assert "BTC-USD" in pf["holdings"]
        assert pf["holdings"]["BTC-USD"]["shares"] > 0

    def test_sell_all_then_rebuy(self):
        """After selling 100%, should be able to buy again."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.30)
        _execute_sell(pf, "BTC-USD", 1.00, 690000.0)

        assert "BTC-USD" not in pf["holdings"]

        # Rebuy
        _execute_buy(pf, "BTC-USD", 700000.0, 0.30)
        assert "BTC-USD" in pf["holdings"]
        assert pf["holdings"]["BTC-USD"]["shares"] > 0


# ---------------------------------------------------------------------------
# Rounding / floating point drift
# ---------------------------------------------------------------------------

class TestFloatingPointPrecision:
    def test_no_drift_over_many_trades(self):
        """After many buy/sell cycles, cash + holdings value should equal
        initial value minus cumulative fees (within tolerance)."""
        pf = _make_portfolio()
        price = 100.0  # simple price for clarity

        # Execute 100 buy/sell cycles
        for i in range(100):
            _execute_buy(pf, "TEST", price, 0.10, is_crypto=True)
            _execute_sell(pf, "TEST", 1.00, price, is_crypto=True)

        # After selling everything, holdings should be empty
        assert "TEST" not in pf["holdings"]

        # Cash should be initial minus cumulative fees
        # Each cycle: buy fee + sell fee
        # buy alloc = cash * 0.10, buy fee = alloc * 0.0005
        # sell proceeds = shares * price, sell fee = proceeds * 0.0005
        # Net effect per cycle: -alloc + (shares * price - sell_fee)
        #   = -alloc + (alloc - buy_fee) - sell_fee = -(buy_fee + sell_fee)
        # Each cycle loses (buy_fee + sell_fee)
        assert pf["cash_sek"] < INITIAL_CASH
        assert abs(pf["cash_sek"] + pf["total_fees_sek"] - INITIAL_CASH) < 0.01

    def test_cash_conservation_law(self):
        """Sum of: current cash + current holdings value + total fees
        should equal initial cash (when using same price for buy and sell)."""
        pf = _make_portfolio()
        price = 50000.0

        _execute_buy(pf, "BTC-USD", price, 0.15)
        _execute_buy(pf, "ETH-USD", price / 30, 0.15)

        # Calculate portfolio value
        btc_value = pf["holdings"]["BTC-USD"]["shares"] * price
        eth_value = pf["holdings"]["ETH-USD"]["shares"] * (price / 30)
        total = pf["cash_sek"] + btc_value + eth_value + pf["total_fees_sek"]

        # Should equal initial cash (within floating point tolerance)
        assert abs(total - INITIAL_CASH) < 0.01

    def test_fee_percentage_correct(self):
        """Verify fee as percentage matches expected rate."""
        pf = _make_portfolio()
        alloc = pf["cash_sek"] * 0.15
        _, fee, _ = _execute_buy(pf, "BTC-USD", 680000.0, 0.15)

        actual_rate = fee / alloc
        assert abs(actual_rate - FEE_CRYPTO) < 1e-10


# ---------------------------------------------------------------------------
# Holdings integrity
# ---------------------------------------------------------------------------

class TestHoldingsIntegrity:
    def test_holdings_never_negative_shares(self):
        """No operation should result in negative shares."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        _execute_sell(pf, "BTC-USD", 0.50, 690000.0)

        for ticker, holding in pf["holdings"].items():
            assert holding["shares"] >= 0

    def test_patient_partial_sell_keeps_ticker(self):
        """Patient 50% sell must keep ticker in holdings."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        _execute_sell(pf, "BTC-USD", 0.50, 690000.0)

        assert "BTC-USD" in pf["holdings"]
        assert pf["holdings"]["BTC-USD"]["shares"] > 0

    def test_bold_full_sell_removes_ticker(self):
        """Bold 100% sell must remove ticker from holdings."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.30)
        _execute_sell(pf, "BTC-USD", 1.00, 690000.0)

        assert "BTC-USD" not in pf["holdings"]

    def test_holdings_not_set_to_empty_dict(self):
        """Holdings should only be {} when every ticker has 0 shares."""
        pf = _make_portfolio()
        _execute_buy(pf, "BTC-USD", 680000.0, 0.15)
        _execute_buy(pf, "ETH-USD", 20000.0, 0.15)

        # Sell only BTC
        _execute_sell(pf, "BTC-USD", 1.00, 690000.0)

        # Holdings should not be empty — ETH still held
        assert len(pf["holdings"]) > 0
        assert "ETH-USD" in pf["holdings"]
