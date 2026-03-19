"""Tests for portfolio.trade_validation -- pre-trade sanity checks.

Covers:
- Valid BUY / SELL trades pass
- Invalid price (0, negative) rejected
- Invalid volume (0, negative) rejected
- Invalid action rejected
- Insufficient cash rejected
- Position too large rejected
- Spread too wide rejected
- Price deviation too high rejected
- Order below minimum rejected
- Warnings for near-limit spread
- Warnings for near-limit price deviation
- Edge: no bid/ask provided (spread check skipped)
- Edge: no last_known_price (deviation check skipped)
- SELL does not check cash sufficiency
- Boundary values (exact limits)
- Multiple warnings accumulate
"""


from portfolio.trade_validation import ValidationResult, validate_trade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VALID_BUY = dict(
    action="BUY",
    price=100.0,
    volume=10.0,           # order_value = 1000 SEK
    cash_available=50_000.0,
)

VALID_SELL = dict(
    action="SELL",
    price=100.0,
    volume=10.0,
    cash_available=0.0,    # SELL doesn't need cash
)


# ===================================================================
# Class 1: Valid trades pass
# ===================================================================
class TestValidTrades:
    def test_valid_buy(self):
        r = validate_trade(**VALID_BUY)
        assert r.valid is True
        assert r.reason == "All checks passed"
        assert r.warnings == []

    def test_valid_sell(self):
        r = validate_trade(**VALID_SELL)
        assert r.valid is True

    def test_valid_buy_with_all_optional_params(self):
        r = validate_trade(
            action="BUY",
            price=100.0,
            volume=5.0,
            cash_available=50_000.0,
            bid=99.5,
            ask=100.0,
            last_known_price=99.0,
        )
        assert r.valid is True

    def test_valid_sell_zero_cash(self):
        """SELL should pass even with zero cash available."""
        r = validate_trade(action="SELL", price=200.0, volume=5.0, cash_available=0.0)
        assert r.valid is True


# ===================================================================
# Class 2: Basic parameter validation
# ===================================================================
class TestBasicParams:
    def test_price_zero(self):
        r = validate_trade(action="BUY", price=0, volume=10, cash_available=50_000)
        assert r.valid is False
        assert "Invalid price" in r.reason

    def test_price_negative(self):
        r = validate_trade(action="BUY", price=-5.0, volume=10, cash_available=50_000)
        assert r.valid is False
        assert "Invalid price" in r.reason

    def test_volume_zero(self):
        r = validate_trade(action="BUY", price=100, volume=0, cash_available=50_000)
        assert r.valid is False
        assert "Invalid volume" in r.reason

    def test_volume_negative(self):
        r = validate_trade(action="BUY", price=100, volume=-1, cash_available=50_000)
        assert r.valid is False
        assert "Invalid volume" in r.reason

    def test_invalid_action(self):
        r = validate_trade(action="HOLD", price=100, volume=10, cash_available=50_000)
        assert r.valid is False
        assert "Invalid action" in r.reason

    def test_invalid_action_lowercase(self):
        r = validate_trade(action="buy", price=100, volume=10, cash_available=50_000)
        assert r.valid is False
        assert "Invalid action" in r.reason


# ===================================================================
# Class 3: Minimum order size
# ===================================================================
class TestMinimumOrder:
    def test_below_minimum(self):
        """Order value 490 SEK < 500 default minimum."""
        r = validate_trade(action="BUY", price=49.0, volume=10, cash_available=50_000)
        assert r.valid is False
        assert "below minimum" in r.reason

    def test_at_minimum(self):
        """Order value exactly 500 SEK should pass."""
        r = validate_trade(action="BUY", price=50.0, volume=10, cash_available=50_000)
        assert r.valid is True

    def test_custom_minimum(self):
        """Custom min_order_sek=1000."""
        r = validate_trade(
            action="BUY", price=100, volume=5, cash_available=50_000,
            min_order_sek=1000.0,
        )
        assert r.valid is False
        assert "below minimum" in r.reason

    def test_sell_below_minimum(self):
        """SELL also rejects orders below minimum."""
        r = validate_trade(action="SELL", price=10.0, volume=1, cash_available=0)
        assert r.valid is False
        assert "below minimum" in r.reason


# ===================================================================
# Class 4: Cash sufficiency (BUY only)
# ===================================================================
class TestCashSufficiency:
    def test_insufficient_cash(self):
        r = validate_trade(action="BUY", price=100, volume=100, cash_available=5_000)
        assert r.valid is False
        assert "Insufficient cash" in r.reason

    def test_exact_cash(self):
        """Order value exactly equals available cash and under max_cash_pct."""
        r = validate_trade(
            action="BUY", price=100, volume=10, cash_available=1_000,
            max_cash_pct=100.0,
        )
        assert r.valid is True

    def test_sell_ignores_cash(self):
        """SELL should NOT check cash sufficiency, even with zero cash."""
        r = validate_trade(action="SELL", price=1000, volume=100, cash_available=0)
        assert r.valid is True


# ===================================================================
# Class 5: Position size (BUY only)
# ===================================================================
class TestPositionSize:
    def test_position_too_large(self):
        """60% of cash with max_cash_pct=50%."""
        r = validate_trade(
            action="BUY", price=100, volume=300, cash_available=50_000,
            max_cash_pct=50.0,
        )
        assert r.valid is False
        assert "Position too large" in r.reason

    def test_position_at_limit(self):
        """Exactly 50% of cash with max_cash_pct=50%."""
        r = validate_trade(
            action="BUY", price=100, volume=250, cash_available=50_000,
            max_cash_pct=50.0,
        )
        assert r.valid is True

    def test_sell_ignores_position_size(self):
        """SELL doesn't check position size as % of cash."""
        r = validate_trade(action="SELL", price=100, volume=1000, cash_available=1_000)
        assert r.valid is True

    def test_custom_max_cash_pct(self):
        """max_cash_pct=15% (Patient strategy)."""
        r = validate_trade(
            action="BUY", price=100, volume=80, cash_available=50_000,
            max_cash_pct=15.0,
        )
        assert r.valid is False
        assert "Position too large" in r.reason
        assert "16.0%" in r.reason


# ===================================================================
# Class 6: Bid/ask spread
# ===================================================================
class TestSpread:
    def test_spread_too_wide(self):
        r = validate_trade(
            **VALID_BUY,
            bid=100.0, ask=103.0,   # 3% spread
            max_spread_pct=2.0,
        )
        assert r.valid is False
        assert "Spread too wide" in r.reason

    def test_spread_within_limit(self):
        r = validate_trade(
            **VALID_BUY,
            bid=100.0, ask=101.0,   # 1% spread
        )
        assert r.valid is True
        assert r.warnings == []

    def test_no_bid_ask_skips_check(self):
        """When bid/ask not provided, spread check is skipped."""
        r = validate_trade(**VALID_BUY)
        assert r.valid is True

    def test_bid_none_ask_provided(self):
        """If only ask is provided (bid=None), spread check is skipped."""
        r = validate_trade(**VALID_BUY, bid=None, ask=105.0)
        assert r.valid is True

    def test_bid_zero_skips_check(self):
        """If bid is 0, spread check is skipped (avoids division by zero)."""
        r = validate_trade(**VALID_BUY, bid=0.0, ask=100.0)
        assert r.valid is True


# ===================================================================
# Class 7: Price deviation
# ===================================================================
class TestPriceDeviation:
    def test_deviation_too_high(self):
        r = validate_trade(
            **VALID_BUY,
            last_known_price=90.0,   # 100 vs 90 = 11.1% deviation
        )
        assert r.valid is False
        assert "Price deviation" in r.reason

    def test_deviation_within_limit(self):
        r = validate_trade(
            **VALID_BUY,
            last_known_price=98.0,   # ~2% deviation
        )
        assert r.valid is True

    def test_no_last_known_price_skips_check(self):
        """When last_known_price not provided, deviation check is skipped."""
        r = validate_trade(**VALID_BUY)
        assert r.valid is True

    def test_last_known_price_zero_skips(self):
        """last_known_price=0 is skipped (avoids division by zero)."""
        r = validate_trade(**VALID_BUY, last_known_price=0.0)
        assert r.valid is True

    def test_custom_max_deviation(self):
        """Custom max_price_deviation_pct=1%."""
        r = validate_trade(
            **VALID_BUY,
            last_known_price=98.5,
            max_price_deviation_pct=1.0,
        )
        assert r.valid is False
        assert "Price deviation" in r.reason


# ===================================================================
# Class 8: Warnings (near-limit conditions)
# ===================================================================
class TestWarnings:
    def test_spread_warning(self):
        """Spread at 1.5% with max 2.0% triggers warning (>70% of limit)."""
        r = validate_trade(
            **VALID_BUY,
            bid=100.0, ask=101.5,   # 1.5% > 2.0*0.7=1.4%
        )
        assert r.valid is True
        assert len(r.warnings) == 1
        assert "Spread warning" in r.warnings[0]

    def test_price_deviation_warning(self):
        """Price deviation at 4% with max 5% triggers warning (>70% of limit)."""
        r = validate_trade(
            **VALID_BUY,
            last_known_price=96.0,   # ~4.17% deviation > 5*0.7=3.5%
        )
        assert r.valid is True
        assert len(r.warnings) == 1
        assert "Price moved" in r.warnings[0]

    def test_multiple_warnings(self):
        """Both spread and deviation near limits => two warnings."""
        r = validate_trade(
            action="BUY",
            price=100.0,
            volume=10.0,
            cash_available=50_000.0,
            bid=100.0,
            ask=101.5,              # 1.5% spread, near 2% limit
            last_known_price=96.0,  # ~4.17% deviation, near 5% limit
        )
        assert r.valid is True
        assert len(r.warnings) == 2

    def test_no_warning_when_well_below_limit(self):
        """Spread at 0.5% and deviation at 1% => no warnings."""
        r = validate_trade(
            **VALID_BUY,
            bid=100.0, ask=100.5,
            last_known_price=99.0,
        )
        assert r.valid is True
        assert r.warnings == []


# ===================================================================
# Class 9: ValidationResult dataclass
# ===================================================================
class TestValidationResult:
    def test_default_warnings_list(self):
        r = ValidationResult(valid=True)
        assert r.warnings == []
        # Each instance should get its own list
        r2 = ValidationResult(valid=False)
        assert r.warnings is not r2.warnings

    def test_custom_warnings(self):
        r = ValidationResult(valid=True, warnings=["test"])
        assert r.warnings == ["test"]

    def test_reason_default_empty(self):
        r = ValidationResult(valid=True)
        assert r.reason == ""


# ===================================================================
# Class 10: Edge cases and boundary conditions
# ===================================================================
class TestEdgeCases:
    def test_very_small_price(self):
        """Very small price (penny stock) should pass if order value meets minimum."""
        r = validate_trade(action="BUY", price=0.01, volume=50_000, cash_available=1_000_000)
        assert r.valid is True

    def test_very_large_order(self):
        """Large order that exceeds max_cash_pct."""
        r = validate_trade(
            action="BUY", price=50_000, volume=10, cash_available=500_000,
            max_cash_pct=50.0,
        )
        assert r.valid is False
        assert "Position too large" in r.reason

    def test_sell_large_value_no_cash_check(self):
        """Large SELL value with zero cash should pass."""
        r = validate_trade(action="SELL", price=50_000, volume=100, cash_available=0)
        assert r.valid is True

    def test_price_exactly_at_deviation_limit(self):
        """Price deviation exactly at 5% should pass (> not >=)."""
        # 100 * 1.05 = 105, deviation = 5.0% exactly
        r = validate_trade(
            action="BUY", price=105.0, volume=10, cash_available=50_000,
            last_known_price=100.0, max_price_deviation_pct=5.0,
        )
        assert r.valid is True

    def test_spread_exactly_at_limit(self):
        """Spread exactly at 2% should pass (> not >=)."""
        r = validate_trade(
            **VALID_BUY,
            bid=100.0, ask=102.0, max_spread_pct=2.0,
        )
        assert r.valid is True

    def test_price_just_above_deviation_limit(self):
        """Price deviation just above limit should fail."""
        r = validate_trade(
            action="BUY", price=105.1, volume=10, cash_available=50_000,
            last_known_price=100.0, max_price_deviation_pct=5.0,
        )
        assert r.valid is False

    def test_negative_price_deviation(self):
        """Price below last known should also be caught by abs()."""
        r = validate_trade(
            action="BUY", price=80.0, volume=10, cash_available=50_000,
            last_known_price=100.0, max_price_deviation_pct=5.0,
        )
        assert r.valid is False
        assert "Price deviation" in r.reason
