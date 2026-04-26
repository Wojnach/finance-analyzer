"""Tests for BUG-228: _estimate_entry_underlying must not permanently save
a fallback value when the warrant return formula inputs are invalid."""

import pytest

from portfolio.fin_snipe_manager import _estimate_entry_underlying


class TestEstimateEntryUnderlying:
    """BUG-228: entry underlying recovery validation."""

    def test_saved_value_takes_priority(self):
        """When entry_underlying is already saved, use it directly."""
        snapshot = {"current_underlying": 33.0}
        state = {"entry_underlying": 32.5}
        assert _estimate_entry_underlying(snapshot, state) == 32.5

    def test_valid_computation(self):
        """Normal case: compute entry underlying from warrant return formula."""
        # Current warrant: bought at 10 SEK, now 11 SEK (10% gain)
        # Leverage 5x → underlying moved 2%
        # Current underlying 33.66 → entry underlying = 33.66 / 1.02 = 33.0
        snapshot = {
            "current_underlying": 33.66,
            "current_instrument_price": 11.0,
            "position_average_price": 10.0,
            "leverage": 5.0,
        }
        state = {}
        result = _estimate_entry_underlying(snapshot, state)
        assert result == pytest.approx(33.0, abs=0.01)

    def test_returns_sentinel_when_entry_price_zero(self):
        """BUG-228: when entry_price is 0, return -1.0 sentinel (not current_underlying)."""
        snapshot = {
            "current_underlying": 33.0,
            "current_instrument_price": 11.0,
            "position_average_price": 0.0,  # Missing from API
            "leverage": 5.0,
        }
        state = {}
        result = _estimate_entry_underlying(snapshot, state)
        assert result == -1.0

    def test_returns_sentinel_when_leverage_zero(self):
        """Return -1.0 sentinel when leverage is missing."""
        snapshot = {
            "current_underlying": 33.0,
            "current_instrument_price": 11.0,
            "position_average_price": 10.0,
            "leverage": 0.0,
        }
        state = {}
        result = _estimate_entry_underlying(snapshot, state)
        assert result == -1.0

    def test_returns_sentinel_when_underlying_zero(self):
        """Return -1.0 sentinel when current_underlying is missing."""
        snapshot = {
            "current_underlying": 0.0,
            "current_instrument_price": 11.0,
            "position_average_price": 10.0,
            "leverage": 5.0,
        }
        state = {}
        result = _estimate_entry_underlying(snapshot, state)
        assert result == -1.0

    def test_returns_sentinel_on_degenerate_base(self):
        """Return -1.0 when the formula denominator is <= 0."""
        # instrument lost more than 100% / leverage → base = 1 + (-2.0) = -1.0
        snapshot = {
            "current_underlying": 33.0,
            "current_instrument_price": 1.0,
            "position_average_price": 100.0,  # 99% loss
            "leverage": 1.0,  # -99% return, base = 0.01 > 0 — this won't trigger
        }
        state = {}
        # Actually, for base <= 0 we need instrument_return/leverage < -1.0
        # price=1, entry=100 → return = -0.99, /lev=1 → base = 0.01 > 0
        # So use leverage=0.5: return = -0.99, /0.5 → -1.98, base = -0.98
        snapshot["leverage"] = 0.5
        result = _estimate_entry_underlying(snapshot, state)
        assert result == -1.0

    def test_saved_zero_treated_as_missing(self):
        """entry_underlying=0 in state should trigger re-estimation."""
        snapshot = {
            "current_underlying": 33.66,
            "current_instrument_price": 11.0,
            "position_average_price": 10.0,
            "leverage": 5.0,
        }
        state = {"entry_underlying": 0.0}
        result = _estimate_entry_underlying(snapshot, state)
        # Should re-compute, not return 0.0
        assert result == pytest.approx(33.0, abs=0.01)
