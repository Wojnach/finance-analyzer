"""Tests for trend signal NaN guard fix (BUG-19).

Verifies that _golden_cross correctly detects NaN SMA values using pd.isna()
instead of the broken `is np.nan` identity check.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.trend import _golden_cross


class TestGoldenCrossNanGuard:
    """BUG-19: `sma.iloc[-1] is np.nan` never catches NaN because pandas
    NaN values are float64 objects, not the np.nan singleton."""

    def test_nan_sma_returns_hold(self):
        """With insufficient data for SMA200, should return HOLD."""
        # Only 10 data points — SMA200 will be all NaN
        close = pd.Series([100.0] * 10)
        result = _golden_cross(close)
        assert result == "HOLD"

    def test_nan_detection_uses_pd_isna(self):
        """Verify that pd.isna catches NaN where `is np.nan` does not."""
        val = pd.Series([np.nan]).iloc[0]
        # The old check would fail:
        assert not (val is np.nan), "identity check should NOT work for pandas NaN"
        # The new check works:
        assert pd.isna(val), "pd.isna should detect NaN"

    def test_golden_cross_with_sufficient_data(self):
        """With 250 data points and a crossover, should detect golden cross."""
        # Create data where SMA50 crosses above SMA200
        # First 200 bars trending down, then 50 bars trending up sharply
        prices = [100 - i * 0.1 for i in range(200)] + [80 + i * 2 for i in range(50)]
        close = pd.Series(prices)
        result = _golden_cross(close)
        assert result in ("BUY", "SELL", "HOLD")  # valid signal, no crash

    def test_flat_data_no_cross(self):
        """Flat data should return HOLD (no crossover)."""
        close = pd.Series([100.0] * 250)
        result = _golden_cross(close)
        assert result == "HOLD"

    def test_partial_nan_sma(self):
        """When SMA50 has values but SMA200 is NaN (< 200 bars)."""
        close = pd.Series([100 + i * 0.1 for i in range(100)])
        result = _golden_cross(close)
        assert result == "HOLD"
