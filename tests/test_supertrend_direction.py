"""Tests for Supertrend direction tracking fix in trend.py.

The fix uses direction[i-1] (integer state) instead of comparing
supertrend[i-1] == upper_band[i-1] (float equality), which avoids
floating-point epsilon issues.
"""

import numpy as np
import pandas as pd

from portfolio.signals.trend import _supertrend


class TestSupertrendDirection:
    """Verify direction flips correctly using integer state, not float equality."""

    def test_downtrend_to_uptrend_flip(self):
        """Price in downtrend that breaks above upper_band flips direction to +1."""
        n = 40
        # Start with a declining trend, then break out upward
        prices = np.concatenate([
            np.linspace(110, 90, 25),   # clear downtrend
            np.linspace(91, 120, 15),   # sharp breakout
        ])
        high = pd.Series(prices + 1.0)
        low = pd.Series(prices - 1.0)
        close = pd.Series(prices)

        _, direction = _supertrend(high, low, close, period=10, multiplier=3.0)

        # During the downtrend section, direction should be -1 at some point
        downtrend_region = direction.iloc[15:24]
        assert (downtrend_region == -1).any(), "Expected downtrend before breakout"

        # After the breakout, direction should flip to +1
        assert direction.iloc[-1] == 1, "Expected uptrend after breakout"

    def test_uptrend_to_downtrend_flip(self):
        """Price in uptrend that breaks below lower_band flips direction to -1."""
        n = 40
        prices = np.concatenate([
            np.linspace(90, 110, 25),   # clear uptrend
            np.linspace(109, 80, 15),   # sharp breakdown
        ])
        high = pd.Series(prices + 1.0)
        low = pd.Series(prices - 1.0)
        close = pd.Series(prices)

        _, direction = _supertrend(high, low, close, period=10, multiplier=3.0)

        # During the uptrend section, direction should be +1 at some point
        uptrend_region = direction.iloc[15:24]
        assert (uptrend_region == 1).any(), "Expected uptrend before breakdown"

        # After the breakdown, direction should flip to -1
        assert direction.iloc[-1] == -1, "Expected downtrend after breakdown"

    def test_direction_is_integer_not_float_comparison(self):
        """Direction values are exactly +1 or -1 (integers), never NaN."""
        prices = np.concatenate([
            np.linspace(100, 105, 20),
            np.linspace(104.9, 95, 20),
        ])
        high = pd.Series(prices + 0.5)
        low = pd.Series(prices - 0.5)
        close = pd.Series(prices)

        _, direction = _supertrend(high, low, close, period=10, multiplier=3.0)

        valid = direction.dropna()
        assert set(valid.unique()).issubset({1, -1}), \
            f"Direction should only contain 1 or -1, got {valid.unique()}"
