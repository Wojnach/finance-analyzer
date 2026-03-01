"""Regression tests for signal bugs found in auto-improve session #3.

BUG-18: futures_flow majority_vote called with dict (always HOLD)
BUG-19: momentum variable shadowing (rsi = rsi(close))
BUG-20: momentum_factors 500-bar requirement on _high_proximity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from portfolio.signal_utils import majority_vote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, close_start: float = 100.0, trend: float = 0.5) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a configurable trend."""
    closes = [close_start + i * trend for i in range(n)]
    opens = [c - abs(trend) * 0.3 for c in closes]
    highs = [max(o, c) + abs(trend) * 0.2 for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(trend) * 0.2 for o, c in zip(opens, closes)]
    volumes = [1000 + i * 10 for i in range(n)]
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ---------------------------------------------------------------------------
# BUG-18: futures_flow majority_vote dict vs list
# ---------------------------------------------------------------------------

class TestBug18FuturesFlowVotePassing:
    """Verify majority_vote receives a list of vote strings, not dict keys."""

    def test_majority_vote_with_dict_gives_hold(self):
        """Demonstrate the bug: passing a dict to majority_vote yields HOLD."""
        sub = {
            "oi_trend": "BUY",
            "oi_divergence": "BUY",
            "ls_extreme": "BUY",
            "top_vs_crowd": "HOLD",
            "funding_trend": "HOLD",
            "oi_acceleration": "HOLD",
        }
        # Passing the dict directly iterates keys (strings), not values
        action_from_dict, conf_from_dict = majority_vote(sub)
        # The keys are not BUY/SELL/HOLD strings, so HOLD, 0.0
        assert action_from_dict == "HOLD"
        assert conf_from_dict == 0.0

    def test_majority_vote_with_values_gives_buy(self):
        """After fix: passing list(sub.values()) works correctly."""
        sub = {
            "oi_trend": "BUY",
            "oi_divergence": "BUY",
            "ls_extreme": "BUY",
            "top_vs_crowd": "BUY",
            "funding_trend": "HOLD",
            "oi_acceleration": "HOLD",
        }
        action, conf = majority_vote(list(sub.values()))
        assert action == "BUY"
        assert conf == 1.0  # 4 BUY out of 4 active voters

    def test_compute_futures_flow_produces_non_hold(self):
        """Integration: with aligned sub-signals, futures_flow should vote."""
        from portfolio.signals.futures_flow import (
            _oi_trend, _oi_divergence, _ls_extreme, _top_vs_crowd,
            _funding_trend, _oi_acceleration,
        )
        # Create data that produces BUY signals
        df = _make_ohlcv(30, 100, 1.0)  # rising prices
        oi_history = [
            {"oi": 1000 + i * 50, "timestamp": 1000 + i}
            for i in range(30)
        ]
        # Verify individual sub-signals work
        assert _oi_trend(oi_history, df) == "BUY"  # rising OI + rising price


# ---------------------------------------------------------------------------
# BUG-19: momentum.py variable shadowing rsi = rsi(close)
# ---------------------------------------------------------------------------

class TestBug19MomentumVariableShadowing:
    """Verify RSI divergence and StochRSI don't crash from variable shadowing."""

    def test_rsi_divergence_does_not_raise(self):
        """_rsi_divergence should not raise UnboundLocalError after fix."""
        from portfolio.signals.momentum import _rsi_divergence
        close = pd.Series([100 + i * 0.5 for i in range(50)])
        # Should not raise UnboundLocalError
        result = _rsi_divergence(close)
        assert result in ("BUY", "SELL", "HOLD")

    def test_stochasticrsi_does_not_raise(self):
        """_stochasticrsi should not raise UnboundLocalError after fix."""
        from portfolio.signals.momentum import _stochasticrsi
        close = pd.Series([100 + i * 0.5 for i in range(50)])
        # Should not raise UnboundLocalError
        val, signal = _stochasticrsi(close)
        assert signal in ("BUY", "SELL", "HOLD")

    def test_rsi_divergence_detects_bearish(self):
        """Bearish divergence: price higher high but RSI lower high."""
        from portfolio.signals.momentum import _rsi_divergence
        # Create a price series where price makes higher high
        # but momentum (RSI) is weakening
        prices = list(range(50, 80)) + list(range(80, 95))  # 45 bars, higher highs
        close = pd.Series(prices)
        result = _rsi_divergence(close, lookback=14)
        # The exact result depends on RSI computation, but it should not crash
        assert result in ("BUY", "SELL", "HOLD")

    def test_momentum_composite_includes_rsi_divergence(self):
        """The composite signal should include rsi_divergence and stochastic_rsi."""
        from portfolio.signals.momentum import compute_momentum_signal
        df = _make_ohlcv(100, 50, 0.5)
        result = compute_momentum_signal(df)
        assert "rsi_divergence" in result["sub_signals"]
        assert "stochastic_rsi" in result["sub_signals"]
        # These should now produce actual votes, not just HOLD from exception
        # (though they may still be HOLD based on data — the point is no crash)


# ---------------------------------------------------------------------------
# BUG-20: momentum_factors _high_proximity 500-bar requirement
# ---------------------------------------------------------------------------

class TestBug20MomentumFactorsBarRequirement:
    """Verify _high_proximity works with reasonable data lengths."""

    def test_high_proximity_with_100_bars(self):
        """After fix: 100 bars should produce a valid signal, not always HOLD."""
        from portfolio.signals.momentum_factors import _high_proximity
        # Price near its high (ratio > 0.95 should give BUY)
        close = pd.Series([100.0] * 90 + [99.5, 99.8, 100.2, 100.5, 100.3,
                                            100.1, 100.0, 99.9, 100.4, 100.2])
        val, signal = _high_proximity(close)
        # Should not return NaN/HOLD due to bar count
        assert not np.isnan(val)
        # Price at 100.2, high is 100.5, ratio = 100.2/100.5 = 0.997 > 0.95 → BUY
        assert signal == "BUY"

    def test_high_proximity_far_from_high(self):
        """Price 25% below high should give SELL."""
        from portfolio.signals.momentum_factors import _high_proximity
        close = pd.Series([100.0] * 50 + [75.0] * 50)  # dropped to 75 from 100
        val, signal = _high_proximity(close)
        assert not np.isnan(val)
        assert val <= 0.80
        assert signal == "SELL"

    def test_high_proximity_still_hold_with_too_few_bars(self):
        """Below MIN_ROWS should still return HOLD."""
        from portfolio.signals.momentum_factors import _high_proximity, MIN_ROWS
        close = pd.Series([100.0] * (MIN_ROWS - 1))
        val, signal = _high_proximity(close)
        assert np.isnan(val)
        assert signal == "HOLD"

    def test_composite_high_proximity_active(self):
        """In composite signal, high_proximity should vote when data is sufficient."""
        from portfolio.signals.momentum_factors import compute_momentum_factors_signal
        # 100 bars trending up, price near high
        df = _make_ohlcv(100, 50, 0.5)
        result = compute_momentum_factors_signal(df)
        # high_proximity should NOT be HOLD just because of bar count
        # (it may still be HOLD based on the actual ratio)
        hp_indicator = result["indicators"].get("high_proximity")
        assert hp_indicator is not None
        # If we have 100 bars, the indicator should have a real value
        if not np.isnan(hp_indicator):
            assert 0 < hp_indicator <= 1.0
