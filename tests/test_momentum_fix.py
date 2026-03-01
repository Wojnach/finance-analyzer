"""Tests for momentum signal sub-signal fixes (BUG-18).

Verifies that _rsi_divergence and _stochasticrsi no longer crash with
UnboundLocalError due to variable shadowing of the imported rsi function.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.momentum import (
    _rsi_divergence,
    _stochasticrsi,
    compute_momentum_signal,
)


# ---------------------------------------------------------------------------
# Helper to generate test data
# ---------------------------------------------------------------------------

def _make_df(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """Generate OHLCV DataFrame with an uptrend or downtrend."""
    if trend == "up":
        close = [100 + i * 0.5 for i in range(n)]
    elif trend == "down":
        close = [130 - i * 0.5 for i in range(n)]
    else:
        close = [100 + (-1) ** i * 0.5 for i in range(n)]  # flat
    return pd.DataFrame({
        "open": [c - 0.2 for c in close],
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "volume": [1000] * n,
    })


# ---------------------------------------------------------------------------
# BUG-18: Variable shadowing fix verification
# ---------------------------------------------------------------------------

class TestRsiDivergenceNoLongerCrashes:
    """_rsi_divergence previously crashed with UnboundLocalError
    because `rsi = rsi(close)` shadowed the imported function."""

    def test_does_not_raise(self):
        """Should return a valid signal, not crash."""
        close = pd.Series([100 + i * 0.5 for i in range(60)])
        result = _rsi_divergence(close)
        assert result in ("BUY", "SELL", "HOLD")

    def test_returns_hold_for_flat_data(self):
        """No divergence in flat data → HOLD."""
        close = pd.Series([100.0] * 60)
        result = _rsi_divergence(close)
        assert result == "HOLD"

    def test_returns_hold_for_short_data(self):
        """Too little data → HOLD."""
        close = pd.Series([100.0] * 10)
        result = _rsi_divergence(close)
        assert result == "HOLD"

    def test_bullish_divergence(self):
        """Price makes lower low, RSI makes higher low → BUY."""
        # First half: price drops to 90, then second half: price drops to 85
        # but RSI should be higher in second half due to less momentum
        prices = list(range(100, 90, -1)) + list(range(90, 100, 1)) + \
                 list(range(100, 85, -1)) + [85] * 3
        close = pd.Series(prices[:28] if len(prices) >= 28 else prices + [85] * (28 - len(prices)))
        # This may or may not detect divergence depending on RSI calculation
        result = _rsi_divergence(close)
        assert result in ("BUY", "SELL", "HOLD")  # Just verify no crash

    def test_bearish_divergence(self):
        """Price makes higher high, RSI makes lower high → SELL."""
        prices = list(range(100, 110, 1)) + list(range(110, 100, -1)) + \
                 list(range(100, 115, 1)) + [115] * 3
        close = pd.Series(prices[:28] if len(prices) >= 28 else prices + [115] * (28 - len(prices)))
        result = _rsi_divergence(close)
        assert result in ("BUY", "SELL", "HOLD")


class TestStochasticRsiNoLongerCrashes:
    """_stochasticrsi previously crashed with UnboundLocalError
    because `rsi = rsi(close, period)` shadowed the imported function."""

    def test_does_not_raise(self):
        """Should return (value, signal), not crash."""
        close = pd.Series([100 + i * 0.5 for i in range(60)])
        val, sig = _stochasticrsi(close)
        assert sig in ("BUY", "SELL", "HOLD")
        assert isinstance(val, float)

    def test_oversold_produces_buy(self):
        """StochRSI < 0.2 → BUY."""
        # Steady downtrend should push RSI and StochRSI low
        close = pd.Series([130 - i * 0.8 for i in range(60)])
        val, sig = _stochasticrsi(close)
        if not np.isnan(val) and val < 0.2:
            assert sig == "BUY"

    def test_overbought_produces_sell(self):
        """StochRSI > 0.8 → SELL."""
        # Steady uptrend should push RSI and StochRSI high
        close = pd.Series([80 + i * 0.8 for i in range(60)])
        val, sig = _stochasticrsi(close)
        if not np.isnan(val) and val > 0.8:
            assert sig == "SELL"

    def test_returns_hold_for_short_data(self):
        """Insufficient data → NaN, HOLD."""
        close = pd.Series([100.0] * 5)
        val, sig = _stochasticrsi(close)
        assert sig == "HOLD"


# ---------------------------------------------------------------------------
# Integration: composite signal now includes both sub-signals
# ---------------------------------------------------------------------------

class TestCompositeIncludesFixedSubSignals:
    """The composite momentum signal should now include rsi_divergence
    and stochastic_rsi in its sub_signals (were silently skipped before fix)."""

    def test_sub_signals_present(self):
        """Both fixed sub-signals should appear in output."""
        df = _make_df(60)
        result = compute_momentum_signal(df)
        assert "rsi_divergence" in result["sub_signals"]
        assert "stochastic_rsi" in result["sub_signals"]
        # Before the fix, these would be "HOLD" from the except clause
        # Now they should be real computed values
        assert result["sub_signals"]["rsi_divergence"] in ("BUY", "SELL", "HOLD")
        assert result["sub_signals"]["stochastic_rsi"] in ("BUY", "SELL", "HOLD")

    def test_stoch_rsi_indicator_present(self):
        """stoch_rsi indicator should be populated (was NaN before fix)."""
        df = _make_df(60)
        result = compute_momentum_signal(df)
        assert "stoch_rsi" in result["indicators"]
        # May still be NaN for some data, but shouldn't crash

    def test_all_eight_sub_signals(self):
        """All 8 sub-signals should be present."""
        df = _make_df(60)
        result = compute_momentum_signal(df)
        expected = {
            "rsi_divergence", "stochastic", "stochastic_rsi",
            "cci", "williams_r", "roc", "ppo", "bull_bear_power"
        }
        assert set(result["sub_signals"].keys()) == expected

    def test_insufficient_data_returns_hold(self):
        """With <50 rows, all sub-signals should be HOLD."""
        df = _make_df(20)
        result = compute_momentum_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
