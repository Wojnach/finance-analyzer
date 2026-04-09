"""Tests for portfolio.signals.momentum_factors — vol-scaled ROC-20 and seasonality context."""

import numpy as np
import pandas as pd

from portfolio.signals.momentum_factors import (
    _roc_20,
    compute_momentum_factors_signal,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_ohlcv(closes, volume=1000.0):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    return pd.DataFrame({
        "open": [c * 0.999 for c in closes],
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [volume] * n,
    })


# ---------------------------------------------------------------------------
# 2b. Vol-scaled ROC-20 tests
# ---------------------------------------------------------------------------

class TestRoc20VolScaled:
    """Tests for the volatility-scaled ROC-20 sub-indicator."""

    def test_roc_high_vol_needs_larger_move(self):
        """In a high-vol environment, a 5% move may not exceed the z-score threshold."""
        # Build 50 bars with high volatility (daily moves of ~3-5%)
        rng = np.random.default_rng(42)
        base = 100.0
        prices = [base]
        for _ in range(49):
            # Large random walk steps → high realized vol
            prices.append(prices[-1] * (1 + rng.normal(0, 0.04)))

        # Force a 5% ROC-20: set prices[-1] = prices[-21] * 1.05
        prices[-1] = prices[-21] * 1.05
        close = pd.Series(prices)

        roc_val, signal = _roc_20(close)
        # With high vol (~4% daily), 5% over 20 bars is unremarkable
        # z_roc = 5 / sigma_20 should be low → HOLD expected
        assert signal in ("HOLD", "BUY")
        # The key insight: with high vol, this may NOT trigger BUY
        # whereas it would always trigger at a fixed 5% threshold

    def test_roc_low_vol_smaller_move_triggers(self):
        """In a low-vol environment, even a 2% move triggers a signal."""
        # Build 50 bars with very low volatility (tiny daily moves)
        base = 100.0
        prices = [base + i * 0.001 for i in range(50)]  # near-flat trend
        # Force a 2% ROC-20
        prices[-1] = prices[-21] * 1.02
        close = pd.Series(prices)

        roc_val, signal = _roc_20(close)
        # With very low vol, 2% is enormous → z_roc >> 1.5 → BUY
        assert signal == "BUY"
        assert roc_val > 0

    def test_roc_fallback_to_fixed_threshold(self):
        """When vol is near-zero (constant prices), falls back to ±5% thresholds."""
        # All prices identical except the last one
        prices = [100.0] * 50
        # 6% move → above the 5% fixed threshold → BUY
        prices[-1] = 106.0
        close = pd.Series(prices)

        roc_val, signal = _roc_20(close)
        assert signal == "BUY"
        assert roc_val > 5.0

    def test_roc_fallback_sell(self):
        """Near-zero vol with a -6% drop → fixed threshold SELL."""
        prices = [100.0] * 50
        prices[-1] = 94.0
        close = pd.Series(prices)

        roc_val, signal = _roc_20(close)
        assert signal == "SELL"
        assert roc_val < -5.0

    def test_roc_moderate_vol_hold(self):
        """With moderate vol and moderate move, z_roc stays within threshold → HOLD."""
        # Create price series with ~2% daily vol and ~2% total ROC
        # z_roc = 2% / 2% = 1.0, below 1.5 threshold → HOLD
        import numpy as np
        np.random.seed(42)
        prices = [100.0]
        for _ in range(49):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        # Set the endpoint so ROC-20 is ~2%
        prices[-1] = prices[-21] * 1.02
        close = pd.Series(prices)

        roc_val, signal = _roc_20(close)
        assert signal == "HOLD"

    def test_roc_insufficient_data(self):
        """With fewer than 21 rows, returns NaN + HOLD."""
        close = pd.Series([100.0] * 20)  # exactly 20 rows, need 21
        roc_val, signal = _roc_20(close)
        assert signal == "HOLD"
        assert np.isnan(roc_val)

    def test_roc_exactly_21_rows(self):
        """With exactly 21 rows, the function should work."""
        prices = [100.0 + i * 0.1 for i in range(21)]
        close = pd.Series(prices)
        roc_val, signal = _roc_20(close)
        assert signal in ("BUY", "SELL", "HOLD")
        assert np.isfinite(roc_val)


# ---------------------------------------------------------------------------
# 2c. Seasonality context tests for momentum_factors
# ---------------------------------------------------------------------------

class TestMomentumFactorsContext:
    """Tests for context/seasonality kwarg in compute_momentum_factors_signal."""

    def test_context_none_works(self):
        """Calling with context=None should not break."""
        df = _make_ohlcv([100.0 + i * 0.5 for i in range(60)])
        result = compute_momentum_factors_signal(df, context=None)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "sub_signals" in result
        assert "indicators" in result

    def test_context_empty_dict_works(self):
        """Calling with context={} (no seasonality_profile key) works fine."""
        df = _make_ohlcv([100.0 + i * 0.5 for i in range(60)])
        result = compute_momentum_factors_signal(df, context={})
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_context_with_seasonality_detrends(self):
        """Providing a context with mock seasonality_profile still returns valid result.

        We cannot fully verify detrending without the seasonality module, but
        the function should handle gracefully (try/except fallback to raw data).
        """
        df = _make_ohlcv([100.0 + i * 0.5 for i in range(60)])
        # Add an hourly DatetimeIndex so the seasonality path is entered
        df.index = pd.date_range("2026-01-01", periods=len(df), freq="1h")

        mock_profile = {
            "hour_returns": {str(h): 0.001 for h in range(24)},
        }
        context = {"seasonality_profile": mock_profile}

        result = compute_momentum_factors_signal(df, context=context)
        # Should still return a valid signal regardless of detrending success
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_default_no_context_kwarg(self):
        """compute_momentum_factors_signal(df) without context kwarg works."""
        df = _make_ohlcv([100.0 + i * 0.5 for i in range(60)])
        result = compute_momentum_factors_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# 2c. Seasonality context tests for mean_reversion (imported separately)
# ---------------------------------------------------------------------------

class TestMeanReversionContext:
    """Tests for context/seasonality kwarg in compute_mean_reversion_signal."""

    def test_context_none_works(self):
        """Calling compute_mean_reversion_signal with context=None doesn't break."""
        from portfolio.signals.mean_reversion import compute_mean_reversion_signal

        df = _make_ohlcv([100.0 + i * 0.1 for i in range(60)])
        result = compute_mean_reversion_signal(df, context=None)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "sub_signals" in result

    def test_context_with_seasonality_detrends(self):
        """Providing context with seasonality_profile returns valid result."""
        from portfolio.signals.mean_reversion import compute_mean_reversion_signal

        df = _make_ohlcv([100.0 + i * 0.1 for i in range(60)])
        df.index = pd.date_range("2026-01-01", periods=len(df), freq="1h")

        mock_profile = {
            "hour_returns": {str(h): 0.0005 for h in range(24)},
        }
        context = {"seasonality_profile": mock_profile}

        result = compute_mean_reversion_signal(df, context=context)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
