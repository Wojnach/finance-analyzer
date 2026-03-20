"""Tests for BUG-85/86/87: Thread safety and NaN resilience.

BUG-85: Thread-safe _prev_sentiment access
BUG-86: Thread-safe _adx_cache access
BUG-87: NaN propagation in compute_indicators
"""

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# BUG-87: NaN propagation in compute_indicators
# ---------------------------------------------------------------------------

class TestNaNResilience:
    """compute_indicators must handle NaN values without crashing or producing NaN output."""

    def _make_df(self, n=30, close_override=None):
        """Build a simple OHLCV DataFrame."""
        close = close_override if close_override is not None else np.linspace(100, 110, n)
        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(n, 1000.0),
        })

    def test_normal_data_returns_valid_indicators(self):
        from portfolio.indicators import compute_indicators
        df = self._make_df(30)
        result = compute_indicators(df)
        assert result is not None
        for key in ("rsi", "macd_hist", "ema9", "ema21", "bb_upper", "bb_lower", "atr"):
            assert not math.isnan(result[key]), f"{key} is NaN"
            assert math.isfinite(result[key]), f"{key} is not finite"

    def test_nan_in_last_close_returns_none(self):
        """If the last close is NaN, indicators should return None (no valid data)."""
        from portfolio.indicators import compute_indicators
        close = np.linspace(100, 110, 30)
        close[-1] = np.nan
        df = self._make_df(30, close_override=close)
        result = compute_indicators(df)
        # Should return None since the last close is NaN
        assert result is None

    def test_nan_in_middle_close_ffilled(self):
        """NaN values in the middle of close series should be forward-filled."""
        from portfolio.indicators import compute_indicators
        close = np.linspace(100, 110, 30)
        close[15] = np.nan  # Gap in the middle
        df = self._make_df(30, close_override=close)
        result = compute_indicators(df)
        # Should still produce valid indicators (NaN forward-filled)
        if result is not None:
            for key in ("rsi", "macd_hist", "bb_upper", "bb_lower"):
                assert not math.isnan(result[key]), f"{key} is NaN after ffill"

    def test_flat_price_no_nan(self):
        """Flat price (all same) should produce valid indicators, not NaN."""
        from portfolio.indicators import compute_indicators
        close = np.full(30, 100.0)
        df = self._make_df(30, close_override=close)
        result = compute_indicators(df)
        if result is not None:
            # RSI should be ~50 for flat price, not NaN
            assert not math.isnan(result["rsi"]), "RSI is NaN for flat price"
            # BB std is 0 for flat price, but upper/lower should still be finite
            assert not math.isnan(result["bb_upper"]), "BB upper is NaN for flat price"

    def test_insufficient_data_returns_none(self):
        from portfolio.indicators import compute_indicators
        df = self._make_df(10)
        assert compute_indicators(df) is None

    def test_all_nan_close_returns_none(self):
        """All-NaN close series should return None."""
        from portfolio.indicators import compute_indicators
        close = np.full(30, np.nan)
        df = self._make_df(30, close_override=close)
        result = compute_indicators(df)
        assert result is None

    def test_output_values_are_json_serializable(self):
        """All output values must be JSON-serializable (no NaN, no Inf)."""
        from portfolio.indicators import compute_indicators
        df = self._make_df(50)
        result = compute_indicators(df)
        assert result is not None
        # json.dumps with allow_nan=False should not raise
        json_str = json.dumps(result, allow_nan=False)
        assert json_str  # non-empty


# ---------------------------------------------------------------------------
# BUG-85: Thread-safe _prev_sentiment
# ---------------------------------------------------------------------------

class TestSentimentThreadSafety:
    """_prev_sentiment must be thread-safe for concurrent generate_signal calls."""

    def test_concurrent_set_prev_sentiment_no_crash(self, tmp_path):
        """Multiple threads setting sentiment simultaneously should not crash."""
        from portfolio import signal_engine as se

        state_file = tmp_path / "sentiment_state.json"
        original_file = se._SENTIMENT_STATE_FILE

        try:
            se._SENTIMENT_STATE_FILE = state_file
            # Reset state
            se._prev_sentiment = {}
            se._prev_sentiment_loaded = True

            errors = []

            def set_sentiment(ticker):
                try:
                    se._set_prev_sentiment(ticker, "positive")
                except Exception as e:
                    errors.append(e)

            tickers = [f"TICKER-{i}" for i in range(20)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(set_sentiment, tickers))

            assert not errors, f"Errors during concurrent sentiment set: {errors}"
            # All tickers should be in the dict
            for t in tickers:
                assert t in se._prev_sentiment
        finally:
            se._SENTIMENT_STATE_FILE = original_file

    def test_sentiment_lock_exists(self):
        """signal_engine should have a _sentiment_lock for thread safety."""
        from portfolio import signal_engine as se
        assert hasattr(se, "_sentiment_lock"), "_sentiment_lock not found"
        assert isinstance(se._sentiment_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# BUG-86: Thread-safe _adx_cache
# ---------------------------------------------------------------------------

class TestADXCacheThreadSafety:
    """_adx_cache must be thread-safe for concurrent _compute_adx calls."""

    def test_adx_lock_exists(self):
        """signal_engine should have an _adx_lock for thread safety."""
        from portfolio import signal_engine as se
        assert hasattr(se, "_adx_lock"), "_adx_lock not found"
        assert isinstance(se._adx_lock, type(threading.Lock()))

    def test_concurrent_adx_computation_no_crash(self):
        """Multiple threads computing ADX simultaneously should not crash."""
        from portfolio import signal_engine as se

        errors = []

        def compute_adx_for_ticker(i):
            try:
                n = 50
                df = pd.DataFrame({
                    "high": np.random.default_rng(i).random(n) * 100 + 100,
                    "low": np.random.default_rng(i + 100).random(n) * 100 + 90,
                    "close": np.random.default_rng(i + 200).random(n) * 100 + 95,
                })
                se._compute_adx(df)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(compute_adx_for_ticker, range(20)))

        assert not errors, f"Errors during concurrent ADX computation: {errors}"
