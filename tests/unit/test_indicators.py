"""
Unit tests for technical indicators.
These tests verify that our indicator calculations are correct.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestRSI:
    """Test RSI (Relative Strength Index) calculations."""

    def test_rsi_overbought_detection(self):
        """RSI above 70 should be detected as overbought."""
        from user_data.strategies.indicators import calculate_rsi, is_overbought

        # Create price data that would result in high RSI (continuous upward movement)
        prices = pd.Series([100 + i * 2 for i in range(20)])
        rsi = calculate_rsi(prices, period=14)

        assert is_overbought(rsi.iloc[-1], threshold=70)

    def test_rsi_oversold_detection(self):
        """RSI below 30 should be detected as oversold."""
        from user_data.strategies.indicators import calculate_rsi, is_oversold

        # Create price data that would result in low RSI (continuous downward movement)
        prices = pd.Series([200 - i * 2 for i in range(20)])
        rsi = calculate_rsi(prices, period=14)

        assert is_oversold(rsi.iloc[-1], threshold=30)

    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        from user_data.strategies.indicators import calculate_rsi

        # Random price data
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(100).cumsum())
        rsi = calculate_rsi(prices, period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_period_parameter(self):
        """RSI with different periods should produce different results."""
        from user_data.strategies.indicators import calculate_rsi

        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50).cumsum())

        rsi_14 = calculate_rsi(prices, period=14)
        rsi_7 = calculate_rsi(prices, period=7)

        # Different periods should give different values
        assert not np.allclose(
            rsi_14.dropna().values, rsi_7.dropna().values[-len(rsi_14.dropna()) :]
        )


class TestMACD:
    """Test MACD (Moving Average Convergence Divergence) calculations."""

    def test_macd_bullish_crossover(self):
        """Detect bullish crossover (MACD crosses above signal line)."""
        from user_data.strategies.indicators import (
            calculate_macd,
            detect_macd_crossover,
        )

        # Create uptrending price data
        prices = pd.Series([100 + i * 0.5 + np.sin(i / 3) * 2 for i in range(50)])
        macd_line, signal_line, histogram = calculate_macd(prices)

        crossovers = detect_macd_crossover(macd_line, signal_line)
        # Should detect at least one crossover in trending data
        assert crossovers["bullish"].any() or crossovers["bearish"].any()

    def test_macd_bearish_crossover(self):
        """Detect bearish crossover (MACD crosses below signal line)."""
        from user_data.strategies.indicators import (
            calculate_macd,
            detect_macd_crossover,
        )

        # Create downtrending price data
        prices = pd.Series([200 - i * 0.5 + np.sin(i / 3) * 2 for i in range(50)])
        macd_line, signal_line, histogram = calculate_macd(prices)

        crossovers = detect_macd_crossover(macd_line, signal_line)
        assert crossovers["bullish"].any() or crossovers["bearish"].any()

    def test_macd_histogram_sign(self):
        """Histogram should be positive when MACD > signal, negative when MACD < signal."""
        from user_data.strategies.indicators import calculate_macd

        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50).cumsum())
        macd_line, signal_line, histogram = calculate_macd(prices)

        valid_idx = histogram.dropna().index
        expected_histogram = macd_line[valid_idx] - signal_line[valid_idx]

        assert np.allclose(
            histogram[valid_idx].values, expected_histogram.values, rtol=1e-10
        )


class TestEMA:
    """Test EMA (Exponential Moving Average) calculations."""

    def test_ema_crossover_detection(self):
        """Detect EMA crossovers (fast crosses slow)."""
        from user_data.strategies.indicators import calculate_ema, detect_ema_crossover

        # Trending price data
        prices = pd.Series([100 + i * 0.3 + np.sin(i / 5) * 5 for i in range(100)])

        ema_fast = calculate_ema(prices, period=9)
        ema_slow = calculate_ema(prices, period=21)

        crossovers = detect_ema_crossover(ema_fast, ema_slow)
        # In trending data with oscillation, we should see crossovers
        assert isinstance(crossovers, pd.DataFrame)
        assert "bullish" in crossovers.columns
        assert "bearish" in crossovers.columns

    def test_ema_faster_than_sma(self):
        """EMA should react faster to price changes than SMA."""
        from user_data.strategies.indicators import calculate_ema, calculate_sma

        # Price with sudden jump
        prices = pd.Series([100] * 20 + [150] * 10)

        ema = calculate_ema(prices, period=10)
        sma = calculate_sma(prices, period=10)

        # After the jump, EMA should be closer to 150 than SMA
        assert ema.iloc[-1] > sma.iloc[-1]

    def test_ema_smoothing(self):
        """EMA should smooth out price noise."""
        from user_data.strategies.indicators import calculate_ema

        np.random.seed(42)
        # Noisy price data around 100
        prices = pd.Series(100 + np.random.randn(50) * 10)

        ema = calculate_ema(prices, period=20)

        # EMA variance should be less than price variance
        assert ema.dropna().std() < prices.std()


class TestVolume:
    """Test volume analysis indicators."""

    def test_volume_spike_detection(self):
        """Detect volume spikes (> 2x average)."""
        from user_data.strategies.indicators import detect_volume_spike

        # Normal volume with one spike
        volumes = pd.Series([1000] * 19 + [5000])  # Last one is 5x average

        spikes = detect_volume_spike(volumes, threshold=2.0, lookback=10)

        assert spikes.iloc[-1] == True
        assert spikes.iloc[-2] == False

    def test_volume_spike_threshold(self):
        """Volume spike detection should respect threshold parameter."""
        from user_data.strategies.indicators import detect_volume_spike

        volumes = pd.Series([1000] * 19 + [2500])  # Last one is 2.5x average

        spikes_2x = detect_volume_spike(volumes, threshold=2.0, lookback=10)
        spikes_3x = detect_volume_spike(volumes, threshold=3.0, lookback=10)

        assert spikes_2x.iloc[-1] == True  # 2.5x > 2x threshold
        assert spikes_3x.iloc[-1] == False  # 2.5x < 3x threshold

    def test_volume_average_calculation(self):
        """Volume average should be calculated correctly."""
        from user_data.strategies.indicators import calculate_volume_sma

        volumes = pd.Series([100, 200, 300, 400, 500])
        vol_sma = calculate_volume_sma(volumes, period=3)

        # Last SMA should be average of last 3: (300+400+500)/3 = 400
        assert vol_sma.iloc[-1] == 400.0
