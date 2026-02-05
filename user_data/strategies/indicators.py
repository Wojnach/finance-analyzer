"""
Technical indicators for the Finance Analyzer.
Implements RSI, MACD, EMA, SMA, and volume analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple

# =============================================================================
# RSI (Relative Strength Index)
# =============================================================================


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def is_overbought(rsi_value: float, threshold: float = 70) -> bool:
    """Check if RSI indicates overbought condition."""
    return rsi_value > threshold


def is_oversold(rsi_value: float, threshold: float = 30) -> bool:
    """Check if RSI indicates oversold condition."""
    return rsi_value < threshold


# =============================================================================
# MACD (Moving Average Convergence Divergence)
# =============================================================================


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def detect_macd_crossover(macd_line: pd.Series, signal_line: pd.Series) -> pd.DataFrame:
    """
    Detect MACD crossovers.

    Args:
        macd_line: MACD line series
        signal_line: Signal line series

    Returns:
        DataFrame with 'bullish' and 'bearish' columns indicating crossovers
    """
    macd_above = macd_line > signal_line
    macd_above_shifted = macd_above.shift(1)

    bullish = macd_above & ~macd_above_shifted  # Was below, now above
    bearish = ~macd_above & macd_above_shifted  # Was above, now below

    return pd.DataFrame({"bullish": bullish, "bearish": bearish})


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate EMA (Exponential Moving Average).

    Args:
        prices: Series of prices
        period: EMA period

    Returns:
        Series of EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate SMA (Simple Moving Average).

    Args:
        prices: Series of prices
        period: SMA period

    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def detect_ema_crossover(fast_ema: pd.Series, slow_ema: pd.Series) -> pd.DataFrame:
    """
    Detect EMA crossovers.

    Args:
        fast_ema: Fast EMA series
        slow_ema: Slow EMA series

    Returns:
        DataFrame with 'bullish' and 'bearish' columns indicating crossovers
    """
    fast_above = fast_ema > slow_ema
    fast_above_shifted = fast_above.shift(1)

    bullish = fast_above & ~fast_above_shifted  # Was below, now above
    bearish = ~fast_above & fast_above_shifted  # Was above, now below

    return pd.DataFrame({"bullish": bullish, "bearish": bearish})


# =============================================================================
# Volume Analysis
# =============================================================================


def calculate_volume_sma(volumes: pd.Series, period: int) -> pd.Series:
    """
    Calculate volume SMA.

    Args:
        volumes: Series of volume data
        period: SMA period

    Returns:
        Series of volume SMA values
    """
    return volumes.rolling(window=period).mean()


def detect_volume_spike(
    volumes: pd.Series, threshold: float = 2.0, lookback: int = 20
) -> pd.Series:
    """
    Detect volume spikes above average.

    Args:
        volumes: Series of volume data
        threshold: Multiplier for average volume (default 2.0 = 200%)
        lookback: Period for calculating average volume

    Returns:
        Boolean series indicating volume spikes
    """
    vol_avg = volumes.rolling(window=lookback).mean()
    return volumes > (vol_avg * threshold)
