"""Tests for portfolio.feature_normalizer — rolling z-score normalization."""

import math

import numpy as np
import pytest

from portfolio import feature_normalizer as fn


@pytest.fixture(autouse=True)
def _clear_buffers():
    """Ensure each test starts with empty normalizer state."""
    fn.clear()
    yield
    fn.clear()


# ---- test_normalize_returns_raw_when_insufficient_data -------------------

def test_normalize_returns_raw_when_insufficient_data():
    """With fewer than 20 samples, normalize() should return the raw value."""
    for i in range(19):
        fn.update("BTC-USD", "rsi_14", 50.0 + i)
    # 19 samples — below _MIN_SAMPLES (20)
    result = fn.normalize("BTC-USD", "rsi_14", 72.0)
    assert result == 72.0


# ---- test_normalize_returns_zscore_with_history --------------------------

def test_normalize_returns_zscore_with_history():
    """With 30 identical values, normalizing a different value gives a non-zero z-score."""
    for _ in range(30):
        fn.update("XAG-USD", "rsi_14", 50.0)
    # The mean is 50.0, std is 0 for identical values — but the code guards
    # std < 1e-12 and returns 0.0.  So we need some variance.
    fn.clear()
    for i in range(30):
        fn.update("XAG-USD", "rsi_14", 50.0 + (i % 5))
    z = fn.normalize("XAG-USD", "rsi_14", 60.0)
    # 60.0 is well above the mean (~52), so z-score should be positive
    assert z != 60.0  # not raw
    assert z > 0      # above mean


# ---- test_update_ignores_nan ---------------------------------------------

def test_update_ignores_nan():
    """update() with float('nan') should not add to the buffer."""
    fn.update("ETH-USD", "macd", float("nan"))
    fn.update("ETH-USD", "macd", float("inf"))
    fn.update("ETH-USD", "macd", float("-inf"))
    stats = fn.get_stats("ETH-USD", "macd")
    assert stats is None  # no data recorded


# ---- test_has_sufficient_history -----------------------------------------

def test_has_sufficient_history():
    """False before 20 samples, True at 20 and beyond."""
    assert fn.has_sufficient_history("A", "x") is False

    for i in range(19):
        fn.update("A", "x", float(i))
    assert fn.has_sufficient_history("A", "x") is False

    fn.update("A", "x", 19.0)
    assert fn.has_sufficient_history("A", "x") is True

    fn.update("A", "x", 20.0)
    assert fn.has_sufficient_history("A", "x") is True


# ---- test_get_stats ------------------------------------------------------

def test_get_stats():
    """get_stats returns dict with expected keys after data is recorded."""
    for i in range(25):
        fn.update("BTC-USD", "volume", float(i))
    stats = fn.get_stats("BTC-USD", "volume")
    assert stats is not None
    assert set(stats.keys()) == {"mean", "std", "count", "min", "max"}
    assert stats["count"] == 25
    assert stats["min"] == 0.0
    assert stats["max"] == 24.0


def test_get_stats_returns_none_for_unknown():
    """get_stats returns None when no data has been recorded."""
    result = fn.get_stats("UNKNOWN", "indicator")
    assert result is None


# ---- test_clear_removes_all ----------------------------------------------

def test_clear_removes_all():
    """clear() with no args removes all ticker+indicator buffers."""
    fn.update("A", "x", 1.0)
    fn.update("B", "y", 2.0)
    fn.clear()
    assert fn.get_stats("A", "x") is None
    assert fn.get_stats("B", "y") is None


# ---- test_clear_by_ticker ------------------------------------------------

def test_clear_by_ticker():
    """clear(ticker='X') only clears that ticker's buffers."""
    fn.update("X", "rsi", 50.0)
    fn.update("X", "macd", 1.0)
    fn.update("Y", "rsi", 60.0)

    fn.clear(ticker="X")

    assert fn.get_stats("X", "rsi") is None
    assert fn.get_stats("X", "macd") is None
    # Y should be untouched
    assert fn.get_stats("Y", "rsi") is not None
    assert fn.get_stats("Y", "rsi")["count"] == 1


# ---- test_window_caps_at_max ---------------------------------------------

def test_window_caps_at_max():
    """Adding 200 values should cap buffer at _DEFAULT_WINDOW (100)."""
    for i in range(200):
        fn.update("BTC-USD", "rsi_14", float(i))
    stats = fn.get_stats("BTC-USD", "rsi_14")
    assert stats["count"] == 100  # capped at _DEFAULT_WINDOW


# ---- edge cases ----------------------------------------------------------

def test_normalize_nan_input_returns_zero():
    """normalize() with NaN input returns 0.0."""
    result = fn.normalize("A", "x", float("nan"))
    assert result == 0.0


def test_normalize_all_identical_returns_zero():
    """When all values identical (std~0), normalize returns 0.0."""
    for _ in range(25):
        fn.update("A", "x", 42.0)
    result = fn.normalize("A", "x", 42.0)
    assert result == 0.0


def test_clear_specific_pair():
    """clear(ticker, indicator) removes only that specific pair."""
    fn.update("X", "rsi", 1.0)
    fn.update("X", "macd", 2.0)
    fn.clear(ticker="X", indicator="rsi")
    assert fn.get_stats("X", "rsi") is None
    assert fn.get_stats("X", "macd") is not None
