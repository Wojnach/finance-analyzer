"""
Integration tests for TABaseStrategy — runs inside the Freqtrade Podman container.

Execute with: ./scripts/ft-test.sh tests/integration/ -v
"""

import importlib
import sys

import pytest
import numpy as np
import pandas as pd


def _have_talib():
    try:
        import talib  # noqa: F401

        return True
    except ImportError:
        return False


def _have_freqtrade():
    try:
        import freqtrade  # noqa: F401

        return True
    except ImportError:
        return False


requires_talib = pytest.mark.skipif(
    not _have_talib(), reason="TA-Lib not available (run inside container)"
)
requires_freqtrade = pytest.mark.skipif(
    not _have_freqtrade(), reason="Freqtrade not available (run inside container)"
)


def _make_ohlcv(rows=200):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "open": close + np.random.randn(rows) * 0.1,
            "high": close + abs(np.random.randn(rows) * 0.3),
            "low": close - abs(np.random.randn(rows) * 0.3),
            "close": close,
            "volume": np.random.randint(100, 10000, rows).astype(float),
        }
    )


@requires_talib
def test_talib_available():
    import talib

    result = talib.RSI(np.random.randn(50), timeperiod=14)
    assert len(result) == 50


@requires_freqtrade
@requires_talib
def test_strategy_loads():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    mod = importlib.import_module("ta_base_strategy")
    assert hasattr(mod, "TABaseStrategy")
    strategy = mod.TABaseStrategy
    assert strategy.INTERFACE_VERSION == 3
    assert strategy.timeframe == "5m"


@requires_freqtrade
@requires_talib
def test_populate_indicators():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = {"strategy": "TABaseStrategy"}
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    result = s.populate_indicators(df, {"pair": "BTC/USDT"})

    for col in [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "ema_fast",
        "ema_slow",
        "volume_sma",
    ]:
        assert col in result.columns, f"Missing column: {col}"

    # RSI should be between 0 and 100 (after warmup)
    valid_rsi = result["rsi"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


@requires_freqtrade
@requires_talib
def test_populate_entry_trend():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = {"strategy": "TABaseStrategy"}
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    df = s.populate_indicators(df, {"pair": "BTC/USDT"})
    result = s.populate_entry_trend(df, {"pair": "BTC/USDT"})

    assert "enter_long" in result.columns


@requires_freqtrade
@requires_talib
def test_populate_exit_trend():
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = {"strategy": "TABaseStrategy"}
    s = TABaseStrategy(config)
    df = _make_ohlcv()
    df = s.populate_indicators(df, {"pair": "BTC/USDT"})
    df = s.populate_entry_trend(df, {"pair": "BTC/USDT"})
    result = s.populate_exit_trend(df, {"pair": "BTC/USDT"})

    assert "exit_long" in result.columns


@requires_freqtrade
@requires_talib
def test_confidence_scoring():
    """Verify that confidence = signal_count * base_confidence matches signals.py logic."""
    sys.path.insert(0, "/freqtrade/user_data/strategies")
    from ta_base_strategy import TABaseStrategy

    config = {"strategy": "TABaseStrategy"}
    s = TABaseStrategy(config)
    bc = s.base_confidence.value

    # With default base_confidence=0.25, 2 full signals = 0.5 >= min_confidence
    assert 2 * bc >= s.min_confidence.value
    # Single signal alone should NOT trigger (0.25 < 0.5)
    assert 1 * bc < s.min_confidence.value
