"""Shared pytest configuration and fixtures."""

import json

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that require live GPU / network (deselect with '-k not integration')"
    )


# ---------------------------------------------------------------------------
# Indicator dictionary helpers
# ---------------------------------------------------------------------------

def make_indicators(**overrides):
    """Return a baseline indicator dict suitable for most signal tests.

    Override any key via keyword args::

        ind = make_indicators(rsi=25.0, close=100_000.0)
    """
    base = {
        "close": 69_000.0,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_hist_prev": 0.0,
        "ema9": 69_000.0,
        "ema21": 69_000.0,
        "bb_upper": 70_000.0,
        "bb_lower": 68_000.0,
        "bb_mid": 69_000.0,
        "price_vs_bb": "inside",
        "atr": 1_500.0,
        "atr_pct": 2.2,
        "rsi_p20": 35.0,
        "rsi_p80": 65.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# OHLCV DataFrame builders
# ---------------------------------------------------------------------------

def make_candles(prices, volume=100.0):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [volume] * n,
        "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
    })


def make_ohlcv_df(n=250, close_base=100.0, trend=0.0, volatility=1.0, seed=42):
    """Generate synthetic OHLCV data with configurable trend and volatility.

    Useful for testing signals that need many bars (e.g., 200-SMA, Ichimoku).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * volatility
    close = close_base + np.cumsum(noise) + np.arange(n) * trend
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n) * volatility)
    low = close - np.abs(rng.standard_normal(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + rng.standard_normal(n) * 0.3
    volume = rng.integers(100, 10_000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Minimal config dict with telegram section (used by many modules)."""
    return {
        "telegram": {"token": "fake-token", "chat_id": "123456"},
    }


@pytest.fixture
def config_file(tmp_path, sample_config):
    """Write sample_config to a temp config.json and return its path."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(sample_config), encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# Temporary data directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory (use with monkeypatch to override DATA_DIR)."""
    return tmp_path
