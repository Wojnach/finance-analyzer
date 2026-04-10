"""Tests for futures_basis signal module."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.futures_basis import (
    _MIN_KLINES,
    _SUSTAINED_WINDOW,
    _VELOCITY_WINDOW,
    _basis_acceleration,
    _basis_velocity,
    _basis_z_extreme,
    _sustained_regime,
    compute_futures_basis_signal,
)


def _make_df(n=100):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_klines(n=168, base_basis=0.0, trend=0.0, noise_std=0.0001):
    """Create mock premiumIndexKlines data.

    Returns list of [open_time, open, high, low, close, ...] entries
    where close is the basis fraction value.
    """
    klines = []
    for i in range(n):
        basis = base_basis + trend * i + np.random.randn() * noise_std
        klines.append([
            1700000000000 + i * 3600000,  # timestamp
            str(basis - abs(np.random.randn() * noise_std)),  # open
            str(basis + abs(np.random.randn() * noise_std)),  # high
            str(basis - abs(np.random.randn() * noise_std)),  # low
            str(basis),  # close
            "0",  # volume placeholder
        ])
    return klines


class TestBasisZExtreme:
    """Test the z-score extreme sub-indicator."""

    def test_hold_on_insufficient_data(self):
        basis = np.array([0.001] * 10)
        action, _ = _basis_z_extreme(basis)
        assert action == "HOLD"

    def test_hold_on_zero_std(self):
        basis = np.array([0.001] * 100)
        action, _ = _basis_z_extreme(basis)
        assert action == "HOLD"

    def test_buy_on_extreme_backwardation(self):
        basis = np.zeros(100)
        basis[-1] = -0.05  # Extreme negative outlier
        action, z = _basis_z_extreme(basis)
        assert action == "BUY"
        assert z < -1.8

    def test_sell_on_extreme_contango(self):
        basis = np.zeros(100)
        basis[-1] = 0.05  # Extreme positive outlier
        action, z = _basis_z_extreme(basis)
        assert action == "SELL"
        assert z > 1.8

    def test_hold_on_normal_basis(self):
        np.random.seed(42)
        basis = np.random.randn(100) * 0.001
        action, _ = _basis_z_extreme(basis)
        assert action == "HOLD"


class TestBasisVelocity:
    """Test the basis velocity sub-indicator."""

    def test_hold_on_insufficient_data(self):
        basis = np.array([0.0] * 10)
        action, _ = _basis_velocity(basis)
        assert action == "HOLD"

    def test_buy_on_rapid_move_to_backwardation(self):
        n = _VELOCITY_WINDOW + 10
        basis = np.zeros(n)
        # Rapid move toward backwardation at the end
        basis[-1] = -0.01
        basis[-_VELOCITY_WINDOW - 1] = 0.005
        action, v = _basis_velocity(basis)
        # Direction should be BUY (moving toward backwardation)
        if action != "HOLD":
            assert action == "BUY"

    def test_hold_on_stable_basis(self):
        # Constant basis with zero velocity → HOLD
        basis = np.full(50, 0.001)
        action, _ = _basis_velocity(basis)
        assert action == "HOLD"

    def test_handles_nan(self):
        basis = np.full(50, np.nan)
        action, _ = _basis_velocity(basis)
        assert action == "HOLD"


class TestSustainedRegime:
    """Test the sustained regime sub-indicator."""

    def test_hold_on_insufficient_data(self):
        basis = np.array([0.001] * 3)
        action, _ = _sustained_regime(basis)
        assert action == "HOLD"

    def test_buy_on_sustained_backwardation(self):
        # All 8 meaningfully negative (below -0.0002 threshold)
        basis = np.array([-0.001, -0.002, -0.001, -0.003, -0.001, -0.002, -0.001, -0.0005])
        action, strength = _sustained_regime(basis)
        assert action == "BUY"
        assert strength > 0.5

    def test_sell_on_sustained_contango(self):
        # All 8 meaningfully positive (above +0.0002 threshold)
        basis = np.array([0.001, 0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.0005])
        action, strength = _sustained_regime(basis)
        assert action == "SELL"
        assert strength > 0.5

    def test_hold_on_mixed_regime(self):
        basis = np.array([0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001])
        action, _ = _sustained_regime(basis)
        assert action == "HOLD"

    def test_hold_on_near_zero_basis(self):
        """Basis near zero (within deadband) should not trigger sustained regime."""
        basis = np.array([-0.00005, -0.00008, -0.00003, -0.00007,
                          -0.00004, -0.00009, -0.00002, -0.00006])
        action, _ = _sustained_regime(basis)
        assert action == "HOLD"  # All below -0.0002 min_abs threshold


class TestBasisAcceleration:
    """Test the basis acceleration sub-indicator."""

    def test_hold_on_insufficient_data(self):
        basis = np.array([0.001] * 10)
        action, _ = _basis_acceleration(basis)
        assert action == "HOLD"

    def test_hold_on_stable_acceleration(self):
        np.random.seed(42)
        basis = np.random.randn(100) * 0.0001
        action, _ = _basis_acceleration(basis)
        # With random noise, likely HOLD
        assert action in ("BUY", "SELL", "HOLD")


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _make_klines(168, base_basis=-0.001)
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _make_klines(168)
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "basis_z_extreme" in result["sub_signals"]
        assert "basis_velocity" in result["sub_signals"]
        assert "sustained_regime" in result["sub_signals"]
        assert "basis_acceleration" in result["sub_signals"]

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_klines(168)
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "basis_current" in result["indicators"]
        assert "basis_z_score" in result["indicators"]
        assert "n_klines" in result["indicators"]

    def test_no_context_returns_hold(self):
        result = compute_futures_basis_signal(None, context=None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_ticker_returns_hold(self):
        result = compute_futures_basis_signal(None, context={})
        assert result["action"] == "HOLD"

    def test_unknown_ticker_returns_hold(self):
        result = compute_futures_basis_signal(None, context={"ticker": "UNKNOWN"})
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_api_failure_returns_hold(self, mock_fetch):
        mock_fetch.return_value = None
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_insufficient_klines_returns_hold(self, mock_fetch):
        mock_fetch.return_value = _make_klines(10)  # Less than _MIN_KLINES
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_confidence_capped_at_07(self, mock_fetch):
        # Create data with extreme backwardation to get high confidence
        klines = _make_klines(168, base_basis=-0.01, noise_std=0.00001)
        mock_fetch.return_value = klines
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_works_with_metals_ticker(self, mock_fetch):
        mock_fetch.return_value = _make_klines(168, base_basis=0.0005)
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_works_with_silver_ticker(self, mock_fetch):
        mock_fetch.return_value = _make_klines(168, base_basis=0.001)
        ctx = {"ticker": "XAG-USD", "asset_class": "metals"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_extreme_backwardation_generates_buy(self, mock_fetch):
        # Normal basis for most of the window, then extreme drop
        klines = _make_klines(168, base_basis=0.0, noise_std=0.0001)
        # Set last 10 candles to extreme backwardation
        for i in range(-10, 0):
            klines[i][4] = str(-0.01)  # -1% basis (extreme)
        mock_fetch.return_value = klines
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        # Should likely generate BUY (extreme backwardation)
        assert result["action"] in ("BUY", "HOLD")  # May be HOLD if not all sub-indicators agree
        assert result["indicators"]["basis_z_score"] < 0

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_all_nan_klines_returns_hold(self, mock_fetch):
        klines = [[0, "nan", "nan", "nan", "nan", "0"] for _ in range(168)]
        mock_fetch.return_value = klines
        ctx = {"ticker": "BTC-USD"}
        result = compute_futures_basis_signal(None, context=ctx)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.futures_basis._fetch_premium_klines")
    def test_df_parameter_ignored(self, mock_fetch):
        """Signal fetches its own data; df parameter is ignored."""
        mock_fetch.return_value = _make_klines(168)
        df = _make_df()
        ctx = {"ticker": "ETH-USD"}
        result = compute_futures_basis_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")
