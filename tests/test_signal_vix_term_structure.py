"""Tests for vix_term_structure signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.vix_term_structure import (
    compute_vix_term_structure_signal,
    _backwardation_flag,
    _contango_depth,
    _ratio_zscore,
    _ratio_slope_5d,
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


def _mock_vix_data(ratio=0.92, vix=19.0, vix3m=20.7, n=25):
    """Create mock VIX data for _cached."""
    ratios = [ratio + np.random.randn() * 0.02 for _ in range(n)]
    ratios[-1] = ratio
    return {
        "vix_current": vix,
        "vix3m_current": vix3m,
        "ratio_current": ratio,
        "ratio_series": ratios,
    }


class TestBackwardationFlag:
    def test_strong_backwardation(self):
        assert _backwardation_flag(1.06) == "SELL"

    def test_backwardation(self):
        assert _backwardation_flag(1.02) == "SELL"

    def test_normal_contango(self):
        assert _backwardation_flag(0.92) == "HOLD"

    def test_deep_contango(self):
        assert _backwardation_flag(0.83) == "BUY"

    def test_boundary_exactly_1(self):
        assert _backwardation_flag(1.0) == "SELL"

    def test_boundary_exactly_0_85(self):
        assert _backwardation_flag(0.85) == "HOLD"


class TestContangoDepth:
    def test_deep_contango_buy(self):
        assert _contango_depth(0.80) == "BUY"

    def test_moderate_contango_buy(self):
        assert _contango_depth(0.88) == "BUY"

    def test_normal_contango_hold(self):
        assert _contango_depth(0.93) == "HOLD"

    def test_backwardation_sell(self):
        assert _contango_depth(1.02) == "SELL"

    def test_mild_backwardation_sell(self):
        assert _contango_depth(1.06) == "SELL"


class TestRatioZscore:
    def test_elevated_zscore_sell(self):
        series = [0.92] * 20 + [1.05]
        z, action = _ratio_zscore(series)
        assert action == "SELL"
        assert z > 0.0

    def test_depressed_zscore_buy(self):
        series = [0.92] * 20 + [0.80]
        z, action = _ratio_zscore(series)
        assert action == "BUY"
        assert z < 0.0

    def test_flat_zscore_hold(self):
        series = [0.92] * 21
        z, action = _ratio_zscore(series)
        assert action == "HOLD"
        assert abs(z) < 0.01

    def test_insufficient_data(self):
        z, action = _ratio_zscore([0.92] * 5)
        assert action == "HOLD"
        assert z == 0.0


class TestRatioSlope:
    def test_rising_slope_sell(self):
        series = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.97]
        slope, action = _ratio_slope_5d(series)
        assert action == "SELL"
        assert slope > 0

    def test_falling_slope_buy(self):
        series = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.85]
        slope, action = _ratio_slope_5d(series)
        assert action == "BUY"
        assert slope < 0

    def test_flat_slope_hold(self):
        series = [0.92] * 7
        slope, action = _ratio_slope_5d(series)
        assert action == "HOLD"

    def test_insufficient_data(self):
        slope, action = _ratio_slope_5d([0.92] * 3)
        assert action == "HOLD"


class TestSignalInterface:
    @patch("portfolio.signals.vix_term_structure._cached")
    def test_returns_dict_with_required_keys(self, mock_cached):
        mock_cached.return_value = _mock_vix_data()
        result = compute_vix_term_structure_signal()
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_has_sub_signals(self, mock_cached):
        mock_cached.return_value = _mock_vix_data()
        result = compute_vix_term_structure_signal()
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "backwardation_flag" in result["sub_signals"]
        assert "contango_depth" in result["sub_signals"]
        assert "ratio_zscore" in result["sub_signals"]
        assert "ratio_slope_5d" in result["sub_signals"]

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_has_indicators(self, mock_cached):
        mock_cached.return_value = _mock_vix_data()
        result = compute_vix_term_structure_signal()
        assert "indicators" in result
        assert "vix" in result["indicators"]
        assert "vix3m" in result["indicators"]
        assert "ratio" in result["indicators"]
        assert "z_score" in result["indicators"]
        assert "in_backwardation" in result["indicators"]

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_data_unavailable_returns_hold(self, mock_cached):
        mock_cached.return_value = None
        result = compute_vix_term_structure_signal()
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_backwardation_returns_sell(self, mock_cached):
        data = _mock_vix_data(ratio=1.08, vix=25.0, vix3m=23.0)
        data["ratio_series"] = [0.92] * 20 + [1.08]
        mock_cached.return_value = data
        result = compute_vix_term_structure_signal()
        assert result["action"] == "SELL"
        assert result["confidence"] > 0.0

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_deep_contango_returns_buy(self, mock_cached):
        data = _mock_vix_data(ratio=0.80, vix=15.0, vix3m=18.75)
        data["ratio_series"] = [0.92] * 20 + [0.80]
        mock_cached.return_value = data
        result = compute_vix_term_structure_signal()
        assert result["action"] == "BUY"

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_confidence_capped_at_0_7(self, mock_cached):
        mock_cached.return_value = _mock_vix_data(ratio=1.10, vix=30.0, vix3m=27.0)
        result = compute_vix_term_structure_signal()
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_strong_backwardation_minimum_confidence(self, mock_cached):
        data = _mock_vix_data(ratio=1.08, vix=28.0, vix3m=26.0)
        data["ratio_series"] = [0.92] * 20 + [1.08]
        mock_cached.return_value = data
        result = compute_vix_term_structure_signal()
        assert result["confidence"] >= 0.6

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_with_context(self, mock_cached):
        mock_cached.return_value = _mock_vix_data()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_vix_term_structure_signal(context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_with_df_parameter(self, mock_cached):
        mock_cached.return_value = _mock_vix_data()
        df = _make_df()
        result = compute_vix_term_structure_signal(df=df)
        assert isinstance(result, dict)

    @patch("portfolio.signals.vix_term_structure._cached")
    def test_normal_contango_hold(self, mock_cached):
        data = _mock_vix_data(ratio=0.92, vix=18.0, vix3m=19.6)
        data["ratio_series"] = [0.92] * 25
        mock_cached.return_value = data
        result = compute_vix_term_structure_signal()
        assert result["action"] == "HOLD"
