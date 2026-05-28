"""Tests for gs_kalman_zscore_regime signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.gs_kalman_zscore_regime import (
    MIN_ROWS,
    _kalman_filter_1d,
    compute_gs_kalman_zscore_regime_signal,
)


def _make_df(n=100, seed=42):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_gs_ratio(n=100, mean=80.0, std=5.0, seed=42):
    """Create a synthetic gold/silver ratio series."""
    np.random.seed(seed)
    return mean + np.cumsum(np.random.randn(n) * 0.3)


class TestKalmanFilter:
    """Test the Kalman filter implementation."""

    def test_output_length_matches_input(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        est, unc = _kalman_filter_1d(obs, Q=0.01, R=1.0, init_P=1.0)
        assert len(est) == len(obs)
        assert len(unc) == len(obs)

    def test_first_estimate_equals_first_observation(self):
        obs = np.array([42.0, 43.0, 44.0])
        est, _ = _kalman_filter_1d(obs, Q=0.01, R=1.0, init_P=1.0)
        assert est[0] == 42.0

    def test_tracks_linear_trend(self):
        obs = np.linspace(70, 90, 200)
        est, _ = _kalman_filter_1d(obs, Q=0.01, R=0.5, init_P=1.0)
        assert abs(est[-1] - obs[-1]) < 2.0

    def test_smooths_noisy_signal(self):
        np.random.seed(42)
        true_signal = np.full(100, 80.0)
        noisy = true_signal + np.random.randn(100) * 3.0
        est, _ = _kalman_filter_1d(noisy, Q=0.001, R=1.0, init_P=1.0)
        noise_var = np.var(noisy - true_signal)
        est_var = np.var(est[-50:] - true_signal[-50:])
        assert est_var < noise_var

    def test_uncertainties_decrease(self):
        obs = np.full(50, 80.0)
        _, unc = _kalman_filter_1d(obs, Q=0.001, R=1.0, init_P=10.0)
        assert unc[-1] < unc[0]


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": _make_gs_ratio()}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "gs_ratio_series": _make_gs_ratio()}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": _make_gs_ratio()}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ctx = {"ticker": "XAG-USD"}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": _make_gs_ratio(n=10)}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_non_metals_ticker_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "gs_ratio_series": _make_gs_ratio()}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_gs_kalman_zscore_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_none_df_returns_hold(self):
        result = compute_gs_kalman_zscore_regime_signal(None, context={"ticker": "XAG-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestSignalBehavior:
    """Test signal behavior under different market conditions."""

    def test_high_ratio_signals_buy_for_silver(self):
        df = _make_df(n=100)
        ratio = np.full(100, 80.0)
        ratio[-10:] = 95.0
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": ratio}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        if result["action"] != "HOLD":
            assert result["action"] == "BUY"

    def test_high_ratio_signals_sell_for_gold(self):
        df = _make_df(n=100)
        ratio = np.full(100, 80.0)
        ratio[-10:] = 95.0
        ctx = {"ticker": "XAU-USD", "gs_ratio_series": ratio}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        if result["action"] != "HOLD":
            assert result["action"] == "SELL"

    def test_confidence_capped_at_07(self):
        df = _make_df(n=200)
        ratio = np.full(200, 80.0)
        ratio[-20:] = 100.0
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": ratio}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["confidence"] <= 0.7

    def test_stable_ratio_returns_hold(self):
        df = _make_df(n=100)
        ratio = np.full(100, 80.0) + np.random.RandomState(42).randn(100) * 0.1
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": ratio}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_indicators_contain_expected_keys(self):
        df = _make_df(n=100)
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": _make_gs_ratio()}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        indicators = result["indicators"]
        for key in ["gs_ratio", "kalman_estimate", "kalman_zscore", "adx", "regime_trending"]:
            assert key in indicators

    def test_sub_signals_contain_expected_keys(self):
        df = _make_df(n=100)
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": _make_gs_ratio()}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        sub = result["sub_signals"]
        for key in ["kalman_zscore", "ratio_level", "kalman_trend", "regime_gate"]:
            assert key in sub

    def test_nan_in_ratio_handled(self):
        df = _make_df(n=100)
        ratio = _make_gs_ratio(n=100)
        ratio[30:35] = np.nan
        ctx = {"ticker": "XAG-USD", "gs_ratio_series": ratio}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_gold_close_context(self):
        n = 100
        np.random.seed(42)
        silver_close = 30 + np.cumsum(np.random.randn(n) * 0.2)
        gold_close = silver_close * 80 + np.random.randn(n) * 10
        df = pd.DataFrame({
            "open": silver_close + 0.1,
            "high": silver_close + 0.3,
            "low": silver_close - 0.3,
            "close": silver_close,
            "volume": np.full(n, 5000.0),
        })
        ctx = {"ticker": "XAG-USD", "gold_close": gold_close}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_silver_close_context(self):
        n = 100
        np.random.seed(42)
        gold_close = 2400 + np.cumsum(np.random.randn(n) * 5)
        silver_close = gold_close / 80 + np.random.randn(n) * 0.2
        df = pd.DataFrame({
            "open": gold_close + 1,
            "high": gold_close + 3,
            "low": gold_close - 3,
            "close": gold_close,
            "volume": np.full(n, 5000.0),
        })
        ctx = {"ticker": "XAU-USD", "silver_close": silver_close}
        result = compute_gs_kalman_zscore_regime_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")
