"""Tests for linear factor model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.linear_factor import LinearFactorModel


def _make_training_data(n=200, seed=42):
    """Generate synthetic signal + return data with known relationships."""
    np.random.seed(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")

    # Create signals with known correlation to returns
    signal_a = np.random.randn(n)  # positively correlated
    signal_b = np.random.randn(n)  # negatively correlated
    signal_c = np.random.randn(n)  # noise (uncorrelated)

    # Returns = 0.003 * signal_a - 0.002 * signal_b + noise
    returns = 0.003 * signal_a - 0.002 * signal_b + np.random.randn(n) * 0.001

    signals_df = pd.DataFrame({
        "signal_a": signal_a,
        "signal_b": signal_b,
        "signal_c": signal_c,
    }, index=idx)

    returns_s = pd.Series(returns, index=idx, name="returns")
    return signals_df, returns_s


class TestLinearFactorFit:
    def test_trains_successfully(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=1.0)
        assert model.fit(signals, returns) is True
        assert model.n_samples == 200
        assert model.r_squared > 0.5  # should explain most variance

    def test_signal_a_positive_weight(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=1.0)
        model.fit(signals, returns)
        assert model.weights["signal_a"] > 0  # positively correlated

    def test_signal_b_negative_weight(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=1.0)
        model.fit(signals, returns)
        assert model.weights["signal_b"] < 0  # negatively correlated

    def test_signal_c_near_zero_weight(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=1.0)
        model.fit(signals, returns)
        # Noise signal should have much smaller weight
        assert abs(model.weights["signal_c"]) < abs(model.weights["signal_a"])

    def test_insufficient_data_returns_false(self):
        signals, returns = _make_training_data(n=10)
        model = LinearFactorModel()
        assert model.fit(signals, returns, min_samples=30) is False

    def test_zero_variance_columns_dropped(self):
        signals, returns = _make_training_data()
        signals["constant"] = 1.0  # zero variance
        model = LinearFactorModel()
        model.fit(signals, returns)
        assert "constant" not in model.weights


class TestLinearFactorPredict:
    def test_positive_signal_a_gives_positive_score(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=1.0)
        model.fit(signals, returns)
        score = model.predict({"signal_a": 2.0, "signal_b": 0.0, "signal_c": 0.0})
        assert score > 0

    def test_positive_signal_b_gives_negative_score(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=1.0)
        model.fit(signals, returns)
        score = model.predict({"signal_a": 0.0, "signal_b": 2.0, "signal_c": 0.0})
        assert score < 0

    def test_empty_model_returns_zero(self):
        model = LinearFactorModel()
        assert model.predict({"signal_a": 1.0}) == 0.0

    def test_missing_signals_treated_as_zero(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel()
        model.fit(signals, returns)
        # Should not crash with missing signals
        score = model.predict({"signal_a": 1.0})
        assert isinstance(score, float)


class TestScoreToAction:
    def test_positive_score_is_buy(self):
        model = LinearFactorModel()
        action, conf = model.score_to_action(0.005, threshold=0.001)
        assert action == "BUY"
        assert conf > 0

    def test_negative_score_is_sell(self):
        model = LinearFactorModel()
        action, conf = model.score_to_action(-0.005, threshold=0.001)
        assert action == "SELL"

    def test_small_score_is_hold(self):
        model = LinearFactorModel()
        action, conf = model.score_to_action(0.0005, threshold=0.001)
        assert action == "HOLD"
        assert conf == 0.0

    def test_confidence_capped(self):
        model = LinearFactorModel()
        _, conf = model.score_to_action(1.0, threshold=0.001)
        assert conf <= 0.8


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        signals, returns = _make_training_data()
        model = LinearFactorModel(alpha=2.0)
        model.fit(signals, returns)
        path = tmp_path / "test_model.json"
        model.save(path)

        model2 = LinearFactorModel()
        assert model2.load(path) is True
        assert model2.weights == model.weights
        assert model2.intercept == pytest.approx(model.intercept)
        assert model2.alpha == 2.0
        assert model2.r_squared == pytest.approx(model.r_squared)

    def test_load_nonexistent_returns_false(self, tmp_path):
        model = LinearFactorModel()
        assert model.load(tmp_path / "nonexistent.json") is False


class TestFeatureImportance:
    def test_sorted_by_absolute_weight(self):
        signals, returns = _make_training_data()
        model = LinearFactorModel()
        model.fit(signals, returns)
        importance = model.feature_importance()
        # Should be sorted descending by |weight|
        weights = [abs(w) for _, w in importance]
        assert weights == sorted(weights, reverse=True)
