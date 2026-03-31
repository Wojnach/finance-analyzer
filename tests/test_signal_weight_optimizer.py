"""Tests for walk-forward signal weight optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signal_weight_optimizer import (
    WalkForwardResult,
    walk_forward_optimize,
    save_results,
    load_results,
)


def _make_data(n=2000, seed=42):
    """Generate synthetic data with known signal relationships."""
    np.random.seed(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")

    strong_signal = np.random.randn(n)
    weak_signal = np.random.randn(n)
    noise_signal = np.random.randn(n)

    returns = 0.003 * strong_signal + 0.0005 * weak_signal + np.random.randn(n) * 0.002

    signals = pd.DataFrame({
        "strong": strong_signal,
        "weak": weak_signal,
        "noise": noise_signal,
    }, index=idx)

    return signals, pd.Series(returns, index=idx)


class TestWalkForwardOptimize:
    def test_runs_multiple_windows(self):
        signals, returns = _make_data(n=2000)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100, step_size=100,
        )
        assert result.n_windows > 5

    def test_positive_r_squared(self):
        signals, returns = _make_data(n=2000)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100, step_size=100,
        )
        assert result.avg_r_squared > 0

    def test_strong_signal_ranked_higher(self):
        signals, returns = _make_data(n=2000)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100, step_size=100,
        )
        # Strong signal should rank higher than noise
        rankings_dict = dict(result.signal_rankings)
        assert rankings_dict.get("strong", 0) > rankings_dict.get("noise", 0)

    def test_weight_stability_between_0_and_1(self):
        signals, returns = _make_data(n=2000)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100, step_size=100,
        )
        for _, stability in result.weight_stability.items():
            assert 0.0 <= stability <= 1.0

    def test_insufficient_data_returns_empty(self):
        signals, returns = _make_data(n=50)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100,
        )
        assert result.n_windows == 0

    def test_recommended_weights_populated(self):
        signals, returns = _make_data(n=2000)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100, step_size=100,
        )
        assert len(result.recommended_weights) == 3
        assert "strong" in result.recommended_weights


class TestSaveLoadResults:
    def test_roundtrip(self, tmp_path):
        signals, returns = _make_data(n=2000)
        result = walk_forward_optimize(
            signals, returns,
            train_window=500, test_window=100, step_size=100,
        )
        path = tmp_path / "wf_test.json"
        save_results(result, path)
        loaded = load_results(path)
        assert loaded is not None
        assert loaded.n_windows == result.n_windows
        assert loaded.avg_r_squared == pytest.approx(result.avg_r_squared)

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_results(tmp_path / "nonexistent.json") is None


class TestWalkForwardResultToDict:
    def test_serializable(self):
        result = WalkForwardResult(
            n_windows=5,
            avg_r_squared=0.6,
            avg_oos_corr=0.3,
            weight_stability={"a": 0.8},
            recommended_weights={"a": 0.005},
            signal_rankings=[("a", 0.004)],
        )
        d = result.to_dict()
        assert d["n_windows"] == 5
        assert d["avg_r_squared"] == 0.6
