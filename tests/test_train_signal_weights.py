"""Tests for signal weight training pipeline."""
from __future__ import annotations

import json

import numpy as np


def _make_signal_log(tmp_path, n=200):
    """Generate a synthetic signal_log.jsonl with outcomes."""
    import pandas as pd
    log_path = tmp_path / "signal_log.jsonl"
    np.random.seed(42)

    lines = []
    for i in range(n):
        ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)
        # Generate synthetic signal votes
        signals = {}
        for sig in ["rsi", "macd", "ema", "trend", "momentum"]:
            r = np.random.random()
            if r < 0.3:
                signals[sig] = "BUY"
            elif r < 0.6:
                signals[sig] = "SELL"
            else:
                signals[sig] = "HOLD"

        # Generate outcome correlated with RSI signal
        rsi_val = 1.0 if signals["rsi"] == "BUY" else (-1.0 if signals["rsi"] == "SELL" else 0.0)
        change_pct = rsi_val * 0.5 + np.random.randn() * 0.3

        entry = {
            "ts": ts.isoformat(),
            "tickers": {
                "XAG-USD": {
                    "price_usd": 30.0 + i * 0.01,
                    "consensus": "HOLD",
                    "signals": signals,
                }
            },
            "outcomes": {
                "XAG-USD": {
                    "1d": {"change_pct": change_pct},
                }
            },
        }
        lines.append(json.dumps(entry))

    log_path.write_text("\n".join(lines))
    return log_path


class TestLoadSignalHistory:
    def test_extracts_signals_and_returns(self, tmp_path):
        from portfolio.train_signal_weights import _load_signal_history
        log_path = _make_signal_log(tmp_path, n=100)
        result = _load_signal_history(log_path, horizon="1d")
        assert result is not None
        signals_df, returns = result
        assert len(signals_df) == 100
        assert "rsi" in signals_df.columns
        assert len(returns) == 100

    def test_insufficient_data_returns_none(self, tmp_path):
        from portfolio.train_signal_weights import _load_signal_history
        log_path = _make_signal_log(tmp_path, n=10)
        result = _load_signal_history(log_path, horizon="1d", min_entries=50)
        assert result is None

    def test_skips_disabled_signals(self, tmp_path):
        from portfolio.train_signal_weights import _load_signal_history
        log_path = _make_signal_log(tmp_path, n=100)
        result = _load_signal_history(log_path, horizon="1d")
        signals_df, _ = result
        assert "ml" not in signals_df.columns
        assert "funding" not in signals_df.columns


class TestTrainWeights:
    def test_trains_and_returns_results(self, tmp_path):
        from portfolio.train_signal_weights import train_weights
        log_path = _make_signal_log(tmp_path, n=200)
        result = train_weights(horizon="1d", log_path=log_path)
        assert result != {}
        assert "model" in result
        assert result["model"]["r_squared"] > 0
        assert result["model"]["n_features"] > 0
        assert len(result["model"]["top_features"]) > 0

    def test_rsi_has_positive_weight(self, tmp_path):
        from portfolio.train_signal_weights import train_weights
        log_path = _make_signal_log(tmp_path, n=200)
        result = train_weights(horizon="1d", log_path=log_path)
        # RSI was synthetically correlated with returns
        top_names = [name for name, _ in result["model"]["top_features"]]
        assert "rsi" in top_names
