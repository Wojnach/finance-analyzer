"""Tests for TinyLoRA training scaffolding."""

import json
from datetime import datetime, timedelta, timezone

from portfolio.tinylora_trainer import (
    collect_training_pairs,
    is_training_allowed,
    prepare_training_config,
)

CET = timezone(timedelta(hours=1))


def _make_signal_entry(ticker, price, consensus, signals, change_pct, horizon="1d"):
    """Build a signal_log JSONL entry."""
    entry = {
        "ts": "2026-03-30T12:00:00Z",
        "tickers": {
            ticker: {
                "price_usd": price,
                "consensus": consensus,
                "signals": signals,
            }
        },
        "outcomes": {
            ticker: {
                horizon: {"change_pct": change_pct},
            }
        },
    }
    return json.dumps(entry)


class TestCollectTrainingPairs:

    def test_extracts_pairs(self, tmp_path):
        """Basic extraction: correct pair count and context format."""
        log = tmp_path / "signal_log.jsonl"
        log.write_text(
            _make_signal_entry("XAG-USD", 30.0, "BUY", {"rsi": "BUY"}, 1.5) + "\n"
            + _make_signal_entry("XAG-USD", 31.0, "SELL", {"macd": "SELL"}, -0.8) + "\n",
            encoding="utf-8",
        )
        pairs = collect_training_pairs(log_path=log, ticker="XAG-USD", horizon="1d")
        assert len(pairs) == 2
        # First pair context should mention ticker and price
        ctx, reward = pairs[0]
        assert "XAG-USD" in ctx
        assert "$30.00" in ctx
        assert "rsi=BUY" in ctx

    def test_positive_reward_for_correct(self, tmp_path):
        """BUY consensus + positive change = reward +1."""
        log = tmp_path / "signal_log.jsonl"
        log.write_text(
            _make_signal_entry("XAG-USD", 30.0, "BUY", {"rsi": "BUY"}, 2.0) + "\n",
            encoding="utf-8",
        )
        pairs = collect_training_pairs(log_path=log, ticker="XAG-USD", horizon="1d")
        assert len(pairs) == 1
        assert pairs[0][1] == 1

    def test_negative_reward_for_wrong(self, tmp_path):
        """BUY consensus + negative change = reward -1."""
        log = tmp_path / "signal_log.jsonl"
        log.write_text(
            _make_signal_entry("XAG-USD", 30.0, "BUY", {"rsi": "BUY"}, -2.0) + "\n",
            encoding="utf-8",
        )
        pairs = collect_training_pairs(log_path=log, ticker="XAG-USD", horizon="1d")
        assert len(pairs) == 1
        assert pairs[0][1] == -1

    def test_neutral_reward_for_hold(self, tmp_path):
        """HOLD consensus = reward 0."""
        log = tmp_path / "signal_log.jsonl"
        log.write_text(
            _make_signal_entry("XAG-USD", 30.0, "HOLD", {"rsi": "HOLD"}, 1.0) + "\n",
            encoding="utf-8",
        )
        pairs = collect_training_pairs(log_path=log, ticker="XAG-USD", horizon="1d")
        assert len(pairs) == 1
        assert pairs[0][1] == 0

    def test_neutral_for_tiny_change(self, tmp_path):
        """Change below threshold (<0.05%) = reward 0."""
        log = tmp_path / "signal_log.jsonl"
        log.write_text(
            _make_signal_entry("XAG-USD", 30.0, "BUY", {"rsi": "BUY"}, 0.01) + "\n",
            encoding="utf-8",
        )
        pairs = collect_training_pairs(log_path=log, ticker="XAG-USD", horizon="1d")
        assert len(pairs) == 1
        assert pairs[0][1] == 0

    def test_missing_log_returns_empty(self, tmp_path):
        """Non-existent log file returns empty list."""
        pairs = collect_training_pairs(log_path=tmp_path / "nope.jsonl")
        assert pairs == []


class TestMarketHoursGuard:

    def test_refuses_during_weekday_market_hours(self):
        """14:00 CET on a weekday is market hours — training not allowed."""
        # Wednesday 14:00 CET
        dt = datetime(2026, 4, 1, 14, 0, tzinfo=CET)
        assert is_training_allowed(dt) is False

    def test_allows_after_hours(self):
        """23:00 CET on a weekday — training allowed."""
        # Wednesday 23:00 CET
        dt = datetime(2026, 4, 1, 23, 0, tzinfo=CET)
        assert is_training_allowed(dt) is True

    def test_allows_weekends(self):
        """Saturday — training always allowed."""
        # Saturday 14:00 CET
        dt = datetime(2026, 4, 4, 14, 0, tzinfo=CET)
        assert is_training_allowed(dt) is True

    def test_allows_early_morning(self):
        """06:00 CET on a weekday — before market hours, training allowed."""
        dt = datetime(2026, 4, 1, 6, 0, tzinfo=CET)
        assert is_training_allowed(dt) is True

    def test_boundary_8am_is_market_hours(self):
        """08:00 CET is the start of market hours — training NOT allowed."""
        dt = datetime(2026, 4, 1, 8, 0, tzinfo=CET)
        assert is_training_allowed(dt) is False

    def test_boundary_22pm_is_after_hours(self):
        """22:00 CET is the start of after-hours — training allowed."""
        dt = datetime(2026, 4, 1, 22, 0, tzinfo=CET)
        assert is_training_allowed(dt) is True


class TestPrepareTrainingConfig:

    def test_returns_valid_config(self):
        """Config dict has expected keys and values."""
        config = prepare_training_config(
            model_path="/models/ministral-8b",
            adapter_dir="/adapters/xag",
            rank=1,
        )
        assert config["method"] == "GRPO"
        assert config["rank"] == 1
        assert config["estimated_params"] == 13
        assert config["model_path"] == "/models/ministral-8b"
        assert config["adapter_dir"] == "/adapters/xag"
        assert config["epochs"] == 3
        assert config["learning_rate"] == 1e-4

    def test_estimated_params_scales_with_rank(self):
        """estimated_params = rank * 13."""
        config = prepare_training_config(
            model_path="/m", adapter_dir="/a", rank=4,
        )
        assert config["estimated_params"] == 52
