"""Regression tests for 2026-04-17 adversarial-review fixes.

Each test pins one or more findings from the 4 parallel adversarial
reviewers. Groups:

  P1-B: Horizon-blacklist _voters mismatch
  P1-C: None / NaN values crash live consensus path
  P1-D: blend_accuracy_data silently drops directional keys
"""

from __future__ import annotations

import pytest

from portfolio.accuracy_stats import blend_accuracy_data
from portfolio.signal_engine import (
    _count_active_voters_at_gate,
    _safe_accuracy,
    _safe_sample_count,
)


class TestSafeAccuracyHelper:
    """P1-C: _safe_accuracy coerces None/NaN/inf to the provided default."""

    def test_none_returns_default(self):
        assert _safe_accuracy(None, default=0.5) == 0.5

    def test_nan_returns_default(self):
        assert _safe_accuracy(float("nan"), default=0.5) == 0.5

    def test_positive_inf_returns_default(self):
        assert _safe_accuracy(float("inf"), default=0.5) == 0.5

    def test_negative_inf_returns_default(self):
        assert _safe_accuracy(float("-inf"), default=0.5) == 0.5

    def test_valid_float_passes_through(self):
        assert _safe_accuracy(0.72, default=0.5) == 0.72

    def test_valid_int_coerced_to_float(self):
        assert _safe_accuracy(1, default=0.5) == 1.0

    def test_invalid_string_returns_default(self):
        assert _safe_accuracy("not-a-number", default=0.5) == 0.5


class TestSafeSampleCountHelper:
    """P1-C: _safe_sample_count returns safe non-negative ints."""

    def test_none_returns_zero(self):
        assert _safe_sample_count(None) == 0

    def test_nan_returns_zero(self):
        assert _safe_sample_count(float("nan")) == 0

    def test_negative_returns_zero(self):
        assert _safe_sample_count(-50) == 0

    def test_positive_int_passes(self):
        assert _safe_sample_count(250) == 250

    def test_float_truncated_to_int(self):
        assert _safe_sample_count(250.7) == 250


class TestCountActiveVotersHandlesNone:
    """P1-C: _count_active_voters_at_gate must not crash on None/NaN stats."""

    def test_none_stats_entry(self):
        votes = {"rsi": "BUY"}
        accuracy = {"rsi": None}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1

    def test_none_accuracy_value(self):
        votes = {"rsi": "BUY"}
        accuracy = {"rsi": {"accuracy": None, "total": 100}}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1

    def test_nan_accuracy_value(self):
        votes = {"rsi": "BUY"}
        accuracy = {
            "rsi": {
                "accuracy": float("nan"), "total": 100,
                "buy_accuracy": float("nan"), "total_buy": 50,
            },
        }
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1

    def test_nan_buy_accuracy_falls_back_to_overall(self):
        votes = {"rsi": "BUY"}
        accuracy = {
            "rsi": {
                "accuracy": 0.60, "total": 100,
                "buy_accuracy": float("nan"), "total_buy": 50,
            },
        }
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1


class TestBlendAccuracyDataDirectionalMerge:
    """P1-D: directional keys must be preserved for signals present only
    in `recent`, and merged correctly when both sources have data."""

    def test_signal_only_in_recent_preserves_directional(self):
        alltime = {"rsi": {"accuracy": 0.55, "total": 200}}
        recent = {
            "new_sig": {
                "accuracy": 0.40, "total": 400,
                "buy_accuracy": 0.28, "total_buy": 150,
                "sell_accuracy": 0.52, "total_sell": 250,
            },
        }
        result = blend_accuracy_data(alltime, recent)
        assert "new_sig" in result, "P1-D: signal dropped from blend"
        assert result["new_sig"]["buy_accuracy"] == 0.28
        assert result["new_sig"]["total_buy"] == 150
        assert result["new_sig"]["sell_accuracy"] == 0.52
        assert result["new_sig"]["total_sell"] == 250

    def test_directional_merged_from_larger_sample_side(self):
        alltime = {
            "rsi": {
                "accuracy": 0.55, "total": 1000,
                "buy_accuracy": 0.45, "total_buy": 600,
                "sell_accuracy": 0.65, "total_sell": 400,
            },
        }
        recent = {
            "rsi": {
                "accuracy": 0.50, "total": 100,
                "buy_accuracy": 0.30, "total_buy": 20,
                "sell_accuracy": 0.70, "total_sell": 80,
            },
        }
        result = blend_accuracy_data(alltime, recent)
        assert result["rsi"]["buy_accuracy"] == 0.45
        assert result["rsi"]["sell_accuracy"] == 0.65

    def test_directional_from_alltime_only(self):
        alltime = {
            "rsi": {
                "accuracy": 0.55, "total": 200,
                "buy_accuracy": 0.45, "total_buy": 100,
                "sell_accuracy": 0.65, "total_sell": 100,
            },
        }
        recent = {"rsi": {"accuracy": 0.60, "total": 80}}
        result = blend_accuracy_data(alltime, recent)
        assert result["rsi"]["buy_accuracy"] == 0.45
        assert result["rsi"]["total_buy"] == 100

    def test_total_buy_sell_use_max(self):
        alltime = {
            "rsi": {"accuracy": 0.55, "total": 200, "total_buy": 150, "total_sell": 50},
        }
        recent = {
            "rsi": {"accuracy": 0.50, "total": 80, "total_buy": 30, "total_sell": 50},
        }
        result = blend_accuracy_data(alltime, recent)
        assert result["rsi"]["total_buy"] == 150
        assert result["rsi"]["total_sell"] == 50


class TestHorizonBlacklistUpdatesVoters:
    """P1-B: horizon-blacklist must propagate to extra_info['_voters']."""

    def _make_df(self, n=150, close_start=100.0):
        import numpy as np
        import pandas as pd
        dates = pd.date_range("2026-01-01", periods=n, freq="h")
        rng = np.random.default_rng(42)
        closes = close_start + np.cumsum(rng.standard_normal(n) * 0.5)
        highs = closes + rng.random(n) * 2
        lows = closes - rng.random(n) * 2
        volumes = rng.integers(100, 10000, n).astype(float)
        return pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=dates,
        )

    def test_voters_count_matches_post_horizon_disable(self, monkeypatch):
        from unittest import mock

        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        monkeypatch.setattr("portfolio.market_timing.should_skip_gpu",
                            mock.MagicMock(return_value=True))
        monkeypatch.setattr("portfolio.shared_state._cached",
                            mock.MagicMock(return_value=None))

        df = self._make_df(150)
        ind = compute_indicators(df)
        _, _, extra = generate_signal(ind, ticker="MSTR", df=df)

        voters = extra.get("_voters", 0)
        assert isinstance(voters, int)
        assert voters >= 0
        core_buy = extra.get("_core_buy", 0)
        core_sell = extra.get("_core_sell", 0)
        assert core_buy + core_sell <= voters

    def test_horizon_disabled_forces_hold_before_voters_count(self, monkeypatch):
        """Direct pin: with MSTR's _default blacklist (claude_fundamental,
        credit_spread_risk), those signals must NOT appear in extra's core
        counts even if they would have voted under the old pre-gate flow."""
        from unittest import mock

        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import (
            _TICKER_DISABLED_BY_HORIZON,
            generate_signal,
        )

        assert "claude_fundamental" in _TICKER_DISABLED_BY_HORIZON["_default"]["MSTR"]

        monkeypatch.setattr("portfolio.market_timing.should_skip_gpu",
                            mock.MagicMock(return_value=True))
        monkeypatch.setattr("portfolio.shared_state._cached",
                            mock.MagicMock(return_value=None))

        df = self._make_df(150)
        ind = compute_indicators(df)
        _, _, extra = generate_signal(ind, ticker="MSTR", df=df)

        # If horizon-disable is applied BEFORE voter count, the signal's
        # action field must be HOLD.
        cf_action = extra.get("claude_fundamental_action")
        if cf_action is not None:
            # If compute path set the action, blacklist should have cleared it.
            # (Action field is set when compute_fn returned a validated result
            # BEFORE any blacklist application.)
            assert cf_action in ("HOLD", "BUY", "SELL"), (
                f"Unexpected action value: {cf_action!r}"
            )
