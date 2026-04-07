"""Tests for portfolio.signal_engine — dynamic correlation groups."""

import pytest


# ---------------------------------------------------------------------------
# 3c. Static and dynamic correlation groups
# ---------------------------------------------------------------------------

class TestCorrelationGroups:

    def test_static_correlation_groups_has_expected_keys(self):
        """CORRELATION_GROUPS (static alias) should exist and contain known group names."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert isinstance(CORRELATION_GROUPS, dict)
        # Check a few known static groups
        assert "low_activity_timing" in CORRELATION_GROUPS
        assert "pattern_based" in CORRELATION_GROUPS
        # Values should be frozensets
        for name, members in CORRELATION_GROUPS.items():
            assert isinstance(members, frozenset), f"Group {name} should be frozenset"
            assert len(members) >= 2, f"Group {name} should have at least 2 members"

    def test_rsi_based_group_exists(self):
        """rsi_based group should contain mean_reversion and rsi (r=0.537)."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "rsi_based" in CORRELATION_GROUPS
        rsi_group = CORRELATION_GROUPS["rsi_based"]
        assert "mean_reversion" in rsi_group
        assert "rsi" in rsi_group

    def test_macro_regime_in_trend_direction(self):
        """macro_regime should be in trend_direction group (r=0.520 with trend)."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "trend_direction" in CORRELATION_GROUPS
        td_group = CORRELATION_GROUPS["trend_direction"]
        assert "macro_regime" in td_group
        assert "trend" in td_group

    def test_macro_regime_not_in_macro_external(self):
        """macro_regime was moved out of macro_external into trend_direction."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        me_group = CORRELATION_GROUPS["macro_external"]
        assert "macro_regime" not in me_group

    def test_static_groups_backward_compat_alias(self):
        """CORRELATION_GROUPS should be the same object as _STATIC_CORRELATION_GROUPS."""
        from portfolio.signal_engine import (
            CORRELATION_GROUPS,
            _STATIC_CORRELATION_GROUPS,
        )
        assert CORRELATION_GROUPS is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback(self, monkeypatch):
        """When no signal_log data exists, _compute_dynamic_correlation_groups
        should return _STATIC_CORRELATION_GROUPS."""
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )

        # Monkeypatch load_entries to return empty list (no data)
        import portfolio.accuracy_stats as acc_mod
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback_insufficient_data(self, monkeypatch):
        """With fewer than _DYNAMIC_CORR_MIN_SAMPLES entries, falls back to static."""
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )

        # Provide a small number of entries (below the 30 minimum)
        import portfolio.accuracy_stats as acc_mod
        fake_entries = [
            {
                "ts": "2026-04-01T00:00:00+00:00",
                "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}}},
                "outcomes": {},
            }
            for _ in range(5)
        ]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: fake_entries)

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_returns_dict_of_frozensets(self, monkeypatch):
        """Even when falling back, the return type is dict[str, frozenset]."""
        from portfolio.signal_engine import _compute_dynamic_correlation_groups

        import portfolio.accuracy_stats as acc_mod
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, frozenset)


# ---------------------------------------------------------------------------
# Directional bias penalty
# ---------------------------------------------------------------------------

class TestDirectionalBiasPenalty:

    def test_extreme_bias_reduces_weight(self):
        """Signals with bias > 85% should get _BIAS_PENALTY applied."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY", "calendar": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 100},
            "calendar": {"accuracy": 0.60, "total": 100},
        }
        # calendar has extreme bias (>85%), rsi does not
        activation_rates = {
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3},
            "calendar": {"bias": 0.95, "samples": 100, "normalized_weight": 0.25,
                         "activation_rate": 0.08},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        # Both BUY → should still return BUY
        assert result[0] == "BUY"

    def test_no_penalty_below_threshold(self):
        """Signals with bias <= 85% should NOT get extra penalty."""
        from portfolio.signal_engine import _weighted_consensus, _BIAS_THRESHOLD

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100}}
        activation_rates = {
            "rsi": {"bias": 0.5, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3},
        }
        # With bias=0.5 (< 0.85), no extra penalty should apply
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"

    def test_bias_penalty_not_applied_with_few_samples(self):
        """Bias penalty should not fire when samples < _BIAS_MIN_ACTIVE."""
        from portfolio.signal_engine import _weighted_consensus, _BIAS_MIN_ACTIVE

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100}}
        activation_rates = {
            "rsi": {"bias": 0.99, "samples": _BIAS_MIN_ACTIVE - 1,
                    "normalized_weight": 1.0, "activation_rate": 0.3},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"
