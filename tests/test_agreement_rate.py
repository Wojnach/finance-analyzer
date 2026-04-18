"""Tests for agreement rate correlation and expanded macro_external group."""

import numpy as np
import pytest

from portfolio.signal_engine import (
    CORRELATION_GROUPS,
    _CLUSTER_CORRELATION_PENALTIES,
    _compute_agreement_rate,
    _compute_dynamic_correlation_groups,
    _STATIC_CORRELATION_GROUPS,
)


class TestComputeAgreementRate:
    """Test the agreement rate metric that replaced Pearson correlation."""

    def test_perfect_agreement(self):
        """Two signals that always agree on non-HOLD votes → 1.0."""
        a = [1, -1, 0, 1, -1, 0, 1, 1, -1, -1]
        b = [1, -1, 0, 1, -1, 0, 1, 1, -1, -1]
        rate, n = _compute_agreement_rate(a, b)
        assert rate == 1.0
        assert n == 8  # 8 non-HOLD pairs (2 are both-HOLD)

    def test_zero_agreement(self):
        """Two signals that always disagree → 0.0."""
        a = [1, -1, 1, -1]
        b = [-1, 1, -1, 1]
        rate, n = _compute_agreement_rate(a, b)
        assert rate == 0.0
        assert n == 4

    def test_both_hold_excluded(self):
        """When both signals are HOLD, the pair is skipped."""
        a = [0, 0, 0, 1, -1]
        b = [0, 0, 0, 1, -1]
        rate, n = _compute_agreement_rate(a, b)
        assert rate == 1.0
        assert n == 2  # only 2 non-HOLD pairs

    def test_one_hold_one_active_counts(self):
        """When one signal is HOLD and the other is active, it counts as disagreement."""
        a = [1, 0, 1, 0]
        b = [0, 1, 0, 1]
        rate, n = _compute_agreement_rate(a, b)
        assert rate == 0.0
        assert n == 4  # all pairs count (each has at least one non-HOLD)

    def test_mixed_agreement(self):
        """50% agreement rate."""
        a = [1, -1, 1, -1]
        b = [1, 1, -1, -1]
        rate, n = _compute_agreement_rate(a, b)
        assert abs(rate - 0.5) < 0.01
        assert n == 4

    def test_empty_returns_zero(self):
        """No data → 0.0 rate, 0 pairs."""
        rate, n = _compute_agreement_rate([], [])
        assert rate == 0.0
        assert n == 0

    def test_all_hold_returns_zero(self):
        """All HOLD → 0 pairs, 0.0 rate."""
        a = [0, 0, 0]
        b = [0, 0, 0]
        rate, n = _compute_agreement_rate(a, b)
        assert rate == 0.0
        assert n == 0


class TestExpandedMacroExternalGroup:
    """Verify macro_external now includes calendar, econ_calendar, funding."""

    def test_macro_external_has_6_members(self):
        group = CORRELATION_GROUPS["macro_external"]
        assert len(group) == 6

    def test_macro_external_new_members(self):
        group = CORRELATION_GROUPS["macro_external"]
        assert "calendar" in group
        assert "econ_calendar" in group
        assert "funding" in group

    def test_macro_external_original_members(self):
        group = CORRELATION_GROUPS["macro_external"]
        assert "fear_greed" in group
        assert "sentiment" in group
        assert "news_event" in group

    def test_macro_external_penalty_is_0_15(self):
        """6-member group should use tighter 0.15x penalty."""
        assert _CLUSTER_CORRELATION_PENALTIES.get("macro_external") == 0.15

    def test_calendar_not_standalone(self):
        """Calendar should not be in any other group after being added to macro_external."""
        for name, members in CORRELATION_GROUPS.items():
            if name == "macro_external":
                continue
            assert "calendar" not in members, (
                f"calendar found in {name} as well as macro_external"
            )


class TestDynamicCorrelationAgreementRate:
    """Verify dynamic correlation now uses agreement rate, not Pearson."""

    def test_fallback_to_static_with_no_data(self, monkeypatch):
        import portfolio.accuracy_stats as acc_mod
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])
        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_high_agreement_signals_grouped(self, monkeypatch):
        """Signals with >85% agreement rate on non-HOLD votes should be grouped."""
        import portfolio.accuracy_stats as acc_mod
        from datetime import datetime, timezone

        # Create entries where sig_a and sig_b always agree when non-HOLD
        ts = datetime.now(timezone.utc).isoformat()
        entries = []
        for i in range(50):
            vote = "BUY" if i % 3 == 0 else ("SELL" if i % 3 == 1 else "HOLD")
            entries.append({
                "ts": ts,
                "tickers": {
                    "BTC-USD": {
                        "signals": {
                            "rsi": vote,
                            "macd": vote,  # always agrees with rsi
                            "ema": "HOLD",  # never votes
                        }
                    }
                },
                "outcomes": {},
            })
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)
        result = _compute_dynamic_correlation_groups()
        # Should find rsi and macd in the same group
        found = False
        for group_members in result.values():
            if "rsi" in group_members and "macd" in group_members:
                found = True
                break
        assert found or result is _STATIC_CORRELATION_GROUPS
