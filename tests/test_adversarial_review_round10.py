"""Regression tests for Codex round-10 findings on the 2026-04-17 merge.

Codex reviewed commit d98fd245 (the merge of fix/adversarial-review-20260417)
and identified three issues the first round of fixes missed:

  P1: _weighted_consensus still crashed on None stats in call sites OTHER
      than _count_active_voters_at_gate.
  P2: blend_accuracy_data recent-only <min_samples path used raw rc_acc
      instead of neutral 0.5 fallback.
  P3: replay regime_distribution counter incremented before outcome validation,
      inflating unscored-row regime coverage.
"""

from __future__ import annotations

import json
import math

import pytest

from portfolio.accuracy_stats import blend_accuracy_data
from portfolio.signal_engine import _weighted_consensus


class TestWeightedConsensusHandlesNoneStats:
    """Codex round-10 P1: _weighted_consensus must tolerate accuracy_data
    values that are None (half-written cache). Previously only
    _count_active_voters_at_gate was hardened; the main function still
    crashed at top-N sort / group-leader selection / crisis-mode check."""

    def _votes(self, accs):
        votes = {f"s{i}": "BUY" for i in range(len(accs))}
        accuracy = {}
        for i, acc in enumerate(accs):
            if acc is None:
                accuracy[f"s{i}"] = None  # simulates half-written cache
            else:
                accuracy[f"s{i}"] = {
                    "accuracy": acc, "total": 200,
                    "buy_accuracy": acc, "sell_accuracy": acc,
                    "total_buy": 100, "total_sell": 100,
                }
        return votes, accuracy

    def test_none_stats_does_not_crash_consensus(self):
        """Mixed: 4 signals with valid stats + 1 with None stats."""
        votes, accuracy = self._votes([0.60, 0.60, 0.60, 0.60, None])
        # Must not raise AttributeError.
        action, conf = _weighted_consensus(votes, accuracy, regime="unknown")
        # The valid signals should still drive consensus.
        assert action in ("BUY", "SELL", "HOLD")

    def test_all_none_stats_does_not_crash(self):
        """All-None accuracy_data must not crash. Signals default to safe
        values (accuracy=0.5, samples=0) and bypass gates."""
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY"}
        accuracy = {"s0": None, "s1": None, "s2": None}
        # Must not raise.
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        assert action in ("BUY", "SELL", "HOLD")

    def test_empty_accuracy_data_does_not_crash(self):
        """_weighted_consensus with totally empty accuracy_data must not crash."""
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY"}
        action, conf = _weighted_consensus(votes, {}, regime="trending-up")
        assert action in ("BUY", "SELL", "HOLD")


class TestBlendAccuracyRecentOnlyFallback:
    """Codex round-10 P2: recent-only signals below min_recent_samples
    must NOT carry raw recent accuracy into the blend - fall back to 0.5."""

    def test_recent_only_below_threshold_uses_neutral(self):
        """Signal only in recent with 20 samples (< 30 default threshold)."""
        alltime = {}
        recent = {
            "new_sig": {"accuracy": 0.80, "total": 20},  # <30 min_recent_samples
        }
        result = blend_accuracy_data(alltime, recent)
        assert "new_sig" in result
        # Must be neutral 0.5, NOT raw rc_acc=0.80.
        assert result["new_sig"]["accuracy"] == 0.5, (
            "Recent-only signal below min_recent_samples should default to "
            "0.5 neutral, not carry raw rc_acc into the blend"
        )

    def test_recent_only_at_threshold_uses_recent(self):
        """At-threshold (30 samples) passes through."""
        alltime = {}
        recent = {
            "new_sig": {"accuracy": 0.80, "total": 30},
        }
        result = blend_accuracy_data(alltime, recent)
        assert result["new_sig"]["accuracy"] == 0.80

    def test_recent_only_above_threshold_uses_recent(self):
        alltime = {}
        recent = {
            "new_sig": {"accuracy": 0.80, "total": 100},
        }
        result = blend_accuracy_data(alltime, recent)
        assert result["new_sig"]["accuracy"] == 0.80

    def test_alltime_only_still_works(self):
        """Pre-existing case: alltime-only signal passes through."""
        alltime = {"sig": {"accuracy": 0.60, "total": 200}}
        recent = {}
        result = blend_accuracy_data(alltime, recent)
        assert result["sig"]["accuracy"] == 0.60

    def test_both_present_below_threshold_uses_alltime(self):
        """When alltime exists and recent has <min_recent_samples, use alltime."""
        alltime = {"sig": {"accuracy": 0.60, "total": 200}}
        recent = {"sig": {"accuracy": 0.80, "total": 20}}  # <30
        result = blend_accuracy_data(alltime, recent)
        assert result["sig"]["accuracy"] == 0.60  # alltime wins


class TestReplayRegimeCounterAfterValidation:
    """Codex round-10 P3: regime_distribution must count only SCORED rows."""

    def test_regime_counter_only_counts_scored_rows(self, tmp_path, monkeypatch):
        """An entry without the requested horizon's outcome must NOT contribute
        to the regime_distribution."""
        log = tmp_path / "signal_log.jsonl"
        entries = [
            # Row 1: trending-up, HAS 1d outcome -> should count.
            {
                "ts": "2026-04-16T12:00:00+00:00",
                "tickers": {
                    "BTC-USD": {
                        "consensus": "BUY",
                        "regime": "trending-up",
                        "signals": {"rsi": "BUY"},
                    },
                },
                "outcomes": {"BTC-USD": {"1d": {"change_pct": 1.5}}},
            },
            # Row 2: trending-up, NO 1d outcome (only 3h) -> must NOT count for 1d.
            {
                "ts": "2026-04-16T13:00:00+00:00",
                "tickers": {
                    "BTC-USD": {
                        "consensus": "SELL",
                        "regime": "trending-up",
                        "signals": {"rsi": "SELL"},
                    },
                },
                "outcomes": {"BTC-USD": {"3h": {"change_pct": 0.3}}},
            },
        ]
        with log.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

        from scripts import replay_consensus as rc
        monkeypatch.setattr(rc, "LOG_FILE", log)

        # Provide a non-empty accuracy cache so replay doesn't raise.
        import portfolio.accuracy_stats as acc_stats
        monkeypatch.setattr(
            acc_stats, "load_cached_accuracy",
            lambda horizon: {"rsi": {"accuracy": 0.55, "total": 200,
                                      "buy_accuracy": 0.55, "sell_accuracy": 0.55,
                                      "total_buy": 100, "total_sell": 100}},
        )

        summary = rc.replay(days=365, horizon="1d")
        regime_dist = summary["regime_distribution"]
        # Only the scored row should have contributed.
        assert regime_dist.get("trending-up", 0) == 1, (
            f"Expected 1 scored trending-up row, got {regime_dist}. "
            f"Row 2 was unscored for 1d horizon and must not inflate the count."
        )
