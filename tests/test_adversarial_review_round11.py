"""Regression tests for Codex round-11 findings on commit a73f1739.

Three issues that the round-10 fixes left open:

  P1: _weighted_consensus sanitized container-level None values but still
      crashed on dict-valued stats with poisoned numeric fields.
  P2: blend_accuracy_data round-10 fix neutralized overall accuracy but
      kept directional stats (buy_accuracy/total_buy/...) for recent-only
      <min_samples signals.
  P3: replay regime_counter incremented past outcome validation but
      still ran for ERROR-simulation rows, inflating the distribution in
      degraded runs.
"""

from __future__ import annotations

import json

import pytest

from portfolio.accuracy_stats import blend_accuracy_data
from portfolio.signal_engine import _weighted_consensus


class TestWeightedConsensusSanitizesNumericFields:
    """Codex round-11 P1: dict-valued stats with None/NaN numeric fields
    must not crash the main gating loop."""

    def test_dict_with_none_accuracy_does_not_crash(self):
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY"}
        accuracy = {
            "s0": {"accuracy": None, "total": 100},
            "s1": {"accuracy": 0.60, "total": 200},
            "s2": {"accuracy": 0.60, "total": 200},
        }
        # Must not raise.
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        assert action in ("BUY", "SELL", "HOLD")

    def test_dict_with_nan_directional_does_not_crash(self):
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY"}
        accuracy = {
            "s0": {
                "accuracy": 0.60, "total": 100,
                "buy_accuracy": float("nan"), "total_buy": 50,
            },
            "s1": {"accuracy": 0.60, "total": 200},
            "s2": {"accuracy": 0.60, "total": 200},
        }
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        assert action in ("BUY", "SELL", "HOLD")

    def test_nan_confidence_not_propagated(self):
        """Even if a signal had NaN in one field, the final confidence must
        be a finite float, not NaN."""
        import math
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY", "s3": "BUY"}
        accuracy = {
            "s0": {"accuracy": float("nan"), "total": 100,
                   "buy_accuracy": float("nan"), "total_buy": 50},
            "s1": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s2": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s3": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        assert not math.isnan(conf), f"confidence became NaN: {conf}"


class TestBlendDirectionalFollowsSampleFloor:
    """Codex round-11 P2: directional stats must also be omitted for
    recent-only signals below min_recent_samples."""

    def test_recent_only_below_threshold_strips_directional(self):
        """A 20-sample recent-only signal shouldn't carry directional stats."""
        alltime = {}
        recent = {
            "new_sig": {
                "accuracy": 0.80, "total": 20,
                "buy_accuracy": 0.85, "total_buy": 12,
                "sell_accuracy": 0.75, "total_sell": 8,
            },
        }
        result = blend_accuracy_data(alltime, recent)
        assert "new_sig" in result
        # Overall blended to neutral 0.5 (round-10 fix).
        assert result["new_sig"]["accuracy"] == 0.5
        # Round-11 fix: directional keys must be absent so downstream
        # .get("buy_accuracy", acc) falls back to the overall 0.5.
        assert "buy_accuracy" not in result["new_sig"], (
            "Round-11 P2: directional keys must follow the same sample floor"
        )
        assert "sell_accuracy" not in result["new_sig"]
        assert "total_buy" not in result["new_sig"]
        assert "total_sell" not in result["new_sig"]

    def test_recent_only_at_threshold_keeps_directional(self):
        """At >= min_recent_samples, directional stats are trustworthy."""
        alltime = {}
        recent = {
            "new_sig": {
                "accuracy": 0.80, "total": 30,
                "buy_accuracy": 0.85, "total_buy": 15,
                "sell_accuracy": 0.75, "total_sell": 15,
            },
        }
        result = blend_accuracy_data(alltime, recent)
        assert result["new_sig"]["buy_accuracy"] == 0.85
        assert result["new_sig"]["total_buy"] == 15

    def test_alltime_present_recent_below_threshold_keeps_directional(self):
        """When alltime exists, directional from alltime flows through."""
        alltime = {
            "sig": {
                "accuracy": 0.60, "total": 200,
                "buy_accuracy": 0.55, "total_buy": 100,
                "sell_accuracy": 0.65, "total_sell": 100,
            },
        }
        recent = {
            "sig": {"accuracy": 0.80, "total": 20,
                    "buy_accuracy": 0.90, "total_buy": 10},
        }
        result = blend_accuracy_data(alltime, recent)
        # Overall uses alltime (recent below threshold).
        assert result["sig"]["accuracy"] == 0.60
        # Directional also from alltime since at_samples > 0.
        assert result["sig"]["buy_accuracy"] == 0.55


class TestReplayRegimeExcludesErrorRows:
    """Codex round-11 P3: regime_counter must not count simulation-error rows."""

    def test_error_row_does_not_inflate_regime_count(self, tmp_path, monkeypatch):
        log = tmp_path / "signal_log.jsonl"
        entries = [
            # Row with trending-up regime, will be forced to ERROR.
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
            # Row with ranging regime, successful sim.
            {
                "ts": "2026-04-16T13:00:00+00:00",
                "tickers": {
                    "BTC-USD": {
                        "consensus": "SELL",
                        "regime": "ranging",
                        "signals": {"rsi": "SELL"},
                    },
                },
                "outcomes": {"BTC-USD": {"1d": {"change_pct": -0.8}}},
            },
        ]
        with log.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

        from scripts import replay_consensus as rc
        monkeypatch.setattr(rc, "LOG_FILE", log)

        import portfolio.accuracy_stats as acc_stats
        monkeypatch.setattr(
            acc_stats, "load_cached_accuracy",
            lambda horizon: {"rsi": {"accuracy": 0.55, "total": 200,
                                      "buy_accuracy": 0.55, "sell_accuracy": 0.55,
                                      "total_buy": 100, "total_sell": 100}},
        )

        # Force the first row (trending-up) to ERROR in simulation.
        import portfolio.signal_engine as se
        real_fn = se._weighted_consensus
        call_count = [0]

        def _error_first(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("synthetic simulation error")
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(se, "_weighted_consensus", _error_first)

        summary = rc.replay(days=365, horizon="1d")
        regime_dist = summary["regime_distribution"]
        # The ERROR row (trending-up) must NOT contribute to regime_distribution.
        # Only the ranging row (successful simulation) should.
        assert regime_dist.get("trending-up", 0) == 0, (
            f"ERROR-simulation row still inflated regime_distribution: "
            f"{regime_dist}"
        )
        assert regime_dist.get("ranging", 0) == 1
