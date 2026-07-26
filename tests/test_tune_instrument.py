"""Tests for scripts/tune_instrument.py — per-instrument component tuning.

The point of these tests is the STATISTICS, not the plumbing: the tool's
whole value is refusing to make a decision from evidence that only looks
strong. Two independent guards must hold:

  1. raw sample floor  (min_samples)
  2. overlap guard     (independent outcome windows must fit the data span)

Both were found the hard way on 2026-07-26: the first draft proposed 15
XAG-USD changes, of which 13 came from 3d/5d horizons whose "n=79" was
~2 non-overlapping windows inside a 6.7-day span.
"""

import json

import pytest

from scripts.tune_instrument import (
    DISABLE,
    ENABLE,
    KEEP,
    SKIP,
    WATCH,
    analyse,
    build_overlay_patch,
    classify,
    effective_windows,
    merge_into_overlay,
    wilson_interval,
)


class TestWilsonInterval:
    def test_degenerate_zero_total(self):
        assert wilson_interval(0, 0) == (0.0, 100.0)

    def test_small_n_interval_is_wide(self):
        """15/16 = 93.8% must NOT be treated as a precise 93.8%."""
        lo, hi = wilson_interval(15, 16)
        assert lo < 75.0 and hi > 95.0

    def test_larger_n_tightens(self):
        lo_small, hi_small = wilson_interval(30, 40)
        lo_big, hi_big = wilson_interval(300, 400)
        assert (hi_big - lo_big) < (hi_small - lo_small)

    def test_never_exceeds_bounds(self):
        lo, hi = wilson_interval(0, 50)
        assert lo >= 0.0
        lo2, hi2 = wilson_interval(50, 50)
        assert hi2 <= 100.0


class TestEffectiveWindows:
    def test_short_horizon_many_windows(self):
        # 6.7 days of data / 3h horizon = ~53 independent windows
        assert 50 < effective_windows("3h", 6.7) < 56

    def test_long_horizon_few_windows(self):
        # THE BUG: 6.7 days cannot hold more than ~2 independent 3d windows
        assert effective_windows("3d", 6.7) == pytest.approx(2.23, abs=0.01)

    def test_unknown_span_returns_none(self):
        assert effective_windows("1d", None) is None

    def test_unknown_horizon_returns_none(self):
        assert effective_windows("nonsense", 30.0) is None


class TestClassifyGuards:
    def test_below_min_samples_is_skip(self):
        d, reason = classify(
            15, 16, False, keep_bar=60, gate=47, min_samples=30, windows=100
        )
        assert d == SKIP
        assert "no evidence" in reason

    def test_overlap_guard_blocks_decision(self):
        """81% on n=79 must be SKIP when only ~2 independent windows fit —
        this is the regression that made the first draft propose 15 changes."""
        d, reason = classify(
            64, 79, False, keep_bar=60, gate=47, min_samples=30, windows=2.23
        )
        assert d == SKIP
        assert "independent outcome windows" in reason

    def test_zero_pct_with_overlap_is_also_skip(self):
        """0.0% on n=71 is NOT 'provably worse than chance' when it is one
        market move counted 71 times."""
        d, _ = classify(0, 71, True, keep_bar=60, gate=47, min_samples=30, windows=2.23)
        assert d == SKIP

    def test_enable_when_ci_low_clears_bar(self):
        d, reason = classify(
            56, 71, False, keep_bar=60, gate=47, min_samples=30, windows=54
        )
        assert d == ENABLE
        assert "keep-bar" in reason

    def test_keep_when_already_enabled_and_strong(self):
        d, _ = classify(56, 71, True, keep_bar=60, gate=47, min_samples=30, windows=54)
        assert d == KEEP

    def test_disable_when_ci_high_below_gate(self):
        d, reason = classify(
            16, 64, True, keep_bar=60, gate=47, min_samples=30, windows=54
        )
        assert d == DISABLE
        assert "gate" in reason

    def test_watch_when_weak_but_already_disabled(self):
        d, _ = classify(16, 64, False, keep_bar=60, gate=47, min_samples=30, windows=54)
        assert d == WATCH

    def test_inconclusive_is_no_op(self):
        """A CI straddling both bars must not move anything."""
        d_en, _ = classify(
            40, 70, True, keep_bar=60, gate=47, min_samples=30, windows=54
        )
        d_dis, _ = classify(
            40, 70, False, keep_bar=60, gate=47, min_samples=30, windows=54
        )
        assert (d_en, d_dis) == (KEEP, WATCH)

    def test_effective_n_widens_interval(self):
        """Same rate, fewer independent windows => wider CI => weaker claim.
        A rate that clears the bar on raw n can fail on effective n."""
        # 46/60 = 76.7%: CI-low 64.6 on raw n (clears 60), but only 53.1 once
        # downscaled to 20 independent windows (fails 60).
        strong, _ = classify(
            46, 60, False, keep_bar=60, gate=47, min_samples=30, windows=1000
        )
        weakened, _ = classify(
            46, 60, False, keep_bar=60, gate=47, min_samples=30, windows=20
        )
        assert strong == ENABLE
        assert weakened in (WATCH, SKIP)


class TestOverlayPatch:
    def test_patch_uses_per_horizon_keys(self):
        changes = [
            {
                "signal": "cubic_trend_persistence",
                "horizon": "3h",
                "decision": ENABLE,
                "reason": "78.9% ...",
            },
            {
                "signal": "drift_regime_gate",
                "horizon": "3h",
                "decision": DISABLE,
                "reason": "25.0% ...",
            },
        ]
        patch = build_overlay_patch("XAG-USD", changes)
        assert patch["XAG-USD"]["cubic_trend_persistence"]["horizons"] == {"3h": True}
        assert patch["XAG-USD"]["drift_regime_gate"]["horizons"] == {"3h": False}
        # top-level `enabled` must NOT be set — per-horizon tuning is the point
        assert "enabled" not in patch["XAG-USD"]["cubic_trend_persistence"]

    def test_multi_horizon_same_signal_merges(self):
        changes = [
            {"signal": "s", "horizon": "3h", "decision": ENABLE, "reason": "a"},
            {"signal": "s", "horizon": "1d", "decision": DISABLE, "reason": "b"},
        ]
        patch = build_overlay_patch("XAG-USD", changes)
        assert patch["XAG-USD"]["s"]["horizons"] == {"3h": True, "1d": False}

    def test_no_changes_yields_empty_patch(self):
        assert build_overlay_patch("XAG-USD", []) == {}

    def test_reason_is_attributed(self):
        changes = [{"signal": "s", "horizon": "3h", "decision": ENABLE, "reason": "x"}]
        patch = build_overlay_patch("XAG-USD", changes)
        assert "tune_instrument.py" in patch["XAG-USD"]["s"]["reason"]


class TestMergeIntoOverlay:
    def test_preserves_unrelated_entries(self, tmp_path):
        overlay = tmp_path / "registry_overrides.json"
        overlay.write_text(
            json.dumps(
                {
                    "BTC-USD": {"ml": {"enabled": True, "reason": "operator"}},
                    "XAG-USD": {"other_sig": {"horizons": {"1d": True}}},
                }
            )
        )
        patch = {"XAG-USD": {"new_sig": {"horizons": {"3h": True}, "reason": "r"}}}
        merged = merge_into_overlay(patch, overlay_file=overlay)
        assert merged["BTC-USD"]["ml"]["enabled"] is True
        assert merged["XAG-USD"]["other_sig"]["horizons"] == {"1d": True}
        assert merged["XAG-USD"]["new_sig"]["horizons"] == {"3h": True}

    def test_unions_horizons_for_existing_signal(self, tmp_path):
        overlay = tmp_path / "registry_overrides.json"
        overlay.write_text(json.dumps({"XAG-USD": {"s": {"horizons": {"1d": False}}}}))
        patch = {"XAG-USD": {"s": {"horizons": {"3h": True}, "reason": "r"}}}
        merged = merge_into_overlay(patch, overlay_file=overlay)
        assert merged["XAG-USD"]["s"]["horizons"] == {"1d": False, "3h": True}

    def test_missing_overlay_file_is_fine(self, tmp_path):
        patch = {"XAG-USD": {"s": {"horizons": {"3h": True}, "reason": "r"}}}
        merged = merge_into_overlay(patch, overlay_file=tmp_path / "absent.json")
        assert merged == patch

    def test_malformed_overlay_does_not_crash(self, tmp_path):
        overlay = tmp_path / "registry_overrides.json"
        overlay.write_text('"just-a-string"')
        patch = {"XAG-USD": {"s": {"horizons": {"3h": True}, "reason": "r"}}}
        merged = merge_into_overlay(patch, overlay_file=overlay)
        assert merged["XAG-USD"]["s"]["horizons"] == {"3h": True}


class TestAnalyseEndToEnd:
    def _cache(self, tmp_path, total=79, correct=64):
        f = tmp_path / "acc.json"
        f.write_text(
            json.dumps(
                {
                    "time": 1784000000,
                    "3h": {
                        "XAG-USD": {
                            "sig_strong": {"correct": 56, "total": 71},
                            "sig_weak": {"correct": 16, "total": 64},
                            "sig_thin": {"correct": 9, "total": 10},
                        }
                    },
                    "3d": {
                        "XAG-USD": {"sig_strong": {"correct": correct, "total": total}}
                    },
                }
            )
        )
        return f

    def test_short_span_suppresses_long_horizon(self, tmp_path):
        """With a 6.7d span, 3h decisions are allowed and 3d ones are not."""
        proposal = analyse("XAG-USD", acc_file=self._cache(tmp_path), span_days=6.7)
        by_hz = {(r["signal"], r["horizon"]): r["decision"] for r in proposal["rows"]}
        assert by_hz[("sig_strong", "3d")] == SKIP
        assert by_hz[("sig_strong", "3h")] == ENABLE
        assert by_hz[("sig_weak", "3h")] in (DISABLE, WATCH)
        # thin sample never decides
        assert by_hz[("sig_thin", "3h")] == SKIP

    def test_long_span_unlocks_long_horizon(self, tmp_path):
        """Same data, a year of span: the 3d cell now has enough windows."""
        proposal = analyse("XAG-USD", acc_file=self._cache(tmp_path), span_days=365.0)
        by_hz = {(r["signal"], r["horizon"]): r["decision"] for r in proposal["rows"]}
        assert by_hz[("sig_strong", "3d")] == ENABLE

    def test_grid_ci_matches_decision_ci(self, tmp_path):
        """The CI columns must be the ones the decision was made from —
        never the raw-n interval (they disagreed in the first draft)."""
        proposal = analyse("XAG-USD", acc_file=self._cache(tmp_path), span_days=6.7)
        row = next(
            r
            for r in proposal["rows"]
            if r["signal"] == "sig_strong" and r["horizon"] == "3d"
        )
        assert row["n_effective"] < row["total"]
        assert f"{row['ci_low']:.1f}" != f"{wilson_interval(64, 79)[0]:.1f}"

    def test_missing_ticker_yields_no_rows(self, tmp_path):
        proposal = analyse("NOPE-USD", acc_file=self._cache(tmp_path), span_days=6.7)
        assert proposal["rows"] == []
        assert proposal["changes"] == []
        assert proposal["overlay_patch"] == {}

    def test_absent_cache_file_is_graceful(self, tmp_path):
        proposal = analyse("XAG-USD", acc_file=tmp_path / "nope.json", span_days=6.7)
        assert proposal["rows"] == []

    def test_proposal_records_thresholds_and_span(self, tmp_path):
        proposal = analyse(
            "XAG-USD", acc_file=self._cache(tmp_path), span_days=6.7, min_samples=25
        )
        assert proposal["thresholds"]["min_samples"] == 25
        assert proposal["data_span_days"] == 6.7
