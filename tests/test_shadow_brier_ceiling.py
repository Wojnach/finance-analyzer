"""Tests for the Brier-score promotion ceiling (2026-05-29).

A shadow signal must NOT be promoted when its logged probability distribution
scores worse than uniform (max_brier, default 0.66) on its directional,
outcome-joined rows — even if its argmax accuracy clears the bar. But a Brier
that cannot be computed (too few joined outcomes) must NOT block promotion on
its own.

Covers:
  * _eligible_for_promotion rejects with an explicit brier reason when brier
    is computed on >= min_samples and exceeds max_brier,
  * passes when brier <= max_brier and accuracy is good,
  * a too-thin sample (matched below the brier bar) does NOT block on brier,
  * the default max_brier (0.66) applies when promotion_criteria omits it,
  * _compute_signal_stats accumulates brier_sum over directional joined rows.

All file I/O is isolated to tmp_path; xdist-safe.
"""
from __future__ import annotations

import datetime as _dt
import json

import pytest

from portfolio import shadow_registry
from scripts import review_shadow_signals as review


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    path = tmp_path / "shadow_registry.json"
    payload = {
        "shadows": {
            # explicit max_brier
            "calib": {
                "entered_shadow_ts": "2026-05-01T00:00:00+00:00",
                "promotion_criteria": {
                    "min_samples": 30, "min_accuracy": 0.55, "max_brier": 0.66,
                },
                "status": "shadow",
                "notes": "max_brier explicit",
                "last_reviewed_ts": "2026-05-01T00:00:00+00:00",
            },
            # NO max_brier key → default 0.66 must apply (backward compat)
            "nobrier": {
                "entered_shadow_ts": "2026-05-01T00:00:00+00:00",
                "promotion_criteria": {"min_samples": 30, "min_accuracy": 0.55},
                "status": "shadow",
                "notes": "no max_brier key",
                "last_reviewed_ts": "2026-05-01T00:00:00+00:00",
            },
        }
    }
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(shadow_registry, "_REGISTRY_FILE", path)
    shadow_registry._invalidate_promoted_cache()
    return path


def test_rejects_when_brier_exceeds_max(tmp_registry):
    entry = shadow_registry.load_registry()["shadows"]["calib"]
    # 60% accuracy clears min_accuracy=0.55, matched=50 clears min_samples=30,
    # but brier_mean = 39.5 / 50 = 0.79 > 0.66 → reject with brier reason.
    stats = {"n": 100, "n_matched": 50, "correct": 30, "brier_sum": 39.5}
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is False
    assert "brier" in reason.lower()
    assert "0.79" in reason or "0.790" in reason
    assert "0.66" in reason


def test_passes_when_brier_under_max_and_accuracy_good(tmp_registry):
    entry = shadow_registry.load_registry()["shadows"]["calib"]
    # 60% accuracy, brier_mean = 25.0 / 50 = 0.50 <= 0.66 → promote.
    stats = {"n": 80, "n_matched": 50, "correct": 30, "brier_sum": 25.0}
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is True
    assert "promote" in reason


def test_missing_brier_does_not_block(tmp_registry):
    """When brier_sum is absent (uncomputable), promotion is decided on the
    accuracy/sample gates alone — it must NOT be blocked just for lacking a
    Brier."""
    entry = shadow_registry.load_registry()["shadows"]["calib"]
    stats = {"n": 80, "n_matched": 50, "correct": 30}  # no brier_sum key
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is True
    assert "promote" in reason


def test_thin_sample_does_not_block_on_brier(tmp_path, monkeypatch):
    """A bad Brier on a sample below the Brier bar must not block — but here
    the sample is also below min_samples, so it fails on min_samples, NOT on
    brier. The point: the brier branch never fires below its sample bar."""
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"shadows": {"thin": {
        "entered_shadow_ts": "2026-05-01T00:00:00+00:00",
        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.55, "max_brier": 0.66},
        "status": "shadow",
    }}}))
    monkeypatch.setattr(shadow_registry, "_REGISTRY_FILE", path)
    entry = shadow_registry.load_registry()["shadows"]["thin"]
    # matched=20 < min_samples=200; brier_mean would be 0.95 (awful) but the
    # gate must report the min_samples failure, not brier.
    stats = {"n": 40, "n_matched": 20, "correct": 15, "brier_sum": 19.0}
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is False
    assert "min_samples" in reason
    assert "brier" not in reason.lower()


def test_no_min_samples_brier_uses_30_floor(tmp_path, monkeypatch):
    """When promotion_criteria has no min_samples, the brier ceiling only
    fires once matched >= 30 — a 10-sample bad Brier must not block."""
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"shadows": {"nofloor": {
        "entered_shadow_ts": "2026-05-01T00:00:00+00:00",
        "promotion_criteria": {"min_accuracy": 0.55, "max_brier": 0.66},
        "status": "shadow",
    }}}))
    monkeypatch.setattr(shadow_registry, "_REGISTRY_FILE", path)
    entry = shadow_registry.load_registry()["shadows"]["nofloor"]
    # matched=10 < 30 → brier not enforced; 60% accuracy → promote.
    stats = {"n": 20, "n_matched": 10, "correct": 6, "brier_sum": 9.5}  # mean 0.95
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is True
    assert "promote" in reason


def test_default_max_brier_applies_without_key(tmp_registry):
    """A shadow whose promotion_criteria omits max_brier still gets the 0.66
    default ceiling (backward compat)."""
    entry = shadow_registry.load_registry()["shadows"]["nobrier"]
    assert "max_brier" not in entry["promotion_criteria"]
    stats = {"n": 100, "n_matched": 50, "correct": 30, "brier_sum": 40.0}  # mean 0.80
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is False
    assert "brier" in reason.lower()
    assert "0.66" in reason


def test_max_brier_none_disables_ceiling(tmp_path, monkeypatch):
    """An explicit max_brier: null opts out of the ceiling entirely."""
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"shadows": {"opt_out": {
        "entered_shadow_ts": "2026-05-01T00:00:00+00:00",
        "promotion_criteria": {"min_samples": 30, "min_accuracy": 0.55, "max_brier": None},
        "status": "shadow",
    }}}))
    monkeypatch.setattr(shadow_registry, "_REGISTRY_FILE", path)
    entry = shadow_registry.load_registry()["shadows"]["opt_out"]
    stats = {"n": 100, "n_matched": 50, "correct": 30, "brier_sum": 40.0}  # mean 0.80
    ok, reason = review._eligible_for_promotion(entry, stats)
    assert ok is True
    assert "promote" in reason


def test_compute_signal_stats_accumulates_brier(tmp_path, monkeypatch):
    """_compute_signal_stats must accumulate brier_sum over the directional,
    outcome-joined rows (same set as correct/n_matched)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    log_path = data_dir / "llm_probability_log.jsonl"
    out_path = data_dir / "llm_probability_outcomes.jsonl"
    ts1 = "2026-05-20T00:00:00+00:00"
    ts2 = "2026-05-20T01:00:00+00:00"
    # Two directional rows for signal Z, both joined to outcomes.
    log_rows = [
        {"ts": ts1, "signal": "Z", "ticker": "BTC-USD", "horizon": "1d",
         "chosen": "BUY", "confidence": 0.8,
         "probs": {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1}},
        {"ts": ts2, "signal": "Z", "ticker": "BTC-USD", "horizon": "1d",
         "chosen": "SELL", "confidence": 0.6,
         "probs": {"BUY": 0.2, "HOLD": 0.2, "SELL": 0.6}},
        # An abstain HOLD row — excluded from directional accumulation.
        {"ts": ts2, "signal": "Z", "ticker": "ETH-USD", "horizon": "1d",
         "chosen": "HOLD", "confidence": 0.0,
         "probs": {"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25}},
    ]
    out_rows = [
        # BUY predicted, outcome SELL (wrong): brier = .8^2 + .1^2 + (.1-1)^2
        #   = 0.64 + 0.01 + 0.81 = 1.46
        {"ts": ts1, "signal": "Z", "ticker": "BTC-USD", "horizon": "1d", "outcome": "SELL"},
        # SELL predicted, outcome SELL (right): brier = .2^2 + .2^2 + (.6-1)^2
        #   = 0.04 + 0.04 + 0.16 = 0.24
        {"ts": ts2, "signal": "Z", "ticker": "BTC-USD", "horizon": "1d", "outcome": "SELL"},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in log_rows))
    out_path.write_text("\n".join(json.dumps(r) for r in out_rows))
    monkeypatch.setattr(review, "_REPO_ROOT", tmp_path)
    stats = review._compute_signal_stats()
    z = stats["Z"]
    assert z["n_matched"] == 2
    assert z["correct"] == 1
    assert z["brier_sum"] == pytest.approx(1.46 + 0.24)
    # brier_mean = 1.70 / 2 = 0.85 → above the 0.66 ceiling.
    assert z["brier_sum"] / z["n_matched"] == pytest.approx(0.85)
