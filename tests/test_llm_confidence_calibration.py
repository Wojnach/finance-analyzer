"""Tests for portfolio.llm_confidence_calibration + its wiring into
derive_probs_from_result.

2026-05-29 (llm-confidence-calibration): ministral/qwen3 log a scalar
confidence that is anti-calibrated; a fitted bin map replaces it with the
empirical P(correct) inside the confidence-split fallback. These tests cover:
  * calibrate() maps confidence through a fitted bin,
  * calibrate() is identity (no map / unknown signal / malformed / OOR),
  * the fallback in derive_probs_from_result still fires when no map exists,
  * derive_probs_from_result produces non-uniform, sum-to-1 probs when the
    map collapses an overconfident value toward a low hit rate.

All file I/O is via tmp_path; the module's TTL cache is invalidated per test
so production state is never read and tests are xdist-safe.
"""
from __future__ import annotations

import json

import pytest

from portfolio import llm_confidence_calibration as cal
from portfolio import llm_probability_log as plog


@pytest.fixture(autouse=True)
def _reset_cache():
    """Drop the module TTL cache before AND after each test."""
    cal._invalidate_cache()
    yield
    cal._invalidate_cache()


@pytest.fixture
def map_path(tmp_path):
    return tmp_path / "llm_confidence_calibration.json"


def _write_map(path, signals):
    path.write_text(json.dumps({
        "fitted_at": "2026-05-29T00:00:00+00:00",
        "method": "equal-width-bins",
        "signals": signals,
    }))


# --- calibrate() -------------------------------------------------------------


def test_calibrate_maps_through_fitted_bin(map_path):
    # 5 equal-width bins; conf 0.7 lands in bin [0.6, 0.8) → p_correct 0.30.
    _write_map(map_path, {
        "ministral": {"n": 1000, "bins": [
            [0.0, 0.2, 0.35, 200],
            [0.2, 0.4, 0.33, 200],
            [0.4, 0.6, 0.31, 200],
            [0.6, 0.8, 0.30, 200],
            [0.8, 1.0, 0.28, 200],
        ]},
    })
    out = cal.calibrate("ministral", "BUY", 0.7, path=map_path)
    assert out == pytest.approx(0.30)


def test_calibrate_top_bin_inclusive_of_one(map_path):
    _write_map(map_path, {
        "qwen3": {"n": 100, "bins": [
            [0.0, 0.5, 0.4, 50],
            [0.5, 1.0, 0.2, 50],
        ]},
    })
    assert cal.calibrate("qwen3", "SELL", 1.0, path=map_path) == pytest.approx(0.2)


def test_calibrate_identity_when_map_missing(map_path):
    # File does not exist → identity passthrough.
    assert not map_path.exists()
    assert cal.calibrate("ministral", "BUY", 0.82, path=map_path) == pytest.approx(0.82)


def test_calibrate_identity_for_unknown_signal(map_path):
    _write_map(map_path, {"ministral": {"n": 10, "bins": [[0.0, 1.0, 0.3, 10]]}})
    assert cal.calibrate("qwen3", "BUY", 0.9, path=map_path) == pytest.approx(0.9)


def test_calibrate_identity_for_undersampled_bin(map_path):
    # p_correct is None (bin too thin to trust) → identity.
    _write_map(map_path, {"ministral": {"n": 3, "bins": [
        [0.0, 0.5, None, 1], [0.5, 1.0, None, 2],
    ]}})
    assert cal.calibrate("ministral", "BUY", 0.6, path=map_path) == pytest.approx(0.6)


def test_calibrate_identity_on_malformed_map(map_path):
    map_path.write_text("{ this is not valid json")
    assert cal.calibrate("ministral", "BUY", 0.6, path=map_path) == pytest.approx(0.6)


def test_calibrate_passes_through_out_of_range(map_path):
    _write_map(map_path, {"ministral": {"n": 10, "bins": [[0.0, 1.0, 0.3, 10]]}})
    # confidence outside [0,1] is returned unchanged (caller clamps elsewhere).
    assert cal.calibrate("ministral", "BUY", 1.5, path=map_path) == 1.5
    assert cal.calibrate("ministral", "BUY", -0.2, path=map_path) == -0.2


def test_calibrate_never_raises_on_garbage_bins(map_path):
    map_path.write_text(json.dumps({"signals": {
        "ministral": {"bins": ["not-a-list", [0.0], {"x": 1}, [0.0, 0.5, "abc", 9]]},
    }}))
    # No usable bin matches → identity, and absolutely no exception.
    assert cal.calibrate("ministral", "BUY", 0.4, path=map_path) == pytest.approx(0.4)


def test_calibrate_non_numeric_confidence_passthrough(map_path):
    _write_map(map_path, {"ministral": {"n": 10, "bins": [[0.0, 1.0, 0.3, 10]]}})
    assert cal.calibrate("ministral", "BUY", "high", path=map_path) == "high"


# --- derive_probs_from_result wiring ----------------------------------------


def test_derive_probs_applies_calibration(map_path, monkeypatch):
    """When a map exists, the chosen action's probability reflects the
    calibrated (low) hit rate, NOT the raw overconfident confidence."""
    monkeypatch.setattr(cal, "_MAP_PATH", map_path)
    cal._invalidate_cache()
    _write_map(map_path, {
        "ministral": {"n": 1000, "bins": [
            [0.0, 0.5, 0.30, 500],
            [0.5, 1.0, 0.25, 500],  # raw conf 0.8 → calibrated 0.25
        ]},
    })
    probs = plog.derive_probs_from_result("ministral", "BUY", 0.8)
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # Chosen action collapsed toward the empirical 0.25, far below raw 0.8.
    assert probs["BUY"] == pytest.approx(0.25)
    # Non-uniform: the other two share the remainder equally here (0.375 each).
    assert probs["HOLD"] == pytest.approx(0.375)
    assert probs["SELL"] == pytest.approx(0.375)
    # Not the uniform 1/3-each distribution.
    assert abs(probs["BUY"] - 1.0 / 3.0) > 0.05


def test_derive_probs_fallback_when_no_map(map_path, monkeypatch):
    """No calibration map on disk → the confidence-split fallback is unchanged
    (identity calibration). This is the safety guarantee: a missing map can
    never make probs worse than the pre-calibration baseline."""
    monkeypatch.setattr(cal, "_MAP_PATH", map_path)
    cal._invalidate_cache()
    assert not map_path.exists()
    probs = plog.derive_probs_from_result("ministral", "BUY", 0.6)
    assert probs is not None
    # Exactly the old confidence-split behavior: 0.6 / 0.2 / 0.2.
    assert probs["BUY"] == pytest.approx(0.6)
    assert probs["HOLD"] == pytest.approx(0.2)
    assert probs["SELL"] == pytest.approx(0.2)


def test_derive_probs_avg_scores_branch_untouched_by_calibration(map_path, monkeypatch):
    """The rich avg_scores branch (sentiment family) must NOT be calibrated —
    its per-class scores are already real."""
    monkeypatch.setattr(cal, "_MAP_PATH", map_path)
    cal._invalidate_cache()
    _write_map(map_path, {"sentiment": {"n": 100, "bins": [[0.0, 1.0, 0.1, 100]]}})
    indicators = {"avg_scores": {"positive": 0.7, "negative": 0.1, "neutral": 0.2}}
    probs = plog.derive_probs_from_result(
        "sentiment", "BUY", 0.7, indicators=indicators,
    )
    assert probs is not None
    # Direct avg_scores mapping, NOT the calibrated 0.1.
    assert probs["BUY"] == pytest.approx(0.7)
    assert probs["SELL"] == pytest.approx(0.1)
    assert probs["HOLD"] == pytest.approx(0.2)


def test_derive_probs_sum_to_one_and_nonuniform_after_calibration(map_path, monkeypatch):
    monkeypatch.setattr(cal, "_MAP_PATH", map_path)
    cal._invalidate_cache()
    _write_map(map_path, {"qwen3": {"n": 400, "bins": [
        [0.0, 1.0, 0.40, 400],
    ]}})
    probs = plog.derive_probs_from_result("qwen3", "SELL", 0.9)
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["SELL"] == pytest.approx(0.40)
    # distribution is not uniform
    assert len({round(v, 6) for v in probs.values()}) > 1
