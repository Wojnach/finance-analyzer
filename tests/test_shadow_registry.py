"""Tests for portfolio.shadow_registry."""
from __future__ import annotations

import datetime as _dt
import json

import pytest

from portfolio import shadow_registry as mod


@pytest.fixture
def reg_path(tmp_path):
    return tmp_path / "shadow_registry.json"


def test_load_empty_returns_shadows_dict(reg_path):
    assert mod.load_registry(path=reg_path) == {"shadows": {}}


def test_add_shadow_round_trips(reg_path):
    mod.add_shadow(
        "fingpt",
        {"min_samples": 200, "min_accuracy": 0.60},
        notes="test",
        path=reg_path,
    )
    reg = mod.load_registry(path=reg_path)
    assert "fingpt" in reg["shadows"]
    entry = reg["shadows"]["fingpt"]
    assert entry["status"] == "shadow"
    assert entry["promotion_criteria"]["min_samples"] == 200
    assert entry["notes"] == "test"


def test_add_shadow_preserves_entered_ts_on_update(reg_path):
    mod.add_shadow(
        "fingpt",
        {"min_samples": 100},
        entered_ts="2026-04-09T00:00:00+00:00",
        path=reg_path,
    )
    # Re-register with different criteria and notes
    mod.add_shadow("fingpt", {"min_samples": 300}, notes="updated", path=reg_path)
    entry = mod.load_registry(path=reg_path)["shadows"]["fingpt"]
    assert entry["entered_shadow_ts"] == "2026-04-09T00:00:00+00:00"
    assert entry["promotion_criteria"]["min_samples"] == 300
    assert entry["notes"] == "updated"


def test_resolve_shadow_returns_true_when_found(reg_path):
    mod.add_shadow("fingpt", {"min_samples": 200}, path=reg_path)
    result = mod.resolve_shadow("fingpt", "promoted", notes="shipped", path=reg_path)
    assert result is True
    entry = mod.load_registry(path=reg_path)["shadows"]["fingpt"]
    assert entry["status"] == "promoted"
    assert entry["notes"] == "shipped"


def test_resolve_shadow_returns_false_when_missing(reg_path):
    assert mod.resolve_shadow("nonexistent", "promoted", path=reg_path) is False


def test_resolve_shadow_rejects_bad_status(reg_path):
    mod.add_shadow("fingpt", {"min_samples": 200}, path=reg_path)
    with pytest.raises(ValueError):
        mod.resolve_shadow("fingpt", "maybe", path=reg_path)


def test_days_in_shadow(reg_path):
    entered = "2026-04-01T00:00:00+00:00"
    mod.add_shadow("fingpt", {"min_samples": 200}, entered_ts=entered, path=reg_path)
    now = _dt.datetime(2026, 4, 21, 0, 0, 0, tzinfo=_dt.UTC)
    days = mod.days_in_shadow("fingpt", path=reg_path, now=now)
    assert days == pytest.approx(20.0, rel=1e-3)


def test_days_in_shadow_returns_none_for_unknown(reg_path):
    assert mod.days_in_shadow("nope", path=reg_path) is None


def test_stale_shadows_filters_by_age_and_status(reg_path):
    now = _dt.datetime(2026, 4, 21, 0, 0, 0, tzinfo=_dt.UTC)
    mod.add_shadow("old_shadow", {"min_samples": 200},
                     entered_ts="2026-03-01T00:00:00+00:00", path=reg_path)
    mod.add_shadow("new_shadow", {"min_samples": 200},
                     entered_ts="2026-04-20T00:00:00+00:00", path=reg_path)
    mod.add_shadow("old_promoted", {"min_samples": 200},
                     entered_ts="2026-02-01T00:00:00+00:00", path=reg_path)
    mod.resolve_shadow("old_promoted", "promoted", path=reg_path)

    stale = mod.stale_shadows(stale_days=30, path=reg_path, now=now)
    assert len(stale) == 1
    assert stale[0]["signal"] == "old_shadow"
    assert stale[0]["days_in_shadow"] > 30


def test_stale_shadows_sorted_oldest_first(reg_path):
    now = _dt.datetime(2026, 4, 21, 0, 0, 0, tzinfo=_dt.UTC)
    mod.add_shadow("medium", {"m": 1}, entered_ts="2026-03-01T00:00:00+00:00", path=reg_path)
    mod.add_shadow("oldest", {"m": 1}, entered_ts="2026-02-01T00:00:00+00:00", path=reg_path)
    mod.add_shadow("newer", {"m": 1}, entered_ts="2026-03-20T00:00:00+00:00", path=reg_path)
    stale = mod.stale_shadows(stale_days=30, path=reg_path, now=now)
    assert [s["signal"] for s in stale] == ["oldest", "medium", "newer"]


def test_seed_defaults_idempotent(reg_path):
    mod.seed_defaults(path=reg_path)
    first = mod.load_registry(path=reg_path)
    assert "fingpt" in first["shadows"]
    assert "kronos" in first["shadows"]
    # Modify one entry — seeding must not overwrite it.
    mod.resolve_shadow("fingpt", "promoted", path=reg_path)
    mod.seed_defaults(path=reg_path)
    second = mod.load_registry(path=reg_path)
    assert second["shadows"]["fingpt"]["status"] == "promoted"


def test_load_handles_corrupt_registry(reg_path):
    reg_path.write_text("not json")
    reg = mod.load_registry(path=reg_path)
    assert reg == {"shadows": {}}
