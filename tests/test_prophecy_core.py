"""Prophecy: schema / strategies / config unit tests.

Targeted suite (run: pytest tests/test_prophecy_core.py) — does not need
config.json or the network. xdist-safe: all file I/O is redirected to tmp_path
via the ``ptmp`` fixture that monkeypatches the prophecy.config path constants.
"""

from __future__ import annotations

import json

import pytest

from prophecy import config as pcfg
from prophecy import schema, strategies


@pytest.fixture
def ptmp(tmp_path, monkeypatch):
    """Redirect every prophecy on-disk path into tmp_path."""
    pdir = tmp_path / "prophecy_runs"
    monkeypatch.setattr(pcfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pcfg, "PROPHECY_DIR", pdir)
    monkeypatch.setattr(pcfg, "CONFIG_FILE", pdir / "prophecy_config.json")
    monkeypatch.setattr(pcfg, "FROZEN_SENTINEL", pdir / "SYSTEM_DISABLED")
    monkeypatch.setattr(pcfg, "JOURNAL_FILE", pdir / "prediction_journal.jsonl")
    monkeypatch.setattr(pcfg, "LATEST_FILE", pdir / "latest.json")
    monkeypatch.setattr(pcfg, "ACCURACY_JSONL", pdir / "accuracy.jsonl")
    monkeypatch.setattr(pcfg, "ACCURACY_FILE", pdir / "accuracy.json")
    monkeypatch.setattr(pcfg, "COST_LOG", pdir / "cost_log.jsonl")
    monkeypatch.setattr(pcfg, "QUARANTINE_DIR", pdir / "quarantine")
    pcfg.ensure_dirs()
    return pdir


# --- schema ----------------------------------------------------------------
def test_horizons_canonical():
    assert schema.HORIZONS == ["1d", "2d", "3d", "4d", "5d", "6d", "7d", "1mo", "2mo", "6mo"]
    assert set(schema.HORIZON_DELTAS) == set(schema.HORIZONS)


@pytest.mark.parametrize("found,required,expected", [
    (7, 7, "high"), (6, 7, "high"), (5, 7, "medium"), (3, 7, "low"),
    (2, 7, "insufficient"), (1, 7, "insufficient"), (0, 7, "insufficient"), (1, 0, "insufficient"),
])
def test_grade_sufficiency(found, required, expected):
    assert schema.grade_sufficiency(found, required) == expected


def test_build_coverage_needs_work_rules():
    assert schema.build_coverage(data_sufficiency="low", has_proper_equation=True)["needs_work"]
    assert schema.build_coverage(data_sufficiency="high", has_proper_equation=False)["needs_work"]
    assert not schema.build_coverage(data_sufficiency="high", has_proper_equation=True)["needs_work"]
    assert schema.build_coverage(data_sufficiency="garbage", has_proper_equation=True)["needs_work"]


def _good_horizon(target=100.0, direction="up"):
    return {"direction": direction, "target": target, "prob_up": 0.6, "prob_down": 0.3,
            "prob_flat": 0.1, "confidence": 0.55, "low": 95, "high": 105, "rationale": "x"}


def _record(horizons, coverage=None, instrument="BTC-USD"):
    rec = {"instrument": instrument, "horizons": horizons}
    if coverage is not None:
        rec["coverage"] = coverage
    return rec


def test_validate_good_record_all_horizons():
    hz = {h: _good_horizon() for h in schema.HORIZONS}
    clean, errs = schema.validate_record(_record(
        hz, coverage={"data_sufficiency": "high", "has_proper_equation": True, "needs_work": False}))
    assert clean is not None
    assert set(clean["horizons"]) == set(schema.HORIZONS)
    assert clean["coverage"]["needs_work"] is False
    assert errs == []


def test_validate_rejects_inf_and_nan_targets():
    hz = {"1d": _good_horizon(target=float("inf")), "2d": _good_horizon(target=float("nan")),
          "3d": _good_horizon(target=100.0)}
    clean, errs = schema.validate_record(_record(hz))
    assert list(clean["horizons"]) == ["3d"]  # inf+nan dropped
    assert "Infinity" not in json.dumps(clean) and "NaN" not in json.dumps(clean)
    assert clean["coverage"]["needs_work"]  # missing horizons -> gap


def test_validate_rejects_nonpositive_and_bad_direction():
    hz = {"1d": _good_horizon(target=-5), "2d": _good_horizon(direction="sideways"),
          "3d": _good_horizon()}
    clean, _ = schema.validate_record(_record(hz))
    assert list(clean["horizons"]) == ["3d"]


def test_validate_missing_instrument_is_none():
    clean, errs = schema.validate_record({"horizons": {"1d": _good_horizon()}})
    assert clean is None and errs


def test_validate_zero_valid_horizons_is_none():
    clean, _ = schema.validate_record(_record({"1d": {"direction": "up", "target": 0}}))
    assert clean is None


def test_validate_probability_normalisation():
    hz = {"1d": {"direction": "up", "target": 100, "prob_up": 6, "prob_down": 3, "prob_flat": 1,
                 "confidence": 2.0}}
    clean, _ = schema.validate_record(_record(hz))
    p = clean["horizons"]["1d"]
    assert abs(p["prob_up"] + p["prob_down"] + p["prob_flat"] - 1.0) < 1e-6
    assert p["confidence"] == 1.0  # clamped


def test_validate_synthesises_missing_probabilities():
    hz = {"1d": {"direction": "down", "target": 100, "confidence": 0.5}}
    clean, errs = schema.validate_record(_record(hz))
    p = clean["horizons"]["1d"]
    assert p["prob_down"] > p["prob_up"]
    assert any("no probabilities" in e for e in errs)


def test_needs_work_true_never_cleared():
    # producer claims needs_work True even though sufficiency is high -> stays True
    clean, _ = schema.validate_record(_record(
        {"1d": _good_horizon()},
        coverage={"data_sufficiency": "high", "has_proper_equation": True, "needs_work": True}))
    assert clean["coverage"]["needs_work"] is True


def test_missing_coverage_defaults_needs_work():
    clean, _ = schema.validate_record(_record({"1d": _good_horizon()}))  # no coverage
    assert clean["coverage"]["needs_work"] is True


# --- strategies ------------------------------------------------------------
def test_thirteen_playbooks_each_well_formed():
    insts = strategies.all_instruments()
    assert len(insts) == 13
    for inst in insts:
        pb = strategies.playbook_for(inst)
        assert pb.strategy_id and pb.asset_class and pb.price_model
        assert pb.required_inputs, f"{inst} has no required_inputs (coverage can't grade)"


def test_warrants_point_at_underlying_and_need_avanza():
    for w in ("XBT-TRACKER", "ETH-TRACKER", "MINI-SILVER"):
        pb = strategies.playbook_for(w)
        assert pb.asset_class == "warrant" and pb.underlying
        assert any(k.startswith("warrant_") for k in pb.required_inputs)
        assert "underlying_prediction" in pb.required_inputs


def test_playbook_for_unknown_is_none():
    assert strategies.playbook_for("NOPE") is None


# --- config ----------------------------------------------------------------
def test_default_config_all_enabled(ptmp):
    cfg = pcfg.load_config()
    assert pcfg.CONFIG_FILE.exists()  # written on first load
    assert len(pcfg.enabled_instruments(cfg)) == 13
    assert pcfg.model(cfg) == "claude-opus-4-8"
    assert pcfg.budget_soft_cap(cfg) is None


def test_disabling_instrument_excludes_it(ptmp):
    from portfolio.file_utils import atomic_write_json
    cfg = pcfg.load_config()
    cfg["instruments"]["BTC-USD"]["enabled"] = False
    atomic_write_json(pcfg.CONFIG_FILE, cfg)
    enabled = pcfg.enabled_instruments(pcfg.load_config())
    assert "BTC-USD" not in enabled and len(enabled) == 12


def test_freeze_sentinel_detected(ptmp):
    assert not pcfg.is_system_frozen()
    pcfg.FROZEN_SENTINEL.write_text("frozen")
    assert pcfg.is_system_frozen()


def test_config_merges_new_instrument_into_old_file(ptmp):
    from portfolio.file_utils import atomic_write_json
    # an old config predating some instruments still picks them up (enabled default)
    atomic_write_json(pcfg.CONFIG_FILE, {"model": "x", "instruments": {"BTC-USD": {"enabled": True}}})
    enabled = pcfg.enabled_instruments(pcfg.load_config())
    assert len(enabled) == 13  # the 12 not in file default to enabled
