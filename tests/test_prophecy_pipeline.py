"""Prophecy: prep / publish / cost / outcomes pipeline tests.

Targeted suite (run: pytest tests/test_prophecy_pipeline.py). Network + module
file paths are mocked/redirected to tmp_path. xdist-safe.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, count_jsonl_lines, load_json
from prophecy import config as pcfg
from prophecy import cost, outcomes, prep, publish
from prophecy.schema import HORIZONS

DATE = "2026-06-06"


@pytest.fixture
def ptmp(tmp_path, monkeypatch):
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


def _hz(base=100.0, up=True):
    return {h: {"direction": "up" if up else "down",
                "target": round(base * (1 + 0.01 * (i + 1) * (1 if up else -1)), 4),
                "prob_up": 0.6 if up else 0.3, "prob_down": 0.3 if up else 0.6, "prob_flat": 0.1,
                "confidence": 0.55, "low": base * 0.95, "high": base * 1.05, "rationale": h}
            for i, h in enumerate(HORIZONS)}


def _raw(instruments):
    return {"date": DATE, "model": "claude-opus-4-8", "instruments": instruments}


def _write_context(ptmp, instruments):
    atomic_write_json(pcfg.context_file(DATE),
                      {"date": DATE, "model": "claude-opus-4-8", "instruments": instruments})


# --- prep ------------------------------------------------------------------
def test_prep_builds_bundle_with_coverage(ptmp, monkeypatch):
    monkeypatch.setattr(prep, "_fetch_price", lambda t: (100.0, "test_feed"))
    atomic_write_json(pcfg.DATA_DIR / "agent_summary.json",
                      {"signals": {"BTC-USD": {"rsi": 55, "regime": "trending-up"}},
                       "fear_greed": 20, "onchain": {"mvrv": 1.2}})
    ctx = prep.build_context(DATE, throttle=False)
    assert len(ctx["instruments"]) == 13
    btc = ctx["instruments"]["BTC-USD"]
    assert btc["live_price"] == 100.0 and btc["price_source"] == "test_feed"
    assert "coverage_seed" in btc and "data_sufficiency" in btc["coverage_seed"]


def test_prep_per_instrument_isolation(ptmp, monkeypatch):
    monkeypatch.setattr(prep, "_fetch_price", lambda t: (100.0, "test_feed"))
    real_seed = prep._seed_coverage

    def boom(inst, price, tokens):
        if inst == "ETH-USD":
            raise RuntimeError("synthetic prep failure")
        return real_seed(inst, price, tokens)

    monkeypatch.setattr(prep, "_seed_coverage", boom)
    ctx = prep.build_context(DATE, throttle=False)
    # whole sweep survives; failed instrument flagged needs_work, not dropped
    assert len(ctx["instruments"]) == 13
    eth = ctx["instruments"]["ETH-USD"]
    assert eth["price_source"] == "prep_error"
    assert eth["coverage_seed"]["needs_work"] is True
    assert ctx["instruments"]["BTC-USD"]["coverage_seed"]["data_sufficiency"] in (
        "high", "medium", "low", "insufficient")


# --- publish ---------------------------------------------------------------
def test_publish_happy_path(ptmp):
    _write_context(ptmp, {"BTC-USD": {"live_price": 61000.0, "price_source": "binance_spot",
                                      "coverage_seed": {"needs_work": False}}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(61000, True),
                    "coverage": {"data_sufficiency": "high", "has_proper_equation": True, "needs_work": False}}}))
    assert publish.publish(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE)
    assert latest["stale"] is False
    assert "BTC-USD" in latest["instruments"]
    assert latest["instruments"]["BTC-USD"]["spot_at_prediction"] == 61000.0
    assert count_jsonl_lines(pcfg.JOURNAL_FILE) == 1
    assert latest["coverage_summary"]["instruments_needing_work"] == []


def test_publish_quarantines_malformed(ptmp):
    _write_context(ptmp, {})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(61000, True)},
        "BAD": {"instrument": "BAD", "horizons": {"1d": {"direction": "x", "target": -1}}}}))
    assert publish.publish(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE)
    assert list(latest["instruments"]) == ["BTC-USD"]
    assert latest["quarantined_count"] == 1
    assert (pcfg.QUARANTINE_DIR / f"quarantine_{DATE}.json").exists()


def test_publish_seed_needs_work_propagates(ptmp):
    _write_context(ptmp, {"SAAB-B": {"live_price": None, "price_source": "no_feed",
                                     "coverage_seed": {"needs_work": True, "note": "no feed"}}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "SAAB-B": {"instrument": "SAAB-B", "horizons": _hz(420, True),
                   "coverage": {"data_sufficiency": "high", "has_proper_equation": True, "needs_work": False}}}))
    publish.publish(DATE)
    latest = load_json(pcfg.LATEST_FILE)
    # prep seed said needs_work -> publish must NOT let the record clear it
    assert latest["instruments"]["SAAB-B"]["coverage"]["needs_work"] is True


def test_publish_missing_raw_marks_stale(ptmp):
    assert publish.publish(DATE) == 1
    latest = load_json(pcfg.LATEST_FILE)
    assert latest["stale"] is True
    assert count_jsonl_lines(pcfg.JOURNAL_FILE) == 0  # no phantom rows
    assert (pcfg.DATA_DIR / "critical_errors.jsonl").exists()


def test_publish_stale_raw_older_than_context(ptmp):
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(61000, True)}}))
    _write_context(ptmp, {"BTC-USD": {"live_price": 61000.0}})
    # force raw mtime to predate context
    ctx_m = pcfg.context_file(DATE).stat().st_mtime
    os.utime(pcfg.raw_file(DATE), (ctx_m - 100, ctx_m - 100))
    assert publish.publish(DATE) == 1
    assert load_json(pcfg.LATEST_FILE)["stale"] is True


def test_publish_empty_records_marks_stale(ptmp):
    _write_context(ptmp, {})
    atomic_write_json(pcfg.raw_file(DATE), _raw({}))
    assert publish.publish(DATE) == 1
    assert load_json(pcfg.LATEST_FILE)["stale"] is True


# --- cost ------------------------------------------------------------------
def test_cost_parses_single_json(ptmp):
    atomic_write_json(pcfg.run_file(DATE), {"type": "result", "is_error": False, "num_turns": 42,
        "total_cost_usd": 7.89, "usage": {"input_tokens": 100, "output_tokens": 200}})
    assert cost.record_cost(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE) or {}
    assert latest["cost_summary"]["last_run_usd"] == 7.89
    assert count_jsonl_lines(pcfg.COST_LOG) == 1


def test_cost_parses_stream_json(ptmp):
    lines = [json.dumps({"type": "system", "subtype": "init"}),
             json.dumps({"type": "assistant"}),
             json.dumps({"type": "result", "is_error": False, "total_cost_usd": 3.21,
                         "num_turns": 10, "usage": {}})]
    pcfg.run_file(DATE).write_text("\n".join(lines))
    assert cost.record_cost(DATE) == 0
    assert load_json(pcfg.LATEST_FILE)["cost_summary"]["last_run_usd"] == 3.21


def test_cost_missing_run_file_alerts(ptmp):
    assert cost.record_cost(DATE) == 1
    assert (pcfg.DATA_DIR / "critical_errors.jsonl").exists()


def test_cost_soft_cap_alert(ptmp):
    atomic_write_json(pcfg.CONFIG_FILE, {**pcfg._default_config(), "budget_usd_soft_cap": 1.0})
    atomic_write_json(pcfg.run_file(DATE), {"type": "result", "is_error": False,
        "total_cost_usd": 50.0, "num_turns": 5, "usage": {}})
    assert cost.record_cost(DATE) == 0  # never blocks
    crit = (pcfg.DATA_DIR / "critical_errors.jsonl").read_text()
    assert "soft cap" in crit


# --- outcomes --------------------------------------------------------------
def _backdated_row(days_ago=9, spot=60000.0, up=True, date="2026-05-28"):
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    return {"run_id": "rid", "instrument": "BTC-USD", "ts": ts, "date": date,
            "spot_at_prediction": spot, "horizons": _hz(spot, up)}


def test_outcomes_scores_matured(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 66000.0)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, _backdated_row())
    assert outcomes.score() == 0
    acc = load_json(pcfg.ACCURACY_FILE)["instruments"]["BTC-USD"]
    cell = acc["1d"]
    assert cell["n"] == 1
    assert cell["dir_hit_rate"] == 1.0  # predicted up, realized 66000 > 60000
    assert cell["brier"] is not None and cell["target_mae"] is not None


def test_outcomes_idempotent(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 66000.0)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, _backdated_row())
    outcomes.score()
    n1 = load_json(pcfg.ACCURACY_FILE)["instruments"]["BTC-USD"]["1d"]["n"]
    outcomes.score()  # run again
    n2 = load_json(pcfg.ACCURACY_FILE)["instruments"]["BTC-USD"]["1d"]["n"]
    assert n1 == n2 == 1


def test_outcomes_skips_unmatured(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 66000.0)
    fresh = _backdated_row(days_ago=0)
    fresh["ts"] = datetime.now(UTC).isoformat()
    atomic_append_jsonl(pcfg.JOURNAL_FILE, fresh)
    outcomes.score()
    assert not pcfg.ACCURACY_JSONL.exists() or count_jsonl_lines(pcfg.ACCURACY_JSONL) == 0


def test_outcomes_unscorable_when_no_price(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: None)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, _backdated_row())
    assert outcomes.score() == 0
    assert not pcfg.ACCURACY_JSONL.exists() or count_jsonl_lines(pcfg.ACCURACY_JSONL) == 0


def test_outcomes_spot_zero_no_crash(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 100.0)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, _backdated_row(spot=0.0))
    assert outcomes.score() == 0  # no ZeroDivisionError
