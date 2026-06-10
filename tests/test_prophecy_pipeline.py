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


# --- audit batch 3 (2026-06-11): alerts level + rate limit ------------------
def _crit_lines():
    p = pcfg.DATA_DIR / "critical_errors.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_alert_default_level_is_critical_and_surfaced(ptmp):
    from prophecy.alerts import log_critical
    log_critical("prophecy_test", "boom", caller="test")
    rows = _crit_lines()
    assert len(rows) == 1 and rows[0]["level"] == "critical"

    # the startup check must actually see it (the original P1: it didn't)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import check_critical_errors as cce
    unresolved = cce.find_unresolved(rows, days=7)
    assert len(unresolved) == 1


def test_alert_rate_limit_one_per_category_per_day(ptmp):
    from prophecy.alerts import log_critical
    log_critical("prophecy_flood", "first", caller="test")
    log_critical("prophecy_flood", "second (suppressed)", caller="test")
    log_critical("prophecy_other", "different category", caller="test")
    rows = _crit_lines()
    cats = [r["category"] for r in rows]
    assert cats.count("prophecy_flood") == 1
    assert cats.count("prophecy_other") == 1


def test_alert_rate_limit_resets_next_day(ptmp):
    from prophecy import alerts
    from prophecy.alerts import log_critical
    log_critical("prophecy_flood", "day one", caller="test")
    # simulate yesterday's mark
    state_path = alerts._ratelimit_path()
    atomic_write_json(state_path, {"prophecy_flood": "2020-01-01"})
    log_critical("prophecy_flood", "next day", caller="test")
    assert sum(1 for r in _crit_lines() if r["category"] == "prophecy_flood") == 2


def test_alert_warning_level_not_rate_limited(ptmp):
    from prophecy.alerts import log_critical
    log_critical("prophecy_cost", "soft cap a", caller="test", level="warning")
    log_critical("prophecy_cost", "soft cap b", caller="test", level="warning")
    rows = [r for r in _crit_lines() if r["level"] == "warning"]
    assert len(rows) == 2


# --- audit batch 3: outcomes coverage reconcile -----------------------------
def test_outcomes_coverage_gap_raises_critical(ptmp, monkeypatch):
    # SAAB-B has no scoring source; un-flag it -> coverage bug -> critical
    cfg = pcfg._default_config()
    cfg["instruments"]["SAAB-B"]["scoreable"] = True
    atomic_write_json(pcfg.CONFIG_FILE, cfg)
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 100.0)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, _backdated_row())
    assert outcomes.score() == 0
    assert any(r["category"] == "prophecy_scoring_coverage" and "SAAB-B" in r["message"]
               for r in _crit_lines())


def test_outcomes_flagged_unscoreable_warns_not_critical(ptmp, monkeypatch, caplog):
    # default config flags warrants/Tier-2 scoreable=false -> single warning,
    # zero criticals, rows skipped without counting as silent noise
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 100.0)
    row = _backdated_row()
    row["instrument"] = "XBT-TRACKER"
    atomic_append_jsonl(pcfg.JOURNAL_FILE, row)
    with caplog.at_level("WARNING", logger="prophecy.outcomes"):
        assert outcomes.score() == 0
    assert any("unscoreable by config" in m for m in caplog.messages)
    assert not any(r["category"] == "prophecy_scoring_coverage" for r in _crit_lines())
    assert not pcfg.ACCURACY_JSONL.exists() or count_jsonl_lines(pcfg.ACCURACY_JSONL) == 0


def test_outcomes_oil_is_scoreable_via_daily_bar(ptmp, monkeypatch):
    assert outcomes._scoring_source("CL=F") == "daily_bar"
    assert outcomes._scoring_source("BZ=F") == "daily_bar"
    assert outcomes._scoring_source("MSTR") == "daily_bar"
    assert outcomes._scoring_source("BTC-USD") == "binance"
    assert outcomes._scoring_source("SAAB-B") is None


# --- audit batch 3: weekend/holiday horizon fix ------------------------------
def test_outcomes_daily_bar_pending_until_next_session(ptmp, monkeypatch):
    # Friday prediction, 1d horizon = Saturday: no bar on/after target yet ->
    # pending, NOT scored against Friday's close (the old ~4h-window bug)
    monkeypatch.setattr(outcomes, "_fetch_daily_bar_close", lambda inst, dt: (None, None))
    row = _backdated_row(days_ago=2, spot=400.0)
    row["instrument"] = "MSTR"
    row["spot_source"] = "price_source"  # trusted -> no reference fetch
    atomic_append_jsonl(pcfg.JOURNAL_FILE, row)
    assert outcomes.score() == 0
    assert not pcfg.ACCURACY_JSONL.exists() or count_jsonl_lines(pcfg.ACCURACY_JSONL) == 0


def test_outcomes_daily_bar_scores_next_session_close(ptmp, monkeypatch):
    monkeypatch.setattr(outcomes, "_fetch_daily_bar_close",
                        lambda inst, dt: (440.0, "2026-06-01"))
    row = _backdated_row(days_ago=9, spot=400.0)
    row["instrument"] = "MSTR"
    row["spot_source"] = "price_source"
    atomic_append_jsonl(pcfg.JOURNAL_FILE, row)
    assert outcomes.score() == 0
    rows = [json.loads(line) for line in pcfg.ACCURACY_JSONL.read_text().splitlines()]
    assert rows and all(r["realized_bar_ts"] == "2026-06-01" for r in rows)
    assert all(r["realized"] == 440.0 for r in rows)


# --- audit batch 3: spot_at_prediction validation ----------------------------
def test_outcomes_poisoned_spot_skipped_not_crash(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 66000.0)
    bad = _backdated_row()
    bad["spot_at_prediction"] = "not-a-number"
    good = _backdated_row()
    atomic_append_jsonl(pcfg.JOURNAL_FILE, bad)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, good)
    assert outcomes.score() == 0  # poisoned row must not abort the loop
    assert any(r["category"] == "prophecy_spot_invalid" for r in _crit_lines())
    # the good row still scored
    assert count_jsonl_lines(pcfg.ACCURACY_JSONL) > 0


def test_outcomes_untrusted_spot_band_reject(ptmp, monkeypatch):
    # reference (scoring source at prediction time) = 100000; claimed spot
    # 60000 breaches the +/-20% band -> record rejected + critical
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 100000.0)
    atomic_append_jsonl(pcfg.JOURNAL_FILE, _backdated_row(spot=60000.0))  # no spot_source -> untrusted
    assert outcomes.score() == 0
    assert any(r["category"] == "prophecy_spot_invalid" for r in _crit_lines())
    assert not pcfg.ACCURACY_JSONL.exists() or count_jsonl_lines(pcfg.ACCURACY_JSONL) == 0


def test_outcomes_trusted_spot_skips_band_check(ptmp, monkeypatch):
    monkeypatch.setattr("portfolio.outcome_tracker._fetch_historical_price", lambda t, ts: 100000.0)
    row = _backdated_row(spot=60000.0)
    row["spot_source"] = "binance_spot"  # prep-sourced -> trusted
    atomic_append_jsonl(pcfg.JOURNAL_FILE, row)
    assert outcomes.score() == 0
    assert count_jsonl_lines(pcfg.ACCURACY_JSONL) > 0


# --- audit batch 3: publish reconcile ----------------------------------------
def test_publish_rejects_hallucinated_instrument(ptmp):
    _write_context(ptmp, {"BTC-USD": {"live_price": 61000.0, "price_source": "binance_spot"}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(61000, True)},
        "DOGE-USD": {"instrument": "DOGE-USD", "horizons": _hz(0.1, True)}}))
    assert publish.publish(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE)
    assert "DOGE-USD" not in latest["instruments"]
    assert latest["quarantined_count"] == 1
    assert count_jsonl_lines(pcfg.JOURNAL_FILE) == 1  # DOGE never journaled
    assert any(r["category"] == "prophecy_publish_unknown_instrument" for r in _crit_lines())


def test_publish_flags_missing_instruments(ptmp):
    _write_context(ptmp, {"BTC-USD": {"live_price": 61000.0, "price_source": "binance_spot"}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(61000, True)}}))
    assert publish.publish(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE)
    missing = latest["coverage_summary"]["missing_instruments"]
    assert "ETH-USD" in missing and "XAG-USD" in missing and len(missing) == 12
    assert latest["stale"] is False  # partial run publishes, but flagged
    assert any(r["category"] == "prophecy_publish_missing_instruments" for r in _crit_lines())


def test_publish_full_run_has_no_missing(ptmp):
    insts = pcfg.enabled_instruments()
    _write_context(ptmp, {i: {"live_price": 100.0, "price_source": "binance_spot"} for i in insts})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        i: {"instrument": i, "horizons": _hz(100, True)} for i in insts}))
    assert publish.publish(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE)
    assert latest["coverage_summary"]["missing_instruments"] == []
    assert not any(r["category"] == "prophecy_publish_missing_instruments" for r in _crit_lines())


# --- audit batch 3: publish spot fallback validation -------------------------
def test_publish_claude_spot_band_breach_dropped(ptmp, monkeypatch):
    monkeypatch.setattr("prophecy.prep._fetch_price", lambda t: (100.0, "binance_spot"))
    _write_context(ptmp, {"BTC-USD": {"live_price": None, "price_source": "no_feed"}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(100, True),
                    "spot_at_prediction": 200.0}}))  # 2x reference -> breach
    assert publish.publish(DATE) == 0
    latest = load_json(pcfg.LATEST_FILE)
    assert latest["instruments"]["BTC-USD"]["spot_at_prediction"] is None
    assert any(r["category"] == "prophecy_spot_invalid" for r in _crit_lines())


def test_publish_claude_spot_within_band_kept_and_tagged(ptmp, monkeypatch):
    monkeypatch.setattr("prophecy.prep._fetch_price", lambda t: (100.0, "binance_spot"))
    _write_context(ptmp, {"BTC-USD": {"live_price": None, "price_source": "no_feed"}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(100, True),
                    "spot_at_prediction": 105.0}}))
    assert publish.publish(DATE) == 0
    rec = load_json(pcfg.LATEST_FILE)["instruments"]["BTC-USD"]
    assert rec["spot_at_prediction"] == 105.0
    assert rec["spot_source"] == "claude_self_reported"


def test_publish_claude_spot_nonnumeric_dropped(ptmp, monkeypatch):
    monkeypatch.setattr("prophecy.prep._fetch_price", lambda t: (None, "no_feed"))
    _write_context(ptmp, {"BTC-USD": {"live_price": None, "price_source": "no_feed"}})
    atomic_write_json(pcfg.raw_file(DATE), _raw({
        "BTC-USD": {"instrument": "BTC-USD", "horizons": _hz(100, True),
                    "spot_at_prediction": {"weird": "dict"}}}))
    assert publish.publish(DATE) == 0
    assert load_json(pcfg.LATEST_FILE)["instruments"]["BTC-USD"]["spot_at_prediction"] is None


# --- audit batch 3: cost 30d window + same-day dedupe ------------------------
def _cost_row(days_ago, usd, session="s1", date=None):
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    return {"ts": ts, "date": date or ts[:10], "total_cost_usd": usd,
            "session_id": session, "is_error": False}


def test_cost_30d_window_excludes_old_rows(ptmp):
    atomic_append_jsonl(pcfg.COST_LOG, _cost_row(40, 100.0, session="old"))
    atomic_append_jsonl(pcfg.COST_LOG, _cost_row(5, 2.0, session="recent"))
    atomic_write_json(pcfg.run_file(DATE), {"type": "result", "is_error": False,
        "total_cost_usd": 3.0, "num_turns": 5, "usage": {}, "session_id": "today"})
    assert cost.record_cost(DATE) == 0
    cum = load_json(pcfg.LATEST_FILE)["cost_summary"]["cumulative_30d_usd"]
    assert cum == pytest.approx(5.0)  # 2 + 3; the 40-day-old 100 excluded


def test_cost_same_day_reparse_counted_once(ptmp):
    # same (date, session_id) appended twice = same claude spend re-parsed
    atomic_write_json(pcfg.run_file(DATE), {"type": "result", "is_error": False,
        "total_cost_usd": 4.0, "num_turns": 5, "usage": {}, "session_id": "sess-a"})
    assert cost.record_cost(DATE) == 0
    assert cost.record_cost(DATE) == 0  # manual re-run, same result file
    assert count_jsonl_lines(pcfg.COST_LOG) == 2  # audit trail keeps both rows
    cum = load_json(pcfg.LATEST_FILE)["cost_summary"]["cumulative_30d_usd"]
    assert cum == pytest.approx(4.0)  # but the spend counts once


def test_cost_same_day_distinct_sessions_both_count(ptmp):
    # a genuine same-day re-run (new claude session) is real additional spend
    atomic_append_jsonl(pcfg.COST_LOG, _cost_row(0, 6.0, session="sess-a"))
    atomic_write_json(pcfg.run_file(DATE), {"type": "result", "is_error": False,
        "total_cost_usd": 4.0, "num_turns": 5, "usage": {}, "session_id": "sess-b"})
    assert cost.record_cost(DATE) == 0
    cum = load_json(pcfg.LATEST_FILE)["cost_summary"]["cumulative_30d_usd"]
    assert cum == pytest.approx(10.0)


def test_cost_records_result_model_when_present(ptmp):
    atomic_write_json(pcfg.run_file(DATE), {"type": "result", "is_error": False,
        "total_cost_usd": 1.0, "num_turns": 5, "usage": {}, "model": "claude-opus-4-9"})
    assert cost.record_cost(DATE) == 0
    rows = [json.loads(line) for line in pcfg.COST_LOG.read_text().splitlines()]
    assert rows[-1]["model"] == "claude-opus-4-9"
