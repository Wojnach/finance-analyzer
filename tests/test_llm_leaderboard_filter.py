"""Tests for `dashboard/app.py:_compute_llm_leaderboard` directional filter.

Regression guard for premortem N1 in the 2026-05-18 plan
(`.worktrees/shadow-gate-lora-20260518/docs/PLAN.md`): the dashboard
leaderboard endpoint must use the same directional filter as the
auto-promotion script, otherwise users see "64% accuracy" on the
dashboard while the gate (correctly) refuses to promote.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def patched_dashboard(monkeypatch, tmp_path):
    """Spin up dashboard/_compute_llm_leaderboard pointed at tmp_path data."""
    import dashboard.app as dashboard_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(dashboard_app, "DATA_DIR", data_dir)

    # The dashboard's _load_jsonl_impl reads with json.loads per line; rebuild
    # a minimal version that doesn't require the cache layer.
    def _stub_load_jsonl_impl(path):
        if not Path(path).exists():
            return []
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    monkeypatch.setattr(dashboard_app, "_load_jsonl_impl", _stub_load_jsonl_impl)

    # Stub the registry so the test doesn't depend on data/shadow_registry.json.
    fake_registry = {
        "shadows": {
            "cryptotrader_lm": {
                "status": "shadow",
                "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.60},
                "notes": "test",
                "entered_ts": "2026-05-15T00:00:00+00:00",
            },
            "ministral": {
                "status": "promoted",
                "promotion_criteria": {"min_samples": 100, "min_accuracy": 0.55},
                "notes": "test",
                "entered_ts": "2026-05-15T00:00:00+00:00",
            },
        }
    }

    def _stub_load_registry():
        return fake_registry

    def _stub_days_in_shadow(sig):
        return 3.0

    from portfolio import shadow_registry as sr_module
    monkeypatch.setattr(sr_module, "load_registry", _stub_load_registry)
    monkeypatch.setattr(sr_module, "days_in_shadow", _stub_days_in_shadow)

    return dashboard_app, data_dir


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_scaffold_abstain_rows_excluded_from_dashboard_accuracy(patched_dashboard):
    """The bug we fixed: scaffold rows scoring 64% on the dashboard."""
    dashboard_app, data_dir = patched_dashboard
    log_rows = [
        {"ts": f"2026-05-15T00:00:{i:02d}+00:00", "signal": "cryptotrader_lm",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25},
         "chosen": "HOLD", "confidence": 0.0, "tier": None}
        for i in range(10)
    ]
    out_rows = [
        {"ts": r["ts"], "signal": "cryptotrader_lm", "ticker": "BTC-USD",
         "horizon": "1d", "outcome": "HOLD" if i < 7 else "BUY"}
        for i, r in enumerate(log_rows)
    ]
    _write_jsonl(data_dir / "llm_probability_log.jsonl", log_rows)
    _write_jsonl(data_dir / "llm_probability_outcomes.jsonl", out_rows)

    payload = dashboard_app._compute_llm_leaderboard()
    by_name = {r["name"]: r for r in payload["signals"]}
    row = by_name["cryptotrader_lm"]
    assert row["n_samples"] == 10
    assert row["n_directional"] == 0
    assert row["n_with_outcome"] == 0
    assert row["accuracy"] is None  # zero directional → no accuracy figure


def test_directional_predictions_counted_correctly(patched_dashboard):
    dashboard_app, data_dir = patched_dashboard
    log_rows = [
        {"ts": "2026-05-15T00:00:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1},
         "chosen": "BUY", "confidence": 0.7, "tier": None},
        {"ts": "2026-05-15T00:01:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1},
         "chosen": "BUY", "confidence": 0.7, "tier": None},
        # one HOLD vote — must NOT count toward accuracy
        {"ts": "2026-05-15T00:02:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.3, "HOLD": 0.6, "SELL": 0.1},
         "chosen": "HOLD", "confidence": 0.6, "tier": None},
    ]
    out_rows = [
        {"ts": "2026-05-15T00:00:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d", "outcome": "BUY"},
        {"ts": "2026-05-15T00:01:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d", "outcome": "SELL"},
        {"ts": "2026-05-15T00:02:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d", "outcome": "HOLD"},
    ]
    _write_jsonl(data_dir / "llm_probability_log.jsonl", log_rows)
    _write_jsonl(data_dir / "llm_probability_outcomes.jsonl", out_rows)

    payload = dashboard_app._compute_llm_leaderboard()
    by_name = {r["name"]: r for r in payload["signals"]}
    row = by_name["ministral"]
    assert row["n_samples"] == 3
    assert row["n_directional"] == 2
    assert row["n_with_outcome"] == 2
    assert row["accuracy"] == 0.5  # 1 of 2 directional was correct
