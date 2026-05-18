"""Tests for `scripts/review_shadow_signals._compute_signal_stats`.

Regression for the 2026-05-18 false-positive promotion: scaffold/abstain
rows (`confidence=0, chosen="HOLD"`) trivially passed the gate at ~64%
"accuracy" because outcome backfill labels ~64% of 1d windows as HOLD.

The fix routes accuracy counting through
`portfolio.llm_probability_log.is_directional_prediction` so only
directional rows (conf>0 AND chosen in {BUY, SELL}) populate the
matched/correct counters.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _import_review_module():
    """Import scripts/review_shadow_signals.py as a module without exec'ing main."""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "review_shadow_signals_under_test",
        str(repo_root / "scripts" / "review_shadow_signals.py"),
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _patch_data_dir(monkeypatch, tmp_path: Path, log_rows, out_rows):
    """Point the review module at tmp_path data files via REPO_ROOT swap."""
    mod = _import_review_module()
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "llm_probability_log.jsonl", log_rows)
    _write_jsonl(data_dir / "llm_probability_outcomes.jsonl", out_rows)
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
    return mod


def test_scaffold_abstain_rows_excluded_from_accuracy(monkeypatch, tmp_path):
    """The exact 2026-05-18 production failure: cryptotrader_lm/meta_trader
    scaffolds emit HOLD/conf=0; outcomes split ~50/50 between HOLD and
    BUY/SELL. The old gate matched HOLD→HOLD as correct and produced
    ~50%+ "accuracy" on zero real predictions. The new gate must drop
    those rows entirely so the promotion gate never fires."""
    log_rows = [
        # 10 abstain rows from a scaffold — all HOLD/conf=0
        {"ts": f"2026-05-{d:02d}T00:00:00+00:00", "signal": "cryptotrader_lm",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25},
         "chosen": "HOLD", "confidence": 0.0, "tier": None}
        for d in range(1, 11)
    ]
    out_rows = [
        # Backfill mostly says HOLD (matches scaffold's HOLD vote)
        {"ts": log_rows[i]["ts"], "signal": "cryptotrader_lm",
         "ticker": "BTC-USD", "horizon": "1d",
         "outcome": "HOLD" if i < 6 else "BUY"}
        for i in range(10)
    ]
    mod = _patch_data_dir(monkeypatch, tmp_path, log_rows, out_rows)
    stats = mod._compute_signal_stats()
    s = stats["cryptotrader_lm"]
    assert s["n"] == 10  # all rows still counted in raw sample count
    assert s["n_directional"] == 0  # but zero directional preds
    assert s["n_matched"] == 0  # so zero matched outcomes
    assert s["correct"] == 0


def test_directional_buy_correct_counted(monkeypatch, tmp_path):
    log_rows = [
        {"ts": "2026-05-01T00:00:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.7, "HOLD": 0.15, "SELL": 0.15},
         "chosen": "BUY", "confidence": 0.7, "tier": None},
        {"ts": "2026-05-02T00:00:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.7, "HOLD": 0.15, "SELL": 0.15},
         "chosen": "BUY", "confidence": 0.7, "tier": None},
    ]
    out_rows = [
        {"ts": "2026-05-01T00:00:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d", "outcome": "BUY"},
        {"ts": "2026-05-02T00:00:00+00:00", "signal": "ministral",
         "ticker": "BTC-USD", "horizon": "1d", "outcome": "SELL"},
    ]
    mod = _patch_data_dir(monkeypatch, tmp_path, log_rows, out_rows)
    stats = mod._compute_signal_stats()
    assert stats["ministral"]["n"] == 2
    assert stats["ministral"]["n_directional"] == 2
    assert stats["ministral"]["n_matched"] == 2
    assert stats["ministral"]["correct"] == 1


def test_hold_vote_with_real_confidence_not_counted(monkeypatch, tmp_path):
    """A model that says HOLD with conf=0.7 is still non-directional —
    HOLD carries no trading direction. Matches accuracy_cache.json
    methodology (correct_buy + correct_sell denominator)."""
    log_rows = [
        {"ts": "2026-05-01T00:00:00+00:00", "signal": "qwen3",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.15, "HOLD": 0.7, "SELL": 0.15},
         "chosen": "HOLD", "confidence": 0.7, "tier": None},
    ]
    out_rows = [
        {"ts": "2026-05-01T00:00:00+00:00", "signal": "qwen3",
         "ticker": "BTC-USD", "horizon": "1d", "outcome": "HOLD"},
    ]
    mod = _patch_data_dir(monkeypatch, tmp_path, log_rows, out_rows)
    stats = mod._compute_signal_stats()
    s = stats["qwen3"]
    assert s["n"] == 1
    assert s["n_directional"] == 0
    assert s["n_matched"] == 0


def test_eligible_for_promotion_blocks_abstain_only_scaffold(monkeypatch, tmp_path):
    """End-to-end: the 2026-05-18 cryptotrader_lm scaffold (zero directional
    preds, 992 abstain rows) must not be eligible for promotion under the
    new gate, even when the criteria would have passed pre-fix."""
    log_rows = [
        {"ts": f"2026-05-01T00:00:{i:02d}+00:00", "signal": "cryptotrader_lm",
         "ticker": "BTC-USD", "horizon": "1d",
         "probs": {"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25},
         "chosen": "HOLD", "confidence": 0.0, "tier": None}
        for i in range(300)
    ]
    out_rows = [
        {"ts": r["ts"], "signal": "cryptotrader_lm", "ticker": "BTC-USD",
         "horizon": "1d", "outcome": "HOLD"}
        for r in log_rows
    ]
    mod = _patch_data_dir(monkeypatch, tmp_path, log_rows, out_rows)
    stats = mod._compute_signal_stats()
    entry = {
        "status": "shadow",
        "promotion_criteria": {"min_samples": 200, "min_accuracy": 0.60},
    }
    ok, reason = mod._eligible_for_promotion(entry, stats["cryptotrader_lm"])
    assert ok is False
    assert "min_samples" in reason
