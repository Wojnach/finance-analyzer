"""Tests for `scripts/pickups/llm_cryptotrader_72h.py`.

Stages a minimal probability log + outcome file in `tmp_path/data` and
asserts the handler returns the correct verdict for the decision-tree
branches:

* `defer` when n_directional < min_directional
* `promote` when accuracy >= promote_bar
* `retire` when accuracy <= retire_bar
* `defer` when accuracy lands in the gap between retire_bar and promote_bar
* `error` (verdict swallowed, NOT raised) on a missing repo
"""

from __future__ import annotations

import json
from pathlib import Path


def _ensure_repo(tmp_path: Path):
    """Symlink the runtime repo's `portfolio` package into tmp_path so the
    handler can import `portfolio.llm_probability_log` without copying
    every module. Falls back to sys.path injection for Windows where
    symlinks need admin."""
    src = Path(__file__).resolve().parent.parent / "portfolio"
    dst = tmp_path / "portfolio"
    if dst.exists():
        return
    try:
        dst.symlink_to(src, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows non-admin can't symlink; use a sys.path shim instead.
        import sys
        sys.path.insert(0, str(src.parent))


def _make_pickup(thresholds: dict | None = None, merged_at: str = "2026-05-18T00:00:00+00:00"):
    return {
        "id": "TEST",
        "title": "test",
        "handler": "llm_cryptotrader_72h",
        "context": {
            "merged_at": merged_at,
            "decision_thresholds": thresholds or {
                "min_directional": 10,
                "min_accuracy_for_promote": 0.60,
                "max_accuracy_for_retire": 0.55,
            },
        },
    }


def _write_log_outcomes(tmp_path: Path, rows):
    """Write log + matching outcomes for a list of (offset_min, chosen, actual)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    log_path = data_dir / "llm_probability_log.jsonl"
    out_path = data_dir / "llm_probability_outcomes.jsonl"
    with log_path.open("w", encoding="utf-8") as lf, out_path.open("w", encoding="utf-8") as of:
        for i, (chosen, actual) in enumerate(rows):
            # Spread across hours so timestamps stay valid (no :60+ minutes).
            ts = f"2026-05-19T{i // 60:02d}:{i % 60:02d}:00+00:00"
            lf.write(json.dumps({
                "ts": ts, "signal": "cryptotrader_lm", "ticker": "BTC-USD",
                "horizon": "1d",
                "probs": {"BUY": 0.6, "HOLD": 0.2, "SELL": 0.2},
                "chosen": chosen, "confidence": 0.7, "tier": None,
            }) + "\n")
            if actual is not None:
                of.write(json.dumps({
                    "ts": ts, "signal": "cryptotrader_lm",
                    "ticker": "BTC-USD", "horizon": "1d", "outcome": actual,
                }) + "\n")


def test_defer_when_too_few_directional(tmp_path):
    _ensure_repo(tmp_path)
    from scripts.pickups import llm_cryptotrader_72h as h

    _write_log_outcomes(tmp_path, [("BUY", "BUY"), ("SELL", "SELL")])
    pickup = _make_pickup({
        "min_directional": 10, "min_accuracy_for_promote": 0.6,
        "max_accuracy_for_retire": 0.5,
    })
    res = h.run(pickup, tmp_path)
    assert res["verdict"] == "defer"
    assert "directional" in res["summary"].lower()


def test_promote_when_accuracy_high(tmp_path):
    _ensure_repo(tmp_path)
    from scripts.pickups import llm_cryptotrader_72h as h

    # 10 directional rows, 7 correct = 70% accuracy
    rows = [("BUY", "BUY")] * 7 + [("SELL", "BUY")] * 3
    _write_log_outcomes(tmp_path, rows)
    pickup = _make_pickup({
        "min_directional": 5, "min_accuracy_for_promote": 0.65,
        "max_accuracy_for_retire": 0.5,
    })
    res = h.run(pickup, tmp_path)
    assert res["verdict"] == "promote"


def test_retire_when_accuracy_low(tmp_path):
    _ensure_repo(tmp_path)
    from scripts.pickups import llm_cryptotrader_72h as h

    rows = [("BUY", "SELL")] * 8 + [("BUY", "BUY")] * 2  # 20% acc
    _write_log_outcomes(tmp_path, rows)
    pickup = _make_pickup({
        "min_directional": 5, "min_accuracy_for_promote": 0.6,
        "max_accuracy_for_retire": 0.4,
    })
    res = h.run(pickup, tmp_path)
    assert res["verdict"] == "retire"


def test_defer_in_gap_zone(tmp_path):
    _ensure_repo(tmp_path)
    from scripts.pickups import llm_cryptotrader_72h as h

    # 50% accuracy: between retire 40 and promote 70
    rows = [("BUY", "BUY")] * 5 + [("BUY", "SELL")] * 5
    _write_log_outcomes(tmp_path, rows)
    pickup = _make_pickup({
        "min_directional": 5, "min_accuracy_for_promote": 0.70,
        "max_accuracy_for_retire": 0.40,
    })
    res = h.run(pickup, tmp_path)
    assert res["verdict"] == "defer"
    assert "gap" in res["summary"].lower()


def test_abstain_rows_excluded(tmp_path):
    """Regression: scaffold abstain rows (conf=0/HOLD) must not count
    toward accuracy -- mirrors the 2026-05-18 gate fix."""
    _ensure_repo(tmp_path)
    from scripts.pickups import llm_cryptotrader_72h as h

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    log_path = data_dir / "llm_probability_log.jsonl"
    with log_path.open("w", encoding="utf-8") as lf:
        for i in range(100):
            lf.write(json.dumps({
                "ts": f"2026-05-19T{i // 60:02d}:{i % 60:02d}:00+00:00", "signal": "cryptotrader_lm",
                "ticker": "BTC-USD", "horizon": "1d",
                "probs": {"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25},
                "chosen": "HOLD", "confidence": 0.0, "tier": None,
            }) + "\n")
    pickup = _make_pickup({
        "min_directional": 50, "min_accuracy_for_promote": 0.6,
        "max_accuracy_for_retire": 0.5,
    })
    res = h.run(pickup, tmp_path)
    assert res["verdict"] == "defer"
    assert res["details"]["stats"]["n_directional"] == 0
    assert res["details"]["stats"]["n_total"] == 100


def test_handler_never_raises_on_bad_repo(tmp_path):
    from scripts.pickups import llm_cryptotrader_72h as h

    pickup = _make_pickup()
    res = h.run(pickup, tmp_path / "does-not-exist")
    # No logs file, no outcomes; should return defer (no rows) or error,
    # but MUST NOT raise.
    assert res["verdict"] in ("defer", "error")
