#!/usr/bin/env python3
"""Fit the LLM confidence calibration map from logged predictions ⨝ outcomes.

Why (2026-05-29)
----------------
ministral & qwen3 log only a scalar self-reported `confidence`, which
`llm_probability_log.derive_probs_from_result` confidence-splits into a
{BUY,HOLD,SELL} distribution. Measured against realized outcomes this is
anti-calibrated — high confidence on wrong calls — and the logged Brier is
WORSE than uniform (ministral 0.785, qwen3 0.793 vs 0.667).

This script bins each signal's raw confidence and computes the empirical
P(chosen action == realized outcome) per bin, writing the result to
`data/llm_confidence_calibration.json`. `portfolio.llm_confidence_calibration.
calibrate()` then substitutes that empirical hit-rate for the raw confidence
inside the confidence-split fallback, collapsing overconfident values toward
the true (low) hit rate and lowering Brier below uniform.

It joins the SAME two files the dashboard / shadow gate use:
  data/llm_probability_log.jsonl   ⨝  data/llm_probability_outcomes.jsonl
on (ts, signal, ticker, horizon), and applies the canonical directional
filter (`is_directional_prediction`) so abstain/HOLD rows don't pollute the
fit — matching `scripts/review_shadow_signals._compute_signal_stats`.

Usage:
    python scripts/fit_llm_confidence_calibration.py [--signals ministral,qwen3]
        [--bins 5] [--min-bin-count 20] [--show-brier] [--dry-run]

Safe to re-run; output is atomically written. If a signal has too few joined
directional samples the bins it can't fill get p_correct=null, which
calibrate() treats as identity (no change) — never worse than the baseline.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.file_utils import atomic_write_json, load_jsonl  # noqa: E402
from portfolio.llm_probability_log import is_directional_prediction  # noqa: E402

_ACTIONS = ("BUY", "HOLD", "SELL")
_DATA_DIR = _REPO_ROOT / "data"
_LOG = _DATA_DIR / "llm_probability_log.jsonl"
_OUTCOMES = _DATA_DIR / "llm_probability_outcomes.jsonl"
_MAP_OUT = _DATA_DIR / "llm_confidence_calibration.json"


def _load_outcomes(path: Path) -> dict:
    outcomes: dict = {}
    for row in load_jsonl(str(path)) or []:
        if not isinstance(row, dict):
            continue
        key = (row.get("ts"), row.get("signal"), row.get("ticker"), row.get("horizon"))
        outcomes[key] = row.get("outcome")
    return outcomes


def fit(
    signals: list[str],
    *,
    n_bins: int = 5,
    min_bin_count: int = 20,
    log_path: Path | None = None,
    outcomes_path: Path | None = None,
) -> dict:
    """Return the calibration-map dict (ready for atomic_write_json).

    Bins are equal-width over [0, 1]. Each bin's p_correct is the empirical
    hit rate of the chosen action vs. realized outcome within that bin, or
    None when the bin has fewer than `min_bin_count` joined samples (the
    consumer treats None as identity).
    """
    log_path = Path(log_path) if log_path else _LOG
    outcomes_path = Path(outcomes_path) if outcomes_path else _OUTCOMES

    outcomes = _load_outcomes(outcomes_path)
    log_rows = load_jsonl(str(log_path)) or []

    # per-signal: list of (confidence, correct_bool) over directional joined rows
    samples: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)
    want = set(signals)
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        sig = row.get("signal")
        if sig not in want:
            continue
        if not is_directional_prediction(row):
            continue
        conf = row.get("confidence")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= conf <= 1.0):
            continue
        key = (row.get("ts"), sig, row.get("ticker"), row.get("horizon"))
        actual = outcomes.get(key)
        if actual not in _ACTIONS:
            continue
        samples[sig].append((conf, row.get("chosen") == actual))

    signals_map: dict = {}
    for sig in signals:
        data = samples.get(sig, [])
        bin_hits = [[0, 0] for _ in range(n_bins)]  # [correct, total]
        for conf, correct in data:
            b = min(n_bins - 1, int(conf * n_bins))
            bin_hits[b][1] += 1
            if correct:
                bin_hits[b][0] += 1
        bins = []
        for i in range(n_bins):
            lo = round(i / n_bins, 6)
            hi = round((i + 1) / n_bins, 6)
            c, t = bin_hits[i]
            p = round(c / t, 6) if t >= min_bin_count else None
            bins.append([lo, hi, p, t])
        signals_map[sig] = {"n": len(data), "bins": bins}

    return {
        "fitted_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "method": "equal-width-bins",
        "n_bins": n_bins,
        "min_bin_count": min_bin_count,
        "signals": signals_map,
    }


def _brier_report(
    signals: list[str],
    calib_map: dict,
    *,
    log_path: Path | None = None,
    outcomes_path: Path | None = None,
) -> None:
    """Print before/after Brier on the directional joined set for each signal.

    Before = current synthetic confidence-split (raw conf). After = the same
    split but with conf replaced by the freshly-fitted bin p_correct. Lets the
    operator see the improvement without recomputing through the live path.
    """
    log_path = Path(log_path) if log_path else _LOG
    outcomes_path = Path(outcomes_path) if outcomes_path else _OUTCOMES
    outcomes = _load_outcomes(outcomes_path)
    log_rows = load_jsonl(str(log_path)) or []
    sig_map = calib_map.get("signals", {})

    def _split(action, p_action, base_other):
        p = {a: 0.0 for a in _ACTIONS}
        p[action] = p_action
        rem = 1.0 - p_action
        others = [a for a in _ACTIONS if a != action]
        w = sum(base_other.get(o, 1.0) for o in others) or 1.0
        for o in others:
            p[o] = rem * base_other.get(o, 1.0) / w
        return p

    def _brier(p, out):
        return sum((p[a] - (1.0 if a == out else 0.0)) ** 2 for a in _ACTIONS)

    def _calib_conf(sig, conf):
        entry = sig_map.get(sig) or {}
        for b in entry.get("bins", []):
            lo, hi, pc = b[0], b[1], b[2]
            if (lo <= conf < hi) or (conf >= hi and hi >= 1.0):
                return pc if pc is not None else conf
        return conf

    per = collections.defaultdict(list)
    base_rates = collections.defaultdict(collections.Counter)
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        sig = row.get("signal")
        if sig not in set(signals) or not is_directional_prediction(row):
            continue
        try:
            conf = float(row.get("confidence"))
        except (TypeError, ValueError):
            continue
        key = (row.get("ts"), sig, row.get("ticker"), row.get("horizon"))
        out = outcomes.get(key)
        if out not in _ACTIONS:
            continue
        per[sig].append((row.get("chosen"), conf, out))
        base_rates[sig][out] += 1

    print("\nBefore/after Brier (directional joined set):")
    for sig in signals:
        data = per.get(sig, [])
        if not data:
            print(f"  {sig:12s} (no joined directional samples)")
            continue
        br = {a: base_rates[sig].get(a, 0) + 1 for a in _ACTIONS}
        before = after = 0.0
        for chosen, conf, out in data:
            c = max(conf, 1.0 / 3.0 + 0.05)
            pb = {a: (1 - c) / 2 for a in _ACTIONS}
            pb[chosen] = c
            before += _brier(pb, out)
            cc = max(0.02, min(0.97, _calib_conf(sig, conf)))
            after += _brier(_split(chosen, cc, br), out)
        n = len(data)
        print(f"  {sig:12s} n={n:5d}  before={before/n:.4f}  after={after/n:.4f}  uniform=0.6667")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals", default="ministral,qwen3",
                    help="comma-separated signal names to fit (default ministral,qwen3)")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--min-bin-count", type=int, default=20)
    ap.add_argument("--show-brier", action="store_true",
                    help="print before/after Brier on the directional joined set")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the map but do not write it")
    args = ap.parse_args(argv)

    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    calib_map = fit(signals, n_bins=args.bins, min_bin_count=args.min_bin_count)

    for sig in signals:
        e = calib_map["signals"].get(sig, {})
        print(f"{sig}: n={e.get('n', 0)} bins={e.get('bins')}")

    if args.show_brier:
        _brier_report(signals, calib_map)

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return 0

    atomic_write_json(str(_MAP_OUT), calib_map)
    print(f"\nWrote {_MAP_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
