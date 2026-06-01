"""Short-horizon LLM/signal accuracy report — 3h / 4h / 12h directional hit rates.

WHY (2026-06-01): the daily `local_llm_report.py` is 1d-centric
(`accuracy_by_signal_ticker(..., horizon="1d")` hardcoded at line 237) and
`accuracy_cache.json` only surfaces 3h/1d/3d/5d/10d — even though
`outcome_tracker.HORIZONS` and `accuracy_stats.HORIZONS` both already list
4h and 12h. The intraday subsystems (metals warrant fishing, iskbets,
grid_fisher) trade on a minutes-to-hours horizon, so 1d accuracy is the
wrong yardstick for them. This script surfaces the sub-1d horizons that the
data already supports but nothing aggregates.

It is read-only: it calls `accuracy_by_signal_ticker` (signal_log + price
snapshots, which DO carry 3h/4h/12h outcomes) for each model × ticker ×
short horizon, prints a table, and writes a JSON snapshot. It does NOT touch
the live loop, the vote path, or any model.

Note the deliberate dual measure (see docs/LLM_FOLLOWUPS_20260518.md §"30pp
gap"): we report sample size alongside every accuracy so a 5-sample 100%
can't masquerade as signal. Numbers here are a fixed `--days` window (raw,
unweighted) — distinct from accuracy_cache's all-time + recency-blended view.

Usage:
    .venv/Scripts/python.exe scripts/short_horizon_llm_report.py
    .venv/Scripts/python.exe scripts/short_horizon_llm_report.py --days 14 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from portfolio.accuracy_stats import accuracy_by_signal_ticker  # noqa: E402
from portfolio.file_utils import atomic_write_json  # noqa: E402

# Short horizons that matter for the intraday subsystems, plus 1d as the
# reference the rest of the system already reports against.
SHORT_HORIZONS = ["3h", "4h", "12h", "1d"]

# Models / signals worth judging at intraday horizons. LLM voters first,
# then the disabled-but-tracked forecast/sentiment, then the active
# non-LLM intraday signals for context.
MODELS = [
    "ministral",
    "qwen3",
    "forecast",
    "sentiment",
    "news_event",
    "momentum",
    "mean_reversion",
    "rsi",
    "bb",
    # shadow finance models — present in accuracy_cache once they log
    # directional rows; harmless if they return empty.
    "finance_llama",
    "cryptotrader_lm",
    "meta_trader",
]


def _aggregate(per_ticker: dict) -> dict:
    """Collapse a per-ticker accuracy dict into one row.

    Sums `correct`/`samples` across tickers so the aggregate is sample-
    weighted, not a mean-of-means (which would over-weight thin tickers).
    """
    if not isinstance(per_ticker, dict) or not per_ticker:
        return {"accuracy": None, "samples": 0, "correct": 0}
    tot = sum((v.get("samples", 0) or 0) for v in per_ticker.values() if isinstance(v, dict))
    cor = sum((v.get("correct", 0) or 0) for v in per_ticker.values() if isinstance(v, dict))
    acc = round(100 * cor / tot, 1) if tot else None
    return {"accuracy": acc, "samples": tot, "correct": cor}


def build_report(days: int = 30) -> dict:
    report: dict = {"days": days, "horizons": SHORT_HORIZONS, "models": {}}
    for model in MODELS:
        row: dict = {"by_horizon": {}, "by_ticker": {}}
        for h in SHORT_HORIZONS:
            try:
                per_ticker = accuracy_by_signal_ticker(model, horizon=h, days=days)
            except Exception as e:  # one bad model must not abort the report
                row["by_horizon"][h] = {"error": f"{type(e).__name__}: {e}"}
                continue
            row["by_horizon"][h] = _aggregate(per_ticker)
            # Keep the per-ticker detail too — intraday metals care about
            # XAU/XAG specifically, which the aggregate can mask.
            row["by_ticker"][h] = {
                t: {"accuracy": round(100 * (v.get("correct", 0) or 0) / s, 1) if (s := v.get("samples", 0) or 0) else None,
                    "samples": s}
                for t, v in (per_ticker or {}).items()
                if isinstance(v, dict)
            }
        report["models"][model] = row
    return report


def _fmt(cell: dict) -> str:
    if not isinstance(cell, dict) or cell.get("error"):
        return "   —   "
    acc, n = cell.get("accuracy"), cell.get("samples", 0)
    if acc is None or not n:
        return "   —   "
    return f"{acc:5.1f}% n={n}"


def print_report(report: dict) -> None:
    hs = report["horizons"]
    print(f"\nShort-horizon directional accuracy ({report['days']}d window, sample-weighted)")
    print("=" * (22 + 14 * len(hs)))
    print(f"{'model':22s}" + "".join(f"{h:>14s}" for h in hs))
    print("-" * (22 + 14 * len(hs)))
    for model, row in report["models"].items():
        cells = "".join(f"{_fmt(row['by_horizon'].get(h, {})):>14s}" for h in hs)
        print(f"{model:22s}{cells}")
    print("=" * (22 + 14 * len(hs)))
    print("Reading: <47% = force-HOLD gate. Compare a model's short cols vs its 1d col —")
    print("monotonic decay toward 1d means the model only fits noise at short range.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    ap.add_argument("--json", default=None, help="optional path to write the JSON snapshot")
    args = ap.parse_args()

    report = build_report(days=args.days)
    print_report(report)

    out = Path(args.json) if args.json else (REPO / "data" / "short_horizon_llm_report.json")
    atomic_write_json(out, report)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
