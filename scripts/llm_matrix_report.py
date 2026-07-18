#!/usr/bin/env python3
"""Model-comparison matrix from backtest jsonl files.

Merges any number of llm_backtest-format jsonl files (LLM, kronos, xgboost
rows share one schema) and prints per (model, interval, ticker, horizon):
votes n, directional accuracy, Wilson 95% CI lower bound, abstain rate.

Usage:
    python scripts/llm_matrix_report.py FILE [FILE ...] [--min-votes 10]
        [--horizon-match]  (score each model only on its own-name horizon,
                            for kronos-<h>h / xgboost-<h>h style rows)
        [--csv OUT.csv]
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def wilson_low(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def horizon_fields(row):
    return [k for k in row if re.fullmatch(r"outcome_[0-9.]+h_pct", k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-votes", type=int, default=10)
    ap.add_argument("--horizon-match", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = []
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"skip missing {f}", file=sys.stderr)
            continue
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        sys.exit("no rows")

    # cells[(model, interval, ticker, horizon)] = [hits, votes, holds, abstains, total]
    cells = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        model, interval, ticker = r["model"], r.get("interval", "?"), r["ticker"]
        for hf in horizon_fields(r):
            h = hf[len("outcome_") : -len("_pct")].rstrip("h")
            if args.horizon_match:
                m = re.search(r"-([0-9.]+)h$", model)
                if m and m.group(1) != h:
                    continue
            key = (model, interval, ticker, h)
            c = cells[key]
            c[4] += 1
            vote, out = r["vote"], r.get(hf)
            if vote == "HOLD":
                c[2] += 1
            elif vote == "ABSTAIN" or vote == "ERROR":
                c[3] += 1
            elif out is not None:
                c[1] += 1
                if (out > 0) == (vote == "BUY"):
                    c[0] += 1

    def hkey(h):
        return float(h)

    out_rows = []
    for (model, interval, ticker, h), (hits, votes, holds, abst, total) in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], hkey(kv[0][3]))
    ):
        if votes < args.min_votes:
            acc = ci = None
        else:
            acc = 100 * hits / votes
            ci = 100 * wilson_low(hits, votes)
        out_rows.append(
            {
                "model": model,
                "interval": interval,
                "ticker": ticker,
                "horizon_h": h,
                "votes": votes,
                "acc_pct": None if acc is None else round(acc, 1),
                "ci_low_pct": None if ci is None else round(ci, 1),
                "hold_pct": round(100 * holds / total, 1) if total else None,
                "abstain_pct": round(100 * abst / total, 1) if total else None,
                "n_rows": total,
            }
        )

    hdr = f"{'model':<26} {'iv':<4} {'ticker':<8} {'hor':>6} {'votes':>6} {'acc%':>6} {'CIlow':>6} {'hold%':>6} {'abst%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        acc = "-" if r["acc_pct"] is None else f"{r['acc_pct']:.1f}"
        ci = "-" if r["ci_low_pct"] is None else f"{r['ci_low_pct']:.1f}"
        star = " *" if r["ci_low_pct"] is not None and r["ci_low_pct"] >= 60 else ""
        print(
            f"{r['model']:<26} {r['interval']:<4} {r['ticker']:<8} {r['horizon_h']:>5}h "
            f"{r['votes']:>6} {acc:>6} {ci:>6} {r['hold_pct']:>6} {r['abstain_pct']:>6}{star}"
        )
    print("\n* = Wilson CI lower bound clears the 60% keep/kill bar")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"csv written: {args.csv}")


if __name__ == "__main__":
    main()
