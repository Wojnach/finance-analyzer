"""Dry-test the grid fisher pipeline against a live Avanza session.

Forces a synthetic signal that meets the placement gate and runs one
``GridFisher.tick()`` in PROBE_ONLY mode. The fisher does its normal
reconcile (hits live ``get_open_orders`` + ``get_positions``), evaluates
the placement decision, and writes ``probe_placement`` entries to
``data/grid_fisher_decisions.jsonl`` — but does NOT send orders to
Avanza.

Usage::

    .venv/Scripts/python.exe scripts/grid_fisher_probe.py
    .venv/Scripts/python.exe scripts/grid_fisher_probe.py --ticker XAU-USD --direction LONG --confidence 0.7

The script restores ``GRID_FISHER_PROBE_ONLY`` to its pre-run value on
exit so re-running production via metals_loop is unaffected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="XAG-USD",
                        choices=["XAG-USD", "XAU-USD", "OIL-USD"])
    parser.add_argument("--direction", default="LONG",
                        choices=["LONG", "SHORT"])
    parser.add_argument("--confidence", type=float, default=0.70,
                        help="Forced signal confidence (default 0.70)")
    parser.add_argument("--state-path", default="data/grid_fisher_probe_state.json",
                        help="Use a separate state file to avoid colliding "
                             "with the live grid fisher state.")
    parser.add_argument("--decisions-path",
                        default="data/grid_fisher_probe_decisions.jsonl")
    args = parser.parse_args()

    # Import only after argparse so --help is fast.
    from portfolio import grid_fisher_config
    from portfolio.grid_fisher import GridFisher
    from portfolio import avanza_session
    from data.fin_fish_config import WARRANT_CATALOG

    prev_probe = grid_fisher_config.GRID_FISHER_PROBE_ONLY
    grid_fisher_config.GRID_FISHER_PROBE_ONLY = True
    try:
        fisher = GridFisher(
            session=avanza_session,
            catalog=WARRANT_CATALOG,
            state_path=args.state_path,
            decisions_path=args.decisions_path,
        )
        # Snap _probe_only off the freshly-imported flag (init read it).
        fisher._probe_only = True

        signal_data = {
            args.ticker: {
                "direction": args.direction,
                "confidence": args.confidence,
            },
        }
        print(f"[probe] tick: ticker={args.ticker} "
              f"direction={args.direction} confidence={args.confidence}")
        t0 = time.time()
        report = fisher.tick(signal_data=signal_data)
        elapsed = time.time() - t0
        print(f"[probe] tick complete in {elapsed:.2f}s")
        print(json.dumps(report, indent=2, default=str))

        # Tail the decisions log to show the probe_placement entries.
        decisions = Path(args.decisions_path)
        if decisions.exists():
            print("\n[probe] last 10 decision-log entries:")
            with decisions.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-10:]:
                try:
                    print("  " + json.dumps(json.loads(line)))
                except json.JSONDecodeError:
                    print("  " + line.rstrip())
        return 0
    finally:
        grid_fisher_config.GRID_FISHER_PROBE_ONLY = prev_probe


if __name__ == "__main__":
    sys.exit(main())
