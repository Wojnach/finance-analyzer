"""Mutation testing runner for finance-analyzer.

WHAT
----
Mutation testing modifies the code under test (e.g. flips ``<`` to ``<=``,
deletes a line, swaps a sign) and re-runs the test suite against each
mutated copy. If a mutated version still passes all tests, the test suite
is "weak" — it claims to verify the behaviour but doesn't actually catch
the bug the mutation introduced.

WHY THIS SCRIPT EXISTS
----------------------
Claude writes both the production code AND the tests for finance-analyzer.
That means a bug in the code can be papered over by an equally-confused
test, and the green CI tells nobody. Mutation testing breaks that loop:
the mutator is a third party with no opinion about what the code "should"
do, so a surviving mutant is unambiguous evidence of a weak test.

WHEN TO RUN
-----------
- During PR review of high-risk module changes (signal_engine,
  risk_management, portfolio_mgr) — manually, scoped to the changed
  module.
- Nightly via CI for the full pilot scope.
- See ``docs/TESTING.md`` ("Mutation Testing (mutmut)") for the
  threshold policy and follow-up triage commands.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

DEFAULT_MODULE = "portfolio/signal_engine.py"
DEFAULT_THRESHOLD_PCT = 50.0


def parse_results(stdout: str) -> tuple[int, int]:
    """Parse ``mutmut results`` output. Return (killed, survived).

    mutmut's text output groups mutants under headings like
    ``Survived (123)`` / ``Killed (456)`` / ``Timeout (...)``. The
    parenthesised count is what we want.
    """
    killed = 0
    survived = 0
    for line in stdout.splitlines():
        m = re.match(r"\s*(Killed|Survived)\s*\((\d+)\)", line, re.IGNORECASE)
        if not m:
            continue
        bucket = m.group(1).lower()
        count = int(m.group(2))
        if bucket == "killed":
            killed = count
        elif bucket == "survived":
            survived = count
    return killed, survived


def run_mutmut(module: str) -> int:
    """Run ``mutmut run`` against a single module path. Return exit code."""
    cmd = ["mutmut", "run", "--paths-to-mutate", module]
    print(f"[mutation] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def collect_results() -> tuple[int, int, str]:
    """Run ``mutmut results`` and parse counts. Return (killed, survived, raw)."""
    proc = subprocess.run(["mutmut", "results"], check=False, capture_output=True, text=True)
    killed, survived = parse_results(proc.stdout)
    return killed, survived, proc.stdout


def main() -> int:
    p = argparse.ArgumentParser(description="Run mutmut against a single module and gate on kill rate.")
    p.add_argument("--module", default=DEFAULT_MODULE,
                   help=f"Path to mutate (default: {DEFAULT_MODULE}). Use forward slashes.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT,
                   help=f"Minimum kill-rate percent to pass (default: {DEFAULT_THRESHOLD_PCT}).")
    p.add_argument("--skip-run", action="store_true",
                   help="Skip 'mutmut run' and only parse existing results (faster smoke check).")
    args = p.parse_args()

    if not args.skip_run:
        rc = run_mutmut(args.module)
        # mutmut returns non-zero when survivors exist; that's data, not a script error.
        print(f"[mutation] mutmut run exited rc={rc}", flush=True)

    killed, survived, raw = collect_results()
    total = killed + survived
    if total == 0:
        print("[mutation] no mutants reported — did mutmut run?")
        print(raw)
        return 1

    kill_rate = (killed / total) * 100.0
    print(f"Killed: {killed} | Survived: {survived} | "
          f"Kill rate: {kill_rate:.1f}% | Threshold: {args.threshold:.1f}%")

    if survived > 0:
        print(
            f"[weak-test] {survived} mutant(s) survived — mutmut changed the code "
            f"(e.g. flipped a comparison or deleted a line) and the tests STILL passed. "
            f"That means the tests don't actually verify the behaviour they claim to. "
            f"Inspect with: mutmut show <id>   then   mutmut apply <id>   to see the diff."
        )

    return 0 if kill_rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
