"""Triage a full-suite run: bucket the failures, separate flakes from real ones.

Why this exists: on 2026-08-21 the suite reported 72 failures against a
`docs/TESTING.md` baseline claiming 24 — written when the suite had 7,730 tests
rather than 11,666. In that state nobody can tell a regression from noise, and
re-deriving it by hand costs a 3.5-minute run plus a pile of one-off greps.

The decisive step is the flake split. `docs/TESTING.md` documents that this
suite reports a *different* 5-10 failures per `-n auto` run from module-level
state leaking across xdist workers. So a failure only counts as real if it
survives a serial re-run of its own file. Everything else is an isolation
artifact.

Usage:
    # full cycle: run parallel, re-run failing files serially, report
    .venv/bin/python scripts/test_triage.py --run --confirm

    # triage a run you already have, no re-execution
    .venv/bin/python scripts/test_triage.py --input /tmp/fullsuite.txt

    # same, and rewrite the baseline block in docs/TESTING.md
    .venv/bin/python scripts/test_triage.py --run --confirm --update-baseline

On herc2 substitute `.venv/Scripts/python.exe`.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTING_DOC = _REPO_ROOT / "docs" / "TESTING.md"

BASELINE_START = "<!-- TRIAGE-BASELINE:START -->"
BASELINE_END = "<!-- TRIAGE-BASELINE:END -->"

# Ordered: first match wins, so put the specific signals above the file-level
# ones. An applicable-count test living in a metals file must not be buried in
# the metals bucket — it is the interesting case.
#
# `unknown` is the bucket that matters. Everything landing there is either a new
# regression or a gap in these rules; both want a human.
_RULES = (
    # (bucket, node-substring match)
    ("applicable-count", ("_total_applicable", "_vote_counts", "_applicable")),
    (
        "llm-infra",
        (
            "test_bert_sentiment",
            "test_llm_prewarmer",
            "test_llama_server",
            "test_model_upgrades",
            "test_local_llm_gate",
            "test_llm_batch",
            "test_chronos",
            "test_forecast_",
            "test_fingpt",
            "Ministral",
            "Qwen3",
            "test_llm_",
        ),
    ),
    ("metals-loop", ("test_metals", "test_fish_engine", "test_grid_")),
)
_UNKNOWN = "unknown"

_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_SUMMARY_RE = re.compile(
    r"(?:(?P<failed>\d+) failed)?[,\s]*"
    r"(?P<passed>\d+) passed"
    r"(?:[,\s]*(?P<skipped>\d+) skipped)?"
    r".*?in (?P<dur>[\d.]+)s"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_failures(text: str) -> list[str]:
    """Node ids from `FAILED`/`ERROR` lines, reason text stripped."""
    out, seen = [], set()
    for nodeid in _FAIL_RE.findall(text or ""):
        nodeid = nodeid.strip()
        if nodeid and nodeid not in seen:
            seen.add(nodeid)
            out.append(nodeid)
    return out


def parse_summary(text: str) -> dict:
    """Counts from pytest's trailing summary line. Empty dict if absent."""
    m = None
    for m in _SUMMARY_RE.finditer(text or ""):
        pass  # keep the LAST match — serial re-runs append their own summary
    if not m:
        return {}
    return {
        "failed": int(m.group("failed") or 0),
        "passed": int(m.group("passed")),
        "skipped": int(m.group("skipped") or 0),
        "duration_s": float(m.group("dur")),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(nodeid: str) -> str:
    for bucket, needles in _RULES:
        if any(n in nodeid for n in needles):
            return bucket
    return _UNKNOWN


def bucketize(nodeids) -> "OrderedDict[str, list[str]]":
    out = OrderedDict((b, []) for b, _ in _RULES)
    out[_UNKNOWN] = []
    for nodeid in nodeids:
        out[classify(nodeid)].append(nodeid)
    return out


def files_to_recheck(nodeids) -> list[str]:
    """Files owning the failures, first-seen order, deduped."""
    out, seen = [], set()
    for nodeid in nodeids:
        path = nodeid.split("::", 1)[0]
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def split_flakes(parallel, serial) -> dict:
    """Real = failed both ways. Anything that recovers serially is an xdist
    isolation artifact, per the clusters documented in docs/TESTING.md."""
    p, s = list(parallel), set(serial)
    real = [n for n in p if n in s]
    return {
        "real": real,
        "xdist_flake": [n for n in p if n not in s],
        "serial_only": [n for n in serial if n not in set(p)],
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _pytest(args, label):
    cmd = [sys.executable, "-m", "pytest", *args]
    print(f"[triage] {label}: {' '.join(cmd[2:])}", flush=True)
    r = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    return r.stdout + r.stderr


def run_parallel() -> str:
    """`-p no:randomly` and `--dist loadfile` so two runs are comparable and
    each file stays on one worker (kills the cheapest class of leak)."""
    return _pytest(
        [
            "tests/",
            "-n",
            "auto",
            "--dist",
            "loadfile",
            "-p",
            "no:randomly",
            "-q",
            "-rf",
        ],
        "full suite, parallel",
    )


def run_serial(files) -> str:
    if not files:
        return ""
    return _pytest(
        [*files, "-p", "no:randomly", "-q", "-rf"],
        f"serial re-run of {len(files)} file(s)",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_baseline(summary, split, buckets, host) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = summary.get("passed", 0) + summary.get("failed", 0)
    lines = [
        BASELINE_START,
        f"### Measured baseline — {ts}, host `{host}`",
        "",
        "Regenerate with `.venv/bin/python scripts/test_triage.py --run --confirm"
        " --update-baseline` (herc2: `.venv/Scripts/python.exe`). Do not hand-edit.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| collected | {total} |",
        f"| passed | {summary.get('passed', 0)} |",
        f"| failed (`-n auto`) | {summary.get('failed', 0)} |",
        f"| skipped | {summary.get('skipped', 0)} |",
        f"| parallel runtime | {summary.get('duration_s', 0):.0f}s |",
        f"| **real failures** (fail serially too) | **{len(split.get('real', []))}** |",
        f"| xdist isolation flakes (pass serially) | {len(split.get('xdist_flake', []))} |",
        f"| serial-only failures | {len(split.get('serial_only', []))} |",
        "",
        "Failures by bucket:",
        "",
        "| bucket | count |",
        "|---|---|",
    ]
    for bucket, items in buckets.items():
        lines.append(f"| {bucket} | {len(items)} |")
    real = split.get("real", [])
    lines += ["", f"**Real failures ({len(real)}):**", ""]
    lines += [f"- `{n}`" for n in real] if real else ["- none"]
    lines += [
        "",
        "`unknown` is the bucket that matters — a new entry there is either a "
        "regression or a gap in the classifier in `scripts/test_triage.py`.",
        BASELINE_END,
    ]
    return "\n".join(lines)


def replace_baseline(doc: str, block: str) -> str:
    """Swap the managed block, or insert after the first heading."""
    start, end = doc.find(BASELINE_START), doc.find(BASELINE_END)
    if start != -1 and end != -1:
        return doc[:start] + block + doc[end + len(BASELINE_END) :]
    lines = doc.splitlines()
    at = 1 if lines and lines[0].startswith("#") else 0
    return "\n".join(lines[:at] + ["", block, ""] + lines[at:]) + "\n"


def _print_report(summary, buckets, split):
    print("\n" + "=" * 66)
    print(
        f"collected {summary.get('passed', 0) + summary.get('failed', 0)}  "
        f"passed {summary.get('passed', 0)}  failed {summary.get('failed', 0)}  "
        f"skipped {summary.get('skipped', 0)}  in {summary.get('duration_s', 0):.0f}s"
    )
    print("=" * 66)
    for bucket, items in buckets.items():
        if items:
            print(f"  {bucket:18} {len(items)}")
    if split is not None:
        print("-" * 66)
        print(f"  REAL (fail serially too) : {len(split['real'])}")
        for n in split["real"]:
            print(f"      {n}")
        print(f"  xdist flake (pass alone) : {len(split['xdist_flake'])}")
        if split["serial_only"]:
            print(f"  serial-only              : {len(split['serial_only'])}")
            for n in split["serial_only"]:
                print(f"      {n}")
    unknown = buckets.get(_UNKNOWN) or []
    if unknown:
        print("-" * 66)
        print(f"  UNKNOWN bucket ({len(unknown)}) — regression or classifier gap:")
        for n in unknown:
            print(f"      {n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="run the full suite now")
    ap.add_argument("--input", help="triage an existing pytest output file instead")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="re-run failing files serially to separate flakes from real",
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help=f"rewrite the managed block in {_TESTING_DOC.name}",
    )
    args = ap.parse_args(argv)

    if args.run:
        text = run_parallel()
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    else:
        ap.error("pass --run or --input")

    failures = parse_failures(text)
    summary = parse_summary(text)
    buckets = bucketize(failures)

    split = None
    if args.confirm:
        serial_text = run_serial(files_to_recheck(failures))
        split = split_flakes(failures, parse_failures(serial_text))

    _print_report(summary, buckets, split)

    if args.update_baseline:
        if split is None:
            ap.error(
                "--update-baseline needs --confirm; a baseline without the "
                "flake split would record noise as real"
            )
        block = render_baseline(summary, split, buckets, platform.node())
        doc = (
            _TESTING_DOC.read_text(encoding="utf-8")
            if _TESTING_DOC.exists()
            else "# Testing Guide\n"
        )
        _TESTING_DOC.write_text(replace_baseline(doc, block), encoding="utf-8")
        print(f"\n[triage] baseline written to {_TESTING_DOC.relative_to(_REPO_ROOT)}")

    # Exit 1 only on real failures. Flakes must not red a CI gate.
    real = split["real"] if split else failures
    return 1 if real else 0


if __name__ == "__main__":
    raise SystemExit(main())
