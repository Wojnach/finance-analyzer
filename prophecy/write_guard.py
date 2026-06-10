"""Filesystem integrity guard around the prophecy headless claude run.

2026-06-11 (review of 68546e7d): defense-in-depth against prompt-injection
writes. The claude invocation in scripts/prophecy-daily.bat is permission-
restricted (Write scoped to data/prophecy_runs/**), but permission-rule
semantics for CLI-passed path patterns cannot be runtime-verified without
spending tokens — this guard catches escaped writes REGARDLESS of how the
permission layer behaves.

Mechanism: snapshot ``git status --porcelain`` immediately before the claude
call (``pre``), re-run it immediately after (``check``) and treat any NEW
status line outside the allowed prefixes as a breach: rate-limited critical
(category ``prophecy_write_breach``) + exit 2, which the .bat turns into
"do NOT publish, exit non-zero".

Known limitations (accepted, documented):
- ``data/`` is excluded wholesale: the main loop + metals loop write tracked
  data files every 60s, so any multi-minute claude run would false-positive
  daily. The guard therefore polices CODE/CONFIG (portfolio/, prophecy/,
  scripts/, dashboard/, .claude/, docs/, tests/, root files); data/ writes
  remain the permission layer's job.
- A file already dirty at ``pre`` that gets dirtier during the run produces
  an identical porcelain line and is not detected (porcelain carries no
  content hash). Pre-run-clean files are fully covered.
- gitignored paths are invisible to git status; same data/-class caveat.

Run:  python -m prophecy.write_guard pre   <snapshot_file>
      python -m prophecy.write_guard check <snapshot_file>
Exit: pre   -> 0 ok | 1 error (caller should NOT start the claude run)
      check -> 0 clean | 1 internal error | 2 breach (caller must quarantine)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from prophecy import config as pcfg
from prophecy.alerts import log_critical

logger = logging.getLogger("prophecy.write_guard")

# New status lines whose path starts with one of these prefixes are expected
# runtime churn and never count as a breach. Forward slashes: git emits them
# even on Windows.
_ALLOWED_PREFIXES = ("data/",)


def _git_status_lines() -> list[str]:
    """Raw ``git status --porcelain`` lines for the repo. Raises on failure."""
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(pcfg.REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git status failed rc={out.returncode}: {out.stderr.strip()[:300]}")
    return [line for line in out.stdout.splitlines() if line.strip()]


def _line_path(line: str) -> str:
    """Path part of a porcelain line (rename: the NEW path; quotes stripped)."""
    body = line[3:] if len(line) > 3 else line
    if " -> " in body:
        body = body.split(" -> ", 1)[1]
    body = body.strip()
    if body.startswith('"') and body.endswith('"') and len(body) >= 2:
        body = body[1:-1]
    return body


def snapshot(snap_path: str) -> int:
    try:
        lines = _git_status_lines()
        Path(snap_path).write_text("\n".join(lines) + ("\n" if lines else ""),
                                   encoding="utf-8")
        print(f"write_guard: snapshot {len(lines)} dirty paths -> {snap_path}")
        return 0
    except Exception as exc:
        log_critical(
            "prophecy_write_guard_error",
            f"pre-run git snapshot failed ({exc!r}) — claude run should be "
            f"skipped (guard cannot verify integrity)",
            caller="write_guard.pre",
            context={"error": repr(exc), "snapshot": snap_path},
        )
        return 1


def check(snap_path: str) -> int:
    try:
        before = set(Path(snap_path).read_text(encoding="utf-8").splitlines())
        after = _git_status_lines()
    except Exception as exc:
        log_critical(
            "prophecy_write_guard_error",
            f"post-run integrity check could not run ({exc!r}) — quarantining "
            f"the run (publish skipped)",
            caller="write_guard.check",
            context={"error": repr(exc), "snapshot": snap_path},
        )
        return 1

    suspicious = [
        line for line in after
        if line not in before
        and not _line_path(line).startswith(_ALLOWED_PREFIXES)
    ]
    if suspicious:
        log_critical(
            "prophecy_write_breach",
            f"claude run modified/created {len(suspicious)} path(s) outside "
            f"data/ — possible prompt-injection write; run quarantined, "
            f"publish skipped. Paths: {[_line_path(s) for s in suspicious[:20]]}",
            caller="write_guard.check",
            context={"suspicious": suspicious[:50], "snapshot": snap_path},
        )
        print("write_guard: BREACH — new/modified paths outside data/:")
        for line in suspicious[:20]:
            print(f"  {line}")
        return 2

    print(f"write_guard: clean ({len(after)} dirty paths, all pre-existing or under data/)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Prophecy claude-run write guard (zero tokens)")
    ap.add_argument("mode", choices=["pre", "check"])
    ap.add_argument("snapshot_file")
    args = ap.parse_args(argv)
    return snapshot(args.snapshot_file) if args.mode == "pre" else check(args.snapshot_file)


if __name__ == "__main__":
    sys.exit(main())
