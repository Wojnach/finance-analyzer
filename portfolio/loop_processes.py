"""Loop-process enumeration for the dashboard duplicate-detection tile.

Backs `/api/loop-processes`. Enumerates running Python (and a couple of
PowerShell) processes via psutil, matches each against the known
finance-analyzer loop signatures (substring on the full command line),
and returns:

    {
      "loops": [
        {
          "name": "main",
          "pattern": "portfolio\\\\main.py --loop",
          "pids": [12345],
          "count": 1,
          "duplicate": false,
          "process_started_at": ["2026-05-17T18:42:11+00:00"],
          "oldest_uptime_seconds": 4521,
        },
        ...
      ],
      "any_duplicate": true,
      "checked_at": "2026-05-18T07:11:02+00:00",
    }

Why a dedicated module instead of inlining in dashboard/app.py:
- Pure-Python, no Flask import — trivially unit-testable with a
  monkeypatched psutil.process_iter().
- Same shape consumed by the dashboard JS tile and a future Telegram
  alerter (premortem N5: if the tile turns red and the user dismisses
  it twice, the watchdog could escalate).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import psutil

logger = logging.getLogger(__name__)


# Known loop signatures. Key = stable loop name (used in JSON keys, JS
# labels, Telegram alerts). Value = substring matched against the full
# command line (case-insensitive, OS-path-separator-agnostic — we
# normalise both sides to forward slashes before comparing).
#
# Patterns are kept conservative: a substring match, not a regex, so
# adding a new loop is one line and an obvious change. If two loops
# need to share words, append the distinguishing token (e.g.
# "data/crypto_loop.py" not just "crypto_loop").
KNOWN_LOOPS: dict[str, tuple[str, ...]] = {
    # main runs as a bare script on Windows (`portfolio\main.py --loop`)
    # but as a module on the Deck systemd unit (`-m portfolio.main --loop`,
    # pf-dataloop.service) — both forms need a pattern (2026-07-18: the
    # module form never matched, so the tile showed the live Deck loop as
    # grey/count=0 while pf-dataloop was running).
    "main":           ("portfolio/main.py --loop", "portfolio.main --loop"),
    "metals":         ("data/metals_loop.py",),
    "crypto":         ("data/crypto_loop.py",),
    "oil":            ("data/oil_loop.py",),
    "mstr":           ("portfolio.mstr_loop",),
    "golddigger":     ("portfolio.golddigger",),
    "silver_monitor": ("data/silver_monitor.py",),
    # dashboard launches via module form on Windows (`-m dashboard.app`)
    # but as a bare script on the Deck systemd unit (`dashboard/app.py`,
    # pf-dashboard.service) — same both-forms fix as `main` above
    # (2026-07-18: the module-only pattern never matched the Deck process).
    "dashboard":      ("dashboard.app", "dashboard/app.py"),
    # hw monitoring is a PowerShell script (PF-HWMonitor → read_temps.ps1),
    # not a python module — the old "hw_monitor.py" never existed as a
    # process (2026-06-06). Windows-only: no equivalent unit/process
    # exists on the Deck, so count=0 there is correct, not a bug
    # (verified 2026-07-18 — no systemd unit, no running process).
    "hw_monitor":     ("data/read_temps.ps1",),
    # fix_agent dispatcher: never installed in production (see CLAUDE.md
    # 2026-06-10 audit) and `data/fix_agent.disabled` flag is set — count=0
    # is correct on the Deck today, not a scanner bug (verified 2026-07-18).
    "fix_agent":      ("scripts/fix_agent_dispatcher.py",),
    # NOTE: telegram_poller is NOT process-checked — it runs as a daemon
    # thread inside the main loop (portfolio/main.py → TelegramPoller),
    # never a standalone process, so it would render a permanent false
    # grey row. Its liveness is implied by `main` being green. Removed
    # 2026-06-06.
    #
    # NOTE: log_rotation is NOT process-checked (removed 2026-07-18, same
    # reasoning as telegram_poller above) — `rotate_all()` runs inline
    # inside the main loop's cycle (portfolio/main.py ~L407), never as a
    # standalone process on either OS, so the old entry could never show
    # anything but a permanent false grey row.
}


def _normalise_cmdline(parts: list[str] | None) -> str:
    """Join argv into a forward-slash, lowercase haystack.

    psutil returns argv as a list; on Windows that's typically
    `["C:\\…\\python.exe", "-u", "Q:\\finance-analyzer\\portfolio\\main.py", "--loop"]`.
    Joining + normalising lets every pattern in KNOWN_LOOPS use forward
    slashes regardless of the actual OS-quoted form.
    """
    if not parts:
        return ""
    joined = " ".join(parts)
    return joined.replace("\\", "/").lower()


def _iter_processes() -> list[dict[str, Any]]:
    """Enumerate processes once with the fields we need.

    Wrapping psutil.process_iter() keeps the test surface tiny: tests
    monkey-patch this and feed in a list of fake process dicts.
    """
    out: list[dict[str, Any]] = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time", "name"]):
        try:
            info = p.info
            out.append(
                {
                    "pid": int(info.get("pid") or 0),
                    "ppid": int(info.get("ppid") or 0),
                    "name": info.get("name") or "",
                    "cmdline": info.get("cmdline") or [],
                    "create_time": float(info.get("create_time") or 0.0),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            # Process disappeared between iter and info access, or we
            # lack permission (e.g. SYSTEM processes). Skip silently;
            # this is exactly the contract psutil documents.
            logger.debug("loop_processes: skip pid=%s err=%s", info.get("pid"), exc)
    return out


def scan(now_utc: datetime | None = None) -> dict[str, Any]:
    """Public entry point. Returns the payload shape documented at top.

    `now_utc` is for testability — production calls omit it.
    """
    if now_utc is None:
        now_utc = datetime.now(UTC)
    now_ts = now_utc.timestamp()

    procs = _iter_processes()

    # OWN-PROCESS GUARD: omit the dashboard's own self-match. Without
    # this the dashboard always reports `dashboard: count=1` (us) plus
    # whatever real dashboard exists, mis-flagging as duplicate.
    own_pid = os.getpid()

    loops: list[dict[str, Any]] = []
    for name, patterns in KNOWN_LOOPS.items():
        pats = tuple(p.replace("\\", "/").lower() for p in patterns)
        matches = []
        for proc in procs:
            if proc["pid"] == own_pid:
                continue
            hay = _normalise_cmdline(proc["cmdline"])
            if not hay:
                continue
            if any(pat in hay for pat in pats):
                matches.append(proc)

        # VENV-LAUNCHER COLLAPSE (added 2026-06-06): on Windows the
        # .venv\Scripts\python.exe shim re-execs the base interpreter, so
        # the same loop script shows up in BOTH the shim (parent) and its
        # child with identical argv tails — every loop on the live box
        # was false-flagged as duplicate 2026-06-05. Drop any matched
        # proc that is the parent of another matched proc; the surviving
        # leaf interpreter is the real loop. A genuine duplicate (two
        # independent pairs) leaves two leaves → still flagged.
        matched_pids = {m["pid"] for m in matches}
        parent_pids = {m["ppid"] for m in matches if m["ppid"] in matched_pids}
        matches = [m for m in matches if m["pid"] not in parent_pids]

        pids = [m["pid"] for m in matches]
        starts = [
            datetime.fromtimestamp(m["create_time"], tz=UTC).isoformat()
            if m["create_time"] > 0 else None
            for m in matches
        ]
        # Compute oldest uptime defensively: matches may all have
        # create_time=0 if psutil couldn't read the field (sandbox /
        # AccessDenied edge); in that case min() on an empty generator
        # raises ValueError and crashes the endpoint.
        valid_starts = [m["create_time"] for m in matches if m["create_time"] > 0]
        if valid_starts:
            oldest = min(valid_starts)
            uptime = max(0, int(now_ts - oldest))
        else:
            uptime = None

        loops.append(
            {
                "name": name,
                "pattern": " | ".join(patterns),
                "pids": pids,
                "count": len(pids),
                "duplicate": len(pids) > 1,
                "process_started_at": starts,
                "oldest_uptime_seconds": uptime,
            }
        )

    any_dup = any(L["duplicate"] for L in loops)
    return {
        "loops": loops,
        "any_duplicate": any_dup,
        "checked_at": now_utc.isoformat(),
    }
