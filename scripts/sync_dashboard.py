#!/usr/bin/env python3
"""Sync finance-analyzer dashboard data to raanman.lol/bets for GitHub Pages.

Fetches all dashboard API endpoints from the running Flask app (localhost:5055),
saves responses as static JSON files in the raanmanlol repo's bets/data/ directory,
then commits and pushes changes to GitHub.

Usage:
    python scripts/sync_dashboard.py           # Single sync
    python scripts/sync_dashboard.py --loop     # Continuous sync every INTERVAL seconds
    python scripts/sync_dashboard.py --loop --interval 600  # Every 10 minutes
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:5055"

# Auto-detect repo paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
FINANCE_DIR = SCRIPT_DIR.parent
RAANMAN_REPO = FINANCE_DIR.parent / "raanmanlol-repo"
BETS_DATA_DIR = RAANMAN_REPO / "bets" / "data"

# Dashboard token (read from config.json if set)
CONFIG_PATH = FINANCE_DIR / "config.json"

# Map: output filename -> API endpoint URL
ENDPOINTS = {
    "summary.json":          "/api/summary",
    "signal-heatmap.json":   "/api/signal-heatmap",
    "equity-curve.json":     "/api/equity-curve",
    "trades.json":           "/api/trades",
    "triggers.json":         "/api/triggers",
    "telegrams.json":        "/api/telegrams?limit=500",
    "signal-log.json":       "/api/signal-log",
    "accuracy.json":         "/api/accuracy",
    "accuracy-history.json": "/api/accuracy-history",
    "decisions.json":        "/api/decisions?limit=200",
    "health.json":           "/api/health",
    "invocations.json":      "/api/invocations",
}

DEFAULT_INTERVAL = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sync] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sync_dashboard")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_token():
    """Read dashboard_token from config.json (if configured)."""
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return cfg.get("dashboard_token") or None
    except Exception:
        return None


def _fetch_all():
    """Fetch all API endpoints and write JSON files to bets/data/."""
    token = _get_token()
    params = {"token": token} if token else {}
    ok_count = 0
    fail_count = 0

    BETS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, endpoint in ENDPOINTS.items():
        url = f"{API_BASE}{endpoint}"
        # Append token to URL if it already has query params
        if token:
            sep = "&" if "?" in url else "?"
            url += f"{sep}token={token}"
        try:
            r = requests.get(url, timeout=30)
            if r.ok:
                data = r.json()
                out_path = BETS_DATA_DIR / filename
                out_path.write_text(
                    json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                ok_count += 1
            else:
                log.warning("HTTP %d for %s", r.status_code, endpoint)
                fail_count += 1
        except requests.ConnectionError:
            log.error("Cannot connect to %s — is the dashboard running?", API_BASE)
            return 0, len(ENDPOINTS)
        except Exception as exc:
            log.warning("Error fetching %s: %s", endpoint, exc)
            fail_count += 1

    # Write sync metadata
    meta = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "endpoints_ok": ok_count,
        "endpoints_failed": fail_count,
    }
    (BETS_DATA_DIR / "sync_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return ok_count, fail_count


def _git_sync():
    """Commit and push data changes to the raanmanlol repo."""
    if not RAANMAN_REPO.exists():
        log.error("Raanman repo not found at %s", RAANMAN_REPO)
        return False

    def _run(cmd, **kwargs):
        return subprocess.run(
            cmd,
            cwd=str(RAANMAN_REPO),
            capture_output=True,
            text=True,
            timeout=60,
            **kwargs,
        )

    # 1. Stage data files first (before pull, to avoid unstaged changes error)
    _run(["git", "add", "bets/data/"])

    # 2. Check if anything changed
    result = _run(["git", "diff", "--cached", "--quiet"])
    if result.returncode == 0:
        log.info("No data changes — skipping commit")
        return True

    # 3. Commit locally
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _run(["git", "commit", "-m", f"sync: dashboard data {ts}"])

    # 4. Stash any leftover unstaged changes, pull, then drop stash
    _run(["git", "stash", "--include-untracked"])
    result = _run(["git", "pull", "--rebase", "origin", "master"])
    _run(["git", "stash", "pop"])  # restore stashed changes (ok if nothing stashed)
    if result.returncode != 0:
        _run(["git", "rebase", "--abort"])
        log.error("git pull --rebase failed: %s", result.stderr.strip())
        return False

    # 5. Push
    result = _run(["git", "push", "origin", "master"])
    if result.returncode != 0:
        log.error("git push failed: %s", result.stderr.strip())
        return False

    log.info("Pushed data update to GitHub")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sync_once():
    """Run a single sync cycle."""
    log.info("Fetching dashboard data from %s ...", API_BASE)
    ok, fail = _fetch_all()

    if ok == 0:
        log.error("All endpoints failed — skipping git sync")
        return False

    log.info("Fetched %d/%d endpoints successfully", ok, ok + fail)

    if not _git_sync():
        log.error("Git sync failed")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Sync dashboard data to GitHub Pages")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between syncs (default: {DEFAULT_INTERVAL})",
    )
    args = parser.parse_args()

    log.info("Raanman repo: %s", RAANMAN_REPO)
    log.info("Data output:  %s", BETS_DATA_DIR)

    if args.loop:
        log.info("Starting continuous sync every %ds", args.interval)
        while True:
            try:
                sync_once()
            except Exception as exc:
                log.error("Sync cycle failed: %s", exc)
            log.info("Next sync in %ds...", args.interval)
            time.sleep(args.interval)
    else:
        success = sync_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
