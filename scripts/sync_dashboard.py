#!/usr/bin/env python3
"""Sync finance-analyzer dashboard data to raanman.lol/bets for GitHub Pages.

Fetches all dashboard API endpoints from the running Flask app (localhost:5055),
saves responses as static JSON files in the raanmanlol repo's bets/data/ directory,
then commits and pushes changes to GitHub only when data has changed.

Usage:
    python scripts/sync_dashboard.py           # Single sync
    python scripts/sync_dashboard.py --loop     # Continuous sync every INTERVAL seconds
    python scripts/sync_dashboard.py --loop --interval 39  # Every 39 seconds (default)
"""

import argparse
import hashlib
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

# raanmanlol repo clone (Wojnach/raanmanlol → master branch → GitHub Pages)
RAANMAN_REPO = Path("Q:/raanmanlol-repo")
BETS_DATA_DIR = RAANMAN_REPO / "bets" / "data"

# Dashboard token (read from config.json if set)
CONFIG_PATH = FINANCE_DIR / "config.json"

# Map: output filename -> API endpoint URL
ENDPOINTS = {
    "summary.json":            "/api/summary",
    "signal-heatmap.json":     "/api/signal-heatmap",
    "equity-curve.json":       "/api/equity-curve",
    "trades.json":             "/api/trades",
    "triggers.json":           "/api/triggers",
    "telegrams.json":          "/api/telegrams?limit=500",
    "signal-log.json":         "/api/signal-log",
    "accuracy.json":           "/api/accuracy",
    "accuracy-history.json":   "/api/accuracy-history",
    "decisions.json":          "/api/decisions?limit=200",
    "health.json":             "/api/health",
    "invocations.json":        "/api/invocations",
    "metals.json":             "/api/metals",
    "metals-accuracy.json":    "/api/metals-accuracy",
    "warrants.json":           "/api/warrants",
    "risk.json":               "/api/risk",
}

DEFAULT_INTERVAL = 39  # seconds

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
# Content hashing for diff detection
# ---------------------------------------------------------------------------

_content_hashes = {}  # {filename: md5_hex} — tracks last-written content


def _hash_content(data_bytes):
    """Return MD5 hex digest of content bytes."""
    return hashlib.md5(data_bytes).hexdigest()


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
    """Fetch all API endpoints, write only files that changed.

    Returns (ok_count, fail_count, changed_count).
    """
    token = _get_token()
    ok_count = 0
    fail_count = 0
    changed_count = 0

    BETS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, endpoint in ENDPOINTS.items():
        url = f"{API_BASE}{endpoint}"
        if token:
            sep = "&" if "?" in url else "?"
            url += f"{sep}token={token}"
        try:
            r = requests.get(url, timeout=30)
            if r.ok:
                data = r.json()
                content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                content_bytes = content.encode("utf-8")
                new_hash = _hash_content(content_bytes)

                # Only write if content actually changed
                old_hash = _content_hashes.get(filename)
                if old_hash != new_hash:
                    out_path = BETS_DATA_DIR / filename
                    out_path.write_bytes(content_bytes)
                    _content_hashes[filename] = new_hash
                    changed_count += 1

                ok_count += 1
            else:
                log.warning("HTTP %d for %s", r.status_code, endpoint)
                fail_count += 1
        except requests.ConnectionError:
            log.error("Cannot connect to %s — is the dashboard running?", API_BASE)
            return 0, len(ENDPOINTS), 0
        except Exception as exc:
            log.warning("Error fetching %s: %s", endpoint, exc)
            fail_count += 1

    # Sync dashboard HTML (patch static path: ./api-data/ → ./data/)
    src_html = FINANCE_DIR / "dashboard" / "static" / "index.html"
    dst_html = RAANMAN_REPO / "bets" / "index.html"
    if src_html.exists():
        html = src_html.read_text(encoding="utf-8")
        html = html.replace("./api-data/", "./data/")
        html_bytes = html.encode("utf-8")
        html_hash = _hash_content(html_bytes)
        if _content_hashes.get("index.html") != html_hash:
            dst_html.write_bytes(html_bytes)
            _content_hashes["index.html"] = html_hash
            changed_count += 1
            log.info("Updated index.html")

    # Write sync metadata (always updated)
    meta = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "endpoints_ok": ok_count,
        "endpoints_failed": fail_count,
        "endpoints_changed": changed_count,
        "endpoints_total": len(ENDPOINTS),
    }
    (BETS_DATA_DIR / "sync_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return ok_count, fail_count, changed_count


def _git_sync():
    """Commit and push data changes to the raanmanlol repo (master branch)."""
    if not RAANMAN_REPO.exists():
        log.error("raanmanlol repo not found at %s", RAANMAN_REPO)
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

    # 0. Ensure git identity is configured (required for commits)
    result = _run(["git", "config", "user.name"])
    if result.returncode != 0 or not result.stdout.strip():
        _run(["git", "config", "user.name", "wojnach"])
        _run(["git", "config", "user.email", "wojnach@users.noreply.github.com"])

    # 1. Stage data files
    _run(["git", "add", "bets/data/", "bets/index.html"])

    # 2. Check if anything changed in git
    result = _run(["git", "diff", "--cached", "--quiet"])
    if result.returncode == 0:
        log.info("No git diff — skipping commit")
        return True

    # 3. Commit locally
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result = _run(["git", "commit", "-m", f"sync: dashboard data {ts}"])
    if result.returncode != 0:
        log.error("git commit failed: %s", result.stderr.strip())
        return False

    # 4. Stash unstaged changes, pull --rebase, pop stash, then push
    _run(["git", "stash", "--include-untracked"])
    pull = _run(["git", "pull", "--rebase", "origin", "master"])
    _run(["git", "stash", "pop"])
    if pull.returncode != 0:
        log.warning("git pull --rebase failed: %s (attempting push anyway)",
                    pull.stderr.strip())
    result = _run(["git", "push", "origin", "master"])
    if result.returncode != 0:
        log.error("git push failed: %s", result.stderr.strip())
        return False

    log.info("Pushed data update to raanmanlol master")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sync_once():
    """Run a single sync cycle."""
    log.info("Fetching dashboard data from %s ...", API_BASE)
    ok, fail, changed = _fetch_all()

    if ok == 0:
        log.error("All endpoints failed — skipping git sync")
        return False

    if changed == 0:
        log.info("Fetched %d/%d endpoints — no data changed, skipping git sync", ok, ok + fail)
        return True

    log.info("Fetched %d/%d endpoints, %d changed — syncing to git", ok, ok + fail, changed)

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

    log.info("raanmanlol repo:   %s", RAANMAN_REPO)
    log.info("Data output:       %s", BETS_DATA_DIR)
    log.info("Endpoints:    %d", len(ENDPOINTS))

    if args.loop:
        log.info("Starting continuous sync every %ds (diff-based)", args.interval)
        while True:
            try:
                sync_once()
            except Exception as exc:
                log.error("Sync cycle failed: %s", exc)
            time.sleep(args.interval)
    else:
        success = sync_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
