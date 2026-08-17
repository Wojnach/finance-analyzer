#!/usr/bin/env python3
"""Check whether the credentials leaked in commit 338b6000 are still live.

config.json was committed on 2026-03-15 and untracked 83 minutes later.
Untracking does not remove a blob from history: as of 2026-08-17 the file was
still fetchable unauthenticated (HTTP 200, 6547 bytes) and ALL ELEVEN
credentials in it were byte-identical to the ones in the live config.

Run this after rotating to confirm each key actually changed:

    .venv/bin/python scripts/check_leaked_creds_rotated.py

Exit 0 when every leaked credential has been rotated (or the blob is no longer
reachable and cannot be compared), 1 while any still match.

No secret is printed or stored — only a short SHA-256 prefix, and only to show
that two values differ. The leaked blob is fetched at runtime rather than
vendored so this script never becomes a second copy of the leak.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

LEAK_URL = (
    "https://raw.githubusercontent.com/Wojnach/finance-analyzer/" "338b6000/config.json"
)

# Rotation order matters: most damaging first. Binance/Alpaca are used
# read-only by this codebase, but key PERMISSIONS are set provider-side — a
# leaked key still matters if trading is enabled on it. The Telegram token is
# not read-only at all: it grants full bot control.
CREDENTIALS = [
    ("exchange.key", "Binance API key"),
    ("exchange.secret", "Binance API secret"),
    ("alpaca.key", "Alpaca API key"),
    ("alpaca.secret", "Alpaca API secret"),
    ("telegram.token", "Telegram bot token (FULL bot control — not read-only)"),
    ("api_server.jwt_secret_key", "api_server JWT secret"),
    ("api_server.ws_token", "api_server websocket token"),
    ("api_server.password", "api_server password"),
    ("newsapi_key", "NewsAPI key"),
    ("alpha_vantage.api_key", "Alpha Vantage key"),
    ("golddigger.fred_api_key", "FRED key"),
]


def _dig(blob: dict, dotted: str):
    cur = blob
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, str) else None


def _short(value: str | None) -> str:
    if not value:
        return "-"
    return hashlib.sha256(value.encode()).hexdigest()[:10]


def main() -> int:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(LEAK_URL, timeout=15) as resp:
            leaked = json.loads(resp.read().decode())
        reachable = True
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            print(
                f"Leaked blob no longer reachable (HTTP {exc.code}) — repo is "
                "private or history was rewritten.\n"
                "That limits FURTHER exposure but does not undo the leak: "
                "anything already cloned, forked or scraped is still out.\n"
                "Rotate anyway, then re-run against a saved copy if you kept one."
            )
            return 0
        print(f"Could not fetch the leaked blob: HTTP {exc.code}")
        return 0
    except Exception as exc:  # network down, DNS, etc.
        print(
            f"Could not fetch the leaked blob ({type(exc).__name__}) — cannot compare."
        )
        return 0

    cfg_path = Path("config.json")
    if not cfg_path.exists():
        print("config.json not found — run from the repo root.")
        return 0
    live = json.loads(cfg_path.read_text())

    print(f"Leaked blob is PUBLICLY FETCHABLE at {LEAK_URL}\n")
    print(f"{'credential':28} {'leaked':11}{'live':11} status")
    unrotated = []
    for dotted, label in CREDENTIALS:
        lk, lv = _dig(leaked, dotted), _dig(live, dotted)
        if lk is None:
            status = "not in leak"
        elif lv is None:
            status = "absent from live config"
        elif lk == lv:
            status = "STILL LIVE — ROTATE"
            unrotated.append(label)
        else:
            status = "rotated"
        print(f"{dotted:28} {_short(lk):11}{_short(lv):11}{status}")

    print()
    if unrotated:
        print(
            f"{len(unrotated)} of {len(CREDENTIALS)} still live. Rotate in this order:"
        )
        for i, label in enumerate(unrotated, 1):
            print(f"  {i}. {label}")
        print(
            "\nMaking the repo private does NOT fix this — rotation is the only "
            "real remedy."
        )
        return 1

    print("All leaked credentials have been rotated.")
    if reachable:
        print(
            "The blob is still public though — consider making the repo private "
            "so the next accident is not immediately world-readable."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
