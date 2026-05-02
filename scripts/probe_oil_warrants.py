"""One-shot Avanza probe for the oil warrant catalog.

Opens a headless Playwright session against the existing
data/avanza_storage_state.json (kept warm by the metals_loop), runs
oil_warrant_refresh.refresh_warrant_catalog(page), and persists results
to data/oil_warrant_catalog.json. Reports the populated warrants to
stdout.

Usage:
    .venv/Scripts/python.exe scripts/probe_oil_warrants.py

Designed to run while metals_loop is also running — Playwright contexts
are independent so this won't conflict with the metals session.
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from playwright.sync_api import sync_playwright

from data.oil_warrant_refresh import (
    CATALOG_FILE,
    TTL_HOURS,
    refresh_warrant_catalog,
)
from portfolio.file_utils import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe_oil_warrants")


STORAGE_STATE = REPO / "data" / "avanza_storage_state.json"


def main() -> int:
    if not STORAGE_STATE.exists():
        print(f"FATAL: no storage state at {STORAGE_STATE}", file=sys.stderr)
        print("Run metals_loop or BankID re-auth first.", file=sys.stderr)
        return 1

    print(f"Opening Avanza session via {STORAGE_STATE}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            context = browser.new_context(storage_state=str(STORAGE_STATE))
            page = context.new_page()
            # Warm the session — visiting the home page validates cookies
            try:
                page.goto("https://www.avanza.se/", wait_until="domcontentloaded",
                          timeout=15000)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Home page load failed (continuing): %s", exc)

            print("Calling refresh_warrant_catalog (oil)")
            new_catalog, covered = refresh_warrant_catalog(page)

            # Merge with any existing entries the way load_catalog_or_fetch
            # does (preserves SHORT entries when LONG was the only covered
            # pair this round, and vice-versa).
            cached = load_json(str(REPO / CATALOG_FILE)) or {}
            existing = cached.get("warrants") or {}
            for key, w in existing.items():
                pair = (w.get("underlying"), w.get("direction"))
                if pair not in covered and key not in new_catalog:
                    new_catalog[key] = w

            payload = {
                "refreshed_ts": datetime.datetime.now(datetime.UTC).isoformat(),
                "ttl_hours": TTL_HOURS,
                "warrants": new_catalog,
            }
            atomic_write_json(str(REPO / CATALOG_FILE), payload)
        finally:
            browser.close()

    print()
    print(f"=== Wrote {len(new_catalog)} oil warrants to {CATALOG_FILE} ===")
    print(f"Covered (underlying, direction) pairs: {covered}")
    print()
    for key, w in sorted(new_catalog.items()):
        print(f"  {key}")
        print(f"    ob_id={w.get('ob_id')} dir={w.get('direction')} "
              f"lev={w.get('leverage')} bar={w.get('barrier')} "
              f"ask={w.get('ask')} bid={w.get('bid')} "
              f"spread%={w.get('spread_pct')} "
              f"barrier_dist%={w.get('barrier_dist_pct')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
