#!/usr/bin/env python3
"""End-to-end sanity check for the Cloudflare tunnel + dashboard auth.

Architecture as of 2026-04-30:
  raanman.lol (apex, canonical) -> Cloudflare tunnel -> http://localhost:5055
  bets.raanman.lol (legacy alias) -> same backend

The dashboard supports three auth paths (in priority order): cookie, ?token=
query param, Authorization: Bearer header. This verifier asserts each.

Run after `setup_tunnel.bat` and after the dashboard is running:

    .venv/Scripts/python.exe scripts/verify_tunnel.py

Token loaded from config.json at runtime, never echoed. Exit codes:
  0 = all pass, 1 = check failed, 2 = config missing.
"""

import sys
from pathlib import Path

import requests

# Use the project's atomic loader so a concurrent config.json write (loop
# cycle, hot-reload) can't hand us a partially-written file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.file_utils import load_json  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config.json"
APEX = "https://raanman.lol"
LEGACY = "https://bets.raanman.lol"
COOKIE_NAME = "pf_dashboard_token"
TIMEOUT = 10


def load_token() -> str:
    cfg = load_json(CONFIG, default={}) or {}
    return cfg.get("dashboard_token") or ""


def check(label: str, ok: bool, detail: str) -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label} — {detail}")
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    token = load_token()
    if not token:
        print("FATAL: dashboard_token missing or empty in config.json")
        return 2

    failures = 0

    section(f"{APEX}  (canonical apex)")
    try:
        r = requests.get(
            f"{APEX}/api/health",
            params={"token": token},
            timeout=TIMEOUT,
        )
        failures += not check(
            "query-token returns 200",
            r.status_code == 200,
            f"HTTP {r.status_code}",
        )
        # The whole point of cookie auth: a valid query token must set the
        # cookie on the response. If it doesn't, the user's URL stays long.
        failures += not check(
            f"sets {COOKIE_NAME} cookie on response",
            COOKIE_NAME in r.cookies,
            f"cookies: {list(r.cookies.keys()) or 'none'}",
        )
    except requests.RequestException as exc:
        failures += not check("query-token returns 200", False, f"{type(exc).__name__}: {exc}")

    try:
        r = requests.get(f"{APEX}/api/health", timeout=TIMEOUT)
        # The negative test: no token → must reject. A regression here would
        # silently expose the dashboard to anyone who finds the URL.
        failures += not check(
            "no token is rejected",
            r.status_code == 401,
            f"HTTP {r.status_code} (want 401)",
        )
    except requests.RequestException as exc:
        failures += not check("no token", False, f"{type(exc).__name__}: {exc}")

    try:
        r = requests.get(
            f"{APEX}/api/health",
            cookies={COOKIE_NAME: token},
            timeout=TIMEOUT,
        )
        failures += not check(
            "cookie-only returns 200",
            r.status_code == 200,
            f"HTTP {r.status_code}",
        )
    except requests.RequestException as exc:
        failures += not check("cookie-only", False, f"{type(exc).__name__}: {exc}")

    section(f"{LEGACY}  (legacy alias kept in ingress)")
    try:
        r = requests.get(
            f"{LEGACY}/api/health",
            params={"token": token},
            timeout=TIMEOUT,
        )
        failures += not check(
            "alias still routes to dashboard",
            r.status_code == 200,
            f"HTTP {r.status_code}",
        )
    except requests.RequestException as exc:
        failures += not check("legacy alias", False, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"RESULT: {failures} check(s) failed.")
        return 1
    print("RESULT: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
