#!/usr/bin/env python3
"""End-to-end sanity check for the dashboard + tunnel + CF Access stack.

Architecture (as of 2026-05-02):
  Browser → CF Access (edge auth) → Cloudflare tunnel → http://localhost:5055

CF Access intercepts external requests and redirects unauthenticated callers
to a login page, so a script that doesn't have a CF Access service token
can't directly test the dashboard's auth from outside. This verifier splits
checks into two halves:

  1. **Local checks** — bypass CF Access by hitting http://localhost:5055
     directly. Tests every dashboard auth path the user can reach via cookie
     or token, plus the negative gate. This is the strongest check we can
     run without a CF Access service token.

  2. **Public liveness checks** — confirm raanman.lol and bets.raanman.lol
     respond with 2xx/3xx (CF Access intercept counts as healthy). 5xx,
     timeouts, or DNS failures = real outage.

Run after `setup_tunnel.bat` and after the dashboard is running:

    .venv/Scripts/python.exe scripts/verify_tunnel.py

Token loaded from config.json at runtime, never echoed. Exit codes:
  0 = all pass, 1 = check failed, 2 = config missing.
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.file_utils import load_json  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config.json"
LOCAL = "http://localhost:5055"
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

    section(f"{LOCAL}  (local — bypasses CF Access, full auth coverage)")
    try:
        r = requests.get(
            f"{LOCAL}/api/health",
            params={"token": token},
            timeout=TIMEOUT,
        )
        failures += not check(
            "query-token returns 200",
            r.status_code == 200,
            f"HTTP {r.status_code}",
        )
        failures += not check(
            f"sets {COOKIE_NAME} cookie on response",
            COOKIE_NAME in r.cookies,
            f"cookies: {list(r.cookies.keys()) or 'none'}",
        )
    except requests.RequestException as exc:
        failures += not check("local query-token", False, f"{type(exc).__name__}: {exc}")

    try:
        r = requests.get(f"{LOCAL}/api/health", timeout=TIMEOUT)
        # Negative test: no token → must reject. A regression here would
        # silently expose the dashboard to anyone on the local network.
        failures += not check(
            "no token is rejected (auth gate fires)",
            r.status_code == 401,
            f"HTTP {r.status_code} (want 401)",
        )
    except requests.RequestException as exc:
        failures += not check("local no-token", False, f"{type(exc).__name__}: {exc}")

    try:
        r = requests.get(
            f"{LOCAL}/api/health",
            cookies={COOKIE_NAME: token},
            timeout=TIMEOUT,
        )
        failures += not check(
            "cookie-only returns 200 (rolling cookie path)",
            r.status_code == 200,
            f"HTTP {r.status_code}",
        )
    except requests.RequestException as exc:
        failures += not check("local cookie-only", False, f"{type(exc).__name__}: {exc}")

    section(f"{APEX}  (public — confirms CF edge + tunnel are alive)")
    try:
        # CF Access intercepts. Any 2xx or 3xx = healthy (CF login page or
        # an authenticated forward to the dashboard). 4xx that isn't 401 is
        # suspicious; 5xx or connection failure means the tunnel or origin
        # is down.
        r = requests.get(APEX, timeout=TIMEOUT, allow_redirects=False)
        failures += not check(
            "responds with 2xx/3xx (CF Access reachable)",
            200 <= r.status_code < 400,
            f"HTTP {r.status_code}",
        )
    except requests.RequestException as exc:
        failures += not check("public reachable", False, f"{type(exc).__name__}: {exc}")

    section(f"{LEGACY}  (legacy alias — also CF-Access-gated)")
    try:
        r = requests.get(LEGACY, timeout=TIMEOUT, allow_redirects=False)
        failures += not check(
            "alias responds with 2xx/3xx",
            200 <= r.status_code < 400,
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
