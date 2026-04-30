#!/usr/bin/env python3
"""End-to-end sanity check for the Cloudflare tunnel + GH Pages redirect.

Run after `setup_tunnel.bat` (step 6) and `setup_gh_pages.py` to confirm:
  1. The tunnel is reachable and the dashboard answers (200 with token).
  2. The auth gate is enforced (no token -> not 200).
  3. raanman.lol/bets serves the redirect page pointing at the tunnel.
  4. The apex (raanman.lol) still serves the game (no 4xx/5xx).

Exit codes: 0 = all pass, 1 = one or more checks failed, 2 = config missing.

Usage (from anywhere in the repo):
    .venv/Scripts/python.exe scripts/verify_tunnel.py

The dashboard_token is loaded from config.json — never passed on the
command line, never echoed to stdout.
"""

import json
import sys
from pathlib import Path

import requests

CONFIG = Path(__file__).resolve().parents[1] / "config.json"
TUNNEL = "https://bets.raanman.lol"
APEX_BETS = "https://raanman.lol/bets/"
APEX = "https://raanman.lol/"
TIMEOUT = 10


def load_token() -> str:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
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

    section(f"{TUNNEL}  (Cloudflare tunnel -> Flask dashboard)")
    try:
        r = requests.get(
            f"{TUNNEL}/api/health",
            params={"token": token},
            timeout=TIMEOUT,
        )
        ok = check("health with token returns 200", r.status_code == 200, f"HTTP {r.status_code}")
        failures += not ok
        if ok:
            try:
                payload = r.json()
                failures += not check(
                    "health body is JSON dict",
                    isinstance(payload, dict),
                    f"keys={list(payload)[:5] if isinstance(payload, dict) else type(payload).__name__}",
                )
            except ValueError:
                failures += not check("health body is JSON dict", False, "non-JSON response")
    except requests.RequestException as exc:
        failures += not check("health with token", False, f"{type(exc).__name__}: {exc}")

    try:
        r = requests.get(f"{TUNNEL}/api/health", timeout=TIMEOUT)
        # Want anything except a clean 200 — 401/403/302 all prove the gate fires.
        failures += not check(
            "health WITHOUT token is rejected",
            r.status_code != 200,
            f"HTTP {r.status_code} (want != 200)",
        )
    except requests.RequestException as exc:
        failures += not check("health without token", False, f"{type(exc).__name__}: {exc}")

    section(f"{APEX_BETS}  (GH Pages redirect to tunnel)")
    try:
        # allow_redirects=False so we inspect the HTML body, not follow it.
        r = requests.get(APEX_BETS, timeout=TIMEOUT, allow_redirects=False)
        ok = check("redirect page returns 200", r.status_code == 200, f"HTTP {r.status_code}")
        failures += not ok
        if ok:
            body = r.text
            failures += not check(
                "body references tunnel URL",
                TUNNEL in body,
                "found" if TUNNEL in body else f"missing string {TUNNEL!r}",
            )
            failures += not check(
                "uses location.replace (no back-button loop)",
                "location.replace" in body,
                "found" if "location.replace" in body else "missing — would create back-button trap",
            )
            failures += not check(
                "has meta-refresh fallback",
                'http-equiv="refresh"' in body,
                "found" if 'http-equiv="refresh"' in body else "missing — no-JS users will see blank page",
            )
    except requests.RequestException as exc:
        failures += not check("redirect page reachable", False, f"{type(exc).__name__}: {exc}")

    section(f"{APEX}  (apex — game must still work)")
    try:
        r = requests.get(APEX, timeout=TIMEOUT, allow_redirects=True)
        failures += not check(
            "apex serves content (final status < 400)",
            r.status_code < 400,
            f"HTTP {r.status_code}, final URL {r.url}",
        )
    except requests.RequestException as exc:
        failures += not check("apex reachable", False, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"RESULT: {failures} check(s) failed.")
        return 1
    print("RESULT: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
