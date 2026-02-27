"""Avanza BankID login via Playwright browser automation.

Opens a Chromium browser, navigates to Avanza login page, waits for the user
to authenticate via BankID on their phone, then captures the session cookies
and security tokens for API access.

Usage:
    python scripts/avanza_login.py

The session is saved to data/avanza_session.json and is valid for ~24 hours.
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "avanza_session.json"

START_URL = "https://www.avanza.se"
# Known login page paths (we wait until user leaves these)
LOGIN_PATHS = ("/logga-in", "/login")
# API base for test calls
API_BASE = "https://www.avanza.se"


def _is_logged_in(captured_tokens: dict, cookies=None) -> bool:
    """Check if user has authenticated via BankID.

    Requires strong evidence: customer_id from auth response, or
    csid+cstoken cookies (only set after BankID authentication).
    The x-securitytoken header alone is NOT enough — it appears
    in pre-login analytics responses too.
    """
    # Strong signal: customer_id captured from auth API response
    if "customer_id" in captured_tokens:
        return True
    # Strong signal: csid + cstoken cookies (only set after BankID)
    if cookies:
        cookie_names = {c["name"] for c in cookies}
        if "csid" in cookie_names and "cstoken" in cookie_names:
            return True
    return False


def run_login():
    """Launch browser, wait for BankID auth, capture session."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run:")
        print("  pip install playwright && playwright install chromium")
        sys.exit(1)

    captured_tokens = {}

    print("Launching Chromium browser...")
    print("1. Click 'Logga in' on the Avanza page")
    print("2. Authenticate with BankID on your phone")
    print("3. The script will detect login automatically and save your session")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="sv-SE",
        )
        page = context.new_page()

        # Intercept both requests AND responses to capture auth flow
        def handle_request(request):
            url = request.url
            if "avanza.se" in url and "_api" in url:
                headers = request.headers
                # Log request auth headers (what the browser SENDS)
                auth_headers = {
                    k: v for k, v in headers.items()
                    if any(t in k.lower() for t in ("securi", "auth", "token", "session", "csrf", "cookie"))
                }
                if auth_headers:
                    print(f"  [REQ] {url[:100]}")
                    for k, v in auth_headers.items():
                        val = v[:120] if len(v) < 200 else v[:120] + "..."
                        print(f"    >> {k}: {val}")
                        # Capture request headers as the "real" auth tokens
                        kl = k.lower()
                        if kl == "x-securitytoken":
                            captured_tokens["security_token"] = v
                        elif kl in ("x-authenticationsession", "x-authenticationssession"):
                            captured_tokens["authentication_session"] = v

        def handle_response(response):
            url = response.url
            headers = response.headers

            # Debug: log API response status
            if "avanza.se" in url and "_api" in url:
                print(f"  [RES] {response.status} {url[:100]}")

            # Capture tokens from response headers
            lower_headers = {k.lower(): v for k, v in headers.items()}
            # Only set from response if not already set from request
            if "x-securitytoken" in lower_headers and "security_token" not in captured_tokens:
                captured_tokens["security_token"] = lower_headers["x-securitytoken"]
            if "x-authenticationsession" in lower_headers and "authentication_session" not in captured_tokens:
                captured_tokens["authentication_session"] = lower_headers[
                    "x-authenticationsession"
                ]

            # Check response body for auth session info
            if "_api" in url and ("auth" in url.lower() or "session" in url.lower()):
                try:
                    body = response.json()
                    if isinstance(body, dict):
                        for key in ("authenticationSession", "authentication_session"):
                            if key in body:
                                captured_tokens["authentication_session"] = body[key]
                        if "customerId" in body:
                            captured_tokens["customer_id"] = str(body["customerId"])
                        for key in ("securityToken", "security_token"):
                            if key in body:
                                captured_tokens["security_token"] = body[key]
                        if "maxInactiveMinutes" in body:
                            captured_tokens["max_inactive_minutes"] = body[
                                "maxInactiveMinutes"
                            ]
                except Exception:
                    pass

        page.on("request", handle_request)
        page.on("response", handle_response)

        # Navigate to Avanza main page (user clicks login themselves)
        page.goto(START_URL, wait_until="domcontentloaded")
        time.sleep(2)

        # Try to dismiss cookie consent banner
        for selector in [
            "button:has-text('Acceptera')",
            "button:has-text('acceptera')",
            "button:has-text('Godkänn')",
            "[data-testid='cookie-accept']",
            "#onetrust-accept-btn-handler",
        ]:
            try:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=1000):
                    btn.click()
                    print("Dismissed cookie banner.")
                    break
            except Exception:
                continue

        print("Browser opened. Click 'Logga in' and authenticate with BankID.")
        print("Waiting for login...")
        print()

        # Wait for login — check captured tokens + URL changes
        max_wait = 300  # 5 minutes
        start = time.time()
        while time.time() - start < max_wait:
            current_url = page.url
            cookies_now = context.cookies()
            if _is_logged_in(captured_tokens, cookies_now):
                print(f"Login detected! URL: {current_url}")
                if captured_tokens:
                    print(f"  Captured tokens: {list(captured_tokens.keys())}")
                break
            time.sleep(1)
        else:
            print("ERROR: Timed out waiting for login (5 minutes).")
            print("Please try again.")
            browser.close()
            sys.exit(1)

        # Give a moment for any final API calls to complete
        time.sleep(3)

        # If we didn't capture tokens from intercepted responses,
        # try making a lightweight API call to trigger token headers
        if "security_token" not in captured_tokens:
            print("Attempting to capture security token via API call...")
            # Navigate to a page that triggers API calls
            page.goto(
                f"{API_BASE}/min-ekonomi/konton", wait_until="domcontentloaded"
            )
            time.sleep(3)

        # Save Playwright storage state (cookies + localStorage)
        # This can be reloaded in a headless browser for API calls
        STORAGE_FILE = DATA_DIR / "avanza_storage_state.json"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(STORAGE_FILE))
        print(f"Browser storage state saved to {STORAGE_FILE}")

        # Also make a test API call from the browser to verify auth works
        print("Verifying API access from browser...")
        test_resp = context.request.get(
            f"{API_BASE}/_api/position-data/positions"
        )
        if test_resp.ok:
            print(f"  API test OK! Status: {test_resp.status}")
        else:
            print(f"  API test failed: {test_resp.status} {test_resp.text[:200]}")

        # Extract cookies from the browser context
        cookies = context.cookies()
        cookie_list = []
        for c in cookies:
            cookie_list.append(
                {
                    "name": c["name"],
                    "value": c["value"],
                    "domain": c["domain"],
                    "path": c["path"],
                    "secure": c.get("secure", False),
                    "httpOnly": c.get("httpOnly", False),
                    "sameSite": c.get("sameSite", "None"),
                }
            )

        # Also try to get customer_id from cookies or page
        if "customer_id" not in captured_tokens:
            for c in cookies:
                if c["name"].lower() in ("customerid", "customer_id", "ava_cid"):
                    captured_tokens["customer_id"] = c["value"]
                    break

        browser.close()

    # Validate we got something useful
    if not cookie_list:
        print("ERROR: No cookies captured. Login may have failed.")
        sys.exit(1)

    # Determine expiry
    max_inactive = captured_tokens.get("max_inactive_minutes", 1440)  # 24h default
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=max_inactive)

    session_data = {
        "cookies": cookie_list,
        "security_token": captured_tokens.get("security_token"),
        "authentication_session": captured_tokens.get("authentication_session"),
        "customer_id": captured_tokens.get("customer_id"),
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "max_inactive_minutes": max_inactive,
    }

    # Save session
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(
        json.dumps(session_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"Session saved to {SESSION_FILE}")
    print(f"  Security token: {'YES' if session_data['security_token'] else 'NO (will try cookie-based auth)'}")
    print(f"  Authentication session: {'YES' if session_data['authentication_session'] else 'NO'}")
    print(f"  Customer ID: {session_data['customer_id'] or 'unknown'}")
    print(f"  Cookies: {len(cookie_list)} captured")
    print(f"  Expires: {expires_at.strftime('%Y-%m-%d %H:%M UTC')} (~{max_inactive // 60}h)")
    print()
    print("You can now use the Avanza API via portfolio/avanza_client.py")

    return session_data


if __name__ == "__main__":
    run_login()
