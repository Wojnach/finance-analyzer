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

LOGIN_URL = "https://www.avanza.se/logga-in"
# After successful BankID login, Avanza redirects to the start page
POST_LOGIN_URLS = ("/start", "/hem", "/min-ekonomi", "/mina-sidor")
# API base for test calls
API_BASE = "https://www.avanza.se"


def _is_logged_in(url: str) -> bool:
    """Check if the current URL indicates successful login."""
    from urllib.parse import urlparse
    path = urlparse(url).path
    # Logged in if NOT on login page and on a known authenticated route
    if "/logga-in" in path:
        return False
    return any(path.startswith(p) for p in POST_LOGIN_URLS) or path == "/"


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
    print("Please log in with BankID when the browser opens.")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="sv-SE",
        )
        page = context.new_page()

        # Intercept API responses to capture security tokens
        def handle_response(response):
            url = response.url
            # Avanza returns auth tokens in response headers of API calls
            headers = response.headers
            if "x-securitytoken" in headers:
                captured_tokens["security_token"] = headers["x-securitytoken"]
            if "x-authenticationSession" in headers:
                captured_tokens["authentication_session"] = headers[
                    "x-authenticationSession"
                ]
            # Also check response body for auth session info
            if "/_api/authentication" in url or "/authentication/sessions" in url:
                try:
                    body = response.json()
                    if isinstance(body, dict):
                        if "authenticationSession" in body:
                            captured_tokens["authentication_session"] = body[
                                "authenticationSession"
                            ]
                        if "customerId" in body:
                            captured_tokens["customer_id"] = str(body["customerId"])
                        if "securityToken" in body:
                            captured_tokens["security_token"] = body["securityToken"]
                        if "maxInactiveMinutes" in body:
                            captured_tokens["max_inactive_minutes"] = body[
                                "maxInactiveMinutes"
                            ]
                except Exception:
                    pass

        page.on("response", handle_response)

        # Navigate to login page
        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        print("Browser opened at Avanza login page.")
        print("Waiting for BankID authentication...")
        print("(This script will detect login automatically)")
        print()

        # Wait for login — poll URL changes
        max_wait = 300  # 5 minutes
        start = time.time()
        while time.time() - start < max_wait:
            current_url = page.url
            if _is_logged_in(current_url):
                print(f"Login detected! Redirected to: {current_url}")
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
