"""Auto-recovering Playwright Page wrapper for Avanza-bound loops.

Problem: long-running loops (`data/metals_loop.py`, `portfolio/golddigger/`,
`portfolio/main.py` via `avanza_session.py`) open a headless Chromium at
startup and hold the `page` reference for days. When the browser dies
(OS sleep, memory pressure, WSL ping hiccup, external BankID re-auth) the
Python process keeps running but every `page.evaluate()` throws
`playwright._impl._errors.TargetClosedError` — silently, for days. The
bug discovered 2026-04-13: metals loop was emitting this error 662 times
between 2026-04-09 and 2026-04-13, making zero trades.

Solution: pass a `ResilientPage` instead of a raw Playwright `Page`. On
`TargetClosedError` (or equivalent browser-dead message) the wrapper tears
down the dead browser+context, relaunches Chromium, reloads the saved
`avanza_storage_state.json`, and retries the failing call once. Only then
does it propagate the error.

This is the minimal surface — `evaluate()` and `context.cookies()` — that
the existing helpers use. Other Page methods pass through unchanged via
`__getattr__`; they get no auto-recovery (good enough: they're only used
during startup, where crash-and-bat-restart is acceptable).

Usage:

    with sync_playwright() as pw:
        page = ResilientPage.open(pw, "data/avanza_storage_state.json")
        # pass `page` to existing helpers — zero call-site changes needed
        fetch_price(page, ob_id, api_type)
        fetch_account_cash(page, account_id)
        # on shutdown:
        page.close()
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger("portfolio.avanza_resilient_page")

_INITIAL_URL = "https://www.avanza.se/min-ekonomi/oversikt.html"
_INITIAL_URL_WAIT_MS = 2000


def is_browser_dead_error(exc: BaseException) -> bool:
    """True if ``exc`` signals a dead Playwright browser/context.

    Checks both the exception class name (Playwright's
    ``TargetClosedError`` name changed across versions) and the message
    (the stable cross-version signal). Exposed for tests and for
    ``avanza_session.py`` which wants the same classifier without
    importing Playwright internals.
    """
    name = type(exc).__name__
    if name == "TargetClosedError":
        return True
    msg = str(exc)
    for marker in (
        "Target page, context or browser has been closed",
        "Target closed",
        "Browser has been closed",
        "has been closed",
    ):
        if marker in msg:
            return True
    return False


class ResilientPage:
    """Playwright ``Page`` wrapper that auto-relaunches the browser on death."""

    def __init__(
        self,
        pw: Any,
        storage_state_path: str,
        *,
        headless: bool = True,
        locale: str = "sv-SE",
        initial_url: str | None = _INITIAL_URL,
        initial_url_wait_ms: int = _INITIAL_URL_WAIT_MS,
    ) -> None:
        self._pw = pw
        self._storage_state_path = storage_state_path
        self._headless = headless
        self._locale = locale
        self._initial_url = initial_url
        self._initial_url_wait_ms = initial_url_wait_ms
        self._browser = None
        self._ctx = None
        self._page = None
        self._relaunch_count = 0
        self._last_relaunch_ts: str | None = None

    @classmethod
    def open(
        cls,
        pw: Any,
        storage_state_path: str,
        *,
        headless: bool = True,
        locale: str = "sv-SE",
        initial_url: str | None = _INITIAL_URL,
        initial_url_wait_ms: int = _INITIAL_URL_WAIT_MS,
    ) -> ResilientPage:
        """Construct and open the browser. Preferred entry point."""
        rp = cls(
            pw,
            storage_state_path,
            headless=headless,
            locale=locale,
            initial_url=initial_url,
            initial_url_wait_ms=initial_url_wait_ms,
        )
        rp._open()
        return rp

    def _open(self) -> None:
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._ctx = self._browser.new_context(
            storage_state=self._storage_state_path,
            locale=self._locale,
        )
        self._page = self._ctx.new_page()
        if self._initial_url:
            self._page.goto(self._initial_url, wait_until="domcontentloaded")
            if self._initial_url_wait_ms:
                self._page.wait_for_timeout(self._initial_url_wait_ms)

    def _close_quietly(self) -> None:
        for closer in (self._ctx, self._browser):
            if closer is None:
                continue
            try:
                closer.close()
            except Exception as exc:
                logger.debug("ResilientPage teardown: %s", exc)
        self._ctx = None
        self._browser = None
        self._page = None

    def _relaunch(self, *, reason: str) -> None:
        self._relaunch_count += 1
        self._last_relaunch_ts = datetime.datetime.now(datetime.UTC).isoformat()
        logger.warning(
            "ResilientPage: browser dead (%s) — relaunching (count=%d)",
            reason, self._relaunch_count,
        )
        self._close_quietly()
        self._open()

    def close(self) -> None:
        """Teardown browser. Safe to call multiple times."""
        self._close_quietly()

    # --- Recovery-aware proxy API ---

    def evaluate(self, script: str, arg: Any = None) -> Any:
        """``page.evaluate(script, arg)`` with one-shot auto-recovery.

        On ``TargetClosedError`` (or equivalent), teardown + relaunch +
        retry. If the retry also fails with a browser-dead error, propagate.
        """
        try:
            if arg is None:
                return self._page.evaluate(script)
            return self._page.evaluate(script, arg)
        except Exception as exc:
            if not is_browser_dead_error(exc):
                raise
            self._relaunch(reason="evaluate")
            if arg is None:
                return self._page.evaluate(script)
            return self._page.evaluate(script, arg)

    def goto(self, *args, **kwargs) -> Any:
        """``page.goto()`` with one-shot auto-recovery."""
        try:
            return self._page.goto(*args, **kwargs)
        except Exception as exc:
            if not is_browser_dead_error(exc):
                raise
            self._relaunch(reason="goto")
            return self._page.goto(*args, **kwargs)

    @property
    def context(self):
        """Return a context proxy whose ``cookies()`` auto-recovers."""
        return _ResilientContextProxy(self)

    # Passthrough for everything else (wait_for_timeout, locator, on, etc.)
    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails — so this does
        # NOT shadow evaluate/goto/context/close defined above.
        if name.startswith("_"):
            raise AttributeError(name)
        target = self.__dict__.get("_page")
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)

    # --- Observability ---

    @property
    def relaunch_count(self) -> int:
        return self._relaunch_count

    @property
    def last_relaunch_ts(self) -> str | None:
        return self._last_relaunch_ts


class _ResilientContextProxy:
    """Proxy for ``BrowserContext`` that auto-recovers ``cookies()`` calls."""

    def __init__(self, resilient_page: ResilientPage) -> None:
        self._rp = resilient_page

    def cookies(self) -> list[dict]:
        try:
            return self._rp._ctx.cookies()
        except Exception as exc:
            if not is_browser_dead_error(exc):
                raise
            self._rp._relaunch(reason="context.cookies")
            return self._rp._ctx.cookies()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._rp._ctx, name)
