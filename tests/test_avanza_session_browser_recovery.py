"""Tests for portfolio.avanza_session browser-death auto-recovery.

Exercises the ``_with_browser_recovery`` wrapper applied to api_get,
api_post, and api_delete. Uses fake Playwright contexts so no actual
Chromium launches. Pairs with test_avanza_resilient_page.py — this file
covers the singleton/lock-based API path, the other covers the per-loop
page-wrapper path.
"""

from __future__ import annotations

import pytest

from portfolio import avanza_session


class FakeTargetClosedError(Exception):
    """Mimics Playwright's TargetClosedError — class name matters."""


FakeTargetClosedError.__name__ = "TargetClosedError"


class FakeResponse:
    def __init__(self, status=200, body="{}", ok_override=None):
        self.status = status
        self._body = body
        self._ok_override = ok_override

    @property
    def ok(self):
        return self._ok_override if self._ok_override is not None else 200 <= self.status < 300

    def text(self):
        return self._body

    def json(self):
        import json as _json
        return _json.loads(self._body)


class FakeRequest:
    def __init__(self, ctx):
        self._ctx = ctx
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict]] = []
        self.delete_calls: list[str] = []

    def get(self, url):
        self.get_calls.append(url)
        if self._ctx._die_on_next_request:
            self._ctx._die_on_next_request = False
            self._ctx._dead = True
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        if self._ctx._dead:
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return FakeResponse(status=200, body='{"ok": true}')

    def post(self, url, data=None, headers=None):
        self.post_calls.append((url, headers or {}))
        if self._ctx._die_on_next_request:
            self._ctx._die_on_next_request = False
            self._ctx._dead = True
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        if self._ctx._dead:
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return FakeResponse(status=200, body='{"posted": true}')

    def delete(self, url, headers=None):
        self.delete_calls.append(url)
        if self._ctx._die_on_next_request:
            self._ctx._die_on_next_request = False
            self._ctx._dead = True
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        if self._ctx._dead:
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return FakeResponse(status=200, body='')


class FakeContext:
    """Stand-in for Playwright BrowserContext."""

    def __init__(self, ctx_id: int):
        self.ctx_id = ctx_id
        self.request = FakeRequest(self)
        self._cookies = [{"name": "AZACSRF", "value": f"token-{ctx_id}"}]
        self._dead = False
        self._die_on_next_request = False
        self.closed = False

    def cookies(self):
        if self._dead:
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return self._cookies

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_avanza_session_state(monkeypatch):
    """Stub _get_playwright_context / close_playwright so tests don't touch real Playwright."""
    ctx_registry = {"next_id": 1, "current": None}

    def fake_get_ctx():
        if ctx_registry["current"] is None:
            ctx = FakeContext(ctx_registry["next_id"])
            ctx_registry["next_id"] += 1
            ctx_registry["current"] = ctx
        return ctx_registry["current"]

    def fake_close_pw():
        if ctx_registry["current"] is not None:
            ctx_registry["current"].close()
        ctx_registry["current"] = None

    monkeypatch.setattr(avanza_session, "_get_playwright_context", fake_get_ctx)
    monkeypatch.setattr(avanza_session, "close_playwright", fake_close_pw)
    yield ctx_registry


# --- _with_browser_recovery helper ---


def test_with_recovery_happy_path(reset_avanza_session_state):
    """Op runs once under the lock, returns successfully, no relaunch."""
    calls = []

    def op(ctx):
        calls.append(ctx.ctx_id)
        return "ok"

    result = avanza_session._with_browser_recovery(op, op_name="test")
    assert result == "ok"
    assert calls == [1]


def test_with_recovery_relaunches_on_browser_dead(reset_avanza_session_state):
    """On TargetClosedError, teardown + relaunch + retry once."""
    attempts = []

    def op(ctx):
        attempts.append(ctx.ctx_id)
        if len(attempts) == 1:
            ctx._dead = True
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return "recovered"

    result = avanza_session._with_browser_recovery(op, op_name="test")
    assert result == "recovered"
    # Two attempts: first on ctx 1, second on ctx 2 after relaunch.
    assert attempts == [1, 2]


def test_with_recovery_propagates_non_browser_errors(reset_avanza_session_state):
    """Normal errors (ValueError, HTTP 500) are NOT caught — propagate immediately."""

    def op(ctx):
        raise ValueError("normal failure")

    with pytest.raises(ValueError, match="normal failure"):
        avanza_session._with_browser_recovery(op, op_name="test")


def test_with_recovery_second_failure_propagates(reset_avanza_session_state):
    """If the retry also fails with browser-dead, propagate (no infinite loop)."""
    attempts = []

    def op(ctx):
        attempts.append(ctx.ctx_id)
        ctx._dead = True
        raise FakeTargetClosedError("Target page, context or browser has been closed")

    with pytest.raises(FakeTargetClosedError):
        avanza_session._with_browser_recovery(op, op_name="test")
    assert len(attempts) == 2  # one original, one retry after relaunch


# --- api_get / api_post / api_delete happy paths ---


def test_api_get_happy_path(reset_avanza_session_state):
    result = avanza_session.api_get("/_api/position-data/positions")
    assert result == {"ok": True}
    ctx = reset_avanza_session_state["current"]
    assert ctx.request.get_calls == ["https://www.avanza.se/_api/position-data/positions"]


def test_api_post_happy_path(reset_avanza_session_state):
    result = avanza_session.api_post("/_api/trading/order/new", {"action": "BUY"})
    assert result == {"posted": True}
    ctx = reset_avanza_session_state["current"]
    assert len(ctx.request.post_calls) == 1
    url, headers = ctx.request.post_calls[0]
    assert url == "https://www.avanza.se/_api/trading/order/new"
    assert headers["X-SecurityToken"] == "token-1"


def test_api_delete_happy_path(reset_avanza_session_state):
    result = avanza_session.api_delete("/_api/trading/stoploss/42")
    assert result == {"http_status": 200, "ok": True}


# --- api_* recovery paths ---


def test_api_get_recovers_from_dead_browser(reset_avanza_session_state):
    """GET on dead browser → teardown, relaunch, retry, succeed."""
    ctx1 = reset_avanza_session_state["current"] = FakeContext(1)
    reset_avanza_session_state["next_id"] = 2
    ctx1._die_on_next_request = True

    result = avanza_session.api_get("/_api/position-data/positions")
    assert result == {"ok": True}
    assert ctx1.closed, "Old dead context should have been closed by recovery"
    # New context was created with id 2.
    current = reset_avanza_session_state["current"]
    assert current.ctx_id == 2


def test_api_post_recovers_from_dead_browser(reset_avanza_session_state):
    """POST on dead browser → recovery uses FRESH CSRF from relaunched ctx."""
    ctx1 = reset_avanza_session_state["current"] = FakeContext(1)
    reset_avanza_session_state["next_id"] = 2
    ctx1._die_on_next_request = True

    result = avanza_session.api_post("/_api/trading/order/new", {"x": 1})
    assert result == {"posted": True}
    # The successful POST used the NEW context's CSRF (token-2), not token-1.
    current = reset_avanza_session_state["current"]
    assert len(current.request.post_calls) == 1
    _, headers = current.request.post_calls[0]
    assert headers["X-SecurityToken"] == "token-2"


def test_api_delete_recovers_from_dead_browser(reset_avanza_session_state):
    ctx1 = reset_avanza_session_state["current"] = FakeContext(1)
    reset_avanza_session_state["next_id"] = 2
    ctx1._die_on_next_request = True

    result = avanza_session.api_delete("/_api/trading/stoploss/42")
    assert result == {"http_status": 200, "ok": True}
    current = reset_avanza_session_state["current"]
    assert current.ctx_id == 2


def test_api_get_401_still_raises_and_closes(reset_avanza_session_state):
    """401 still triggers AvanzaSessionError (not treated as browser-dead)."""
    ctx = reset_avanza_session_state["current"] = FakeContext(1)
    reset_avanza_session_state["next_id"] = 2

    def bad_get(url):
        return FakeResponse(status=401, body="unauth", ok_override=False)

    ctx.request.get = bad_get
    with pytest.raises(avanza_session.AvanzaSessionError):
        avanza_session.api_get("/foo")
    # close_playwright was called during 401 path.
    assert reset_avanza_session_state["current"] is None


def test_get_csrf_ctx_arg_bypasses_lock(reset_avanza_session_state):
    """_get_csrf(ctx=<explicit>) reads cookies directly — no singleton fetch."""
    explicit_ctx = FakeContext(99)
    result = avanza_session._get_csrf(explicit_ctx)
    assert result == "token-99"
    # Registry's current context was NEVER fetched.
    assert reset_avanza_session_state["current"] is None
