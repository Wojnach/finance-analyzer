"""Unit tests for portfolio.avanza_resilient_page.ResilientPage.

Uses a fake Playwright API surface — no actual Chromium launch — so the
tests run fast in CI without needing a browser.
"""

from __future__ import annotations

import pytest

from portfolio.avanza_resilient_page import ResilientPage, is_browser_dead_error


class FakeTargetClosedError(Exception):
    """Mimics Playwright's TargetClosedError — same class name matters."""


FakeTargetClosedError.__name__ = "TargetClosedError"


class FakePage:
    """Stand-in for Playwright's Page. Configurable to die on command."""

    def __init__(self, ctx):
        self._ctx = ctx
        self.evaluate_calls = []
        self.goto_calls = []
        self.wait_for_timeout_calls = []
        self._alive = True
        self._evaluate_die_on_next = False
        self._goto_die_on_next = False

    def evaluate(self, script, arg=None):
        self.evaluate_calls.append((script, arg))
        if self._evaluate_die_on_next:
            self._evaluate_die_on_next = False
            self._alive = False
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        if not self._alive:
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        if arg is None:
            return {"script": script}
        return {"script": script, "arg": arg}

    def goto(self, url, wait_until="load"):
        self.goto_calls.append((url, wait_until))
        if self._goto_die_on_next:
            self._goto_die_on_next = False
            self._alive = False
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return None

    def wait_for_timeout(self, ms):
        self.wait_for_timeout_calls.append(ms)


class FakeContext:
    def __init__(self, browser):
        self._browser = browser
        self._cookies = [{"name": "AZACSRF", "value": "token-123"}]
        self._closed = False
        self._cookies_die_on_next = False

    def cookies(self):
        if self._cookies_die_on_next:
            self._cookies_die_on_next = False
            self._closed = True
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        if self._closed:
            raise FakeTargetClosedError("Target page, context or browser has been closed")
        return self._cookies

    def new_page(self):
        return FakePage(self)

    def close(self):
        self._closed = True


class FakeBrowser:
    def __init__(self, pw):
        self._pw = pw
        self._contexts = []
        self._closed = False

    def new_context(self, storage_state=None, locale=None):
        ctx = FakeContext(self)
        self._contexts.append(ctx)
        return ctx

    def close(self):
        self._closed = True


class FakeChromium:
    def __init__(self, pw):
        self._pw = pw
        self.launched_browsers: list[FakeBrowser] = []

    def launch(self, headless=True):
        b = FakeBrowser(self._pw)
        self.launched_browsers.append(b)
        return b


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium(self)


# Tests use FakePlaywright so no actual browser launches and the
# storage_state path is never read from disk. A stub string is sufficient.
_FAKE_STATE_PATH = "FAKE_STORAGE_STATE"


@pytest.fixture
def pw():
    return FakePlaywright()


@pytest.fixture
def rp(pw):
    return ResilientPage.open(
        pw,
        _FAKE_STATE_PATH,
        headless=True,
        initial_url=None,  # skip goto in tests
    )


# --- is_browser_dead_error classifier ---


def test_classifier_detects_fake_target_closed_by_class_name():
    assert is_browser_dead_error(FakeTargetClosedError("x"))


def test_classifier_detects_by_message():
    assert is_browser_dead_error(Exception("Target page, context or browser has been closed"))
    assert is_browser_dead_error(Exception("Browser has been closed"))
    assert is_browser_dead_error(Exception("Target closed: transient wrap"))


def test_classifier_ignores_normal_errors():
    assert not is_browser_dead_error(Exception("normal failure"))
    assert not is_browser_dead_error(ValueError("bad input"))
    assert not is_browser_dead_error(RuntimeError("http 500"))


# --- ResilientPage.open happy path ---


def test_open_launches_browser_and_creates_context(pw):
    rp = ResilientPage.open(pw, _FAKE_STATE_PATH, initial_url=None)
    assert len(pw.chromium.launched_browsers) == 1
    assert rp.relaunch_count == 0
    assert rp.last_relaunch_ts is None


# --- evaluate auto-recovery ---


def test_evaluate_happy_path(rp):
    result = rp.evaluate("return 42;")
    assert result == {"script": "return 42;"}


def test_evaluate_with_arg(rp):
    result = rp.evaluate("return arg;", [1, 2, 3])
    assert result == {"script": "return arg;", "arg": [1, 2, 3]}


def test_evaluate_recovers_from_dead_browser(rp, pw):
    rp._page._evaluate_die_on_next = True
    result = rp.evaluate("return 42;")
    # Recovery happened: new browser launched, evaluate returned on retry.
    assert rp.relaunch_count == 1
    assert rp.last_relaunch_ts is not None
    assert len(pw.chromium.launched_browsers) == 2
    # The new page returns the mocked result from the fresh context.
    assert result == {"script": "return 42;"}


def test_evaluate_does_not_retry_on_normal_error(rp):
    """Only TargetClosedError triggers relaunch; other errors propagate."""

    def boom(script, arg=None):
        raise ValueError("not a browser death")

    rp._page.evaluate = boom
    with pytest.raises(ValueError, match="not a browser death"):
        rp.evaluate("x")
    assert rp.relaunch_count == 0


def test_evaluate_gives_up_after_one_retry(pw):
    """If the relaunched browser ALSO fails immediately, we propagate."""
    # Make every FakePage die on first evaluate (persistent failure).
    original_new_page = FakeContext.new_page

    def make_dying_page(self):
        page = original_new_page(self)
        page._evaluate_die_on_next = True
        return page

    FakeContext.new_page = make_dying_page
    try:
        rp = ResilientPage.open(pw, _FAKE_STATE_PATH, initial_url=None)
        with pytest.raises(FakeTargetClosedError):
            rp.evaluate("return 1;")
        assert rp.relaunch_count == 1  # tried relaunch once, retry also died
    finally:
        FakeContext.new_page = original_new_page


# --- context.cookies auto-recovery ---


def test_cookies_happy_path(rp):
    cookies = rp.context.cookies()
    assert cookies == [{"name": "AZACSRF", "value": "token-123"}]


def test_cookies_recovers_from_dead_context(rp, pw):
    rp._ctx._cookies_die_on_next = True
    cookies = rp.context.cookies()
    assert rp.relaunch_count == 1
    assert len(pw.chromium.launched_browsers) == 2
    # Fresh context returned fresh cookies.
    assert cookies == [{"name": "AZACSRF", "value": "token-123"}]


# --- close() ---


def test_close_tears_down_browser_and_context(rp):
    browser = rp._browser
    ctx = rp._ctx
    rp.close()
    assert browser._closed
    assert ctx._closed
    assert rp._page is None


def test_close_is_idempotent(rp):
    rp.close()
    rp.close()  # no error


# --- passthrough ---


def test_passthrough_wait_for_timeout(rp):
    rp.wait_for_timeout(500)
    assert rp._page.wait_for_timeout_calls == [500]
