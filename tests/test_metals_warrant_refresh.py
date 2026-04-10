"""Regression tests for the metals warrant refresh page-based HTTP path.

Locks in the 2026-04-10 sync_playwright fix: refresh_warrant_catalog(page)
must issue HTTP via page.context.request, not via
portfolio.avanza_session.api_post (which would open a second sync_playwright
instance and crash with "Sync API inside the asyncio loop").
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_warrant_refresh as mwr


def _make_fake_page(post_responses=None, get_responses=None, csrf: str = "test-csrf"):
    """Build a MagicMock page whose context.request.post/get returns canned data.

    post_responses and get_responses are dicts keyed by URL suffix (the part
    after API_BASE) → dict that will be returned by .json(). Pass None to
    simulate a failed request.
    """
    post_responses = post_responses or {}
    get_responses = get_responses or {}

    def _build_response(body):
        resp = MagicMock()
        if body is None:
            resp.ok = False
            resp.status = 500
            resp.json.side_effect = ValueError("no body")
        else:
            resp.ok = True
            resp.status = 200
            resp.json.return_value = body
        return resp

    def _post(url, data=None, headers=None):
        suffix = url.replace(mwr.API_BASE, "")
        body = post_responses.get(suffix)
        return _build_response(body)

    def _get(url):
        suffix = url.replace(mwr.API_BASE, "")
        body = get_responses.get(suffix)
        return _build_response(body)

    page = MagicMock()
    page.context.request.post.side_effect = _post
    page.context.request.get.side_effect = _get
    page.context.cookies.return_value = [{"name": "AZACSRF", "value": csrf}]
    return page


def test_csrf_from_page_extracts_cookie():
    page = _make_fake_page(csrf="abc123")
    assert mwr._csrf_from_page(page) == "abc123"


def test_csrf_from_page_returns_none_when_missing():
    page = MagicMock()
    page.context.cookies.return_value = [{"name": "OTHER", "value": "x"}]
    assert mwr._csrf_from_page(page) is None


def test_page_api_post_uses_page_context_and_includes_csrf():
    page = _make_fake_page(post_responses={"/foo": {"ok": True}}, csrf="tok")
    result = mwr._page_api_post(page, "/foo", {"q": 1})

    assert result == {"ok": True}
    # Verify the helper NEVER reached for portfolio.avanza_session. The call
    # must have gone through page.context.request.post.
    page.context.request.post.assert_called_once()
    call_kwargs = page.context.request.post.call_args
    assert call_kwargs.kwargs["headers"]["X-SecurityToken"] == "tok"
    assert call_kwargs.kwargs["headers"]["Content-Type"] == "application/json"


def test_page_api_get_uses_page_context():
    page = _make_fake_page(get_responses={"/bar": [1, 2, 3]})
    result = mwr._page_api_get(page, "/bar")

    assert result == [1, 2, 3]
    page.context.request.get.assert_called_once()


def test_search_warrants_threads_page_through():
    page = _make_fake_page(post_responses={
        "/_api/search/filtered-search": {
            "hits": [
                {"type": "WARRANT", "orderBookId": "111"},
                {"type": "STOCK", "orderBookId": "222"},  # should be filtered
            ]
        }
    })

    hits = mwr._search_warrants("MINI L SILVER AVA", page)

    assert len(hits) == 1
    assert hits[0]["orderBookId"] == "111"


def test_probe_warrant_threads_page_through():
    probe_data = {
        "name": "MINI L SILVER AVA 330",
        "isin": "SE000000TEST",
        "tradable": "BUYABLE_AND_SELLABLE",
        "quote": {"buy": 10.0, "sell": 10.1},
        "keyIndicators": {
            "leverage": 7.5,
            "barrierLevel": 60.0,
            "parity": 10,
            "direction": "Lång",
            "isAza": True,
        },
        "underlying": {"quote": {"last": 75.0}},
    }
    page = _make_fake_page(get_responses={
        "/_api/market-guide/warrant/999": probe_data
    })

    probe = mwr._probe_warrant("999", page)

    assert probe is not None
    assert probe["name"] == "MINI L SILVER AVA 330"
    assert probe["leverage"] == 7.5
    assert probe["isAza"] is True


def test_refresh_warrant_catalog_requires_page_arg():
    # The API contract: refresh_warrant_catalog takes a page positional arg.
    # A TypeError on missing-page call is the guard against regressing to the
    # old parameterless version that imported api_post from avanza_session.
    import inspect
    sig = inspect.signature(mwr.refresh_warrant_catalog)
    assert "page" in sig.parameters
    assert list(sig.parameters)[0] == "page"


def test_load_catalog_or_fetch_without_page_returns_cache_only(monkeypatch):
    # When called without a page (e.g. from a context that can't own one),
    # the function must skip the refresh and return whatever's in the cache.
    monkeypatch.setattr(
        mwr,
        "_read_cache",
        lambda: {
            "refreshed_ts": "2020-01-01T00:00:00+00:00",  # stale on purpose
            "ttl_hours": 6,
            "warrants": {"SOMETHING": {"underlying": "XAG-USD", "direction": "LONG"}},
        },
    )

    result = mwr.load_catalog_or_fetch(page=None)

    # Returns the cached warrants (even though stale) — better than nothing.
    assert "SOMETHING" in result


def test_metals_warrant_refresh_does_not_import_avanza_session_api():
    # The whole point of the fix: this module MUST NOT pull in
    # portfolio.avanza_session.api_post / api_get, because doing so will
    # trigger a second sync_playwright().start() at runtime.
    with open(mwr.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert "from portfolio.avanza_session import api_get" not in source
    assert "from portfolio.avanza_session import api_post" not in source
    assert "avanza_session import api_get, api_post" not in source
