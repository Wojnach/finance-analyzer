"""Tests for data/oil_warrant_refresh.py.

Mirrors test_metals_warrant_refresh.py / test_crypto_swing_trader.py patterns.
Verifies:
  - Page-based HTTP path (no second sync_playwright instance).
  - TTL freshness gate.
  - Fallback to oil_swing_config.WARRANT_CATALOG_FALLBACK when no page +
    no cache exists.
  - Atomic catalog write on refresh.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from unittest.mock import MagicMock, patch

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import oil_warrant_refresh as owr


def _make_fake_page(post_responses=None, get_responses=None, csrf: str = "test-csrf"):
    """Build a MagicMock page whose context.request.post/get returns canned data."""
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
        suffix = url.replace(owr.API_BASE, "")
        return _build_response(post_responses.get(suffix))

    def _get(url):
        suffix = url.replace(owr.API_BASE, "")
        return _build_response(get_responses.get(suffix))

    page = MagicMock()
    page.context.request.post.side_effect = _post
    page.context.request.get.side_effect = _get
    page.context.cookies.return_value = [{"name": "AZACSRF", "value": csrf}]
    return page


# ---------------------------------------------------------------------------
# CSRF / page wiring
# ---------------------------------------------------------------------------
def test_csrf_from_page_extracts_cookie():
    page = _make_fake_page(csrf="abc123")
    assert owr._csrf_from_page(page) == "abc123"


def test_csrf_from_page_returns_none_when_missing():
    page = MagicMock()
    page.context.cookies.return_value = [{"name": "OTHER", "value": "x"}]
    assert owr._csrf_from_page(page) is None


def test_page_api_post_includes_csrf_header():
    page = _make_fake_page(post_responses={"/foo": {"ok": True}}, csrf="tok")
    result = owr._page_api_post(page, "/foo", {"q": 1})

    assert result == {"ok": True}
    page.context.request.post.assert_called_once()
    call = page.context.request.post.call_args
    assert call.kwargs["headers"]["X-SecurityToken"] == "tok"


# ---------------------------------------------------------------------------
# Search & probe
# ---------------------------------------------------------------------------
def test_search_warrants_filters_to_warrant_certificate_etf():
    page = _make_fake_page(post_responses={
        "/_api/search/filtered-search": {
            "hits": [
                {"type": "WARRANT", "orderBookId": "111"},
                {"type": "STOCK", "orderBookId": "222"},  # filtered
                {"type": "CERTIFICATE", "orderBookId": "333"},
                {"type": "ETF", "orderBookId": "444"},
            ]
        }
    })
    hits = owr._search_warrants("MINI L OLJA", page)
    assert len(hits) == 3
    assert {h["orderBookId"] for h in hits} == {"111", "333", "444"}


def test_probe_warrant_returns_none_for_zero_bid():
    page = _make_fake_page(get_responses={
        "/_api/market-guide/warrant/X": {
            "name": "MINI L OLJA AVA Z",
            "tradable": "BUYABLE_AND_SELLABLE",
            "quote": {"buy": 0, "sell": 100, "last": 50, "spread": 100},
            "keyIndicators": {"leverage": 2.0, "barrierLevel": 60.0, "parity": 1},
            "underlying": {"quote": {"last": 80}},
        }
    })
    assert owr._probe_warrant("X", page, "WARRANT") is None


def test_probe_warrant_extracts_canonical_fields():
    page = _make_fake_page(get_responses={
        "/_api/market-guide/warrant/X": {
            "name": "MINI L OLJA AVA Z",
            "isin": "SE0000",
            "tradable": "BUYABLE_AND_SELLABLE",
            "quote": {"buy": 100.0, "sell": 100.5, "last": 100.2, "spread": 0.5},
            "keyIndicators": {
                "leverage": 2.5, "barrierLevel": 60.0, "parity": 1,
                "direction": "Lång", "isAza": True, "subType": "MINI_FUTURE",
            },
            "underlying": {"quote": {"last": 80.0}},
        }
    })
    probe = owr._probe_warrant("X", page, "WARRANT")
    assert probe["name"] == "MINI L OLJA AVA Z"
    assert probe["bid"] == 100.0
    assert probe["ask"] == 100.5
    assert probe["leverage"] == 2.5
    assert probe["barrier"] == 60.0
    assert probe["api_type"] == "warrant"


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------
def test_is_valid_candidate_passes_tracker_with_tight_spread():
    probe = {
        "tradable": "BUYABLE_AND_SELLABLE",
        "api_type": "etf",
        "leverage": 1.0,
        "barrier": None,
        "bid": 100.0,
        "ask": 100.5,
        "underlying_last": 80.0,
    }
    ok, reason = owr._is_valid_candidate(probe, "LONG")
    assert ok, reason


def test_is_valid_candidate_rejects_too_close_to_barrier():
    probe = {
        "tradable": "BUYABLE_AND_SELLABLE",
        "api_type": "warrant",
        "leverage": 3.0,
        "barrier": 75.0,        # 80 -> 75 = 6.25% — too close (need 12%)
        "bid": 100.0,
        "ask": 100.5,
        "underlying_last": 80.0,
    }
    ok, reason = owr._is_valid_candidate(probe, "LONG")
    assert not ok
    assert "barrier" in reason


def test_is_valid_candidate_rejects_wide_spread():
    probe = {
        "tradable": "BUYABLE_AND_SELLABLE",
        "api_type": "warrant",
        "leverage": 3.0,
        "barrier": 50.0,        # generous distance
        "bid": 100.0,
        "ask": 105.0,           # 5% spread
        "underlying_last": 80.0,
    }
    ok, reason = owr._is_valid_candidate(probe, "LONG")
    assert not ok
    assert "spread" in reason


def test_is_valid_candidate_rejects_not_tradable():
    probe = {
        "tradable": "SUSPENDED",
        "api_type": "warrant",
        "leverage": 2.0,
        "barrier": 50.0,
        "bid": 100.0,
        "ask": 100.5,
        "underlying_last": 80.0,
    }
    ok, reason = owr._is_valid_candidate(probe, "LONG")
    assert not ok
    assert "tradable" in reason.lower()


# ---------------------------------------------------------------------------
# load_catalog_or_fetch — TTL + fallback
# ---------------------------------------------------------------------------
def test_load_catalog_returns_fresh_cache(tmp_path, monkeypatch):
    """Within TTL, load_catalog_or_fetch must NOT call refresh."""
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps({
        "refreshed_ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "ttl_hours": 6,
        "warrants": {"MINI_L_OLJA_AVA_624": {"ob_id": "2370189",
                                              "underlying": "OIL-USD",
                                              "direction": "LONG"}},
    }))
    monkeypatch.setattr(owr, "CATALOG_FILE", str(catalog_file))

    with patch.object(owr, "refresh_warrant_catalog") as refresh:
        result = owr.load_catalog_or_fetch(page=MagicMock())

    refresh.assert_not_called()
    assert "MINI_L_OLJA_AVA_624" in result


def test_load_catalog_falls_back_when_no_page_and_no_cache(tmp_path, monkeypatch):
    catalog_file = tmp_path / "missing.json"
    monkeypatch.setattr(owr, "CATALOG_FILE", str(catalog_file))

    result = owr.load_catalog_or_fetch(page=None)

    # Fallback should include seed warrants from oil_swing_config
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_load_catalog_force_refresh_bypasses_ttl(tmp_path, monkeypatch):
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps({
        "refreshed_ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "ttl_hours": 6,
        "warrants": {"OLD": {"underlying": "OIL-USD", "direction": "LONG"}},
    }))
    monkeypatch.setattr(owr, "CATALOG_FILE", str(catalog_file))

    with patch.object(owr, "refresh_warrant_catalog",
                      return_value=({"NEW": {"underlying": "OIL-USD",
                                              "direction": "LONG"}},
                                     {("OIL-USD", "LONG")})) as refresh:
        result = owr.load_catalog_or_fetch(page=MagicMock(), force_refresh=True)

    refresh.assert_called_once()
    assert "NEW" in result


def test_load_catalog_merges_uncovered_pairs_on_partial_refresh(tmp_path, monkeypatch):
    """If a refresh covers only LONG, prior SHORT entries must be preserved."""
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps({
        # Stale (>TTL) so refresh runs.
        "refreshed_ts": (datetime.datetime.now(datetime.UTC)
                         - datetime.timedelta(hours=24)).isoformat(),
        "ttl_hours": 6,
        "warrants": {
            "OLD_SHORT": {"underlying": "OIL-USD", "direction": "SHORT",
                          "ob_id": "111"},
            "OLD_LONG": {"underlying": "OIL-USD", "direction": "LONG",
                         "ob_id": "222"},
        },
    }))
    monkeypatch.setattr(owr, "CATALOG_FILE", str(catalog_file))

    with patch.object(owr, "refresh_warrant_catalog",
                      return_value=({"NEW_LONG": {"underlying": "OIL-USD",
                                                   "direction": "LONG",
                                                   "ob_id": "333"}},
                                     {("OIL-USD", "LONG")})):  # SHORT not covered
        result = owr.load_catalog_or_fetch(page=MagicMock())

    # SHORT side preserved from old cache, LONG side replaced.
    assert "OLD_SHORT" in result
    assert "NEW_LONG" in result
    # OLD_LONG should be gone (LONG was covered, replaced)
    assert "OLD_LONG" not in result


# ---------------------------------------------------------------------------
# Search query coverage
# ---------------------------------------------------------------------------
def test_search_queries_cover_long_and_short():
    pairs = {(u, d) for u, d, _ in owr.SEARCH_QUERIES}
    assert ("OIL-USD", "LONG") in pairs
    assert ("OIL-USD", "SHORT") in pairs


def test_search_queries_include_swedish_olja():
    queries = [q for _, _, q in owr.SEARCH_QUERIES]
    assert any("OLJA" in q for q in queries)


def test_search_queries_include_brent_and_wti_english():
    queries = [q for _, _, q in owr.SEARCH_QUERIES]
    assert any("BRENT" in q for q in queries)
    assert any("WTI" in q for q in queries)
