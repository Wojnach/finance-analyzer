"""Tests for dashboard/house_blueprint.py — read-only viewer at /house
over the househunting project's findapartments runs and heatmap.

Coverage:
  - Every route is gated by `require_auth` (no-cookie → 401).
  - With the auth cookie, all routes return their expected content.
  - Path validation rejects traversal attempts on run_id and slug.
  - Manifest + summary + per-candidate flows render correctly.
  - Missing run / missing slug → 404 (no 500s).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dashboard.app import COOKIE_NAME, app
from dashboard import house_blueprint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_TOKEN = "test-token-for-house-blueprint"


@pytest.fixture
def fake_house_root(tmp_path: Path) -> Path:
    """Build a synthetic househunting tree with one run + heatmap."""
    runs = tmp_path / "data" / "findapartments" / "2026-05-01-0032"
    runs.mkdir(parents=True)

    # _manifest.json
    slug_a = "lagenhet-3rum-test-slug-a-123"
    slug_b = "lagenhet-3rum-test-slug-b-456"
    # slug_c: data.json present but NO markdown report and NO address field —
    # exercises (1) the hub gating the detail link off _resolve_md (no link to
    # candidate_detail()'s 404) and (2) the address-fallback to the slug.
    slug_c = "lagenhet-3rum-test-slug-c-789"
    (runs / "_manifest.json").write_text(json.dumps([slug_a, slug_b, slug_c]))

    # _summary.thesis.md (preferred over .md). Explicit utf-8 encoding —
    # without it Windows writes cp1252 by default, which then fails the
    # route's read_text(encoding="utf-8") on the non-ASCII characters
    # (`—`, `friköpt`).
    (runs / "_summary.thesis.md").write_text(
        "# /findapartments — thesis-weighted ranking\n\n"
        "| Rank | Address | Tenure |\n|---|---|---|\n"
        "| 1 | Test 1 | friköpt |\n",   # matches slug_a's data.json address
        encoding="utf-8",
    )

    # Per-candidate reports
    (runs / f"{slug_a}.thesis.md").write_text(
        "# #1: Test Address — composite 66\n\n"
        "## Thesis fit\n\n- friköpt + skuldfri\n",
        encoding="utf-8",
    )
    # Slug B has only the legacy non-thesis report; verify fallback works.
    (runs / f"{slug_b}.md").write_text(
        "# Test Address B\n\nLegacy report.\n",
        encoding="utf-8",
    )

    # _raw/<slug>/data.json — slug_a carries the full structured shape so the
    # table-cell rendering (Score, kr/m², CAGR%, vs-fair) is actually asserted.
    raw_a = runs / "_raw" / slug_a
    raw_a.mkdir(parents=True)
    (raw_a / "data.json").write_text(json.dumps({
        "slug": slug_a, "address": "Test 1",
        "url": "https://www.hemnet.se/bostad/test-a-123",
        "price": 6_995_000, "sqm": 70, "fee": 3210, "construction_year": 1929,
        "composite_score": {"composite": 66},
        "weighted_cagr": {"composite": 2.6},
        "bid_advisor": {"fair_value": 7_300_000},   # price < fair → -4% (cheap)
        "premium_structured": {"tier": 1},
        "premium_llm": {"tier": 2},
    }))
    # slug_c has data.json (so the hub renders a row) but no `address` key and
    # no markdown report — see slug_c comment above. Lower score so it sorts
    # after slug_a; slug_b has no data.json so its score is None (sorts last).
    raw_c = runs / "_raw" / slug_c
    raw_c.mkdir(parents=True)
    (raw_c / "data.json").write_text(json.dumps({
        "slug": slug_c, "price": 4_500_000, "sqm": 60,
        "composite_score": {"composite": 40},
    }))

    # output/heatmap.html
    output = tmp_path / "output"
    output.mkdir()
    (output / "heatmap.html").write_text(
        "<!doctype html><html><body>FAKE HEATMAP</body></html>"
    )

    # data/kellgrensgatan/CURRENT_APARTMENT_BRIEF.md — the /house/k10 page.
    k10 = tmp_path / "data" / "kellgrensgatan"
    k10.mkdir(parents=True)
    (k10 / "CURRENT_APARTMENT_BRIEF.md").write_text(
        "# Current apartment brief — Kellgrensgatan 10\n\nBRF Gladan baseline.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def client(fake_house_root: Path):
    """Flask test client with token auth + house_root pointed at the
    synthetic tree. Uses `mock.patch` on `_get_config` so we don't have
    to mutate the real config.json (or worry about its 60s cache)."""
    app.config["TESTING"] = True

    def fake_get_config():
        return {"dashboard_token": _TOKEN, "house_root": str(fake_house_root)}

    # Patch auth's _get_config so require_auth's token check uses the fake;
    # patch house_blueprint's _get_config so route handlers (e.g. _house_root)
    # also see the fake. Both are necessary after the 2026-05-02 refactor that
    # split auth out of dashboard.app — neither module looks at app.py's
    # _get_config anymore.
    with patch("dashboard.auth._get_config", fake_get_config), \
         patch("dashboard.house_blueprint._get_config", fake_get_config), \
         app.test_client() as c:
        yield c


def _auth_cookie() -> dict:
    return {COOKIE_NAME: _TOKEN}


# ---------------------------------------------------------------------------
# Auth gate — every route must reject unauthenticated requests
# ---------------------------------------------------------------------------


# Blueprint registers `/` (rendered as `/house/` with the prefix). Flask
# 308-redirects `/house` → `/house/` before any view runs, so we probe
# the trailing-slash form to actually hit the auth gate.
_HOUSE_ROUTES_TO_PROBE = [
    "/house/",
    "/house/runs",
    "/house/runs/2026-05-01-0032",
    "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123",
    "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123/raw",
    "/house/runs/2026-05-01-0032/_manifest.json",
    "/house/heatmap",
    "/house/k10",
    "/house/api/runs",
    "/house/api/runs/2026-05-01-0032",
    "/house/api/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123",
]


@pytest.mark.parametrize("path", _HOUSE_ROUTES_TO_PROBE)
def test_route_requires_auth(client, path):
    """No cookie → 401. Catches future routes added without auth."""
    resp = client.get(path)
    assert resp.status_code == 401, (
        f"{path} returned {resp.status_code} without auth — "
        f"missing @require_auth?"
    )


def test_blueprint_url_map_covers_probe_list():
    """Sanity: the probed routes match the blueprint's actual URL map.
    If a new route is added without updating _HOUSE_ROUTES_TO_PROBE, this
    fails — keeping the auth-gate parametrization comprehensive."""
    bp_rules = {
        rule.rule for rule in app.url_map.iter_rules()
        if rule.endpoint.startswith("house.")
    }
    # Strip <var> placeholders to compare against concrete probe paths.
    probed = set()
    for path in _HOUSE_ROUTES_TO_PROBE:
        probed.add(path)
    # Every blueprint rule should have at least one probed path that
    # would match it (handle <run_id>/<slug> etc.).
    bp_static = {
        r for r in bp_rules
        if "<" not in r
    }
    bp_dynamic = bp_rules - bp_static
    # Static rules must literally appear in our probe list.
    assert bp_static.issubset(probed), (
        f"Static blueprint rules not covered by probe list: "
        f"{bp_static - probed}"
    )
    # Dynamic rules must each match at least one probe path.
    import re as _re
    for rule_pattern in bp_dynamic:
        pat = _re.sub(r"<[^>]+>", r"[^/]+", rule_pattern)
        regex = _re.compile(f"^{pat}$")
        if not any(regex.match(p) for p in probed):
            pytest.fail(f"Dynamic rule {rule_pattern!r} not covered by any probe path")


# ---------------------------------------------------------------------------
# Happy-path content
# ---------------------------------------------------------------------------


def test_index_renders_hub(client):
    """The /house/ index is the hub: a ranked apartment table built from each
    candidate's _raw/<slug>/data.json, plus the embedded heatmap."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/")
    assert resp.status_code == 200
    body = resp.data.decode()

    # --- table cells are load-bearing: each derives from slug_a's data.json ---
    assert "Test 1" in body                        # address field
    assert ">66</td>" in body                       # composite_score.composite
    assert "99 929" in body                    # kr/m² = 6_995_000 / 70 (nbsp-grouped)
    assert ">2.6</td>" in body                      # weighted_cagr.composite
    assert "-4%" in body                            # price 6.995M vs fair 7.30M
    assert "https://www.hemnet.se/bostad/test-a-123" in body   # Hemnet ↗

    # --- row ordering: slug_a (66) sorts above slug_c (40) ---
    assert body.index(">66</td>") < body.index(">40</td>")

    # --- slug_a has a report → linked; slug_c has data.json but NO report →
    # rendered but NOT linked (the high-sev fix: never link to a 404). ---
    assert "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123" in body
    assert "lagenhet-3rum-test-slug-c-789" in body
    assert "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-c-789" not in body

    # --- heatmap embedded + linked ---
    assert "/house/heatmap" in body
    assert "<iframe" in body

    # --- build/revision marker in the footer (so we can tell which build is live) ---
    assert "rev " in body
    assert "<footer" in body

    # --- Est. value column (bid_advisor.fair_value = 7.30M for slug_a) ---
    assert "Est." in body
    assert "7.30M" in body

    # --- sortable tables: the sort script + the K10 link are present ---
    assert "data-sort" in body          # from _SORT_JS
    assert "/house/k10" in body          # current-home link (footer + hub card)


def test_index_strips_token_query(client):
    """?token= bootstrap on the hub redirects to a clean, token-free URL."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get(f"/house/?token={_TOKEN}", follow_redirects=False)
    assert resp.status_code == 302
    loc = resp.headers["Location"]
    assert loc.endswith("/house")
    assert "token" not in loc          # the whole point — token must not leak
    assert _TOKEN not in loc


def test_index_empty_runs_shows_heatmap_link(tmp_path):
    """No findapartments runs → empty-state hub still serves and links the
    heatmap (the rewritten no-runs branch)."""
    app.config["TESTING"] = True
    cfg = {"dashboard_token": _TOKEN, "house_root": str(tmp_path)}
    with patch("dashboard.auth._get_config", lambda: cfg), \
         patch("dashboard.house_blueprint._get_config", lambda: cfg), \
         app.test_client() as c:
        c.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
        resp = c.get("/house/")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "No findapartments runs yet" in body
    assert "/house/heatmap" in body


def test_runs_list_renders(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "2026-05-01-0032" in body
    assert "3 candidates" in body  # from manifest length


def test_run_detail_links_summary_addresses_to_hemnet(client):
    """Each Addr cell in the ranked-comparison table links to the candidate's
    Hemnet listing (url from data.json), matched by exact address."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    body = client.get("/house/runs/2026-05-01-0032").get_data(as_text=True)
    assert '<a href="https://www.hemnet.se/bostad/test-a-123"' in body
    assert ">Test 1</a>" in body
    # The plain (unlinked) cell must be gone.
    assert "<td>Test 1</td>" not in body


def test_run_detail_drops_slug_list(client):
    """The raw 'All candidates' slug list is removed (redundant now that the
    table addresses link to Hemnet); summary + heatmap footer link remain."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs/2026-05-01-0032")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "thesis-weighted" in body                 # summary still renders
    assert "All candidates" not in body               # slug list gone
    assert "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123" not in body
    assert "/house/heatmap" in body                   # heatmap footer link kept


def test_candidate_detail_renders_thesis_md(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Thesis fit" in body
    assert "friköpt" in body
    # Raw-link footer
    assert "/raw" in body


def test_candidate_detail_falls_back_to_legacy_md(client):
    """Slug B has only `<slug>.md`, no `<slug>.thesis.md`. Should still 200."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-b-456")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Legacy report" in body


def test_k10_renders_current_apartment_brief(client):
    """/house/k10 renders data/kellgrensgatan/CURRENT_APARTMENT_BRIEF.md."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/k10")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Kellgrensgatan 10" in body
    assert "BRF Gladan baseline" in body


def test_format_oneliners_breaks_blob_into_list():
    """The run-together Top-5 paragraph becomes a bulleted list."""
    html = (
        "<h2>Top-5 one-liners</h2>"
        "<p><strong>#1</strong> — A: score 61. "
        "<strong>#2</strong> — B: score 57. "
        "<strong>#3</strong> — C: score 52.</p>"
    )
    out = house_blueprint._format_oneliners(html)
    assert '<ul class="oneliners">' in out
    assert out.count("<li>") == 3
    assert "<p><strong>#1</strong>" not in out   # original blob paragraph gone


def test_candidate_raw_returns_data_json(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get(
        "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123/raw"
    )
    assert resp.status_code == 200
    payload = json.loads(resp.get_data())
    assert payload["address"] == "Test 1"


def test_heatmap_returns_html(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/heatmap")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    assert b"FAKE HEATMAP" in resp.get_data()


def test_api_runs_returns_json(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/api/runs")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["runs"][0]["run_id"] == "2026-05-01-0032"
    assert payload["runs"][0]["candidate_count"] == 3


def test_api_run_returns_candidate_list(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/api/runs/2026-05-01-0032")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["run_id"] == "2026-05-01-0032"
    assert "lagenhet-3rum-test-slug-a-123" in payload["candidates"]


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_run", [
    "../../etc",
    "2026-05-01-0032..",
    "abc",                # wrong shape
    "2026.05.01",         # dots, not hyphens
    "2026-13-01-0032",    # well, regex doesn't validate month range
                          # but the dir won't exist → 404 either way
])
def test_invalid_run_id_returns_404(client, bad_run):
    """Run-IDs that fail the regex shape OR don't match a real dir → 404.

    Note: variants containing literal `/` (e.g. `/etc/passwd`,
    `2026/05/01`) are rejected upstream by Flask's URL normalization
    with a 308 redirect — those never reach our handler. The cases
    above all stay within a single path segment so they exercise
    `_validate_run_id`."""
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get(f"/house/runs/{bad_run}")
    assert resp.status_code == 404


@pytest.mark.parametrize("bad_slug", [
    "..",
    "../etc",
    "/etc/passwd",
    "x",   # too short
    "Foo_Bar",  # underscore + caps not in slug shape
])
def test_invalid_slug_returns_404(client, bad_slug):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get(f"/house/runs/2026-05-01-0032/{bad_slug}")
    assert resp.status_code == 404


def test_missing_run_returns_404(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs/2099-01-01-0000")
    assert resp.status_code == 404


def test_missing_candidate_returns_404(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get(
        "/house/runs/2026-05-01-0032/lagenhet-3rum-nonexistent-slug-999"
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Hub data loader — the tolerant branches that keep the landing page from 500ing
# ---------------------------------------------------------------------------


def test_load_candidates_tolerates_garbage(tmp_path):
    """A non-dict data.json, corrupt bytes, and bad/non-string manifest entries
    must NOT raise — good rows survive, junk slugs are dropped."""
    run = tmp_path / "data" / "findapartments" / "2026-05-01-0032"
    good = run / "_raw" / "good-slug-aaa-1"
    good.mkdir(parents=True)
    (good / "data.json").write_text(json.dumps(
        {"address": "Good", "price": 1_000_000, "composite_score": {"composite": 50}}
    ))
    nondict = run / "_raw" / "listjson-slug-2"
    nondict.mkdir(parents=True)
    (nondict / "data.json").write_text("[1, 2, 3]")           # valid JSON, not a dict
    corrupt = run / "_raw" / "corrupt-slug-3"
    corrupt.mkdir(parents=True)
    (corrupt / "data.json").write_bytes(b"\xff\xfe not json") # undecodable / invalid
    (run / "_manifest.json").write_text(json.dumps([
        "good-slug-aaa-1", "listjson-slug-2", "corrupt-slug-3",
        "../etc/passwd", 123, "ab",   # bad slugs: traversal, non-str, too short
    ]))

    cfg = {"house_root": str(tmp_path)}
    with patch("dashboard.house_blueprint._get_config", lambda: cfg):
        rows = house_blueprint._load_candidates("2026-05-01-0032")

    slugs = [r["slug"] for r in rows]
    assert "good-slug-aaa-1" in slugs        # full row
    assert "listjson-slug-2" in slugs        # non-dict tolerated → slug fallback
    assert "corrupt-slug-3" in slugs         # corrupt bytes tolerated
    assert "../etc/passwd" not in slugs      # dropped by _SLUG_RE (traversal)
    assert 123 not in slugs                  # dropped (non-string)
    assert "ab" not in slugs                 # dropped (too short for _SLUG_RE)
    good_row = next(r for r in rows if r["slug"] == "good-slug-aaa-1")
    assert good_row["address"] == "Good"
    assert good_row["score"] == 50


def test_load_candidates_missing_manifest_returns_empty(tmp_path):
    (tmp_path / "data" / "findapartments" / "2026-05-01-0032").mkdir(parents=True)
    cfg = {"house_root": str(tmp_path)}
    with patch("dashboard.house_blueprint._get_config", lambda: cfg):
        assert house_blueprint._load_candidates("2026-05-01-0032") == []


def test_load_candidates_non_list_manifest_returns_empty(tmp_path):
    run = tmp_path / "data" / "findapartments" / "2026-05-01-0032"
    run.mkdir(parents=True)
    (run / "_manifest.json").write_text(json.dumps({"not": "a list"}))
    cfg = {"house_root": str(tmp_path)}
    with patch("dashboard.house_blueprint._get_config", lambda: cfg):
        assert house_blueprint._load_candidates("2026-05-01-0032") == []
