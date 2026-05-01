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
    (runs / "_manifest.json").write_text(json.dumps([slug_a, slug_b]))

    # _summary.thesis.md (preferred over .md)
    (runs / "_summary.thesis.md").write_text(
        "# /findapartments — thesis-weighted ranking\n\n"
        "| Rank | Address | Tenure |\n|---|---|---|\n"
        "| 1 | Test Address 1 | friköpt |\n"
    )

    # Per-candidate reports
    (runs / f"{slug_a}.thesis.md").write_text(
        "# #1: Test Address — composite 66\n\n"
        "## Thesis fit\n\n- friköpt + skuldfri\n"
    )
    # Slug B has only the legacy non-thesis report; verify fallback works.
    (runs / f"{slug_b}.md").write_text(
        "# Test Address B\n\nLegacy report.\n"
    )

    # _raw/<slug>/data.json
    raw_a = runs / "_raw" / slug_a
    raw_a.mkdir(parents=True)
    (raw_a / "data.json").write_text(json.dumps({
        "slug": slug_a, "address": "Test 1", "price": 6_995_000,
    }))

    # output/heatmap.html
    output = tmp_path / "output"
    output.mkdir()
    (output / "heatmap.html").write_text(
        "<!doctype html><html><body>FAKE HEATMAP</body></html>"
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

    with patch("dashboard.app._get_config", fake_get_config), \
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


def test_index_redirects_to_most_recent_run(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/house/runs/2026-05-01-0032")


def test_runs_list_renders(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "2026-05-01-0032" in body
    assert "2 candidates" in body  # from manifest length


def test_run_detail_renders_summary_and_candidate_links(client):
    client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
    resp = client.get("/house/runs/2026-05-01-0032")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Summary content
    assert "thesis-weighted" in body
    # Both candidate slugs link
    assert "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-a-123" in body
    assert "/house/runs/2026-05-01-0032/lagenhet-3rum-test-slug-b-456" in body
    # Manifest link present
    assert "_manifest.json" in body


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
    assert payload["runs"][0]["candidate_count"] == 2


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
