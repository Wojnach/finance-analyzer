"""Skeleton-integrity tests for the mobile dashboard.

Verifies the new index.html declares everything the runtime needs:
manifest link, viewport-fit=cover, theme-color, apple-touch-icon, Chart.js
UMD before the module entry, all 5 CSS files linked, the bottom-nav and
bottom-sheet shell elements.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")


REQUIRED_HEAD_FRAGMENTS = [
    'name="viewport"',
    "viewport-fit=cover",
    'name="theme-color"',
    'name="apple-mobile-web-app-capable"',
    'rel="apple-touch-icon"',
    'rel="manifest"',
    "/static/manifest.webmanifest",
    "/static/css/tokens.css",
    "/static/css/base.css",
    "/static/css/layout.css",
    "/static/css/components.css",
    "/static/css/responsive.css",
    "chart.umd.min.js",
    'type="module"',
    "/static/js/main.js",
]


@pytest.mark.parametrize("fragment", REQUIRED_HEAD_FRAGMENTS)
def test_skeleton_contains(fragment):
    assert fragment in INDEX, f"index.html missing required fragment: {fragment}"


def test_skeleton_chart_js_loaded_before_module():
    """Chart.js must register window.Chart BEFORE the module entry runs."""
    chart_pos = INDEX.find("chart.umd.min.js")
    module_pos = INDEX.find("/static/js/main.js")
    assert chart_pos > 0 and module_pos > 0
    assert chart_pos < module_pos, (
        "Chart.js UMD must be loaded before the ES module entry "
        "(modules will check window.Chart at first chart render)."
    )


def test_skeleton_bottom_nav_has_four_items():
    """Mobile-first nav must have Home / Decisions / Signals / More."""
    for route in ("home", "decisions", "signals", "more"):
        assert f'data-route="{route}"' in INDEX, f"bottom-nav missing route: {route}"


def test_skeleton_has_bottom_sheet_container():
    """The universal long-press bottom-sheet element must exist in the shell."""
    assert 'id="bottom-sheet"' in INDEX
    assert 'id="bottom-sheet-content"' in INDEX
    assert "data-bottom-sheet-close" in INDEX


def test_skeleton_links_legacy_for_fallback():
    assert "/legacy" in INDEX, "Skeleton must mention /legacy as a fallback"


def test_skeleton_size_under_target():
    """Cold first-paint budget (Track-6): ≤100 KB. The shell HTML alone
    should be a small fraction of that."""
    assert len(INDEX.encode("utf-8")) < 8_000, (
        f"index.html is {len(INDEX)} bytes — should be under 8KB. "
        "Don't inline view content into the skeleton."
    )
