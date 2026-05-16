"""House-hunting dashboard — read-only viewer at /house over the
househunting project's findapartments runs and innerstad heatmap.

Mounted as a Flask Blueprint on the main dashboard app. Reuses the same
`require_auth` cookie/token gate, so the entire `/house/*` surface is
protected by the same `pf_dashboard_token` that gates the finance dashboard.

Reads files directly from disk under `<house_root>`:
  - <house_root>/data/findapartments/<run-id>/_manifest.json
  - <house_root>/data/findapartments/<run-id>/_summary.thesis.md  (preferred)
                                            /_summary.md           (fallback)
  - <house_root>/data/findapartments/<run-id>/<slug>.thesis.md     (preferred)
                                            /<slug>.md             (fallback)
  - <house_root>/data/findapartments/<run-id>/_raw/<slug>/data.json
  - <house_root>/output/heatmap.html

`house_root` is configured via `config.json[house_root]`, defaulting to
`Q:\\househunting`. The blueprint never imports from the househunting
project — it's a pure file viewer, so the two repos stay decoupled.

SECURITY: Every route is wrapped with `@require_auth`. There's a unit
test (`test_dashboard_house.py::test_every_route_requires_auth`) that
walks the blueprint's URL map and asserts each route returns 401 without
a cookie. Asset files are streamed through authenticated routes —
NEVER copied or symlinked into `dashboard/static/` (that path is served
by Flask's default static handler without auth, per docs/TUNNEL_SETUP.md).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional

from flask import (
    Blueprint, abort, jsonify, redirect, request, send_file,
)
from werkzeug.utils import secure_filename

import markdown as md_lib  # type: ignore[import-not-found]

from dashboard.auth import _get_config, require_auth

bp = Blueprint("house", __name__, url_prefix="/house")


# ---------------------------------------------------------------------------
# Config + path helpers
# ---------------------------------------------------------------------------


def _house_root() -> Path:
    """Root of the house-hunting project on disk. Configurable via
    `config.json[house_root]`. Defaults to Q:\\househunting (the local
    canonical path on the dashboard host)."""
    cfg = _get_config()
    return Path(cfg.get("house_root", r"Q:\househunting"))


def _runs_dir() -> Path:
    return _house_root() / "data" / "findapartments"


def _heatmap_path() -> Path:
    return _house_root() / "output" / "heatmap.html"


# Run IDs look like 2026-05-01-0032 — YYYY-MM-DD-HHMM. Slugs look like
# lagenhet-3rum-kungsholmen-... — lowercase ASCII + digits + hyphens.
# Both are URL-safe but we still validate before using as a path component.
_RUN_ID_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(?:-[0-9]{4})?$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,200}$")


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        abort(404)
    return run_id


def _validate_slug(slug: str) -> str:
    # secure_filename strips path traversal; the regex enforces shape.
    cleaned = secure_filename(slug)
    if cleaned != slug or not _SLUG_RE.match(cleaned):
        abort(404)
    return cleaned


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _list_runs() -> list[dict]:
    """List all run directories newest-first. Returns empty list if the
    findapartments dir doesn't exist (fresh install, etc.)."""
    runs_dir = _runs_dir()
    if not runs_dir.exists():
        return []
    runs: list[dict] = []
    for entry in runs_dir.iterdir():
        if not entry.is_dir() or not _RUN_ID_RE.match(entry.name):
            continue
        manifest = entry / "_manifest.json"
        try:
            slugs = json.loads(manifest.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            slugs = []
        runs.append({
            "run_id": entry.name,
            "candidate_count": len(slugs),
            "has_summary": (entry / "_summary.thesis.md").exists()
                           or (entry / "_summary.md").exists(),
            "modified_iso": datetime.fromtimestamp(
                entry.stat().st_mtime
            ).isoformat(timespec="seconds"),
        })
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


def _resolve_md(run_id: str, basename: str) -> Optional[Path]:
    """Resolve <run-id>/<basename>.thesis.md, falling back to .md."""
    base = _runs_dir() / run_id
    thesis = base / f"{basename}.thesis.md"
    if thesis.exists():
        return thesis
    legacy = base / f"{basename}.md"
    if legacy.exists():
        return legacy
    return None


# ---------------------------------------------------------------------------
# HTML shell — minimal, mobile-friendly
# ---------------------------------------------------------------------------


_PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI",
        system-ui, sans-serif;
  max-width: 900px; margin: 0 auto; padding: 16px 20px 80px;
  color: #222; background: #fff;
}
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #111; }
  a { color: #6cf; }
  a:visited { color: #c9f; }
  table { border-color: #333; }
  th, td { border-color: #333; }
  thead th { background: #1a1a1a; }
  code { background: #1a1a1a; }
}
nav.crumbs { font-size: 14px; padding: 8px 0; border-bottom: 1px solid #ddd;
             margin-bottom: 16px; }
nav.crumbs a { margin-right: 8px; }
h1 { font-size: 26px; line-height: 1.25; margin-top: 8px; }
h2 { font-size: 20px; margin-top: 28px; padding-bottom: 4px;
     border-bottom: 1px solid #eee; }
h3 { font-size: 17px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left;
         vertical-align: top; font-size: 14px; }
thead th { background: #f4f4f4; }
code, pre { font-family: ui-monospace, "Menlo", "Consolas", monospace;
            font-size: 13px; }
pre { background: #f6f8fa; padding: 10px; overflow-x: auto;
      border-radius: 4px; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
blockquote { border-left: 3px solid #88c; padding: 4px 12px;
             margin: 12px 0; color: #444; }
@media (prefers-color-scheme: dark) {
  blockquote { color: #aaa; }
}
.runs-list { list-style: none; padding: 0; }
.runs-list li { padding: 10px 12px; border: 1px solid #ddd;
                border-radius: 4px; margin-bottom: 8px; }
.runs-list .meta { color: #666; font-size: 13px; }
@media (prefers-color-scheme: dark) {
  .runs-list li { border-color: #333; }
  .runs-list .meta { color: #aaa; }
}
"""


def _shell(title: str, body_html: str, breadcrumbs: list[tuple[str, str]]) -> str:
    crumbs_html = " ›\n".join(
        f'<a href="{escape(href)}">{escape(label)}</a>'
        for href, label in breadcrumbs
    )
    return (
        f"<!doctype html>\n"
        f"<html lang=\"en\"><head>\n"
        f"<meta charset=\"utf-8\">\n"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{escape(title)}</title>\n"
        f"<style>{_PAGE_CSS}</style>\n"
        f"</head><body>\n"
        f"<nav class=\"crumbs\">{crumbs_html}</nav>\n"
        f"{body_html}\n"
        f"</body></html>\n"
    )


def _render_markdown(text: str) -> str:
    return md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )


# ---------------------------------------------------------------------------
# Routes — HTML
# ---------------------------------------------------------------------------


@bp.route("/")
@require_auth
def index():
    """Land on the most recent run; if no runs exist, show the empty list."""
    runs = _list_runs()
    if request.args.get("token"):
        # User auth-bootstrapped via ?token= — strip it from the URL.
        return redirect("/house", code=302)
    if not runs:
        body = (
            "<h1>House</h1>"
            "<p>No findapartments runs yet. From "
            "<code>Q:\\househunting</code>, run "
            "<code>.venv\\Scripts\\python -m scripts.findapartments_scan</code>.</p>"
        )
        return _shell("House — no runs", body, [("/house", "House")])
    return redirect(f"/house/runs/{runs[0]['run_id']}", code=302)


@bp.route("/runs")
@require_auth
def runs_list():
    runs = _list_runs()
    if not runs:
        body = "<h1>House</h1><p>No findapartments runs yet.</p>"
    else:
        items = []
        for r in runs:
            items.append(
                f"<li>"
                f"<a href=\"/house/runs/{escape(r['run_id'])}\">"
                f"{escape(r['run_id'])}</a>"
                f"<div class=\"meta\">{r['candidate_count']} candidates · "
                f"updated {escape(r['modified_iso'])}"
                f"{' · summary available' if r['has_summary'] else ''}"
                f"</div></li>"
            )
        body = (
            f"<h1>House — {len(runs)} run(s)</h1>"
            f"<ul class=\"runs-list\">{''.join(items)}</ul>"
            f"<p><a href=\"/house/heatmap\">Innerstad appreciation heatmap →</a></p>"
        )
    return _shell(
        "House — runs",
        body,
        [("/", "Dashboard"), ("/house", "House"), ("/house/runs", "Runs")],
    )


@bp.route("/runs/<run_id>")
@require_auth
def run_detail(run_id: str):
    run_id = _validate_run_id(run_id)
    summary = _resolve_md(run_id, "_summary")
    if not summary:
        abort(404)
    text = summary.read_text(encoding="utf-8")
    # Rewrite plain-text references to candidate slugs into hyperlinks. The
    # summary's table column is `<address>` text — slugs themselves aren't
    # in the rendered table but ARE in the per-candidate report file names.
    # Add a "candidates" footer with explicit links sourced from the manifest.
    manifest = _runs_dir() / run_id / "_manifest.json"
    candidate_links = ""
    try:
        slugs = json.loads(manifest.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        slugs = []
    if slugs:
        link_items = "".join(
            # Suppressed false-positive: Values are escape()d via markupsafe before interpolation; pattern false positive.
            # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
            f"<li><a href=\"/house/runs/{escape(run_id)}/{escape(s)}\">"
            f"{escape(s)}</a></li>"
            for s in slugs
        )
        candidate_links = (
            # Suppressed false-positive: Static count + already-escaped link_items; behind require_auth.
            # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
            f"<h2>All candidates ({len(slugs)})</h2>"
            f"<ul>{link_items}</ul>"
            # Suppressed false-positive: escape(run_id) used; run_id also validated by _validate_run_id earlier in handler.
            # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
            f"<p><a href=\"/house/runs/{escape(run_id)}/_manifest.json\">"
            f"manifest.json</a> · "
            f"<a href=\"/house/heatmap\">heatmap</a></p>"
        )
    body = _render_markdown(text) + candidate_links
    return _shell(
        f"House — {run_id}",
        body,
        [("/", "Dashboard"), ("/house/runs", "Runs"), (f"/house/runs/{run_id}", run_id)],
    )


@bp.route("/runs/<run_id>/_manifest.json")
@require_auth
def run_manifest(run_id: str):
    run_id = _validate_run_id(run_id)
    manifest = _runs_dir() / run_id / "_manifest.json"
    if not manifest.exists():
        abort(404)
    return send_file(manifest, mimetype="application/json")


@bp.route("/runs/<run_id>/<slug>")
@require_auth
def candidate_detail(run_id: str, slug: str):
    run_id = _validate_run_id(run_id)
    slug = _validate_slug(slug)
    md_path = _resolve_md(run_id, slug)
    if not md_path:
        abort(404)
    text = md_path.read_text(encoding="utf-8")
    body = (
        _render_markdown(text)
        # Suppressed false-positive: escape(run_id) and escape(slug); both validated via _validate_* before this line.
        # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
        + f"<p><a href=\"/house/runs/{escape(run_id)}/{escape(slug)}/raw\">"
          "raw data.json →</a></p>"
    )
    return _shell(
        f"House — {slug}",
        body,
        [
            ("/", "Dashboard"),
            ("/house/runs", "Runs"),
            (f"/house/runs/{run_id}", run_id),
            (f"/house/runs/{run_id}/{slug}", slug),
        ],
    )


@bp.route("/runs/<run_id>/<slug>/raw")
@require_auth
def candidate_raw(run_id: str, slug: str):
    run_id = _validate_run_id(run_id)
    slug = _validate_slug(slug)
    raw = _runs_dir() / run_id / "_raw" / slug / "data.json"
    if not raw.exists():
        abort(404)
    return send_file(raw, mimetype="application/json")


@bp.route("/heatmap")
@require_auth
def heatmap():
    path = _heatmap_path()
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="text/html")


# ---------------------------------------------------------------------------
# Routes — JSON API (for programmatic / phone clients)
# ---------------------------------------------------------------------------


@bp.route("/api/runs")
@require_auth
def api_runs():
    return jsonify({"runs": _list_runs()})


@bp.route("/api/runs/<run_id>")
@require_auth
def api_run(run_id: str):
    run_id = _validate_run_id(run_id)
    manifest = _runs_dir() / run_id / "_manifest.json"
    if not manifest.exists():
        abort(404)
    try:
        slugs = json.loads(manifest.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        abort(500)
    return jsonify({"run_id": run_id, "candidates": slugs})


@bp.route("/api/runs/<run_id>/<slug>")
@require_auth
def api_candidate(run_id: str, slug: str):
    run_id = _validate_run_id(run_id)
    slug = _validate_slug(slug)
    raw = _runs_dir() / run_id / "_raw" / slug / "data.json"
    if not raw.exists():
        abort(404)
    return send_file(raw, mimetype="application/json")
