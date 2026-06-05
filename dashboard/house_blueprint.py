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

# Build marker shown in every page footer. The dashboard has NO auto-reload, so
# this is how we tell which build is actually live: it's the mtime of this
# source file, captured once at import — it changes only when the file is
# redeployed AND the Flask process restarts. If the footer rev hasn't changed,
# you're still on the old process.
_REVISION = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime(
    "%Y-%m-%d %H:%M:%S"
)


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
thead th { cursor: pointer; user-select: none; white-space: nowrap; }
thead th:hover { text-decoration: underline; }
thead th[data-sort="asc"]::after  { content: " ▲"; font-size: 10px; }
thead th[data-sort="desc"]::after { content: " ▼"; font-size: 10px; }
.oneliners { margin: 8px 0; padding-left: 20px; }
.oneliners li { margin: 5px 0; }
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
p.meta { color: #666; font-size: 13px; margin-top: 4px; }
span.meta { color: #666; font-size: 13px; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.cheap { color: #1a7f37; font-weight: 600; }
.rich  { color: #c0392b; }
.score { font-weight: 700; }
details.sold { margin: 14px 0; }
details.sold summary { cursor: pointer; color: #666; font-weight: 600; }
details.sold ul { margin: 6px 0; }
iframe.heatmap { width: 100%; height: 70vh; border: 1px solid #ccc;
                 border-radius: 6px; }
@media (prefers-color-scheme: dark) {
  p.meta, span.meta { color: #aaa; }
  .cheap { color: #3fb950; }
  .rich  { color: #f85149; }
  iframe.heatmap { border-color: #333; }
}
"""


# Client-side sortable tables: click a header to sort by that column, click
# again to flip direction. Numeric-aware (handles "7.70M", "99 929" with nbsp,
# "-4%", "n/a"/"—" sort last). Applies to every table on the page — the hub
# apartment table and the run-summary ranked-comparison table alike.
_SORT_JS = r"""<script>
(function(){
  function num(td){
    var t=(td.textContent||"").replace(/ /g,"").trim();
    if(!t||t==="—"||t.toLowerCase()==="n/a") return null;
    var mult=/m$/i.test(t)?1e6:1;
    var n=parseFloat(t.replace(/[^0-9.\-]/g,""));
    return isNaN(n)?null:n*mult;
  }
  document.querySelectorAll("table").forEach(function(table){
    var thead=table.tHead, tbody=table.tBodies[0];
    if(!thead||!tbody||!thead.rows.length) return;
    var ths=thead.rows[0].cells;
    Array.prototype.forEach.call(ths,function(th,ci){
      th.addEventListener("click",function(){
        var asc=th.getAttribute("data-sort")!=="asc";
        Array.prototype.forEach.call(ths,function(o){o.removeAttribute("data-sort");});
        th.setAttribute("data-sort",asc?"asc":"desc");
        var rows=Array.prototype.slice.call(tbody.rows);
        rows.sort(function(a,b){
          var x=num(a.cells[ci]),y=num(b.cells[ci]);
          if(x!==null&&y!==null) return asc?x-y:y-x;
          if(x!==null) return -1;
          if(y!==null) return 1;
          var sx=(a.cells[ci].textContent||"").trim().toLowerCase();
          var sy=(b.cells[ci].textContent||"").trim().toLowerCase();
          return asc?sx.localeCompare(sy):sy.localeCompare(sx);
        });
        rows.forEach(function(r){tbody.appendChild(r);});
      });
    });
  });
})();
</script>"""


def _shell(title: str, body_html: str, breadcrumbs: list[tuple[str, str]],
           wide: bool = False) -> str:
    crumbs_html = " ›\n".join(
        f'<a href="{escape(href)}">{escape(label)}</a>'
        for href, label in breadcrumbs
    )
    # The hub (apartment table + heatmap) needs more horizontal room than the
    # 900px reading column used by the markdown report pages.
    wide_css = "<style>body{max-width:1200px}</style>" if wide else ""
    return (
        f"<!doctype html>\n"
        f"<html lang=\"en\"><head>\n"
        f"<meta charset=\"utf-8\">\n"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{escape(title)}</title>\n"
        f"<style>{_PAGE_CSS}</style>\n"
        f"{wide_css}\n"
        f"</head><body>\n"
        f"<nav class=\"crumbs\">{crumbs_html}</nav>\n"
        f"{body_html}\n"
        f"{_SORT_JS}\n"
        f"<footer style=\"margin-top:40px;padding-top:8px;"
        f"border-top:1px solid #444;color:#888;font-size:12px\">"
        f"rev {escape(_REVISION)} · "
        f"<a href=\"/house/k10\">current home (K10)</a></footer>\n"
        f"</body></html>\n"
    )


# Bare http(s) URL not already inside a tag/attribute or an anchor's text. The
# lookbehind rejects the three contexts a URL is already linked in: href="URL
# ("), >URL (anchor text after the opening tag), =URL (any attribute). The char
# class stops at the next tag/space/quote/paren so trailing markup isn't swallowed.
_BARE_URL_RE = re.compile(r"""(?<![">=])(https?://[^\s<>"')]+)""")


def _autolink(html: str) -> str:
    """Wrap bare URLs in rendered markdown as clickable links. Python-Markdown
    has no core autolinker, so report lines like `Hemnet: https://…` render as
    dead text — this makes them clickable without a new dependency."""
    # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
    return _BARE_URL_RE.sub(
        r'<a href="\1" target="_blank" rel="noopener">\1</a>', html
    )


def _render_markdown(text: str) -> str:
    return _autolink(md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    ))


# ---------------------------------------------------------------------------
# Hub — ranked apartment table from each candidate's _raw/<slug>/data.json
# ---------------------------------------------------------------------------


def _nested_num(data: dict, *keys: str):
    """Walk nested dict keys; return the leaf only if it's a real number."""
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def _num_or_none(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _candidate_row(run_id: str, slug: str, data: dict) -> dict:
    """Flatten one candidate's data.json into the table-row fields. Every
    field is optional — the findapartments pipeline writes partial records
    (e.g. no BRF financials, no bid advisor when comps are missing)."""
    price = _num_or_none(data.get("price"))
    sqm = _num_or_none(data.get("sqm"))
    fair = _nested_num(data, "bid_advisor", "fair_value")
    score = _nested_num(data, "composite_score", "composite")
    url = data.get("url")
    return {
        "slug": slug,
        # Only the top-N deep-dived candidates get a markdown report; data.json
        # is written for every scanned listing. Gate the detail hyperlink on
        # the same resolver candidate_detail() uses so we never link to a 404.
        "has_report": _resolve_md(run_id, slug) is not None,
        "address": data.get("address") or slug,
        "url": url if isinstance(url, str) and url.startswith("http") else None,
        "price": price,
        "sqm": sqm,
        "kr_m2": round(price / sqm) if price and sqm else None,
        "fee": _num_or_none(data.get("fee")),
        "built": _num_or_none(data.get("construction_year")),
        "score": int(score) if score is not None else None,
        "cagr": _nested_num(data, "weighted_cagr", "composite"),
        "prem_s": _nested_num(data, "premium_structured", "tier"),
        "prem_l": _nested_num(data, "premium_llm", "tier"),
        "est_value": fair,   # our calculated estimated value (bid_advisor.fair_value)
        # bid_advisor prices living area only — flag listings whose terrace /
        # uterum / bastu makes that a lower bound, not a fair value.
        "est_understated": bool(data.get("bid_advisor_understated")),
        "est_understated_terms": data.get("understatement_terms") or [],
        "booli_est": _num_or_none(data.get("booli_estimate")),  # Booli värdering
        # "estimate" (Värdekollen, for-sale ads) or "sold" (recent slutpris, when
        # the unit isn't for-sale on Booli so no model estimate exists).
        "booli_kind": data.get("booli_estimate_kind") or "estimate",
        "fair_delta": (price / fair - 1.0) if price and fair else None,
    }


def _load_candidates(run_id: str) -> list[dict]:
    """Build score-sorted (desc) table rows for a run. Tolerates a missing
    manifest, missing/garbage per-candidate data.json, and bad slugs."""
    manifest = _runs_dir() / run_id / "_manifest.json"
    # Broad guard: this is the landing page now, so a single unreadable file
    # (missing, a directory, perms, bad bytes, bad JSON) must degrade to an
    # empty/partial table, never 500 the hub. json.JSONDecodeError and
    # UnicodeDecodeError are both ValueError subclasses; the OS-level ones are
    # all OSError subclasses.
    try:
        slugs = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(slugs, list):
        return []
    rows: list[dict] = []
    for slug in slugs:
        if not isinstance(slug, str) or not _SLUG_RE.match(slug):
            continue
        data: dict = {}
        path = _runs_dir() / run_id / "_raw" / slug / "data.json"
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
        rows.append(_candidate_row(run_id, slug, data))
    # None scores sort last; within scored rows, highest first.
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return rows


_DASH = "—"


def _opt(value) -> str:
    if value is None:
        return _DASH
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fmt_price(price) -> str:
    return f"{price / 1e6:.2f}M" if isinstance(price, (int, float)) else _DASH


def _fmt_sint(num) -> str:
    """Space-grouped thousands (Swedish convention), non-breaking so cells
    don't wrap mid-number."""
    if not isinstance(num, (int, float)):
        return _DASH
    return f"{int(round(num)):,}".replace(",", " ")


def _fmt_cagr(cagr) -> str:
    return f"{cagr:.1f}" if isinstance(cagr, (int, float)) else _DASH


def _sold_tag(row: dict) -> str:
    """A small '(sold)' marker when the Booli figure is a recent slutpris rather
    than a Värdekollen estimate (the unit isn't for-sale on Booli). Keeps the
    column honest — a transacted price is not a model estimate."""
    if row.get("booli_est") is not None and row.get("booli_kind") == "sold":
        return " <span class=\"meta\">(sold)</span>"
    return ""


def _understated_tag(row: dict) -> str:
    """A ⚠ after our Est. value when bid_advisor (living-area only) likely
    understates it — the listing's terrace / uterum / bastu etc. isn't priced.
    Read the figure as a floor, not a fair value."""
    if not row.get("est_understated"):
        return ""
    terms = ", ".join(row.get("est_understated_terms") or [])
    title = f"Est. value is a floor — bid_advisor ignores biarea/amenities: {terms}"
    return f" <span class=\"meta\" title=\"{escape(title)}\">⚠</span>"


def _render_apartment_table(run_id: str, rows: list[dict]) -> str:
    if not rows:
        return "<p>No candidates in this run.</p>"
    head = (
        "<tr><th>#</th><th>Score</th><th>Address</th><th>Price</th>"
        "<th>Est.<br>value</th><th>Booli<br>est.</th>"
        "<th>kr/m²</th><th>m²</th><th>Fee</th><th>Built</th>"
        "<th>Prem<br>S/L</th><th>CAGR%</th><th>vs<br>fair</th><th></th></tr>"
    )
    body_rows = []
    for i, r in enumerate(rows, 1):
        # run_id is a directory name already matched against _RUN_ID_RE in
        # _list_runs(); slug is matched against _SLUG_RE in _load_candidates;
        # address/url still escaped as defence-in-depth. Link the address only
        # when a per-candidate report exists, else render it as plain text so
        # we don't emit a link to candidate_detail()'s 404.
        # Address links to the canonical Hemnet ad — present for EVERY candidate,
        # so all rows get a link (the report link, top-N only, moves to the last
        # column). Prior behaviour linked the address to the internal report and
        # left non-deep-dived rows as dead plain text.
        addr_text = escape(str(r["address"]))
        if r["url"]:
            addr_cell = (
                f"<a href=\"{escape(r['url'])}\" target=\"_blank\" rel=\"noopener\">"
                f"{addr_text}</a>"
            )
        else:
            addr_cell = addr_text
        report = (
            f"<a href=\"/house/runs/{escape(run_id)}/{escape(r['slug'])}\">report</a>"
            if r["has_report"] else ""
        )
        delta = r["fair_delta"]
        if delta is None:
            fair_cell = _DASH
        else:
            cls = "cheap" if delta < 0 else "rich"
            fair_cell = f"<span class=\"{cls}\">{delta * 100:+.0f}%</span>"
        prem = f"{_opt(r['prem_s'])}/{_opt(r['prem_l'])}"
        # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
        body_rows.append(
            "<tr>"
            f"<td class=\"num\">{i}</td>"
            f"<td class=\"num score\">{_opt(r['score'])}</td>"
            f"<td>{addr_cell}</td>"
            f"<td class=\"num\">{_fmt_price(r['price'])}</td>"
            f"<td class=\"num\">{_fmt_price(r['est_value'])}{_understated_tag(r)}</td>"
            f"<td class=\"num\">{_fmt_price(r['booli_est'])}{_sold_tag(r)}</td>"
            f"<td class=\"num\">{_fmt_sint(r['kr_m2'])}</td>"
            f"<td class=\"num\">{_opt(r['sqm'])}</td>"
            f"<td class=\"num\">{_fmt_sint(r['fee'])}</td>"
            f"<td class=\"num\">{_opt(r['built'])}</td>"
            f"<td class=\"num\">{escape(prem)}</td>"
            f"<td class=\"num\">{_fmt_cagr(r['cagr'])}</td>"
            f"<td class=\"num\">{fair_cell}</td>"
            f"<td>{report}</td>"
            "</tr>"
        )
    return (
        "<div style=\"overflow-x:auto\"><table>"
        f"<thead>{head}</thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _load_sold(run_id: str) -> list[dict]:
    """Archived (sold/withdrawn) candidates for a run, from <run>/_sold.json.
    Written by scripts/archive_sold_listings.py. Tolerant: missing/garbage → []."""
    try:
        data = json.loads((_runs_dir() / run_id / "_sold.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _render_sold_section(sold: list[dict]) -> str:
    """Collapsed <details> 'Sold (N)' block: address (Hemnet-linked) + tag."""
    if not sold:
        return ""
    items = []
    for r in sold:
        addr = escape(str(r.get("address") or r.get("slug") or "?"))
        status = escape(str(r.get("status") or "sold"))
        url = r.get("url")
        link = (
            f"<a href=\"{escape(url)}\" target=\"_blank\" rel=\"noopener\">{addr}</a>"
            if isinstance(url, str) and url.startswith("http") else addr
        )
        # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
        items.append(f"<li>{link} <span class=\"meta\">— {status}</span></li>")
    return (
        f"<details class=\"sold\"><summary>Sold ({len(sold)})</summary>"
        f"<ul>{''.join(items)}</ul></details>"
    )


def _mark_sold_in_summary(html: str, sold: list[dict]) -> str:
    """Tag each sold listing's row in the (frozen) summary table with a
    '(sold)'/'(withdrawn)' marker. Best-effort: the address must still appear in
    the static _summary.md table. Runs after linkify, so it matches the linked
    cell `>ADDR</a>` (and falls back to plain text)."""
    for r in sold:
        addr = escape(str(r.get("address") or ""))
        if not addr:
            continue
        tag = escape(str(r.get("status") or "sold"))
        marker = f" <span class=\"meta\">({tag})</span>"
        linked = f">{addr}</a>"
        plain = f"<td>{addr}</td>"
        if linked in html:
            html = html.replace(linked, f">{addr}</a>{marker}", 1)
        elif plain in html:
            html = html.replace(plain, f"<td>{addr}{marker}</td>", 1)
    return html


# ---------------------------------------------------------------------------
# Routes — HTML
# ---------------------------------------------------------------------------


@bp.route("/")
@require_auth
def index():
    """Hub: ranked apartment table from the most recent run + the innerstad
    heatmap + links to every run. This is the landing aliased as /hh."""
    if request.args.get("token"):
        # User auth-bootstrapped via ?token= — strip it from the URL.
        return redirect("/house", code=302)
    runs = _list_runs()
    if not runs:
        body = (
            "<h1>House</h1>"
            "<p>No findapartments runs yet. From "
            "<code>Q:\\househunting</code>, run "
            "<code>.venv\\Scripts\\python -m scripts.findapartments_scan</code>.</p>"
            "<p><a href=\"/house/heatmap\">Innerstad appreciation heatmap →</a></p>"
        )
        return _shell("House — no runs", body, [("/", "Dashboard"), ("/house", "House")])

    latest = runs[0]["run_id"]
    rows = _load_candidates(latest)
    # run_id comes from a directory name already matched against _RUN_ID_RE in
    # _list_runs(); escape() on top is belt-and-suspenders.
    # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
    body = (
        "<h1>House — apartments we're looking at</h1>"
        f"<p class=\"meta\">Latest run "
        f"<a href=\"/house/runs/{escape(latest)}\">{escape(latest)}</a> · "
        f"{len(rows)} candidate(s) · "
        f"<a href=\"/house/runs/{escape(latest)}\">full summary &amp; caveats →</a></p>"
        "<p><a href=\"/house/k10\">🏠 Your current home — Kellgrensgatan 10 "
        "(BRF Gladan), the apartment being sold →</a></p>"
        "<h2>Candidates</h2>"
        f"{_render_apartment_table(latest, rows)}"
        f"{_render_sold_section(_load_sold(latest))}"
        "<h2>Innerstad appreciation heatmap</h2>"
        "<p><a href=\"/house/heatmap\">Open fullscreen ↗</a></p>"
        "<iframe class=\"heatmap\" src=\"/house/heatmap\" loading=\"lazy\" "
        "title=\"Innerstad appreciation heatmap\"></iframe>"
        f"{_render_runs_links(runs)}"
    )
    return _shell(
        "House", body, [("/", "Dashboard"), ("/house", "House")], wide=True,
    )


def _render_runs_links(runs: list[dict]) -> str:
    items = "".join(
        f"<li><a href=\"/house/runs/{escape(r['run_id'])}\">"
        f"{escape(r['run_id'])}</a> "
        f"<span class=\"meta\">{r['candidate_count']} candidate(s)"
        f"{' · summary' if r['has_summary'] else ''}</span></li>"
        for r in runs
    )
    return (
        f"<h2>All runs ({len(runs)})</h2>"
        f"<ul class=\"runs-list\">{items}</ul>"
    )


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


def _hemnet_urls_by_address(run_id: str) -> dict:
    """Map each candidate's exact `address` → its Hemnet `url` (from data.json),
    for hyperlinking the summary table's Addr column. Addresses in the table are
    written verbatim from data.json, so an exact-string map is reliable."""
    out: dict = {}
    for row in _load_candidates(run_id):
        if row["url"] and row["address"]:
            out[str(row["address"])] = row["url"]
    return out


def _linkify_addr_cells(html: str, addr_urls: dict) -> str:
    """Wrap whole `<td>{address}</td>` cells in a Hemnet link. Only exact,
    whole-cell matches are replaced, so the same address mentioned elsewhere in
    the prose (e.g. the Top-5 one-liners) is left untouched."""
    for address, url in addr_urls.items():
        cell = f"<td>{escape(address)}</td>"
        if cell in html:
            # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
            link = (
                f"<td><a href=\"{escape(url)}\" target=\"_blank\" "
                f"rel=\"noopener\">{escape(address)}</a></td>"
            )
            html = html.replace(cell, link)
    return html


def _format_oneliners(html: str) -> str:
    """Break the run-together "Top-5 one-liners" paragraph (#1 … #2 … #3 …) that
    markdown renders as one blob into a bulleted list, one candidate per line."""
    m = re.search(r"<p>(<strong>#1</strong>.*?)</p>", html, re.S)
    if not m:
        return html
    parts = re.split(r"(?=<strong>#\d)", m.group(1))
    items = "".join(f"<li>{p.strip()}</li>" for p in parts if p.strip())
    # nosemgrep: python.flask.security.injection.raw-html-concat.raw-html-format
    return f"{html[:m.start()]}<ul class=\"oneliners\">{items}</ul>{html[m.end():]}"


@bp.route("/runs/<run_id>")
@require_auth
def run_detail(run_id: str):
    run_id = _validate_run_id(run_id)
    summary = _resolve_md(run_id, "_summary")
    if not summary:
        abort(404)
    text = summary.read_text(encoding="utf-8")
    # The raw slug list is gone: per-candidate reports are reachable from the hub
    # (address links), and the ranked-table addresses now link straight to Hemnet.
    # Keep only a static heatmap link (no interpolation → nothing to escape).
    footer_links = '<p class="meta"><a href="/house/heatmap">heatmap</a></p>'
    # Linkify addresses from active candidates AND archived (sold) ones, so the
    # static summary table keeps working after a listing is archived out of the
    # manifest. Archived listings also get a collapsed "Sold (N)" section.
    sold = _load_sold(run_id)
    addr_urls = _hemnet_urls_by_address(run_id)
    for r in sold:
        url = r.get("url")
        if r.get("address") and isinstance(url, str) and url.startswith("http"):
            addr_urls.setdefault(str(r["address"]), url)
    summary_html = _mark_sold_in_summary(
        _format_oneliners(_linkify_addr_cells(_render_markdown(text), addr_urls)),
        sold,
    )
    body = summary_html + _render_sold_section(sold) + footer_links
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


@bp.route("/k10")
@require_auth
def k10():
    """The current apartment being sold to fund the upgrade — Kellgrensgatan 10,
    BRF Gladan. Renders <house_root>/data/kellgrensgatan/CURRENT_APARTMENT_BRIEF.md."""
    path = _house_root() / "data" / "kellgrensgatan" / "CURRENT_APARTMENT_BRIEF.md"
    if not path.exists():
        abort(404)
    body = _render_markdown(path.read_text(encoding="utf-8"))
    return _shell(
        "Current home — Kellgrensgatan 10", body,
        [("/", "Dashboard"), ("/house", "House"), ("/house/k10", "K10")],
    )


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
