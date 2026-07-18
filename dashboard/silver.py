"""dashboard/silver.py — read-only XAG-USD accuracy matrix for the #silver
command page (Phase 6, 2026-07-18).

Blueprint at /api/silver. Fills a gap Phase 6 needs that no existing route
covers: per-signal accuracy broken down by horizon for ONE ticker.
``/api/accuracy`` (dashboard/app.py) only exposes GLOBAL per-signal accuracy
plus per-ticker CONSENSUS accuracy — never per-ticker PER-SIGNAL. That data
already exists (``accuracy_stats.accuracy_by_ticker_signal[_cached]()`` reads
straight from ``signal_log.db``), it's just never been surfaced for horizons
other than 1d — ``data/ticker_signal_accuracy_cache.json`` only had a "1d"
key before this route started asking for "3h"/"3d"/"5d" too.

Single route: GET /api/silver/accuracy?ticker=XAG-USD (defaults to
XAG-USD — this blueprint exists for the silver page, not as a general
ticker-accuracy API). Every horizon in HORIZONS goes through the same 1h-TTL
cache accuracy_stats already maintains, so a first request after a cache
miss costs one signal_log.db scan per horizon (~35ms each, benchmarked
2026-07-18) and every request after that is a cache hit.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from dashboard.auth import require_auth
from portfolio import accuracy_stats
from portfolio.accuracy_stats import accuracy_by_ticker_signal_cached
from portfolio.file_utils import load_json
from portfolio.tickers import ALL_TICKERS

bp = Blueprint("silver", __name__, url_prefix="/api/silver")

HORIZONS = ("3h", "1d", "3d", "5d")
DEFAULT_TICKER = "XAG-USD"


@bp.route("/accuracy", methods=["GET"])
@require_auth
def api_silver_accuracy():
    """Per-signal accuracy for one ticker across HORIZONS.

    Response: ``{"ticker": ..., "updated_ts": <shared cache mtime, or
    None>, "horizons": {h: {"signals": {name: {...accuracy_by_ticker_signal
    shape...}}, "n_signals": int}}}``. A horizon with no accuracy data for
    this ticker yet (e.g. too few outcome rows) comes back with an empty
    ``signals`` dict, not a missing key — same "show it, don't drop it"
    rule as ``/api/accuracy``'s 10d handling.
    """
    ticker = request.args.get("ticker", DEFAULT_TICKER)
    if ticker not in ALL_TICKERS:
        return jsonify({"error": f"unknown ticker: {ticker!r}"}), 400

    horizons = {}
    for h in HORIZONS:
        signals = accuracy_by_ticker_signal_cached(h).get(ticker, {})
        horizons[h] = {"signals": signals, "n_signals": len(signals)}

    # The cache file stores one shared "time" across every horizon (see
    # accuracy_stats.write_ticker_accuracy_cache) — not per-horizon, so this
    # is a single age for the whole payload, not a per-horizon one. Read the
    # path off the module (not a copied-in constant) so test fixtures that
    # redirect accuracy_stats.TICKER_ACCURACY_CACHE_FILE to a tmp dir apply
    # here too.
    cache = load_json(accuracy_stats.TICKER_ACCURACY_CACHE_FILE, default={}) or {}
    updated_ts = cache.get("time") if isinstance(cache, dict) else None

    return jsonify({"ticker": ticker, "updated_ts": updated_ts, "horizons": horizons})
