"""Dynamic warrant catalog refresher for the crypto swing trader.

Mirrors `data/metals_warrant_refresh.py` for BTC/ETH instruments. Searches
Avanza for crypto trackers and (where they exist) BULL/BEAR or MINI L/S
certificates linked to BTC/ETH, probes each for current quote/leverage/
barrier, and caches the result to `data/crypto_warrant_catalog.json` with a
TTL (default 6h).

Why this exists separately from metals_warrant_refresh:
- Avanza issues warrants under different naming conventions per underlying.
  Silver/Gold use "MINI L SILVER AVA n" / "MINI L GULD AVA n"; BTC uses
  "XBT TRACKER" / "BULL BITCOIN" / "BEAR BITCOIN" prefixes. ETH uses
  "ETH TRACKER" / "BULL ETHEREUM" / "BEAR ETHEREUM".
- We don't want to bloat the metals refresher with unrelated queries that
  would slow down its 6h cycle.

Usage:
    from data.crypto_warrant_refresh import load_catalog_or_fetch
    catalog = load_catalog_or_fetch(page)  # page = Playwright page from crypto_loop

    # Falls back to crypto_swing_config.WARRANT_CATALOG_FALLBACK if the
    # refresh fails AND no cache exists.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("crypto_warrant_refresh")

CATALOG_FILE = "data/crypto_warrant_catalog.json"
TTL_HOURS = 6
API_BASE = "https://www.avanza.se"

# Search queries -> (underlying, direction) tuples. Same approach as metals:
# Avanza filtered-search caps at 10 hits per query, so we use multiple
# prefixes to cover the full universe.
_BASE_PREFIXES = [
    ("BTC-USD", "LONG",  "XBT TRACKER"),
    ("BTC-USD", "LONG",  "BITCOIN TRACKER"),
    ("BTC-USD", "LONG",  "BULL BITCOIN"),
    ("BTC-USD", "LONG",  "BULL BTC"),
    ("BTC-USD", "SHORT", "BEAR BITCOIN"),
    ("BTC-USD", "SHORT", "BEAR BTC"),
    ("BTC-USD", "LONG",  "MINI L BITCOIN"),
    ("BTC-USD", "SHORT", "MINI S BITCOIN"),
    ("ETH-USD", "LONG",  "ETH TRACKER"),
    ("ETH-USD", "LONG",  "ETHEREUM TRACKER"),
    ("ETH-USD", "LONG",  "BULL ETHEREUM"),
    ("ETH-USD", "LONG",  "BULL ETH"),
    ("ETH-USD", "SHORT", "BEAR ETHEREUM"),
    ("ETH-USD", "SHORT", "BEAR ETH"),
    ("ETH-USD", "LONG",  "MINI L ETHEREUM"),
    ("ETH-USD", "SHORT", "MINI S ETHEREUM"),
]

SEARCH_QUERIES: list[tuple[str, str, str]] = list(_BASE_PREFIXES)

# Reject candidates below this barrier buffer. Crypto wicks are wider than
# metals, so we use 15% (vs 10% for metals).
MIN_BARRIER_DISTANCE_PCT = 15.0
# Crypto warrants tend to have wider spreads than metals MINIs.
MAX_SPREAD_PCT = 2.0


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _csrf_from_page(page) -> str | None:
    try:
        for c in page.context.cookies():
            if c.get("name") == "AZACSRF":
                return c.get("value")
    except Exception as exc:  # noqa: BLE001
        logger.debug("_csrf_from_page failed: %s", exc)
    return None


def _page_api_post(page, path: str, payload: dict) -> Any | None:
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    headers = {"Content-Type": "application/json"}
    csrf = _csrf_from_page(page)
    if csrf:
        headers["X-SecurityToken"] = csrf
    resp = page.context.request.post(url, data=json.dumps(payload), headers=headers)
    if not resp.ok:
        logger.debug("_page_api_post %s -> HTTP %s", path, resp.status)
        return None
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("_page_api_post %s decode: %s", path, exc)
        return None


def _page_api_get(page, path: str) -> Any | None:
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    resp = page.context.request.get(url)
    if not resp.ok:
        logger.debug("_page_api_get %s -> HTTP %s", path, resp.status)
        return None
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("_page_api_get %s decode: %s", path, exc)
        return None


def _search_warrants(query: str, page, limit: int = 40) -> list[dict]:
    """Call Avanza filtered-search and return WARRANT/CERTIFICATE hits."""
    try:
        result = _page_api_post(
            page,
            "/_api/search/filtered-search",
            {"query": query, "limit": limit},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("search %r failed: %s", query, exc)
        return []
    if not result:
        return []
    return [
        h for h in result.get("hits", [])
        if h.get("type") in ("WARRANT", "CERTIFICATE", "ETF")
    ]


def _probe_warrant(ob_id: str, page, hit_type: str) -> dict | None:
    """Fetch full instrument detail.

    For WARRANT use market-guide/warrant; for CERTIFICATE use
    market-guide/certificate; for ETF (e.g. XBT-ETF trackers like XBTH4) use
    market-guide/exchange-traded-fund. Return None on any error so callers
    can drop the candidate.
    """
    if hit_type == "WARRANT":
        path = f"/_api/market-guide/warrant/{ob_id}"
    elif hit_type == "CERTIFICATE":
        path = f"/_api/market-guide/certificate/{ob_id}"
    elif hit_type == "ETF":
        path = f"/_api/market-guide/exchange-traded-fund/{ob_id}"
    else:
        return None

    try:
        data = _page_api_get(page, path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("probe %s (%s) failed: %s", ob_id, hit_type, exc)
        return None
    if not data:
        return None

    quote = data.get("quote") or {}
    ki = data.get("keyIndicators") or {}
    underlying = (data.get("underlying") or {}).get("quote") or {}

    bid = quote.get("buy")
    ask = quote.get("sell")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return None

    return {
        "ob_id": str(ob_id),
        "name": data.get("name"),
        "isin": data.get("isin"),
        "tradable": data.get("tradable"),
        "bid": float(bid),
        "ask": float(ask),
        "last": quote.get("last"),
        "spread_sek": quote.get("spread"),
        "leverage": ki.get("leverage"),
        "barrier": ki.get("barrierLevel"),
        "parity": ki.get("parity"),
        "direction_raw": ki.get("direction"),  # "Lång" or "Kort"
        "isAza": bool(ki.get("isAza")),
        "sub_type": ki.get("subType"),
        "underlying_last": underlying.get("last"),
        "api_type": hit_type.lower(),  # "warrant" | "certificate" | "etf"
    }


def _barrier_distance_pct(probe: dict, direction: str) -> float:
    barrier = probe.get("barrier")
    und = probe.get("underlying_last")
    if not barrier or not und or und <= 0:
        return 0.0
    if direction == "LONG":
        return (float(und) - float(barrier)) / float(und) * 100.0
    return (float(barrier) - float(und)) / float(und) * 100.0


def _spread_pct(probe: dict) -> float:
    bid = probe.get("bid") or 0
    ask = probe.get("ask") or 0
    if bid <= 0:
        return 999.0
    return (ask - bid) / bid * 100.0


def _is_valid_candidate(probe: dict, direction: str) -> tuple[bool, str]:
    """Filter gate. Trackers (1x, no barrier) pass without barrier check."""
    if probe.get("tradable") != "BUYABLE_AND_SELLABLE":
        return False, f"not tradable ({probe.get('tradable')})"

    is_tracker = (probe.get("api_type") == "etf"
                  or probe.get("leverage") in (None, 0, 1, 1.0)
                  or probe.get("barrier") in (None, 0))

    if is_tracker:
        sp = _spread_pct(probe)
        if sp > MAX_SPREAD_PCT:
            return False, f"spread {sp:.2f}% > {MAX_SPREAD_PCT}%"
        return True, "ok (tracker)"

    if probe.get("leverage") in (None, 0):
        return False, "missing leverage"
    bd = _barrier_distance_pct(probe, direction)
    if bd < MIN_BARRIER_DISTANCE_PCT:
        return False, f"barrier dist {bd:.1f}% < {MIN_BARRIER_DISTANCE_PCT}%"
    sp = _spread_pct(probe)
    if sp > MAX_SPREAD_PCT:
        return False, f"spread {sp:.2f}% > {MAX_SPREAD_PCT}%"
    return True, "ok"


def _make_key(name: str) -> str:
    """'XBT TRACKER AVA' -> 'XBT_TRACKER_AVA'."""
    return (name or "").replace(" ", "_").upper()


def refresh_warrant_catalog(page) -> tuple[dict[str, dict], set[tuple[str, str]]]:
    """Fetch live crypto warrant universe.

    Returns (catalog, covered_pairs) where covered_pairs is the set of
    (underlying, direction) tuples that produced at least one valid
    candidate. The caller can merge with the previous cache for any
    uncovered pair instead of overwriting it.
    """
    catalog: dict[str, dict] = {}
    covered: set[tuple[str, str]] = set()
    for underlying, direction, query in SEARCH_QUERIES:
        hits = _search_warrants(query, page)
        if not hits:
            logger.info("no hits for %r", query)
            continue

        for hit in hits:
            ob_id = str(hit.get("orderBookId") or "")
            hit_type = hit.get("type", "WARRANT")
            if not ob_id:
                continue
            probe = _probe_warrant(ob_id, page, hit_type)
            if not probe:
                continue
            ok, reason = _is_valid_candidate(probe, direction)
            if not ok:
                logger.debug("%s rejected: %s", probe.get("name"), reason)
                continue

            key = _make_key(probe["name"])
            catalog[key] = {
                "ob_id": ob_id,
                "api_type": probe["api_type"],
                "underlying": underlying,
                "direction": direction,
                "leverage": float(probe.get("leverage") or 1.0),
                "barrier": float(probe["barrier"]) if probe.get("barrier") else None,
                "parity": int(probe.get("parity") or 1),
                "name": probe["name"],
                "isAza": bool(probe.get("isAza")),
                "spread_pct": round(_spread_pct(probe), 2),
                "barrier_dist_pct": round(_barrier_distance_pct(probe, direction), 2),
                "last_probe": _now_utc().isoformat(),
            }
            covered.add((underlying, direction))

    return catalog, covered


def load_catalog_or_fetch(page=None, force_refresh: bool = False) -> dict[str, dict]:
    """Return the catalog, refreshing from Avanza when stale.

    Public entry-point used by the crypto swing trader. Behavior:
        1. If on-disk catalog is fresh (within TTL_HOURS), return it.
        2. Else if `page` provided, refresh from Avanza, write atomically,
           merge any previously-covered pairs missing from this refresh.
        3. Else (no page, no fresh cache), return the static fallback from
           crypto_swing_config.WARRANT_CATALOG_FALLBACK so the loop can
           still bootstrap.
    """
    cached = load_json(CATALOG_FILE) or {}
    refreshed_ts = cached.get("refreshed_ts")
    is_fresh = False
    if refreshed_ts:
        try:
            ts = datetime.datetime.fromisoformat(refreshed_ts)
            age_hours = (_now_utc() - ts).total_seconds() / 3600.0
            is_fresh = age_hours < TTL_HOURS
        except (ValueError, TypeError):
            is_fresh = False

    if is_fresh and not force_refresh:
        return cached.get("warrants") or {}

    if page is None:
        # No way to refresh — return cache (stale ok) or fallback
        if cached.get("warrants"):
            return cached["warrants"]
        try:
            from data.crypto_swing_config import WARRANT_CATALOG_FALLBACK
            return WARRANT_CATALOG_FALLBACK
        except Exception:
            return {}

    new_catalog, covered = refresh_warrant_catalog(page)

    # Merge: keep previously-covered pairs that didn't appear this round
    # (network failure on a single query shouldn't drop those warrants).
    if cached.get("warrants"):
        for key, w in cached["warrants"].items():
            pair = (w.get("underlying"), w.get("direction"))
            if pair not in covered and key not in new_catalog:
                new_catalog[key] = w  # preserve

    payload = {
        "refreshed_ts": _now_utc().isoformat(),
        "ttl_hours": TTL_HOURS,
        "warrants": new_catalog,
    }
    try:
        atomic_write_json(CATALOG_FILE, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("atomic_write %s failed: %s", CATALOG_FILE, exc)

    return new_catalog


__all__ = [
    "CATALOG_FILE",
    "TTL_HOURS",
    "SEARCH_QUERIES",
    "MIN_BARRIER_DISTANCE_PCT",
    "MAX_SPREAD_PCT",
    "refresh_warrant_catalog",
    "load_catalog_or_fetch",
]
