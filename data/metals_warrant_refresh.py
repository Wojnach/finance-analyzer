"""Dynamic warrant catalog refresher for metals swing trader.

Replaces the static hardcoded WARRANT_CATALOG in metals_swing_config.py with
a live fetch from Avanza. Searches for MINI L/S SILVER AVA and MINI L/S GULD AVA
warrants, probes each one for current quote/leverage/barrier, and caches the
result to data/metals_warrant_catalog.json with a TTL.

Why this exists:
- metals_swing_config.WARRANT_CATALOG had MINI L SILVER AVA 301 hardcoded.
- That warrant got knocked out when silver crashed in March; its orderbook
  returns HTTP 404 now.
- MINI L SILVER AVA 332-338 (new issues with 7-11x leverage) exist on Avanza
  but the static catalog never gets refreshed, so the system only ever saw
  MINI L SILVER SG (SG-issued, 1.5x, courtage applies) as the sole candidate.
- This module fixes the blind spot by querying the live universe.

Usage:
    from metals_warrant_refresh import load_catalog_or_fetch
    catalog = load_catalog_or_fetch()  # dict keyed by warrant name
    # Falls back to the static config catalog if the refresh fails.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from portfolio.avanza_session import api_get, api_post
from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("metals_warrant_refresh")

CATALOG_FILE = "data/metals_warrant_catalog.json"
TTL_HOURS = 6

# Search queries -> (underlying, direction) tuples. Avanza's filtered-search
# returns only 10 hits per query (hard API cap), so we use digit-prefixed
# sub-queries to reach the full universe of issued warrants. Without these
# narrower prefixes, newer high-leverage issues (e.g. MINI L SILVER AVA 330s)
# get filtered out by the top-10 cut and the system never discovers them.
_BASE_PREFIXES = [
    ("XAG-USD", "LONG",  "MINI L SILVER AVA"),
    ("XAG-USD", "SHORT", "MINI S SILVER AVA"),
    ("XAU-USD", "LONG",  "MINI L GULD AVA"),
    ("XAU-USD", "SHORT", "MINI S GULD AVA"),
]
# Generate <prefix> + " " + <digits> queries so the search can reach the
# 300s, 400s, 500s series without being dominated by older 70-99 issues that
# always sort first. We query 1-digit (0-9) AND 2-digit (30-49) prefixes to
# catch newer high-leverage issues that otherwise hide past the top-10 cut.
SEARCH_QUERIES: list[tuple[str, str, str]] = []
for _und, _dir, _pref in _BASE_PREFIXES:
    SEARCH_QUERIES.append((_und, _dir, _pref))
    for _digit in range(10):
        SEARCH_QUERIES.append((_und, _dir, f"{_pref} {_digit}"))
    # 2-digit prefixes for newer issue ranges (e.g. 330s, 400s, 500s)
    for _two in (30, 31, 32, 33, 34, 35, 40, 41, 42, 49, 50):
        SEARCH_QUERIES.append((_und, _dir, f"{_pref} {_two}"))

# Reject candidates below this barrier buffer. MINI leverage ~= 1/barrier_dist,
# so 10% allows up to ~10x products and blocks anything about to knock out.
MIN_BARRIER_DISTANCE_PCT = 10.0
# Reject candidates with spreads wider than this (too expensive to trade).
MAX_SPREAD_PCT = 1.5


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _search_warrants(query: str, limit: int = 40) -> list[dict]:
    """Call Avanza filtered-search and return hit list (may be capped at 10)."""
    try:
        result = api_post("/_api/search/filtered-search", {"query": query, "limit": limit})
    except Exception as exc:  # noqa: BLE001
        logger.warning("search %r failed: %s", query, exc)
        return []
    if not result:
        return []
    return [h for h in result.get("hits", []) if h.get("type") == "WARRANT"]


def _probe_warrant(ob_id: str) -> dict | None:
    """Fetch full warrant detail from market-guide API.

    Returns None on 404 (dead/delisted) or any other error, so callers can
    drop the candidate without special-casing.
    """
    try:
        data = api_get(f"/_api/market-guide/warrant/{ob_id}")
    except Exception as exc:  # noqa: BLE001
        logger.debug("probe %s failed: %s", ob_id, exc)
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
    }


def _barrier_distance_pct(probe: dict, direction: str) -> float:
    """Compute barrier distance as a positive percentage.

    LONG: underlying is above barrier (returns +%)
    SHORT: underlying is below barrier (returns +%)
    """
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
    """Filter gate: tradable, barrier buffer, spread. Returns (ok, reason)."""
    if probe.get("tradable") != "BUYABLE_AND_SELLABLE":
        return False, f"not tradable ({probe.get('tradable')})"
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
    """Convert 'MINI L SILVER AVA 332' -> 'MINI_L_SILVER_AVA_332'."""
    return (name or "").replace(" ", "_").upper()


def refresh_warrant_catalog() -> tuple[dict[str, dict], set[tuple[str, str]]]:
    """Fetch live warrant universe for all configured queries.

    Returns:
        (catalog, covered_pairs) where catalog is keyed by canonical warrant
        name and covered_pairs is the set of (underlying, direction) tuples
        that successfully produced at least one valid candidate. The caller
        uses covered_pairs to detect partial failures and merge with the
        previous cache for any uncovered pair, instead of overwriting it.
    """
    catalog: dict[str, dict] = {}
    covered: set[tuple[str, str]] = set()
    for underlying, direction, query in SEARCH_QUERIES:
        hits = _search_warrants(query)
        if not hits:
            logger.warning("no hits for %r", query)
            continue

        for hit in hits:
            ob_id = str(hit.get("orderBookId") or "")
            if not ob_id:
                continue
            probe = _probe_warrant(ob_id)
            if not probe:
                continue
            ok, reason = _is_valid_candidate(probe, direction)
            if not ok:
                logger.debug("%s rejected: %s", probe.get("name"), reason)
                continue

            key = _make_key(probe["name"])
            catalog[key] = {
                "ob_id": ob_id,
                "api_type": "warrant",
                "underlying": underlying,
                "direction": direction,
                "leverage": float(probe["leverage"]),
                "barrier": float(probe["barrier"]) if probe.get("barrier") else None,
                "parity": probe.get("parity") or 10,
                "name": probe["name"],
                "isAza": probe["isAza"],
                "spread_pct": round(_spread_pct(probe), 3),
                "barrier_dist_pct": round(_barrier_distance_pct(probe, direction), 2),
                "last_probe": _now_utc().isoformat(),
            }
            covered.add((underlying, direction))
    return catalog, covered


# Set of all (underlying, direction) pairs the refresher tries to cover.
# Used by load_catalog_or_fetch to detect partial-failure refreshes.
ALL_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (underlying, direction) for (underlying, direction, _query) in SEARCH_QUERIES
)


def _write_cache(catalog: dict[str, dict]) -> None:
    payload = {
        "refreshed_ts": _now_utc().isoformat(),
        "ttl_hours": TTL_HOURS,
        "warrants": catalog,
    }
    try:
        atomic_write_json(CATALOG_FILE, payload, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to write catalog cache: %s", exc)


def _read_cache() -> dict[str, Any] | None:
    return load_json(CATALOG_FILE)


def _is_stale(payload: dict[str, Any]) -> bool:
    ts_str = payload.get("refreshed_ts")
    if not ts_str:
        return True
    try:
        ts = datetime.datetime.fromisoformat(ts_str)
    except ValueError:
        return True
    age = _now_utc() - ts
    return age > datetime.timedelta(hours=payload.get("ttl_hours", TTL_HOURS))


def _merge_with_cache(
    fresh: dict[str, dict],
    covered: set[tuple[str, str]],
    cached: dict[str, Any] | None,
) -> dict[str, dict]:
    """Merge a (possibly partial) fresh catalog with the previous cache.

    For each (underlying, direction) pair the fresh refresh DID cover, use
    the fresh entries (they replace whatever was cached). For pairs the
    fresh refresh did NOT cover (search failed or returned all-rejected),
    fall back to the previous cache so we don't lose coverage of e.g. gold
    shorts during a transient API hiccup.
    """
    if not cached:
        return fresh
    cached_warrants = cached.get("warrants") or {}
    if not cached_warrants:
        return fresh
    uncovered = ALL_PAIRS - covered
    if not uncovered:
        return fresh
    merged = dict(fresh)
    rescued = 0
    for key, w in cached_warrants.items():
        pair = (w.get("underlying"), w.get("direction"))
        if pair in uncovered and key not in merged:
            merged[key] = w
            rescued += 1
    if rescued:
        logger.warning(
            "partial refresh: rescued %d entries for uncovered pairs %s",
            rescued, sorted(uncovered),
        )
    return merged


def load_catalog_or_fetch(force_refresh: bool = False) -> dict[str, dict]:
    """Return warrant catalog, refreshing from Avanza when the cache is stale.

    On any error (network, search failure, empty result), falls back to the
    last-known-good cache and logs a warning. Partial refreshes (some pairs
    succeed, others fail) merge with the previous cache so dropped pairs
    don't disappear from the trader's universe. If no cache exists at all,
    returns an empty dict — the caller must handle that by falling back to
    metals_swing_config.WARRANT_CATALOG.
    """
    cached = _read_cache()

    if not force_refresh and cached and not _is_stale(cached):
        warrants = cached.get("warrants") or {}
        if warrants:
            logger.info("using cached catalog: %d warrants (age: fresh)", len(warrants))
            return warrants

    logger.info("refreshing warrant catalog from Avanza...")
    try:
        fresh, covered = refresh_warrant_catalog()
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh failed: %s", exc)
        fresh, covered = {}, set()

    if fresh:
        # Merge with previous cache for any (underlying, direction) pair the
        # fresh refresh did not cover — protects against partial-failure cache
        # wipes (e.g. all silver shorts succeed, all gold shorts fail).
        merged = _merge_with_cache(fresh, covered, cached)
        _write_cache(merged)
        if merged is not fresh and len(merged) > len(fresh):
            logger.info("catalog refreshed: %d fresh + %d rescued = %d total",
                        len(fresh), len(merged) - len(fresh), len(merged))
        else:
            logger.info("catalog refreshed: %d warrants (covered %d/%d pairs)",
                        len(merged), len(covered), len(ALL_PAIRS))
        return merged

    # Refresh failed entirely — fall back to last-known-good cache if we have one.
    if cached and cached.get("warrants"):
        logger.warning("refresh failed, using stale cache (%d warrants)", len(cached["warrants"]))
        return cached["warrants"]

    logger.warning("refresh failed and no cache available; caller must fall back")
    return {}


if __name__ == "__main__":
    # Allow manual invocation for testing/refreshing cache.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    catalog = load_catalog_or_fetch(force_refresh=True)
    print(f"\n=== Refreshed catalog: {len(catalog)} warrants ===")
    for key, w in sorted(catalog.items(), key=lambda kv: (kv[1]["underlying"], kv[1]["direction"], -kv[1]["leverage"])):
        print(f"  {key:<30s} {w['underlying']}  {w['direction']:<5s}  "
              f"lev={w['leverage']:>5.2f}x  bardist={w['barrier_dist_pct']:>5.1f}%  "
              f"spread={w['spread_pct']:>5.2f}%  AVA={w['isAza']}")
