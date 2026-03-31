"""Instrument scanner — find and rank the best warrants/certificates.

Chains search → detail fetch → ranking to answer questions like:
"Find the best bull mini-future for oil right now"

Works with EITHER auth method:
- TOTP (AvanzaClient) — preferred, faster
- BankID session (avanza_session.api_get/api_post) — fallback

Usage:
    from portfolio.avanza.scanner import scan_instruments

    results = scan_instruments(
        query="OLJA",           # underlying asset keyword
        direction="BULL",       # BULL or BEAR
        instrument_type="certificate",  # certificate, warrant, or None for both
        sort_by="spread",       # spread, leverage, price, barrier_distance
        limit=10,
    )
    for r in results:
        print(f"{r['name']:40s} lev={r['leverage']:5.1f}x  spread={r['spread_pct']:.2f}%  bid={r['bid']}")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from portfolio.avanza.types import _val

logger = logging.getLogger("portfolio.avanza.scanner")


# ---------------------------------------------------------------------------
# Dual-auth API helpers — try TOTP first, fall back to BankID session
# ---------------------------------------------------------------------------

def _get_api():
    """Return (search_fn, instrument_fn, marketdata_fn) that work with
    whichever auth is currently available.

    Returns:
        Tuple of three callables:
        - search(instrument_type_str, query, limit) -> dict or list
        - get_instrument(api_type, ob_id) -> dict
        - get_market_data(ob_id) -> dict
    """
    # Try TOTP client first
    try:
        from portfolio.avanza.client import AvanzaClient
        client = AvanzaClient.get_instance()
        avanza = client.avanza

        def _search(itype_str, query, limit):
            from avanza.constants import InstrumentType
            return avanza.search_for_instrument(InstrumentType(itype_str), query, limit)

        def _instrument(api_type, ob_id):
            return avanza.get_instrument(api_type, ob_id)

        def _marketdata(ob_id):
            return avanza.get_market_data(ob_id)

        logger.debug("Scanner using TOTP client")
        return _search, _instrument, _marketdata
    except Exception:
        pass

    # Fall back to BankID session (Playwright)
    try:
        from portfolio.avanza_session import api_get, api_post

        def _search(itype_str, query, limit):
            return api_post("/_api/search/filtered-search", {"query": query, "limit": limit})

        def _instrument(api_type, ob_id):
            return api_get(f"/_api/market-guide/{api_type}/{ob_id}")

        def _marketdata(ob_id):
            return api_get(f"/_api/trading-critical/rest/marketdata/{ob_id}")

        logger.debug("Scanner using BankID session")
        return _search, _instrument, _marketdata
    except Exception as e:
        raise RuntimeError(
            "No Avanza auth available. Either configure TOTP credentials "
            "or run scripts/avanza_login.py for BankID session."
        ) from e


@dataclass
class ScannedInstrument:
    """Rich instrument data combining search + market-guide + marketdata."""

    orderbook_id: str
    name: str
    instrument_type: str  # CERTIFICATE, WARRANT, etc.
    direction: str  # BULL, BEAR, LONG, SHORT, or ""

    # Price
    bid: float | None
    ask: float | None
    last: float | None
    spread_pct: float | None  # (ask-bid)/bid * 100

    # Instrument details
    leverage: float | None
    barrier: float | None
    barrier_distance_pct: float | None  # distance from last to barrier

    # Underlying
    underlying_name: str
    underlying_price: float | None

    # Market quality
    volume_today: int
    turnover_today: float
    market_maker: bool

    # Order depth (best level)
    bid_volume: int
    ask_volume: int

    # Computed score (lower = better for spread, higher = better for leverage)
    score: float


def scan_instruments(
    query: str,
    direction: str = "",
    instrument_type: str | None = None,
    sort_by: str = "spread",
    limit: int = 10,
    max_search: int = 30,
    min_leverage: float = 0,
    max_spread_pct: float = 100,
    workers: int = 6,
) -> list[ScannedInstrument]:
    """Search Avanza and fetch details for the best instruments.

    Args:
        query: Search keyword (e.g. "OLJA", "SILVER", "GULD", "TSMC").
        direction: "BULL" or "BEAR" (filters results by name). Empty = both.
        instrument_type: "certificate", "warrant", or None for both.
        sort_by: Ranking criterion — "spread", "leverage", "price", "barrier_distance".
        limit: Max results to return (after filtering and ranking).
        max_search: How many search results to fetch before filtering.
        min_leverage: Minimum leverage to include.
        max_spread_pct: Maximum spread % to include (filters illiquid instruments).
        workers: Thread pool size for parallel detail fetching.

    Returns:
        List of ScannedInstrument, sorted by the chosen criterion.
    """
    search_fn, instrument_fn, marketdata_fn = _get_api()

    # --- Step 1: Search ---
    search_query = f"{direction} {query}".strip() if direction else query
    types_to_search = []
    if instrument_type:
        types_to_search.append(instrument_type)
    else:
        types_to_search.extend(["certificate", "warrant"])

    all_hits: list[dict] = []
    for itype in types_to_search:
        try:
            raw = search_fn(itype, search_query, max_search)
            hits = raw.get("hits", raw) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
            all_hits.extend(hits)
        except Exception as e:
            logger.warning("Search failed for type=%s query=%r: %s", itype, search_query, e)

    if not all_hits:
        logger.info("No search results for query=%r direction=%s", query, direction)
        return []

    # Filter by direction if specified
    dir_upper = direction.upper()
    if dir_upper:
        all_hits = [h for h in all_hits if dir_upper in (h.get("title", "") or "").upper()]

    # Filter tradeable only
    all_hits = [h for h in all_hits if h.get("tradeable", h.get("tradable", True))]

    # Deduplicate by orderbook ID
    seen = set()
    unique_hits = []
    for h in all_hits:
        ob_id = str(h.get("orderBookId", h.get("id", "")))
        if ob_id and ob_id not in seen:
            seen.add(ob_id)
            unique_hits.append(h)
    all_hits = unique_hits[:max_search]

    logger.info("Scanner: %d candidates after search+filter for %r %s", len(all_hits), query, direction)

    # --- Step 2: Fetch details in parallel ---
    def fetch_detail(hit: dict) -> ScannedInstrument | None:
        ob_id = str(hit.get("orderBookId", hit.get("id", "")))
        name = hit.get("title", hit.get("name", ""))
        itype = hit.get("type", hit.get("instrumentType", ""))

        # Determine API type for market-guide
        api_type = "certificate"
        type_lower = itype.lower() if itype else ""
        if "warrant" in type_lower or "mini" in name.upper():
            api_type = "warrant"
        elif "stock" in type_lower:
            api_type = "stock"

        try:
            # Fetch instrument details (leverage, barrier, underlying)
            info = instrument_fn(api_type, ob_id)
            if not info or not isinstance(info, dict):
                return None

            # Extract quote
            quote = info.get("quote", {})
            bid = _val(quote.get("buy"))
            ask = _val(quote.get("sell"))
            last = _val(quote.get("last"))

            # Compute spread
            spread_pct = None
            if bid and ask and bid > 0:
                spread_pct = round((ask - bid) / bid * 100, 3)

            # Extract leverage and barrier
            ki = info.get("keyIndicators", {})
            leverage = _val(ki.get("leverage"))
            barrier = _val(ki.get("barrierLevel"))

            # Barrier distance
            barrier_dist_pct = None
            if barrier and last and last > 0:
                barrier_dist_pct = round(abs(last - barrier) / last * 100, 2)

            # Underlying
            underlying = info.get("underlying", {})
            underlying_name = underlying.get("name", "")
            underlying_quote = underlying.get("quote", {})
            underlying_price = _val(underlying_quote.get("last"))

            # Volume/turnover
            volume = _val(quote.get("totalVolumeTraded"), 0) or 0
            turnover = _val(quote.get("totalValueTraded"), 0) or 0

            # Detect direction from name
            name_upper = name.upper()
            detected_dir = ""
            for d in ("BULL", "BEAR", "MINI L", "MINI S"):
                if d in name_upper:
                    detected_dir = "BULL" if d in ("BULL", "MINI L") else "BEAR"
                    break

            # Also try market data for order depth (fast call)
            bid_vol = 0
            ask_vol = 0
            mm = False
            try:
                md = marketdata_fn(ob_id)
                if isinstance(md, dict):
                    od = md.get("orderDepth", md.get("orderDepthLevels", {}))
                    levels = od.get("levels", od) if isinstance(od, dict) else od
                    if isinstance(levels, list) and levels:
                        first = levels[0]
                        bid_side = first.get("buySide", first.get("buy", {}))
                        ask_side = first.get("sellSide", first.get("sell", {}))
                        bid_vol = int(bid_side.get("volume", 0))
                        ask_vol = int(ask_side.get("volume", 0))
                    mm = md.get("marketMakerExpected", False)
            except Exception:
                pass

            return ScannedInstrument(
                orderbook_id=ob_id,
                name=name,
                instrument_type=itype,
                direction=detected_dir,
                bid=bid,
                ask=ask,
                last=last,
                spread_pct=spread_pct,
                leverage=leverage,
                barrier=barrier,
                barrier_distance_pct=barrier_dist_pct,
                underlying_name=underlying_name,
                underlying_price=underlying_price,
                volume_today=int(volume),
                turnover_today=float(turnover),
                market_maker=mm,
                bid_volume=bid_vol,
                ask_volume=ask_vol,
                score=0.0,
            )
        except Exception as e:
            logger.debug("Detail fetch failed for %s (%s): %s", ob_id, name, e)
            return None

    results: list[ScannedInstrument] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_detail, h): h for h in all_hits}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    dt = (time.perf_counter() - t0) * 1000
    logger.info("Scanner: fetched %d instrument details in %.0fms", len(results), dt)

    # --- Step 3: Filter ---
    if min_leverage > 0:
        results = [r for r in results if r.leverage and r.leverage >= min_leverage]
    if max_spread_pct < 100:
        results = [r for r in results if r.spread_pct is not None and r.spread_pct <= max_spread_pct]

    # Filter out instruments with no bid/ask (not tradeable right now)
    results = [r for r in results if r.bid is not None and r.ask is not None]

    # --- Step 4: Score and sort ---
    for r in results:
        if sort_by == "spread":
            r.score = r.spread_pct if r.spread_pct is not None else 999
        elif sort_by == "leverage":
            r.score = -(r.leverage or 0)  # negative so higher leverage sorts first
        elif sort_by == "price":
            r.score = r.last or 999
        elif sort_by == "barrier_distance":
            r.score = -(r.barrier_distance_pct or 0)  # negative = larger distance first
        else:
            r.score = r.spread_pct if r.spread_pct is not None else 999

    results.sort(key=lambda r: r.score)
    return results[:limit]


def format_scan_results(results: list[ScannedInstrument]) -> str:
    """Format scan results as a readable table string."""
    if not results:
        return "No instruments found."

    lines = []
    lines.append(f"{'Name':45s} {'ID':>8s} {'Lev':>5s} {'Bid':>8s} {'Ask':>8s} "
                 f"{'Spread':>7s} {'Barrier':>8s} {'Dist%':>6s} {'Vol':>6s} {'MM':>3s}")
    lines.append("-" * 115)

    for r in results:
        lev = f"{r.leverage:.1f}x" if r.leverage else "  -  "
        bid = f"{r.bid:.2f}" if r.bid else "   -   "
        ask = f"{r.ask:.2f}" if r.ask else "   -   "
        spread = f"{r.spread_pct:.2f}%" if r.spread_pct is not None else "  -  "
        barrier = f"{r.barrier:.1f}" if r.barrier else "   -   "
        dist = f"{r.barrier_distance_pct:.1f}%" if r.barrier_distance_pct is not None else "  -  "
        vol = f"{r.volume_today:,}" if r.volume_today else "  0"
        mm = "Yes" if r.market_maker else " No"

        lines.append(f"{r.name[:45]:45s} {r.orderbook_id:>8s} {lev:>5s} {bid:>8s} {ask:>8s} "
                     f"{spread:>7s} {barrier:>8s} {dist:>6s} {vol:>6s} {mm:>3s}")

    return "\n".join(lines)
