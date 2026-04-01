"""Auto-discover best fishing instruments via Avanza search API.

Searches for BULL/BEAR/MINI S/MINI L silver and gold instruments,
fetches quotes for each, and ranks by spread + barrier distance.

Usage:
    from portfolio.fish_instrument_finder import find_best_instruments
    instruments = find_best_instruments("XAG-USD", "SHORT")
    # Returns ranked list: [{name, ob_id, price, spread_pct, leverage, barrier, ...}]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio.fish_instrument_finder")

# Search queries per underlying + direction
_SEARCH_QUERIES = {
    ("XAG-USD", "SHORT"): [
        "MINI S SILVER AVA",
        "BEAR SILVER AVA",
    ],
    ("XAG-USD", "LONG"): [
        "MINI L SILVER AVA",
        "BULL SILVER AVA",
    ],
    ("XAU-USD", "SHORT"): [
        "MINI S GULD AVA",
        "BEAR GULD AVA",
    ],
    ("XAU-USD", "LONG"): [
        "MINI L GULD AVA",
        "BULL GULD AVA",
    ],
}


def _search_avanza(query: str, limit: int = 20) -> list[dict]:
    """Search Avanza via the session API."""
    try:
        from portfolio.avanza_session import api_post
        result = api_post("/_api/search/filtered-search", {"query": query, "limit": limit})
        if result and isinstance(result, dict):
            return result.get("hits", [])
    except Exception as e:
        logger.warning("Avanza search failed: %s", e)
    return []


def _get_quote(ob_id: str) -> dict | None:
    """Get quote for an instrument."""
    try:
        from portfolio.avanza_session import get_quote
        return get_quote(ob_id)
    except Exception:
        return None


def _get_warrant_details(ob_id: str) -> dict | None:
    """Get warrant details (barrier, leverage, parity) from Avanza API."""
    try:
        from portfolio.avanza_session import api_get
        data = api_get(f"/_api/market-guide/warrant/{ob_id}")
        if data and isinstance(data, dict):
            ki = data.get("keyIndicators", {})
            return {
                "leverage": ki.get("leverage"),
                "barrier": ki.get("barrierLevel"),
                "parity": ki.get("parity"),
                "direction": data.get("direction"),
                "issuer": data.get("issuerName"),
                "isin": data.get("isin"),
            }
    except Exception:
        pass
    return None


def find_best_instruments(
    ticker: str,
    direction: str,
    underlying_price: float = 0,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Find and rank the best fishing instruments for a ticker + direction.

    Args:
        ticker: e.g. "XAG-USD"
        direction: "LONG" or "SHORT"
        underlying_price: current underlying price (for barrier distance calc)
        max_results: max instruments to return

    Returns:
        List of dicts sorted by quality (lowest spread, safest barrier first):
        [{name, ob_id, bid, ask, spread_pct, leverage, barrier,
          barrier_distance_pct, instrument_type, issuer}]
    """
    queries = _SEARCH_QUERIES.get((ticker, direction), [])
    if not queries:
        return []

    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for query in queries:
        hits = _search_avanza(query, limit=20)
        for hit in hits:
            ob_id = str(hit.get("orderBookId", ""))
            if not ob_id or ob_id in seen_ids:
                continue
            seen_ids.add(ob_id)

            name = hit.get("title", "")
            itype = hit.get("type", "").lower()

            # Filter: only warrants and certificates
            if itype not in ("warrant", "certificate"):
                continue

            # Get quote for spread
            quote = _get_quote(ob_id)
            if not quote:
                continue

            bid = float(quote.get("buy", 0) or 0)
            ask = float(quote.get("sell", 0) or 0)
            if bid <= 0 or ask <= 0:
                continue

            spread_pct = (ask - bid) / bid * 100

            # Get warrant details for barrier/leverage
            details = _get_warrant_details(ob_id)
            leverage = None
            barrier = None
            barrier_dist = None

            if details:
                leverage = details.get("leverage")
                barrier = details.get("barrier")
                if barrier and underlying_price > 0:
                    barrier_dist = abs(underlying_price - barrier) / underlying_price * 100

            # Volume / activity check — detect dead instruments
            volume_traded = float(quote.get("totalVolumeTraded", 0) or 0)
            value_traded = float(quote.get("totalValueTraded", 0) or 0)
            updated = quote.get("updated", 0) or quote.get("timeOfLast", 0)
            is_active = volume_traded > 0 or value_traded > 0

            candidates.append({
                "name": name,
                "ob_id": ob_id,
                "bid": bid,
                "ask": ask,
                "last": float(quote.get("last", 0) or 0),
                "spread_pct": round(spread_pct, 2),
                "leverage": leverage,
                "barrier": barrier,
                "barrier_distance_pct": round(barrier_dist, 1) if barrier_dist else None,
                "instrument_type": itype,
                "issuer": details.get("issuer", "?") if details else "?",
                "change_pct": quote.get("changePercent", 0),
                "volume_today": int(volume_traded),
                "value_traded_sek": round(value_traded, 0),
                "is_active": is_active,
            })

    # Rank: prefer low spread, then high barrier distance (safer)
    def sort_key(c: dict) -> tuple:
        spread = c["spread_pct"]
        # Barrier distance: higher is safer, use negative for descending sort
        barrier = -(c["barrier_distance_pct"] or 0)
        return (spread, barrier)

    candidates.sort(key=sort_key)
    return candidates[:max_results]


def print_instruments(ticker: str, direction: str, underlying_price: float = 0) -> None:
    """Print a formatted table of available instruments."""
    instruments = find_best_instruments(ticker, direction, underlying_price)
    if not instruments:
        print(f"No instruments found for {ticker} {direction}")
        return

    print(f"\n{'='*80}")
    print(f"  {ticker} {direction} instruments (ranked by spread + barrier safety)")
    print(f"{'='*80}")
    print(f"{'OB_ID':>10s}  {'Spread':>7s}  {'Lev':>5s}  {'Barrier':>8s}  {'Dist':>6s}  {'Vol':>6s}  {'Bid':>7s}  {'Ask':>7s}  Name")
    print("-" * 90)
    for inst in instruments:
        lev = f"{inst['leverage']:.0f}x" if inst['leverage'] else "?"
        barrier = f"${inst['barrier']:.1f}" if inst['barrier'] else "N/A"
        dist = f"{inst['barrier_distance_pct']:.1f}%" if inst['barrier_distance_pct'] else "N/A"
        vol = str(inst.get("volume_today", 0))
        dead = " DEAD" if not inst.get("is_active", True) else ""
        print(
            f"{inst['ob_id']:>10s}  {inst['spread_pct']:>6.1f}%  {lev:>5s}  "
            f"{barrier:>8s}  {dist:>6s}  {vol:>6s}  {inst['bid']:>7.2f}  {inst['ask']:>7.2f}  {inst['name']}{dead}"
        )


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "XAG-USD"
    direction = sys.argv[2] if len(sys.argv) > 2 else "SHORT"
    price = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
    print_instruments(ticker, direction, price)
