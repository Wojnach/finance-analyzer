"""Tick-size rules — price rounding for Avanza order books.

Caches tick tables per orderbook ID so repeated rounding calls do not
hit the API.  Uses integer arithmetic internally to avoid floating-point
drift.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import TickEntry

logger = logging.getLogger("portfolio.avanza.tick_rules")

# Module-level cache: ob_id -> list of TickEntry
_cache: dict[str, list[TickEntry]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tick_rules(ob_id: str) -> list[TickEntry]:
    """Fetch (and cache) the tick-size table for an orderbook.

    Args:
        ob_id: Avanza orderbook ID.

    Returns:
        List of :class:`~portfolio.avanza.types.TickEntry` sorted by
        ``min_price``.
    """
    if ob_id in _cache:
        return _cache[ob_id]

    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.get_order_book_raw(ob_id)

    tick_list_raw: list[dict[str, Any]] = raw.get("tickSizeList", raw.get("tickSizes", []))
    entries = [TickEntry.from_api(t) for t in tick_list_raw]
    entries.sort(key=lambda e: e.min_price)

    _cache[ob_id] = entries
    logger.debug("get_tick_rules ob_id=%s entries=%d (cached)", ob_id, len(entries))
    return entries


def round_to_tick(price: float, ob_id: str, direction: str = "down") -> float:
    """Round a price to the nearest valid tick.

    Uses integer arithmetic (multiply -> floor/ceil -> divide) to avoid
    floating-point drift.

    Args:
        price: The price to round.
        ob_id: Avanza orderbook ID (needed to fetch the tick table).
        direction: ``"down"`` (floor) or ``"up"`` (ceil).

    Returns:
        The rounded price.

    Raises:
        ValueError: If *direction* is not ``"down"`` or ``"up"``.
        ValueError: If no tick rule matches *price*.
    """
    if direction not in ("down", "up"):
        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")

    entries = get_tick_rules(ob_id)
    tick = _find_tick_for_price(price, entries)

    if tick is None:
        raise ValueError(f"No tick rule found for price {price} (ob_id={ob_id})")

    # Integer arithmetic to avoid float drift:
    # steps = price / tick  ->  round to int  ->  result = steps * tick
    # We use a precision multiplier derived from the tick's decimal places.
    precision = _decimal_places(tick)
    multiplier = 10 ** precision

    # Convert to integer domain
    price_int = price * multiplier
    tick_int = round(tick * multiplier)

    if tick_int == 0:
        return price  # degenerate tick; return unchanged

    if direction == "down":
        steps = math.floor(price_int / tick_int)
    else:
        steps = math.ceil(price_int / tick_int)

    result = (steps * tick_int) / multiplier
    return round(result, precision)


def clear_cache() -> None:
    """Clear the module-level tick-rule cache."""
    _cache.clear()
    logger.debug("tick_rules cache cleared")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_tick_for_price(price: float, entries: list[TickEntry]) -> float | None:
    """Find the tick size applicable for *price*.

    Returns ``None`` if no entry matches.
    """
    for entry in entries:
        if entry.min_price <= price <= entry.max_price:
            return entry.tick_size
        # Handle unbounded upper range (max_price == 0 means infinity)
        if entry.min_price <= price and entry.max_price == 0:
            return entry.tick_size
    # Fallback: if price exceeds all ranges, use the last entry
    if entries:
        return entries[-1].tick_size
    return None


def _decimal_places(value: float) -> int:
    """Count the number of significant decimal places in *value*."""
    s = f"{value:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0
