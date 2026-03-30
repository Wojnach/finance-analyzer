"""Instrument search — find stocks, certificates, warrants, etc.

Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
for instrument discovery.
"""

from __future__ import annotations

import logging
from typing import Any

from avanza.constants import InstrumentType

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import SearchHit

logger = logging.getLogger("portfolio.avanza.search")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    query: str,
    limit: int = 10,
    instrument_type: str | None = None,
) -> list[SearchHit]:
    """Search for instruments on Avanza.

    Args:
        query: Search string (ISIN, ticker, name fragment, etc.).
        limit: Maximum number of results.
        instrument_type: Optional filter (e.g. ``"certificate"``,
            ``"stock"``, ``"warrant"``).  When *None*, all types are
            searched.

    Returns:
        List of :class:`~portfolio.avanza.types.SearchHit`.
    """
    client = AvanzaClient.get_instance()

    inst_type = (
        InstrumentType(instrument_type) if instrument_type else InstrumentType.ANY
    )

    raw: Any = client.avanza.search_for_instrument(inst_type, query, limit)
    logger.debug(
        "search query=%r type=%s limit=%d hits=%d",
        query,
        inst_type.name,
        limit,
        len(raw) if isinstance(raw, list) else 0,
    )

    hits: list[dict[str, Any]]
    if isinstance(raw, list):
        hits = raw
    elif isinstance(raw, dict):
        hits = raw.get("hits", raw.get("results", []))
    else:
        hits = []

    return [SearchHit.from_api(h) for h in hits]


def find_warrants(query: str = "", limit: int = 20) -> list[SearchHit]:
    """Search specifically for warrants.

    Convenience wrapper around :func:`search` with
    ``instrument_type="warrant"``.
    """
    return search(query=query, limit=limit, instrument_type="warrant")


def find_certificates(query: str = "", limit: int = 20) -> list[SearchHit]:
    """Search specifically for certificates.

    Convenience wrapper around :func:`search` with
    ``instrument_type="certificate"``.
    """
    return search(query=query, limit=limit, instrument_type="certificate")
