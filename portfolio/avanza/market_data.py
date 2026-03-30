"""Market data retrieval — quotes, depth, OHLC, instrument info, news.

Thin typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
raw delegators.  Every function returns our own dataclasses from
:mod:`portfolio.avanza.types`.
"""

from __future__ import annotations

import logging
from typing import Any

from avanza.constants import Resolution, TimePeriod

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import (
    OHLC,
    InstrumentInfo,
    MarketData,
    NewsArticle,
    Quote,
)

logger = logging.getLogger("portfolio.avanza.market_data")

# ---------------------------------------------------------------------------
# Resolution lookup (period -> sensible default resolution)
# ---------------------------------------------------------------------------

_DEFAULT_RESOLUTION: dict[str, Resolution] = {
    "TODAY": Resolution.THIRTY_MINUTES,
    "ONE_WEEK": Resolution.THIRTY_MINUTES,
    "ONE_MONTH": Resolution.DAY,
    "THREE_MONTHS": Resolution.DAY,
    "THIS_YEAR": Resolution.WEEK,
    "ONE_YEAR": Resolution.WEEK,
    "THREE_YEARS": Resolution.MONTH,
    "FIVE_YEARS": Resolution.MONTH,
    "INFINITY": Resolution.MONTH,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_quote(ob_id: str, instrument_type: str = "certificate") -> Quote:
    """Fetch a live quote for the given orderbook ID.

    Calls ``client.avanza.get_instrument(type, id)`` and parses the
    result into a :class:`~portfolio.avanza.types.Quote`.
    """
    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.avanza.get_instrument(instrument_type, ob_id)
    logger.debug("get_quote ob_id=%s raw_keys=%s", ob_id, list(raw.keys()))
    return Quote.from_api(raw)


def get_market_data(ob_id: str) -> MarketData:
    """Fetch full market data (quote + depth + recent trades).

    Calls ``client.get_market_data_raw(id)`` and parses the result
    into a :class:`~portfolio.avanza.types.MarketData`.
    """
    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.get_market_data_raw(ob_id)
    logger.debug("get_market_data ob_id=%s raw_keys=%s", ob_id, list(raw.keys()))
    return MarketData.from_api(raw)


def get_ohlc(
    ob_id: str,
    period: str = "ONE_MONTH",
    resolution: str | None = None,
) -> list[OHLC]:
    """Fetch OHLCV candles for the given orderbook ID.

    Args:
        ob_id: Avanza orderbook ID.
        period: Time period string (e.g. ``"ONE_MONTH"``, ``"ONE_WEEK"``).
        resolution: Optional resolution override.  When *None* a sensible
            default is chosen based on *period*.

    Returns:
        List of :class:`~portfolio.avanza.types.OHLC` candles.
    """
    client = AvanzaClient.get_instance()

    tp = TimePeriod[period]
    if resolution is not None:
        res = Resolution[resolution]
    else:
        res = _DEFAULT_RESOLUTION.get(period, Resolution.DAY)

    raw: Any = client.avanza.get_chart_data(ob_id, tp, res)
    logger.debug(
        "get_ohlc ob_id=%s period=%s resolution=%s candles=%d",
        ob_id,
        period,
        res.name,
        len(raw) if isinstance(raw, list) else 0,
    )

    # The API may return a dict with an "ohlc" key or a plain list.
    candles: list[dict[str, Any]]
    if isinstance(raw, dict):
        candles = raw.get("ohlc", raw.get("dataPoints", []))
    elif isinstance(raw, list):
        candles = raw
    else:
        candles = []

    return [OHLC.from_api(c) for c in candles]


def get_instrument_info(
    ob_id: str,
    instrument_type: str = "certificate",
) -> InstrumentInfo:
    """Fetch instrument metadata (leverage, barrier, underlying, etc.).

    Calls ``client.avanza.get_instrument(type, id)`` and parses the
    result into a :class:`~portfolio.avanza.types.InstrumentInfo`.
    """
    client = AvanzaClient.get_instance()
    raw: dict[str, Any] = client.avanza.get_instrument(instrument_type, ob_id)
    logger.debug("get_instrument_info ob_id=%s name=%s", ob_id, raw.get("name"))
    return InstrumentInfo.from_api(raw)


def get_news(ob_id: str) -> list[NewsArticle]:
    """Fetch news articles linked to the given orderbook ID.

    Calls ``client.get_news_raw(id)`` and parses the result into a
    list of :class:`~portfolio.avanza.types.NewsArticle`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_news_raw(ob_id)
    logger.debug("get_news ob_id=%s", ob_id)

    # The API may return a list directly or a dict with an "articles" key.
    articles: list[dict[str, Any]]
    if isinstance(raw, dict):
        articles = raw.get("articles", raw.get("news", []))
    elif isinstance(raw, list):
        articles = raw
    else:
        articles = []

    return [NewsArticle.from_api(a) for a in articles]
