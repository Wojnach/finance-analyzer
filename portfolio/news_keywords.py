"""Shared keyword dictionaries and sector mappings for news/event signals.

Provides:
- Tiered keyword lists with severity weights
- headline scoring function
- Sector-to-ticker and ticker-to-sector mappings
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Keyword tiers: word/phrase -> weight multiplier for sentiment scoring
# ---------------------------------------------------------------------------

CRITICAL_KEYWORDS = {
    "war": 3.0,
    "crash": 3.0,
    "tariff": 3.0,
    "tariffs": 3.0,
    "sanctions": 3.0,
    "sanction": 3.0,
    "hack": 3.0,
    "hacked": 3.0,
    "default": 3.0,
    "bankruptcy": 3.0,
    "bankrupt": 3.0,
    "collapse": 3.0,
    "ban": 3.0,
    "banned": 3.0,
    "invasion": 3.0,
    "nuclear": 3.0,
}

HIGH_KEYWORDS = {
    "rate hike": 2.0,
    "rate cut": 2.0,
    "cpi": 2.0,
    "inflation": 2.0,
    "recession": 2.0,
    "delisting": 2.0,
    "delisted": 2.0,
    "lawsuit": 2.0,
    "indictment": 2.0,
    "sec investigation": 2.0,
    "trade war": 2.0,
    "downgrade": 2.0,
    "debt ceiling": 2.0,
    "margin call": 2.0,
    "layoffs": 2.0,
    "fraud": 2.0,
    "subpoena": 2.0,
}

MODERATE_KEYWORDS = {
    "etf approval": 1.5,
    "etf approved": 1.5,
    "regulation": 1.5,
    "regulatory": 1.5,
    "liquidation": 1.5,
    "liquidated": 1.5,
    "upgrade": 1.5,
    "earnings miss": 1.5,
    "earnings beat": 1.5,
    "guidance cut": 1.5,
    "guidance raise": 1.5,
    "buyback": 1.5,
    "stock split": 1.5,
    "merger": 1.5,
    "acquisition": 1.5,
    "ipo": 1.5,
}

# Combined dict for fast lookup (longer phrases checked first)
ALL_KEYWORDS = {**MODERATE_KEYWORDS, **HIGH_KEYWORDS, **CRITICAL_KEYWORDS}

# Pre-compile patterns sorted by length descending (match longer phrases first)
_KEYWORD_PATTERNS = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
    for kw, weight in sorted(ALL_KEYWORDS.items(), key=lambda x: -len(x[0]))
]

# ---------------------------------------------------------------------------
# Sector mappings
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    "semiconductor": {"NVDA", "AMD", "MU", "AVGO", "MRVL", "TSM", "SMCI", "IONQ"},
    "crypto": {"BTC-USD", "ETH-USD", "MSTR"},
    "china": {"BABA", "TSM"},
    "defense": {"LMT"},
    "big_tech": {"GOOGL", "AMZN", "AAPL", "META"},
    "ai": {"NVDA", "AMD", "GOOGL", "META", "PLTR", "SOUN", "IONQ", "MRVL", "AVGO"},
    "software": {"PLTR", "UPST", "RXRX", "VERI", "PONY", "GRRR", "TEM"},
    "gaming": {"TTWO"},
    "infrastructure": {"VRT"},
    "metals": {"XAU-USD", "XAG-USD"},
    "etf": {"QQQ"},
}

# Reverse map: ticker -> set of sectors
TICKER_SECTORS: dict[str, set[str]] = {}
for _sector, _tickers in SECTOR_MAP.items():
    for _t in _tickers:
        TICKER_SECTORS.setdefault(_t, set()).add(_sector)

# Keywords that imply directional impact on specific sectors
KEYWORD_SECTOR_IMPACT = {
    "tariff": {"semiconductor": "SELL", "china": "SELL", "metals": "BUY"},
    "tariffs": {"semiconductor": "SELL", "china": "SELL", "metals": "BUY"},
    "trade war": {"semiconductor": "SELL", "china": "SELL", "metals": "BUY"},
    "sanctions": {"china": "SELL", "crypto": "SELL"},
    "sanction": {"china": "SELL", "crypto": "SELL"},
    "ban": {"crypto": "SELL", "china": "SELL"},
    "banned": {"crypto": "SELL", "china": "SELL"},
    "hack": {"crypto": "SELL"},
    "hacked": {"crypto": "SELL"},
    "rate hike": {"big_tech": "SELL", "crypto": "SELL", "metals": "SELL"},
    "rate cut": {"big_tech": "BUY", "crypto": "BUY", "metals": "BUY"},
    "recession": {"etf": "SELL", "big_tech": "SELL", "defense": "BUY"},
    "inflation": {"metals": "BUY", "crypto": "BUY"},
    "etf approval": {"crypto": "BUY"},
    "etf approved": {"crypto": "BUY"},
    "regulation": {"crypto": "SELL"},
    "regulatory": {"crypto": "SELL"},
    "war": {"defense": "BUY", "metals": "BUY", "etf": "SELL"},
    "invasion": {"defense": "BUY", "metals": "BUY", "etf": "SELL"},
}

# Credible financial news sources (weight 1.5x)
CREDIBLE_SOURCES = {
    "reuters", "bloomberg", "wsj", "wall street journal",
    "cnbc", "associated press", "ap", "financial times", "ft",
    "bbc", "new york times", "nyt", "the economist",
    "marketwatch", "barron's", "barrons",
}


def score_headline(title: str) -> tuple[float, list[str]]:
    """Score a headline by keyword severity.

    Returns:
        (max_weight, matched_keywords) — max_weight is the highest keyword
        weight found (1.0 if no keywords match), matched_keywords lists all
        matched keyword strings.
    """
    if not title:
        return 1.0, []

    matched = []
    max_weight = 1.0

    for pattern, weight in _KEYWORD_PATTERNS:
        if pattern.search(title):
            matched.append(pattern.pattern.replace(r"\b", "").replace("\\", ""))
            if weight > max_weight:
                max_weight = weight

    return max_weight, matched


def keyword_severity(title: str) -> str:
    """Classify headline severity: critical, high, moderate, or normal."""
    weight, _ = score_headline(title)
    if weight >= 3.0:
        return "critical"
    if weight >= 2.0:
        return "high"
    if weight >= 1.5:
        return "moderate"
    return "normal"


def is_credible_source(source: str) -> bool:
    """Check if a source name matches a credible financial news source."""
    if not source:
        return False
    lower = source.lower().strip()
    return any(cs in lower for cs in CREDIBLE_SOURCES)


def get_sector_impact(keyword: str, ticker: str) -> str | None:
    """Get the directional impact of a keyword on a specific ticker.

    Returns "BUY", "SELL", or None if no sector-specific impact.
    """
    impacts = KEYWORD_SECTOR_IMPACT.get(keyword.lower(), {})
    ticker_secs = TICKER_SECTORS.get(ticker, set())
    for sector, direction in impacts.items():
        if sector in ticker_secs:
            return direction
    return None
