"""Shared keyword dictionaries and sector mappings for news/event signals.

Provides:
- Tiered keyword lists with severity weights
- headline scoring function
- Sector-to-ticker and ticker-to-sector mappings
"""

from __future__ import annotations

import re
from datetime import UTC

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
    "semiconductor": {"NVDA", "AMD", "MU", "AVGO", "TSM", "SMCI"},
    "crypto": {"BTC-USD", "ETH-USD", "MSTR"},
    "defense": {"LMT"},
    "big_tech": {"GOOGL", "AMZN", "AAPL", "META"},
    "ai": {"NVDA", "AMD", "GOOGL", "META", "PLTR", "SOUN", "AVGO"},
    "software": {"PLTR"},
    "gaming": {"TTWO"},
    "infrastructure": {"VRT"},
    "metals": {"XAU-USD", "XAG-USD"},
}

# Reverse map: ticker -> set of sectors
TICKER_SECTORS: dict[str, set[str]] = {}
for _sector, _tickers in SECTOR_MAP.items():
    for _t in _tickers:
        TICKER_SECTORS.setdefault(_t, set()).add(_sector)

# Keywords that imply directional impact on specific sectors
KEYWORD_SECTOR_IMPACT = {
    "tariff": {"semiconductor": "SELL", "metals": "BUY"},
    "tariffs": {"semiconductor": "SELL", "metals": "BUY"},
    "trade war": {"semiconductor": "SELL", "metals": "BUY"},
    "sanctions": {"crypto": "SELL"},
    "sanction": {"crypto": "SELL"},
    "ban": {"crypto": "SELL"},
    "banned": {"crypto": "SELL"},
    "hack": {"crypto": "SELL"},
    "hacked": {"crypto": "SELL"},
    "rate hike": {"big_tech": "SELL", "crypto": "SELL", "metals": "SELL"},
    "rate cut": {"big_tech": "BUY", "crypto": "BUY", "metals": "BUY"},
    "recession": {"big_tech": "SELL", "defense": "BUY"},
    "inflation": {"metals": "BUY", "crypto": "BUY"},
    "etf approval": {"crypto": "BUY"},
    "etf approved": {"crypto": "BUY"},
    "regulation": {"crypto": "SELL"},
    "regulatory": {"crypto": "SELL"},
    "war": {"defense": "BUY", "metals": "BUY", "big_tech": "SELL"},
    "invasion": {"defense": "BUY", "metals": "BUY", "big_tech": "SELL"},
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


def dissemination_score(articles: list[dict]) -> float:
    """Score how widely news has spread (FinGPT dissemination-aware pattern).

    Factors:
    1. Unique source count — more sources = wider spread
    2. Source diversity — credible sources (Reuters, Bloomberg) weight more
    3. Time clustering — articles within 1h of each other = breaking news

    Returns:
        Float multiplier (1.0 = normal, up to 3.0 for breaking news with wide coverage).
        Used to amplify headline weights in sentiment aggregation.
    """
    if not articles or len(articles) < 2:
        return 1.0

    # Factor 1: Unique source count
    sources = set()
    for a in articles:
        src = a.get("source", "unknown").lower().strip()
        if src:
            sources.add(src)
    source_count = len(sources)
    # 1 source = 1.0, 3+ sources = 1.5, 5+ = 2.0
    source_factor = min(1.0 + (source_count - 1) * 0.25, 2.0)

    # Factor 2: Source diversity — credible source presence
    credible_count = sum(1 for s in sources if any(cs in s for cs in CREDIBLE_SOURCES))
    diversity_factor = 1.0
    if credible_count >= 2:
        diversity_factor = 1.5
    elif credible_count >= 1:
        diversity_factor = 1.25

    # Factor 3: Time clustering — articles within 1h of each other
    from datetime import datetime
    timestamps = []
    for a in articles:
        pub = a.get("published", "")
        if not pub:
            continue
        try:
            if isinstance(pub, (int, float)):
                ts = datetime.fromtimestamp(pub, tz=UTC)
            else:
                # Try ISO format
                pub_str = str(pub).replace("Z", "+00:00")
                ts = datetime.fromisoformat(pub_str)
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError, OSError):
            continue

    clustering_factor = 1.0
    if len(timestamps) >= 3:
        timestamps.sort()
        # Check if most articles appeared within a 1-hour window
        window = 3600  # 1 hour
        max_cluster = 1
        for i in range(len(timestamps)):
            cluster = sum(1 for t in timestamps if abs(t - timestamps[i]) <= window)
            max_cluster = max(max_cluster, cluster)
        # If 60%+ of articles are in a 1h cluster, it's breaking news
        cluster_ratio = max_cluster / len(timestamps)
        if cluster_ratio >= 0.6:
            clustering_factor = 1.5

    # Combined score (multiplicative, capped at 3.0)
    score = source_factor * diversity_factor * clustering_factor
    return min(round(score, 2), 3.0)


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


# ---------------------------------------------------------------------------
# Headline relevance (added 2026-04-28 for sentiment regression fix)
# ---------------------------------------------------------------------------
#
# Background: shadow LLM accuracy investigation found that the sentiment
# pipeline was scoring every wire-feed headline returned by Yahoo/CryptoCompare,
# including bare price-tickers like "Bitcoin: $67,123" and generic
# "Markets mixed in afternoon trade" boilerplate. Models correctly labeled
# these neutral, but the neutral mass drowned out the few decisive headlines
# in the average. Sentiment regressed from 75.3% -> ~42% over W16-W17.
#
# A headline is "relevant" to a ticker if:
#   1. It triggers a keyword from score_headline (weight > 1.0), OR
#   2. It mentions the ticker symbol or a known synonym (Bitcoin/BTC,
#      Ethereum/ETH, gold/XAU, silver/XAG, or the stock symbol)
# Source-credibility lives in the wrapper in portfolio/sentiment.py, not here.

_TICKER_SYNONYMS: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin", "bitcoins"],
    "ETH": ["eth", "ethereum", "ether"],
    "XAU": ["xau", "gold", "bullion"],
    "XAG": ["xag", "silver"],
    "MSTR": ["mstr", "microstrategy"],
    "NVDA": ["nvda", "nvidia"],
    "AMD": ["amd"],
    "GOOGL": ["googl", "google", "alphabet"],
    "AMZN": ["amzn", "amazon"],
    "AAPL": ["aapl", "apple"],
    "META": ["meta", "facebook", "instagram"],
    "AVGO": ["avgo", "broadcom"],
    "TSM": ["tsm", "tsmc"],
    "MU": ["mu", "micron"],
    "PLTR": ["pltr", "palantir"],
    "SMCI": ["smci"],
    "TTWO": ["ttwo", "rockstar"],
    "VRT": ["vrt", "vertiv"],
    "LMT": ["lmt", "lockheed"],
    "SOUN": ["soun", "soundhound"],
}


def _ticker_synonym_pattern(ticker: str) -> re.Pattern | None:
    short = ticker.upper().replace("-USD", "")
    syns = _TICKER_SYNONYMS.get(short)
    if not syns:
        if not short or not short.isalnum():
            return None
        return re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)
    pattern = "|".join(re.escape(s) for s in syns)
    return re.compile(r"\b(" + pattern + r")\b", re.IGNORECASE)


# Memoize per-ticker patterns; tickers are a fixed small set so the cache
# never grows large. None values are cached too (use sentinel-via-membership).
_PATTERN_CACHE: dict[str, re.Pattern | None] = {}


def is_relevant_headline(title: str, ticker: str) -> bool:
    """Return True if the headline is plausibly relevant to the ticker.

    Used by the sentiment pipeline to filter wire-noise before model inference.
    See the module-level comment block above for background.

    Minimum content gate: even when a ticker synonym matches, the headline
    must have at least 3 word tokens AFTER stripping the synonym itself.
    This drops bare price-tickers like "Bitcoin: $67,123" (1 token after
    removing "Bitcoin") while keeping "Bitcoin treasury firm adds 500 BTC"
    (5 tokens after removing the synonyms).
    """
    if not title or not title.strip():
        return False

    weight, _ = score_headline(title)
    if weight > 1.0:
        return True

    short = ticker.upper().replace("-USD", "")
    if short not in _PATTERN_CACHE:
        _PATTERN_CACHE[short] = _ticker_synonym_pattern(short)
    pat = _PATTERN_CACHE[short]
    if pat is None or not pat.search(title):
        return False

    # Synonym matched — guard against bare price-ticker noise. Strip the
    # synonym occurrences and count the remaining word tokens; need >=3 to
    # be considered real content.
    stripped = pat.sub(" ", title)
    tokens = [t for t in re.findall(r"\b[A-Za-z]{2,}\b", stripped) if t.lower() not in {"the", "and", "for", "from"}]
    return len(tokens) >= 3
