"""Social media sentiment — Reddit headline fetcher.

Uses Reddit's public Atom feeds, no authentication needed.

2026-08-16: moved off the `.json` API. Reddit now 403s every unauthenticated
request to it — verified against five User-Agents (including a real browser
string) on both www and old hosts, all returning the same HTML block page
citing "network security". It is a policy block on the endpoint, not a
User-Agent problem, so no header tweak can recover it; the only anonymous
route left is the Atom feed.

The feed is rate-limited hard: back-to-back calls alternate 200/429 even
spaced 10s apart, hence the single retry. Atom carries no score or comment
count, which is why those fields are gone — nothing downstream read them
(sentiment.py consumes titles only).
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "finance-analyzer/1.0 (portfolio intelligence bot)"

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_RETRY_WAIT_S = 2.0

# Reddit serves roughly one Atom request per 20s per client; a signal pass wants
# 8 feeds at once, so most would 429. The cache is what makes the source usable:
# each feed is fetched at most once per FEED_TTL_S, and when a refresh does get
# 429'd we keep serving the last good copy rather than going blind. Headlines
# move far slower than the 600s loop, so stale-by-minutes costs nothing.
FEED_TTL_S = 600.0
_feed_cache: dict[str, tuple[float, str]] = {}


def clear_feed_cache():
    """Test seam — production never needs this."""
    _feed_cache.clear()


# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
TICKER_SUBREDDITS = {
    "BTC": [("Bitcoin", True), ("CryptoCurrency", False)],
    "ETH": [("ethereum", True), ("CryptoCurrency", False)],
    "PLTR": [("PLTR", True), ("wallstreetbets", False)],
    "NVDA": [("wallstreetbets", False), ("stocks", False)],
}

TICKER_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "PLTR": ["palantir", "pltr"],
    "NVDA": ["nvidia", "nvda"],
}


def _get_feed(url):
    """GET an Atom feed: cache-first, one retry, stale-on-error.

    Raises only when a feed is rate-limited AND we have never fetched it.
    """
    now = time.monotonic()
    cached = _feed_cache.get(url)
    if cached and (now - cached[0]) < FEED_TTL_S:
        return cached[1]

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if resp.status_code == 429:
            time.sleep(_RETRY_WAIT_S)
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        resp.raise_for_status()
    except Exception:
        if cached:
            logger.debug("reddit: refresh failed, serving cached feed for %s", url)
            return cached[1]
        raise

    _feed_cache[url] = (now, resp.text)
    return resp.text


def _parse_atom(xml_text, sub, keywords, dedicated):
    """Turn an Atom feed into post dicts, applying the keyword filter."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("reddit: unparseable feed for r/%s: %s", sub, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title_el = entry.find("atom:title", _ATOM_NS)
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue
        if not dedicated and not any(kw in title.lower() for kw in keywords):
            continue
        updated = entry.find("atom:updated", _ATOM_NS)
        published = entry.find("atom:published", _ATOM_NS)
        stamp = None
        for el in (updated, published):
            if el is not None and el.text:
                stamp = el.text.strip()
                break
        posts.append(
            {
                "title": title,
                "source": f"reddit/r/{sub}",
                "published": stamp or datetime.now(UTC).isoformat(),
            }
        )
    return posts


def _fetch_subreddit(sub, keywords, dedicated, per_sub):
    url = f"https://www.reddit.com/r/{sub}/hot.rss?limit={per_sub + 5}"
    return _parse_atom(_get_feed(url), sub, keywords, dedicated)


def _search_subreddit(sub, keywords, limit=10):
    query = quote(" OR ".join(keywords))
    url = (
        f"https://www.reddit.com/r/{sub}/search.rss"
        f"?q={query}&sort=new&restrict_sr=on&limit={limit}"
    )
    # dedicated=True: the query already restricts to the keywords, so a second
    # title-level filter would only drop matches phrased differently.
    return _parse_atom(_get_feed(url), sub, keywords, True)


def get_reddit_posts(ticker, limit=20):
    short = ticker.upper().replace("-USD", "")
    subreddits = TICKER_SUBREDDITS.get(short, [])
    keywords = TICKER_KEYWORDS.get(short, [short.lower()])
    if not subreddits:
        return []

    posts = []
    seen = set()
    per_sub = max(5, limit // len(subreddits))

    for sub, dedicated in subreddits:
        try:
            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
            for p in fetched:
                if p["title"] not in seen:
                    seen.add(p["title"])
                    posts.append(p)
        except Exception as e:
            print(f"    [Reddit r/{sub}] error: {e}")

    # Fallback: if keyword filtering yielded nothing, try search
    if not posts:
        for sub, _dedicated in subreddits[:1]:
            try:
                fetched = _search_subreddit(sub, keywords, limit=per_sub)
                for p in fetched:
                    if p["title"] not in seen:
                        seen.add(p["title"])
                        posts.append(p)
            except Exception as e:
                print(f"    [Reddit search r/{sub}] error: {e}")

    # Was sorted by score; Atom carries none, so rank by recency instead.
    posts.sort(key=lambda p: p.get("published") or "", reverse=True)
    return posts[:limit]


if __name__ == "__main__":
    for ticker in ["BTC", "ETH", "PLTR", "NVDA"]:
        print(f"\n{'='*60}")
        print(f"  Reddit posts for {ticker}")
        print(f"{'='*60}")
        posts = get_reddit_posts(ticker)
        print(f"  Found {len(posts)} posts")
        for p in posts[:5]:
            print(f"  [{p['published'][:16]}] {p['source']:>25}  {p['title'][:70]}")
