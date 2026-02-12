"""Social media sentiment — Reddit and Twitter/X headline fetchers.

Reddit: uses public JSON API, no authentication needed.
Twitter/X: requires Bearer Token (Basic tier $200/mo). Gracefully skips if not configured.
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "finance-analyzer/1.0 (portfolio intelligence bot)"

# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
TICKER_SUBREDDITS = {
    "BTC": [("Bitcoin", True), ("CryptoCurrency", False)],
    "ETH": [("ethereum", True), ("CryptoCurrency", False)],
    "MSTR": [("wallstreetbets", False), ("stocks", False)],
    "PLTR": [("PLTR", True), ("wallstreetbets", False)],
    "NVDA": [("wallstreetbets", False), ("stocks", False)],
}

TICKER_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "MSTR": ["microstrategy", "mstr", "saylor"],
    "PLTR": ["palantir", "pltr"],
    "NVDA": ["nvidia", "nvda"],
}


def _fetch_subreddit(sub, keywords, dedicated, per_sub):
    posts = []
    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub + 5}&raw_json=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        title = post.get("title", "").strip()
        if not title or post.get("stickied"):
            continue
        if not dedicated and not any(kw in title.lower() for kw in keywords):
            continue
        created = post.get("created_utc", 0)
        posts.append(
            {
                "title": title,
                "source": f"reddit/r/{sub}",
                "published": (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created
                    else datetime.now(timezone.utc).isoformat()
                ),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
            }
        )
    return posts


def _search_subreddit(sub, keywords, limit=10):
    query = urllib.parse.quote(" OR ".join(keywords))
    url = (
        f"https://www.reddit.com/r/{sub}/search.json"
        f"?q={query}&sort=new&restrict_sr=on&limit={limit}&raw_json=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        title = post.get("title", "").strip()
        if not title:
            continue
        created = post.get("created_utc", 0)
        posts.append(
            {
                "title": title,
                "source": f"reddit/r/{sub}",
                "published": (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created
                    else datetime.now(timezone.utc).isoformat()
                ),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
            }
        )
    return posts


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

    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
    return posts[:limit]


def get_twitter_posts(ticker, bearer_token, limit=20):
    """Fetch recent tweets. Requires X API Basic tier ($200/mo) bearer token."""
    if not bearer_token:
        return []
    short = ticker.upper().replace("-USD", "")
    keywords = TICKER_KEYWORDS.get(short, [short])
    cashtag = f"${short}"
    parts = [f'"{cashtag}"'] + [f'"{kw}"' for kw in keywords[:2]]
    query = f"({' OR '.join(parts)}) -is:retweet lang:en"

    try:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "max_results": min(max(limit, 10), 100),
                "tweet.fields": "created_at,public_metrics",
            }
        )
        url = f"https://api.twitter.com/2/tweets/search/recent?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        tweets = data.get("data", [])
        posts = []
        for tweet in tweets[:limit]:
            text = tweet.get("text", "").strip()
            if not text:
                continue
            metrics = tweet.get("public_metrics", {})
            posts.append(
                {
                    "title": text[:280],
                    "source": "twitter/x",
                    "published": tweet.get(
                        "created_at", datetime.now(timezone.utc).isoformat()
                    ),
                    "score": metrics.get("like_count", 0)
                    + metrics.get("retweet_count", 0),
                }
            )
        return posts
    except Exception as e:
        print(f"    [Twitter/X] error: {e}")
        return []


if __name__ == "__main__":
    for ticker in ["BTC", "ETH", "PLTR", "NVDA"]:
        print(f"\n{'='*60}")
        print(f"  Reddit posts for {ticker}")
        print(f"{'='*60}")
        posts = get_reddit_posts(ticker)
        print(f"  Found {len(posts)} posts")
        for p in posts[:5]:
            score = p.get("score", 0)
            print(f"  [{score:>5}] {p['source']:>25}  {p['title'][:70]}")
