"""Regression tests for the 2026-08-16 news-source outage.

Three independent failures had blinded the news/sentiment path:

1. Reddit began 403-ing every unauthenticated `.json` API call. Verified
   against five different User-Agents on both www and old hosts — it is a
   policy block on the endpoint, not a UA problem, so the fetcher moved to
   the Atom (`.rss`) feed, which still serves anonymously.
2. `config` was never threaded into the Swedbank/watchlist signal path, so
   `signal_engine` read `cryptocompare_api_key` (and `newsapi_key`) off an
   empty dict and issued unauthenticated requests -> HTTP 401.
3. The crypto headline path only ever consulted CryptoCompare, so when that
   account went over its monthly quota there was no second source, even
   though a working NewsAPI key was sitting in the same config.
"""

import pytest

from portfolio import sentiment, social_sentiment

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Bitcoin rips higher on ETF inflows</title>
    <link href="https://reddit.com/r/Bitcoin/x"/>
    <updated>2026-08-16T10:00:00+00:00</updated>
  </entry>
  <entry>
    <title>Daily discussion thread</title>
    <link href="https://reddit.com/r/Bitcoin/y"/>
    <updated>2026-08-16T09:00:00+00:00</updated>
  </entry>
</feed>
"""


@pytest.fixture(autouse=True)
def _clean_feed_cache():
    """The Reddit feed cache is module-level; keep tests independent."""
    social_sentiment.clear_feed_cache()
    yield
    social_sentiment.clear_feed_cache()


class _Resp:
    def __init__(self, status=200, text="", json_data=None):
        self.status_code = status
        self.text = text
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Client Error")


# --- 1. Reddit: Atom feed instead of the blocked .json API -----------------


def test_reddit_fetch_uses_rss_not_json(monkeypatch):
    """The .json endpoint is 403-blocked by Reddit; we must request .rss."""
    seen = []

    def fake_get(url, **kw):
        seen.append(url)
        return _Resp(200, ATOM)

    monkeypatch.setattr(social_sentiment.requests, "get", fake_get)
    social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)

    assert seen, "no request issued"
    assert ".rss" in seen[0], f"expected an .rss feed, requested {seen[0]}"
    assert ".json" not in seen[0], f"still hitting the blocked JSON API: {seen[0]}"


def test_reddit_parses_atom_entries(monkeypatch):
    monkeypatch.setattr(
        social_sentiment.requests, "get", lambda url, **kw: _Resp(200, ATOM)
    )
    posts = social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)

    titles = [p["title"] for p in posts]
    assert "Bitcoin rips higher on ETF inflows" in titles
    assert all(p["source"] == "reddit/r/Bitcoin" for p in posts)
    assert all(p["published"] for p in posts)


def test_reddit_retries_once_on_429(monkeypatch):
    """Reddit rate-limits the Atom feed aggressively; one retry recovers it."""
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return _Resp(429, "") if calls["n"] == 1 else _Resp(200, ATOM)

    monkeypatch.setattr(social_sentiment.requests, "get", fake_get)
    monkeypatch.setattr(social_sentiment.time, "sleep", lambda _s: None)

    posts = social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)
    assert calls["n"] == 2, "should retry once after a 429"
    assert posts, "retry should have produced posts"


def test_reddit_feed_is_cached_between_calls(monkeypatch):
    """One fetch per feed per TTL — otherwise Reddit 429s the whole pass."""
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return _Resp(200, ATOM)

    monkeypatch.setattr(social_sentiment.requests, "get", fake_get)
    for _ in range(3):
        social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)
    assert calls["n"] == 1, f"expected 1 network call, made {calls['n']}"


def test_reddit_serves_stale_feed_when_refresh_is_rate_limited(monkeypatch):
    """A 429 on refresh must not blank the signal — serve the last good copy."""
    state = {"n": 0}

    def fake_get(url, **kw):
        state["n"] += 1
        return _Resp(200, ATOM) if state["n"] == 1 else _Resp(429, "")

    monkeypatch.setattr(social_sentiment.requests, "get", fake_get)
    monkeypatch.setattr(social_sentiment.time, "sleep", lambda _s: None)

    first = social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)
    assert first

    # expire the cache without waiting out the real TTL
    url = next(iter(social_sentiment._feed_cache))
    ts, text = social_sentiment._feed_cache[url]
    social_sentiment._feed_cache[url] = (ts - social_sentiment.FEED_TTL_S - 1, text)

    second = social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)
    assert [p["title"] for p in second] == [p["title"] for p in first]


def test_reddit_raises_when_rate_limited_with_no_cache(monkeypatch):
    """No cached copy and a hard 429 is a genuine failure the caller logs."""
    import requests as _rq

    monkeypatch.setattr(
        social_sentiment.requests, "get", lambda url, **kw: _Resp(429, "")
    )
    monkeypatch.setattr(social_sentiment.time, "sleep", lambda _s: None)
    with pytest.raises(_rq.HTTPError):
        social_sentiment._fetch_subreddit("Bitcoin", ["bitcoin"], True, 5)


def test_reddit_keyword_filter_still_applies_to_general_subs(monkeypatch):
    monkeypatch.setattr(
        social_sentiment.requests, "get", lambda url, **kw: _Resp(200, ATOM)
    )
    posts = social_sentiment._fetch_subreddit("CryptoCurrency", ["bitcoin"], False, 5)
    assert [p["title"] for p in posts] == ["Bitcoin rips higher on ETF inflows"]


# --- 2. config must reach the Swedbank/watchlist signal path ---------------


def test_swedbank_evaluate_forwards_config_to_signal_engine(monkeypatch):
    """evaluate() must hand a real config to generate_signal, or keys vanish."""
    import numpy as np
    import pandas as pd

    from portfolio.swedbank import ohlcv as ohlcv_mod
    from portfolio.swedbank import signals as sigmod

    n = 260
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 140.0, n), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000.0), index=idx),
        }
    )
    monkeypatch.setattr(ohlcv_mod, "fetch", lambda *a, **kw: (df, "test"))

    seen = {}
    import portfolio.signal_engine as se

    def fake_generate_signal(ind, ticker=None, config=None, df=None, **kw):
        seen["config"] = config
        return ("HOLD", 0.0, {})

    monkeypatch.setattr(se, "generate_signal", fake_generate_signal)

    from portfolio.swedbank.instruments import INSTRUMENTS

    cfg = {"cryptocompare_api_key": "CC", "newsapi_key": "NA"}
    row = sigmod.evaluate(INSTRUMENTS["NVDA"], horizon="1d", config=cfg)
    assert "error" not in row, row
    assert seen.get("config") == cfg, "config did not reach generate_signal"


def test_swedbank_evaluate_defaults_config_from_disk(monkeypatch):
    """The real callers pass nothing; the default must still carry the keys."""
    from portfolio.swedbank import signals as sigmod

    monkeypatch.setattr(
        sigmod, "_default_config", lambda: {"cryptocompare_api_key": "FROM_DISK"}
    )
    seen = {}
    import portfolio.signal_engine as se

    monkeypatch.setattr(
        se,
        "generate_signal",
        lambda ind, ticker=None, config=None, df=None, **kw: (
            seen.update(config=config) or ("HOLD", 0.0, {})
        ),
    )
    from portfolio.swedbank import ohlcv as ohlcv_mod

    import numpy as np
    import pandas as pd

    n = 260
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 140.0, n), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000.0), index=idx),
        }
    )
    monkeypatch.setattr(ohlcv_mod, "fetch", lambda *a, **kw: (df, "test"))

    from portfolio.swedbank.instruments import INSTRUMENTS

    sigmod.evaluate(INSTRUMENTS["NVDA"], horizon="1d")
    assert seen.get("config") == {"cryptocompare_api_key": "FROM_DISK"}


def test_watchlist_binds_config_into_evaluate(monkeypatch):
    """evaluate_watchlist used to call evaluate() bare, dropping every key."""
    import inspect

    from portfolio.swedbank import watchlist as wlmod

    src = inspect.getsource(wlmod.evaluate_watchlist)
    assert "_default_config" in src, "watchlist still never resolves a config"
    assert "partial" in src, "watchlist does not bind the config onto evaluate"


def test_swedbank_loop_loads_config_for_signals():
    """The loop must supply a real config, not fall through on the None default."""
    import inspect

    import portfolio.swedbank.signals as sigmod

    src = inspect.getsource(sigmod.evaluate_universe)
    assert (
        "_default_config" in src or "load_json" in src
    ), "evaluate_universe still has no way to obtain a config"


def test_signal_engine_reads_keys_off_config():
    """Guard the exact lookup that silently yielded None and caused the 401."""
    import inspect

    import portfolio.signal_engine as se

    src = inspect.getsource(se)
    assert 'get("cryptocompare_api_key"' in src
    assert 'get("newsapi_key"' in src


# --- 3. crypto headlines: CryptoCompare AND NewsAPI, freshest first --------


@pytest.fixture
def newsapi_open(monkeypatch):
    """Open the NewsAPI quota/TTL gates and bypass the process-wide _cached.

    The crypto path deliberately reuses shared_state's budget machinery, so a
    test that patched _fetch_newsapi_headlines directly would never be reached.
    """
    from portfolio import shared_state

    monkeypatch.setattr(shared_state, "newsapi_quota_ok", lambda: True)
    monkeypatch.setattr(shared_state, "newsapi_ttl_for_ticker", lambda t: 600)
    monkeypatch.setattr(
        shared_state, "_cached", lambda key, ttl, fn, *a, **kw: fn(*a, **kw)
    )
    # Yahoo is a merge participant now, so silence it by default — a test that
    # cares about Yahoo overrides this itself.
    monkeypatch.setattr(
        sentiment, "_fetch_crypto_headlines_yahoo_fallback", lambda t, limit=20: []
    )
    return shared_state


def _set_newsapi(monkeypatch, articles):
    monkeypatch.setattr(
        sentiment,
        "_fetch_newsapi_with_tracking",
        lambda ticker, key, limit=10, query=None: list(articles),
    )


def _set_cryptocompare(monkeypatch, articles):
    monkeypatch.setattr(
        sentiment,
        "_fetch_cryptocompare_headlines",
        lambda t, limit, key: list(articles),
    )


def test_crypto_headlines_merges_both_sources(monkeypatch, newsapi_open):
    _set_cryptocompare(
        monkeypatch,
        [
            {
                "title": "CC story",
                "source": "cc",
                "published": "2026-08-16T08:00:00+00:00",
            }
        ],
    )
    _set_newsapi(
        monkeypatch,
        [
            {
                "title": "NA story",
                "source": "na",
                "published": "2026-08-16T09:00:00+00:00",
            }
        ],
    )
    out = sentiment._fetch_crypto_headlines(
        "BTC", cryptocompare_api_key="CC", newsapi_key="NA"
    )
    titles = [a["title"] for a in out]
    assert "CC story" in titles and "NA story" in titles, titles


def test_crypto_headlines_freshest_first(monkeypatch, newsapi_open):
    _set_cryptocompare(
        monkeypatch,
        [{"title": "older", "source": "cc", "published": "2026-08-16T01:00:00+00:00"}],
    )
    _set_newsapi(
        monkeypatch,
        [{"title": "newer", "source": "na", "published": "2026-08-16T23:00:00+00:00"}],
    )
    out = sentiment._fetch_crypto_headlines(
        "BTC", cryptocompare_api_key="CC", newsapi_key="NA"
    )
    assert [a["title"] for a in out] == ["newer", "older"]


def test_crypto_headlines_dedupes_same_story(monkeypatch, newsapi_open):
    dup = "Bitcoin ETF sees record inflow"
    _set_cryptocompare(
        monkeypatch,
        [{"title": dup, "source": "cc", "published": "2026-08-16T01:00:00+00:00"}],
    )
    _set_newsapi(
        monkeypatch,
        [
            {
                "title": dup + "  ",
                "source": "na",
                "published": "2026-08-16T02:00:00+00:00",
            }
        ],
    )
    out = sentiment._fetch_crypto_headlines(
        "BTC", cryptocompare_api_key="CC", newsapi_key="NA"
    )
    assert len(out) == 1, out
    assert out[0]["published"].startswith("2026-08-16T02"), "kept the staler copy"


def test_crypto_headlines_survives_one_dead_source(monkeypatch, newsapi_open):
    """CryptoCompare being over quota must not blank out the feed."""
    _set_cryptocompare(monkeypatch, [])
    _set_newsapi(
        monkeypatch,
        [
            {
                "title": "NA story",
                "source": "na",
                "published": "2026-08-16T09:00:00+00:00",
            }
        ],
    )
    out = sentiment._fetch_crypto_headlines(
        "BTC", cryptocompare_api_key="CC", newsapi_key="NA"
    )
    assert [a["title"] for a in out] == ["NA story"]


def test_crypto_headlines_falls_back_to_yahoo_when_both_dead(monkeypatch, newsapi_open):
    _set_cryptocompare(monkeypatch, [])
    _set_newsapi(monkeypatch, [])
    monkeypatch.setattr(
        sentiment,
        "_fetch_crypto_headlines_yahoo_fallback",
        lambda t, limit=20: [
            {
                "title": "YF story",
                "source": "yf",
                "published": "2026-08-16T09:00:00+00:00",
            }
        ],
    )
    out = sentiment._fetch_crypto_headlines(
        "BTC", cryptocompare_api_key="CC", newsapi_key="NA"
    )
    assert [a["title"] for a in out] == ["YF story"]


def test_get_sentiment_passes_newsapi_key_into_crypto_path(monkeypatch):
    """The crypto branch used to drop newsapi_key on the floor."""
    seen = {}

    def fake(ticker="BTC", limit=20, *, cryptocompare_api_key=None, newsapi_key=None):
        seen["cc"] = cryptocompare_api_key
        seen["na"] = newsapi_key
        return []

    monkeypatch.setattr(sentiment, "_fetch_crypto_headlines", fake)
    sentiment.get_sentiment("BTC", newsapi_key="NA", cryptocompare_api_key="CC")
    assert seen == {"cc": "CC", "na": "NA"}, seen


@pytest.mark.parametrize(
    "quota_msg", ["You are over your rate limit please upgrade your account!"]
)
def test_cryptocompare_over_quota_returns_empty_not_exception(monkeypatch, quota_msg):
    monkeypatch.setattr(
        sentiment,
        "fetch_json",
        lambda *a, **kw: {"Response": "Error", "Message": quota_msg},
    )
    assert sentiment._fetch_cryptocompare_headlines("BTC", 20, "CC") == []


# --- 4. news_event: the second, independent headline call site -------------


def test_news_event_crypto_branch_passes_api_keys(monkeypatch):
    """news_event fetches headlines itself and used to drop every key.

    The stock branch already forwarded newsapi_key; the crypto branch called
    _fetch_crypto_headlines(short) bare, which is the second source of the
    401 storm (the first being the Swedbank config drop).
    """
    from portfolio.signals import news_event

    seen = {}

    def fake_crypto(ticker, limit=20, *, cryptocompare_api_key=None, newsapi_key=None):
        seen["cc"] = cryptocompare_api_key
        seen["na"] = newsapi_key
        return []

    monkeypatch.setattr("portfolio.sentiment._fetch_crypto_headlines", fake_crypto)
    monkeypatch.setattr(
        news_event, "_cached", lambda key, ttl, fn, *a, **kw: fn(*a, **kw)
    )

    cfg = {"cryptocompare_api_key": "CC", "newsapi_key": "NA"}
    news_event._fetch_headlines("BTC-USD", cfg)
    assert seen == {"cc": "CC", "na": "NA"}, seen


def test_yahoo_participates_in_merge_not_just_fallback(monkeypatch, newsapi_open):
    """Yahoo must compete on recency, not sit behind the others.

    Observed live 2026-08-16: NewsAPI returned Aug-15 stories while Yahoo had
    Aug-16 ones. Fallback-only ordering served the staler set.
    """
    _set_cryptocompare(monkeypatch, [])
    _set_newsapi(
        monkeypatch,
        [
            {
                "title": "stale NA",
                "source": "na",
                "published": "2026-08-15T20:00:00+00:00",
            }
        ],
    )
    monkeypatch.setattr(
        sentiment,
        "_fetch_crypto_headlines_yahoo_fallback",
        lambda t, limit=20: [
            {
                "title": "fresh YF",
                "source": "yf",
                "published": "2026-08-16T18:58:00+00:00",
            }
        ],
    )
    out = sentiment._fetch_crypto_headlines(
        "BTC", cryptocompare_api_key="CC", newsapi_key="NA"
    )
    assert [a["title"] for a in out] == ["fresh YF", "stale NA"], out
