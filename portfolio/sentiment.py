import json
import subprocess
from datetime import datetime, timezone

import platform
from pathlib import Path

from portfolio.http_retry import fetch_with_retry

CRYPTO_TICKERS = {"BTC", "ETH"}

if platform.system() == "Windows":
    MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
    TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
else:
    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"

TICKER_CATEGORIES = {
    "BTC": "BTC",
    "ETH": "ETH",
    "XAU": "GOLD",
    "XAG": "SILVER",
    "MSTR": "BTC",
    "PLTR": "TECHNOLOGY",
    "NVDA": "TECHNOLOGY",
    "AMD": "TECHNOLOGY",
    "BABA": "TECHNOLOGY",
    "GOOGL": "TECHNOLOGY",
    "AMZN": "TECHNOLOGY",
    "AAPL": "TECHNOLOGY",
    "AVGO": "TECHNOLOGY",
    "GRRR": "TECHNOLOGY",
    "IONQ": "TECHNOLOGY",
    "MRVL": "TECHNOLOGY",
    "META": "TECHNOLOGY",
    "MU": "TECHNOLOGY",
    "PONY": "TECHNOLOGY",
    "RXRX": "TECHNOLOGY",
    "SOUN": "TECHNOLOGY",
    "SMCI": "TECHNOLOGY",
    "TSM": "TECHNOLOGY",
    "TTWO": "TECHNOLOGY",
    "TEM": "TECHNOLOGY",
    "UPST": "TECHNOLOGY",
    "VERI": "TECHNOLOGY",
    "VRT": "TECHNOLOGY",
    "QQQ": "TECHNOLOGY",
    "LMT": "TECHNOLOGY",
}

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"


def _is_crypto(ticker):
    return ticker.upper().replace("-USD", "") in CRYPTO_TICKERS


def _fetch_crypto_headlines(ticker="BTC", limit=20):
    category = TICKER_CATEGORIES.get(ticker.upper(), ticker.upper())
    url = f"{CRYPTOCOMPARE_URL}&categories={category}"
    r = fetch_with_retry(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    if r is None:
        return []
    data = r.json()
    raw = data.get("Data", [])
    articles = list(raw)[:limit] if isinstance(raw, list) else []
    return [
        {
            "title": a["title"],
            "source": a.get("source", "unknown"),
            "published": datetime.fromtimestamp(
                a["published_on"], tz=timezone.utc
            ).isoformat(),
        }
        for a in articles
    ]


def _fetch_yahoo_headlines(ticker, limit=10):
    import yfinance as yf

    stock = yf.Ticker(ticker)
    news = stock.news or []
    articles = []
    for item in news[:limit]:
        content = item.get("content", item)
        title = content.get("title", "")
        if not title:
            continue
        pub = content.get("pubDate") or content.get("displayTime", "")
        provider = content.get("provider", {})
        source = (
            provider.get("displayName", "Yahoo Finance")
            if isinstance(provider, dict)
            else "Yahoo Finance"
        )
        articles.append(
            {
                "title": title,
                "source": source,
                "published": pub or datetime.now(timezone.utc).isoformat(),
            }
        )
    return articles


def _fetch_newsapi_headlines(ticker, api_key, limit=10):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&language=en&sortBy=publishedAt&pageSize={limit}"
    )
    r = fetch_with_retry(
        url,
        headers={"User-Agent": "Mozilla/5.0", "X-Api-Key": api_key},
        timeout=15,
    )
    if r is None:
        return []
    data = r.json()
    articles = data.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", "unknown"),
            "published": a.get("publishedAt", datetime.now(timezone.utc).isoformat()),
        }
        for a in articles
        if a.get("title")
    ]


def _fetch_stock_headlines(ticker, newsapi_key=None, limit=20):
    articles = []
    try:
        articles.extend(_fetch_yahoo_headlines(ticker, limit=limit))
    except Exception as e:
        print(f"    [Yahoo News] error: {e}")
    if newsapi_key:
        try:
            remaining = max(0, limit - len(articles))
            if remaining > 0:
                articles.extend(
                    _fetch_newsapi_headlines(ticker, newsapi_key, limit=remaining)
                )
        except Exception as e:
            print(f"    [NewsAPI] error: {e}")
    return articles[:limit]


def _run_model(script, texts):
    proc = subprocess.run(
        [MODELS_PYTHON, script],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Model failed: {proc.stderr}")
    return json.loads(proc.stdout)


def _aggregate_sentiments(sentiments):
    pos_sum = sum(s["scores"]["positive"] for s in sentiments)
    neg_sum = sum(s["scores"]["negative"] for s in sentiments)
    neu_sum = sum(s["scores"]["neutral"] for s in sentiments)
    n = len(sentiments)
    avg = {"positive": pos_sum / n, "negative": neg_sum / n, "neutral": neu_sum / n}
    overall = max(avg, key=avg.get)
    return overall, avg


def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None) -> dict:
    short = ticker.upper().replace("-USD", "")
    is_crypto = _is_crypto(short)

    if is_crypto:
        articles = _fetch_crypto_headlines(short)
        model_script = CRYPTOBERT_SCRIPT
        model_name = "CryptoBERT"
    else:
        articles = _fetch_stock_headlines(short, newsapi_key=newsapi_key)
        model_script = TRADING_HERO_SCRIPT
        model_name = "Trading-Hero-LLM"

    social = social_posts or []
    all_articles = articles + social
    sources = {
        "news": len(articles),
        "reddit": sum(1 for p in social if "reddit" in p.get("source", "")),
    }

    if not all_articles:
        return {
            "overall_sentiment": "unknown",
            "confidence": 0.0,
            "num_articles": 0,
            "model": model_name,
            "sources": sources,
            "details": [],
        }

    titles = [a["title"] for a in all_articles]
    sentiments = _run_model(model_script, titles)

    overall, avg = _aggregate_sentiments(sentiments)

    details = []
    for article, sent in zip(all_articles, sentiments):
        details.append(
            {
                "title": article["title"],
                "source": article["source"],
                "published": article["published"],
                "sentiment": sent.get("sentiment") or sent.get("label", "unknown"),
                "confidence": sent["confidence"],
                "scores": sent["scores"],
            }
        )

    return {
        "overall_sentiment": overall,
        "confidence": round(avg[overall], 4),
        "num_articles": len(sentiments),
        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
        "model": model_name,
        "sources": sources,
        "details": details,
    }


def get_crypto_sentiment(ticker="BTC") -> dict:
    return get_sentiment(ticker)


if __name__ == "__main__":
    for ticker in ["BTC", "ETH"]:
        print(f"\n{'='*60}")
        print(f"  Sentiment for {ticker}")
        print(f"{'='*60}")
        result = get_sentiment(ticker)
        print(
            f"Overall: {result['overall_sentiment']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        print(f"Model: {result['model']}")
        print(f"Articles analyzed: {result['num_articles']}")
        if result.get("avg_scores"):
            s = result["avg_scores"]
            print(
                f"Avg scores: pos={s['positive']:.3f} neg={s['negative']:.3f} neu={s['neutral']:.3f}"
            )
        print("\nTop headlines:")
        for d in result["details"][:5]:
            emoji = {"positive": "+", "negative": "-", "neutral": "~"}.get(
                d["sentiment"], "?"
            )
            print(
                f"  [{emoji}] {d['sentiment']:>8} ({d['confidence']:.2%}) {d['title']}"
            )
