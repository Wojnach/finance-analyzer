"""Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.

Runs multiple sentiment models in shadow mode and logs results for comparison.
The primary (voting) model is the legacy model; shadow models are logged to
data/sentiment_ab_log.jsonl for accuracy tracking. Once a shadow model proves
superior (>60% on 200+ samples), it can be promoted.

Models:
  - CryptoBERT: crypto headlines (legacy primary)
  - Trading-Hero-LLM: stock headlines (legacy primary)
  - FinGPT (Finance-Llama-8B / FinGPT-MT): GGUF via llama-cpp-python (shadow)
  - FinBERT: CPU fallback (shadow)

Phase 3B: Cumulative headline clustering — groups related headlines and scores
them as a batch for richer "drumbeat effect" detection.
"""

import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

import platform
from pathlib import Path

from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.sentiment")

CRYPTO_TICKERS = {"BTC", "ETH"}

if platform.system() == "Windows":
    MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
    TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
    FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
    FINGPT_PYTHON = r"Q:\models\.venv-llm\Scripts\python.exe"
    FINGPT_SCRIPT = r"Q:\models\fingpt_infer.py"
else:
    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"
    FINGPT_PYTHON = "/home/deck/models/.venv-llm/bin/python"
    FINGPT_SCRIPT = "/home/deck/models/fingpt_infer.py"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AB_LOG_FILE = DATA_DIR / "sentiment_ab_log.jsonl"

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
    "META": "TECHNOLOGY",
    "MU": "TECHNOLOGY",
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
        logger.debug("[Yahoo News] error: %s", e)
    if newsapi_key:
        try:
            remaining = max(0, limit - len(articles))
            if remaining > 0:
                articles.extend(
                    _fetch_newsapi_headlines(ticker, newsapi_key, limit=remaining)
                )
        except Exception as e:
            logger.debug("[NewsAPI] error: %s", e)
    return articles[:limit]


def _run_model(script, texts):
    """Run a sentiment model via subprocess (legacy pattern)."""
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


def _run_fingpt(texts, model_name=None, cumulative=False, ticker="unknown"):
    """Run FinGPT sentiment inference via llama-cpp-python subprocess.

    Uses FINGPT_PYTHON (the .venv-llm venv with llama-cpp-python CUDA).

    Args:
        texts: List of headline strings
        model_name: Which GGUF model to use (auto-detect if None)
        cumulative: If True, run cumulative analysis instead of per-headline
        ticker: Ticker for cumulative context

    Returns:
        List of sentiment result dicts (per-headline) or single dict (cumulative)
    """
    cmd = [FINGPT_PYTHON, FINGPT_SCRIPT]
    if model_name:
        cmd.extend(["--model", model_name])
    if cumulative:
        cmd.extend(["--cumulative", "--ticker", ticker])

    proc = subprocess.run(
        cmd,
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FinGPT failed: {proc.stderr[:200]}")
    return json.loads(proc.stdout)


def _run_finbert(texts):
    """Run FinBERT sentiment inference (CPU fallback)."""
    proc = subprocess.run(
        [MODELS_PYTHON, FINBERT_SCRIPT],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FinBERT failed: {proc.stderr[:200]}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Headline clustering (Phase 3B)
# ---------------------------------------------------------------------------

def _cluster_headlines(articles):
    """Group headlines by keyword overlap and time proximity.

    Clusters enable cumulative sentiment analysis — many mildly negative
    headlines about the same topic together signal stronger negativity
    than scoring each independently.

    Returns:
        List of clusters, each a list of article dicts
    """
    if not articles or len(articles) < 3:
        return [articles] if articles else []

    from portfolio.news_keywords import score_headline

    # Extract keywords per headline
    headline_keywords = []
    for a in articles:
        _, matched = score_headline(a.get("title", ""))
        # Also extract significant words (>4 chars, not stopwords)
        words = set()
        for word in a.get("title", "").lower().split():
            clean = word.strip(".,!?;:'\"()[]")
            if len(clean) > 4 and clean not in _STOPWORDS:
                words.add(clean)
        words.update(kw.lower() for kw in matched)
        headline_keywords.append(words)

    # Simple greedy clustering by keyword overlap
    clusters = []
    assigned = set()

    for i in range(len(articles)):
        if i in assigned:
            continue
        cluster = [articles[i]]
        assigned.add(i)
        kw_i = headline_keywords[i]

        for j in range(i + 1, len(articles)):
            if j in assigned:
                continue
            kw_j = headline_keywords[j]
            overlap = len(kw_i & kw_j)
            # Cluster if they share 2+ keywords or 1 matched keyword
            if overlap >= 2 or (overlap >= 1 and kw_i & kw_j & _SIGNIFICANT_KEYWORDS):
                cluster.append(articles[j])
                assigned.add(j)

        clusters.append(cluster)

    return clusters


_STOPWORDS = {
    "about", "after", "again", "being", "between", "could", "during",
    "every", "first", "their", "there", "these", "those", "under",
    "which", "while", "would", "other", "still", "where", "before",
    "should", "since", "until", "years", "might", "price", "stock",
    "market", "shares", "today", "could", "report", "quarter",
}

_SIGNIFICANT_KEYWORDS = {
    "tariff", "tariffs", "war", "crash", "sanctions", "hack", "recession",
    "inflation", "rate", "cut", "hike", "layoffs", "earnings", "fomc",
    "bitcoin", "ethereum", "crypto", "nvidia", "semiconductor",
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_sentiments(sentiments, headlines=None, dissemination_mult=1.0):
    """Aggregate sentiment scores, weighted by headline keywords and dissemination.

    When headlines are provided, each sentiment score is multiplied by the
    keyword weight from news_keywords.score_headline(). High-impact keywords
    (tariff, war, crash) get 2-3x amplification.

    dissemination_mult applies the FinGPT dissemination-aware multiplier
    to all weights (wider news spread = stronger signal).
    """
    if headlines and len(headlines) == len(sentiments):
        from portfolio.news_keywords import score_headline
        weights = []
        for h in headlines:
            title = h.get("title", "") if isinstance(h, dict) else str(h)
            w, _ = score_headline(title)
            weights.append(w * dissemination_mult)
    else:
        weights = [dissemination_mult] * len(sentiments)

    pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
    neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
    neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
    total_w = sum(weights)
    if total_w == 0:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    avg = {"positive": pos_sum / total_w, "negative": neg_sum / total_w, "neutral": neu_sum / total_w}
    overall = max(avg, key=avg.get)
    return overall, avg


def _log_ab_result(ticker, primary_result, shadow_results):
    """Log A/B test results to sentiment_ab_log.jsonl for accuracy comparison."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "primary": {
                "model": primary_result.get("model", "unknown"),
                "sentiment": primary_result.get("overall_sentiment", "unknown"),
                "confidence": primary_result.get("confidence", 0.0),
            },
            "shadow": shadow_results,
        }
        with open(AB_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to log A/B result", exc_info=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None) -> dict:
    """Get sentiment for a ticker using primary model + shadow A/B models.

    Primary model (votes): CryptoBERT (crypto) or Trading-Hero-LLM (stocks)
    Shadow models (logged only): FinGPT, FinBERT

    Returns the primary model's result. Shadow results are logged to
    data/sentiment_ab_log.jsonl for accuracy tracking.
    """
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

    # Compute dissemination score for weight amplification
    diss_mult = 1.0
    try:
        from portfolio.news_keywords import dissemination_score
        diss_mult = dissemination_score(all_articles)
    except Exception:
        pass

    # --- Primary model (votes in consensus) ---
    sentiments = _run_model(model_script, titles)
    overall, avg = _aggregate_sentiments(sentiments, headlines=all_articles,
                                         dissemination_mult=diss_mult)

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

    primary_result = {
        "overall_sentiment": overall,
        "confidence": round(avg[overall], 4),
        "num_articles": len(sentiments),
        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
        "model": model_name,
        "sources": sources,
        "details": details,
        "dissemination_score": diss_mult,
    }

    # --- Shadow models (A/B testing — logged only, don't affect consensus) ---
    shadow_results = []

    # Shadow: FinGPT (per-headline)
    try:
        fingpt_results = _run_fingpt(titles)
        if fingpt_results:
            fg_overall, fg_avg = _aggregate_sentiments(
                fingpt_results, headlines=all_articles, dissemination_mult=diss_mult
            )
            shadow_results.append({
                "model": fingpt_results[0].get("model", "fingpt") if fingpt_results else "fingpt",
                "sentiment": fg_overall,
                "confidence": round(fg_avg[fg_overall], 4),
                "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
            })
    except Exception as e:
        logger.debug("FinGPT shadow failed: %s", e)

    # Shadow: FinGPT cumulative (clustered headlines)
    try:
        clusters = _cluster_headlines(all_articles)
        for cluster in clusters:
            if len(cluster) >= 3:
                cluster_titles = [a["title"] for a in cluster]
                cum_result = _run_fingpt(
                    cluster_titles, cumulative=True, ticker=short
                )
                if cum_result and isinstance(cum_result, dict):
                    shadow_results.append({
                        "model": "fingpt:cumulative",
                        "sentiment": cum_result.get("sentiment", "neutral"),
                        "confidence": cum_result.get("confidence", 0.0),
                        "headline_count": len(cluster),
                    })
    except Exception as e:
        logger.debug("FinGPT cumulative shadow failed: %s", e)

    # Shadow: FinBERT (CPU, fast)
    try:
        finbert_results = _run_finbert(titles)
        if finbert_results:
            fb_overall, fb_avg = _aggregate_sentiments(
                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
            )
            shadow_results.append({
                "model": "FinBERT",
                "sentiment": fb_overall,
                "confidence": round(fb_avg[fb_overall], 4),
                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
            })
    except Exception as e:
        logger.debug("FinBERT shadow failed: %s", e)

    # Log A/B comparison
    if shadow_results:
        _log_ab_result(short, primary_result, shadow_results)

    return primary_result


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
        print(f"Dissemination score: {result.get('dissemination_score', 1.0):.2f}")
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
