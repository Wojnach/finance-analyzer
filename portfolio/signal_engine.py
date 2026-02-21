"""Signal generation engine — 25-signal voting system with weighted consensus."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio.shared_state import _cached, FEAR_GREED_TTL, SENTIMENT_TTL, MINISTRAL_TTL, ML_SIGNAL_TTL, FUNDING_RATE_TTL, VOLUME_TTL
from portfolio.indicators import detect_regime
from portfolio.tickers import CRYPTO_SYMBOLS, STOCK_SYMBOLS, METALS_SYMBOLS

logger = logging.getLogger("portfolio.signal_engine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_enhanced_signal_modules = {}  # cache for importlib.import_module results

# --- Signal (full 25-signal for "Now" timeframe) ---

MIN_VOTERS_CRYPTO = 3  # crypto has 24 signals (10 core + 14 enhanced, custom_lora removed) — need 3 active voters
MIN_VOTERS_STOCK = 3  # stocks have 21 signals (7 original + 14 enhanced) — need 3 active voters

# Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
_prev_sentiment = {}  # in-memory cache; seeded from trigger_state.json on first call
_prev_sentiment_loaded = False


def _load_prev_sentiments():
    global _prev_sentiment, _prev_sentiment_loaded
    if _prev_sentiment_loaded:
        return
    try:
        ts_file = DATA_DIR / "trigger_state.json"
        if ts_file.exists():
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
            _prev_sentiment = ts.get("prev_sentiment", {})
    except Exception:
        pass
    _prev_sentiment_loaded = True


def _get_prev_sentiment(ticker):
    _load_prev_sentiments()
    return _prev_sentiment.get(ticker)


def _set_prev_sentiment(ticker, direction):
    _load_prev_sentiments()
    _prev_sentiment[ticker] = direction
    # Persist to trigger_state.json alongside other trigger state
    try:
        ts_file = DATA_DIR / "trigger_state.json"
        ts = json.loads(ts_file.read_text(encoding="utf-8")) if ts_file.exists() else {}
        ts["prev_sentiment"] = _prev_sentiment
        import tempfile as _tmp, os as _os

        fd, tmp = _tmp.mkstemp(dir=ts_file.parent, suffix=".tmp")
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ts, f, indent=2, default=str)
        _os.replace(tmp, ts_file)
    except Exception:
        pass


REGIME_WEIGHTS = {
    "trending-up": {"ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7},
    "trending-down": {"ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7},
    "ranging": {"rsi": 1.5, "bb": 1.5, "ema": 0.5, "macd": 0.5},
    "high-vol": {"bb": 1.5, "volume": 1.3, "funding": 1.3, "ema": 0.5},
}


def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None):
    """Compute weighted consensus using accuracy, regime, and activation frequency.

    Weight per signal = accuracy_weight * regime_mult * normalized_weight
    where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
    Rare, balanced signals get more weight; noisy/biased signals get less.
    """
    buy_weight = 0.0
    sell_weight = 0.0
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        # Accuracy weight
        stats = accuracy_data.get(signal_name, {})
        acc = stats.get("accuracy", 0.5)
        samples = stats.get("total", 0)
        weight = acc if samples >= 20 else 0.5
        # Regime adjustment
        weight *= regime_mults.get(signal_name, 1.0)
        # Activation frequency normalization (rarity * bias correction)
        act_data = activation_rates.get(signal_name, {})
        norm_weight = act_data.get("normalized_weight", 1.0)
        weight *= norm_weight
        if vote == "BUY":
            buy_weight += weight
        elif vote == "SELL":
            sell_weight += weight
    total_weight = buy_weight + sell_weight
    if total_weight == 0:
        return "HOLD", 0.0
    buy_conf = buy_weight / total_weight
    sell_conf = sell_weight / total_weight
    if buy_conf > sell_conf and buy_conf >= 0.5:
        return "BUY", round(buy_conf, 4)
    if sell_conf > buy_conf and sell_conf >= 0.5:
        return "SELL", round(sell_conf, 4)
    return "HOLD", round(max(buy_conf, sell_conf), 4)


def _confluence_score(votes, indicators):
    active = {k: v for k, v in votes.items() if v != "HOLD"}
    if not active:
        return 0.0
    buy_count = sum(1 for v in active.values() if v == "BUY")
    sell_count = sum(1 for v in active.values() if v == "SELL")
    majority = max(buy_count, sell_count)
    score = majority / len(active)
    if indicators.get("volume_action") in ("BUY", "SELL"):
        vol_dir = indicators.get("volume_action")
        majority_dir = "BUY" if buy_count >= sell_count else "SELL"
        if vol_dir == majority_dir:
            score += 0.1
    return min(round(score, 4), 1.0)


def _time_of_day_factor():
    hour = datetime.now(timezone.utc).hour
    if 2 <= hour <= 6:
        return 0.8
    return 1.0


def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None):
    votes = {}
    extra_info = {}

    # RSI — only votes at extremes (adaptive thresholds from rolling percentiles)
    rsi_lower = ind.get("rsi_p20", 30)
    rsi_upper = ind.get("rsi_p80", 70)
    rsi_lower = max(rsi_lower, 15)
    rsi_upper = min(rsi_upper, 85)
    if ind["rsi"] < rsi_lower:
        votes["rsi"] = "BUY"
    elif ind["rsi"] > rsi_upper:
        votes["rsi"] = "SELL"
    else:
        votes["rsi"] = "HOLD"

    # MACD — only votes on crossover
    if ind["macd_hist"] > 0 and ind["macd_hist_prev"] <= 0:
        votes["macd"] = "BUY"
    elif ind["macd_hist"] < 0 and ind["macd_hist_prev"] >= 0:
        votes["macd"] = "SELL"
    else:
        votes["macd"] = "HOLD"

    # EMA trend — votes only when gap is meaningful (>0.5%)
    ema_gap_pct = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100 if ind["ema21"] != 0 else 0
    )
    if ema_gap_pct >= 0.5:
        votes["ema"] = "BUY" if ind["ema9"] > ind["ema21"] else "SELL"
    else:
        votes["ema"] = "HOLD"

    # Bollinger Bands — only votes at extremes
    if ind["price_vs_bb"] == "below_lower":
        votes["bb"] = "BUY"
    elif ind["price_vs_bb"] == "above_upper":
        votes["bb"] = "SELL"
    else:
        votes["bb"] = "HOLD"

    # --- Extended signals from tools (optional) ---

    # Fear & Greed Index (per-ticker: crypto->alternative.me, stocks->VIX)
    votes["fear_greed"] = "HOLD"
    try:
        from portfolio.fear_greed import get_fear_greed

        fg_key = f"fear_greed_{ticker}" if ticker else "fear_greed"
        fg = _cached(fg_key, FEAR_GREED_TTL, get_fear_greed, ticker)
        if fg:
            extra_info["fear_greed"] = fg["value"]
            extra_info["fear_greed_class"] = fg["classification"]
            if fg["value"] <= 20:
                votes["fear_greed"] = "BUY"
            elif fg["value"] >= 80:
                votes["fear_greed"] = "SELL"
    except ImportError:
        pass

    # Social media posts (Reddit) — fetched separately, merged into sentiment
    social_posts = []
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.social_sentiment import get_reddit_posts

            reddit = _cached(
                f"reddit_{short_ticker}",
                SENTIMENT_TTL,
                get_reddit_posts,
                short_ticker,
            )
            if reddit:
                social_posts.extend(reddit)
        except ImportError:
            pass

    # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM) — includes social posts
    # Hysteresis: flipping direction requires confidence > 0.55, same direction > 0.40
    votes["sentiment"] = "HOLD"
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.sentiment import get_sentiment

            newsapi_key = (config or {}).get("newsapi_key", "")
            sent = _cached(
                f"sentiment_{short_ticker}",
                SENTIMENT_TTL,
                get_sentiment,
                short_ticker,
                newsapi_key or None,
                social_posts or None,
            )
            if sent and sent.get("num_articles", 0) > 0:
                extra_info["sentiment"] = sent["overall_sentiment"]
                extra_info["sentiment_conf"] = sent["confidence"]
                extra_info["sentiment_model"] = sent.get("model", "unknown")
                if sent.get("sources"):
                    extra_info["sentiment_sources"] = sent["sources"]

                prev_sent_dir = _get_prev_sentiment(ticker)
                current_dir = sent["overall_sentiment"]
                if (
                    prev_sent_dir
                    and current_dir != prev_sent_dir
                    and current_dir != "neutral"
                ):
                    sent_threshold = 0.55
                else:
                    sent_threshold = 0.40

                if (
                    sent["overall_sentiment"] == "positive"
                    and sent["confidence"] > sent_threshold
                ):
                    votes["sentiment"] = "BUY"
                    _set_prev_sentiment(ticker, "positive")
                elif (
                    sent["overall_sentiment"] == "negative"
                    and sent["confidence"] > sent_threshold
                ):
                    votes["sentiment"] = "SELL"
                    _set_prev_sentiment(ticker, "negative")
        except ImportError:
            pass

    # ML Classifier (HistGradientBoosting on BTC/ETH 1h data)
    votes["ml"] = "HOLD"
    if ticker:
        try:
            from portfolio.ml_signal import get_ml_signal

            ml = _cached(f"ml_{ticker}", ML_SIGNAL_TTL, get_ml_signal, ticker)
            if ml:
                extra_info["ml_action"] = ml["action"]
                extra_info["ml_confidence"] = ml["confidence"]
                votes["ml"] = ml["action"]
        except ImportError:
            pass

    # Funding Rate (Binance perpetuals, crypto only — contrarian)
    votes["funding"] = "HOLD"
    if ticker:
        try:
            from portfolio.funding_rate import get_funding_rate

            fr = _cached(
                f"funding_{ticker}", FUNDING_RATE_TTL, get_funding_rate, ticker
            )
            if fr:
                extra_info["funding_rate"] = fr["rate_pct"]
                extra_info["funding_action"] = fr["action"]
                votes["funding"] = fr["action"]
        except ImportError:
            pass

    # Volume Confirmation (spike + price direction = vote)
    votes["volume"] = "HOLD"
    if ticker:
        try:
            from portfolio.macro_context import get_volume_signal

            vs = _cached(f"volume_{ticker}", VOLUME_TTL, get_volume_signal, ticker)
            if vs:
                extra_info["volume_ratio"] = vs["ratio"]
                extra_info["volume_action"] = vs["action"]
                votes["volume"] = vs["action"]
        except ImportError:
            pass

    # Ministral-8B LLM reasoning (original CryptoTrader-LM, crypto only)
    # custom_lora fully disabled: 20.9% accuracy, 97% SELL bias (worse than random).
    # Model no longer invoked. Shadow A/B testing data preserved in ab_test_log.jsonl.
    votes["ministral"] = "HOLD"
    if ticker and ticker in CRYPTO_SYMBOLS:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.ministral_signal import get_ministral_signal

            tf_summary = ""
            if timeframes:
                parts = []
                for label, entry in timeframes:
                    if (
                        isinstance(entry, dict)
                        and "action" in entry
                        and entry["action"]
                    ):
                        ti = entry.get("indicators", {})
                        parts.append(
                            f"{label}: {entry['action']} (RSI={ti.get('rsi', 0):.0f})"
                        )
                if parts:
                    tf_summary = " | ".join(parts)

            ema_gap = (
                abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100
                if ind["ema21"] != 0
                else 0
            )

            ctx = {
                "ticker": short_ticker,
                "price_usd": ind["close"],
                "rsi": round(ind["rsi"], 1),
                "macd_hist": round(ind["macd_hist"], 2),
                "ema_bullish": ind["ema9"] > ind["ema21"],
                "ema_gap_pct": round(ema_gap, 2),
                "bb_position": ind["price_vs_bb"],
                "fear_greed": extra_info.get("fear_greed", "N/A"),
                "fear_greed_class": extra_info.get("fear_greed_class", ""),
                "news_sentiment": extra_info.get("sentiment", "N/A"),
                "sentiment_confidence": extra_info.get("sentiment_conf", "N/A"),
                "volume_ratio": extra_info.get("volume_ratio", "N/A"),
                "funding_rate": extra_info.get("funding_action", "N/A"),
                "timeframe_summary": tf_summary,
                "headlines": "",
            }
            ms = _cached(
                f"ministral_{short_ticker}",
                MINISTRAL_TTL,
                get_ministral_signal,
                ctx,
            )
            if ms:
                orig = ms.get("original") or ms
                extra_info["ministral_action"] = orig["action"]
                extra_info["ministral_reasoning"] = orig.get("reasoning", "")
                votes["ministral"] = orig["action"]

                # custom_lora fully disabled — not even stored in extra.
                # Shadow A/B data preserved in data/ab_test_log.jsonl.
        except ImportError:
            pass

    # --- Enhanced signal modules (composite indicators computed from raw OHLCV) ---
    _enhanced_modules = [
        ("trend", "portfolio.signals.trend", "compute_trend_signal"),
        ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
        ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
        ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
        ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
        ("structure", "portfolio.signals.structure", "compute_structure_signal"),
        ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
        ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
        ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
        ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
        ("mean_reversion", "portfolio.signals.mean_reversion", "compute_mean_reversion_signal"),
        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
        ("momentum_factors", "portfolio.signals.momentum_factors", "compute_momentum_factors_signal"),
    ]
    # macro_regime is special — it takes an extra macro dict parameter
    _macro_regime_module = ("macro_regime", "portfolio.signals.macro_regime", "compute_macro_regime_signal")

    if df is not None and isinstance(df, pd.DataFrame) and len(df) >= 26:
        for sig_name, module_path, func_name in _enhanced_modules:
            try:
                import importlib
                if module_path not in _enhanced_signal_modules:
                    _enhanced_signal_modules[module_path] = importlib.import_module(module_path)
                mod = _enhanced_signal_modules[module_path]
                compute_fn = getattr(mod, func_name)
                result = compute_fn(df)
                if result and isinstance(result, dict):
                    extra_info[f"{sig_name}_action"] = result.get("action", "HOLD")
                    extra_info[f"{sig_name}_confidence"] = result.get("confidence", 0.0)
                    extra_info[f"{sig_name}_sub_signals"] = result.get("sub_signals", {})
                    votes[sig_name] = result.get("action", "HOLD")
                else:
                    votes[sig_name] = "HOLD"
            except Exception:
                votes[sig_name] = "HOLD"

        # macro_regime gets macro context from cache if available
        try:
            import importlib
            macro_data = None
            try:
                from portfolio.macro_context import get_dxy, get_fed_calendar, get_treasury
                macro_data = {}
                dxy = _cached("dxy", 3600, get_dxy)
                if dxy:
                    macro_data["dxy"] = dxy
                treasury = _cached("treasury", 3600, get_treasury)
                if treasury:
                    macro_data["treasury"] = treasury
                fed = get_fed_calendar()
                if fed:
                    macro_data["fed"] = fed
            except Exception:
                pass
            mr_name, mr_path, mr_func = _macro_regime_module
            if mr_path not in _enhanced_signal_modules:
                _enhanced_signal_modules[mr_path] = importlib.import_module(mr_path)
            mod = _enhanced_signal_modules[mr_path]
            compute_fn = getattr(mod, mr_func)
            result = compute_fn(df, macro=macro_data or None)
            if result and isinstance(result, dict):
                votes[mr_name] = result.get("action", "HOLD")
                extra_info[f"{mr_name}_action"] = result.get("action", "HOLD")
                extra_info[f"{mr_name}_confidence"] = result.get("confidence", 0.0)
                extra_info[f"{mr_name}_sub_signals"] = result.get("sub_signals", {})
            else:
                votes[mr_name] = "HOLD"
        except Exception:
            votes[_macro_regime_module[0]] = "HOLD"
    else:
        for sig_name, _, _ in _enhanced_modules:
            votes[sig_name] = "HOLD"
        votes[_macro_regime_module[0]] = "HOLD"

    # Derive buy/sell counts from named votes
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")

    # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
    # Enhanced signals can strengthen/weaken a consensus but never create one alone.
    CORE_SIGNAL_NAMES = {
        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
        "ml", "funding", "volume", "ministral",
    }
    core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
    core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
    core_active = core_buy + core_sell

    # Total applicable signals:
    # Crypto: 10 core (11 original - custom_lora removed) + 14 enhanced (all voting) = 24
    # Metals: 7 core + 14 enhanced = 21
    # Stocks: 7 core + 14 enhanced = 21
    is_crypto = ticker in CRYPTO_SYMBOLS
    is_metal = ticker in METALS_SYMBOLS
    if is_crypto:
        total_applicable = 24  # 10 core + 14 enhanced
    elif is_metal:
        total_applicable = 21  # 7 core + 14 enhanced
    else:
        total_applicable = 21  # 7 core + 14 enhanced

    active_voters = buy + sell
    if ticker in STOCK_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK
    elif ticker in METALS_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK  # metals use same threshold
    else:
        min_voters = MIN_VOTERS_CRYPTO

    # Core gate: if no core signal is active, force HOLD regardless of enhanced votes
    if core_active == 0:
        action = "HOLD"
        conf = 0.0
    elif active_voters < min_voters:
        action = "HOLD"
        conf = 0.0
    else:
        buy_conf = buy / active_voters
        sell_conf = sell / active_voters
        if buy_conf > sell_conf and buy_conf >= 0.5:
            action = "BUY"
            conf = buy_conf
        elif sell_conf > buy_conf and sell_conf >= 0.5:
            action = "SELL"
            conf = sell_conf
        else:
            action = "HOLD"
            conf = max(buy_conf, sell_conf)

    # Weighted consensus using accuracy data, regime, and activation frequency
    regime = detect_regime(ind, is_crypto=ticker in CRYPTO_SYMBOLS)
    accuracy_data = {}
    activation_rates = {}
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy, signal_accuracy, write_accuracy_cache,
            load_cached_activation_rates,
        )

        cached = load_cached_accuracy("1d")
        if cached:
            accuracy_data = cached
        else:
            accuracy_data = signal_accuracy("1d")
            if accuracy_data:
                write_accuracy_cache("1d", accuracy_data)
        activation_rates = load_cached_activation_rates()
    except Exception:
        pass
    weighted_action, weighted_conf = _weighted_consensus(
        votes, accuracy_data, regime, activation_rates
    )

    # Apply core gate AND MIN_VOTERS gate to weighted consensus too
    if core_active == 0 or active_voters < min_voters:
        weighted_action = "HOLD"
        weighted_conf = 0.0

    # Confluence score
    confluence = _confluence_score(votes, extra_info)

    # Time-of-day confidence adjustment
    tod_factor = _time_of_day_factor()
    conf *= tod_factor
    weighted_conf *= tod_factor

    # Store raw consensus in extra for debugging, then use weighted as primary
    extra_info["_raw_action"] = action
    extra_info["_raw_confidence"] = conf
    extra_info["_voters"] = active_voters
    extra_info["_total_applicable"] = total_applicable
    extra_info["_buy_count"] = buy
    extra_info["_sell_count"] = sell
    extra_info["_core_buy"] = core_buy
    extra_info["_core_sell"] = core_sell
    extra_info["_core_active"] = core_active
    extra_info["_votes"] = votes
    extra_info["_regime"] = regime
    extra_info["_weighted_action"] = weighted_action
    extra_info["_weighted_confidence"] = weighted_conf
    extra_info["_confluence_score"] = confluence

    # Primary action = weighted consensus (accounts for accuracy + bias penalties)
    action = weighted_action
    conf = weighted_conf
    return action, conf, extra_info
