"""Signal generation engine — 30-signal voting system with weighted consensus."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import numpy as np

from portfolio.shared_state import _cached, FEAR_GREED_TTL, SENTIMENT_TTL, MINISTRAL_TTL, ML_SIGNAL_TTL, FUNDING_RATE_TTL, VOLUME_TTL
from portfolio.indicators import detect_regime
from portfolio.tickers import CRYPTO_SYMBOLS, STOCK_SYMBOLS, METALS_SYMBOLS
from portfolio.signal_registry import get_enhanced_signals, load_signal_func
from portfolio.signal_utils import true_range

logger = logging.getLogger("portfolio.signal_engine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- Signal (full 30-signal for "Now" timeframe) ---

MIN_VOTERS_CRYPTO = 3  # crypto has 27 signals (8 core + 19 enhanced; custom_lora, ml, funding disabled) — need 3
MIN_VOTERS_STOCK = 3  # stocks have 25 signals (7 core + 18 enhanced) — need 3 active voters

# Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
_prev_sentiment = {}  # in-memory cache; seeded from sentiment_state.json on first call
_prev_sentiment_loaded = False

_SENTIMENT_STATE_FILE = DATA_DIR / "sentiment_state.json"


def _load_prev_sentiments():
    global _prev_sentiment, _prev_sentiment_loaded
    if _prev_sentiment_loaded:
        return
    try:
        # Primary: own state file (no race with trigger.py)
        if _SENTIMENT_STATE_FILE.exists():
            data = json.loads(_SENTIMENT_STATE_FILE.read_text(encoding="utf-8"))
            _prev_sentiment = data.get("prev_sentiment", {})
        else:
            # Migration: read from trigger_state.json if sentiment_state.json doesn't exist yet
            ts_file = DATA_DIR / "trigger_state.json"
            if ts_file.exists():
                ts = json.loads(ts_file.read_text(encoding="utf-8"))
                _prev_sentiment = ts.get("prev_sentiment", {})
    except Exception:
        logger.debug("Failed to load prev sentiments", exc_info=True)
    _prev_sentiment_loaded = True


def _get_prev_sentiment(ticker):
    _load_prev_sentiments()
    return _prev_sentiment.get(ticker)


def _set_prev_sentiment(ticker, direction):
    _load_prev_sentiments()
    _prev_sentiment[ticker] = direction
    # Persist to own state file (avoids racing with trigger.py on trigger_state.json)
    try:
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(_SENTIMENT_STATE_FILE, {"prev_sentiment": _prev_sentiment})
    except Exception:
        logger.debug("Failed to persist sentiment", exc_info=True)


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

    Signals with accuracy below 50% (with >=20 samples) are **inverted**: a 30%
    accurate BUY signal becomes a 70% accurate SELL signal. This turns consistently
    wrong signals into useful contrarian indicators.
    """
    buy_weight = 0.0
    sell_weight = 0.0
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        # Accuracy weight — invert signals below 50%
        stats = accuracy_data.get(signal_name, {})
        acc = stats.get("accuracy", 0.5)
        samples = stats.get("total", 0)
        if samples < 20:
            weight = 0.5
            invert = False
        else:
            invert = acc < 0.5
            weight = (1.0 - acc) if invert else acc
        # Regime adjustment
        weight *= regime_mults.get(signal_name, 1.0)
        # Activation frequency normalization (rarity * bias correction)
        act_data = activation_rates.get(signal_name, {})
        norm_weight = act_data.get("normalized_weight", 1.0)
        weight *= norm_weight
        effective_vote = vote
        if invert:
            effective_vote = "SELL" if vote == "BUY" else "BUY"
        if effective_vote == "BUY":
            buy_weight += weight
        elif effective_vote == "SELL":
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


def _compute_adx(df, period=14):
    """Compute ADX (Average Directional Index) from a DataFrame with high/low/close.

    Returns the latest ADX value, or None if insufficient data.
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < period * 2:
        return None
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = true_range(high, low, close)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        alpha = 1.0 / period
        atr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        val = adx.iloc[-1]
        return float(val) if pd.notna(val) and np.isfinite(val) else None
    except Exception:
        return None


def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
    """Apply a 4-stage multiplicative confidence penalty cascade.

    Stages:
      1. Regime penalty — dampens confidence in choppy/volatile markets
      2. Volume/ADX gate — rejects low-conviction signals
      3. Trap detection — catches bull/bear traps (price vs volume divergence)
      4. Dynamic MIN_VOTERS — raises the bar in uncertain markets

    Returns (action, conf, penalty_log) where penalty_log is a list of applied penalties.
    """
    cfg = (config or {}).get("confidence_penalties", {})
    if cfg.get("enabled") is False:
        return action, conf, []

    penalty_log = []

    # --- Stage 1: Regime penalties ---
    if regime == "ranging":
        conf *= 0.75
        penalty_log.append({"stage": "regime", "regime": "ranging", "mult": 0.75})
    elif regime == "high-vol":
        conf *= 0.80
        penalty_log.append({"stage": "regime", "regime": "high-vol", "mult": 0.80})
    elif regime in ("trending-up", "trending-down"):
        # Bonus only if action aligns with trend direction
        trending_buy = regime == "trending-up" and action == "BUY"
        trending_sell = regime == "trending-down" and action == "SELL"
        if trending_buy or trending_sell:
            conf *= 1.10
            penalty_log.append({"stage": "regime", "regime": regime, "aligned": True, "mult": 1.10})

    # --- Stage 2: Volume/ADX gate ---
    volume_ratio = extra_info.get("volume_ratio")
    adx = _compute_adx(df)
    extra_info["_adx"] = adx

    if volume_ratio is not None and action != "HOLD":
        if volume_ratio < 0.5:
            # Very low volume — force HOLD
            penalty_log.append({"stage": "volume_gate", "rvol": volume_ratio, "effect": "force_hold"})
            action = "HOLD"
            conf = 0.0
        elif volume_ratio < 0.8 and (adx is not None and adx < 20) and conf < 0.65:
            # Low volume + weak trend + marginal confidence — force HOLD
            penalty_log.append({
                "stage": "volume_adx_gate", "rvol": volume_ratio,
                "adx": round(adx, 1), "conf": round(conf, 4), "effect": "force_hold",
            })
            action = "HOLD"
            conf = 0.0
        elif volume_ratio > 1.5:
            # High volume — slight confidence boost
            conf *= 1.15
            penalty_log.append({"stage": "volume_boost", "rvol": volume_ratio, "mult": 1.15})

    # --- Stage 3: Trap detection ---
    if action != "HOLD" and df is not None and isinstance(df, pd.DataFrame) and len(df) >= 5:
        try:
            recent_close = df["close"].iloc[-5:]
            recent_vol = df["volume"].iloc[-5:] if "volume" in df.columns else None
            price_up = recent_close.iloc[-1] > recent_close.iloc[0]
            price_down = recent_close.iloc[-1] < recent_close.iloc[0]

            if recent_vol is not None and len(recent_vol) >= 5:
                vol_declining = recent_vol.iloc[-1] < recent_vol.iloc[0] * 0.8

                if action == "BUY" and price_up and vol_declining:
                    conf *= 0.5
                    penalty_log.append({"stage": "trap", "type": "bull_trap", "mult": 0.5})
                elif action == "SELL" and price_down and vol_declining:
                    conf *= 0.5
                    penalty_log.append({"stage": "trap", "type": "bear_trap", "mult": 0.5})
        except Exception:
            pass

    # --- Stage 4: Dynamic MIN_VOTERS ---
    active_voters = extra_info.get("_voters", 0)
    buy_count = extra_info.get("_buy_count", 0)
    sell_count = extra_info.get("_sell_count", 0)

    if regime in ("trending-up", "trending-down"):
        dynamic_min = 3
    elif regime == "high-vol":
        dynamic_min = 4
    else:  # ranging or unknown
        dynamic_min = 5

    if action != "HOLD" and active_voters < dynamic_min:
        penalty_log.append({
            "stage": "dynamic_min_voters", "regime": regime,
            "required": dynamic_min, "actual": active_voters, "effect": "force_hold",
        })
        action = "HOLD"
        conf = 0.0

    # Clamp confidence to [0, 1]
    conf = max(0.0, min(1.0, conf))

    return action, conf, penalty_log


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

    # ML Classifier — disabled: 28.2% accuracy (1,027 samples, 1d horizon).
    # Worse than coin flip; actively harmful to consensus. Still tracked for
    # accuracy monitoring but never votes.
    votes["ml"] = "HOLD"

    # Funding Rate — disabled: 27.0% accuracy (512 samples, 1d horizon).
    # Contrarian logic consistently wrong in current regime. Still tracked for
    # accuracy monitoring but never votes.
    votes["funding"] = "HOLD"

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
    # Loaded from signal_registry — no hardcoded list needed here.
    _enhanced_entries = get_enhanced_signals()

    if df is not None and isinstance(df, pd.DataFrame) and len(df) >= 26:
        # Fetch macro context once for any signal that requires it
        macro_data = None
        has_macro_signals = any(e.get("requires_macro") for e in _enhanced_entries.values())
        if has_macro_signals:
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
                logger.warning("Macro context fetch failed", exc_info=True)

        # Build context data once for signals that need it
        context_data = {"ticker": ticker, "config": config or {}, "macro": macro_data}

        for sig_name, entry in _enhanced_entries.items():
            try:
                _sig_t0 = time.monotonic()
                compute_fn = load_signal_func(entry)
                if compute_fn is None:
                    votes[sig_name] = "HOLD"
                    continue
                if entry.get("requires_context"):
                    result = compute_fn(df, context=context_data)
                elif entry.get("requires_macro"):
                    result = compute_fn(df, macro=macro_data or None)
                else:
                    result = compute_fn(df)
                _sig_dt = time.monotonic() - _sig_t0
                if _sig_dt > 1.0:
                    logger.info("[SLOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
                if result and isinstance(result, dict):
                    extra_info[f"{sig_name}_action"] = result.get("action", "HOLD")
                    extra_info[f"{sig_name}_confidence"] = result.get("confidence", 0.0)
                    extra_info[f"{sig_name}_sub_signals"] = result.get("sub_signals", {})
                    votes[sig_name] = result.get("action", "HOLD")
                else:
                    votes[sig_name] = "HOLD"
            except Exception as e:
                logger.warning("Signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
    else:
        for sig_name in _enhanced_entries:
            votes[sig_name] = "HOLD"

    # Derive buy/sell counts from named votes
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")

    # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
    # Enhanced signals can strengthen/weaken a consensus but never create one alone.
    # ml and funding removed — disabled due to <30% accuracy.
    CORE_SIGNAL_NAMES = {
        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
        "volume", "ministral", "claude_fundamental",
    }
    core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
    core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
    core_active = core_buy + core_sell

    # Total applicable signals:
    # Crypto: 8 core (11 original - custom_lora, ml, funding disabled) + 19 enhanced = 27
    # Metals: 7 core + 18 enhanced = 25 (futures_flow only applies to crypto)
    # Stocks: 7 core + 18 enhanced = 25
    # (enhanced: 16 original + forecast + claude_fundamental + futures_flow[crypto only])
    is_crypto = ticker in CRYPTO_SYMBOLS
    is_metal = ticker in METALS_SYMBOLS
    if is_crypto:
        total_applicable = 27  # 8 core + 19 enhanced
    elif is_metal:
        total_applicable = 25  # 7 core + 18 enhanced
    else:
        total_applicable = 25  # 7 core + 18 enhanced

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
            load_cached_accuracy, signal_accuracy, signal_accuracy_recent,
            write_accuracy_cache, load_cached_activation_rates,
        )

        # Load all-time accuracy
        alltime = load_cached_accuracy("1d")
        if not alltime:
            alltime = signal_accuracy("1d")
            if alltime:
                write_accuracy_cache("1d", alltime)

        # Load recent accuracy (7d window) — more responsive to regime changes
        recent = load_cached_accuracy("1d_recent")
        if not recent:
            recent = signal_accuracy_recent("1d", days=7)
            if recent:
                write_accuracy_cache("1d_recent", recent)

        # Blend: 70% recent + 30% all-time (prefer recent performance)
        if alltime and recent:
            accuracy_data = {}
            for sig_name in alltime:
                at = alltime.get(sig_name, {})
                rc = recent.get(sig_name, {})
                at_acc = at.get("accuracy", 0.5)
                rc_acc = rc.get("accuracy", 0.5)
                rc_samples = rc.get("total", 0)
                at_samples = at.get("total", 0)
                # Only blend if recent has enough data; otherwise use all-time
                if rc_samples >= 50:
                    blended = 0.7 * rc_acc + 0.3 * at_acc
                else:
                    blended = at_acc
                accuracy_data[sig_name] = {
                    "accuracy": blended,
                    "total": max(at_samples, rc_samples),
                    "correct": at.get("correct", 0),
                    "pct": round(blended * 100, 1),
                }
        elif alltime:
            accuracy_data = alltime
        elif recent:
            accuracy_data = recent

        activation_rates = load_cached_activation_rates()
    except Exception:
        logger.warning("Accuracy stats load failed", exc_info=True)
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

    # Apply confidence penalty cascade (regime, volume/ADX, trap, dynamic min_voters)
    action, conf, penalty_log = apply_confidence_penalties(
        action, conf, regime, ind, extra_info, ticker, df, config
    )
    if penalty_log:
        extra_info["_penalty_log"] = penalty_log

    return action, conf, extra_info
