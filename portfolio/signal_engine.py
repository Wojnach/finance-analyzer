"""Signal generation engine — 30-signal voting system with weighted consensus."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.indicators import detect_regime
from portfolio.shared_state import FEAR_GREED_TTL, MINISTRAL_TTL, SENTIMENT_TTL, VOLUME_TTL, _cached
from portfolio.signal_registry import get_enhanced_signals, load_signal_func
from portfolio.signal_utils import true_range
from portfolio.tickers import CRYPTO_SYMBOLS, DISABLED_SIGNALS, GPU_SIGNALS, METALS_SYMBOLS, SIGNAL_NAMES, STOCK_SYMBOLS

logger = logging.getLogger("portfolio.signal_engine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOCAL_MODEL_ACCURACY_TTL = 1800

# ADX computation cache — keyed by id(df) so each DataFrame is computed at most once.
# Naturally expires when DataFrames are garbage-collected between cycles.
_adx_cache: dict[int, float | None] = {}
_adx_lock = threading.Lock()  # BUG-86: protect concurrent access from ThreadPoolExecutor
_ADX_CACHE_MAX = 200  # prevent unbounded growth
_LOCAL_MODEL_HOLD_THRESHOLD = 0.55
_LOCAL_MODEL_MIN_SAMPLES = 30
_LOCAL_MODEL_LOOKBACK_DAYS = 30

# Accuracy gate: signals with blended accuracy below this threshold are
# force-HOLD (treated like DISABLED_SIGNALS but dynamically). A signal at
# 44% is noise, not a reliable contrarian indicator — inverting it just
# produces different noise with whiplash as accuracy oscillates around 50%.
ACCURACY_GATE_THRESHOLD = 0.45
ACCURACY_GATE_MIN_SAMPLES = 30  # need enough data before gating

# --- Signal (full 30-signal for "Now" timeframe) ---

MIN_VOTERS_CRYPTO = 3  # crypto has 27 signals (8 core + 19 enhanced; custom_lora, ml, funding disabled) — need 3
MIN_VOTERS_STOCK = 3  # stocks have 25 signals (7 core + 18 enhanced) — need 3 active voters

# Core signals that must have at least 1 active voter for non-HOLD consensus.
# Enhanced signals can strengthen/weaken but never create consensus alone.
CORE_SIGNAL_NAMES = frozenset({
    "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
    "volume", "ministral", "qwen3", "claude_fundamental",
})

# Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
_prev_sentiment = {}  # in-memory cache; seeded from sentiment_state.json on first call
_prev_sentiment_loaded = False
_sentiment_lock = threading.Lock()  # BUG-85: protect concurrent access from ThreadPoolExecutor
_sentiment_dirty = False  # Track whether in-memory state diverged from disk

_SENTIMENT_STATE_FILE = DATA_DIR / "sentiment_state.json"


def _load_prev_sentiments():
    global _prev_sentiment, _prev_sentiment_loaded
    with _sentiment_lock:
        if _prev_sentiment_loaded:
            return
        try:
            from portfolio.file_utils import load_json as _load_json
            data = _load_json(str(_SENTIMENT_STATE_FILE), default=None)
            if data and isinstance(data, dict):
                _prev_sentiment = data.get("prev_sentiment", {})
            # Prune entries for removed tickers
            from portfolio.tickers import ALL_TICKERS
            removed = [k for k in _prev_sentiment if k not in ALL_TICKERS]
            for k in removed:
                del _prev_sentiment[k]
        except Exception:
            logger.warning("Failed to load prev sentiments", exc_info=True)
        _prev_sentiment_loaded = True


def _get_prev_sentiment(ticker):
    _load_prev_sentiments()
    with _sentiment_lock:
        return _prev_sentiment.get(ticker)


def _set_prev_sentiment(ticker, direction):
    """Set sentiment direction for a ticker (thread-safe, batched disk write)."""
    global _sentiment_dirty
    _load_prev_sentiments()
    with _sentiment_lock:
        _prev_sentiment[ticker] = direction
        _sentiment_dirty = True


def flush_sentiment_state():
    """Persist sentiment state to disk. Call once per cycle, not per-ticker.

    BUG-85 fix: batching prevents concurrent per-ticker writes that clobber each other.
    BUG-101 fix: dirty flag cleared only AFTER successful write, so a failed write
    will be retried on the next cycle instead of silently losing state.
    """
    global _sentiment_dirty
    with _sentiment_lock:
        if not _sentiment_dirty:
            return
        snapshot = dict(_prev_sentiment)
    # Write outside the lock to avoid holding it during I/O
    try:
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(_SENTIMENT_STATE_FILE, {"prev_sentiment": snapshot})
        # BUG-101: Only clear dirty flag after successful write
        with _sentiment_lock:
            _sentiment_dirty = False
    except Exception:
        # Dirty flag remains True — next cycle will retry the write
        logger.warning("Failed to persist sentiment state (will retry next cycle)", exc_info=True)


REGIME_WEIGHTS = {
    "trending-up": {"ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7},
    "trending-down": {"ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7},
    "ranging": {"rsi": 1.5, "bb": 1.5, "ema": 0.5, "macd": 0.5},
    "high-vol": {"bb": 1.5, "volume": 1.3, "ema": 0.5},
}


# Signals that only apply to specific asset classes
_CRYPTO_ONLY_SIGNALS = {"futures_flow", "funding"}
_CORE_SIGNAL_SET = {"rsi", "macd", "ema", "bb", "fear_greed", "sentiment", "ministral", "qwen3", "ml", "funding", "volume", "claude_fundamental"}


def _compute_applicable_count(ticker: str, skip_gpu: bool = False) -> int:
    """Compute total applicable signals for a ticker dynamically.

    Accounts for disabled signals, per-asset-class restrictions,
    and GPU signals skipped outside market hours.
    """
    is_crypto = ticker in CRYPTO_SYMBOLS
    count = 0
    for sig in SIGNAL_NAMES:
        if sig in DISABLED_SIGNALS:
            continue
        # futures_flow only applies to crypto
        if sig in _CRYPTO_ONLY_SIGNALS and not is_crypto:
            continue
        # ministral (CryptoTrader-LM) only runs for crypto
        if sig == "ministral" and not is_crypto:
            continue
        # GPU signals skipped for stocks outside market hours
        if skip_gpu and sig in GPU_SIGNALS:
            continue
        count += 1
    return count


_VALID_ACTIONS = frozenset({"BUY", "SELL", "HOLD"})


def _validate_signal_result(result, sig_name=None, max_confidence=1.0):
    """Normalize and validate a signal's return dict.

    Ensures action is a valid string, confidence is a finite float in [0, 1],
    and sub_signals is a dict. Returns a clean dict, always.
    """
    if not result or not isinstance(result, dict):
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    action = result.get("action")
    if action not in _VALID_ACTIONS:
        if sig_name:
            logger.warning("Signal %s returned invalid action=%r, defaulting to HOLD", sig_name, action)
        action = "HOLD"

    conf = result.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    if not np.isfinite(conf):
        if sig_name:
            logger.warning("Signal %s returned non-finite confidence=%r, defaulting to 0.0", sig_name, conf)
        conf = 0.0
    conf = max(0.0, min(max_confidence, conf))

    sub_signals = result.get("sub_signals")
    if not isinstance(sub_signals, dict):
        sub_signals = {}

    return {
        "action": action,
        "confidence": conf,
        "sub_signals": sub_signals,
        "indicators": result.get("indicators") or {},
    }


def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
                        accuracy_gate=None):
    """Compute weighted consensus using accuracy, regime, and activation frequency.

    Weight per signal = accuracy_weight * regime_mult * normalized_weight
    where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
    Rare, balanced signals get more weight; noisy/biased signals get less.

    Signals below the accuracy gate (with sufficient samples) are force-skipped —
    they are noise, not useful contrarian indicators.
    """
    gate = accuracy_gate if accuracy_gate is not None else ACCURACY_GATE_THRESHOLD
    buy_weight = 0.0
    sell_weight = 0.0
    gated_signals = []
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        stats = accuracy_data.get(signal_name, {})
        acc = stats.get("accuracy", 0.5)
        samples = stats.get("total", 0)
        # Accuracy gate: skip signals that are below threshold with enough data
        if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < gate:
            gated_signals.append(signal_name)
            continue
        # Weight = accuracy (or 0.5 default for new signals with insufficient data)
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
    if gated_signals:
        logger.debug("Accuracy-gated signals (<%s%%): %s", ACCURACY_GATE_THRESHOLD * 100, gated_signals)
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
    hour = datetime.now(UTC).hour
    if 2 <= hour <= 6:
        return 0.8
    return 1.0


def _load_local_model_accuracy(signal_name, horizon="1d", days=None, cache_ttl=None):
    """Load per-ticker accuracy for a local model signal."""
    lookback_days = days if days is not None else _LOCAL_MODEL_LOOKBACK_DAYS
    ttl = cache_ttl or _LOCAL_MODEL_ACCURACY_TTL
    cache_key = f"local_model_accuracy_{signal_name}_{horizon}_{lookback_days}"

    def _fetch():
        try:
            from portfolio.accuracy_stats import accuracy_by_signal_ticker

            return accuracy_by_signal_ticker(signal_name, horizon=horizon, days=lookback_days)
        except Exception:
            logger.warning("Failed to load %s accuracy", signal_name, exc_info=True)
            return {}

    return _cached(cache_key, ttl, _fetch)


def _gate_local_model_vote(signal_name, vote, ticker, config=None):
    """Apply accuracy-based abstention to local model votes."""
    info = {
        "gating": "raw",
        "accuracy": None,
        "samples": 0,
    }
    if vote == "HOLD" or not ticker:
        return vote, info

    cfg = ((config or {}).get("local_models", {}) or {}).get(signal_name, {})
    hold_threshold = cfg.get("hold_threshold", _LOCAL_MODEL_HOLD_THRESHOLD)
    min_samples = cfg.get("min_samples", _LOCAL_MODEL_MIN_SAMPLES)
    days = cfg.get("accuracy_days", _LOCAL_MODEL_LOOKBACK_DAYS)
    cache_ttl = cfg.get("accuracy_cache_ttl", _LOCAL_MODEL_ACCURACY_TTL)

    accuracy_data = _load_local_model_accuracy(
        signal_name, horizon=cfg.get("horizon", "1d"), days=days, cache_ttl=cache_ttl
    )
    ticker_stats = (accuracy_data or {}).get(ticker)
    if not ticker_stats or ticker_stats.get("samples", 0) < min_samples:
        info["gating"] = "insufficient_data"
        if ticker_stats:
            info["accuracy"] = round(ticker_stats.get("accuracy", 0.0), 3)
            info["samples"] = ticker_stats.get("samples", 0)
        return vote, info

    accuracy = float(ticker_stats.get("accuracy", 0.0))
    samples = int(ticker_stats.get("samples", 0))
    info["accuracy"] = round(accuracy, 3)
    info["samples"] = samples
    if accuracy < hold_threshold:
        info["gating"] = "held"
        return "HOLD", info

    return vote, info


def _compute_adx(df, period=14):
    """Compute ADX (Average Directional Index) from a DataFrame with high/low/close.

    Returns the latest ADX value, or None if insufficient data.
    Cached per DataFrame identity to avoid recomputation within a cycle.
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < period * 2:
        return None

    df_id = id(df)
    with _adx_lock:
        if df_id in _adx_cache:
            return _adx_cache[df_id]

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
        # Use clip(lower=1e-10) instead of replace(0, np.nan) to avoid NaN propagation
        atr_clipped = atr_smooth.clip(lower=1e-10)
        plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_clipped
        minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_clipped

        di_sum = (plus_di + minus_di).clip(lower=1e-10)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        val = adx.iloc[-1]
        result = float(val) if pd.notna(val) and np.isfinite(val) else None
        # BUG-86: Thread-safe cache write with eviction
        with _adx_lock:
            if len(_adx_cache) >= _ADX_CACHE_MAX:
                _adx_cache.clear()
            _adx_cache[df_id] = result
        return result
    except Exception:
        logger.warning("ADX computation failed", exc_info=True)
        with _adx_lock:
            _adx_cache[df_id] = None
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
    # BUG-90: Clamp after Stage 1 so inflated confidence doesn't bypass Stage 2 gates
    conf = min(1.0, conf)

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
    # BUG-90: Clamp after Stage 2
    conf = min(1.0, conf)

    # --- Stage 3: Trap detection ---
    # NOTE: df must be the "Now" timeframe (15m candles, 100 bars ≈ 25h).
    # Last 5 bars = 75 minutes — appropriate for intraday trap detection.
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
            logger.warning("Trap detection failed for %s", ticker, exc_info=True)
    # BUG-90: Clamp after Stage 3
    conf = min(1.0, conf)

    # --- Stage 4: Dynamic MIN_VOTERS ---
    active_voters = extra_info.get("_voters", 0)

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

    # Check if GPU-intensive signals should be skipped (stocks outside market hours)
    from portfolio.market_timing import should_skip_gpu
    skip_gpu = should_skip_gpu(ticker, config=config) if ticker else False
    if skip_gpu:
        extra_info["_gpu_signals_skipped"] = True

    # Compute regime early so F&G gating and other sections can use it
    regime = detect_regime(ind, is_crypto=ticker in CRYPTO_SYMBOLS)

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
    # Gated: F&G is contrarian (buy fear, sell greed) which fights trends.
    # Only allow F&G to vote in ranging/high-vol regimes where mean reversion works.
    votes["fear_greed"] = "HOLD"
    try:
        from portfolio.fear_greed import get_fear_greed

        fg_key = f"fear_greed_{ticker}" if ticker else "fear_greed"
        fg = _cached(fg_key, FEAR_GREED_TTL, get_fear_greed, ticker)
        if fg:
            extra_info["fear_greed"] = fg["value"]
            extra_info["fear_greed_class"] = fg["classification"]
            # Gate: suppress F&G votes in trending regimes
            if regime in ("trending-up", "trending-down"):
                extra_info["fear_greed_gated"] = regime
                votes["fear_greed"] = "HOLD"
            elif fg["value"] <= 20:
                votes["fear_greed"] = "BUY"
            elif fg["value"] >= 80:
                votes["fear_greed"] = "SELL"
    except ImportError:
        logger.debug("Optional module %s not available", "fear_greed")

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
            logger.debug("Optional module %s not available", "social_sentiment")

    # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM) — includes social posts
    # Hysteresis: flipping direction requires confidence > 0.55, same direction > 0.40
    votes["sentiment"] = "HOLD"
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from functools import partial

            from portfolio.sentiment import get_sentiment

            newsapi_key = (config or {}).get("newsapi_key", "")
            cc_api_key = (config or {}).get("cryptocompare_api_key", "") or None
            _sent_fn = partial(get_sentiment, cryptocompare_api_key=cc_api_key)
            sent = _cached(
                f"sentiment_{short_ticker}",
                SENTIMENT_TTL,
                _sent_fn,
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
            logger.debug("Optional module %s not available", "sentiment")

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
            logger.debug("Optional module %s not available", "macro_context")

    # Ministral-3-8B LLM reasoning (all tickers — crypto, stocks, metals)
    # Upgraded from legacy Ministral-8B (44% accuracy) to Ministral-3-8B.
    # custom_lora fully disabled: 20.9% accuracy, 97% SELL bias (worse than random).
    votes["ministral"] = "HOLD"
    if ticker and not skip_gpu:
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
                raw_action = orig["action"]
                gated_action, gating = _gate_local_model_vote(
                    "ministral", raw_action, ticker, config=config
                )
                extra_info["ministral_raw_action"] = raw_action
                extra_info["ministral_action"] = gated_action
                extra_info["ministral_reasoning"] = orig.get("reasoning", "")
                extra_info["ministral_accuracy"] = gating.get("accuracy")
                extra_info["ministral_samples"] = gating.get("samples", 0)
                extra_info["ministral_gating"] = gating.get("gating", "raw")
                if orig.get("confidence") is not None:
                    extra_info["ministral_confidence"] = orig["confidence"]
                votes["ministral"] = gated_action

                # custom_lora fully disabled — not even stored in extra.
                # Shadow A/B data preserved in data/ab_test_log.jsonl.
        except ImportError:
            logger.debug("Optional module %s not available", "ministral_signal")

    # Qwen3-8B LLM reasoning (all tickers — crypto, stocks, metals)
    # General financial model providing ensemble diversification vs Ministral.
    # Config: config.json → local_models.qwen3 (hold_threshold, min_samples)
    votes["qwen3"] = "HOLD"
    qwen3_enabled = (config or {}).get("local_models", {}).get("qwen3", {}).get("enabled", True)
    if ticker and qwen3_enabled and not skip_gpu:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.qwen3_signal import get_qwen3_signal

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

            # Determine asset type for prompt context
            if ticker in CRYPTO_SYMBOLS:
                asset_type = "cryptocurrency"
            elif ticker in METALS_SYMBOLS:
                asset_type = "precious metal"
            else:
                asset_type = "stock"

            ctx = {
                "ticker": short_ticker,
                "asset_type": asset_type,
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
            q3 = _cached(
                f"qwen3_{short_ticker}",
                MINISTRAL_TTL,
                get_qwen3_signal,
                ctx,
            )
            if q3:
                raw_action = q3.get("action", "HOLD")
                gated_action, gating = _gate_local_model_vote(
                    "qwen3", raw_action, ticker, config=config
                )
                extra_info["qwen3_raw_action"] = raw_action
                extra_info["qwen3_action"] = gated_action
                extra_info["qwen3_reasoning"] = q3.get("reasoning", "")
                extra_info["qwen3_accuracy"] = gating.get("accuracy")
                extra_info["qwen3_samples"] = gating.get("samples", 0)
                extra_info["qwen3_gating"] = gating.get("gating", "raw")
                if q3.get("confidence") is not None:
                    extra_info["qwen3_confidence"] = q3["confidence"]
                votes["qwen3"] = gated_action

        except ImportError:
            logger.debug("Optional module %s not available", "qwen3_signal")

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

        _signal_failures = []
        for sig_name, entry in _enhanced_entries.items():
            # Skip GPU-intensive enhanced signals for stocks outside market hours
            if skip_gpu and sig_name in GPU_SIGNALS:
                votes[sig_name] = "HOLD"
                continue
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
                max_conf = entry.get("max_confidence", 1.0)
                validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
                extra_info[f"{sig_name}_action"] = validated["action"]
                extra_info[f"{sig_name}_confidence"] = validated["confidence"]
                extra_info[f"{sig_name}_sub_signals"] = validated["sub_signals"]
                if validated["indicators"]:
                    extra_info[f"{sig_name}_indicators"] = validated["indicators"]
                votes[sig_name] = validated["action"]
            except Exception as e:
                logger.warning("Signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
                _signal_failures.append(sig_name)
        if _signal_failures:
            extra_info["_signal_failures"] = _signal_failures
            if len(_signal_failures) > 3:
                logger.warning(
                    "%s: %d enhanced signals failed: %s",
                    ticker, len(_signal_failures), ", ".join(_signal_failures),
                )

        # Persist signal health (single batch write for all enhanced signals)
        try:
            from portfolio.health import update_signal_health_batch
            health_results = {
                sig_name: (sig_name not in _signal_failures)
                for sig_name in _enhanced_entries
            }
            update_signal_health_batch(health_results)
        except Exception:
            logger.debug("Signal health tracking failed", exc_info=True)
    else:
        for sig_name in _enhanced_entries:
            votes[sig_name] = "HOLD"

    # Derive buy/sell counts from named votes
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")

    # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
    # Enhanced signals can strengthen/weaken a consensus but never create one alone.
    core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
    core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
    core_active = core_buy + core_sell

    # Total applicable signals: computed dynamically from SIGNAL_NAMES
    # minus DISABLED_SIGNALS minus per-asset-class exclusions.
    total_applicable = _compute_applicable_count(ticker, skip_gpu=skip_gpu)

    active_voters = buy + sell
    if ticker in STOCK_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK
    elif ticker in METALS_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK  # metals use same threshold
    else:
        min_voters = MIN_VOTERS_CRYPTO

    # Core gate: if no core signal is active, force HOLD regardless of enhanced votes
    if core_active == 0 or active_voters < min_voters:
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
    # (regime already computed early in the function for F&G gating)
    accuracy_data = {}
    activation_rates = {}
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy,
            load_cached_activation_rates,
            signal_accuracy,
            signal_accuracy_ewma,
            write_accuracy_cache,
        )

        # Read halflife from config (default 5 days)
        halflife_days = (config or {}).get("signals", {}).get("accuracy_halflife_days", 5)

        # Try EWMA-weighted accuracy first (smooth exponential decay, no hard 7d boundary)
        ewma = load_cached_accuracy("1d_ewma")
        if not ewma:
            ewma = signal_accuracy_ewma("1d", halflife_days=halflife_days)
            if ewma:
                write_accuracy_cache("1d_ewma", ewma)

        # Fall back to flat all-time accuracy if EWMA has no data
        if ewma and any(v.get("total", 0) > 0 for v in ewma.values()):
            accuracy_data = ewma
        else:
            alltime = load_cached_accuracy("1d")
            if not alltime:
                alltime = signal_accuracy("1d")
                if alltime:
                    write_accuracy_cache("1d", alltime)
            if alltime:
                accuracy_data = alltime

        activation_rates = load_cached_activation_rates()
    except Exception:
        logger.error("Accuracy stats load failed", exc_info=True)

    # Override global accuracy with per-ticker accuracy for LLM signals.
    # The global number averages across all tickers, but Qwen3/Ministral
    # performance varies hugely per ticker (e.g., Qwen3: MU 90%, PLTR 44%).
    for llm_sig in ("qwen3", "ministral"):
        per_ticker_acc = extra_info.get(f"{llm_sig}_accuracy")
        per_ticker_samples = extra_info.get(f"{llm_sig}_samples", 0)
        if per_ticker_acc is not None and per_ticker_samples >= 20:
            accuracy_data[llm_sig] = {
                "accuracy": per_ticker_acc,
                "total": per_ticker_samples,
                "correct": int(per_ticker_acc * per_ticker_samples),
                "pct": round(per_ticker_acc * 100, 1),
            }

    sig_cfg = (config or {}).get("signals", {})
    accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
    weighted_action, weighted_conf = _weighted_consensus(
        votes, accuracy_data, regime, activation_rates,
        accuracy_gate=accuracy_gate,
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
