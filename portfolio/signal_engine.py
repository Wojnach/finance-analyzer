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

# Adaptive recency blend: when recent accuracy diverges from all-time by more
# than this threshold, increase recent weight for faster regime adaptation.
# Normal: 70% recent + 30% all-time. Fast: 90% recent + 10% all-time.
_RECENCY_DIVERGENCE_THRESHOLD = 0.15  # 15% absolute divergence triggers fast blend
_RECENCY_WEIGHT_NORMAL = 0.7
_RECENCY_WEIGHT_FAST = 0.9

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
    "trending-up": {
        "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
        # Enhanced: boost trend-following, dampen mean-reversion
        "trend": 1.4, "momentum_factors": 1.3, "heikin_ashi": 1.2,
        "structure": 1.2, "smart_money": 1.1,
        "mean_reversion": 0.6, "fibonacci": 0.7,
    },
    "trending-down": {
        "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
        # Enhanced: same as trending-up (trend signals work both ways)
        "trend": 1.4, "momentum_factors": 1.3, "heikin_ashi": 1.2,
        "structure": 1.2, "smart_money": 1.1,
        "mean_reversion": 0.6, "fibonacci": 0.7,
    },
    "ranging": {
        "rsi": 1.5, "bb": 1.5, "ema": 0.5, "macd": 0.5,
        # Enhanced: boost mean-reversion and level-based signals
        "mean_reversion": 1.5, "fibonacci": 1.4, "calendar": 1.2,
        "oscillators": 1.2,
        "trend": 0.5, "momentum_factors": 0.6, "heikin_ashi": 0.6,
        "structure": 0.7,
    },
    "high-vol": {
        "bb": 1.5, "volume": 1.3, "ema": 0.5,
        # Enhanced: boost volatility-aware and smart money signals
        "volatility_sig": 1.4, "smart_money": 1.3, "volume_flow": 1.2,
        "candlestick": 1.2,
        "trend": 0.6, "calendar": 0.7, "mean_reversion": 0.7,
    },
}

# Regime-gated signals: completely silenced (forced HOLD) in certain regimes
# because they produce negative alpha.  Horizon-aware since 2026-03-29:
# BUG-149: trend has 61.6% accuracy on 3h even in ranging — short-term trends
# exist within range-bound markets, so only gate on longer horizons.
# Structure: {regime: {horizon: frozenset(signals), ...}}
# "_default" key applies to horizons not explicitly listed.
REGIME_GATED_SIGNALS: dict[str, dict[str, frozenset[str]]] = {
    "ranging": {
        # trend 1d_recent=40.7%, momentum_factors 1d_recent=41.4% — gate on daily
        "_default": frozenset({"trend", "momentum_factors"}),
        # trend 3h_recent=61.6%, momentum_factors 3h_recent=60.1% — do NOT gate
        "3h": frozenset(),
        "4h": frozenset(),
    },
    "trending-up": {
        # BUG-152: SELL-biased signals have 0-11% accuracy in trending-up.
        # Gating at 1d prevents false SELL consensus during breakouts.
        # trend ~0%, ema ~11%, volume_flow ~10%, macro_regime 11.1%, momentum_factors low
        # claude_fundamental 5.9% trending-up (34 samples) — BUG-154
        "_default": frozenset({
            "trend", "ema", "volume_flow", "macro_regime",
            "momentum_factors", "claude_fundamental",
        }),
        # mean_reversion 3h_recent=45.5% — gate on short horizons
        # SELL-biased signals work short-term even in uptrends — do NOT gate at 3h
        "3h": frozenset({"mean_reversion"}),
        "4h": frozenset({"mean_reversion"}),
    },
    "trending-down": {
        # BUG-155: bb 21.7% in trending-down (false reversal signals)
        # BUG-154: claude_fundamental 30.4% in trending-down
        "_default": frozenset({"bb", "claude_fundamental"}),
        "3h": frozenset({"mean_reversion"}),
        "4h": frozenset({"mean_reversion"}),
    },
}


def _get_regime_gated(regime: str, horizon: str | None = None) -> frozenset[str]:
    """Get the set of signals to gate for a regime+horizon combination."""
    regime_dict = REGIME_GATED_SIGNALS.get(regime, {})
    if not regime_dict:
        return frozenset()
    if horizon and horizon in regime_dict:
        return regime_dict[horizon]
    return regime_dict.get("_default", frozenset())

# Horizon-specific signal weight multipliers.
# Signals with >15pp accuracy divergence between horizons get adjusted.
# Updated: 2026-03-29 accuracy audit (3h_recent vs 1d_recent).
HORIZON_SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {
    "3h": {
        "news_event": 1.4,      # 70.0% at 3h (vs 29.5% at 1d — pure short-term signal)
        "smart_money": 1.2,     # 63.2% at 3h (vs 39.6% at 1d) — NEW 2026-03-29
        "ema": 1.3,             # 62.9% at 3h (vs 40.8% at 1d)
        "ministral": 1.2,       # 62.6% at 3h
        "qwen3": 1.2,           # 61.8% at 3h — NEW 2026-03-29
        "trend": 1.2,           # 61.6% at 3h (vs 40.7% at 1d) — NEW 2026-03-29
        "volatility_sig": 1.2,  # 60.2% at 3h (vs 35.0% at 1d) — NEW 2026-03-29
        "momentum_factors": 1.2, # 60.1% at 3h (vs 41.4% at 1d) — NEW 2026-03-29
        "sentiment": 0.5,       # 33.8% at 3h — worst performer
        "fibonacci": 0.6,       # 38.3% at 3h (but 68.2% at 1d)
        "forecast": 0.5,        # 38.3% at 3h — tightened from 0.6
        "oscillators": 0.7,     # 39.4% at 3h — NEW 2026-03-29
        "bb": 0.6,              # 41.7% at 3h (but 60.8% at 1d) — NEW 2026-03-29
        "mean_reversion": 0.7,  # 45.5% at 3h (but 65.4% at 1d) — NEW 2026-03-29
    },
    "4h": {
        "news_event": 1.4,
        "smart_money": 1.2,
        "ema": 1.3,
        "ministral": 1.2,
        "qwen3": 1.2,
        "trend": 1.2,
        "volatility_sig": 1.2,
        "momentum_factors": 1.2,
        "sentiment": 0.5,
        "fibonacci": 0.6,
        "forecast": 0.5,
        "oscillators": 0.7,
        "bb": 0.6,
        "mean_reversion": 0.7,
    },
    "1d": {
        "fibonacci": 1.4,       # 68.2% at 1d
        "ministral": 1.3,       # 68.0% at 1d — NEW 2026-03-29 (was only 3h boost)
        "mean_reversion": 1.3,  # 65.4% at 1d
        "calendar": 1.2,        # 62.8% at 1d
        "bb": 1.2,              # 60.8% at 1d (vs 41.7% at 3h!) — NEW 2026-03-29
        "macd": 1.2,            # 58.7% at 1d — NEW 2026-03-29
        "news_event": 0.5,      # 29.5% at 1d (reversal of 3h edge)
        "fear_greed": 0.4,      # 25.9% at 1d — collapsed, tightened from 0.5
        "macro_regime": 0.5,    # 30.3% at 1d
        "volatility_sig": 0.5,  # 35.0% at 1d — NEW 2026-03-29
        "structure": 0.6,       # 36.1% at 1d
        "forecast": 0.5,        # 36.1% at 1d — NEW 2026-03-29
        "smart_money": 0.6,     # 39.6% at 1d (vs 63.2% at 3h) — NEW 2026-03-29
        "ema": 0.6,             # 40.8% at 1d (vs 62.9% at 3h) — BUG-151
        "trend": 0.6,           # 40.7% at 1d — NEW 2026-03-29
        "heikin_ashi": 0.7,     # 42.0% at 1d — NEW 2026-03-29
    },
}

# Activity rate cap: signals with activation rate above this threshold get
# an additional penalty to prevent a single high-activity signal from
# dominating consensus.  Targets volume_flow (83.1% activity, 49.2% accuracy).
_ACTIVITY_RATE_CAP = 0.70
_ACTIVITY_RATE_PENALTY = 0.5

# Dynamic horizon weight computation settings
_DYNAMIC_HORIZON_WEIGHT_TTL = 3600  # 1 hour cache
_DYNAMIC_HORIZON_MIN_SAMPLES = 50   # need enough data per signal per horizon
_DYNAMIC_HORIZON_CLAMP_LOW = 0.4    # minimum multiplier
_DYNAMIC_HORIZON_CLAMP_HIGH = 1.5   # maximum multiplier
_DYNAMIC_HORIZON_DEADBAND = 0.1     # ignore multipliers within ±10% of 1.0

# Cross-horizon pairs: for a given horizon, which other horizons to compare against
_CROSS_HORIZON_PAIRS = {
    "3h": ["1d"],
    "4h": ["1d"],
    "1d": ["3h"],
}


def _compute_dynamic_horizon_weights(horizon: str) -> dict[str, float]:
    """Compute horizon-specific signal weight multipliers from accuracy cache.

    For each signal, computes the ratio of its accuracy on this horizon vs
    the comparison horizon(s). Signals that perform much better on this
    horizon get boosted; signals that perform much worse get penalized.

    Returns a dict of {signal_name: multiplier} for multipliers outside
    the deadband (i.e., > 1.1 or < 0.9). Falls back to static
    HORIZON_SIGNAL_WEIGHTS if accuracy cache is unavailable.
    """
    try:
        from portfolio.file_utils import load_json
        cache = load_json(DATA_DIR / "accuracy_cache.json")
        if not cache:
            return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})

        # Get recent accuracy for this horizon and comparison horizons
        this_key = f"{horizon}_recent"
        this_data = cache.get(this_key, {})
        if not this_data:
            return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})

        cross_horizons = _CROSS_HORIZON_PAIRS.get(horizon, [])
        if not cross_horizons:
            return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})

        # Gather comparison accuracies (true mean across comparison horizons)
        cross_sum: dict[str, float] = {}
        cross_count: dict[str, int] = {}
        for ch in cross_horizons:
            ch_key = f"{ch}_recent"
            ch_acc = cache.get(ch_key, {})
            for sig, stats in ch_acc.items():
                if stats.get("total", 0) >= _DYNAMIC_HORIZON_MIN_SAMPLES:
                    acc = stats.get("accuracy", 0.5)
                    cross_sum[sig] = cross_sum.get(sig, 0.0) + acc
                    cross_count[sig] = cross_count.get(sig, 0) + 1
        cross_data = {sig: cross_sum[sig] / cross_count[sig] for sig in cross_sum}

        # Compute multipliers
        weights = {}
        for sig, stats in this_data.items():
            samples = stats.get("total", 0)
            if samples < _DYNAMIC_HORIZON_MIN_SAMPLES:
                continue
            this_acc = stats.get("accuracy", 0.5)
            cross_acc = cross_data.get(sig)
            if cross_acc is None or cross_acc < 0.01:
                continue

            # Ratio of this-horizon accuracy to cross-horizon accuracy
            ratio = this_acc / cross_acc
            # Clamp
            ratio = max(_DYNAMIC_HORIZON_CLAMP_LOW, min(_DYNAMIC_HORIZON_CLAMP_HIGH, ratio))
            # Deadband: only include if meaningfully different from 1.0
            if abs(ratio - 1.0) > _DYNAMIC_HORIZON_DEADBAND:
                weights[sig] = round(ratio, 2)

        return weights if weights else HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
    except Exception:
        logger.debug("Dynamic horizon weights unavailable, using static fallback", exc_info=True)
        return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})


def _get_horizon_weights(horizon: str | None) -> dict[str, float]:
    """Get horizon-specific signal weight multipliers, preferring dynamic computation.

    Uses cached dynamic weights when available, falling back to static dict.
    """
    if not horizon:
        return {}
    cache_key = f"dynamic_horizon_weights_{horizon}"
    return _cached(cache_key, _DYNAMIC_HORIZON_WEIGHT_TTL, lambda: _compute_dynamic_horizon_weights(horizon))


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


# Correlation groups: signals that frequently agree (>90% in recent data).
# Within a group, only the highest-accuracy signal gets full weight;
# others get a penalty to prevent correlated signals inflating consensus.
CORRELATION_GROUPS = {
    # BUG-153: Split low_activity_timing — calendar+econ_calendar are excellent
    # (62.8%/86.8%) while forecast+futures_flow are broken (36.1%/33.3%).
    # Mixing them risks forecast becoming leader and suppressing calendar.
    "low_activity_timing": frozenset({"calendar", "econ_calendar"}),
    "rare_technical": frozenset({"volatility_sig", "oscillators"}),
    # Discovered 2026-03-27: ema/trend corr=0.55, all share SELL bias (37-40%)
    "trend_direction": frozenset({"ema", "trend", "heikin_ashi"}),
    # Discovered 2026-03-27: both permanent SELL lean (volume_flow 69% SELL, macro_regime 44% SELL)
    "high_volume_sell": frozenset({"volume_flow", "macro_regime"}),
    # Discovered 2026-03-29: all depend on external data quality. sentiment 33.8% 3h,
    # fear_greed 25.9% 1d, news_event varies wildly by horizon. When external data
    # degrades, all fail together.
    "macro_external": frozenset({"fear_greed", "sentiment", "news_event"}),
}
_CORRELATION_PENALTY = 0.3  # secondary signals in a group get 30% of normal weight


def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
                        accuracy_gate=None, max_signals=None, horizon=None):
    """Compute weighted consensus using accuracy, regime, and activation frequency.

    Weight per signal = accuracy_weight * regime_mult * normalized_weight
                        * horizon_mult * activity_cap
    where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
    Rare, balanced signals get more weight; noisy/biased signals get less.

    Signals below the accuracy gate (with sufficient samples) are force-skipped —
    they are noise, not useful contrarian indicators.

    Regime gating: signals in REGIME_GATED_SIGNALS for the current regime are
    forced to HOLD before vote processing — they produce negative alpha.

    Correlation deduplication: within defined correlation groups, only the
    highest-accuracy signal gets full weight. Others get 0.3x penalty.

    Horizon-specific weights: signals with divergent accuracy across horizons
    get boosted or penalized via HORIZON_SIGNAL_WEIGHTS.

    Activity rate cap: signals with >70% activation rate get 0.5x penalty
    to prevent a single high-activity signal from dominating consensus.

    Top-N gate: when max_signals is set, only the top max_signals non-HOLD
    signals (ranked by accuracy) participate in the consensus. This focuses
    the vote on the best performers and ignores marginal contributors.
    """
    gate = accuracy_gate if accuracy_gate is not None else ACCURACY_GATE_THRESHOLD
    buy_weight = 0.0
    sell_weight = 0.0
    gated_signals = []
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    horizon_mults = _get_horizon_weights(horizon)

    # Regime gating: force-HOLD signals that produce negative alpha in this regime.
    # BUG-149: now horizon-aware — e.g., trend works at 3h in ranging (61.6%)
    regime_gated = _get_regime_gated(regime, horizon)
    votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}

    # Top-N gate: only let the top max_signals (by accuracy) participate
    active_votes = {k: v for k, v in votes.items() if v != "HOLD"}
    if max_signals and len(active_votes) > max_signals:
        ranked = sorted(
            active_votes.keys(),
            key=lambda s: accuracy_data.get(s, {}).get("accuracy", 0.5),
            reverse=True,
        )
        excluded = set(ranked[max_signals:])
    else:
        excluded = set()

    # Pre-compute which signal is the "leader" (highest accuracy) in each
    # correlation group, considering only signals that are actively voting.
    active_non_hold = {s for s, v in votes.items() if v != "HOLD"}
    group_leaders = {}
    for group_name, group_sigs in CORRELATION_GROUPS.items():
        active_in_group = active_non_hold & group_sigs
        if len(active_in_group) <= 1:
            continue
        best_sig = max(
            active_in_group,
            key=lambda s: accuracy_data.get(s, {}).get("accuracy", 0.5),
        )
        group_leaders[group_name] = best_sig

    # Build a set of signals that should get the correlation penalty
    penalized_signals = set()
    for group_name, group_sigs in CORRELATION_GROUPS.items():
        leader = group_leaders.get(group_name)
        if leader:
            for s in group_sigs:
                if s != leader and s in active_non_hold:
                    penalized_signals.add(s)

    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        if signal_name in excluded:
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
        # Horizon-specific weight adjustment
        if signal_name in horizon_mults:
            weight *= horizon_mults[signal_name]
        # Activation frequency normalization (rarity * bias correction)
        act_data = activation_rates.get(signal_name, {})
        norm_weight = act_data.get("normalized_weight", 1.0)
        weight *= norm_weight
        # Activity rate cap: penalize signals with extremely high activation rates
        act_rate = act_data.get("activation_rate", 0.0)
        if act_rate > _ACTIVITY_RATE_CAP:
            weight *= _ACTIVITY_RATE_PENALTY
        # Correlation penalty: secondary signals in a group get reduced weight
        if signal_name in penalized_signals:
            weight *= _CORRELATION_PENALTY
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


def _time_of_day_factor(horizon=None):
    hour = datetime.now(UTC).hour
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import time_of_day_scale_3h
        return time_of_day_scale_3h(hour)
    # Default 1d behavior
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


def _build_llm_context(ticker, ind, timeframes, extra_info):
    """Build shared context dict for local LLM signals (Ministral, Qwen3)."""
    tf_summary = ""
    if timeframes:
        parts = []
        for label, entry in timeframes:
            if isinstance(entry, dict) and "action" in entry and entry["action"]:
                ti = entry.get("indicators", {})
                parts.append(f"{label}: {entry['action']} (RSI={ti.get('rsi', 0):.0f})")
        if parts:
            tf_summary = " | ".join(parts)

    ema_gap = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100
        if ind["ema21"] != 0
        else 0
    )

    return {
        "ticker": ticker.replace("-USD", ""),
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

    # --- Stage 5: Unanimity penalty ---
    # When all signals agree, the move is often already priced in.
    # 90%+ confidence has 28-32% actual accuracy across all horizons.
    if action != "HOLD" and conf > 0.0:
        buy_count = extra_info.get("_buy_count", 0)
        sell_count = extra_info.get("_sell_count", 0)
        total_voters = buy_count + sell_count
        if total_voters > 0:
            agreement_ratio = max(buy_count, sell_count) / total_voters
            if agreement_ratio >= 0.9:  # 90%+ agreement
                conf *= 0.6
                penalty_log.append({"stage": "unanimity", "agreement": round(agreement_ratio, 3), "mult": 0.6})
            elif agreement_ratio >= 0.8:  # 80-90% agreement
                conf *= 0.75
                penalty_log.append({"stage": "unanimity", "agreement": round(agreement_ratio, 3), "mult": 0.75})

    # Clamp confidence to [0, 1]
    conf = max(0.0, min(1.0, conf))

    return action, conf, penalty_log


def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None, horizon=None):
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
    if horizon in ("3h", "4h"):
        # 3h: RSI(7) is more sensitive — use fixed 25/75 thresholds
        rsi_lower = 25
        rsi_upper = 75
    else:
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

            ctx = _build_llm_context(ticker, ind, timeframes, extra_info)
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

            ctx = _build_llm_context(ticker, ind, timeframes, extra_info)
            # Qwen3 gets asset_type for prompt diversification
            if ticker in CRYPTO_SYMBOLS:
                ctx["asset_type"] = "cryptocurrency"
            elif ticker in METALS_SYMBOLS:
                ctx["asset_type"] = "precious metal"
            else:
                ctx["asset_type"] = "stock"
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
        # BUG-144: Include regime so enhanced signals (forecast.py) can apply
        # regime-specific confidence discounts.
        context_data = {"ticker": ticker, "config": config or {}, "macro": macro_data, "regime": regime}

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

    # 3h horizon: gate slow signals that are noise at short timeframes
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import is_slow_signal_3h
        for sig_name in list(votes.keys()):
            if is_slow_signal_3h(sig_name) and votes[sig_name] != "HOLD":
                votes[sig_name] = "HOLD"

    # BUG-143: Apply regime gating BEFORE computing buy/sell counts so that
    # all downstream code (core gate, min_voters, unanimity penalty) sees
    # post-gated counts.  _weighted_consensus also applies this internally
    # (idempotent — gating HOLD→HOLD is a no-op).
    # BUG-149: now horizon-aware via _get_regime_gated()
    regime_gated = _get_regime_gated(regime, horizon)
    for sig_name in regime_gated:
        if sig_name in votes and votes[sig_name] != "HOLD":
            votes[sig_name] = "HOLD"

    # Derive buy/sell counts from named votes (post-gating)
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
            blend_accuracy_data,
            load_cached_accuracy,
            load_cached_activation_rates,
            signal_accuracy,
            signal_accuracy_recent,
            write_accuracy_cache,
        )

        # Select accuracy horizon — use 3h accuracy when predicting 3h moves
        acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"

        # Load all-time accuracy
        alltime = load_cached_accuracy(acc_horizon)
        if not alltime:
            alltime = signal_accuracy(acc_horizon)
            if alltime:
                write_accuracy_cache(acc_horizon, alltime)

        # Load recent accuracy (7d window) — more responsive to regime changes
        recent = load_cached_accuracy(f"{acc_horizon}_recent")
        if not recent:
            recent = signal_accuracy_recent(acc_horizon, days=7)
            if recent:
                write_accuracy_cache(f"{acc_horizon}_recent", recent)

        # ARCH-23: Use shared blend function (replaces inline logic).
        accuracy_data = blend_accuracy_data(
            alltime, recent,
            divergence_threshold=_RECENCY_DIVERGENCE_THRESHOLD,
            normal_weight=_RECENCY_WEIGHT_NORMAL,
            fast_weight=_RECENCY_WEIGHT_FAST,
        )

        activation_rates = load_cached_activation_rates()
    except Exception:
        logger.error("Accuracy stats load failed", exc_info=True)

    # Overlay regime-specific accuracy when available
    try:
        from portfolio.accuracy_stats import (
            load_cached_regime_accuracy,
            signal_accuracy_by_regime,
            write_regime_accuracy_cache,
        )
        # BUG-134: Use acc_horizon (not hardcoded "1d") so regime accuracy
        # matches the prediction horizon (3h/4h/12h/1d).
        regime_acc = load_cached_regime_accuracy(acc_horizon)
        if not regime_acc:
            regime_acc = signal_accuracy_by_regime(acc_horizon)
            if regime_acc:
                write_regime_accuracy_cache(acc_horizon, regime_acc)
        current_regime_data = regime_acc.get(regime, {})
        for sig_name, rdata in current_regime_data.items():
            if rdata.get("total", 0) >= 30:
                accuracy_data[sig_name] = rdata
    except Exception:
        logger.debug("Regime-conditional accuracy unavailable", exc_info=True)

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

    # Utility boost: scale accuracy weight by return-based utility score.
    # Signals that catch large moves (high avg_return) get a confidence boost
    # capped at 1.5x, applied only when >= 30 samples exist and avg_return > 0.
    try:
        from portfolio.accuracy_stats import signal_utility
        # BUG-135: Use acc_horizon (not hardcoded "1d") so utility boost
        # reflects the actual prediction horizon's return profile.
        utility_data = signal_utility(acc_horizon)
        for sig_name in list(accuracy_data.keys()):
            u = utility_data.get(sig_name, {})
            u_score = u.get("avg_return", 0.0)
            samples = u.get("samples", 0)
            if samples >= 30 and u_score > 0:
                boost = min(1.0 + u_score, 1.5)
                if sig_name in accuracy_data:
                    # BUG-136: Build a new dict instead of mutating in-place.
                    # The accuracy_data may be a reference to cached alltime data.
                    boosted_acc = min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)
                    accuracy_data[sig_name] = {
                        **accuracy_data[sig_name],
                        "accuracy": boosted_acc,
                    }
    except Exception:
        logger.debug("Utility weighting unavailable", exc_info=True)

    # Multi-horizon: optionally use each signal's best horizon accuracy
    sig_cfg = (config or {}).get("signals", {})
    if sig_cfg.get("use_best_horizon", False):
        try:
            from portfolio.accuracy_stats import signal_best_horizon_accuracy
            best_hz = signal_best_horizon_accuracy(min_samples=50)
            for sig_name, bh_data in best_hz.items():
                if bh_data.get("total", 0) >= 30:
                    # Only override if best-horizon accuracy is meaningfully better
                    current = accuracy_data.get(sig_name, {}).get("accuracy", 0.5)
                    if bh_data["accuracy"] > current + 0.03:
                        accuracy_data[sig_name] = bh_data
        except Exception:
            logger.debug("Best-horizon accuracy unavailable", exc_info=True)
    accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
    max_signals = sig_cfg.get("max_active_signals")
    weighted_action, weighted_conf = _weighted_consensus(
        votes, accuracy_data, regime, activation_rates,
        accuracy_gate=accuracy_gate,
        max_signals=max_signals,
        horizon=horizon,
    )

    # Apply core gate AND MIN_VOTERS gate to weighted consensus too
    if core_active == 0 or active_voters < min_voters:
        weighted_action = "HOLD"
        weighted_conf = 0.0

    # Confluence score
    confluence = _confluence_score(votes, extra_info)

    # Time-of-day confidence adjustment
    tod_factor = _time_of_day_factor(horizon=horizon)
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
    if horizon:
        extra_info["_horizon"] = horizon
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

    # Global confidence cap — calibration data shows >80% confidence is
    # anti-correlated with accuracy at every horizon (70-80% bucket is the
    # best performing at 57-59% actual accuracy)
    conf = min(conf, 0.80)

    # 3h horizon: cap confidence to prevent overconfident short-term predictions
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import CONFIDENCE_CAP_3H
        conf = min(conf, CONFIDENCE_CAP_3H)

    return action, conf, extra_info
