"""Signal generation engine — 32-signal voting system with weighted consensus."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from portfolio.indicators import detect_regime
from portfolio.shared_state import (
    FEAR_GREED_TTL,
    FUNDING_RATE_TTL,
    MINISTRAL_TTL,
    ONCHAIN_TTL,
    SENTIMENT_TTL,
    VOLUME_TTL,
    _cached,
    _cached_or_enqueue,
)
from portfolio.signal_registry import get_enhanced_signals, load_signal_func
from portfolio.signal_utils import true_range
from portfolio.tickers import CRYPTO_SYMBOLS, DISABLED_SIGNALS, GPU_SIGNALS, METALS_SYMBOLS, SIGNAL_NAMES, STOCK_SYMBOLS

logger = logging.getLogger("portfolio.signal_engine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOCAL_MODEL_ACCURACY_TTL = 1800

# ADX computation cache — content-keyed by (len, first_close, last_close)
# to prevent stale hits from Python address reuse after GC.
_adx_cache: dict[tuple[int, float, float], float | None] = {}
_adx_lock = threading.Lock()  # BUG-86: protect concurrent access from ThreadPoolExecutor
_ADX_CACHE_MAX = 200  # prevent unbounded growth

# BUG-178 diagnostics: per-ticker last-signal tracker.
# Updated right before each enhanced signal's compute_fn() is called so that
# when the BUG-178 ticker pool timeout fires, main.py can ask which signal
# each stuck ticker was running. Surfaces silent hangs (signals that never
# complete and therefore never trip the [SLOW] >1s logger).
# Added 2026-04-10 after a 49-event BUG-178 audit traced silent hangs to the
# disabled-signals dispatch path. Cheap (single dict write per signal call).
_last_signal_per_ticker: dict[str, tuple[str, float]] = {}
_last_signal_lock = threading.Lock()


def _set_last_signal(ticker: str, sig_name: str) -> None:
    """Record the signal currently being computed for a ticker (BUG-178 diag)."""
    with _last_signal_lock:
        _last_signal_per_ticker[ticker] = (sig_name, time.monotonic())


def get_last_signal(ticker: str) -> tuple[str, float] | None:
    """Return (sig_name, elapsed_seconds) for the most recent signal start
    on this ticker, or None if no signal has been recorded.

    Used by main.py's BUG-178 timeout handler to identify which signal hung.
    """
    with _last_signal_lock:
        entry = _last_signal_per_ticker.get(ticker)
    if entry is None:
        return None
    sig_name, started = entry
    return sig_name, time.monotonic() - started


# BUG-178 phase log (added 2026-04-15): records per-ticker phase durations
# inside generate_signal()'s post-dispatch code. The __post_dispatch__ marker
# above was too coarse — it collapsed 7+ distinct post-dispatch operations
# (accuracy load, weighted consensus, penalties, linear factor, etc.) into
# a single "after dispatch" bucket, so slow cycles with elapsed_since_set
# ~170s out of a 180s pool timeout gave us zero signal about which phase
# was actually slow.
#
# Each phase records (phase_name, duration_seconds). main.py's slow-cycle
# diagnostic reads this log when the pool timeout fires so we can see the
# full phase breakdown retrospectively. Bounded per-ticker (replaced on
# each generate_signal call) so memory is constant.
_phase_log_per_ticker: dict[str, list[tuple[str, float]]] = {}
_phase_log_lock = threading.Lock()

_PHASE_WARN_THRESHOLD_S = 2.0

# Defensive bound on the number of distinct ticker keys kept. In production
# this is 5 (Tier-1 symbols); tests and probes may pass arbitrary names and
# slowly grow the dict. When we exceed the cap, prune the least-recently-
# used entries (reset cycles refresh them, so LRU by last-reset is fine).
# Prior callers were silently leaking one small list per unique ticker name.
_PHASE_LOG_MAX_TICKERS = 64


def _reset_phase_log(ticker: str) -> None:
    """Clear the phase log for a ticker at the start of generate_signal.

    Also enforces _PHASE_LOG_MAX_TICKERS by pruning older entries when the
    dict grows past the cap — cheap per-call O(n) but n is bounded and the
    prune happens at most once per generate_signal invocation.
    """
    if not ticker:
        return
    with _phase_log_lock:
        if len(_phase_log_per_ticker) >= _PHASE_LOG_MAX_TICKERS and ticker not in _phase_log_per_ticker:
            # Evict oldest half — we don't need true LRU, just bounded memory.
            # `iter(dict)` yields insertion order in CPython 3.7+; dropping
            # the first half gives us amortized O(1) per call.
            evict_count = len(_phase_log_per_ticker) // 2
            for old_key in list(_phase_log_per_ticker)[:evict_count]:
                del _phase_log_per_ticker[old_key]
        _phase_log_per_ticker[ticker] = []


def _record_phase(ticker: str, phase: str, start_mono: float) -> float:
    """Record a phase completion for a ticker. Returns the phase duration.

    Logs WARNING if duration > _PHASE_WARN_THRESHOLD_S so that slow
    individual phases (e.g., cold accuracy_stats load, lock contention)
    are visible in portfolio.log without waiting for a BUG-178 timeout.
    """
    if not ticker:
        return 0.0
    dur = time.monotonic() - start_mono
    with _phase_log_lock:
        _phase_log_per_ticker.setdefault(ticker, []).append((phase, dur))
    if dur > _PHASE_WARN_THRESHOLD_S:
        logger.warning("[SLOW-PHASE] %s/%s: %.1fs", ticker, phase, dur)
    return dur


def get_phase_log(ticker: str) -> list[tuple[str, float]]:
    """Return the phase breakdown for a ticker's last generate_signal call.

    Used by main.py's BUG-178 slow-cycle diagnostic to dump per-phase timing
    when the ticker pool times out. Returns an empty list if no log exists.
    """
    with _phase_log_lock:
        return list(_phase_log_per_ticker.get(ticker, []))

_LOCAL_MODEL_HOLD_THRESHOLD = 0.55
_LOCAL_MODEL_MIN_SAMPLES = 30
_LOCAL_MODEL_LOOKBACK_DAYS = 30

# 2026-05-11 — Stage 2 Batch 1: dead-zone HOLD-bias reduction.
# Investigation found EMA / BB / MACD core voters abstaining 88-94% on
# metals: _total_applicable=20, _voters=5. User rationale: HOLD is for
# managing open positions, not for entry decisions — entries should
# always pick a direction. So when these signals would otherwise HOLD
# (gap < 0.5% on EMA, price inside the BB band, |MACD hist| below
# threshold), we now emit a *weak* directional vote based on the
# secondary derivative (slope of EMA9 vs EMA21, distance from BB mid,
# slope of MACD histogram). The strong-vote paths are unchanged.
#
# The "conf" returned by the helpers is informational — it goes into
# extra_info so consumers can see the soft confidence. Downstream
# weighting (accuracy gate, regime mult, horizon mult, correlation
# group leader) still applies, so a soft vote on a force-HOLD signal
# is still force-HOLD'd. We are NOT bypassing the accuracy gate.
#
# Stays well below the typical strong-vote confidence (0.5-0.8) so
# the weight ranking continues to prefer real conviction signals.
EMA_DEAD_ZONE_SOFT_CONF = 0.20
BB_INSIDE_SOFT_CONF = 0.15
MACD_DEAD_ZONE_SOFT_CONF = 0.20
# 2026-05-11 — Stage 2 Batch 2: extend soft-directional pattern to two
# more high-HOLD enhanced voters. Candlestick abstained 87.6% on metals
# (HOLD when no recognised pattern); forecast abstained 87.0% (HOLD when
# Chronos low-conf or gated). Soft confs deliberately LOWER than Batch 1
# (0.12-0.15) because the secondary derivative we read (raw body
# direction / price+EMA slope) is even weaker than EMA9/EMA21 slope
# divergence — i.e. less of a "true regime hint", more of a tiebreaker.
# Strong-vote paths are unchanged; soft branches only fire when the
# enhanced compute_fn returned HOLD AND the secondary derivative has a
# clear direction. Mixed/ambiguous secondary signals keep HOLD.
CANDLESTICK_DEAD_ZONE_SOFT_CONF = 0.15
FORECAST_DEAD_ZONE_SOFT_CONF = 0.12
# How many of the last N candle bodies must agree (close vs open sign)
# before the candlestick soft branch emits a directional vote. 3/3 is a
# clean unanimity; 2/3 keeps HOLD because that's a coinflip+1, well
# within candle noise — the soft branch is not supposed to fight a
# 2-vs-1 split.
_CANDLE_BODY_LOOKBACK = 3
# How many bars the forecast soft branch reads to estimate the
# price+EMA21 slope alignment. 5 bars is the minimum that filters
# single-bar noise without lagging the regime — same rationale as
# _DEAD_ZONE_SLOPE_LOOKBACK but a touch longer because we want a
# slower, "actually still rising / falling" read, not a tick.
_FORECAST_SLOPE_LOOKBACK = 5
# How many bars to look back when measuring EMA slope and MACD slope
# inside the dead-zone helpers. 3 bars = enough to filter single-bar
# noise without lagging the regime.
_DEAD_ZONE_SLOPE_LOOKBACK = 3
# Slope tie-break tolerance. EMA slopes within `price * 1e-5` of each
# other count as "flat" and keep HOLD. MACD histogram slope within
# `1e-9` (absolute) of zero counts as flat.
_EMA_FLAT_EPS_REL = 1e-5
_MACD_FLAT_EPS = 1e-9
# 2026-05-11 (Codex review Fix A): MACD dead-zone helper must gate on
# *magnitude* of the current histogram before considering slope. Without
# this, any non-crossover bar with non-flat slope produces a soft vote —
# including bars where |hist| is large and the strong-vote path is the
# correct authority. The strong path handles crossovers (sign flip), so
# whenever |current_hist| is comfortably above zero but no crossover
# occurred, the right behaviour is HOLD and let the strong-path own it.
# 0.05 is a small absolute fraction picked to cover typical noisy zero
# crossings without bleeding into normal histogram amplitudes.
MACD_DEAD_ZONE_MAGNITUDE_THRESHOLD = 0.05


def _ema_dead_zone_vote(ind, df, lookback=_DEAD_ZONE_SLOPE_LOOKBACK):
    """Return (vote, conf) when EMA gap is in the dead zone (<0.5%).

    Compares the slope of EMA9 vs EMA21 over the last `lookback` bars.
    EMA9 rising faster than EMA21 -> soft BUY. EMA9 falling faster
    -> soft SELL. Slopes within `tiny_eps` of each other -> HOLD.

    Returns ("HOLD", 0.0) if df is missing/short or slopes truly flat.

    2026-05-11: introduced to convert dead-zone HOLDs into weak
    directional votes. See module-level rationale comment above.
    """
    if df is None or "close" not in df or len(df) < lookback + 21:
        return "HOLD", 0.0
    try:
        close = df["close"]
        ema9_series = close.ewm(span=9, adjust=False).mean()
        ema21_series = close.ewm(span=21, adjust=False).mean()
        # Slope = last - value-`lookback`-bars-ago. Linear; sign is what
        # matters, not magnitude.
        ema9_slope = float(ema9_series.iloc[-1] - ema9_series.iloc[-1 - lookback])
        ema21_slope = float(ema21_series.iloc[-1] - ema21_series.iloc[-1 - lookback])
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    price = float(ind.get("close", 0.0)) or float(close.iloc[-1])
    tiny_eps = abs(price) * _EMA_FLAT_EPS_REL
    if ema9_slope > ema21_slope + tiny_eps:
        return "BUY", EMA_DEAD_ZONE_SOFT_CONF
    if ema9_slope < ema21_slope - tiny_eps:
        return "SELL", EMA_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


def _bb_inside_band_vote(ind):
    """Return (vote, conf) when price is inside the Bollinger band.

    Uses normalized band position: (price - mid) / (upper - mid),
    clamped to [-1, +1]. Position > 0.6 -> soft SELL (near upper),
    < -0.6 -> soft BUY (near lower), else HOLD (mid-band).

    2026-05-11: introduced to convert dead-zone HOLDs into weak
    directional votes. See module-level rationale comment above.
    """
    try:
        price = float(ind["close"])
        mid = float(ind["bb_mid"])
        upper = float(ind["bb_upper"])
    except (KeyError, TypeError, ValueError):
        return "HOLD", 0.0
    half_width = upper - mid
    if half_width <= 0:
        return "HOLD", 0.0
    band_position = (price - mid) / half_width
    # Clamp to [-1, +1] so degenerate / wick data can't produce
    # arbitrarily large positions.
    band_position = max(-1.0, min(1.0, band_position))
    if band_position > 0.6:
        return "SELL", BB_INSIDE_SOFT_CONF
    if band_position < -0.6:
        return "BUY", BB_INSIDE_SOFT_CONF
    return "HOLD", 0.0


def _macd_dead_zone_vote(ind, df, lookback=_DEAD_ZONE_SLOPE_LOOKBACK):
    """Return (vote, conf) when |MACD hist| is in the dead zone.

    Measures histogram slope over the last `lookback` bars. Rising
    histogram -> soft BUY; falling -> soft SELL. Flat (within
    `_MACD_FLAT_EPS` absolute) -> HOLD.

    Recomputes the histogram from `df["close"]` because `ind` only
    carries the last two histogram values.

    2026-05-11: introduced to convert dead-zone HOLDs into weak
    directional votes. See module-level rationale comment above.

    2026-05-11 (Codex Fix A): added magnitude gate. The strong-vote
    path owns any bar where the histogram has meaningful amplitude;
    this helper must only fire when |current_hist| is actually in the
    dead zone (small absolute value). Without this gate, every non-
    crossover bar with non-flat slope produced a soft vote — even on
    large histogram magnitudes where the strong path's "no crossover"
    decision should win and the answer is genuinely HOLD.
    """
    if df is None or "close" not in df or len(df) < lookback + 26:
        return "HOLD", 0.0
    # 2026-05-11 Codex Fix A: prefer ind's current histogram (already
    # computed by the standard pipeline) for the magnitude gate. Falls
    # through to recomputing if ind didn't carry it.
    current_hist = None
    try:
        current_hist = float(ind.get("macd_hist", 0.0))
    except (TypeError, ValueError):
        current_hist = None
    try:
        close = df["close"]
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - macd_signal
        hist_now = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-1 - lookback])
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    # If ind lacked a usable current hist, fall back to the recomputed
    # value so the magnitude gate still applies.
    if current_hist is None:
        current_hist = hist_now
    # Magnitude gate: only fire in the actual dead zone. Outside the
    # dead zone the strong path is responsible; we return HOLD here
    # rather than emitting a soft vote that would overlap with (or
    # contradict) the strong-vote logic.
    if abs(current_hist) >= MACD_DEAD_ZONE_MAGNITUDE_THRESHOLD:
        return "HOLD", 0.0
    delta = hist_now - hist_prev
    if delta > _MACD_FLAT_EPS:
        return "BUY", MACD_DEAD_ZONE_SOFT_CONF
    if delta < -_MACD_FLAT_EPS:
        return "SELL", MACD_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


def _candlestick_dead_zone_vote(df, lookback=_CANDLE_BODY_LOOKBACK):
    """Return (vote, conf) when no candlestick pattern matched.

    Fallback secondary derivative: count bullish (close>open) vs bearish
    (close<open) bodies over the last `lookback` bars. Unanimous bullish
    -> soft BUY at CANDLESTICK_DEAD_ZONE_SOFT_CONF. Unanimous bearish ->
    soft SELL. Anything mixed (incl. doji-style equality) stays HOLD.

    2026-05-11 (Stage 2 Batch 2): introduced because the
    `compute_candlestick_signal` strong path abstains ~87% of the time
    when none of hammer / engulfing / doji / star matches. A 3-of-3
    body-direction agreement is a *weak* tie-breaker, not a pattern
    claim — hence the 0.15 soft conf and the 3/3 unanimity gate (no
    2/1 splits). df is the same OHLCV frame the strong path consumed.
    """
    if df is None or "close" not in df or "open" not in df:
        return "HOLD", 0.0
    try:
        if len(df) < lookback:
            return "HOLD", 0.0
        tail = df[["open", "close"]].iloc[-lookback:]
        closes = tail["close"].astype(float).to_numpy()
        opens = tail["open"].astype(float).to_numpy()
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    bull = int((closes > opens).sum())
    bear = int((closes < opens).sum())
    if bull == lookback:
        return "BUY", CANDLESTICK_DEAD_ZONE_SOFT_CONF
    if bear == lookback:
        return "SELL", CANDLESTICK_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


def _forecast_dead_zone_vote(df, forecast_indicators,
                             lookback=_FORECAST_SLOPE_LOOKBACK):
    """Return (vote, conf) when forecast voted HOLD but price+EMA align.

    Fallback secondary derivative: compare the slope of close prices to
    the slope of EMA21 over the last `lookback` bars. Both rising ->
    soft BUY; both falling -> soft SELL; mixed slopes -> HOLD.

    Critical guard: this branch ONLY fires when the forecast pipeline
    actually produced data. If `forecast_indicators` is empty / lacks
    Chronos output / has `models_disabled` or `error` keys, the soft
    branch is skipped entirely so we don't substitute slope-following
    for an unrun forecast. The user-visible contract: HOLD when the
    forecast model couldn't run, soft vote only when it ran but
    couldn't muster enough confidence.

    2026-05-11 (Stage 2 Batch 2): introduced because Chronos
    + accuracy gating force ~87% HOLD on metals (low confidence, vol
    gate, accuracy gate). When the forecast ran but didn't commit, a
    price+EMA21 slope-alignment read is a defensible weak tiebreaker.
    """
    if df is None or "close" not in df:
        return "HOLD", 0.0
    indicators = forecast_indicators or {}
    # Skip when the forecast pipeline didn't run at all. Any of these
    # markers indicates "no Chronos data" — we must not substitute.
    if indicators.get("models_disabled"):
        return "HOLD", 0.0
    if indicators.get("error"):
        return "HOLD", 0.0
    # Chronos is the live composite voter (Kronos is shadow per current
    # config). Require its 1h output to be present — that's the cheap
    # "did Chronos produce a forecast this cycle?" check.
    if indicators.get("chronos_1h_pct") is None:
        # Belt-and-suspenders: also accept chronos_ok flag if present.
        if not indicators.get("chronos_ok"):
            return "HOLD", 0.0
    try:
        if len(df) < lookback + 21:
            return "HOLD", 0.0
        close = df["close"].astype(float)
        ema21_series = close.ewm(span=21, adjust=False).mean()
        # Linear: last - value-`lookback`-bars-ago. Sign is what
        # matters, magnitude is informational.
        price_slope = float(close.iloc[-1] - close.iloc[-1 - lookback])
        ema21_slope = float(
            ema21_series.iloc[-1] - ema21_series.iloc[-1 - lookback]
        )
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    if price_slope > 0 and ema21_slope > 0:
        return "BUY", FORECAST_DEAD_ZONE_SOFT_CONF
    if price_slope < 0 and ema21_slope < 0:
        return "SELL", FORECAST_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


# Accuracy gate: signals with blended accuracy below this threshold are
# force-HOLD (treated like DISABLED_SIGNALS but dynamically). A signal at
# 44% is noise, not a reliable contrarian indicator — inverting it just
# produces different noise with whiplash as accuracy oscillates around 50%.
# 2026-04-11 (A-PR-batch-5): raised 0.45 → 0.47. The signal audit on
# 2026-04-10 found four signals sitting in the 45-47% band that the
# previous gate let through (volatility_sig 0.453, trend 0.454, etc.).
# Tightening the gate by 2pp removes ~4 coin-flip-adjacent signals from
# consensus while leaving the well-performing tier untouched.
ACCURACY_GATE_THRESHOLD = 0.47
ACCURACY_GATE_MIN_SAMPLES = 30  # need enough data before gating
# 2026-04-12: Tiered gate for high-confidence coin-flips. With 5000+ samples,
# a signal at 49.8% is coin-flip with p < 0.001 — no amount of waiting will
# fix it. Raising the gate to 50% for established signals removes structure
# (49.8%, 12K sam), heikin_ashi (49.6%, 23K sam) etc. while letting newer
# signals with <5000 samples prove themselves at the standard 47% threshold.
# 2026-04-16: raised high-sample min 5000 -> 10000. Investigation of W15/W16
# consensus collapse found the 5000 threshold catching signals during regime
# transitions where 5000 samples is too few to distinguish true coin-flip
# from transient degradation. 10000 samples reduces false-positive gating
# (e.g., a signal at 49.5% over 6000 samples may still have real edge in
# specific regimes that the aggregate accuracy hides).
_ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD = 0.50
_ACCURACY_GATE_HIGH_SAMPLE_MIN = 7000

# Directional accuracy gate: signals whose BUY or SELL accuracy is below this
# threshold get that direction force-HOLD'd while the other direction can still
# vote.  E.g., qwen3 BUY=30% (gated) but SELL=74.2% (votes normally).
# Uses the same min-samples threshold as the overall gate.
# 2026-04-10: raised from 0.35 → 0.40 to catch macro_regime BUY (38.9%),
# fibonacci SELL (35.9%), futures_flow both (36-37%).  Now with per-ticker
# directional data, this gate also works per-instrument.
_DIRECTIONAL_GATE_THRESHOLD = 0.40
_DIRECTIONAL_GATE_MIN_SAMPLES = 30

# Directional rescue (2026-04-28): when a signal fails the overall accuracy
# gate but its vote direction has strong accuracy, rescue it at reduced weight.
# E.g., heikin_ashi overall=42.6% (gated) but SELL=55.7% → rescue SELL vote.
# Only triggers when direction accuracy >= 55% with >= 30 samples, giving
# a 5pp safety margin above coin-flip.  Rescued signals get a 0.7x weight
# penalty so they contribute less than fully-passing signals.
_DIRECTIONAL_RESCUE_THRESHOLD = 0.55
_DIRECTIONAL_RESCUE_MIN_SAMPLES = 30
_DIRECTIONAL_RESCUE_WEIGHT_PENALTY = 0.70

# Adaptive recency blend: when recent accuracy diverges from all-time by more
# than this threshold, increase recent weight for faster regime adaptation.
# Normal: 70% recent + 30% all-time. Fast: 90% recent + 10% all-time.
# 2026-04-15: raised normal 0.70→0.75, fast 0.90→0.95 to better capture
# recent-regime signals like trend (40.3% alltime → 61.6% recent).
# 2026-04-16: REVERTED to 0.70/0.90. The 0.75/0.95 tuning amplified noise
# during the W12-W13 crash -> W14-W16 recovery transition. A 7-day window
# with only 170 samples was dominating a 10K-sample all-time baseline,
# triggering gates on signals whose "bad recent" was just the crash tail
# rolling through the window. 0.70/0.90 gives regime adaptation while
# leaving enough all-time anchor to damp single-week noise.
_RECENCY_DIVERGENCE_THRESHOLD = 0.15  # 15% absolute divergence triggers fast blend
_RECENCY_WEIGHT_NORMAL = 0.70
_RECENCY_WEIGHT_FAST = 0.90
_RECENCY_MIN_SAMPLES = 30  # match ACCURACY_GATE_MIN_SAMPLES (was 50 default)

# Crisis regime: when multiple macro-external signals are simultaneously
# degraded (recent accuracy < 35% with 50+ samples), the market is in a
# regime that breaks fundamental assumptions (e.g., wartime, systemic crisis).
# In crisis mode, apply extra penalty to trend-following signals and boost
# mean-reversion/calendar signals.
_CRISIS_THRESHOLD = 0.35  # signal accuracy below this counts as "broken"
_CRISIS_MIN_BROKEN = 3  # need at least 3 broken macro signals for crisis flag
_CRISIS_TREND_PENALTY = 0.6  # 0.6x weight for trend signals in crisis
_CRISIS_MR_BOOST = 1.3  # 1.3x weight for mean-reversion in crisis

# Directional bias penalty: signals with extreme BUY or SELL bias (>85% of
# their non-HOLD votes in one direction) get penalized because their
# "accuracy" may just reflect market drift rather than genuine edge.
# E.g., calendar (100% BUY) in a ranging-up market looks accurate by luck.
_BIAS_THRESHOLD = 0.85  # >85% BUY or >85% SELL triggers penalty
_BIAS_PENALTY = 0.5  # 0.5x weight for high-bias signals (85-95%)
_BIAS_EXTREME_THRESHOLD = 0.95  # >95% triggers stronger penalty
_BIAS_EXTREME_PENALTY = 0.2  # 0.2x weight for extreme-bias signals (>95%)
_BIAS_MIN_ACTIVE = 30  # need enough active (non-HOLD) votes to judge bias

# IC-based weight multiplier (2026-04-18): adjusts signal weight based on
# Information Coefficient — the rank correlation between a signal's votes and
# actual return magnitude. A signal with 55% accuracy but IC=0.15 catches big
# moves; one with 58% accuracy but IC=0.00 is riding market drift.
_IC_ALPHA = 2.0         # IC sensitivity: IC=0.10 → 1.20x boost
_IC_MULT_FLOOR = 0.6    # never zero out a signal via IC alone
_IC_MULT_CAP = 1.5      # cap to prevent IC from dominating
_IC_MIN_SAMPLES = 100   # need reliable IC estimate
_IC_STABILITY_MIN = 0.10  # minimum |ICIR| to trust the IC value
_IC_ZERO_PENALTY = 0.85   # phantom performers (|IC|<0.01, 500+ samples) get 0.85x
_IC_ZERO_MIN_SAMPLES = 500  # sample floor for zero-IC penalty
_IC_DATA_TTL = 3600     # IC cache TTL (matches ic_computation.py)

# Signal persistence filter (2026-04-20): require signals to maintain their
# vote for MIN_PERSISTENCE_CYCLES consecutive cycles before counting in
# consensus. Eliminates documented "single-check MACD/RSI/volume improvements
# are noise" pattern. Raw votes are still recorded for accuracy tracking —
# only the consensus input is filtered.
#
# Design: in-memory dict tracks {ticker: {signal: {"vote": X, "cycles": N}}}.
# When a signal's vote matches its previous non-HOLD vote, cycles increments.
# When it flips direction or goes HOLD→non-HOLD for the first time, cycles=1.
# Only signals with cycles >= _PERSISTENCE_MIN_CYCLES get their vote passed
# to consensus; others are treated as HOLD for consensus purposes only.
_PERSISTENCE_MIN_CYCLES = 2        # require 2+ consecutive same-direction votes
_PERSISTENCE_ENABLED = True        # toggle for easy disable
_PERSISTENCE_MAX_TICKERS = 32      # bound on tracked tickers (prod=5, cap guards tests/probes)
_persistence_state: dict[str, dict[str, dict]] = {}  # {ticker: {signal: {"vote": str, "cycles": int}}}
_persistence_lock = threading.Lock()

# 2026-05-11: per-asset relaxation. Metals + crypto run intraday; one cycle
# of confirmation already costs a minute. Stocks keep 2 cycles because
# market-hours-only and short windows make whipsaw more expensive.
_PERSISTENCE_CYCLES_BY_ASSET = {
    "METALS": 1,
    "CRYPTO": 1,
    "STOCK": 2,
}
def _persistence_cycles_for(ticker: str | None) -> int:
    """Return how many same-direction cycles a signal must hold to count."""
    if ticker is None:
        return _PERSISTENCE_MIN_CYCLES
    # Local import-style references — these names are defined at module scope
    # (imported from portfolio.tickers at the top of this file).
    if ticker in METALS_SYMBOLS:
        return _PERSISTENCE_CYCLES_BY_ASSET["METALS"]
    if ticker in CRYPTO_SYMBOLS:
        return _PERSISTENCE_CYCLES_BY_ASSET["CRYPTO"]
    return _PERSISTENCE_CYCLES_BY_ASSET["STOCK"]

# Cross-ticker consensus cache: stores the most recent consensus action per
# ticker so synthetic cross-asset signals can reference other tickers' results.
# Stale reads (MSTR processing before BTC in the same cycle) are acceptable —
# the 60s loop ensures data is at most one cycle old.
_cross_ticker_consensus: dict[str, dict] = {}  # {ticker: {"action": str, "confidence": float}}
_cross_ticker_lock = threading.Lock()


def _apply_persistence_filter(votes: dict[str, str], ticker: str | None) -> dict[str, str]:
    """Filter votes to only include signals that persisted for MIN_PERSISTENCE_CYCLES.

    Returns a new dict with non-persistent signals forced to HOLD.
    The original votes dict is not modified (needed for accuracy tracking).

    Cold-start: on the first cycle for a ticker (no prior state), all signals
    pass through unfiltered. Filtering only activates once we have history.
    """
    if not _PERSISTENCE_ENABLED or not ticker:
        return votes

    # 2026-05-11: resolve the per-asset persistence threshold once per call.
    # _persistence_cycles_for() returns 1 for metals+crypto, 2 for stocks,
    # 2 fallback for ticker=None.
    min_cycles = _persistence_cycles_for(ticker)

    with _persistence_lock:
        # Cold start: if we have NO history for this ticker, seed state and
        # pass all votes through. The filter only applies from cycle 2 onward.
        if ticker not in _persistence_state:
            if len(_persistence_state) >= _PERSISTENCE_MAX_TICKERS:
                evict_count = len(_persistence_state) // 2
                for old_key in list(_persistence_state)[:evict_count]:
                    del _persistence_state[old_key]
            _persistence_state[ticker] = {
                sig: {"vote": vote, "cycles": min_cycles if vote != "HOLD" else 0}
                for sig, vote in votes.items()
            }
            return votes  # first cycle — trust all signals

        ticker_state = _persistence_state[ticker]
        filtered = {}
        for sig, vote in votes.items():
            prev = ticker_state.get(sig)

            if vote == "HOLD":
                # Signal went quiet — reset persistence
                ticker_state[sig] = {"vote": "HOLD", "cycles": 0}
                filtered[sig] = "HOLD"
            elif prev is None or prev["vote"] != vote:
                # New direction or first appearance — start counting
                ticker_state[sig] = {"vote": vote, "cycles": 1}
                # 2026-05-11: when min_cycles == 1 (metals/crypto), a freshly
                # appearing non-HOLD vote already meets the threshold and
                # should be passed through immediately. Stocks keep the
                # original 2-cycle confirmation requirement.
                if min_cycles <= 1:
                    filtered[sig] = vote
                else:
                    filtered[sig] = "HOLD"
            else:
                # Same direction as previous cycle — increment
                prev["cycles"] += 1
                if prev["cycles"] >= min_cycles:
                    filtered[sig] = vote  # persistent — let it vote
                else:
                    filtered[sig] = "HOLD"  # still provisional

        return filtered


# Disabled signal per-ticker rescue (2026-04-18): signals in DISABLED_SIGNALS
# that have proven accuracy on specific tickers. These are re-enabled for
# compute+consensus on the listed ticker only. The standard accuracy gate
# (47%) still protects against degradation.
# Format: {(signal_name, ticker)} — if (sig, ticker) is in this set, the
# signal is computed and votes for that ticker despite being globally disabled.
# Evidence: data/disabled_signal_rescue_2026-04-18.json
_DISABLED_SIGNAL_OVERRIDES: frozenset[tuple[str, str]] = frozenset({
    # ml on ETH-USD: 55.1% at 3h (1206 samples). Globally disabled at 41.7%
    # because BTC-USD pulls it down to 26.4%. ETH-USD has genuine edge.
    ("ml", "ETH-USD"),
})

# Shadow-safe signals: disabled signals that are pure math (no network I/O)
# and safe to compute every cycle without impacting cycle time. Their votes
# are stored in _shadow_votes (not consensus votes) so outcome_tracker can
# track accuracy while they remain force-HOLD in consensus.
# Network-heavy disabled signals (futures_basis, vix_term_structure,
# gold_real_yield_paradox, cross_asset_tsmom, copper_gold_ratio,
# network_momentum, ovx_metals_spillover, xtrend_equity_spillover,
# orderbook_flow) are NOT shadow-safe — they do
# yfinance/FRED/Binance calls that would blow the 60s cycle budget.
_SHADOW_SAFE_SIGNALS = frozenset({
    "hurst_regime",
    "shannon_entropy",
    # "statistical_jump_regime" — RE-ENABLED 2026-04-29 (52.7% at 110 sam)
    "realized_skewness",
    "oscillators",
    # 2026-04-29: Added compute-only signals to accumulate accuracy data.
    # These use local OHLCV data only — no network calls.
    "complexity_gap_regime",
    "mahalanobis_turbulence",
    "crypto_evrp",
    "hash_ribbons",
    "fibonacci",  # newly disabled, shadow-track to confirm continued poor accuracy
    "calendar",  # 2026-05-09: disabled at 29.3%, shadow-track for recovery
    # 2026-05-11: Expanded shadow set — all pending-validation OHLCV-only signals.
    # residual_pair_reversion excluded (uses yfinance).
    "drift_regime_gate",
    "vol_ratio_regime",
    "williams_vix_fix",
    "intraday_seasonality",
    "cubic_trend_persistence",
    "vwap_zscore_mr",
    "gold_overnight_bias",
})

# Per-ticker consensus gate: BUG-164.  Suppress all non-HOLD consensus for
# tickers where the system's overall consensus is historically harmful.
# AMD 24.8%, GOOGL 31.3%, META 34.2% — actively wrong.
_PER_TICKER_CONSENSUS_GATE = 0.38  # below 38% = force HOLD
_PER_TICKER_CONSENSUS_MIN_SAMPLES = 50

# Voter-count circuit breaker (2026-04-16, Batch 2 of accuracy gating reconfig).
# When cascaded gates would leave fewer than _MIN_ACTIVE_VOTERS_SOFT active voters
# for a ticker, progressively relax the accuracy gate by _GATE_RELAXATION_STEP
# until the voter floor is met or _GATE_RELAXATION_MAX is reached. Rationale:
# losing voter diversity is worse than letting a borderline signal vote, because
# the consensus is a weighted sum of possibly-correlated signals — 3 correlated
# voters aren't as informative as 5 independent ones.
#
# Expected impact: kicks in during regime transitions where the 47% gate is
# silencing several voters whose recent accuracy dipped to 45-47%. Keeps at
# least 5 voters active by relaxing the gate by up to 6pp (to 41% floor).
# Signals with directional or per-ticker gating are NOT un-gated by this —
# only the overall accuracy gate is relaxed.
_MIN_ACTIVE_VOTERS_SOFT = 5
_GATE_RELAXATION_STEP = 0.02  # relax by 2pp per step
_GATE_RELAXATION_MAX = 0.02   # cap at 2pp below base gate (0.47 -> 0.45)

# Per-ticker signal disable: force HOLD for specific signal+ticker combos
# where accuracy data shows the signal is actively harmful for that instrument.
#
# 2026-04-16 (Batch 4): horizon-specific per-ticker blacklists via
# _TICKER_DISABLED_BY_HORIZON. Structure:
#   {"3h": {ticker: frozenset(bad_signals_at_3h)},
#    "1d": {ticker: frozenset(bad_signals_at_1d)},
#    "_default": {ticker: frozenset(bad_at_ALL_horizons)}}
#
# Compute-time (signal dispatch loop): uses the _default list only. Signals
# compute once per ticker per cycle and their vote is reused across horizons,
# so disabling at compute time requires the signal to be bad at EVERY horizon.
# Horizon-specific entries do NOT skip compute (the vote still exists, but
# is force-HOLD'd per-horizon at consensus time).
#
# Consensus-time: when building consensus for horizon H, apply
# (_default[ticker] | _TICKER_DISABLED_BY_HORIZON[H][ticker]).
#
# Why this structure: the Apr 14 MSTR blacklist was built from 3h accuracy
# data but applied globally, causing W15/W16 consensus collapse (1d dropped
# to 21.9%). 5 of 7 blacklisted MSTR signals were 66-81% accurate at 1d.
# Batch 1 trimmed the list to 2 entries; Batch 4 (this) enables per-horizon
# entries so future audits can say "bad at 3h, fine at 1d" without global
# penalty.
_TICKER_DISABLED_BY_HORIZON: dict[str, dict[str, frozenset]] = {
    # Disabled at ALL horizons — bad everywhere, safe to skip even at compute.
    "_default": {
        # 2026-04-15 audit: per-ticker 3h accuracy gating, retained pending
        # per-horizon audit of 1d/3d/5d behaviors.
        # 2026-04-24 after-hours audit: added structure (metals), credit_spread_risk
        # and macro_regime (XAU), ema (crypto/metals), futures_flow (crypto).
        "ETH-USD": frozenset({"news_event", "qwen3", "smart_money",
                              "ema",           # 17.6% 1d (51 sam)
                              "futures_flow",  # 32.6% 1d (675 sam)
                              }),
        "BTC-USD": frozenset({"smart_money", "heikin_ashi",
                              "futures_flow",  # 39.7% 1d (511 sam)
                              }),
        "XAG-USD": frozenset({"ministral", "credit_spread_risk",
                              "metals_cross_asset", "smart_money",
                              "structure",     # 29.9% 1d (723 sam)
                              "ema",           # 14.7% 1d (34 sam)
                              "sentiment",     # 33.3% 1d (285 sam), 94% BUY-only
                              }),
        "XAU-USD": frozenset({"ministral", "metals_cross_asset",
                              "structure",           # 30.4% 1d (827 sam)
                              "credit_spread_risk",  # 35.4% 1d (413 sam), 38.8% 3h — bad everywhere
                              "macro_regime",        # 34.3% 1d (484 sam)
                              }),
        # 2026-04-16: trimmed from 7 to 2 (Batch 1). Full history in commit
        # fd504d4. Kept: bad at both 3h (33.2%) and 1d (47.8%).
        "MSTR": frozenset({"claude_fundamental", "credit_spread_risk",
                          "statistical_jump_regime",  # 27.0% 1d (74 sam)
                          "realized_skewness",        # 36.0% 1d (50 sam)
                          # 2026-05-10: crashed 40-58pp after never-sell policy broken May 5
                          "sentiment",          # 90.4% -> 39.2%
                          "volume_flow",        # 82.3% -> 33.7%
                          "heikin_ashi",        # 78.6% -> 35.0%
                          "momentum_factors",   # 88.7% -> 30.4%
                          }),
    },
    # 2026-04-16 after-hours audit: signals that PASS global gate (>0.47)
    # but FAIL per-ticker (<0.45 with >=50 samples).
    # Source: accuracy_by_ticker_signal_cached() cross-referenced with
    # global accuracy. Each entry justified by per-ticker accuracy data:
    #
    # 3h: BTC volatility_sig 43.3%/342, bb 44.8%/536;
    #     ETH credit_spread_risk 43.5%/186;
    #     XAU credit_spread_risk 38.8%/170;
    #     XAG forecast 40.3%/248, qwen3 44.8%/413;
    #     MSTR volume 35.6%/1400, volatility_sig 42.9%/319.
    # 1d: BTC news_event 40.8%/671, forecast 42.0%/300;
    #     XAU candlestick 43.3%/656;
    #     MSTR ema 41.4%/1405, bb 44.5%/245.
    "3h": {
        # 2026-04-30 audit: added sentiment (33.8% 3h_recent, 3629 sam, 94.9% BUY).
        # Also added bb for more tickers, forecast for BTC (38.3% 3h_recent).
        "BTC-USD": frozenset({"volatility_sig", "bb",
                              "sentiment",  # 33.8% 3h_recent (3629 sam), 94.9% BUY-only
                              }),
        "ETH-USD": frozenset({"credit_spread_risk",
                              "sentiment",  # 33.8% 3h_recent (3629 sam), 94.9% BUY-only
                              }),
        # credit_spread_risk promoted to _default (2026-04-24)
        "XAU-USD": frozenset({"sentiment",  # 33.8% 3h_recent (3629 sam)
                              }),
        "XAG-USD": frozenset({"forecast", "qwen3",
                              "sentiment",  # 33.8% 3h_recent (3629 sam)
                              }),
        "MSTR": frozenset({"volume", "volatility_sig",
                           "sentiment",  # 33.8% 3h_recent (3629 sam)
                           }),
    },
    "4h": {},
    "12h": {},
    "1d": {
        # 2026-04-24 audit: added econ_calendar (1.8% BTC/ETH), ema (BTC),
        # funding (ETH 12.5%), econ_calendar (XAG 29.5%).
        # 2026-04-30 audit: added signals with <40% 1d_recent accuracy (50+ sam).
        # Key finding: systemic BUY bias — calendar (100% BUY, 30.8%),
        # claude_fundamental (99.3% BUY, 34.2%), momentum_factors (32.7%),
        # volume_flow (35.8%), heikin_ashi (38.2%), crypto_macro (33.8%).
        # These signals are already auto-gated by the blended accuracy gate,
        # but per-horizon blacklists provide defense-in-depth.
        "BTC-USD": frozenset({"news_event", "forecast",
                              "econ_calendar",       # 1.8% 1d (113 sam)
                              "ema",                 # 23.8% 1d (42 sam)
                              "claude_fundamental",  # 34.2% 1d_recent (730 sam), 99.3% BUY-only
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam), 60.1% at 3h — horizon divergence
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              "crypto_macro",        # 33.8% 1d_recent (476 sam)
                              "structure",           # 33.1% 1d_recent (758 sam)
                              }),
        "ETH-USD": frozenset({"econ_calendar",       # 1.8% 1d (113 sam)
                              "funding",             # 12.5% 1d (64 sam)
                              "claude_fundamental",  # 34.2% 1d_recent (730 sam), 99.3% BUY-only
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam)
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              "crypto_macro",        # 33.8% 1d_recent (476 sam)
                              "structure",           # 33.1% 1d_recent (758 sam)
                              }),
        "XAU-USD": frozenset({"candlestick",
                              "claude_fundamental",  # 2026-04-27: metals have no earnings/guidance
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam)
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              "smart_money",         # 34.2% 1d_recent (155 sam)
                              }),
        "XAG-USD": frozenset({"econ_calendar",       # 29.5% 1d (112 sam)
                              "claude_fundamental",  # 2026-04-27: metals have no earnings/guidance
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam)
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              }),
        "MSTR": frozenset({"ema", "bb",
                           "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                           "momentum_factors",    # 32.7% 1d_recent (910 sam)
                           "volume_flow",         # 35.8% 1d_recent (924 sam)
                           "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                           "smart_money",         # 34.2% 1d_recent (155 sam)
                           "structure",           # 33.1% 1d_recent (758 sam)
                           "macro_regime",        # 40.3% 1d (1475 sam) — moved from _default to preserve 3h
                           }),
    },
    "3d": {
        # 2026-05-10: signals with <45% accuracy at 3d horizon (global).
        # ministral 37.2% (6214 sam), credit_spread_risk 38.6% (1545 sam),
        # ema 39.9% (16662 sam) — all actively harmful at this horizon.
        "BTC-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "ETH-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "XAG-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "XAU-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "MSTR": frozenset({"ministral", "credit_spread_risk", "ema"}),
    },
    "5d": {
        # 2026-05-10: signals with <45% accuracy at 5d horizon.
        # funding 32.1% (728), news_event 42.2% (8251), ema 42.3% (15596),
        # credit_spread_risk 43.1% (1455), heikin_ashi 44.1% (24761).
        "BTC-USD": frozenset({"funding", "news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "ETH-USD": frozenset({"funding", "news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "XAG-USD": frozenset({"news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "XAU-USD": frozenset({"news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "MSTR": frozenset({"news_event", "ema",
                           "credit_spread_risk", "heikin_ashi"}),
    },
    "10d": {},
}


# P2-H (2026-04-17): module-load validation of _TICKER_DISABLED_BY_HORIZON
# shape. Catches structural errors (missing _default, invalid horizon keys,
# non-frozenset values) at import time rather than silently at runtime.
_VALID_HORIZON_KEYS = frozenset({"_default", "3h", "4h", "12h", "1d", "3d", "5d", "10d"})
assert "_default" in _TICKER_DISABLED_BY_HORIZON, (
    "_TICKER_DISABLED_BY_HORIZON missing required '_default' key")
for _k, _inner in _TICKER_DISABLED_BY_HORIZON.items():
    assert _k in _VALID_HORIZON_KEYS, (
        f"_TICKER_DISABLED_BY_HORIZON has unknown horizon key {_k!r}; "
        f"valid keys: {sorted(_VALID_HORIZON_KEYS)}")
    assert isinstance(_inner, dict), (
        f"_TICKER_DISABLED_BY_HORIZON[{_k!r}] must be a dict")
    for _tk, _sigs in _inner.items():
        assert isinstance(_sigs, frozenset), (
            f"_TICKER_DISABLED_BY_HORIZON[{_k!r}][{_tk!r}] must be a "
            f"frozenset (got {type(_sigs).__name__})")
del _k, _inner, _tk, _sigs


def _get_horizon_disabled_signals(ticker: str | None, horizon: str | None) -> frozenset:
    """Return signals to force-HOLD for (ticker, horizon). Union of default + horizon-specific.

    P3-1 (2026-04-17): uses .get('_default', {}) defensively instead of []
    subscript. If _default is ever removed at runtime (shouldn't happen —
    module-load assertion prevents it — but defensive), we return an empty
    set rather than crash the hot consensus path.
    """
    if not ticker:
        return frozenset()
    default_map = _TICKER_DISABLED_BY_HORIZON.get("_default", {})
    default_set = default_map.get(ticker, frozenset())
    if not horizon:
        return default_set
    horizon_set = _TICKER_DISABLED_BY_HORIZON.get(horizon, {}).get(ticker, frozenset())
    return default_set | horizon_set


# Backward-compat alias: the compute-time (signal dispatch) gate. Equal to the
# _default list — the minimum set of signals that are bad at every horizon.
# Existing callers reference this name; keep it as a view of _default.
_TICKER_DISABLED_SIGNALS = _TICKER_DISABLED_BY_HORIZON["_default"]


# --- Macro-window regime overlay (2026-04-28) ---
#
# When a high-impact macro event (FOMC/CPI/NFP) is within ~24h past or
# ~72h future, technical/sentiment signals trained on price-pattern
# continuity systematically misvote because price is being driven by
# news. The 2026-04-28 audit found 19 of 21 flagged per-ticker
# degradations real (sentiment 60.9%→42.7%, momentum_factors 44.5%→30.4%,
# structure 41.7%→30.0%, claude_fundamental 63.5%→38.9%) — coincident
# with the densest macro week of 2026.
#
# This overlay reduces the influence of those signals during the
# macro window, then auto-reverts when the window passes. It composes
# multiplicatively with the existing regime/horizon weight chain.
MACRO_WINDOW_DOWNWEIGHT_SIGNALS = frozenset({
    "sentiment", "momentum_factors", "structure",
})
MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER = 0.5

# claude_fundamental has a known >75% BUY bias caught by its own
# tier-bias gate. During macro windows the bias dominates because the
# 30-120min LLM cascade (Haiku/Sonnet/Opus) lags real-time regime
# shifts. Force-HOLD instead of down-weighting (stricter than the
# others — its accuracy collapses the most).
MACRO_WINDOW_FORCE_HOLD_SIGNALS = frozenset({"claude_fundamental"})

# 5-minute cache. econ_dates is hardcoded so the underlying data
# doesn't change between cycles, but we still pay an iteration over
# ECON_EVENTS each call. Caching avoids that hit per signal per ticker
# per cycle.
_MACRO_WINDOW_CACHE_TTL_S = 300


# --- Signal (full 32-signal for "Now" timeframe) ---

MIN_VOTERS_CRYPTO = 3  # crypto has 30 signals (8 core + 22 enhanced; ml disabled) — need 3
MIN_VOTERS_STOCK = 3  # stocks have 24-26 signals (7 core + 17-19 enhanced, GPU-dependent) — need 3
MIN_VOTERS_METALS = 2  # 2026-05-11: metals run at noisier intraday horizon
                       # (1m-1h target) where the standard 3-voter floor
                       # almost never fires after persistence filter.
                       # Empirical: XAG sees 5 raw voters → 2 post-persistence;
                       # MIN_VOTERS=3 produced 0 trades in 20 days.

# P2-F (2026-04-17 adversarial review): derived floors used by the
# circuit-breaker precondition. Placing here (after MIN_VOTERS_*) keeps the
# relationship explicit and prevents silent drift if the base MIN_VOTERS_*
# changes.
_MIN_VOTERS_BASE = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)
# Slate viability floor: the post-exclusion candidate count below which
# relaxation would produce a consensus thinner than any asset class's quorum.
_POST_EXCLUSION_MIN = _MIN_VOTERS_BASE
# Lone-signal escape floor: raised from 2 to _MIN_VOTERS_BASE (3) because a
# 2-voter relaxed consensus is still thinner than any asset class's outer
# quorum, so letting it emit trades was inconsistent with the system's
# design. Codex rounds 6-9 each flagged variants of this issue.
_LONE_SIGNAL_FLOOR = _MIN_VOTERS_BASE

# P2-G (2026-04-17): module-load assertions on constant relationships.
# These catch misconfigurations at import time rather than producing silent
# wrong behavior at runtime.
assert MIN_VOTERS_CRYPTO > 0 and MIN_VOTERS_STOCK > 0, (
    "MIN_VOTERS_* must be positive")
assert _POST_EXCLUSION_MIN <= _MIN_ACTIVE_VOTERS_SOFT, (
    f"_POST_EXCLUSION_MIN ({_POST_EXCLUSION_MIN}) must be <= "
    f"_MIN_ACTIVE_VOTERS_SOFT ({_MIN_ACTIVE_VOTERS_SOFT}); "
    f"otherwise the circuit breaker requires more candidates than it can "
    f"ever accept."
)
assert _GATE_RELAXATION_STEP > 0, (
    "_GATE_RELAXATION_STEP must be positive (else ZeroDivisionError in "
    "circuit-breaker step-count math).")
assert _GATE_RELAXATION_MAX > 0, "_GATE_RELAXATION_MAX must be positive."
assert (ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX) > _DIRECTIONAL_GATE_THRESHOLD, (
    f"Relaxed overall accuracy gate "
    f"({ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX:.2f}) must remain "
    f"above _DIRECTIONAL_GATE_THRESHOLD ({_DIRECTIONAL_GATE_THRESHOLD}); "
    f"otherwise directional gating becomes tighter than the relaxed "
    f"accuracy gate and the claim that the directional gate is NEVER "
    f"relaxed becomes meaningless."
)
# Step must divide max cleanly so iteration lands on the intended max.
_rel_ratio = _GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP
assert abs(_rel_ratio - round(_rel_ratio)) < 1e-9, (
    f"_GATE_RELAXATION_STEP ({_GATE_RELAXATION_STEP}) must divide "
    f"_GATE_RELAXATION_MAX ({_GATE_RELAXATION_MAX}) cleanly.")
del _rel_ratio

# Core signals that must have at least 1 active voter for non-HOLD consensus.
# Enhanced signals can strengthen/weaken but never create consensus alone.
CORE_SIGNAL_NAMES = frozenset({
    "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
    "volume", "ministral", "qwen3", "claude_fundamental",
})

# Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
_prev_sentiment: dict[str, str] = {}  # in-memory cache; seeded from sentiment_state.json on first call
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
        "mean_reversion": 0.6,  # fibonacci removed — disabled 2026-04-29
    },
    "trending-down": {
        "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
        # Enhanced: same as trending-up (trend signals work both ways)
        "trend": 1.4, "momentum_factors": 1.3, "heikin_ashi": 1.2,
        "structure": 1.2, "smart_money": 1.1,
        "mean_reversion": 0.6,  # fibonacci removed — disabled 2026-04-29
    },
    "ranging": {
        "rsi": 1.5, "bb": 1.5, "ema": 0.5,
        # 2026-04-05 audit: macd 58.7% recent (crossover catches range turns)
        "macd": 1.3,
        # Enhanced: boost mean-reversion and level-based signals
        # 2026-04-05 audit: fibonacci 68.2% recent — boost to 1.8 (was 1.6)
        # mean_reversion 65.4% recent — boost to 1.7 (was 1.5)
        # ministral 68.0% recent (Apr 5) — was 1.4x boost but collapsed to 41.5%
        # recent (Apr 26 audit, 41 sam). Removed boost, added to regime gate.
        "mean_reversion": 1.7, "calendar": 1.2,  # fibonacci removed — disabled 2026-04-29
        # 2026-04-05 audit: momentum 58.9% in ranging (2196 samples) — untapped edge
        "momentum": 1.3,
        # 2026-04-04: BUG-161 — oscillators 34-39% per-ticker in ranging.
        # Was 1.2x (boosted), now 0.3x (heavily penalized).
        "oscillators": 0.3,
        # 2026-04-28 research: actual degradation 50-70% (BUY acc 13-25%),
        # previous 0.5-0.7x was insufficient. Lowered to 0.3x.
        "trend": 0.3, "momentum_factors": 0.3, "heikin_ashi": 0.3,
        "structure": 0.4, "fear_greed": 0.3,
    },
    "high-vol": {
        "bb": 1.5, "volume": 1.3, "ema": 0.5,
        # Enhanced: boost volatility-aware and smart money signals
        # 2026-03-31: mean_reversion works in high-vol too (war overshoot/reversion)
        "volatility_sig": 1.4, "smart_money": 1.3, "volume_flow": 1.2,
        "candlestick": 1.2, "mean_reversion": 1.2,
        "trend": 0.6, "calendar": 0.7,
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
        # 2026-04-02 audit: 13 signals below 45% on 1d_recent. Gate the worst.
        # trend 40.7%, momentum_factors 41.4%, ema 40.8%, heikin_ashi 42.0%,
        # structure 36.1%, fear_greed 25.9%, macro_regime 30.3%,
        # news_event 29.5%, volatility_sig 35.0%, forecast 36.1%,
        # candlestick 44.5%, smart_money 39.6%.
        # The dynamic 45% accuracy gate also catches these, but explicit
        # regime gating is clearer and doesn't depend on blending math.
        "_default": frozenset({
            "trend", "momentum_factors", "ema", "heikin_ashi", "structure",
            "fear_greed", "macro_regime",
            # 2026-04-02: added based on 1d_recent audit
            "news_event", "volatility_sig", "forecast", "smart_money",
            # 2026-04-04: BUG-161/163 — oscillators 34-39% per-ticker,
            # candlestick 44.5% recent (292 sam). Both noise in ranging.
            "oscillators", "candlestick",
            # 2026-04-09: funding 29.9% at 1d (536 sam) but 74.2% at 3h (535 sam).
            # Gate at 1d, let it vote at 3h/4h.
            "funding",
            # 2026-04-12: econ_calendar 34.2% in ranging (1911 sam). SELL-only signal
            # is actively harmful in range-bound markets. 62.8% overall is inflated by
            # unknown-regime (86.8%, 2562 sam) and trending-up (25.9%) dominance.
            "econ_calendar",
            # 2026-04-26: volume_flow collapsed to 40.8% recent (-10.0pp, 1310 sam).
            # credit_spread_risk collapsed to 39.0% recent (-15.2pp, 249 sam).
            # ministral collapsed to 41.5% recent (41 sam) from 58.4% all-time.
            # All three are noise in the current 141h+ ranging regime.
            "volume_flow", "credit_spread_risk", "ministral",
            # 2026-04-27: claude_fundamental 40.5% at 1d_recent (1178 sam),
            # 78-83% BUY bias. Was only gated at 3h/4h but also harms 12h/1d/3d/5d.
            "claude_fundamental",
            # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), 33.8% at 3h.
            # BUY-only bias. Was gated at 3h/4h in all regimes but still active
            # at _default in ranging where it pushes false BUY consensus.
            "sentiment",
        }),
        # 3h: news_event 58.5%, smart_money 53.1% — decent at short horizons.
        # volatility_sig 47.2%, forecast 47.2% — marginal, let accuracy gate
        # handle them dynamically at 3h.
        # funding 74.2% at 3h (535 sam) — NOT gated here.
        # 2026-04-11: sentiment added — 33.8% at 3h_recent (3629 sam). The 0.5x
        # horizon weight is insufficient; this signal actively harms 3h consensus.
        # 2026-04-25: claude_fundamental added — 0 accuracy samples at 3h so
        # the accuracy gate defaults to 0.5 and passes the 47% gate. At 1d the
        # fast-blended accuracy is ~40% (correctly gated), but at 3h it escapes.
        # Fundamentals operate on hours/days timescale, not 3h. Sonnet/Opus have
        # 78-83% BUY bias (500-entry audit), so the ungateed BUY vote at 3h is
        # pure noise. Gate at 3h/4h in ranging; let it vote at 12h/1d/3d/5d.
        "3h": frozenset({"fear_greed", "macro_regime", "sentiment", "claude_fundamental"}),
        "4h": frozenset({"fear_greed", "macro_regime", "sentiment", "claude_fundamental"}),
    },
    "trending-up": {
        # BUG-152: SELL-biased signals have 0-11% accuracy in trending-up.
        # Gating at 1d prevents false SELL consensus during breakouts.
        # trend ~0%, ema ~11%, volume_flow ~10%, macro_regime 11.1%, momentum_factors low
        # claude_fundamental 5.9% trending-up (34 samples) — BUG-154
        # 2026-04-09: funding gated at 1d (29.9%), active at 3h (74.2%)
        # 2026-04-13: fear_greed 25.9% at 1d (170 sam) — destructive in ALL regimes at 1d
        "_default": frozenset({
            "trend", "ema", "volume_flow", "macro_regime",
            "momentum_factors", "claude_fundamental",
            "funding", "fear_greed",
        }),
        # mean_reversion 3h_recent=45.5% — gate on short horizons
        # SELL-biased signals work short-term even in uptrends — do NOT gate at 3h
        # 2026-04-13: sentiment 33.8% at 3h (3629 sam) — destructive at 3h in ALL regimes
        "3h": frozenset({"mean_reversion", "sentiment"}),
        "4h": frozenset({"mean_reversion", "sentiment"}),
    },
    "trending-down": {
        # BUG-155: bb 21.7% in trending-down (false reversal signals)
        # BUG-154: claude_fundamental 30.4% in trending-down
        # BUG-156: volume_flow (0%), macro_regime (0%), ema (0%), trend (0%)
        # on MSTR/PLTR in trending-down. These are SELL-biased and catastrophically
        # wrong when the downtrend classification is stale or stocks are recovering.
        # BUG-165: smart_money 10.0% in trending-down (130 samples) — worst signal
        # 2026-04-09: funding gated at 1d, active at 3h
        # 2026-04-13: fear_greed 25.9% at 1d (170 sam) — destructive in ALL regimes at 1d
        "_default": frozenset({
            "bb", "claude_fundamental",
            "volume_flow", "macro_regime", "ema", "trend", "heikin_ashi",
            "smart_money",  # BUG-165: 10.0% accuracy in trending-down
            "funding", "fear_greed",
            # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), BUY-only bias.
            # Was only gated at 3h/4h; actively harmful at longer horizons too.
            "sentiment",
        }),
        # 3h: trend signals may still work short-term; keep mean_reversion gated
        # 2026-04-13: sentiment 33.8% at 3h (3629 sam) — destructive at 3h in ALL regimes
        "3h": frozenset({"mean_reversion", "bb", "claude_fundamental", "sentiment"}),
        "4h": frozenset({"mean_reversion", "bb", "claude_fundamental", "sentiment"}),
    },
    "high-vol": {
        # 2026-04-09: funding gated at 1d, active at 3h
        # 2026-04-13: fear_greed 25.9% at 1d (170 sam) — destructive in ALL regimes at 1d
        "_default": frozenset({"funding", "fear_greed"}),
        # 2026-04-13: sentiment 33.8% at 3h (3629 sam) — destructive at 3h in ALL regimes
        "3h": frozenset({"sentiment"}),
        "4h": frozenset({"sentiment"}),
    },
    "unknown": {
        "3h": frozenset({"sentiment"}),
        "4h": frozenset({"sentiment"}),
    },
}


def _get_regime_gated(regime: str, horizon: str | None = None) -> frozenset[str]:
    """Get the set of signals to gate for a regime+horizon combination.

    Intentional semantics: horizon-specific override REPLACES `_default`,
    NOT unions with it. This is by design (BUG-149, 2026-03-29):
    `_default` lists signals that are bad at long horizons (1d/3d/5d) in
    a regime, while a horizon override (3h/4h) is the FINER-grained list
    of what should still be gated at that intraday horizon. Example:
    `trend` has 40.7% accuracy at 1d ranging (gate via _default) but
    61.6% at 3h ranging (allow via no-mention in 3h override).

    2026-05-02 audit: 04-24 P0-1 / 04-29 SC-P1-1 / 05-01 P0 (carryover)
    framed this as a "union bug" by analogy to `_get_horizon_disabled_signals`.
    Re-reading the docstring at line 762-767 and the per-signal comments
    in REGIME_GATED_SIGNALS confirms the intent is replace-semantics.
    Verified: funding 74.2% at 3h (in _default for 1d but excluded from
    3h override on purpose), trend 61.6% at 3h (same pattern).
    Finding REJECTED — leaving behavior unchanged. No fix needed.
    """
    regime_dict = REGIME_GATED_SIGNALS.get(regime, {})
    if not regime_dict:
        return frozenset()
    if horizon and horizon in regime_dict:
        return regime_dict[horizon]
    return regime_dict.get("_default", frozenset())

# Horizon-specific signal weight multipliers.
# Signals with >15pp accuracy divergence between horizons get adjusted.
# Updated: 2026-04-27 accuracy audit (3h_recent vs 1d_recent).
HORIZON_SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {
    "3h": {
        "news_event": 1.4,      # 70.0% at 3h_recent (1762 sam)
        "ema": 1.3,             # 62.9% at 3h (vs 48.6% at 1d)
        "ministral": 1.3,       # 62.6% at 3h (vs 42.4% at 1d) — boosted from 1.2
        "qwen3": 1.2,           # 61.8% at 3h
        "trend": 1.2,           # 61.6% at 3h (vs 37.7% at 1d)
        "volatility_sig": 1.2,  # 60.2% at 3h (304 sam)
        "momentum_factors": 1.2, # 60.1% at 3h (vs 35.4% at 1d)
        "momentum": 1.1,        # 56.1% at 3h (378 sam) — NEW 2026-04-27
        "heikin_ashi": 1.1,     # 55.0% at 3h (vs 42.7% at 1d) — NEW 2026-04-27
        "sentiment": 0.4,       # 33.8% at 3h — tightened from 0.5
        "forecast": 0.5,        # 38.3% at 3h
        "bb": 0.6,              # 41.7% at 3h (but 62.5% at 1d)
        "mean_reversion": 0.7,  # 45.5% at 3h (but 51.8% at 1d)
        "volume_flow": 0.7,     # 46.4% at 3h — NEW 2026-04-27
    },
    "4h": {
        "news_event": 1.4,
        "ema": 1.3,
        "ministral": 1.3,
        "qwen3": 1.2,
        "trend": 1.2,
        "volatility_sig": 1.2,
        "momentum_factors": 1.2,
        "momentum": 1.1,
        "heikin_ashi": 1.1,
        "sentiment": 0.4,
        "forecast": 0.5,
        "bb": 0.6,
        "mean_reversion": 0.7,
        "volume_flow": 0.7,
    },
    "1d": {
        "bb": 1.3,              # 62.5% at 1d_recent (120 sam) — boosted from 1.2
        "rsi": 1.1,             # 56.2% at 1d_recent (569 sam) — NEW 2026-04-27
        "credit_spread_risk": 1.1,  # 56.4% at 1d_recent (140 sam), SELL 77.9% — NEW 2026-04-27
        "volume": 1.1,          # 54.7% at 1d_recent (265 sam) — NEW 2026-04-27
        "macd": 1.1,            # 54.8% at 1d_recent (93 sam)
        "mean_reversion": 1.1,  # 51.8% at 1d_recent — reduced from 1.3
        "news_event": 1.4,      # 70.0% at 1d_recent (340 sam)! — was 0.5 (SELL-focused works now)
        "claude_fundamental": 0.5,  # 40.5% at 1d_recent (1178 sam) — NEW 2026-04-27 penalty
        "sentiment": 0.4,       # 40.1% at 1d_recent (202 sam) — NEW 2026-04-27
        "fear_greed": 0.4,      # 25.9% at 1d — still terrible
        "macro_regime": 0.5,    # 36.8% at 1d_recent
        "volatility_sig": 0.5,  # 45.5% at 1d_recent
        "structure": 0.5,       # 33.7% at 1d_recent — tightened from 0.6
        "forecast": 0.5,        # 44.6% at 1d_recent
        "ema": 0.5,             # 48.6% at 1d_recent — tightened from 0.6
        "trend": 0.5,           # 37.7% at 1d_recent — tightened from 0.6
        "heikin_ashi": 0.6,     # 42.7% at 1d_recent — tightened from 0.7
        "momentum_factors": 0.5, # 35.4% at 1d_recent — NEW 2026-04-27
        "volume_flow": 0.5,     # 40.0% at 1d_recent — NEW 2026-04-27
        "crypto_macro": 0.7,    # 46.9% at 1d_recent — NEW 2026-04-27
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
        cross_data = {
            sig: cross_sum[sig] / cross_count[sig]
            for sig in cross_sum
            if cross_count.get(sig, 0) > 0
        }

        # Compute multipliers
        weights = {}
        for sig, stats in this_data.items():
            samples = stats.get("total", 0)
            if samples < _DYNAMIC_HORIZON_MIN_SAMPLES:
                continue
            this_acc = stats.get("accuracy", 0.5)
            cross_acc = cross_data.get(sig)
            if cross_acc is None or not (0.01 <= cross_acc <= 1.0):
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
    # Codex 2026-05-10: _cached returns None on dogpile/timeout/error
    # paths (shared_state.py:88, 109, 123, 126). The previous bare cast
    # silenced the type but left ``signal_name in horizon_mults`` to
    # crash at runtime when None leaked through. Coerce to {} here so
    # the contract — "horizon weights are always a dict" — holds at the
    # boundary where the lie used to live.
    weights = _cached(cache_key, _DYNAMIC_HORIZON_WEIGHT_TTL,
                      lambda: _compute_dynamic_horizon_weights(horizon))
    return cast(dict[str, float], weights) if weights else {}


# Signals that only apply to specific asset classes
_CRYPTO_ONLY_SIGNALS = {"futures_flow", "funding", "crypto_macro", "onchain"}
_METALS_ONLY_SIGNALS = {"metals_cross_asset"}
_NON_STOCK_SIGNALS = {"orderbook_flow"}  # metals + crypto only


def _compute_applicable_count(ticker: str, skip_gpu: bool = False) -> int:
    """Compute total applicable signals for a ticker dynamically.

    Accounts for disabled signals, per-asset-class restrictions,
    and GPU signals skipped outside market hours.
    """
    is_crypto = ticker in CRYPTO_SYMBOLS
    is_metal = ticker in METALS_SYMBOLS
    is_stock = ticker in STOCK_SYMBOLS
    count = 0
    for sig in SIGNAL_NAMES:
        if sig in DISABLED_SIGNALS and (sig, ticker) not in _DISABLED_SIGNAL_OVERRIDES:
            continue
        # Per-ticker blacklist: check _default horizon for signals bad at all horizons
        if sig in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
            continue
        # crypto-only signals (futures_flow, funding, crypto_macro)
        if sig in _CRYPTO_ONLY_SIGNALS and not is_crypto:
            continue
        # metals-only signals (metals_cross_asset)
        if sig in _METALS_ONLY_SIGNALS and not is_metal:
            continue
        # non-stock signals (orderbook_flow — metals + crypto only)
        if sig in _NON_STOCK_SIGNALS and is_stock:
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


# Dynamic correlation group computation TTL and thresholds
_DYNAMIC_CORR_TTL = 7200  # 2h cache for dynamic correlation groups
# 2026-04-18: changed from Pearson r > 0.7 to agreement rate > 0.85.
# Pearson on vote encoding (BUY=1, HOLD=0, SELL=-1) was diluted by 70-90%
# HOLD dominance — max observed r=0.538 (ema↔trend), making the 0.7
# threshold unreachable and dynamic groups always falling back to static.
# Agreement rate only counts pairs where at least one signal voted non-HOLD.
_DYNAMIC_CORR_THRESHOLD = 0.85  # agreement rate threshold for clustering
_DYNAMIC_CORR_MIN_SAMPLES = 30  # minimum signal log entries for reliable correlation
_DYNAMIC_CORR_MIN_PAIRS = 20    # minimum non-HOLD pairs to trust agreement rate


def _compute_agreement_rate(votes_a, votes_b):
    """Compute agreement rate between two signal vote lists.

    Only counts pairs where at least one signal voted non-HOLD.
    Returns (agreement_rate, n_pairs) where n_pairs is the count of
    non-HOLD pairs.
    """
    agree = 0
    total = 0
    for va, vb in zip(votes_a, votes_b):
        if va == 0 and vb == 0:
            continue  # both HOLD — skip
        total += 1
        if va == vb:
            agree += 1
    if total == 0:
        return 0.0, 0
    return agree / total, total


def _compute_dynamic_correlation_groups() -> dict[str, frozenset[str]]:
    """Compute signal correlation groups from recent signal_log data.

    Uses agreement rate (not Pearson correlation) on non-HOLD vote pairs.
    Signals that agree > 85% of the time on non-HOLD votes are clustered.

    Falls back to static CORRELATION_GROUPS if insufficient data.
    """
    try:
        from datetime import datetime, timedelta

        from portfolio.accuracy_stats import load_entries
        entries = load_entries()
        cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        recent = [e for e in entries if e.get("ts", "") >= cutoff]
        if len(recent) < _DYNAMIC_CORR_MIN_SAMPLES:
            return _STATIC_CORRELATION_GROUPS

        # Build signal vote matrix: each row is a (entry, ticker) pair
        vote_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        from portfolio.tickers import SIGNAL_NAMES as _SN
        active_signals = [s for s in _SN if s not in DISABLED_SIGNALS]

        rows = []
        for entry in recent:
            for _tk, tdata in entry.get("tickers", {}).items():
                signals = tdata.get("signals", {})
                row = {s: vote_map.get(signals.get(s, "HOLD"), 0) for s in active_signals}
                rows.append(row)

        if len(rows) < _DYNAMIC_CORR_MIN_SAMPLES:
            return _STATIC_CORRELATION_GROUPS

        df = pd.DataFrame(rows)
        # Drop signals that are always HOLD (no variance)
        df = df.loc[:, df.std() > 0.01]
        if df.shape[1] < 3:
            return _STATIC_CORRELATION_GROUPS

        # Compute pairwise agreement rate (non-HOLD pairs only)
        from collections import defaultdict
        signal_to_group: dict[str, int] = {}
        groups: dict[int, set[str]] = defaultdict(set)
        next_group = 0

        sig_list = list(df.columns)
        for i, s1 in enumerate(sig_list):
            for j in range(i + 1, len(sig_list)):
                s2 = sig_list[j]
                agree_rate, n_pairs = _compute_agreement_rate(
                    df[s1].values, df[s2].values,
                )
                if n_pairs < _DYNAMIC_CORR_MIN_PAIRS:
                    continue
                if agree_rate > _DYNAMIC_CORR_THRESHOLD:
                    g1 = signal_to_group.get(s1)
                    g2 = signal_to_group.get(s2)
                    if g1 is None and g2 is None:
                        gid = next_group
                        next_group += 1
                        groups[gid] = {s1, s2}
                        signal_to_group[s1] = gid
                        signal_to_group[s2] = gid
                    elif g1 is not None and g2 is None:
                        groups[g1].add(s2)
                        signal_to_group[s2] = g1
                    elif g1 is None and g2 is not None:
                        groups[g2].add(s1)
                        signal_to_group[s1] = g2
                    elif g1 != g2:
                        # Merge groups. By elimination of the prior elifs both
                        # g1 and g2 are non-None here, but mypy can't narrow
                        # via ``!=``; assert makes the invariant explicit.
                        assert g1 is not None and g2 is not None
                        merged = groups[g1] | groups[g2]
                        groups[g1] = merged
                        del groups[g2]
                        for s in merged:
                            signal_to_group[s] = g1

        # Convert to named frozensets (only groups with 2+ members)
        result = {}
        for gid, members in groups.items():
            if len(members) >= 2:
                name = f"dynamic_{gid}"
                result[name] = frozenset(members)

        return result if result else _STATIC_CORRELATION_GROUPS

    except Exception:
        logger.debug("Dynamic correlation groups unavailable, using static", exc_info=True)
        return _STATIC_CORRELATION_GROUPS


def _get_correlation_groups() -> dict[str, frozenset[str]]:
    """Get current correlation groups, preferring dynamic over static.

    Codex 2026-05-10: ``_cached`` can return None on dogpile/error; the
    sole caller already does ``... or _STATIC_CORRELATION_GROUPS`` so a
    None leak isn't a runtime crash here, but the bare cast lied about
    the actual return type. Treat empty/None as "no dynamic groups
    available" so the type matches reality.
    """
    groups = _cached("dynamic_corr_groups", _DYNAMIC_CORR_TTL,
                     _compute_dynamic_correlation_groups)
    return cast(dict[str, frozenset[str]], groups) if groups else {}


# Static correlation groups (fallback when dynamic computation unavailable).
# Updated 2026-04-08: empirical audit of 200 recent signal_log entries.
_STATIC_CORRELATION_GROUPS = {
    # 2026-04-14: Measured correlation analysis (300 snapshots, 1308 obs):
    # volatility_sig only weakly correlates with volume (r=0.38). Oscillators
    # moved to trend_direction (0.463 with heikin_ashi, 83.4% agreement).
    # Structure moved to trend_direction (0.608 with trend, 96.5% with macro_regime).
    # RES-2026-04-21: REMOVED volatility_cluster. r=0.38 is too weak for a
    # correlation group. volume (52.1% acc) was unfairly penalized by
    # volatility_sig (46.8% acc). Let both vote independently.
    # 2026-04-30: SPLIT trend_direction mega-cluster (was 9 members, 1.96x weight).
    # At 0.12x per follower, one directional signal family dominated consensus.
    # Split into 3 semantically distinct sub-clusters:
    #   pure_trend (MA-based): trend/ema/heikin_ashi — 85-90% agreement
    #   oscillator_trend: macd/momentum_factors/oscillators — oscillation methods
    #   structural_flow: volume_flow/macro_regime/structure — market structure
    # Each sub-cluster gets independent leader selection, preventing a single
    # broken trend signal from poisoning all 9 members' direction.
    # Previous: 1.0 + 8*0.12 = 1.96x total. Now: 3 * (1.0 + 2*0.20) = 4.20x
    # total, but with 3 independent leaders — better captures disagreement
    # between trend, momentum, and structural signals.
    # 2026-05-07: trend disabled (46.1% 1d, 17880 sam), heikin_ashi borderline (49.3%)
    # but ema is the sole surviving member. Kept as group in case heikin_ashi re-enabled.
    "pure_trend": frozenset({"ema", "heikin_ashi"}),
    # 2026-05-07: macd disabled (44.2% 1d). momentum_factors is the leader.
    "oscillator_trend": frozenset({"momentum_factors", "oscillators"}),
    "structural_flow": frozenset({"volume_flow", "macro_regime", "structure"}),
    # 2026-04-18: Expanded from 3→6 members. Research (2026-04-17 after-hours)
    # found calendar↔fear_greed 100% agreement (501 sam), funding↔fear_greed
    # 100% (543 sam), news_event↔econ_calendar 100% (714 sam). These orphaned
    # signals were voting with full weight despite being completely redundant.
    "macro_external": frozenset({
        "fear_greed", "sentiment", "news_event",
        "calendar", "econ_calendar", "funding",
    }),
    # 2026-04-04: BUG-162 — candlestick-fibonacci correlation 0.708 on BTC.
    # fibonacci disabled 2026-04-29 (43.6%, 17K sam). Group dissolved —
    # candlestick now votes unclustered at full weight.
    # "pattern_based" removed: single-member groups are invalid.
    # 2026-04-26: bb removed from all clusters — now unclustered (full 1.0x weight).
    # BB thrives in ranging (+15.2pp to 69.5% recent) independently of ema/macd
    # (which are regime-gated). Correlation with ema/macd is superficial (BB bands
    # track MA) but BB's edge is overbought/oversold detection. Putting it in
    # trend_direction (0.12x) destroyed its edge; standalone cluster is semantically
    # wrong (1 member). Unclustered = full weight, which matches its value.
    # 2026-04-08: rsi+bb agree 100%, bb+mean_reversion 100%, bb+momentum 98.8%.
    # 2026-04-25: Moved bb to trend_direction, 2026-04-26: removed from all clusters.
    "momentum_cluster": frozenset({"mean_reversion", "rsi", "momentum"}),
    # 2026-04-13: claude_fundamental + crypto_macro agree 92-100%.
    # structure removed (now in trend_direction where correlations are stronger).
    "fundamental_cluster": frozenset({"claude_fundamental", "crypto_macro"}),
    # 2026-04-19: Measured correlation (50 snapshots): credit_spread_risk
    # agrees 100% with macro_regime in ETH/XAU, 100% with news_event in BTC.
    # futures_flow agrees 100% with credit_spread_risk + macro_regime in ETH.
    # Both were orphaned and getting full 1.0x weight despite redundancy.
    # Grouped together rather than in trend_direction to avoid inflating that
    # mega-cluster further (already 9 members at 0.12x).
    # 2026-05-07: futures_flow disabled (38.3% 1d). credit_spread_risk (53.7%)
    # now unclustered — single-member groups are invalid per convention.
    # credit_spread_risk votes at full 1.0x weight independently.
}
# Public alias for backward compatibility (used by tests and reporting)
CORRELATION_GROUPS = _STATIC_CORRELATION_GROUPS
_CORRELATION_PENALTY = 0.3  # secondary signals in a group get 30% of normal weight
# Per-cluster overrides: momentum_cluster signals agree 88-100% of the time.
# 2026-04-25: momentum_cluster now 3 members (bb moved to trend_direction).
# At 0.15x: 1.0 + 2*0.15 = 1.30x effective weight (was 1.45x with 4 members).
_CLUSTER_CORRELATION_PENALTIES: dict[str, float] = {
    "momentum_cluster": 0.15,
    # 2026-04-30: split trend_direction (9 members, 0.12x) into 3 sub-clusters.
    # Each has 3 members at 0.20x: effective weight per cluster = 1.0 + 2*0.20 = 1.40x.
    # Previous total: 1.96x from 1 cluster. New total: 3 * 1.40x = 4.20x BUT with
    # 3 independent leaders — momentum/structural can disagree with pure trend.
    "pure_trend": 0.20,
    "oscillator_trend": 0.20,
    "structural_flow": 0.20,
    # 2026-04-18: macro_external expanded from 3→6 members. At 0.15x per follower:
    # effective weight = 1.0 + 5*0.15 = 1.75x. Previously 3 members at 0.3x gave
    # 1.0 + 2*0.3 = 1.6x. Slightly higher total accounts for 3 truly independent
    # information sources (sentiment, calendar/econ timing, macro FG) being merged
    # with their highly-correlated variants.
    "macro_external": 0.15,
}

# Meta-clusters: groups of correlation sub-clusters whose LEADERS frequently
# agree (100% cross-cluster leader agreement measured 2026-05-01, 20 snapshots).
# When all meta-cluster leaders vote the same direction, only the highest-accuracy
# leader retains full weight; others get the meta-cluster penalty.
# This prevents the trend mega-view from getting 3.0x effective leader weight
# when pure_trend, oscillator_trend, and structural_flow leaders vote identically.
# When leaders DISAGREE, no penalty is applied — that's informative diversity.
_META_CLUSTER_GROUPS: dict[str, list[str]] = {
    "trend_mega": ["pure_trend", "oscillator_trend", "structural_flow"],
}
_META_CLUSTER_PENALTY = 0.35  # 2nd/3rd agreeing leaders get 35% weight


def _safe_accuracy(value, default):
    """Coerce an accuracy value to a clean float, mapping None/NaN/inf to `default`.

    2026-04-17 (P1-C): the live consensus path previously crashed with
    TypeError when `accuracy_data[sig]` held explicit None (e.g., from a
    half-written cache), and with a silent fall-through-as-valid when it
    held NaN (every comparison with NaN is False). This helper normalizes.
    """
    import math
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _safe_sample_count(value):
    """Coerce a sample count to a non-negative int; None/NaN/negative -> 0."""
    import math
    if value is None:
        return 0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(f) or math.isinf(f) or f < 0:
        return 0
    return int(f)


def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
                                  base_gate, relaxation):
    """Count how many signals would pass gating at gate=(base_gate - relaxation).

    Counts only voters that survive the full gate cascade:
      1) excluded (top-N)
      2) group-gated (correlation leader below group-leader gate)
      3) accuracy gate at (base - relaxation), tiered for high-sample signals
      4) directional gate (unchanged by relaxation)

    Returns int — the number of signals still voting BUY/SELL.
    """
    gate_val = base_gate - relaxation
    # SC-P1-2 (2026-05-02 adversarial follow-ups): high-sample tier is NOT
    # relaxed. A signal with 10K+ samples at sub-50% accuracy has measurable
    # negative edge — circuit-breaker relaxation must not promote it back to
    # voting. Standard tier (under 10K samples) still relaxes so borderline
    # newer signals can be rescued during regime transitions. Must mirror
    # the same logic in `_weighted_consensus` (line ~2068).
    high_gate_val = _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD
    active = 0
    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        if signal_name in excluded:
            continue
        if signal_name in group_gated:
            continue
        stats = accuracy_data.get(signal_name) or {}
        # P1-C (2026-04-17 adversarial review): coerce None / NaN values to
        # safe defaults. The live path previously crashed with TypeError when
        # `accuracy_data[sig]` contained explicit None values (e.g., from a
        # half-written cache). Replay had `except TypeError` but live didn't.
        acc = _safe_accuracy(stats.get("accuracy"), default=0.5)
        samples = _safe_sample_count(stats.get("total"))
        effective_gate = gate_val
        if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
            effective_gate = max(gate_val, high_gate_val)
        if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
            continue
        # Directional gate is not relaxed by the circuit breaker — those gates
        # catch signals that are actively wrong in one direction.
        if vote == "BUY":
            dir_acc = _safe_accuracy(stats.get("buy_accuracy"), default=acc)
            dir_n = _safe_sample_count(stats.get("total_buy"))
        else:
            dir_acc = _safe_accuracy(stats.get("sell_accuracy"), default=acc)
            dir_n = _safe_sample_count(stats.get("total_sell"))
        if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
            continue
        active += 1
    return active


def _normalize_regime(regime):
    """P2-D (2026-04-17): normalize regime strings to a canonical lowercase form.

    Protects against case/typo variants ("TRENDING-UP", " trending-up ",
    "trending_up") that would otherwise silently fall through to the
    strictest-quorum default. Returns None unchanged.
    """
    if regime is None:
        return None
    if not isinstance(regime, str):
        return regime  # Let downstream default handle non-strings.
    normalized = regime.strip().lower().replace("_", "-")
    # Common alias fixups.
    if normalized in ("trendingup", "trending"):
        normalized = "trending-up"
    elif normalized == "trendingdown":
        normalized = "trending-down"
    elif normalized in ("highvol", "high-volatility", "high_vol"):
        normalized = "high-vol"
    return normalized


def _dynamic_min_voters_for_regime(regime):
    """Regime-dependent final quorum. Single source of truth - called by both
    the circuit breaker and apply_confidence_penalties.

    This is the minimum voter count the OUTER consensus path requires before
    emitting a non-HOLD action. The circuit breaker uses it to size its
    recovery floor so relaxation is only engaged when it could reach the
    regime's actual quorum.

    2026-04-17 (P2-C/P2-D): de-duplicated. apply_confidence_penalties
    previously had an inline copy at line ~1623 that had to stay in lockstep
    manually - now it calls this helper. Also accepts case/typo-variant
    regime strings via _normalize_regime.
    """
    canonical = _normalize_regime(regime)
    if canonical in ("trending-up", "trending-down"):
        return 3
    if canonical == "high-vol":
        return 4
    return 5  # ranging, unknown, None


def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
                              regime=None):
    """Compute circuit-breaker relaxation to preserve voter diversity.

    Progressively tests relaxation values 0, step, 2*step, ..., up to
    _GATE_RELAXATION_MAX. Returns the smallest relaxation that yields at
    least _MIN_ACTIVE_VOTERS_SOFT active voters.

    Decision tree:
      - baseline >= floor                    -> 0.0 (no relaxation needed)
      - best_possible <= baseline            -> 0.0 (relaxation doesn't help;
                                                either a low-signal scenario
                                                or a genuine regime break
                                                where remaining signals are
                                                below even the 41% relaxed
                                                gate — letting them vote
                                                would be wrong)
      - best_possible >= floor               -> smallest step that meets floor
      - baseline < best_possible < floor     -> _GATE_RELAXATION_MAX (partial
                                                recovery: a single
                                                irrecoverable outlier must
                                                not veto relaxation for the
                                                rest - Codex P2 fix)

    Uses `_count_active_voters_at_gate` which applies directional gating,
    so signals gated on BUY-accuracy=30% don't inflate the decision.

    Returns float - relaxation in absolute accuracy points (e.g., 0.02).
    """
    # Defensive: caller may pass None for either set (older paths or a future
    # refactor). Treat as empty to avoid `in None` TypeErrors in a hot path.
    excluded = excluded or set()
    group_gated = group_gated or set()

    # Three guards, in increasing strictness, all applied:
    #
    #   Guard A (raw vs regime quorum):
    #     Matches downstream's `apply_confidence_penalties` which checks
    #     `extra_info["_voters"]` against dynamic_min. `_voters` is raw
    #     non-HOLD count post-regime, pre top-N/group-gate, so this check
    #     must NOT subtract `excluded` or `group_gated`.
    #
    #   Guard B (post-exclusion slate viability):
    #     Downstream's raw `_voters` doesn't account for top-N or
    #     correlation-group exclusions. If the POST-exclusion slate is
    #     below MIN_VOTERS_BASE (3) — the floor across all asset classes —
    #     a relaxed consensus would be built from a too-thin slate even
    #     though downstream would accept the raw count. Codex round 9
    #     (2026-04-17) caught this with a 3-signal correlation cluster
    #     gated out, leaving only 2 voters to drive consensus.
    #
    #   Guard C (lone-signal escape):
    #     Even with a large post-exclusion slate, directional gating can
    #     leave a single accuracy-passing signal. `best_possible >= 2`
    #     catches this case.
    min_regime_quorum = _dynamic_min_voters_for_regime(regime)
    raw_candidates = sum(1 for v in votes.values() if v != "HOLD")
    if raw_candidates < min_regime_quorum:
        return 0.0

    # P2-F (2026-04-17): derived from MIN_VOTERS_CRYPTO/STOCK rather than
    # hardcoded. If the base quorum changes, this follows automatically.
    post_exclusion_candidates = sum(
        1 for sn, v in votes.items()
        if v != "HOLD" and sn not in excluded and sn not in group_gated
    )
    if post_exclusion_candidates < _POST_EXCLUSION_MIN:
        return 0.0

    baseline = _count_active_voters_at_gate(
        votes, accuracy_data, excluded, group_gated, base_gate, 0.0,
    )
    if baseline >= _MIN_ACTIVE_VOTERS_SOFT:
        return 0.0

    best_possible = _count_active_voters_at_gate(
        votes, accuracy_data, excluded, group_gated,
        base_gate, _GATE_RELAXATION_MAX,
    )

    # Lone-signal escape guard. Even when raw candidates meet the downstream
    # quorum, directional gating can leave a thin set of recoverable voters.
    # P2-A (2026-04-17): raised from 2 to MIN_VOTERS_BASE (3). A 2-voter
    # "consensus" is still exposure-worthy in trending markets where
    # dynamic_min=3 — but any relaxation that only recovers 2 voters from a
    # large slate is catching signals that the downstream quorum would
    # accept as a weak consensus. Require at least as many as the base
    # MIN_VOTERS_* to avoid creating "relaxed" sub-quorum consensuses.
    if best_possible < _LONE_SIGNAL_FLOOR:
        return 0.0

    # Regime break: relaxation recovers nothing beyond baseline. Keep the
    # strict gate so the event shows up in logs rather than silently opening
    # to sub-41% signals.
    if best_possible <= baseline:
        return 0.0

    # Integer steps up to and including max - use int steps to avoid float drift.
    n_steps = int(round(_GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP))
    for i in range(1, n_steps + 1):
        candidate_rel = round(i * _GATE_RELAXATION_STEP, 6)
        active = _count_active_voters_at_gate(
            votes, accuracy_data, excluded, group_gated, base_gate, candidate_rel,
        )
        if active >= _MIN_ACTIVE_VOTERS_SOFT:
            return candidate_rel
    # Partial-recovery case (Codex P2 fix): best_possible > baseline but
    # still < floor. A single irrecoverable outlier shouldn't veto recovery
    # of the recoverable majority - apply max relaxation to get as many
    # voters back as possible. Logs still carry the relaxation value so
    # operators can distinguish this from a clean relaxation-to-floor.
    return _GATE_RELAXATION_MAX


# ---------------------------------------------------------------------------
# IC-based weight multiplier (2026-04-18)
# ---------------------------------------------------------------------------

def _compute_ic_mult(ic: float, icir: float, samples: int) -> float:
    """Compute IC-based weight multiplier for a signal.

    Returns a multiplicative adjustment based on the signal's Information
    Coefficient:
    - IC > 0 with stable ICIR → boost (catches big moves)
    - IC ≈ 0 with many samples → slight penalty (phantom performer)
    - IC < 0 with stable ICIR → penalty (contrarian, accuracy gate handles)
    - Insufficient data or unstable → 1.0 (no adjustment)

    Clamped to [_IC_MULT_FLOOR, _IC_MULT_CAP].
    """
    if samples < _IC_MIN_SAMPLES:
        return 1.0
    # Zero-IC penalty for phantom performers: signals with many samples but
    # no return-magnitude predictive power (e.g., calendar, econ_calendar).
    if abs(ic) < 0.01 and samples >= _IC_ZERO_MIN_SAMPLES:
        return _IC_ZERO_PENALTY
    if abs(icir) < _IC_STABILITY_MIN:
        return 1.0
    raw = 1.0 + _IC_ALPHA * ic
    return max(_IC_MULT_FLOOR, min(_IC_MULT_CAP, raw))


# IC data cache: reuse ic_computation.py infrastructure with in-memory TTL.
_ic_data_cache: dict = {}
_ic_data_lock = threading.Lock()


def _get_ic_data(horizon: str) -> dict | None:
    """Load IC data for the given horizon, computing if cache is stale.

    Returns the full cache dict {"global": {...}, "per_ticker": {...}}
    or None if IC data is unavailable.
    """
    now = time.time()
    with _ic_data_lock:
        cached = _ic_data_cache.get(horizon)
        if cached and now - cached.get("_loaded_at", 0) < _IC_DATA_TTL:
            return cast(dict[Any, Any], cached)

    try:
        from portfolio.ic_computation import compute_and_cache_ic, load_cached_ic
        cache = load_cached_ic(horizon)
        if cache is None:
            cache = compute_and_cache_ic(horizon)
        if cache:
            cache["_loaded_at"] = now
            with _ic_data_lock:
                _ic_data_cache[horizon] = cache
            return cast(dict[Any, Any], cache)
    except Exception:
        logger.info("IC data unavailable for horizon=%s — signals will use default 1.0x weight", horizon)
    return None


_macro_window_cache: dict = {"value": False, "ts": 0.0}
_macro_window_cache_lock = threading.Lock()
_macro_window_last_state: dict = {"active": None}  # transition logger


def _is_macro_window_cached(now_ts: float | None = None) -> bool:
    """Return whether we're inside a macro event window, with TTL caching.

    The underlying ``portfolio.econ_dates.is_macro_window`` iterates
    ``ECON_EVENTS`` linearly. That's cheap, but called per signal per
    ticker per cycle becomes wasteful. Cache the result for
    ``_MACRO_WINDOW_CACHE_TTL_S`` (5 minutes by default) — events have
    hourly cadence at fastest, so 5min staleness is acceptable.

    Logs once per state transition so the operational log shows when
    we entered/exited a macro window without spamming every cycle.
    """
    import time as _time
    if now_ts is None:
        now_ts = _time.time()
    with _macro_window_cache_lock:
        if now_ts - _macro_window_cache["ts"] < _MACRO_WINDOW_CACHE_TTL_S:
            return bool(_macro_window_cache["value"])
        try:
            from portfolio.econ_dates import is_macro_window
            active = bool(is_macro_window())
        except Exception as e:
            logger.warning("macro window detection failed (treating as inactive): %s", e)
            active = False
        _macro_window_cache["value"] = active
        _macro_window_cache["ts"] = now_ts
        last = _macro_window_last_state["active"]
        if last is None:
            _macro_window_last_state["active"] = active
        elif last != active:
            logger.info(
                "macro_window state transition: %s -> %s",
                "ACTIVE" if last else "inactive",
                "ACTIVE" if active else "inactive",
            )
            _macro_window_last_state["active"] = active
        return active


def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
                        accuracy_gate=None, max_signals=None, horizon=None,
                        regime_gated_override=None, ticker=None,
                        soft_confidences=None):
    """Compute weighted consensus using accuracy, IC, regime, and activation frequency.

    Weight per signal = accuracy_weight * ic_mult * regime_mult * normalized_weight
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

    2026-05-11 (Codex Fix B) — soft-confidence dampening:
    The Stage 2 dead-zone helpers (EMA / BB / MACD) emit *weak* directional
    votes when the strong path would HOLD, and stash a small per-vote
    confidence (0.15-0.20) into extra_info under the keys
    "_soft_conf_ema" / "_soft_conf_bb" / "_soft_conf_macd". Without
    propagation, _weighted_consensus treated those soft votes as full-
    strength votes (just direction × accuracy weight), so an all-soft
    slate could produce full directional confidence — defeating the
    "weak weight" contract. We now scale each soft vote's contribution
    by its soft_conf, so e.g. 3 × 0.18 ≈ 0.54 < 1.0 (a single strong
    vote). Pass the soft_confidences dict to opt in; strong votes (no
    key present) keep their original weight × accuracy × regime mult.
    """
    soft_confidences = soft_confidences or {}
    gate = accuracy_gate if accuracy_gate is not None else ACCURACY_GATE_THRESHOLD
    buy_weight = 0.0
    sell_weight = 0.0
    gated_signals = []
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    horizon_mults = _get_horizon_weights(horizon)

    # Codex round 10/11/12 (2026-04-17 follow-up): deep-sanitize accuracy_data
    # at function entry.
    #   Round 10: coerced non-dict container values to {}.
    #   Round 11: found dict values with poisoned numeric fields still
    #             crashed. Added per-field coercion.
    #   Round 12: coerce-with-0.5-default silently promoted partially-
    #             written cache rows ({"accuracy": null, "total": 200}) into
    #             mature 50% signals that cleared the min-samples gate.
    #             Now: if a numeric field is poisoned, DROP that field so
    #             downstream `.get(..., default)` falls back cleanly. A row
    #             whose overall accuracy is poisoned but total=200 becomes
    #             {"total": 200} - the gate sees no accuracy, the downstream
    #             code default to the safe fallback. The row no longer
    #             masquerades as a 50%-accurate mature signal.
    # Codex round 13 (2026-04-17): a poisoned accuracy must invalidate its
    # PAIRED sample count too. Otherwise `{"accuracy": None, "total": 200}`
    # becomes `{"total": 200}` which downstream still reads as a mature
    # 50% signal (accuracy defaults to 0.5, samples=200 clears the gate).
    # Drop-together semantics: overall acc poisoned -> drop (accuracy, total);
    # buy_accuracy poisoned -> drop (buy_accuracy, total_buy); likewise for
    # sell. Fields whose pair is clean but themselves clean pass through.
    import math as _math

    def _coerce_sample_count(val):
        """Return int >= 0, or None if val is missing/poisoned/invalid."""
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if _math.isnan(f) or _math.isinf(f) or f < 0:
            return None
        return int(f)

    _PAIRED = (
        ("accuracy", "total"),
        ("buy_accuracy", "total_buy"),
        ("sell_accuracy", "total_sell"),
    )
    if accuracy_data:
        _sanitized: dict[Any, dict[str, Any]] = {}
        for _k, _v in accuracy_data.items():
            if not isinstance(_v, dict):
                _sanitized[_k] = {}
                continue
            _clean = dict(_v)  # start from a copy, then prune.
            for _acc_key, _cnt_key in _PAIRED:
                _acc_has = _acc_key in _clean
                _cnt_has = _cnt_key in _clean
                if _acc_has:
                    _clean_acc = _safe_accuracy(_clean.get(_acc_key), default=None)
                else:
                    _clean_acc = None
                if _cnt_has:
                    _clean_cnt = _coerce_sample_count(_clean.get(_cnt_key))
                else:
                    _clean_cnt = None
                # Decide whether to keep each field:
                #   Both clean      -> keep both.
                #   Only acc clean  -> keep acc; drop cnt (if it was present-and-poisoned).
                #   Only cnt clean  -> drop BOTH (count without trustworthy accuracy
                #                      must not promote the row to a mature signal).
                #   Neither clean   -> drop both.
                if _clean_acc is not None and _clean_cnt is not None:
                    _clean[_acc_key] = _clean_acc
                    _clean[_cnt_key] = _clean_cnt
                elif _clean_acc is not None and not _cnt_has:
                    # Accuracy present (clean), count field absent - keep acc.
                    _clean[_acc_key] = _clean_acc
                else:
                    # Poisoned accuracy OR poisoned count: drop both so the
                    # row doesn't masquerade as a mature signal. Downstream
                    # .get() calls then use their safe defaults.
                    _clean.pop(_acc_key, None)
                    _clean.pop(_cnt_key, None)
            _sanitized[_k] = _clean
        accuracy_data = _sanitized
    else:
        accuracy_data = {}

    # Regime gating: force-HOLD signals that produce negative alpha in this regime.
    # BUG-149: now horizon-aware — e.g., trend works at 3h in ranging (61.6%)
    # SC-I-001: when caller provides regime_gated_override (with BUG-158 per-ticker
    # exemptions already applied), use it instead of recomputing from scratch.
    regime_gated = regime_gated_override if regime_gated_override is not None else _get_regime_gated(regime, horizon)
    votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}

    # Horizon-specific per-ticker blacklist (2026-04-16, Batch 4). Extends the
    # compute-time _default blacklist with horizon-specific entries. Compute time
    # can't see horizon (one vote reused across 3h/4h/12h/1d/3d/5d/10d consensus),
    # so per-horizon gating must happen here.
    horizon_disabled = _get_horizon_disabled_signals(ticker, horizon)
    if horizon_disabled:
        votes = {k: ("HOLD" if k in horizon_disabled else v) for k, v in votes.items()}

    # Macro-window force-HOLD pre-pass (2026-04-28). When a high-impact
    # event is within ~24h past or ~72h future, force-HOLD the signals
    # whose lag/bias makes them dominantly wrong in news-driven regimes.
    # The downweight branch for the other macro-fragile signals lives in
    # the weight loop below so it composes with regime/horizon multipliers.
    macro_active = _is_macro_window_cached()
    if macro_active and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
        votes = {
            k: ("HOLD" if k in MACRO_WINDOW_FORCE_HOLD_SIGNALS else v)
            for k, v in votes.items()
        }

    # Top-N gate: only let the top max_signals (by accuracy) participate.
    # Codex round 2 P2 (2026-04-28): rank with macro-adjusted accuracy so
    # downweighted signals lose Top-N slots to healthier peers during a
    # macro window. Without this, sentiment can keep its slot at full
    # raw accuracy and exclude a peer that would have voted more reliably.
    def _topn_accuracy_key(s: str) -> float:
        base = float(accuracy_data.get(s, {}).get("accuracy", 0.5))
        if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
            base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        return base

    active_votes = {k: v for k, v in votes.items() if v != "HOLD"}
    if max_signals and len(active_votes) > max_signals:
        ranked = sorted(
            active_votes.keys(),
            key=_topn_accuracy_key,
            reverse=True,
        )
        excluded = set(ranked[max_signals:])
    else:
        excluded = set()

    # Pre-compute which signal is the "leader" (highest accuracy) in each
    # correlation group, considering only signals that are actively voting.
    # Prefer dynamic groups (from signal_log correlations) over static.
    active_non_hold = {s for s, v in votes.items() if v != "HOLD"}
    _active_corr_groups = _get_correlation_groups() or _STATIC_CORRELATION_GROUPS

    # Codex P2 (2026-04-28): apply the macro-window downweight to the
    # leader-selection key BEFORE picking the leader. Otherwise sentiment
    # (lifetime ~70% acc) stays leader of macro_external during a macro
    # window — and the 0.15x follower penalty pushes healthier peers
    # below sentiment's already-halved weight, making the overlay
    # actively reinforce the wrong signal.
    def _leader_accuracy_key(s: str) -> float:
        base = float(accuracy_data.get(s, {}).get("accuracy", 0.5))
        if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
            base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        return base

    group_leaders = {}
    for group_name, group_sigs in _active_corr_groups.items():
        active_in_group = active_non_hold & group_sigs
        if len(active_in_group) <= 1:
            continue
        best_sig = max(active_in_group, key=_leader_accuracy_key)
        group_leaders[group_name] = best_sig

    # Correlation group leader gating: when the best signal in a group has
    # accuracy below threshold (with enough samples), gate the ENTIRE group.
    # Prevents the "least bad" broken signal from voting.
    # 2026-03-31: macro_external group (fear_greed 25.9%, sentiment 46.8%,
    # news_event 29.5%) — even the leader is near noise.
    # 2026-04-06: Lowered from 0.47 → 0.46 to catch borderline cases where
    # sentiment (blended ~46.4%) barely escapes as group leader.
    _GROUP_LEADER_GATE_THRESHOLD = 0.46
    group_gated_signals: set[str] = set()
    for group_name, group_sigs in _active_corr_groups.items():
        leader = group_leaders.get(group_name)
        if leader:
            leader_stats = accuracy_data.get(leader, {})
            leader_acc = leader_stats.get("accuracy", 0.5)
            leader_samples = leader_stats.get("total", 0)
            if leader_samples >= ACCURACY_GATE_MIN_SAMPLES and leader_acc < _GROUP_LEADER_GATE_THRESHOLD:
                group_gated_signals.update(group_sigs & active_non_hold)
                logger.debug(
                    "Correlation group %s gated: leader %s at %.1f%% < %.0f%% threshold",
                    group_name, leader, leader_acc * 100, _GROUP_LEADER_GATE_THRESHOLD * 100,
                )

    # Build a mapping of signal → correlation penalty (per-cluster override).
    # When a signal is in multiple groups, use the harshest (lowest) penalty.
    penalized_signals: dict[str, float] = {}
    for group_name, group_sigs in _active_corr_groups.items():
        leader = group_leaders.get(group_name)
        if leader:
            penalty = _CLUSTER_CORRELATION_PENALTIES.get(group_name, _CORRELATION_PENALTY)
            for s in group_sigs:
                if s != leader and s in active_non_hold:
                    penalized_signals[s] = min(penalized_signals.get(s, 1.0), penalty)

    # Meta-cluster deduplication (2026-05-01): when leaders from related
    # sub-clusters agree on direction, apply penalty to redundant leaders.
    # Prevents the trend mega-view from getting 3.0x effective leader weight
    # when pure_trend/oscillator_trend/structural_flow leaders vote identically.
    for meta_name, sub_clusters in _META_CLUSTER_GROUPS.items():
        meta_leaders: dict[str, str] = {}
        for sc_name in sub_clusters:
            leader = group_leaders.get(sc_name)
            if leader and leader in active_non_hold:
                meta_leaders[sc_name] = leader
        if len(meta_leaders) < 2:
            continue
        # Check if all leaders agree on direction
        leader_directions = {sc: votes.get(ldr, "HOLD")
                            for sc, ldr in meta_leaders.items()}
        active_dirs = set(leader_directions.values()) - {"HOLD"}
        if len(active_dirs) != 1:
            continue  # Leaders disagree — informative diversity, no penalty
        # All leaders agree: keep best-accuracy leader, penalize others
        best_sc = max(meta_leaders,
                      key=lambda sc: _leader_accuracy_key(meta_leaders[sc]))
        for sc_name, leader in meta_leaders.items():
            if sc_name != best_sc:
                current = penalized_signals.get(leader, 1.0)
                penalized_signals[leader] = min(current, _META_CLUSTER_PENALTY)
                logger.debug(
                    "Meta-cluster %s: %s leader %s agrees with %s leader %s "
                    "— penalized to %.0f%%",
                    meta_name, sc_name, leader, best_sc,
                    meta_leaders[best_sc], _META_CLUSTER_PENALTY * 100,
                )

    # Crisis mode detection: when multiple macro-external signals have degraded
    # accuracy, the market is in an abnormal regime (war, systemic crisis) where
    # trend-following breaks and mean-reversion becomes more reliable.
    #
    # 2026-04-19: Made crisis response conditional on trend signal performance.
    # When macro signals are broken but trend signals have >55% accuracy, the
    # crisis is in the macro indicators, not in the trend — penalizing trend
    # signals that are winning is actively harmful (observed: trend 61.6%,
    # EMA 62.9% being penalized 0.6x while crisis mode was active).
    _MACRO_CRISIS_SIGNALS = {"fear_greed", "macro_regime", "structure", "news_event", "sentiment"}
    broken_count = sum(
        1 for s in _MACRO_CRISIS_SIGNALS
        if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
        and accuracy_data.get(s, {}).get("accuracy", 0.5) < _CRISIS_THRESHOLD
    )
    crisis_mode = broken_count >= _CRISIS_MIN_BROKEN

    _TREND_SIGNALS = {"ema", "trend", "heikin_ashi", "volume_flow"}
    _MR_SIGNALS = {"mean_reversion", "calendar"}

    # Check if trend signals are actually underperforming before penalizing.
    # If avg trend accuracy > 55%, trend is capturing edge despite macro chaos.
    _CRISIS_TREND_ACCURACY_FLOOR = 0.55
    crisis_penalize_trend = False
    if crisis_mode:
        trend_accs = [
            accuracy_data.get(s, {}).get("accuracy", 0.5)
            for s in _TREND_SIGNALS
            if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
        ]
        avg_trend_acc = sum(trend_accs) / len(trend_accs) if trend_accs else 0.5
        crisis_penalize_trend = avg_trend_acc < _CRISIS_TREND_ACCURACY_FLOOR
        if crisis_penalize_trend:
            logger.info(
                "Crisis mode active (full): %d/%d macro signals broken, "
                "trend avg %.1f%% < %.0f%% floor — penalizing trend, boosting MR",
                broken_count, len(_MACRO_CRISIS_SIGNALS),
                avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
            )
        else:
            logger.info(
                "Crisis mode active (partial): %d/%d macro signals broken, but "
                "trend avg %.1f%% >= %.0f%% floor — NOT penalizing trend signals",
                broken_count, len(_MACRO_CRISIS_SIGNALS),
                avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
            )

    # Voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy gating reconfig).
    # Only the overall accuracy gate is relaxable — directional and correlation
    # gates still fire. Prevents regime-transition over-gating that silenced
    # ~8 voters in W15/W16.
    relaxation = _compute_gate_relaxation(
        votes=votes,
        accuracy_data=accuracy_data,
        excluded=excluded,
        group_gated=group_gated_signals,
        base_gate=gate,
        regime=regime,
    )
    if relaxation > 0:
        logger.debug(
            "Circuit breaker: relaxing accuracy gate by %.0fpp "
            "(base=%.2f -> effective=%.2f) to preserve voter diversity",
            relaxation * 100, gate, gate - relaxation,
        )

    # IC-based weight multiplier (2026-04-18): load IC data once per consensus
    # call. Returns {"global": {sig: {ic, icir, samples}}, "per_ticker": {...}}
    # or None if IC computation is unavailable.
    ic_cache = _get_ic_data(horizon) if horizon else None
    ic_global = ic_cache.get("global", {}) if ic_cache else {}
    ic_per_ticker = ic_cache.get("per_ticker", {}) if ic_cache else {}

    for signal_name, vote in votes.items():
        # P1-1 (2026-05-02 adversarial follow-ups): defensive — initialize
        # _rescued at the TOP of every iteration so a future contributor who
        # adds a third branch to the gate-check below cannot leak a stale
        # True from a prior iteration into line 2123 (`if _rescued: weight
        # *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY`). Today both arms of the
        # if/else at line 2072 set _rescued, so the bug doesn't manifest in
        # production — but the structural guarantee is now hardcoded.
        _rescued = False
        if vote == "HOLD":
            continue
        if signal_name in excluded:
            continue
        # Correlation group leader gating: entire group silenced
        if signal_name in group_gated_signals:
            gated_signals.append(signal_name)
            continue
        stats = accuracy_data.get(signal_name, {})
        acc = stats.get("accuracy", 0.5)
        samples = stats.get("total", 0)
        # Accuracy gate: skip signals that are below threshold with enough data.
        # Tiered: established signals (10000+ samples) use a tighter 50% gate;
        # newer signals use the standard 47% gate.
        # SC-P1-2 (2026-05-02 adversarial follow-ups): the high-sample tier
        # (10K+ samples, 0.50 gate) is NOT relaxed. A signal with 10K+ samples
        # at sub-50% accuracy has statistically demonstrated negative edge —
        # circuit-breaker relaxation must not let it back in. The standard
        # tier still relaxes uniformly so newer borderline signals can be
        # rescued during regime transitions.
        effective_gate = gate - relaxation
        if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
            effective_gate = max(
                gate - relaxation,
                _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD,
            )
        if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
            # Directional rescue: before gating, check if the vote direction
            # has strong enough accuracy to justify a reduced-weight vote.
            if vote == "BUY":
                rescue_acc = stats.get("buy_accuracy", 0.0)
                rescue_n = stats.get("total_buy", 0)
            else:
                rescue_acc = stats.get("sell_accuracy", 0.0)
                rescue_n = stats.get("total_sell", 0)
            if (rescue_n >= _DIRECTIONAL_RESCUE_MIN_SAMPLES
                    and rescue_acc >= _DIRECTIONAL_RESCUE_THRESHOLD):
                logger.debug(
                    "Directional rescue: %s overall=%.1f%% (gated) but "
                    "%s=%.1f%% (%d sam) — rescued at %.0f%% weight",
                    signal_name, acc * 100, vote,
                    rescue_acc * 100, rescue_n,
                    _DIRECTIONAL_RESCUE_WEIGHT_PENALTY * 100,
                )
                # Fall through to weighting with rescue penalty applied later
                _rescued = True
            else:
                gated_signals.append(signal_name)
                continue
        else:
            _rescued = False
        # Directional accuracy gate: gate individual BUY/SELL direction when
        # direction-specific accuracy is very poor, even if overall accuracy passes.
        # E.g., qwen3 overall=59.8% passes, but BUY=30.0% → gate BUY only.
        if vote == "BUY":
            dir_acc = stats.get("buy_accuracy", acc)
            dir_n = stats.get("total_buy", 0)
        else:
            dir_acc = stats.get("sell_accuracy", acc)
            dir_n = stats.get("total_sell", 0)
        if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
            gated_signals.append(f"{signal_name}_{vote}")
            continue
        # BUG-182: Use direction-specific accuracy as weight when available.
        # A signal with overall 60% accuracy may be 30% for BUY and 75% for SELL.
        # Using overall accuracy over-weights the weak direction.
        _DIR_WEIGHT_MIN_SAMPLES = 20
        if vote == "BUY" and stats.get("total_buy", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
            weight = stats.get("buy_accuracy", acc)  # BUG-185: .get() for cache safety
        elif vote == "SELL" and stats.get("total_sell", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
            weight = stats.get("sell_accuracy", acc)  # BUG-185: .get() for cache safety
        elif samples >= 20:
            weight = acc
        else:
            weight = 0.5
        # Apply directional rescue penalty: rescued signals contribute at
        # reduced weight since their overall accuracy failed the gate.
        if _rescued:
            weight *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY
        # IC-based weight adjustment: boost signals with high return-magnitude
        # predictive power, penalize phantom performers with zero IC.
        if ic_global:
            # Prefer per-ticker IC when available with enough samples
            _ic_info = None
            if ticker and ic_per_ticker:
                _ic_info = ic_per_ticker.get(ticker, {}).get(signal_name)
                if _ic_info and _ic_info.get("samples", 0) < _IC_MIN_SAMPLES:
                    _ic_info = None  # fall back to global
            if _ic_info is None:
                _ic_info = ic_global.get(signal_name, {})
            _ic = _ic_info.get("ic", 0.0)
            _icir = _ic_info.get("icir", 0.0)
            _ic_n = _ic_info.get("samples", 0)
            ic_mult = _compute_ic_mult(_ic, _icir, _ic_n)
            weight *= ic_mult
        # Regime adjustment
        weight *= regime_mults.get(signal_name, 1.0)
        # Horizon-specific weight adjustment
        if signal_name in horizon_mults:
            weight *= horizon_mults[signal_name]
        # Macro-window downweight (2026-04-28). Composes with regime/
        # horizon multipliers — e.g., during a macro window in ranging
        # regime, sentiment hits 0.5 (macro) × 0.X (regime) × Y (horizon).
        # Only applies to MACRO_WINDOW_DOWNWEIGHT_SIGNALS — the
        # FORCE_HOLD signals were already mutated to HOLD above and won't
        # reach this branch.
        if macro_active and signal_name in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
            weight *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        # Crisis mode adjustments: penalize trend signals (only if they're
        # underperforming), boost mean-reversion. See 2026-04-19 fix above.
        if crisis_mode:
            if signal_name in _TREND_SIGNALS and crisis_penalize_trend:
                weight *= _CRISIS_TREND_PENALTY
            elif signal_name in _MR_SIGNALS:
                weight *= _CRISIS_MR_BOOST
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
            weight *= penalized_signals[signal_name]
        # Directional bias penalty (2026-05-02 research): signals with extreme
        # BUY/SELL bias get penalized ONLY when voting in their bias direction.
        # Contrarian votes (rare, high-value) keep full weight.
        # E.g., calendar is 100% BUY — its BUY votes get 0.5x, but a rare
        # SELL (if it ever emits one) keeps 1.0x because that's genuinely
        # informative. Previous version penalized ALL votes equally.
        signal_bias = act_data.get("bias", 0.0)
        signal_samples = act_data.get("samples", 0)
        if signal_samples >= _BIAS_MIN_ACTIVE and signal_bias > _BIAS_THRESHOLD:
            buy_rate = act_data.get("buy_rate", 0.0)
            sell_rate = act_data.get("sell_rate", 0.0)
            bias_direction = "BUY" if buy_rate >= sell_rate else "SELL"
            if vote == bias_direction:
                penalty = _BIAS_EXTREME_PENALTY if signal_bias > _BIAS_EXTREME_THRESHOLD else _BIAS_PENALTY
                weight *= penalty
        # 2026-05-11 (Codex Fix B): apply soft-vote dampening LAST so it
        # composes with all upstream multipliers (accuracy, IC, regime,
        # horizon, macro, crisis, activity, correlation, bias). The
        # soft_conf is small (0.15-0.20) for dead-zone votes — a strong
        # vote has no soft_conf key and so this branch is skipped,
        # preserving the existing strong-vote weight contract.
        soft = soft_confidences.get(f"_soft_conf_{signal_name}")
        if soft is not None:
            try:
                weight *= float(soft)
            except (TypeError, ValueError):
                pass
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
        # 2026-04-10: bumped to 5 decimals — see portfolio/reporting.py:114
        # for the root-cause explanation (MACD-improving gate rounding).
        "macd_hist": round(ind["macd_hist"], 5),
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
    # 2026-05-10 (codex re-review): explicit dict[str, Any] — initial
    # values mix str / None / int and later rounds add float (accuracy)
    # and int (samples). Without annotation mypy locks the value type
    # to the union of the literal initialisers and rejects every later
    # assignment.
    info: dict[str, Any] = {
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

    df_id = (
        len(df),
        float(df["close"].iloc[0]) if len(df) > 0 else 0.0,
        float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
    )
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
        # BUG-180: Insertion-order eviction — drop oldest 50% instead of clearing all
        with _adx_lock:
            if len(_adx_cache) >= _ADX_CACHE_MAX:
                keys = list(_adx_cache.keys())
                for k in keys[:len(keys) // 2]:
                    del _adx_cache[k]
            _adx_cache[df_id] = result
        return result
    except Exception:
        logger.warning("ADX computation failed", exc_info=True)
        with _adx_lock:
            _adx_cache[df_id] = None
        return None


def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
    """Apply an 8-stage multiplicative confidence penalty cascade.

    Stages:
      1. Regime penalty — dampens confidence in choppy/volatile markets
      2. Volume/ADX gate — rejects low-conviction signals
      3. Trap detection — catches bull/bear traps (price vs volume divergence)
      4. Dynamic MIN_VOTERS — raises the bar in uncertain markets
      5. Unanimity penalty — over-agreement often means the move is priced in
      5b. Ensemble entropy — high 3-way disagreement caps confidence
      6. Per-ticker consensus — penalizes tickers where ensemble accuracy < 50%
      7. Calibration compression — compress overconfident predictions to honest levels

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
    # P2-C (2026-04-17): delegate to shared helper to avoid drift with the
    # circuit breaker's recovery-floor logic. Same semantic as before.
    # BUG-227: Use post-persistence voter count (not pre-filter) so the gate
    # reflects the actual participating voters after debounce filtering.
    active_voters = extra_info.get("_voters_post_filter",
                                    extra_info.get("_voters", 0))
    dynamic_min = _dynamic_min_voters_for_regime(regime)

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

    # --- Stage 5b: Ensemble entropy guard ---
    # High Shannon entropy across BUY/SELL/HOLD means signals genuinely disagree —
    # the winning direction is near-random. Cap confidence.
    _ENTROPY_MIN_APPLICABLE = 5
    if action != "HOLD" and conf > 0.0:
        import math as _math_ent
        _ent_buy = extra_info.get("_buy_count", 0)
        _ent_sell = extra_info.get("_sell_count", 0)
        _ent_total = extra_info.get("_total_applicable", _ent_buy + _ent_sell)
        _ent_hold = max(0, _ent_total - _ent_buy - _ent_sell)
        if _ent_total >= _ENTROPY_MIN_APPLICABLE:
            _probs = [c / _ent_total for c in (_ent_buy, _ent_sell, _ent_hold) if c > 0]
            _entropy = -sum(p * _math_ent.log2(p) for p in _probs)
            _norm_entropy = _entropy / _math_ent.log2(3)
            extra_info["_ensemble_entropy"] = round(_norm_entropy, 4)
            if _norm_entropy >= 0.9:
                conf *= 0.6
                penalty_log.append({"stage": "ensemble_entropy", "entropy": round(_norm_entropy, 3), "mult": 0.6})
            elif _norm_entropy >= 0.8:
                conf *= 0.8
                penalty_log.append({"stage": "ensemble_entropy", "entropy": round(_norm_entropy, 3), "mult": 0.8})

    # --- Stage 6: Per-ticker consensus accuracy penalty ---
    # RES-2026-04-17: The consensus system has below-coinflip accuracy for some
    # tickers (ETH-USD 47.7% at 3h, MSTR 45.9%). When this happens, the ensemble
    # is net-negative — acting on its signals loses money. Apply a confidence
    # penalty proportional to how far below 52% the consensus accuracy is.
    # Don't force HOLD (too aggressive) — just reduce confidence.
    # RES-2026-04-21: Raised threshold 0.50→0.52 to catch coin-flip tickers
    # (XAG-USD 50.0%, XAU-USD 49.6% were getting zero penalty). Steepened
    # the curve and lowered floor (0.3→0.2) for truly broken instruments.
    _PTC_MIN_SAMPLES = 500
    _PTC_PENALTY_THRESHOLD = 0.52
    if action != "HOLD":
        ptc_acc = extra_info.get("_ptc_accuracy")
        ptc_samples = extra_info.get("_ptc_samples", 0)
        if ptc_acc is not None and ptc_samples >= _PTC_MIN_SAMPLES and ptc_acc < _PTC_PENALTY_THRESHOLD:
            # Scale penalty: 52% acc → 0.6x, 50% acc → 0.52x, 48% acc → 0.44x, 40% acc → 0.2x
            ptc_mult = max(0.2, 0.6 + (ptc_acc - _PTC_PENALTY_THRESHOLD) * 4.0)
            conf *= ptc_mult
            penalty_log.append({
                "stage": "per_ticker_consensus",
                "ticker": ticker,
                "ptc_accuracy": round(ptc_acc, 4),
                "ptc_samples": ptc_samples,
                "mult": round(ptc_mult, 4),
            })

    # --- Stage 7: Confidence calibration compression ---
    # RES-2026-04-18: Calibration analysis shows confidence is meaningless
    # above 60% — all bands (60-69%, 70-79%, 80-89%) have ~50% actual
    # accuracy. The system is massively overconfident. Compress high-confidence
    # predictions to honest levels while preserving relative ordering.
    # Formula: conf = 0.55 + (conf - 0.55) * 0.3  (for conf > 0.55)
    # Maps: 60% → 56.5%, 70% → 59.5%, 80% → 62.5%, 90% → 65.5%
    _CALIBRATION_THRESHOLD = 0.55
    _CALIBRATION_COMPRESSION = 0.3
    if action != "HOLD" and conf > _CALIBRATION_THRESHOLD:
        raw_conf = conf
        conf = _CALIBRATION_THRESHOLD + (conf - _CALIBRATION_THRESHOLD) * _CALIBRATION_COMPRESSION
        penalty_log.append({
            "stage": "calibration_compression",
            "raw_conf": round(raw_conf, 4),
            "compressed_conf": round(conf, 4),
        })

    # Clamp confidence to [0, 1]
    conf = max(0.0, min(1.0, conf))

    return action, conf, penalty_log


def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None, horizon=None):
    # CRITICAL-2 guard (2026-04-17 adversarial review): empty/None ticker
    # slipped through scattered `if ticker:` checks in production before.
    # All real callers (main.py:486, agent_invocation, backtester) pass a
    # non-empty string; only tests/test_signal_engine.py:651 passes None
    # intentionally (BUG-178 regression guard for tracker pollution).
    # Warn on empty so future regressions surface rather than silently
    # degrading the signal pipeline.
    if not ticker:
        logger.warning(
            "generate_signal called with empty ticker=%r — "
            "tracker/phase updates will be skipped",
            ticker,
        )

    # 2026-05-10 (codex re-review): explicit annotations so the strict pilot
    # sees the heterogeneous shape we rely on. Without these mypy infers
    # ``dict[Any, bool]`` from the first truthy assignment and reports a
    # cascade of false-positive [assignment]/[arg-type] errors at every
    # subsequent extra_info["..."] = <non-bool> line (Codex flagged 30+).
    votes: dict[str, str] = {}
    shadow_votes: dict[str, str] = {}  # disabled signals tracked for accuracy
    extra_info: dict[str, Any] = {}

    # BUG-178 diagnostic phase marker (added 2026-04-10, diag/bug178-end-of-cycle-snapshot).
    # The per-ticker last-signal tracker is updated inside the enhanced-signal
    # dispatch loop, but slow cycles can also hang BEFORE the loop (sentiment,
    # fear_greed, news_event, _cached() macro fetches) or AFTER it (accuracy_stats
    # loading, weighted consensus, per-ticker gate). Writing `__pre_dispatch__`
    # here and `__post_dispatch__` after the loop gives the end-of-cycle slow
    # diagnostic in main.py three distinct phases to point at:
    #   - __pre_dispatch__  → hang is in the pre-loop block
    #   - <signal name>     → hang is in the dispatch loop at that signal
    #   - __post_dispatch__ → hang is in accuracy/consensus code after the loop
    # The double-underscore prefix is reserved; no registered signal uses it.
    #
    # 2026-04-15 (bug178-instrumentation-timeout): also reset the per-phase log
    # so the post-dispatch code can record granular timing between __post_dispatch__
    # and the return. See _record_phase() and _reset_phase_log() above.
    if ticker:
        _set_last_signal(ticker, "__pre_dispatch__")
        _reset_phase_log(ticker)

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

    # MACD — strong vote on histogram crossover; soft directional vote in
    # the dead-zone (2026-05-11 Stage 2 Batch 1). The strong-vote path is
    # unchanged. When |hist| is small and there is no crossover, the
    # dead-zone helper inspects the histogram slope over the last few
    # bars and emits a weak BUY/SELL (conf=MACD_DEAD_ZONE_SOFT_CONF) when
    # there is directional drift, or HOLD when truly flat. Rationale:
    # entries should pick a direction; HOLD is for managing positions.
    if ind["macd_hist"] > 0 and ind["macd_hist_prev"] <= 0:
        votes["macd"] = "BUY"
    elif ind["macd_hist"] < 0 and ind["macd_hist_prev"] >= 0:
        votes["macd"] = "SELL"
    else:
        macd_vote, macd_soft_conf = _macd_dead_zone_vote(ind, df)
        votes["macd"] = macd_vote
        if macd_vote != "HOLD":
            extra_info["_soft_conf_macd"] = macd_soft_conf

    # EMA trend — strong vote when gap >= 0.5%; soft directional vote in
    # the dead-zone (2026-05-11 Stage 2 Batch 1). The strong-vote path is
    # unchanged. When the gap is small, compare EMA9 slope to EMA21
    # slope over the last few bars; faster EMA9 -> weak BUY, slower ->
    # weak SELL, flat -> HOLD. Rationale: entries should pick a
    # direction; HOLD is for managing positions.
    ema_gap_pct = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100 if ind["ema21"] != 0 else 0
    )
    if ema_gap_pct >= 0.5:
        votes["ema"] = "BUY" if ind["ema9"] > ind["ema21"] else "SELL"
    else:
        ema_vote, ema_soft_conf = _ema_dead_zone_vote(ind, df)
        votes["ema"] = ema_vote
        if ema_vote != "HOLD":
            extra_info["_soft_conf_ema"] = ema_soft_conf

    # Bollinger Bands — strong vote at band touches; soft directional
    # vote when price is inside the band but biased toward an edge
    # (2026-05-11 Stage 2 Batch 1). Normalized band position
    # (price - mid) / (upper - mid) clamped to [-1, +1]:
    # > 0.6 -> weak SELL, < -0.6 -> weak BUY, else HOLD. Rationale:
    # entries should pick a direction; HOLD is for managing positions.
    if ind["price_vs_bb"] == "below_lower":
        votes["bb"] = "BUY"
    elif ind["price_vs_bb"] == "above_upper":
        votes["bb"] = "SELL"
    else:
        bb_vote, bb_soft_conf = _bb_inside_band_vote(ind)
        votes["bb"] = bb_vote
        if bb_vote != "HOLD":
            extra_info["_soft_conf_bb"] = bb_soft_conf

    # --- Extended signals from tools (optional) ---

    # Fear & Greed Index (per-ticker: crypto->alternative.me, stocks->VIX)
    # Gated: F&G is contrarian (buy fear, sell greed) which fights trends.
    # Only allow F&G to vote in ranging/high-vol regimes where mean reversion works.
    # 2026-04-02: Added sustained fear duration gate — during prolonged extreme
    # fear (46+ consecutive days as of Apr 2), contrarian BUY signals are noise.
    # Historical data: buying at F&G <15 yields median +38.4% over 90 days,
    # but during sustained fear (2022), prices dropped another -40% after signal.
    votes["fear_greed"] = "HOLD"
    try:
        from portfolio.fear_greed import get_fear_greed, get_sustained_fear_days, update_fear_streak

        fg_key = f"fear_greed_{ticker}" if ticker else "fear_greed"
        fg = _cached(fg_key, FEAR_GREED_TTL, get_fear_greed, ticker)
        if fg:
            extra_info["fear_greed"] = fg["value"]
            extra_info["fear_greed_class"] = fg["classification"]
            # Read streak BEFORE updating — use previous cycle's state for voting
            fear_days = get_sustained_fear_days()
            # Update streak tracker (once per cycle, not per ticker)
            if ticker in ("BTC-USD", None):
                update_fear_streak(fg["value"])
            extra_info["fear_greed_streak_days"] = fear_days
            # Gate: suppress F&G votes in trending regimes
            if regime in ("trending-up", "trending-down"):
                extra_info["fear_greed_gated"] = regime
                votes["fear_greed"] = "HOLD"
            # Gate: sustained extreme fear — contrarian BUY is unreliable
            # during first 30 days of prolonged fear (whiplash risk).
            # After 30 days, allow BUY but at reduced confidence (handled
            # by the existing 0.3x ranging regime weight).
            elif fg["value"] <= 20 and fear_days > 30:
                votes["fear_greed"] = "BUY"
                extra_info["fear_greed_note"] = f"sustained_fear_{fear_days}d_allowing_contrarian"
            elif fg["value"] <= 20 and fear_days > 0:
                extra_info["fear_greed_gated"] = f"sustained_fear_{fear_days}d"
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
            # 2026-05-15 LLM shadow-enrollment: pass the full ticker
            # (e.g. "BTC-USD") so per-sub-voter rows in
            # llm_probability_log.jsonl use the same ticker key as the
            # aggregate `sentiment` row emitted ~30 lines below.
            # _log_ticker_full is consumed only by _log_sub_vote() — it
            # does NOT change which headlines get fetched (sentiment.py
            # still keys headline lookup off the short ticker arg).
            _sent_fn = partial(
                get_sentiment,
                cryptocompare_api_key=cc_api_key,
                _log_ticker_full=ticker,
            )
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
                # 2026-04-21: carry avg_scores forward for the LLM probability
                # logger so it can write a rich P(positive)/P(negative)/P(neutral)
                # distribution instead of a confidence-split fallback.
                if sent.get("avg_scores"):
                    extra_info["sentiment_avg_scores"] = sent["avg_scores"]

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

    # Funding Rate — 29.9% accuracy at 1d but 74.2% at 3h (535 samples).
    # Re-enabled 2026-04-09: horizon-gated via REGIME_GATED_SIGNALS (_default
    # gates it at 1d across all regimes; active only at 3h/4h horizons).
    # Crypto-only (BTC, ETH). The regime gate handles suppression at 1d.
    votes["funding"] = "HOLD"
    if ticker and ticker in CRYPTO_SYMBOLS:
        try:
            from portfolio.funding_rate import get_funding_rate

            fr = _cached(f"funding_{ticker}", FUNDING_RATE_TTL, get_funding_rate, ticker)
            if fr:
                extra_info["funding_rate"] = fr["rate_pct"]
                extra_info["funding_action"] = fr["action"]
                votes["funding"] = fr["action"]
        except ImportError:
            logger.debug("Optional module %s not available", "funding_rate")

    # On-Chain BTC Signal — MVRV Z-Score, SOPR, NUPL, Exchange Netflow.
    # BTC-only. Data from BGeometrics (12h cache, 15 req/day).
    # Added 2026-04-09: promotes existing on-chain data to a voting signal.
    votes["onchain"] = "HOLD"
    if ticker == "BTC-USD":
        try:
            from portfolio.onchain_data import get_onchain_data

            oc = _cached("onchain_btc_signal", ONCHAIN_TTL, get_onchain_data)
            if oc:
                sub_votes = []
                # MVRV Z-Score: <1 undervalued (BUY), >5 overheated (SELL)
                zscore = oc.get("mvrv_zscore")
                if zscore is not None:
                    if zscore < 1.0:
                        sub_votes.append("BUY")
                    elif zscore > 5.0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_mvrv_zscore"] = round(zscore, 2)
                # SOPR: <0.97 capitulation (BUY), >1.05 profit-taking (SELL)
                sopr = oc.get("sopr")
                if sopr is not None:
                    if sopr < 0.97:
                        sub_votes.append("BUY")
                    elif sopr > 1.05:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_sopr"] = round(sopr, 4)
                # NUPL: <0 capitulation (BUY), >0.75 euphoria (SELL)
                nupl = oc.get("nupl")
                if nupl is not None:
                    if nupl < 0:
                        sub_votes.append("BUY")
                    elif nupl > 0.75:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                # Exchange netflow: negative = accumulation (BUY)
                netflow = oc.get("netflow")
                if netflow is not None:
                    if netflow < 0:
                        sub_votes.append("BUY")
                    elif netflow > 0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                # Majority vote
                buy_count = sub_votes.count("BUY")
                sell_count = sub_votes.count("SELL")
                total = buy_count + sell_count
                if total >= 2:
                    if buy_count > sell_count:
                        votes["onchain"] = "BUY"
                    elif sell_count > buy_count:
                        votes["onchain"] = "SELL"
                    extra_info["onchain_sub_votes"] = f"{buy_count}B/{sell_count}S"
        except ImportError:
            logger.debug("Optional module %s not available", "onchain_data")

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
    # Uses batch queue: on cache miss, enqueues for post-cycle flush instead of
    # calling model inline (avoids model swap ping-pong between threads).
    votes["ministral"] = "HOLD"
    if ticker and not skip_gpu:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.llm_batch import enqueue_ministral, is_llm_on_cycle

            ctx = _build_llm_context(ticker, ind, timeframes, extra_info)
            # 2026-04-10 (perf/llama-swap-reduction): gate the enqueue with
            # rotation predicate. When ministral is off-cycle this cycle,
            # _cached_or_enqueue skips the enqueue and returns stale data.
            # max_stale_factor=5 gives 5 * 15 min = 75 min of stale tolerance,
            # comfortably covering the 3-cycle rotation period (~45-60 min
            # depending on cycle slippage) with slack.
            ms = _cached_or_enqueue(
                f"ministral_{short_ticker}",
                MINISTRAL_TTL,
                enqueue_ministral,
                ctx,
                should_enqueue_fn=lambda: is_llm_on_cycle("ministral"),
                max_stale_factor=5,
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
    # Uses batch queue: same pattern as Ministral above.
    votes["qwen3"] = "HOLD"
    qwen3_enabled = (config or {}).get("local_models", {}).get("qwen3", {}).get("enabled", True)
    if ticker and qwen3_enabled and not skip_gpu:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.llm_batch import enqueue_qwen3, is_llm_on_cycle

            ctx = _build_llm_context(ticker, ind, timeframes, extra_info)
            # Qwen3 gets asset_type for prompt diversification
            if ticker in CRYPTO_SYMBOLS:
                ctx["asset_type"] = "cryptocurrency"
            elif ticker in METALS_SYMBOLS:
                ctx["asset_type"] = "precious metal"
            else:
                ctx["asset_type"] = "stock"
            # 2026-04-10 (perf/llama-swap-reduction): rotation gate via
            # is_llm_on_cycle; see ministral block above for rationale.
            q3 = _cached_or_enqueue(
                f"qwen3_{short_ticker}",
                MINISTRAL_TTL,
                enqueue_qwen3,
                ctx,
                should_enqueue_fn=lambda: is_llm_on_cycle("qwen3"),
                max_stale_factor=5,
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

        # Load seasonality profile for metals tickers (detrending)
        seasonality_profile = None
        if ticker in METALS_SYMBOLS:
            try:
                from portfolio.seasonality import get_profile
                seasonality_profile = get_profile(ticker)
            except Exception:
                logger.debug("Seasonality profile load failed for %s", ticker, exc_info=True)

        # Build context data once for signals that need it
        # BUG-144: Include regime so enhanced signals (forecast.py) can apply
        # regime-specific confidence discounts.
        context_data = {
            "ticker": ticker, "config": config or {}, "macro": macro_data,
            "regime": regime, "seasonality_profile": seasonality_profile,
        }

        _signal_failures = []
        for sig_name, entry in _enhanced_entries.items():
            # BUG-178 fix (2026-04-10): respect DISABLED_SIGNALS in the dispatch
            # loop. Previously this loop iterated *every* registered enhanced
            # signal regardless of disabled status, which meant the three
            # "registered but force-HOLD pending live validation" signals
            # (crypto_macro, cot_positioning, credit_spread_risk) were doing
            # network I/O on every cycle. Their late position in iteration
            # order matches the silent gap before all 49 BUG-178 events
            # observed 2026-04-09/10 (last [SLOW] log line is always the
            # signal *before* this set, then 150+s of silence). Skipping them
            # restores the documented behavior. Other DISABLED_SIGNALS-aware
            # call sites: count_active_signals():468, dynamic correlation:558,
            # accuracy_stats.py, ticker_accuracy.py, backtester.py, reporting.py.
            if sig_name in DISABLED_SIGNALS and (sig_name, ticker) not in _DISABLED_SIGNAL_OVERRIDES:
                # Shadow-safe signals: compute but don't let them vote.
                # Their predictions go into _shadow_votes for accuracy tracking.
                if sig_name in _SHADOW_SAFE_SIGNALS:
                    try:
                        _sig_t0 = time.monotonic()
                        compute_fn = load_signal_func(entry)
                        if compute_fn is not None:
                            if ticker:
                                _set_last_signal(ticker, f"shadow:{sig_name}")
                            if entry.get("requires_context"):
                                result = compute_fn(df, context=context_data)
                            elif entry.get("requires_macro"):
                                result = compute_fn(df, macro=macro_data or None)
                            else:
                                result = compute_fn(df)
                            _sig_dt = time.monotonic() - _sig_t0
                            if _sig_dt > 1.0:
                                logger.info("[SLOW-SHADOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
                            max_conf = entry.get("max_confidence", 1.0)
                            validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
                            extra_info[f"{sig_name}_action"] = validated["action"]
                            extra_info[f"{sig_name}_confidence"] = validated["confidence"]
                            extra_info[f"shadow_{sig_name}"] = True
                            shadow_votes[sig_name] = validated["action"]
                    except Exception as e:
                        logger.debug("Shadow signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
                continue
            if sig_name in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
                votes[sig_name] = "HOLD"
                continue
            # Skip GPU-intensive enhanced signals for stocks outside market hours
            if skip_gpu and sig_name in GPU_SIGNALS:
                votes[sig_name] = "HOLD"
                continue
            # 2026-05-15 LLM shadow-enrollment: cycle-modulo throttle for
            # shadow signals. Expensive shadow LLMs (forecast, meta_trader,
            # finance_llama, etc.) declare cycle_modulo in shadow_registry
            # so we only compute them every Nth cycle and the cycle budget
            # stays near 60s. Counter is UTC epoch minute — stateless, so
            # a loop restart picks up the right phase automatically. Only
            # skipped signals get force-HOLD here; signals not in shadow
            # status fall through unchanged.
            try:
                from portfolio.shadow_registry import (
                    cycle_count_now,
                    get_status,
                    should_run_this_cycle,
                )
                if get_status(sig_name) == "shadow":
                    if not should_run_this_cycle(sig_name, cycle_count_now()):
                        votes[sig_name] = "HOLD"
                        extra_info[f"{sig_name}_throttled"] = True
                        continue
            except Exception:
                logger.debug("shadow-throttle check failed for %s", sig_name, exc_info=True)
            try:
                _sig_t0 = time.monotonic()
                compute_fn = load_signal_func(entry)
                if compute_fn is None:
                    votes[sig_name] = "HOLD"
                    continue
                # BUG-178 diagnostic: track which signal each ticker is currently
                # running so main.py's pool-timeout handler can name the culprit.
                # Ticker guard added 2026-04-10 in the phase-marker diag commit
                # to prevent leaking a None-keyed entry when callers pass
                # ticker=None (legacy test harnesses and backtester paths).
                if ticker:
                    _set_last_signal(ticker, sig_name)
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

        # 2026-05-11 Stage 2 Batch 2: candlestick + forecast dead-zone
        # soft directional votes. Same pattern as Batch 1 (EMA / BB /
        # MACD) but on enhanced signals — the helper only fires when
        # the strong path returned HOLD AND a secondary derivative
        # (candle body direction / price+EMA slope alignment) points
        # cleanly one way. Strong votes are untouched. The soft conf is
        # written into extra_info so `_weighted_consensus` dampens the
        # vote's weight via the existing soft_confidences path.
        if "candlestick" in votes and votes["candlestick"] == "HOLD" \
                and "candlestick" not in DISABLED_SIGNALS \
                and "candlestick" not in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
            cs_vote, cs_soft_conf = _candlestick_dead_zone_vote(df)
            if cs_vote != "HOLD":
                votes["candlestick"] = cs_vote
                extra_info["_soft_conf_candlestick"] = cs_soft_conf
                extra_info["candlestick_action"] = cs_vote
        if "forecast" in votes and votes["forecast"] == "HOLD" \
                and "forecast" not in DISABLED_SIGNALS \
                and "forecast" not in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
            fc_indicators = extra_info.get("forecast_indicators") or {}
            fc_vote, fc_soft_conf = _forecast_dead_zone_vote(df, fc_indicators)
            if fc_vote != "HOLD":
                votes["forecast"] = fc_vote
                extra_info["_soft_conf_forecast"] = fc_soft_conf
                extra_info["forecast_action"] = fc_vote
    else:
        for sig_name in _enhanced_entries:
            votes[sig_name] = "HOLD"

    # MSTR BTC cross-asset proxy (2026-04-29): MSTR is a BTC treasury company
    # (818K BTC, 0.58 price correlation). Its per-ticker consensus accuracy is
    # 47.8% — worst of all Tier 1. Injecting BTC-USD's consensus as a synthetic
    # signal provides cross-asset information the signal system otherwise ignores.
    # The vote goes through all normal gates (accuracy, regime, persistence).
    if ticker == "MSTR":
        with _cross_ticker_lock:
            btc_cons = _cross_ticker_consensus.get("BTC-USD")
        if btc_cons is not None:
            btc_action = btc_cons.get("action", "HOLD")
            if btc_action in ("BUY", "SELL", "HOLD"):
                votes["btc_proxy"] = btc_action
                extra_info["btc_proxy_action"] = btc_action
                extra_info["btc_proxy_confidence"] = btc_cons.get("confidence", 0.0)
                extra_info["btc_proxy_source"] = "cross_ticker_cache"

    # BUG-178 diagnostic phase marker (added 2026-04-10, see docstring above
    # at the __pre_dispatch__ writer). We made it through the enhanced-signal
    # dispatch loop; any remaining slow code is in the post-dispatch accuracy /
    # consensus path. The end-of-cycle diagnostic in main.py uses this marker
    # to identify WHICH phase is slow on cycles > 120 s.
    if ticker:
        _set_last_signal(ticker, "__post_dispatch__")

    # BUG-178 phase timing (added 2026-04-15): _phase_start_*  track wall
    # time at the boundary between named post-dispatch phases so _record_phase()
    # can log per-phase duration. See module-level _phase_log_per_ticker and
    # main.py's slow-cycle diagnostic for how this log is consumed.
    _phase_start = time.monotonic()

    # 2026-04-21: Log per-vote probability distribution for every LLM-family
    # signal. Central hook before any gating logic rewrites votes. The
    # probability log is how we distinguish "confidently wrong" from
    # "uncertainly wrong" for calibration analysis — argmax accuracy alone
    # can't tell shadow models apart. Fire-and-forget: failures never abort
    # the signal cycle.
    try:
        from portfolio.llm_probability_log import (
            derive_probs_from_result,
            llm_signals,
            log_vote,
        )
        for sig_name in llm_signals():
            action = votes.get(sig_name)
            if not action or action not in ("BUY", "HOLD", "SELL"):
                continue
            if sig_name == "sentiment":
                conf = extra_info.get("sentiment_conf", 0.0)
                indicators = {
                    "avg_scores": extra_info.get("sentiment_avg_scores"),
                } if extra_info.get("sentiment_avg_scores") else None
            else:
                conf = extra_info.get(f"{sig_name}_confidence", 0.0)
                indicators = extra_info.get(f"{sig_name}_indicators")
            probs = derive_probs_from_result(
                sig_name, action, conf, indicators=indicators,
            )
            if probs is None:
                continue
            tier = None
            if sig_name == "claude_fundamental" and indicators:
                tier = indicators.get("tier")
            # Default horizon to "1d" when caller passes None so the log is
            # queryable by horizon without null-handling at every join site.
            log_vote(
                sig_name, ticker or "", probs,
                horizon=horizon or "1d", chosen=action, confidence=conf, tier=tier,
            )
    except Exception:
        logger.debug("llm probability logging failed", exc_info=True)

    # C10: Capture raw pre-gate votes BEFORE any gating rewrites them to HOLD.
    # This allows accuracy tracking for regime-gated signals, breaking the
    # dead-signal trap where gated signals can never accumulate accuracy data.
    raw_votes = dict(votes)
    # Merge shadow votes so outcome_tracker can track accuracy for disabled
    # signals that were shadow-computed (math-only, no network I/O).
    raw_votes.update(shadow_votes)
    if shadow_votes:
        extra_info["_shadow_votes"] = shadow_votes

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
    # BUG-158: Per-ticker exemption — if a signal has ≥60% accuracy with ≥50
    # samples on THIS ticker, exempt it from regime gating. fear_greed is 93.8%
    # on XAG-USD but globally gated in ranging — this recovers that alpha.
    regime_gated = _get_regime_gated(regime, horizon)
    _ticker_acc_data = {}
    try:
        from portfolio.accuracy_stats import accuracy_by_ticker_signal_cached
        acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
        _ticker_acc_data = (accuracy_by_ticker_signal_cached(acc_horizon) or {}).get(ticker, {})
    except Exception:
        logger.debug("Per-ticker accuracy unavailable for regime gating exemption", exc_info=True)
    _TICKER_EXEMPT_ACC = 0.60
    _TICKER_EXEMPT_MIN_SAMPLES = 50
    # RES-2026-04-21: Recent-accuracy override for regime gating. When a signal's
    # 7d recent accuracy is significantly above the gate threshold (>55%, 50+ samples),
    # exempt it from regime gating even if all-time data is bad. Prevents stale regime
    # gates from suppressing signals that have recovered in a new market regime.
    # Example: fibonacci went from 43% all-time to 68.2% recent — should not be gated.
    _RECENT_EXEMPT_ACC = 0.55
    _RECENT_EXEMPT_MIN_SAMPLES = 50
    _recent_acc_data: dict[str, Any] = {}
    try:
        from portfolio.accuracy_stats import get_or_compute_recent_accuracy
        # get_or_compute_recent_accuracy expects the base horizon, not the cache key
        base_hz = "3h" if horizon in ("3h", "4h") else "1d"
        _recent_acc_data = get_or_compute_recent_accuracy(base_hz) or {}
    except Exception:
        logger.debug("Recent accuracy unavailable for regime gating override", exc_info=True)

    regime_gated_effective = set(regime_gated)
    for sig_name in list(regime_gated_effective):
        # Per-ticker exemption (BUG-158)
        t_stats = _ticker_acc_data.get(sig_name, {})
        t_acc = t_stats.get("accuracy", 0)
        t_samples = t_stats.get("total", 0)
        if t_samples >= _TICKER_EXEMPT_MIN_SAMPLES and t_acc >= _TICKER_EXEMPT_ACC:
            regime_gated_effective.discard(sig_name)
            logger.debug(
                "BUG-158: %s exempt from %s regime gating for %s (%.1f%%, %d samples)",
                sig_name, regime, ticker, t_acc * 100, t_samples,
            )
            continue
        # Recent-accuracy override (RES-2026-04-21)
        r_stats = _recent_acc_data.get(sig_name, {})
        r_acc = r_stats.get("accuracy", 0)
        r_samples = r_stats.get("total", 0)
        if r_samples >= _RECENT_EXEMPT_MIN_SAMPLES and r_acc >= _RECENT_EXEMPT_ACC:
            regime_gated_effective.discard(sig_name)
            logger.debug(
                "RES-2026-04-21: %s exempt from %s regime gating — recent 7d "
                "accuracy %.1f%% (%d sam) overrides stale gate",
                sig_name, regime, r_acc * 100, r_samples,
            )
    for sig_name in regime_gated_effective:
        if sig_name in votes and votes[sig_name] != "HOLD":
            votes[sig_name] = "HOLD"

    # P1-B (2026-04-17 adversarial review): apply horizon-specific blacklist
    # HERE, before buy/sell counting, so `active_voters` reflects the post-
    # horizon-disable state. Previously this gating only happened inside
    # _weighted_consensus, leaving `extra_info["_voters"]` stale - Stage 4's
    # dynamic_min check could pass a 5-voter count even though only 2 voters
    # actually drove the consensus. _weighted_consensus still applies the
    # same gating internally (idempotent: HOLD->HOLD is a no-op) as defense
    # in depth for callers that bypass generate_signal.
    horizon_disabled_effective = _get_horizon_disabled_signals(ticker, horizon)
    for sig_name in horizon_disabled_effective:
        if sig_name in votes and votes[sig_name] != "HOLD":
            votes[sig_name] = "HOLD"

    # Codex round 2 P1 (2026-04-28): macro-window force-HOLD must mutate
    # `votes` BEFORE buy/sell/core_active are computed below — otherwise
    # those counts come from the pre-mutation state and the gate at
    # line ~3333 ("core_active == 0 ...") sees a stale 1 even when the
    # only core voter (e.g. claude_fundamental) was suppressed by macro.
    # Mirrors the existing regime_gate / horizon_disabled mutation
    # pattern above.
    macro_active_effective = _is_macro_window_cached()
    if macro_active_effective and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
        for sig_name in MACRO_WINDOW_FORCE_HOLD_SIGNALS:
            if sig_name in votes and votes[sig_name] != "HOLD":
                votes[sig_name] = "HOLD"

    if ticker:
        _record_phase(ticker, "regime_gate", _phase_start)
        _phase_start = time.monotonic()

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
        # 2026-05-11: metals lowered from MIN_VOTERS_STOCK(3) to
        # MIN_VOTERS_METALS(2). Intraday horizon + persistence filter
        # leaves only 2 voters in steady-state on XAG; the old 3-voter
        # floor produced 0 trades in 20 days.
        min_voters = MIN_VOTERS_METALS
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
    # H3: Define acc_horizon before the try/except so the except block and
    # subsequent code can reference it even if the import fails.
    acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
    try:
        from portfolio.accuracy_stats import (
            blend_accuracy_data,
            get_or_compute_accuracy,
            get_or_compute_per_ticker_accuracy,
            get_or_compute_recent_accuracy,
            load_cached_activation_rates,
        )

        # BUG-178 (2026-04-16): the get_or_compute_* helpers serialize the
        # cache-miss compute via _accuracy_compute_lock so 5 parallel ticker
        # threads don't each pay the 7s+ cost of loading 50,000 signal-log
        # entries when the 1h TTL expires (was 215s wall before the fix).
        # See accuracy_stats.py for the lock rationale.
        alltime = get_or_compute_accuracy(acc_horizon)
        recent = get_or_compute_recent_accuracy(acc_horizon, days=7)
        # BUG-164 lazy-populate per-ticker consensus accuracy — _ptc_key
        # convention preserved by get_or_compute_per_ticker_accuracy.
        _ptc_data = get_or_compute_per_ticker_accuracy(acc_horizon)
        # RES-2026-04-17: Pass per-ticker consensus accuracy into extra_info
        # so apply_confidence_penalties can penalize tickers where the consensus
        # system itself has below-coinflip accuracy (e.g. ETH-USD 47.7% at 3h,
        # MSTR 45.9%). The consensus is the aggregated output, not individual
        # signals — if it's below 50%, the ensemble is net-negative for this ticker.
        if _ptc_data and ticker and isinstance(_ptc_data, dict):
            _ticker_ptc = _ptc_data.get(ticker)
            if isinstance(_ticker_ptc, dict):
                extra_info["_ptc_accuracy"] = _ticker_ptc.get("accuracy")
                extra_info["_ptc_samples"] = _ticker_ptc.get("total", 0)

        # ARCH-23: Use shared blend function (replaces inline logic).
        accuracy_data = blend_accuracy_data(
            alltime, recent,
            divergence_threshold=_RECENCY_DIVERGENCE_THRESHOLD,
            normal_weight=_RECENCY_WEIGHT_NORMAL,
            fast_weight=_RECENCY_WEIGHT_FAST,
            min_recent_samples=_RECENCY_MIN_SAMPLES,
        )

        activation_rates = load_cached_activation_rates()
        _accuracy_failed = False
    except Exception:
        logger.error("Accuracy stats load failed", exc_info=True)
        # H3: Fail-closed: gate all signals (0% accuracy, 999 samples) rather than
        # leaving accuracy_data = {} which bypasses the accuracy gate entirely.
        accuracy_data = {sig: {"accuracy": 0.0, "total": 999} for sig in SIGNAL_NAMES}
        _accuracy_failed = True

    if ticker:
        _record_phase(ticker, "acc_load", _phase_start)
        _phase_start = time.monotonic()

    # Overlay regime-specific accuracy when available.
    # H3: Skip all overlays when primary load failed — they would silently restore
    # real accuracy values for cached signals, negating the fail-closed gate.
    if not _accuracy_failed:
        try:
            from portfolio.accuracy_stats import get_or_compute_regime_accuracy
            # BUG-134: Use acc_horizon (not hardcoded "1d") so regime accuracy
            # matches the prediction horizon (3h/4h/12h/1d).
            # 2026-05-04: switched from manual L2-only dance to L1+L2 wrapper.
            # The previous code re-read disk on every ticker call (~10-50ms
            # JSON parse) and on TTL miss all 5 ticker threads cold-computed
            # in parallel (~30s × 5). The wrapper adds an in-memory L1 with
            # the same dogpile-resistant pattern as signal_utility, dropping
            # 2nd-through-Nth ticker calls per cycle to <1ms. Empty dict on
            # failure preserves the pre-existing fall-through behavior.
            regime_acc = get_or_compute_regime_accuracy(acc_horizon)
            current_regime_data = regime_acc.get(regime, {})
            for sig_name, rdata in current_regime_data.items():
                if rdata.get("total", 0) >= 30:
                    accuracy_data[sig_name] = rdata
        except Exception:
            logger.debug("Regime-conditional accuracy unavailable", exc_info=True)

    # BUG-158: Override global accuracy with per-ticker accuracy for ALL signals.
    # Per-ticker variance is enormous: fear_greed is 93.8% on XAG-USD but 25.9%
    # globally. Using global accuracy throws away alpha on specific instruments.
    # H3: Skip when primary load failed to preserve fail-closed gate.
    _PER_TICKER_MIN_SAMPLES = 30
    if not _accuracy_failed and _ticker_acc_data:
        for sig_name, t_stats in _ticker_acc_data.items():
            if t_stats.get("total", 0) >= _PER_TICKER_MIN_SAMPLES:
                override = {
                    "accuracy": t_stats["accuracy"],
                    "total": t_stats["total"],
                    "correct": t_stats.get("correct", 0),
                    "pct": t_stats.get("pct", round(t_stats["accuracy"] * 100, 1)),
                }
                # Copy directional fields for per-ticker directional gating.
                # Without these, _weighted_consensus directional gate falls back
                # to overall per-ticker accuracy, missing direction-specific
                # weaknesses (e.g., ministral BUY 15% on XAG even if overall 20%).
                for field in ("correct_buy", "total_buy", "buy_accuracy",
                              "correct_sell", "total_sell", "sell_accuracy"):
                    if field in t_stats:
                        override[field] = t_stats[field]
                accuracy_data[sig_name] = override
    elif not _accuracy_failed:
        # Fallback: LLM-specific per-ticker data from extra_info
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
    # Utility boost and best-horizon overlay.
    # H3: Skip when primary load failed to preserve fail-closed gate.
    if not _accuracy_failed:
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

    if ticker:
        _record_phase(ticker, "utility_overlay", _phase_start)
        _phase_start = time.monotonic()

    # Multi-horizon: optionally use each signal's best horizon accuracy.
    # H3: Skip when primary load failed to preserve fail-closed gate.
    sig_cfg = (config or {}).get("signals", {})
    if not _accuracy_failed and sig_cfg.get("use_best_horizon", False):
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

    # Signal persistence filter: only let signals that maintained their vote
    # for 2+ consecutive cycles participate in consensus. Raw votes are kept
    # intact for accuracy tracking (signal_log records unfiltered votes).
    consensus_votes = _apply_persistence_filter(votes, ticker)
    # Track how many signals were filtered for debugging
    _filtered_count = sum(
        1 for s in votes
        if votes[s] != "HOLD" and consensus_votes.get(s) == "HOLD"
    )
    if _filtered_count > 0:
        extra_info["_persistence_filtered"] = _filtered_count

    # Macro-window force-HOLD has already been applied to `votes` above
    # (before buy/sell/core_active counting), so `consensus_votes` —
    # which derives from `votes` via persistence filter — already
    # carries the suppression. No additional mutation needed here.
    # `_weighted_consensus` runs its own pre-pass as defense-in-depth.

    # BUG-224: compute post-persistence voter count so downstream consumers
    # (accuracy tracking, Layer 2) see the actual participating voter count,
    # not the inflated pre-filter number.
    post_persistence_voters = sum(
        1 for v in consensus_votes.values() if v in ("BUY", "SELL")
    )

    weighted_action, weighted_conf = _weighted_consensus(
        consensus_votes, accuracy_data, regime, activation_rates,
        accuracy_gate=accuracy_gate,
        max_signals=max_signals,
        horizon=horizon,
        regime_gated_override=regime_gated_effective,
        ticker=ticker,
        # 2026-05-11 (Codex Fix B): pass extra_info so _weighted_consensus
        # can dampen ema/bb/macd soft votes by their _soft_conf_* values.
        soft_confidences=extra_info,
    )

    if ticker:
        _record_phase(ticker, "weighted_consensus", _phase_start)
        _phase_start = time.monotonic()

    # BUG-227: Apply core gate AND MIN_VOTERS gate to weighted consensus.
    # Use post_persistence_voters (not pre-filter active_voters) because the
    # persistence filter may have reduced voters below the threshold.
    if core_active == 0 or post_persistence_voters < min_voters:
        weighted_action = "HOLD"
        weighted_conf = 0.0

    # Confluence score
    confluence = _confluence_score(votes, extra_info)

    # Time-of-day confidence adjustment
    tod_factor = _time_of_day_factor(horizon=horizon)
    weighted_conf *= tod_factor

    # Store raw consensus in extra for debugging, then use weighted as primary
    extra_info["_raw_action"] = action
    extra_info["_raw_confidence"] = conf
    extra_info["_voters"] = active_voters  # pre-filter (compatibility)
    extra_info["_voters_post_filter"] = post_persistence_voters
    extra_info["_total_applicable"] = total_applicable
    extra_info["_buy_count"] = buy
    extra_info["_sell_count"] = sell
    extra_info["_core_buy"] = core_buy
    extra_info["_core_sell"] = core_sell
    extra_info["_core_active"] = core_active
    extra_info["_votes"] = votes
    extra_info["_raw_votes"] = raw_votes  # C10: pre-gate votes for accuracy recovery
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

    if ticker:
        _record_phase(ticker, "penalties", _phase_start)
        _phase_start = time.monotonic()

    # --- Market health confidence penalty ---
    # Penalizes BUY signals when broad market is unhealthy (distribution days,
    # broken FTD, etc.).  Only affects BUY; SELL and HOLD pass through.
    try:
        from portfolio.market_health import get_confidence_penalty, get_market_health
        mh = get_market_health()
        mh_mult = get_confidence_penalty(action, mh)
        if mh_mult != 1.0:
            conf *= mh_mult
            extra_info.setdefault("_penalty_log", []).append(
                {"stage": "market_health", "zone": mh.get("zone") if mh else "unknown",
                 "score": mh.get("score") if mh else None, "mult": mh_mult}
            )
    except Exception:
        logger.debug("Market health penalty failed", exc_info=True)

    # --- Earnings proximity gate (stocks only) ---
    # Force HOLD if ticker has earnings within 2 calendar days.
    if ticker and action != "HOLD":
        try:
            from portfolio.earnings_calendar import should_gate_earnings
            if should_gate_earnings(ticker):
                extra_info.setdefault("_penalty_log", []).append(
                    {"stage": "earnings_gate", "ticker": ticker, "effect": "force_hold"}
                )
                extra_info["_earnings_gated"] = True
                action = "HOLD"
                conf = 0.0
        except Exception:
            logger.debug("Earnings gate failed for %s", ticker, exc_info=True)

    # --- Linear factor model score (supplementary, not overriding) ---
    # Provides a continuous predicted-return score from ridge regression
    # weights trained on historical signal+return data. Used to:
    # 1. Confirm or weaken consensus (agreement = boost, disagreement = dampen)
    # 2. Provide alternative ranking in agent_summary for Layer 2 decisions
    try:
        from portfolio.linear_factor import LinearFactorModel
        _lf_model = LinearFactorModel()
        if _lf_model.load():
            # Convert votes to numeric: BUY=+1, SELL=-1, HOLD=0
            _vote_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
            numeric_votes = {k: _vote_map.get(v, 0.0) for k, v in votes.items()}
            lf_score = _lf_model.predict(numeric_votes)
            lf_action, lf_conf = _lf_model.score_to_action(lf_score)
            extra_info["_linear_factor_score"] = round(lf_score, 6)
            extra_info["_linear_factor_action"] = lf_action
            extra_info["_linear_factor_confidence"] = lf_conf
            # Confirmation boost/dampen: if linear model agrees with consensus,
            # boost confidence by 10%. If it disagrees, dampen by 10%.
            if lf_action == action and action != "HOLD" and lf_conf > 0.3:
                conf *= 1.10
                extra_info.setdefault("_penalty_log", []).append(
                    {"stage": "linear_factor", "effect": "confirm_boost",
                     "lf_action": lf_action, "lf_score": round(lf_score, 6)})
            elif lf_action != "HOLD" and lf_action != action and action != "HOLD":
                conf *= 0.90
                extra_info.setdefault("_penalty_log", []).append(
                    {"stage": "linear_factor", "effect": "disagree_dampen",
                     "lf_action": lf_action, "lf_score": round(lf_score, 6)})
    except Exception:
        logger.debug("Linear factor model failed", exc_info=True)

    if ticker:
        _record_phase(ticker, "linear_factor", _phase_start)
        _phase_start = time.monotonic()

    # --- Per-ticker consensus accuracy gate ---
    # BUG-164: AMD 24.8%, GOOGL 31.3%, META 34.2% consensus accuracy.
    # The system is actively harmful on these tickers.  Gate consensus
    # when per-ticker historical accuracy is below threshold.
    if ticker and action != "HOLD":
        try:
            from portfolio.accuracy_stats import load_cached_accuracy

            # H1: Match the horizon-scoped key written above.
            _ptc_acc = load_cached_accuracy(f"per_ticker_consensus_{acc_horizon}")
            if _ptc_acc:
                _ptc_stats = _ptc_acc.get(ticker, {})
                _ptc_total = _ptc_stats.get("total", 0)
                _ptc_accuracy = _ptc_stats.get("accuracy", 0.5)
                if _ptc_total >= _PER_TICKER_CONSENSUS_MIN_SAMPLES and _ptc_accuracy < _PER_TICKER_CONSENSUS_GATE:
                    extra_info.setdefault("_penalty_log", []).append({
                        "stage": "per_ticker_consensus_gate",
                        "ticker": ticker,
                        "accuracy": round(_ptc_accuracy, 3),
                        "samples": _ptc_total,
                        "threshold": _PER_TICKER_CONSENSUS_GATE,
                        "effect": "force_hold",
                    })
                    extra_info["_ticker_consensus_gated"] = True
                    action = "HOLD"
                    conf = 0.0
        except Exception:
            logger.debug("Per-ticker consensus gate failed for %s", ticker, exc_info=True)

    # Global confidence cap — calibration data shows >80% confidence is
    # anti-correlated with accuracy at every horizon (70-80% bucket is the
    # best performing at 57-59% actual accuracy)
    conf = min(conf, 0.80)

    # 3h horizon: cap confidence to prevent overconfident short-term predictions
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import CONFIDENCE_CAP_3H
        conf = min(conf, CONFIDENCE_CAP_3H)

    if ticker:
        _record_phase(ticker, "consensus_gate", _phase_start)

    # Update cross-ticker consensus cache for synthetic cross-asset signals
    if ticker:
        with _cross_ticker_lock:
            _cross_ticker_consensus[ticker] = {"action": action, "confidence": conf}

    return action, conf, extra_info
