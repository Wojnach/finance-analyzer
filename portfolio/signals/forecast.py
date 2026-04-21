"""Forecast signal — price direction prediction from time-series models.

Combines Kronos (K-line foundation model), Chronos (Amazon), and Prophet
into a majority-vote composite signal with four sub-signals:
  1. kronos_1h  — Kronos 1-hour prediction
  2. kronos_24h — Kronos 24-hour prediction
  3. chronos_1h — Chronos 1-hour prediction
  4. chronos_24h — Chronos 24-hour prediction

Reuses candle loading from portfolio.forecast_signal. Confidence capped at 0.7.
Registered as enhanced signal #28 with weight=0 (shadow mode) initially.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from portfolio.file_utils import atomic_append_jsonl
from portfolio.gpu_gate import gpu_gate
from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.forecast")

# Cache TTL — forecasts don't change fast
_FORECAST_TTL = 300  # 5 minutes

# Confidence cap (same as news_event, econ_calendar)
_MAX_CONFIDENCE = 0.7

# Default Chronos timeout (seconds) — reduced from 120 to avoid long hangs
_CHRONOS_TIMEOUT = 60

# Default Kronos subprocess timeout (seconds) — reduced from 90; fails fast
_KRONOS_TIMEOUT = 30

# Forecast models master switch. Set to True to disable all model calls (early-return HOLD).
# Circuit breakers remain as secondary protection — auto-trip on failure, 5min TTL.
_FORECAST_MODELS_DISABLED = False

# Kronos inference — RETIRED 2026-04-21.
# Why: subprocess success rate collapsed to 59.2% (3250 ok / 2241 fail out of
# 5491 attempts over 30d) due to VRAM contention with Chronos/Ministral/Qwen3
# sharing gpu_gate, plus stdout contamination from HuggingFace loading that
# _extract_json_from_stdout can only partially scrub. Separately, the shadow
# mode (config value "shadow") forced every successful prediction to HOLD at
# lines ~811/820, so over 3668 logged predictions not one contributed a raw
# BUY/SELL vote to the composite. Net effect: Kronos was HOLD-diluting the
# Chronos-only useful signal (3 of 6 slots in _health_weighted_vote when
# kronos_ok was True).
# Decision: permanently disable at runtime. Subprocess code (_run_kronos,
# _run_kronos_inner) kept in place so existing tests continue to run, and so
# a future session can re-evaluate if the model is retrained or moved to a
# dedicated venv. Config flag is now ignored — override via direct monkey-
# patch of _KRONOS_ENABLED in tests only.
_KRONOS_ENABLED = False
_KRONOS_SHADOW = False


def _init_kronos_enabled():
    """No-op since 2026-04-21 Kronos retire. Kept as a named function so tests
    can still call it without error; the only effect is to reassert the
    disabled state."""
    global _KRONOS_ENABLED, _KRONOS_SHADOW
    _KRONOS_ENABLED = False
    _KRONOS_SHADOW = False


_init_kronos_enabled()

if platform.system() == "Windows":
    _KRONOS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    _KRONOS_SCRIPT = r"Q:\models\kronos_infer.py"
else:
    _KRONOS_PYTHON = "/home/deck/models/.venv/bin/python"
    _KRONOS_SCRIPT = "/home/deck/models/kronos_infer.py"

# Prediction log
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PREDICTIONS_FILE = _DATA_DIR / "forecast_predictions.jsonl"
_HEALTH_FILE = _DATA_DIR / "forecast_health.jsonl"

# Circuit breaker — after first failure, skip remaining tickers in this loop cycle.
# Prevents 27 x 6s GPU timeouts when CUDA is broken.
_CIRCUIT_BREAKER_TTL = 30  # 30 seconds before retry
_kronos_tripped_until = 0.0  # monotonic timestamp when breaker resets
_chronos_tripped_until = 0.0

# BUG-102: Lock protects circuit breaker state and dedup cache from ThreadPoolExecutor races.
# The read-check-write pattern in _log_health() is not atomic without a lock.
_forecast_lock = threading.Lock()

# Prediction dedup — track last logged timestamp per ticker to avoid
# logging cached replays. Key: ticker, value: ISO-8601 timestamp.
_PREDICTION_DEDUP_TTL = 60  # seconds — don't re-log within this window
_PREDICTION_DEDUP_EVICT_AGE = 600  # BUG-106: evict entries older than 10 minutes
_last_prediction_ts: dict[str, float] = {}  # ticker -> monotonic timestamp


def _extract_json_from_stdout(stdout: str | None) -> dict | None:
    """Extract JSON from potentially contaminated subprocess stdout.

    HuggingFace's from_pretrained() prints to stdout during model loading,
    which contaminates the subprocess output before the JSON result.
    This function handles that by finding the first '{' and parsing from there.

    Returns parsed dict on success, None on failure.
    """
    if not stdout:
        return None

    text = stdout.strip()
    if not text:
        return None

    # Fast path: stdout starts with '{' — clean JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Slow path: find first '{' and try parsing from there
    brace_idx = text.find("{")
    if brace_idx > 0:
        try:
            parsed = json.loads(text[brace_idx:])
            logger.debug("JSON extracted via brace-offset fallback (offset=%d, len=%d)", brace_idx, len(text))
            return parsed
        except json.JSONDecodeError:
            pass

    # Last resort: scan lines in reverse for a JSON line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                logger.debug("JSON extracted via reverse-line-scan fallback (len=%d)", len(text))
                return parsed
            except json.JSONDecodeError:
                continue

    logger.debug("JSON extraction failed — all 3 strategies exhausted (len=%d)", len(text) if text else 0)
    return None


def _kronos_circuit_open() -> bool:
    with _forecast_lock:
        return time.monotonic() < _kronos_tripped_until


def _trip_kronos():
    global _kronos_tripped_until
    with _forecast_lock:
        _kronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
    logger.warning("Kronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)


def _chronos_circuit_open() -> bool:
    with _forecast_lock:
        return time.monotonic() < _chronos_tripped_until


def _trip_chronos():
    global _chronos_tripped_until
    with _forecast_lock:
        _chronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
    logger.warning("Chronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)


def reset_circuit_breakers():
    """Reset both circuit breakers (for testing or manual recovery)."""
    global _kronos_tripped_until, _chronos_tripped_until
    with _forecast_lock:
        _kronos_tripped_until = 0.0
        _chronos_tripped_until = 0.0


def _log_health(model: str, ticker: str, success: bool, duration_ms: int, error: str = ""):
    """Append a line to forecast_health.jsonl for persistent success/failure tracking.

    On success, auto-resets the relevant circuit breaker so recovered models
    resume immediately instead of waiting for the full TTL (BUG-56 fix).
    """
    global _kronos_tripped_until, _chronos_tripped_until
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model,
            "ticker": ticker,
            "ok": success,
            "ms": duration_ms,
        }
        if error:
            entry["error"] = error[:200]
        atomic_append_jsonl(_HEALTH_FILE, entry)
    except Exception as e:
        logger.debug("Forecast health logging failed: %s", e)

    # Auto-reset circuit breaker on success — faster recovery from transient failures
    # BUG-102: Use lock to make read-check-write atomic
    if success:
        with _forecast_lock:
            if model == "kronos" and _kronos_tripped_until > 0:
                _kronos_tripped_until = 0.0
                logger.info("Kronos circuit breaker RESET on successful %s", ticker)
            elif model == "chronos" and _chronos_tripped_until > 0:
                _chronos_tripped_until = 0.0
                logger.info("Chronos circuit breaker RESET on successful %s", ticker)


def _load_candles_ohlcv(ticker: str, periods: int = 168,
                        interval: str = "1h") -> list[dict] | None:
    """Load recent OHLCV candles as list of dicts.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD")
        periods: Number of candles to fetch
        interval: Candle interval ("1h", "5m", "15m", etc.)
    """
    from portfolio.tickers import SYMBOLS

    source_info = SYMBOLS.get(ticker, {})

    # Determine the data source — needed to apply source-specific interval constraints
    if "binance" in source_info:
        source = "binance"
    elif "binance_fapi" in source_info:
        source = "binance_fapi"
    elif "alpaca" in source_info:
        source = "alpaca"
    else:
        source = None

    # Alpaca minimum supported interval is 15m — fall back if configured interval is smaller.
    # alpaca_klines() does its own mapping; pass the raw internal interval directly.
    if source == "alpaca" and interval in ("1m", "3m", "5m"):
        logger.debug(
            "Alpaca does not support %s interval for %s — falling back to 15m", interval, ticker
        )
        interval = "15m"

    try:
        if source == "binance":
            from portfolio.data_collector import binance_klines
            symbol = source_info["binance"]
            df = binance_klines(symbol, interval=interval, limit=periods)
        elif source == "binance_fapi":
            from portfolio.data_collector import binance_fapi_klines
            symbol = source_info["binance_fapi"]
            df = binance_fapi_klines(symbol, interval=interval, limit=periods)
        elif source == "alpaca":
            # Pass the raw internal interval — alpaca_klines() handles the mapping itself.
            from portfolio.data_collector import alpaca_klines
            symbol = source_info["alpaca"]
            df = alpaca_klines(symbol, interval=interval, limit=periods)
        else:
            return None

        if df is not None and len(df) > 30:
            candles = []
            for _, row in df.iterrows():
                candles.append({
                    "open": float(row.get("open", row.get("close", 0))),
                    "high": float(row.get("high", row.get("close", 0))),
                    "low": float(row.get("low", row.get("close", 0))),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
            return candles
    except Exception as e:
        logger.debug("OHLCV fetch failed for %s (interval=%s): %s", ticker, interval, e)

    return None


def _run_kronos(candles: list[dict], horizons: tuple = (1, 24), _ticker: str = "") -> dict | None:
    """Run Kronos inference via subprocess with GPU gating."""
    if not _KRONOS_ENABLED:
        return None
    if _kronos_circuit_open():
        return None
    t0 = time.time()
    try:
        with gpu_gate("kronos", timeout=90) as acquired:
            if not acquired:
                logger.warning("GPU gate timeout for Kronos %s", _ticker)
                return None
            return _run_kronos_inner(candles, horizons, _ticker, t0)
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        logger.warning("Kronos GPU gate error: %s", e)
        _log_health("kronos", _ticker, False, ms, str(e)[:200])
        _trip_kronos()
        return None


def _run_kronos_inner(candles, horizons, _ticker, t0):
    """Kronos inference (called inside GPU gate)."""
    try:
        # Read tunable params from config
        try:
            from portfolio.file_utils import load_json
            cfg = load_json(str(Path(__file__).resolve().parent.parent.parent / "config.json"), {})
            fc = cfg.get("forecast", {})
        except Exception:
            fc = {}

        input_data = json.dumps({
            "candles": candles,
            "prices_close": [c["close"] for c in candles],
            "temperature": fc.get("kronos_temperature", 1.0),
            "top_p": fc.get("kronos_top_p", 0.9),
            "sample_count": fc.get("kronos_samples", 3),
        })
        proc = subprocess.run(
            [_KRONOS_PYTHON, _KRONOS_SCRIPT,
             "--horizons", ",".join(str(h) for h in horizons)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=_KRONOS_TIMEOUT,
        )
        ms = round((time.time() - t0) * 1000)
        if proc.returncode != 0:
            err = proc.stderr[:200]
            logger.warning("Kronos subprocess failed: %s", err)
            _log_health("kronos", _ticker, False, ms, err)
            _trip_kronos()
            return None
        if not proc.stdout or not proc.stdout.strip():
            _log_health("kronos", _ticker, False, ms, "empty_stdout")
            _trip_kronos()
            return None
        result = _extract_json_from_stdout(proc.stdout)
        if result is None:
            # JSON extraction failed — log actual stdout for diagnostics
            preview = repr(proc.stdout[:200])
            logger.warning("Kronos stdout not valid JSON for %s: %s", _ticker, preview)
            _log_health("kronos", _ticker, False, ms, f"json_extract_failed: {preview[:150]}")
            _trip_kronos()
            return None
        if not result or not result.get("results"):
            _log_health("kronos", _ticker, False, ms, "empty_results")
            _trip_kronos()
            return None
        _log_health("kronos", _ticker, True, ms)
        return result
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        logger.warning("Kronos subprocess error (v2): %s", e)
        _log_health("kronos", _ticker, False, ms, str(e)[:200])
        _trip_kronos()
        return None


def _run_chronos(prices: list[float], horizons: tuple = (1, 24), _ticker: str = "",
                 timeout: int | None = None) -> dict | None:
    """Run Chronos forecast (in-process, lazy-loaded) with GPU gating and timeout."""
    if _chronos_circuit_open():
        return None

    with gpu_gate("chronos", timeout=120) as acquired:
        if not acquired:
            logger.warning("GPU gate timeout for Chronos %s", _ticker)
            return None
        return _run_chronos_inner(prices, horizons, _ticker, timeout)


def _run_chronos_inner(prices, horizons, _ticker, timeout):
    """Chronos inference (called inside GPU gate)."""
    t0 = time.time()
    _timeout = timeout or _CHRONOS_TIMEOUT
    try:
        from portfolio.forecast_signal import forecast_chronos

        # Run in thread with timeout to prevent hangs
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(forecast_chronos, "", prices, horizons=horizons)
            try:
                result = future.result(timeout=_timeout)
            except FuturesTimeout:
                ms = round((time.time() - t0) * 1000)
                logger.warning("Chronos timed out after %ds for %s", _timeout, _ticker)
                _log_health("chronos", _ticker, False, ms, f"timeout_{_timeout}s")
                _trip_chronos()
                return None

        ms = round((time.time() - t0) * 1000)
        if result is None:
            _log_health("chronos", _ticker, False, ms, "returned_none")
            _trip_chronos()
        else:
            _log_health("chronos", _ticker, True, ms)
        return result
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        logger.warning("Chronos failed: %s", e)
        _log_health("chronos", _ticker, False, ms, str(e)[:200])
        _trip_chronos()
        return None


def _health_weighted_vote(sub_signals, kronos_ok, chronos_ok):
    """Vote only using sub-signals from healthy (working) models.

    When Kronos is dead (99.5% failure rate), its 2 permanent HOLD votes
    dilute the 4-vote majority and make the signal always return HOLD.
    This function excludes dead models from the vote.

    1h horizon gets 2x weight (counted twice) because short-term predictions
    are more actionable and Chronos 24h predictions are less reliable.
    """
    alive_votes = []
    if kronos_ok:
        # 1h gets double weight
        alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("kronos_24h", "HOLD"))
    if chronos_ok:
        # 1h gets double weight
        alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
        alive_votes.append(sub_signals.get("chronos_24h", "HOLD"))

    if not alive_votes:
        return "HOLD", 0.0

    return majority_vote(alive_votes)


# Per-ticker accuracy cache TTL
_ACCURACY_CACHE_TTL = 1800  # 30 minutes

# Default thresholds for accuracy gating
_HOLD_THRESHOLD = 0.55        # Below this: force HOLD (signal can't predict)
_MIN_SAMPLES = 10             # Below this: use raw vote (insufficient data)

# Volatility gate — force HOLD when ATR% exceeds threshold
# Chronos predicts negligible moves (~0.1% avg), so high-volatility environments
# where actual moves are 3-5% make the signal useless.
_VOL_GATE_CRYPTO = 0.03       # 3% ATR for crypto
_VOL_GATE_DEFAULT = 0.02      # 2% ATR for metals/stocks

# Regime-aware confidence discount — Chronos has a mean-reversion bias
# (predicts small moves back to mean). In trending markets this is wrong.
_REGIME_DISCOUNT_TRENDING = 0.5   # Halve confidence in trending regimes
_REGIME_DISCOUNT_HIGH_VOL = 0.6   # Reduce confidence in high-vol regimes
_REGIME_NEUTRAL = 1.0             # No discount in ranging/neutral regimes


def _compute_atr_pct(close_prices: list[float], period: int = 14) -> float | None:
    """Compute ATR% from close prices (approximation using close-to-close).

    Returns ATR as fraction of current price (e.g. 0.03 = 3%), or None
    if insufficient data.
    """
    if not close_prices or len(close_prices) < period + 1:
        return None
    # Approximate true range from close-to-close changes
    trs = [abs(close_prices[i] - close_prices[i - 1]) for i in range(1, len(close_prices))]
    if len(trs) < period:
        return None
    # EMA-smoothed ATR over last `period` values
    recent_trs = trs[-period * 2:]  # use more data for EMA warmup
    atr = recent_trs[0]
    alpha = 2.0 / (period + 1)
    for tr in recent_trs[1:]:
        atr = alpha * tr + (1 - alpha) * atr
    current = close_prices[-1]
    if current <= 0:
        return None
    return atr / current


def _is_crypto_ticker(ticker: str) -> bool:
    """Check if ticker is crypto (BTC-USD, ETH-USD)."""
    try:
        from portfolio.tickers import CRYPTO_SYMBOLS
        return ticker in CRYPTO_SYMBOLS
    except ImportError:
        return ticker in {"BTC-USD", "ETH-USD"}


def _load_forecast_accuracy(cache_ttl=None):
    """Load per-ticker forecast accuracy, cached via _cached().

    Returns dict: {ticker: {accuracy, samples}} or empty dict on error.
    """
    ttl = cache_ttl or _ACCURACY_CACHE_TTL

    def _fetch():
        try:
            from portfolio.forecast_accuracy import get_all_ticker_accuracies
            return get_all_ticker_accuracies(horizon="24h", days=14)
        except Exception as e:
            logger.debug("Failed to load forecast accuracy: %s", e)
            return {}

    return _cached("forecast_ticker_accuracy", ttl, _fetch)


def _load_forecast_subsignal_accuracy(cache_ttl=None, days=30):
    """Load raw sub-signal accuracy for 1h and 24h forecast votes."""
    ttl = cache_ttl or _ACCURACY_CACHE_TTL
    cache_key = f"forecast_subsignal_accuracy_{days}"

    def _fetch():
        try:
            from portfolio.forecast_accuracy import compute_forecast_accuracy

            return {
                "1h": compute_forecast_accuracy(
                    horizon="1h", days=days, use_raw_sub_signals=True
                ),
                "24h": compute_forecast_accuracy(
                    horizon="24h", days=days, use_raw_sub_signals=True
                ),
            }
        except Exception as e:
            logger.debug("Failed to load forecast sub-signal accuracy: %s", e)
            return {}

    return _cached(cache_key, ttl, _fetch)


def _gate_subsignal_votes_by_accuracy(sub_signals, ticker, config_forecast=None):
    """Gate individual forecast sub-signals using raw historical accuracy."""
    cfg = config_forecast or {}
    hold_threshold = cfg.get("subsignal_hold_threshold", cfg.get("hold_threshold", _HOLD_THRESHOLD))
    min_samples = cfg.get("subsignal_min_samples", cfg.get("min_samples", _MIN_SAMPLES))
    lookback_days = cfg.get("subsignal_accuracy_days", 30)
    cache_ttl = cfg.get("subsignal_accuracy_cache_ttl", _ACCURACY_CACHE_TTL)

    gated = dict(sub_signals)
    info = {}
    if not ticker:
        return gated, info

    accuracy_matrix = _load_forecast_subsignal_accuracy(cache_ttl=cache_ttl, days=lookback_days)
    for sub_name, vote in sub_signals.items():
        if vote == "HOLD":
            continue

        horizon = "1h" if sub_name.endswith("_1h") else "24h"
        horizon_stats = ((accuracy_matrix or {}).get(horizon) or {}).get(sub_name) or {}
        ticker_stats = (horizon_stats.get("by_ticker") or {}).get(ticker)

        accuracy = None
        samples = 0
        source = None
        if ticker_stats and ticker_stats.get("total", 0) >= min_samples:
            accuracy = float(ticker_stats["accuracy"])
            samples = int(ticker_stats["total"])
            source = "ticker"
        elif horizon_stats.get("total", 0) >= min_samples:
            accuracy = float(horizon_stats["accuracy"])
            samples = int(horizon_stats["total"])
            source = "global"

        gating = "insufficient_data"
        if accuracy is not None:
            gating = "held" if accuracy < hold_threshold else "raw"
            if gating == "held":
                gated[sub_name] = "HOLD"

        info[sub_name] = {
            "gating": gating,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "samples": samples,
            "source": source,
        }

    return gated, info


def _regime_discount(regime: str, config_forecast: dict | None = None) -> float:
    """Return confidence multiplier based on market regime.

    Chronos has a mean-reversion bias — it predicts small moves back to mean.
    In trending markets, this is wrong, so we discount confidence.
    """
    cfg = config_forecast or {}
    if not regime:
        return _REGIME_NEUTRAL
    r = regime.lower()
    if r in ("trending-up", "trending-down", "breakout"):
        return cfg.get("regime_discount_trending", _REGIME_DISCOUNT_TRENDING)
    elif r in ("high-vol", "capitulation"):
        return cfg.get("regime_discount_high_vol", _REGIME_DISCOUNT_HIGH_VOL)
    else:
        # range-bound, neutral — mean-reversion is appropriate
        return _REGIME_NEUTRAL


def _accuracy_weighted_vote(sub_signals, kronos_ok, chronos_ok, ticker="",
                            config_forecast=None, atr_pct=None, regime=None):
    """Vote with per-ticker accuracy gating, volatility gate, and regime discount.

    Extends _health_weighted_vote with:
    - Volatility gate: high ATR% → force HOLD (Chronos can't predict big moves)
    - Regime discount: trending markets → reduce confidence (mean-reversion bias)
    - Accuracy gate: accuracy < hold_threshold → force HOLD
    - Good accuracy: use raw vote, scale confidence by accuracy
    - Insufficient samples: use raw vote (not enough data to judge)

    Bad tickers abstain (HOLD) rather than invert — inversion games the
    accuracy metric without fixing the underlying prediction quality.

    Returns (action, confidence, gating_info) where gating_info is a dict with
    accuracy metadata for logging.
    """
    cfg = config_forecast or {}
    hold_thresh = cfg.get("hold_threshold", _HOLD_THRESHOLD)
    min_samples = cfg.get("min_samples", _MIN_SAMPLES)

    # Start with health-weighted vote as baseline
    base_action, base_conf = _health_weighted_vote(
        sub_signals, kronos_ok, chronos_ok
    )

    gating_info = {
        "forecast_accuracy": None,
        "forecast_samples": 0,
        "forecast_gating": "raw",
        "forecast_inverted": False,
        "base_action": base_action,
        "base_confidence": base_conf,
        "atr_pct": atr_pct,
    }

    if not ticker:
        return base_action, base_conf, gating_info

    # Volatility gate — Chronos predicts negligible moves (~0.1% avg),
    # so high-vol environments make the signal useless
    if atr_pct is not None:
        vol_thresh = cfg.get("vol_gate_crypto", _VOL_GATE_CRYPTO) \
            if _is_crypto_ticker(ticker) \
            else cfg.get("vol_gate_default", _VOL_GATE_DEFAULT)
        if atr_pct > vol_thresh:
            gating_info["forecast_gating"] = "vol_gated"
            return "HOLD", 0.0, gating_info

    # Load per-ticker accuracy
    all_acc = _load_forecast_accuracy(cfg.get("accuracy_cache_ttl"))
    ticker_acc = all_acc.get(ticker) if all_acc else None

    if ticker_acc is None or ticker_acc.get("samples", 0) < min_samples:
        gating_info["forecast_gating"] = "insufficient_data"
        if ticker_acc:
            gating_info["forecast_accuracy"] = ticker_acc["accuracy"]
            gating_info["forecast_samples"] = ticker_acc["samples"]
        return base_action, base_conf, gating_info

    acc = ticker_acc["accuracy"]
    samples = ticker_acc["samples"]
    gating_info["forecast_accuracy"] = acc
    gating_info["forecast_samples"] = samples

    if acc < hold_thresh:
        # Below threshold — signal can't predict this ticker, abstain
        gating_info["forecast_gating"] = "held"
        return "HOLD", 0.0, gating_info

    else:
        # Good accuracy — use raw vote, scale confidence by accuracy
        gating_info["forecast_gating"] = "raw"
        # Apply regime discount (trending → lower confidence for mean-reversion bias)
        r_discount = _regime_discount(regime, cfg)
        gating_info["regime_discount"] = r_discount
        scaled_conf = base_conf * acc * r_discount
        return base_action, min(scaled_conf, _MAX_CONFIDENCE), gating_info


def _direction_to_action(direction: str) -> str:
    """Convert direction string to action."""
    if direction in ("up", "BUY"):
        return "BUY"
    if direction in ("down", "SELL"):
        return "SELL"
    return "HOLD"


def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the composite forecast signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used as fallback if candle fetch fails).
    context : dict, optional
        Dict with keys: ticker, config, macro.

    Returns
    -------
    dict
        action, confidence, sub_signals, indicators
    """
    result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "kronos_1h": "HOLD",
            "kronos_24h": "HOLD",
            "chronos_1h": "HOLD",
            "chronos_24h": "HOLD",
        },
        "indicators": {},
    }

    # Bulletproof early return — skip ALL work when models are disabled
    if _FORECAST_MODELS_DISABLED:
        result["indicators"]["models_disabled"] = True
        return result

    ticker = (context or {}).get("ticker", "")
    if not ticker:
        return result

    # Apply Chronos model config if specified
    chronos_model = (context or {}).get("config", {}).get("forecast", {}).get("chronos_model")
    if chronos_model:
        try:
            from portfolio.forecast_signal import set_chronos_model
            set_chronos_model(chronos_model)
        except Exception as e:
            logger.debug("Chronos model config override failed: %s", e)

    config_forecast = (context or {}).get("config", {}).get("forecast", {})

    # Load candles (1h for Chronos, optionally 5m for Kronos)
    cache_key = f"forecast_candles_{ticker}"
    candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)

    # Load 5m candles for Kronos if configured (more granular context)
    kronos_interval = config_forecast.get("kronos_interval", "1h")
    if kronos_interval != "1h" and _KRONOS_ENABLED:
        kronos_periods = config_forecast.get("kronos_periods", 500)
        kronos_cache_key = f"forecast_candles_{ticker}_{kronos_interval}"
        kronos_candles = _cached(kronos_cache_key, _FORECAST_TTL,
                                  _load_candles_ohlcv, ticker, kronos_periods,
                                  kronos_interval)
    else:
        kronos_candles = None

    if not candles or len(candles) < 50:
        # Fallback to df close prices if available
        if df is not None and len(df) >= 50 and "close" in df.columns:
            close_prices = df["close"].values.tolist()
        else:
            result["indicators"]["error"] = "insufficient_candle_data"
            return result
    else:
        close_prices = [c["close"] for c in candles]

    # If Kronos-specific candle fetch failed but df has full OHLCV data, build candle dicts
    # from the DataFrame so Kronos still gets richer data than just close prices.
    if kronos_candles is None and df is not None and len(df) >= 50:
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        if ohlcv_cols.issubset(df.columns):
            try:
                kronos_candles = [
                    {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                    for _, row in df.iterrows()
                ]
                logger.debug(
                    "Kronos candle fallback from df for %s (%d candles)", ticker, len(kronos_candles)
                )
                result["indicators"]["kronos_candles_source"] = "df_fallback"
            except Exception as e:
                logger.debug("Kronos df candle fallback failed for %s: %s", ticker, e)
                kronos_candles = None

    current_price = close_prices[-1]
    result["indicators"]["current_price"] = current_price
    result["indicators"]["candle_count"] = len(close_prices)
    result["indicators"]["kronos_circuit_open"] = _kronos_circuit_open()
    result["indicators"]["chronos_circuit_open"] = _chronos_circuit_open()

    # Run Kronos — use 5m candles if available, otherwise 1h
    t0 = time.time()
    kronos_key = f"kronos_forecast_{ticker}"
    kronos_input = kronos_candles if kronos_candles and len(kronos_candles) >= 50 else (candles or [])
    kronos = _cached(kronos_key, _FORECAST_TTL, _run_kronos, kronos_input, (1, 24), ticker)
    if kronos_candles and len(kronos_candles) >= 50:
        result["indicators"]["kronos_interval"] = kronos_interval
    kronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["kronos_time_ms"] = kronos_ms

    if kronos and kronos.get("results"):
        kr = kronos["results"]
        result["indicators"]["kronos_method"] = kronos.get("method", "unknown")
        result["indicators"]["kronos_shadow"] = _KRONOS_SHADOW

        if "1h" in kr:
            k1h_action = _direction_to_action(kr["1h"].get("direction", "neutral"))
            # Shadow mode: log the real prediction but vote HOLD
            result["sub_signals"]["kronos_1h"] = "HOLD" if _KRONOS_SHADOW else k1h_action
            result["indicators"]["kronos_1h_raw"] = k1h_action
            result["indicators"]["kronos_1h_pct"] = kr["1h"].get("pct_move", 0)
            result["indicators"]["kronos_1h_conf"] = kr["1h"].get("confidence", 0)
            result["indicators"]["kronos_1h_range_pct"] = kr["1h"].get("predicted_range_pct", 0)
            result["indicators"]["kronos_1h_range_skew"] = kr["1h"].get("range_skew", 0)

        if "24h" in kr:
            k24h_action = _direction_to_action(kr["24h"].get("direction", "neutral"))
            result["sub_signals"]["kronos_24h"] = "HOLD" if _KRONOS_SHADOW else k24h_action
            result["indicators"]["kronos_24h_raw"] = k24h_action
            result["indicators"]["kronos_24h_pct"] = kr["24h"].get("pct_move", 0)
            result["indicators"]["kronos_24h_conf"] = kr["24h"].get("confidence", 0)
            result["indicators"]["kronos_24h_range_pct"] = kr["24h"].get("predicted_range_pct", 0)
            result["indicators"]["kronos_24h_range_skew"] = kr["24h"].get("range_skew", 0)
            result["indicators"]["kronos_24h_predicted_high"] = kr["24h"].get("predicted_high", 0)
            result["indicators"]["kronos_24h_predicted_low"] = kr["24h"].get("predicted_low", 0)

    # Run Chronos (skip entirely if circuit breaker is open)
    t0 = time.time()
    chronos_key = f"chronos_forecast_{ticker}"
    chronos = _cached(chronos_key, _FORECAST_TTL, _run_chronos, close_prices, (1, 24), ticker)
    chronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["chronos_time_ms"] = chronos_ms

    if chronos:
        if "1h" in chronos:
            result["sub_signals"]["chronos_1h"] = chronos["1h"].get("action", "HOLD")
            result["indicators"]["chronos_1h_pct"] = chronos["1h"].get("pct_move", 0)
            result["indicators"]["chronos_1h_conf"] = chronos["1h"].get("confidence", 0)

        if "24h" in chronos:
            result["sub_signals"]["chronos_24h"] = chronos["24h"].get("action", "HOLD")
            result["indicators"]["chronos_24h_pct"] = chronos["24h"].get("pct_move", 0)
            result["indicators"]["chronos_24h_conf"] = chronos["24h"].get("confidence", 0)

    raw_sub_signals = dict(result["sub_signals"])
    gated_sub_signals, subsignal_gating = _gate_subsignal_votes_by_accuracy(
        raw_sub_signals, ticker, config_forecast=config_forecast
    )
    result["sub_signals"] = gated_sub_signals
    result["indicators"]["forecast_subsignal_gating"] = subsignal_gating

    # Accuracy-weighted vote — per-ticker accuracy gating + health exclusion
    kronos_ok = kronos is not None and bool(kronos.get("results"))
    chronos_ok = chronos is not None
    result["indicators"]["kronos_ok"] = kronos_ok
    result["indicators"]["chronos_ok"] = chronos_ok

    # Compute ATR% for volatility gate
    atr_pct = _compute_atr_pct(close_prices)
    result["indicators"]["forecast_atr_pct"] = round(atr_pct, 4) if atr_pct else None

    regime = (context or {}).get("regime", "")
    result["action"], result["confidence"], gating_info = _accuracy_weighted_vote(
        result["sub_signals"], kronos_ok, chronos_ok,
        ticker=ticker, config_forecast=config_forecast,
        atr_pct=atr_pct, regime=regime,
    )

    # Store gating metadata in indicators
    result["indicators"]["forecast_accuracy"] = gating_info.get("forecast_accuracy")
    result["indicators"]["forecast_samples"] = gating_info.get("forecast_samples", 0)
    result["indicators"]["forecast_gating"] = gating_info.get("forecast_gating", "raw")
    result["indicators"]["forecast_inverted"] = gating_info.get("forecast_inverted", False)

    # Cap confidence (already capped inside _accuracy_weighted_vote, but belt-and-suspenders)
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    # Log prediction for accuracy tracking (with dedup)
    # BUG-102: Lock protects _last_prediction_ts from concurrent ThreadPoolExecutor access
    # BUG-106: Evict stale entries to prevent unbounded dict growth
    try:
        now_mono = time.monotonic()
        with _forecast_lock:
            last_ts = _last_prediction_ts.get(ticker, 0.0)
            should_log = now_mono - last_ts >= _PREDICTION_DEDUP_TTL
        if should_log:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "ticker": ticker,
                "current_price": current_price,
                "sub_signals": result["sub_signals"],
                "raw_sub_signals": raw_sub_signals,
                "subsignal_gating": subsignal_gating,
                "action": result["action"],
                "confidence": result["confidence"],
                "per_ticker_accuracy": gating_info.get("forecast_accuracy"),
                "gating_action": gating_info.get("forecast_gating", "raw"),
            }
            if kronos and kronos.get("results"):
                entry["kronos"] = kronos["results"]
            if chronos:
                entry["chronos"] = chronos
            atomic_append_jsonl(_PREDICTIONS_FILE, entry)
            with _forecast_lock:
                _last_prediction_ts[ticker] = now_mono
                # BUG-106: Evict stale entries older than 10 minutes
                stale = [k for k, v in _last_prediction_ts.items()
                         if now_mono - v > _PREDICTION_DEDUP_EVICT_AGE]
                for k in stale:
                    del _last_prediction_ts[k]
    except Exception:
        logger.debug("Failed to log forecast prediction", exc_info=True)

    return result
