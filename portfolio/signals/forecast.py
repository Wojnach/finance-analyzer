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
import subprocess
import platform
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio.signal_utils import majority_vote
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.signals.forecast")

# Cache TTL — forecasts don't change fast
_FORECAST_TTL = 300  # 5 minutes

# Confidence cap (same as news_event, econ_calendar)
_MAX_CONFIDENCE = 0.7

# Default Chronos timeout (seconds) — prevents hangs
_CHRONOS_TIMEOUT = 60

# Default Kronos subprocess timeout (seconds) — lower since it usually fails fast
_KRONOS_TIMEOUT = 30

# Forecast models master switch. Set to True to disable all model calls (early-return HOLD).
# Circuit breakers remain as secondary protection — auto-trip on failure, 5min TTL.
_FORECAST_MODELS_DISABLED = False

# Kronos inference script — runs via subprocess calling Q:/models/kronos_infer.py
# Enable via config.json → forecast.kronos_enabled = true
_KRONOS_ENABLED = False

def _init_kronos_enabled():
    """Read kronos_enabled from config.json at import time."""
    global _KRONOS_ENABLED
    try:
        import json as _json
        _cfg = _json.load(open(Path(__file__).resolve().parent.parent.parent / "config.json"))
        _KRONOS_ENABLED = bool(_cfg.get("forecast", {}).get("kronos_enabled", False))
    except Exception:
        pass

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
_CIRCUIT_BREAKER_TTL = 300  # 5 minutes before retry
_kronos_tripped_until = 0.0  # monotonic timestamp when breaker resets
_chronos_tripped_until = 0.0

# Prediction dedup — track last logged timestamp per ticker to avoid
# logging cached replays. Key: ticker, value: ISO-8601 timestamp.
_PREDICTION_DEDUP_TTL = 60  # seconds — don't re-log within this window
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
            return json.loads(text[brace_idx:])
        except json.JSONDecodeError:
            pass

    # Last resort: scan lines in reverse for a JSON line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return None


def _kronos_circuit_open() -> bool:
    return time.monotonic() < _kronos_tripped_until


def _trip_kronos():
    global _kronos_tripped_until
    _kronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
    logger.warning("Kronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)


def _chronos_circuit_open() -> bool:
    return time.monotonic() < _chronos_tripped_until


def _trip_chronos():
    global _chronos_tripped_until
    _chronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
    logger.warning("Chronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)


def reset_circuit_breakers():
    """Reset both circuit breakers (for testing or manual recovery)."""
    global _kronos_tripped_until, _chronos_tripped_until
    _kronos_tripped_until = 0.0
    _chronos_tripped_until = 0.0


def _log_health(model: str, ticker: str, success: bool, duration_ms: int, error: str = ""):
    """Append a line to forecast_health.jsonl for persistent success/failure tracking."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "ticker": ticker,
            "ok": success,
            "ms": duration_ms,
        }
        if error:
            entry["error"] = error[:200]
        with open(_HEALTH_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # health logging must never break the signal


def _load_candles_ohlcv(ticker: str, periods: int = 168) -> list[dict] | None:
    """Load recent 1h OHLCV candles as list of dicts.

    Reuses data sources from forecast_signal._load_candles but returns
    full OHLCV dicts instead of just close prices.
    """
    from portfolio.tickers import SYMBOLS

    source_info = SYMBOLS.get(ticker, {})
    try:
        if "binance" in source_info:
            from portfolio.data_collector import binance_klines
            symbol = source_info["binance"]
            df = binance_klines(symbol, interval="1h", limit=periods)
        elif "binance_fapi" in source_info:
            from portfolio.data_collector import binance_fapi_klines
            symbol = source_info["binance_fapi"]
            df = binance_fapi_klines(symbol, interval="1h", limit=periods)
        elif "alpaca" in source_info:
            from portfolio.data_collector import alpaca_klines
            symbol = source_info["alpaca"]
            df = alpaca_klines(symbol, interval="1h", limit=periods)
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
        logger.debug("OHLCV fetch failed for %s: %s", ticker, e)

    return None


def _run_kronos(candles: list[dict], horizons: tuple = (1, 24), _ticker: str = "") -> dict | None:
    """Run Kronos inference via subprocess."""
    if not _KRONOS_ENABLED:
        return None
    if _kronos_circuit_open():
        return None
    t0 = time.time()
    try:
        input_data = json.dumps({
            "candles": candles,
            "prices_close": [c["close"] for c in candles],
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
    """Run Chronos forecast (in-process, lazy-loaded) with timeout protection."""
    if _chronos_circuit_open():
        return None
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
            return get_all_ticker_accuracies(horizon="24h", days=7)
        except Exception as e:
            logger.debug("Failed to load forecast accuracy: %s", e)
            return {}

    return _cached("forecast_ticker_accuracy", ttl, _fetch)


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
        except Exception:
            pass

    # Load candles
    cache_key = f"forecast_candles_{ticker}"
    candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)

    if not candles or len(candles) < 50:
        # Fallback to df close prices if available
        if df is not None and len(df) >= 50 and "close" in df.columns:
            close_prices = df["close"].values.tolist()
        else:
            result["indicators"]["error"] = "insufficient_candle_data"
            return result
    else:
        close_prices = [c["close"] for c in candles]

    current_price = close_prices[-1]
    result["indicators"]["current_price"] = current_price
    result["indicators"]["candle_count"] = len(close_prices)
    result["indicators"]["kronos_circuit_open"] = _kronos_circuit_open()
    result["indicators"]["chronos_circuit_open"] = _chronos_circuit_open()

    # Run Kronos (skip entirely if circuit breaker is open)
    t0 = time.time()
    kronos_key = f"kronos_forecast_{ticker}"
    kronos = _cached(kronos_key, _FORECAST_TTL, _run_kronos, candles or [], (1, 24), ticker)
    kronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["kronos_time_ms"] = kronos_ms

    if kronos and kronos.get("results"):
        kr = kronos["results"]
        result["indicators"]["kronos_method"] = kronos.get("method", "unknown")

        if "1h" in kr:
            result["sub_signals"]["kronos_1h"] = _direction_to_action(kr["1h"].get("direction", "neutral"))
            result["indicators"]["kronos_1h_pct"] = kr["1h"].get("pct_move", 0)
            result["indicators"]["kronos_1h_conf"] = kr["1h"].get("confidence", 0)

        if "24h" in kr:
            result["sub_signals"]["kronos_24h"] = _direction_to_action(kr["24h"].get("direction", "neutral"))
            result["indicators"]["kronos_24h_pct"] = kr["24h"].get("pct_move", 0)
            result["indicators"]["kronos_24h_conf"] = kr["24h"].get("confidence", 0)

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

    # Accuracy-weighted vote — per-ticker accuracy gating + health exclusion
    kronos_ok = kronos is not None and bool(kronos.get("results"))
    chronos_ok = chronos is not None
    result["indicators"]["kronos_ok"] = kronos_ok
    result["indicators"]["chronos_ok"] = chronos_ok

    # Compute ATR% for volatility gate
    atr_pct = _compute_atr_pct(close_prices)
    result["indicators"]["forecast_atr_pct"] = round(atr_pct, 4) if atr_pct else None

    config_forecast = (context or {}).get("config", {}).get("forecast", {})
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
    try:
        now_mono = time.monotonic()
        last_ts = _last_prediction_ts.get(ticker, 0.0)
        if now_mono - last_ts >= _PREDICTION_DEDUP_TTL:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "current_price": current_price,
                "sub_signals": result["sub_signals"],
                "action": result["action"],
                "confidence": result["confidence"],
                "per_ticker_accuracy": gating_info.get("forecast_accuracy"),
                "gating_action": gating_info.get("forecast_gating", "raw"),
            }
            if kronos and kronos.get("results"):
                entry["kronos"] = kronos["results"]
            if chronos:
                entry["chronos"] = chronos
            with open(_PREDICTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _last_prediction_ts[ticker] = now_mono
    except Exception:
        logger.debug("Failed to log forecast prediction", exc_info=True)

    return result
