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

# Forecast models master switch. Set to True to disable all model calls (early-return HOLD).
# Circuit breakers remain as secondary protection — auto-trip on failure, 5min TTL.
_FORECAST_MODELS_DISABLED = False

# Kronos inference script — runs via subprocess calling Q:/models/kronos_infer.py
_KRONOS_ENABLED = True

if platform.system() == "Windows":
    _KRONOS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    _KRONOS_SCRIPT = r"Q:\models\kronos_infer.py"
else:
    _KRONOS_PYTHON = "/home/deck/models/.venv/bin/python"
    _KRONOS_SCRIPT = "/home/deck/models/kronos_infer.py"

# Prediction log
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PREDICTIONS_FILE = _DATA_DIR / "forecast_predictions.jsonl"

# Circuit breaker — after first failure, skip remaining tickers in this loop cycle.
# Prevents 27 x 6s GPU timeouts when CUDA is broken.
_CIRCUIT_BREAKER_TTL = 300  # 5 minutes before retry
_kronos_tripped_until = 0.0  # monotonic timestamp when breaker resets
_chronos_tripped_until = 0.0


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


def _run_kronos(candles: list[dict], horizons: tuple = (1, 24)) -> dict | None:
    """Run Kronos inference via subprocess."""
    if not _KRONOS_ENABLED:
        return None
    if _kronos_circuit_open():
        return None
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
            timeout=120,
        )
        if proc.returncode != 0:
            logger.warning("Kronos subprocess failed: %s", proc.stderr[:200])
            _trip_kronos()
            return None
        result = json.loads(proc.stdout)
        if not result or not result.get("results"):
            _trip_kronos()
            return None
        return result
    except Exception as e:
        logger.warning("Kronos subprocess error (v2): %s", e)
        _trip_kronos()
        return None


def _run_chronos(prices: list[float], horizons: tuple = (1, 24)) -> dict | None:
    """Run Chronos forecast (in-process, lazy-loaded)."""
    if _chronos_circuit_open():
        return None
    try:
        from portfolio.forecast_signal import forecast_chronos
        result = forecast_chronos("", prices, horizons=horizons)
        if result is None:
            _trip_chronos()
        return result
    except Exception as e:
        logger.warning("Chronos failed: %s", e)
        _trip_chronos()
        return None


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
    kronos = _cached(kronos_key, _FORECAST_TTL, _run_kronos, candles or [], (1, 24))
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
    chronos = _cached(chronos_key, _FORECAST_TTL, _run_chronos, close_prices, (1, 24))
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

    # Majority vote across sub-signals
    votes = list(result["sub_signals"].values())
    result["action"], result["confidence"] = majority_vote(votes)

    # Cap confidence
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    # Log prediction for accuracy tracking
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "current_price": current_price,
            "sub_signals": result["sub_signals"],
            "action": result["action"],
            "confidence": result["confidence"],
        }
        if kronos and kronos.get("results"):
            entry["kronos"] = kronos["results"]
        if chronos:
            entry["chronos"] = chronos
        with open(_PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to log forecast prediction", exc_info=True)

    return result
