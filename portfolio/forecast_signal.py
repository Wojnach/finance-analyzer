"""Time-series forecast signal — Prophet (CPU) + Chronos (GPU).

Generates 1h-ahead and 24h-ahead price forecasts for each ticker.
Logs predictions to data/forecast_predictions.jsonl for accuracy tracking.
Can be run standalone or called from the main loop.
"""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.file_utils import atomic_append_jsonl, load_json

logger = logging.getLogger("portfolio.forecast")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"

# Chronos model — loaded lazily on first use
# Configurable via config.json → forecast.chronos_model
# Options: "amazon/chronos-t5-tiny", "amazon/chronos-t5-small",
#          "amazon/chronos-t5-base", "amazon/chronos-t5-large",
#          "amazon/chronos-2" (preferred, requires chronos-forecasting>=2.0)
_CHRONOS_MODEL = "amazon/chronos-2"
_chronos_pipeline = None
_chronos_version = 0  # 0=not loaded, 1=v1 (T5), 2=v2 (Chronos-2)
_prophet_cache = {}  # ticker -> last fit time, to avoid refitting every minute


def _load_candles(ticker, periods=168):
    """Load recent 1h candle closes using the appropriate data source."""
    from portfolio.tickers import SYMBOLS

    source_info = SYMBOLS.get(ticker, {})
    try:
        if "binance" in source_info:
            from portfolio.data_collector import binance_klines
            symbol = source_info["binance"]
            df = binance_klines(symbol, interval="1h", limit=periods)
            if df is not None and len(df) > 30:
                return df["close"].values.tolist()
        elif "binance_fapi" in source_info:
            from portfolio.data_collector import binance_fapi_klines
            symbol = source_info["binance_fapi"]
            df = binance_fapi_klines(symbol, interval="1h", limit=periods)
            if df is not None and len(df) > 30:
                return df["close"].values.tolist()
        elif "alpaca" in source_info:
            from portfolio.data_collector import alpaca_klines
            symbol = source_info["alpaca"]
            df = alpaca_klines(symbol, interval="1h", limit=periods)
            if df is not None and len(df) > 30:
                return df["close"].values.tolist()
    except Exception as e:
        logger.debug("Candle fetch failed for %s: %s", ticker, e)

    return None


def set_chronos_model(model_name: str):
    """Override the Chronos model (e.g. from config). Resets cached pipeline."""
    global _CHRONOS_MODEL, _chronos_pipeline, _chronos_version
    if model_name and model_name != _CHRONOS_MODEL:
        _CHRONOS_MODEL = model_name
        _chronos_pipeline = None  # force reload on next call
        _chronos_version = 0
        logger.info("Chronos model set to %s", model_name)


def _get_chronos_pipeline():
    """Lazy-load Chronos pipeline. Tries Chronos-2 first, falls back to v1."""
    global _chronos_pipeline, _chronos_version
    if _chronos_pipeline is not None:
        return _chronos_pipeline

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try Chronos-2 first (newer, multivariate, better accuracy)
        try:
            from chronos import Chronos2Pipeline
            logger.info("Loading Chronos-2 model %s on %s...", _CHRONOS_MODEL, device)
            _chronos_pipeline = Chronos2Pipeline.from_pretrained(
                _CHRONOS_MODEL,
                device_map=device,
            )
            _chronos_version = 2
            logger.info("Chronos-2 model loaded: %s", _CHRONOS_MODEL)
            return _chronos_pipeline
        except (ImportError, Exception) as e:
            logger.info("Chronos-2 not available (%s), falling back to v1", e)

        # Fallback to v1 (ChronosPipeline)
        from chronos import ChronosPipeline
        logger.info("Loading Chronos v1 model %s on %s...", _CHRONOS_MODEL, device)
        _chronos_pipeline = ChronosPipeline.from_pretrained(
            _CHRONOS_MODEL,
            device_map=device,
            dtype=torch.float32,
        )
        _chronos_version = 1
        logger.info("Chronos v1 model loaded: %s", _CHRONOS_MODEL)
        return _chronos_pipeline
    except Exception as e:
        logger.warning("Failed to load Chronos (%s): %s", _CHRONOS_MODEL, e)
        return None


def forecast_chronos(ticker, prices, horizons=(1, 24)):
    """Generate probabilistic forecasts using Chronos (v1 or v2).

    Args:
        ticker: Instrument ticker
        prices: List of recent hourly close prices
        horizons: Tuple of forecast horizons in hours

    Returns:
        Dict with forecast results per horizon, or None on failure
    """
    pipeline = _get_chronos_pipeline()
    if pipeline is None:
        return None

    try:
        if _chronos_version == 2:
            return _forecast_chronos_v2(pipeline, ticker, prices, horizons)
        else:
            return _forecast_chronos_v1(pipeline, ticker, prices, horizons)
    except Exception as e:
        logger.warning("Chronos forecast failed for %s: %s", ticker, e)
        return None


def _forecast_chronos_v1(pipeline, ticker, prices, horizons=(1, 24)):
    """Chronos v1 (T5) sample-based forecasting."""
    import torch
    context = torch.tensor([prices], dtype=torch.float32)
    max_h = max(horizons)

    # Generate forecast samples
    forecast = pipeline.predict(context, max_h, num_samples=100)
    # forecast shape: (1, num_samples, max_h)
    samples = forecast[0].numpy()  # (num_samples, max_h)

    results = {}
    current_price = prices[-1]
    for h in horizons:
        h_samples = samples[:, h - 1]
        median = float(np.median(h_samples))
        low = float(np.percentile(h_samples, 10))
        high = float(np.percentile(h_samples, 90))

        # Signal: if current price is below lower band -> BUY
        #         if current price is above upper band -> SELL
        if current_price < low:
            action = "BUY"
            confidence = min((low - current_price) / current_price * 10, 1.0)
        elif current_price > high:
            action = "SELL"
            confidence = min((current_price - high) / current_price * 10, 1.0)
        else:
            # Direction from median
            pct_move = (median - current_price) / current_price
            if abs(pct_move) < 0.002:  # <0.2% = noise
                action = "HOLD"
                confidence = 0.0
            elif pct_move > 0:
                action = "BUY"
                confidence = min(abs(pct_move) * 20, 1.0)
            else:
                action = "SELL"
                confidence = min(abs(pct_move) * 20, 1.0)

        results[f"{h}h"] = {
            "median": round(median, 4),
            "low_10": round(low, 4),
            "high_90": round(high, 4),
            "pct_move": round((median - current_price) / current_price * 100, 3),
            "action": action,
            "confidence": round(confidence, 3),
        }

    return results


def _forecast_chronos_v2(pipeline, ticker, prices, horizons=(1, 24)):
    """Chronos-2 DataFrame-based forecasting with quantile output."""
    n = len(prices)
    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")
    context_df = pd.DataFrame({
        "timestamp": timestamps,
        "target": prices,
        "id": ticker or "default",
    })

    max_h = max(horizons)
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=max_h,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    results = {}
    current_price = prices[-1]

    for h in horizons:
        # pred_df has columns: id, timestamp, 0.1, 0.5, 0.9
        row = pred_df.iloc[h - 1]
        median = float(row["0.5"])
        low = float(row["0.1"])
        high = float(row["0.9"])

        if current_price < low:
            action = "BUY"
            confidence = min((low - current_price) / current_price * 10, 1.0)
        elif current_price > high:
            action = "SELL"
            confidence = min((current_price - high) / current_price * 10, 1.0)
        else:
            pct_move = (median - current_price) / current_price
            if abs(pct_move) < 0.002:
                action = "HOLD"
                confidence = 0.0
            elif pct_move > 0:
                action = "BUY"
                confidence = min(abs(pct_move) * 20, 1.0)
            else:
                action = "SELL"
                confidence = min(abs(pct_move) * 20, 1.0)

        results[f"{h}h"] = {
            "median": round(median, 4),
            "low_10": round(low, 4),
            "high_90": round(high, 4),
            "pct_move": round((median - current_price) / current_price * 100, 3),
            "action": action,
            "confidence": round(confidence, 3),
        }

    return results


def forecast_prophet(ticker, prices, horizons=(1, 24)):
    """Generate forecasts using Meta Prophet.

    Args:
        ticker: Instrument ticker
        prices: List of recent hourly close prices
        horizons: Tuple of forecast horizons in hours

    Returns:
        Dict with forecast results per horizon, or None on failure
    """
    try:
        import logging as _logging

        from prophet import Prophet
        # Suppress Prophet's verbose stdout
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        # Build dataframe with hourly timestamps (tz-naive, Prophet requirement)
        now = datetime.now(UTC).replace(tzinfo=None)
        n = len(prices)
        ds = pd.date_range(end=now, periods=n, freq="h")
        df = pd.DataFrame({"ds": ds, "y": prices})

        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1,
            interval_width=0.80,
        )
        m.fit(df)

        max_h = max(horizons)
        future = m.make_future_dataframe(periods=max_h, freq="h")
        forecast = m.predict(future)

        results = {}
        current_price = prices[-1]
        for h in horizons:
            row = forecast.iloc[-(max_h - h + 1)]
            median = float(row["yhat"])
            low = float(row["yhat_lower"])
            high = float(row["yhat_upper"])

            if current_price < low:
                action = "BUY"
                confidence = min((low - current_price) / current_price * 10, 1.0)
            elif current_price > high:
                action = "SELL"
                confidence = min((current_price - high) / current_price * 10, 1.0)
            else:
                pct_move = (median - current_price) / current_price
                if abs(pct_move) < 0.002:
                    action = "HOLD"
                    confidence = 0.0
                elif pct_move > 0:
                    action = "BUY"
                    confidence = min(abs(pct_move) * 20, 1.0)
                else:
                    action = "SELL"
                    confidence = min(abs(pct_move) * 20, 1.0)

            results[f"{h}h"] = {
                "median": round(median, 4),
                "low_80": round(low, 4),
                "high_80": round(high, 4),
                "pct_move": round((median - current_price) / current_price * 100, 3),
                "action": action,
                "confidence": round(confidence, 3),
            }

        return results
    except Exception as e:
        logger.warning("Prophet forecast failed for %s: %s", ticker, e)
        return None


def run_forecasts(tickers=None):
    """Run both Prophet and Chronos forecasts for all tickers.

    Logs predictions to forecast_predictions.jsonl for accuracy tracking.
    """
    if tickers is None:
        # Load tickers from agent_summary
        summary = load_json(AGENT_SUMMARY_FILE)
        if summary is None:
            logger.error("Could not load tickers from agent_summary.json")
            return
        tickers = list(summary.get("signals", {}).keys())

    ts = datetime.now(UTC).isoformat()
    results = []

    for ticker in tickers:
        prices = _load_candles(ticker)
        if not prices or len(prices) < 50:
            logger.debug("Skipping %s: insufficient candle data (%d)", ticker, len(prices) if prices else 0)
            continue

        current_price = prices[-1]
        entry = {
            "ts": ts,
            "ticker": ticker,
            "current_price": current_price,
        }

        # Chronos (GPU)
        t0 = time.time()
        chronos_result = forecast_chronos(ticker, prices)
        if chronos_result:
            entry["chronos"] = chronos_result
            entry["chronos_time_ms"] = round((time.time() - t0) * 1000)

        # Prophet (CPU)
        t0 = time.time()
        prophet_result = forecast_prophet(ticker, prices)
        if prophet_result:
            entry["prophet"] = prophet_result
            entry["prophet_time_ms"] = round((time.time() - t0) * 1000)

        results.append(entry)
        logger.info(
            f"{ticker}: Chronos {chronos_result.get('1h', {}).get('action', '?') if chronos_result else 'FAIL'} "
            f"/ Prophet {prophet_result.get('1h', {}).get('action', '?') if prophet_result else 'FAIL'}"
        )

    # Append all predictions
    if results:
        for entry in results:
            atomic_append_jsonl(PREDICTIONS_FILE, entry)
        logger.info("Logged %d forecast predictions", len(results))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    results = run_forecasts()
    if results:
        print(f"\n{'Ticker':<10} {'Price':>10} {'Chronos 1h':>12} {'Prophet 1h':>12} {'Chronos 24h':>13} {'Prophet 24h':>13}")
        print("-" * 72)
        for r in results:
            ticker = r["ticker"]
            price = r["current_price"]
            c1 = r.get("chronos", {}).get("1h", {})
            p1 = r.get("prophet", {}).get("1h", {})
            c24 = r.get("chronos", {}).get("24h", {})
            p24 = r.get("prophet", {}).get("24h", {})
            print(
                f"{ticker:<10} {price:>10.2f} "
                f"{c1.get('action', '?'):>4} {c1.get('pct_move', 0):>+.2f}% "
                f"{p1.get('action', '?'):>4} {p1.get('pct_move', 0):>+.2f}% "
                f"{c24.get('action', '?'):>4} {c24.get('pct_move', 0):>+.2f}% "
                f"{p24.get('action', '?'):>4} {p24.get('pct_move', 0):>+.2f}%"
            )
