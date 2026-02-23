"""Time-series forecast signal — Prophet (CPU) + Chronos (GPU).

Generates 1h-ahead and 24h-ahead price forecasts for each ticker.
Logs predictions to data/forecast_predictions.jsonl for accuracy tracking.
Can be run standalone or called from the main loop.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio.forecast")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"

# Chronos model — loaded lazily on first use
_chronos_pipeline = None
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
        logger.debug(f"Candle fetch failed for {ticker}: {e}")

    return None


def _get_chronos_pipeline():
    """Lazy-load Chronos pipeline with GPU if available."""
    global _chronos_pipeline
    if _chronos_pipeline is not None:
        return _chronos_pipeline

    try:
        import torch
        from chronos import ChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Chronos model on {device}...")
        _chronos_pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=device,
            dtype=torch.float32,
        )
        logger.info("Chronos model loaded")
        return _chronos_pipeline
    except Exception as e:
        logger.warning(f"Failed to load Chronos: {e}")
        return None


def forecast_chronos(ticker, prices, horizons=(1, 24)):
    """Generate probabilistic forecasts using Chronos.

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
        import torch
        context = torch.tensor([prices], dtype=torch.float32)
        max_h = max(horizons)

        # Generate forecast samples
        forecast = pipeline.predict(context, max_h, num_samples=20)
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
    except Exception as e:
        logger.warning(f"Chronos forecast failed for {ticker}: {e}")
        return None


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
        from prophet import Prophet
        import logging as _logging
        # Suppress Prophet's verbose stdout
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        # Build dataframe with hourly timestamps (tz-naive, Prophet requirement)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
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
        logger.warning(f"Prophet forecast failed for {ticker}: {e}")
        return None


def run_forecasts(tickers=None):
    """Run both Prophet and Chronos forecasts for all tickers.

    Logs predictions to forecast_predictions.jsonl for accuracy tracking.
    """
    if tickers is None:
        # Load tickers from agent_summary
        try:
            summary = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
            tickers = list(summary.get("signals", {}).keys())
        except Exception:
            logger.error("Could not load tickers from agent_summary.json")
            return

    ts = datetime.now(timezone.utc).isoformat()
    results = []

    for ticker in tickers:
        prices = _load_candles(ticker)
        if not prices or len(prices) < 50:
            logger.debug(f"Skipping {ticker}: insufficient candle data ({len(prices) if prices else 0})")
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
        with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Logged {len(results)} forecast predictions")

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
