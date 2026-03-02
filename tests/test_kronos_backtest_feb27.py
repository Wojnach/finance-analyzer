"""Kronos backtest on Feb 27 2026 data with 5-min candles.

Fetches historical 5-min OHLCV from Binance, runs Kronos predictions
at multiple time points throughout Feb 27, and measures direction accuracy
against actual price outcomes.

Tests parameter variations (T, top_p, sample_count) to find optimal settings.
"""

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

# Add Kronos source to path
KRONOS_SRC = Path(r"Q:\models\kronos\kronos-src")
if str(KRONOS_SRC) not in sys.path:
    sys.path.insert(0, str(KRONOS_SRC))

logger = logging.getLogger("kronos_backtest")

# --- Data fetching ---

def fetch_binance_klines(symbol: str, interval: str, start_dt: datetime,
                          end_dt: datetime, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance spot API."""
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_candles.extend(data)
        # Move start past last candle
        current_start = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.2)  # rate limit

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df[["timestamp", "open", "high", "low", "close", "volume", "amount"]]


def load_kronos_predictor():
    """Load Kronos model (cached after first call)."""
    import torch
    from model import KronosTokenizer, Kronos, KronosPredictor

    model_path = Path(r"Q:\models\kronos\kronos-base")
    tokenizer_path = Path(r"Q:\models\kronos\kronos-tokenizer")

    if not model_path.exists() or not tokenizer_path.exists():
        pytest.skip("Kronos model files not found")

    # Redirect stdout during model loading
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
        model = Kronos.from_pretrained(str(model_path))
    finally:
        sys.stdout = saved_stdout

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device)
    return predictor


def run_kronos_prediction(predictor, candles_df: pd.DataFrame,
                          pred_len: int = 6, T: float = 1.0,
                          top_p: float = 0.9, sample_count: int = 3):
    """Run a single Kronos prediction.

    Args:
        predictor: KronosPredictor instance
        candles_df: DataFrame with open,high,low,close,volume,amount,timestamp
        pred_len: Number of candles to predict ahead
        T: Sampling temperature
        top_p: Nucleus sampling threshold
        sample_count: Number of parallel samples to average

    Returns:
        DataFrame with predicted OHLCV
    """
    ohlcv = candles_df[["open", "high", "low", "close", "volume", "amount"]].copy()

    x_timestamp = pd.DatetimeIndex(candles_df["timestamp"].values)

    # Infer interval from last two candles
    if len(x_timestamp) >= 2:
        interval = x_timestamp[-1] - x_timestamp[-2]
    else:
        interval = timedelta(minutes=5)

    last_time = x_timestamp[-1]
    y_timestamp = pd.DatetimeIndex([
        last_time + interval * (i + 1) for i in range(pred_len)
    ])

    pred_df = predictor.predict(
        ohlcv,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    return pred_df


def direction_correct(predicted_close: float, actual_close: float,
                      entry_price: float, threshold_pct: float = 0.1) -> bool | None:
    """Check if predicted direction matches actual direction.

    Returns True/False for correct/incorrect, None if move is below threshold.
    """
    pred_pct = (predicted_close - entry_price) / entry_price * 100
    actual_pct = (actual_close - entry_price) / entry_price * 100

    # Skip if actual move is negligible
    if abs(actual_pct) < threshold_pct:
        return None

    # Check if same direction
    return (pred_pct > 0) == (actual_pct > 0)


# --- Tests ---

@pytest.fixture(scope="module")
def feb27_btc_5m():
    """Fetch BTC 5-min candles covering Feb 26-28 (context + test + outcomes)."""
    # Feb 25 00:00 to Feb 28 12:00 — plenty of context + outcomes
    start = datetime(2026, 2, 25, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    df = fetch_binance_klines("BTCUSDT", "5m", start, end)
    if df.empty:
        pytest.skip("Could not fetch BTC 5m candles from Binance")
    print(f"\nFetched {len(df)} BTC 5-min candles ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
    return df


@pytest.fixture(scope="module")
def feb27_eth_5m():
    """Fetch ETH 5-min candles covering Feb 26-28."""
    start = datetime(2026, 2, 25, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    df = fetch_binance_klines("ETHUSDT", "5m", start, end)
    if df.empty:
        pytest.skip("Could not fetch ETH 5m candles from Binance")
    return df


@pytest.fixture(scope="module")
def feb27_xag_5m():
    """Fetch XAG (silver) 5-min candles from Binance FAPI."""
    start = datetime(2026, 2, 25, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    # XAG on Binance FAPI
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_candles = []
    current_start = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while current_start < end_ms:
        params = {
            "symbol": "XAGUSDT",
            "interval": "5m",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            pytest.skip("Could not fetch XAG 5m candles from Binance FAPI")
            return pd.DataFrame()
        if not data:
            break
        all_candles.extend(data)
        current_start = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.2)

    if not all_candles:
        pytest.skip("No XAG 5m candles returned")

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df[["timestamp", "open", "high", "low", "close", "volume", "amount"]]


@pytest.fixture(scope="module")
def kronos_predictor():
    """Load Kronos predictor (cached for the module)."""
    return load_kronos_predictor()


class TestKronosBacktestFeb27:
    """Backtest Kronos on Feb 27 2026 with 5-min candles."""

    def _run_sliding_backtest(self, df, predictor, ticker,
                              lookback=400, pred_len=6, step=12,
                              T=1.0, top_p=0.9, sample_count=3,
                              test_start_str="2026-02-27",
                              test_end_str="2026-02-28"):
        """Run sliding window backtest.

        Args:
            df: Full OHLCV DataFrame with timestamp
            predictor: KronosPredictor
            ticker: Ticker name for logging
            lookback: Number of candles as history input
            pred_len: Number of candles to predict ahead
            step: Candles between prediction windows (12 = every 1h for 5m candles)
            T, top_p, sample_count: Kronos parameters
            test_start_str, test_end_str: Date range for predictions

        Returns:
            list of result dicts with prediction outcomes
        """
        results = []
        test_start = pd.Timestamp(test_start_str, tz="UTC")
        test_end = pd.Timestamp(test_end_str, tz="UTC")

        # Find valid prediction windows
        for i in range(lookback, len(df) - pred_len, step):
            pred_time = df.iloc[i]["timestamp"]
            if pred_time < test_start or pred_time >= test_end:
                continue

            context = df.iloc[i - lookback:i].reset_index(drop=True)
            actual_future = df.iloc[i:i + pred_len].reset_index(drop=True)

            entry_price = context.iloc[-1]["close"]
            actual_close_1h = actual_future.iloc[-1]["close"]

            try:
                t0 = time.time()
                pred_df = run_kronos_prediction(
                    predictor, context,
                    pred_len=pred_len, T=T, top_p=top_p,
                    sample_count=sample_count,
                )
                elapsed = time.time() - t0

                predicted_close = float(pred_df.iloc[-1]["close"])
                pred_pct = (predicted_close - entry_price) / entry_price * 100
                actual_pct = (actual_close_1h - entry_price) / entry_price * 100

                correct = direction_correct(predicted_close, actual_close_1h, entry_price)

                results.append({
                    "time": str(pred_time),
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "predicted_close": predicted_close,
                    "actual_close": actual_close_1h,
                    "pred_pct": round(pred_pct, 4),
                    "actual_pct": round(actual_pct, 4),
                    "direction_correct": correct,
                    "elapsed_s": round(elapsed, 2),
                    "params": {"T": T, "top_p": top_p, "sample_count": sample_count},
                })
            except Exception as e:
                results.append({
                    "time": str(pred_time),
                    "ticker": ticker,
                    "error": str(e)[:200],
                })

        return results

    @pytest.mark.slow
    def test_btc_default_params(self, feb27_btc_5m, kronos_predictor, tmp_path):
        """BTC backtest with default parameters (T=1.0, top_p=0.9, samples=3)."""
        results = self._run_sliding_backtest(
            feb27_btc_5m, kronos_predictor, "BTC-USD",
            lookback=400, pred_len=6, step=12,  # every 1h
            T=1.0, top_p=0.9, sample_count=3,
        )

        valid = [r for r in results if "direction_correct" in r and r["direction_correct"] is not None]
        correct = sum(1 for r in valid if r["direction_correct"])
        total = len(valid)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\nBTC default params: {correct}/{total} = {accuracy:.1f}% direction accuracy")
        print(f"  Total predictions: {len(results)}, valid: {total}, errors: {len(results) - len(valid)}")
        if valid:
            avg_elapsed = np.mean([r["elapsed_s"] for r in valid])
            print(f"  Avg prediction time: {avg_elapsed:.1f}s")

        # Save results
        out_path = tmp_path / "kronos_backtest_btc_default.json"
        out_path.write_text(json.dumps({
            "ticker": "BTC-USD",
            "params": {"T": 1.0, "top_p": 0.9, "sample_count": 3},
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, indent=2, default=str))
        print(f"  Results saved to {out_path}")

        assert total > 0, "No valid predictions were made"

    @pytest.mark.slow
    def test_btc_low_temperature(self, feb27_btc_5m, kronos_predictor, tmp_path):
        """BTC backtest with lower temperature (sharper predictions)."""
        results = self._run_sliding_backtest(
            feb27_btc_5m, kronos_predictor, "BTC-USD",
            lookback=400, pred_len=6, step=12,
            T=0.5, top_p=0.9, sample_count=3,
        )

        valid = [r for r in results if "direction_correct" in r and r["direction_correct"] is not None]
        correct = sum(1 for r in valid if r["direction_correct"])
        total = len(valid)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\nBTC T=0.5: {correct}/{total} = {accuracy:.1f}% direction accuracy")

        out_path = tmp_path / "kronos_backtest_btc_lowT.json"
        out_path.write_text(json.dumps({
            "ticker": "BTC-USD",
            "params": {"T": 0.5, "top_p": 0.9, "sample_count": 3},
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, indent=2, default=str))

        assert total > 0

    @pytest.mark.slow
    def test_btc_high_samples(self, feb27_btc_5m, kronos_predictor, tmp_path):
        """BTC backtest with more samples (better averaging)."""
        results = self._run_sliding_backtest(
            feb27_btc_5m, kronos_predictor, "BTC-USD",
            lookback=400, pred_len=6, step=12,
            T=0.7, top_p=0.85, sample_count=5,
        )

        valid = [r for r in results if "direction_correct" in r and r["direction_correct"] is not None]
        correct = sum(1 for r in valid if r["direction_correct"])
        total = len(valid)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\nBTC T=0.7/p=0.85/s=5: {correct}/{total} = {accuracy:.1f}% direction accuracy")

        out_path = tmp_path / "kronos_backtest_btc_highsamples.json"
        out_path.write_text(json.dumps({
            "ticker": "BTC-USD",
            "params": {"T": 0.7, "top_p": 0.85, "sample_count": 5},
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, indent=2, default=str))

        assert total > 0

    @pytest.mark.slow
    def test_eth_default_params(self, feb27_eth_5m, kronos_predictor, tmp_path):
        """ETH backtest with default parameters."""
        results = self._run_sliding_backtest(
            feb27_eth_5m, kronos_predictor, "ETH-USD",
            lookback=400, pred_len=6, step=12,
            T=1.0, top_p=0.9, sample_count=3,
        )

        valid = [r for r in results if "direction_correct" in r and r["direction_correct"] is not None]
        correct = sum(1 for r in valid if r["direction_correct"])
        total = len(valid)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\nETH default params: {correct}/{total} = {accuracy:.1f}% direction accuracy")

        out_path = tmp_path / "kronos_backtest_eth_default.json"
        out_path.write_text(json.dumps({
            "ticker": "ETH-USD",
            "params": {"T": 1.0, "top_p": 0.9, "sample_count": 3},
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, indent=2, default=str))

        assert total > 0

    @pytest.mark.slow
    def test_xag_default_params(self, feb27_xag_5m, kronos_predictor, tmp_path):
        """XAG (silver) backtest with default parameters."""
        results = self._run_sliding_backtest(
            feb27_xag_5m, kronos_predictor, "XAG-USD",
            lookback=400, pred_len=6, step=12,
            T=1.0, top_p=0.9, sample_count=3,
        )

        valid = [r for r in results if "direction_correct" in r and r["direction_correct"] is not None]
        correct = sum(1 for r in valid if r["direction_correct"])
        total = len(valid)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\nXAG default params: {correct}/{total} = {accuracy:.1f}% direction accuracy")

        out_path = tmp_path / "kronos_backtest_xag_default.json"
        out_path.write_text(json.dumps({
            "ticker": "XAG-USD",
            "params": {"T": 1.0, "top_p": 0.9, "sample_count": 3},
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, indent=2, default=str))

        assert total > 0


class TestKronosParamSweep:
    """Parameter sweep to find optimal Kronos settings."""

    @pytest.mark.slow
    def test_param_sweep_btc(self, feb27_btc_5m, kronos_predictor, tmp_path):
        """Sweep T and sample_count to find best BTC accuracy."""
        param_combos = [
            {"T": 0.3, "top_p": 0.9, "sample_count": 3},
            {"T": 0.5, "top_p": 0.9, "sample_count": 3},
            {"T": 0.7, "top_p": 0.9, "sample_count": 3},
            {"T": 1.0, "top_p": 0.9, "sample_count": 3},
            {"T": 0.5, "top_p": 0.8, "sample_count": 5},
            {"T": 0.7, "top_p": 0.85, "sample_count": 5},
            {"T": 0.5, "top_p": 0.9, "sample_count": 7},
        ]

        sweep_results = []
        backtest_cls = TestKronosBacktestFeb27()

        for params in param_combos:
            results = backtest_cls._run_sliding_backtest(
                feb27_btc_5m, kronos_predictor, "BTC-USD",
                lookback=400, pred_len=6, step=24,  # every 2h (faster sweep)
                **params,
            )

            valid = [r for r in results if "direction_correct" in r and r["direction_correct"] is not None]
            correct = sum(1 for r in valid if r["direction_correct"])
            total = len(valid)
            accuracy = correct / total * 100 if total > 0 else 0

            sweep_results.append({
                "params": params,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            })
            print(f"  T={params['T']}, p={params['top_p']}, s={params['sample_count']}: "
                  f"{correct}/{total} = {accuracy:.1f}%")

        # Sort by accuracy
        sweep_results.sort(key=lambda x: x["accuracy"], reverse=True)

        print(f"\n=== BEST PARAMS ===")
        best = sweep_results[0]
        print(f"  T={best['params']['T']}, top_p={best['params']['top_p']}, "
              f"sample_count={best['params']['sample_count']}: {best['accuracy']:.1f}%")

        out_path = tmp_path / "kronos_param_sweep.json"
        out_path.write_text(json.dumps(sweep_results, indent=2))
        print(f"  Sweep results saved to {out_path}")

        assert len(sweep_results) > 0
