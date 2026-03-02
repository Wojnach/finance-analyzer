"""
Local LLM inference for metals trading loop.

Runs Ministral-8B (metals-adapted prompt) and Chronos (time-series forecast)
every 5 minutes in a background thread. Tracks prediction accuracy at 1h and 3h
horizons. Weights model signals by their proven accuracy.

Usage from metals_loop.py:
    from metals_llm import start_llm_thread, get_llm_signals, get_llm_accuracy

    start_llm_thread()  # launch background inference thread
    signals = get_llm_signals()  # get latest cached results
    accuracy = get_llm_accuracy()  # get rolling accuracy per model
"""

import json
import os
import sys
import time
import datetime
import threading
import subprocess
import traceback
import platform

os.chdir(r"Q:/finance-analyzer")

import requests

# --- CONFIG ---
LLM_INTERVAL = 300       # run models every 5 minutes
ACCURACY_WINDOW = 50     # rolling window for accuracy calculation
PREDICTION_LOG = "data/metals_llm_predictions.jsonl"

# Binance FAPI for metals klines
FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"
METALS_SYMBOLS = {
    "XAG-USD": "XAGUSDT",
    "XAU-USD": "XAUUSDT",
}

# Ministral subprocess paths
if platform.system() == "Windows":
    MINISTRAL_PYTHON = r"Q:\models\.venv-llm\Scripts\python.exe"
    MINISTRAL_SCRIPT = r"Q:\models\ministral_trader.py"
    MAIN_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
else:
    MINISTRAL_PYTHON = "/home/deck/models/.venv-llm/bin/python"
    MINISTRAL_SCRIPT = "/home/deck/models/ministral_trader.py"
    MAIN_PYTHON = "/home/deck/.venv/bin/python"

# --- SHARED STATE (thread-safe) ---
_lock = threading.Lock()
_llm_signals = {}        # latest model predictions
_llm_accuracy = {}       # rolling accuracy per model
_llm_last_run = 0        # timestamp of last inference run
_llm_thread = None
_llm_stop = threading.Event()

# Accuracy horizons in seconds
HORIZONS = {
    "1h": 3600,
    "3h": 10800,
}


def _log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [LLM] {msg}", flush=True)


def _fetch_fapi_klines(symbol, interval="1h", limit=200):
    """Fetch klines from Binance FAPI (futures) for metals."""
    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        r = requests.get(FAPI_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        # Extract close prices + OHLCV
        candles = []
        for k in data:
            candles.append({
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        return candles
    except Exception as e:
        _log(f"FAPI kline error {symbol}: {e}")
        return None


def _run_ministral_metals(context):
    """Run Ministral-8B with metals-adapted prompt via subprocess.

    Returns {"action": "BUY/SELL/HOLD", "reasoning": "...", "confidence": 0.0-1.0}
    """
    try:
        # Build metals-specific context for Ministral
        # We modify the context to make the prompt metals-aware
        metals_context = {
            "ticker": context.get("ticker", "XAG-USD"),
            "price_usd": context.get("price_usd", 0),
            "change_24h": context.get("change_24h", "N/A"),
            "rsi": context.get("rsi", "N/A"),
            "macd_hist": context.get("macd_hist", "N/A"),
            "ema_bullish": context.get("ema_bullish", True),
            "ema_gap_pct": context.get("ema_gap_pct", "N/A"),
            "bb_position": context.get("bb_position", "N/A"),
            "fear_greed": context.get("fear_greed", "N/A"),
            "fear_greed_class": context.get("fear_greed_class", ""),
            "news_sentiment": context.get("news_sentiment", "neutral"),
            "sentiment_confidence": context.get("sentiment_confidence", 0),
            "timeframe_summary": context.get("timeframe_summary", "N/A"),
            "headlines": context.get("headlines", "N/A"),
        }

        proc = subprocess.run(
            [MINISTRAL_PYTHON, MINISTRAL_SCRIPT],
            input=json.dumps(metals_context),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            _log(f"Ministral failed: {proc.stderr[:200]}")
            return None

        result = json.loads(proc.stdout.strip())
        return {
            "action": result.get("action", "HOLD"),
            "reasoning": result.get("reasoning", "")[:200],
            "confidence": 0.6,  # base confidence, adjusted by accuracy
        }
    except Exception as e:
        _log(f"Ministral error: {e}")
        return None


def _run_chronos_metals(ticker, close_prices, horizons=(1, 3)):
    """Run Chronos forecast for metals using close prices.

    Returns dict of {horizon_key: {"direction": "up/down", "pct_move": float, "confidence": float}}
    """
    try:
        # Write a small script that runs Chronos and returns results
        script = r"""
import json, sys
sys.path.insert(0, r"Q:/finance-analyzer")
try:
    from portfolio.forecast_signal import forecast_chronos
    prices = json.loads(sys.stdin.read())
    result = forecast_chronos("", prices["close_prices"], horizons=tuple(prices["horizons"]))
    if result:
        print(json.dumps(result))
    else:
        print(json.dumps({}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"""
        input_data = json.dumps({
            "close_prices": close_prices,
            "horizons": list(horizons),
        })

        proc = subprocess.run(
            [MAIN_PYTHON, "-c", script],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if proc.returncode != 0:
            _log(f"Chronos failed for {ticker}: {proc.stderr[:200]}")
            return None

        stdout = proc.stdout.strip()
        if not stdout:
            return None

        # Handle contaminated stdout (HuggingFace prints during load)
        brace_idx = stdout.find("{")
        if brace_idx > 0:
            stdout = stdout[brace_idx:]

        result = json.loads(stdout)
        if "error" in result:
            _log(f"Chronos error for {ticker}: {result['error']}")
            return None

        return result

    except Exception as e:
        _log(f"Chronos subprocess error for {ticker}: {e}")
        return None


def _log_prediction(model, ticker, direction, confidence, current_price, horizon_key="1h"):
    """Log a prediction for accuracy tracking."""
    try:
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": model,
            "ticker": ticker,
            "direction": direction,  # "up" or "down"
            "confidence": round(confidence, 3),
            "price_at_prediction": round(current_price, 4),
            "horizon": horizon_key,
            "outcome": None,  # filled in later by accuracy checker
            "outcome_price": None,
            "correct": None,
        }
        with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def check_prediction_accuracy(current_prices):
    """Check old predictions against current prices.

    Looks at predictions from ~1h and ~3h ago, compares predicted direction
    with actual direction, updates the log entries.

    Returns dict: {model: {horizon: {correct: int, total: int, accuracy: float}}}
    """
    now = time.time()
    accuracy = {}

    try:
        if not os.path.exists(PREDICTION_LOG):
            return accuracy

        # Read all predictions
        entries = []
        with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except:
                    pass

        if not entries:
            return accuracy

        updated = False
        for entry in entries:
            if entry.get("outcome") is not None:
                continue  # already resolved

            # Parse timestamp
            try:
                pred_ts = datetime.datetime.fromisoformat(entry["ts"])
                pred_epoch = pred_ts.timestamp()
            except:
                continue

            horizon_key = entry.get("horizon", "1h")
            horizon_secs = HORIZONS.get(horizon_key, 3600)

            # Check if enough time has passed
            elapsed = now - pred_epoch
            if elapsed < horizon_secs * 0.9:  # allow 10% tolerance
                continue

            # Get current price for this ticker
            ticker = entry.get("ticker", "")
            if ticker not in current_prices:
                continue

            actual_price = current_prices[ticker]
            pred_price = entry.get("price_at_prediction", 0)
            if pred_price <= 0 or actual_price <= 0:
                continue

            # Determine actual direction
            actual_direction = "up" if actual_price > pred_price else "down"
            pred_direction = entry.get("direction", "flat")

            if pred_direction == "flat":
                entry["outcome"] = "skipped"
                updated = True
                continue

            correct = (pred_direction == actual_direction)
            entry["outcome"] = actual_direction
            entry["outcome_price"] = round(actual_price, 4)
            entry["correct"] = correct
            updated = True

        # Write back updated entries
        if updated:
            with open(PREDICTION_LOG, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Compute accuracy stats (last ACCURACY_WINDOW resolved predictions per model)
        for entry in entries:
            if entry.get("correct") is None:
                continue

            model = entry.get("model", "?")
            horizon = entry.get("horizon", "1h")
            key = f"{model}_{horizon}"

            if key not in accuracy:
                accuracy[key] = {"correct": 0, "total": 0, "accuracy": 0.0}

            accuracy[key]["total"] += 1
            if entry["correct"]:
                accuracy[key]["correct"] += 1

        # Compute accuracy percentages
        for key in accuracy:
            total = accuracy[key]["total"]
            if total > 0:
                accuracy[key]["accuracy"] = round(accuracy[key]["correct"] / total, 3)

        return accuracy

    except Exception as e:
        _log(f"Accuracy check error: {e}")
        return accuracy


def _compute_model_weights(accuracy_stats):
    """Compute model weights based on accuracy.

    Models with higher accuracy get higher weight.
    Default weight = 0.5 (neutral) until enough samples.
    """
    weights = {}
    for key, stats in accuracy_stats.items():
        total = stats.get("total", 0)
        acc = stats.get("accuracy", 0.5)

        if total < 10:
            # Not enough data — use neutral weight
            weights[key] = 0.5
        elif acc < 0.45:
            # Worse than coin flip — low weight (but don't invert yet)
            weights[key] = 0.1
        elif acc < 0.55:
            # Coin flip territory — low weight
            weights[key] = 0.3
        else:
            # Above coin flip — scale by accuracy
            weights[key] = acc

    return weights


def _build_signal_context(signal_data, ticker):
    """Build Ministral context from main loop's signal data."""
    ctx = {
        "ticker": ticker,
        "price_usd": 0,
        "rsi": "N/A",
        "macd_hist": "N/A",
        "ema_bullish": True,
        "ema_gap_pct": "N/A",
        "bb_position": "N/A",
        "fear_greed": "N/A",
        "fear_greed_class": "",
        "news_sentiment": "neutral",
        "sentiment_confidence": 0,
        "timeframe_summary": "N/A",
        "headlines": "N/A",
    }

    if ticker in signal_data:
        s = signal_data[ticker]
        ctx["rsi"] = s.get("rsi", "N/A")
        ctx["macd_hist"] = s.get("macd_hist", "N/A")
        ctx["bb_position"] = s.get("bb_position", "N/A")

        # Build timeframe summary if available
        tfs = s.get("timeframes", {})
        if tfs:
            tf_parts = []
            for tf_name, tf_val in tfs.items():
                tf_parts.append(f"{tf_name}={tf_val}")
            ctx["timeframe_summary"] = ", ".join(tf_parts)

    return ctx


def _run_inference_cycle(signal_data, underlying_prices):
    """Run one complete inference cycle for all metals tickers.

    Args:
        signal_data: dict from read_signal_data() in metals_loop
        underlying_prices: {"XAG-USD": float, "XAU-USD": float}
    """
    global _llm_signals, _llm_accuracy, _llm_last_run

    results = {}

    for ticker, binance_sym in METALS_SYMBOLS.items():
        current_price = underlying_prices.get(ticker, 0)
        if current_price <= 0:
            continue

        ticker_result = {
            "ticker": ticker,
            "price": current_price,
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ministral": None,
            "chronos_1h": None,
            "chronos_3h": None,
            "consensus": {"direction": "flat", "confidence": 0, "weighted_action": "HOLD"},
        }

        # --- Ministral inference ---
        try:
            ctx = _build_signal_context(signal_data, ticker)
            ctx["price_usd"] = current_price
            # Add 24h change from klines if we can
            ministral_result = _run_ministral_metals(ctx)
            if ministral_result:
                action = ministral_result["action"]
                direction = "up" if action == "BUY" else ("down" if action == "SELL" else "flat")
                ticker_result["ministral"] = {
                    "action": action,
                    "direction": direction,
                    "reasoning": ministral_result.get("reasoning", ""),
                    "confidence": ministral_result.get("confidence", 0.5),
                }
                if direction != "flat":
                    _log_prediction("ministral", ticker, direction,
                                    ministral_result.get("confidence", 0.5),
                                    current_price, "1h")
                    _log_prediction("ministral", ticker, direction,
                                    ministral_result.get("confidence", 0.5),
                                    current_price, "3h")
                _log(f"Ministral {ticker}: {action} — {ministral_result.get('reasoning', '')[:60]}")
        except Exception as e:
            _log(f"Ministral {ticker} failed: {e}")

        # --- Chronos inference ---
        try:
            candles = _fetch_fapi_klines(binance_sym, interval="1h", limit=200)
            if candles and len(candles) >= 50:
                close_prices = [c["close"] for c in candles]

                chronos_result = _run_chronos_metals(ticker, close_prices, horizons=(1, 3))
                if chronos_result:
                    for h_key, h_data in chronos_result.items():
                        if isinstance(h_data, dict):
                            direction = h_data.get("direction", "neutral")
                            pct_move = h_data.get("pct_move", 0)
                            conf = h_data.get("confidence", 0)

                            horizon_label = f"{h_key}h" if not h_key.endswith("h") else h_key
                            out_key = f"chronos_{horizon_label}"
                            ticker_result[out_key] = {
                                "direction": direction,
                                "pct_move": round(pct_move, 4) if pct_move else 0,
                                "confidence": round(conf, 3) if conf else 0,
                            }

                            # Log prediction for accuracy tracking
                            if direction in ("up", "down"):
                                log_horizon = "1h" if h_key in ("1", "1h") else "3h"
                                _log_prediction(f"chronos", ticker, direction,
                                                conf, current_price, log_horizon)

                    _log(f"Chronos {ticker}: {json.dumps({k: v.get('direction', '?') for k, v in chronos_result.items() if isinstance(v, dict)})}")
        except Exception as e:
            _log(f"Chronos {ticker} failed: {e}")

        # --- Compute weighted consensus ---
        try:
            votes = []  # (direction, weight)

            # Get accuracy-based weights
            weights = _compute_model_weights(_llm_accuracy)

            # Ministral vote
            if ticker_result.get("ministral") and ticker_result["ministral"]["direction"] != "flat":
                w = weights.get("ministral_1h", 0.5)
                votes.append((ticker_result["ministral"]["direction"], w))

            # Chronos votes
            for h in ["1h", "3h"]:
                key = f"chronos_{h}"
                if ticker_result.get(key) and ticker_result[key].get("direction") in ("up", "down"):
                    w = weights.get(f"chronos_{h}", 0.5)
                    votes.append((ticker_result[key]["direction"], w))

            if votes:
                up_weight = sum(w for d, w in votes if d == "up")
                down_weight = sum(w for d, w in votes if d == "down")
                total_weight = up_weight + down_weight

                if total_weight > 0:
                    if up_weight > down_weight:
                        direction = "up"
                        confidence = up_weight / total_weight
                    elif down_weight > up_weight:
                        direction = "down"
                        confidence = down_weight / total_weight
                    else:
                        direction = "flat"
                        confidence = 0

                    action = "BUY" if direction == "up" else ("SELL" if direction == "down" else "HOLD")
                    ticker_result["consensus"] = {
                        "direction": direction,
                        "confidence": round(confidence, 3),
                        "weighted_action": action,
                        "up_weight": round(up_weight, 3),
                        "down_weight": round(down_weight, 3),
                        "model_weights": weights,
                        "n_votes": len(votes),
                    }
                    _log(f"Consensus {ticker}: {action} ({confidence:.0%}) "
                         f"[up={up_weight:.2f} down={down_weight:.2f}]")

        except Exception as e:
            _log(f"Consensus error {ticker}: {e}")

        results[ticker] = ticker_result

    # Check accuracy of old predictions
    accuracy = check_prediction_accuracy(underlying_prices)

    # Update shared state
    with _lock:
        _llm_signals = results
        _llm_accuracy = accuracy
        _llm_last_run = time.time()

    # Log accuracy summary
    if accuracy:
        parts = []
        for key, stats in sorted(accuracy.items()):
            if stats["total"] >= 5:
                parts.append(f"{key}: {stats['accuracy']:.0%} ({stats['total']})")
        if parts:
            _log(f"Accuracy: {' | '.join(parts)}")


def _llm_worker(signal_data_fn, underlying_prices_fn):
    """Background worker that runs inference on a schedule.

    Args:
        signal_data_fn: callable that returns signal data dict
        underlying_prices_fn: callable that returns {"XAG-USD": float, "XAU-USD": float}
    """
    _log("LLM worker started")
    while not _llm_stop.is_set():
        try:
            signal_data = signal_data_fn()
            prices = underlying_prices_fn()
            if prices:
                _run_inference_cycle(signal_data, prices)
            else:
                _log("No underlying prices available, skipping")
        except Exception as e:
            _log(f"LLM worker error: {e}")
            traceback.print_exc()

        # Wait for next cycle (check stop event every 10s)
        for _ in range(LLM_INTERVAL // 10):
            if _llm_stop.is_set():
                break
            time.sleep(10)

    _log("LLM worker stopped")


def start_llm_thread(signal_data_fn, underlying_prices_fn):
    """Start the background LLM inference thread.

    Args:
        signal_data_fn: callable() -> signal data dict
        underlying_prices_fn: callable() -> {"XAG-USD": float, "XAU-USD": float}
    """
    global _llm_thread
    if _llm_thread and _llm_thread.is_alive():
        _log("LLM thread already running")
        return

    _llm_stop.clear()
    _llm_thread = threading.Thread(
        target=_llm_worker,
        args=(signal_data_fn, underlying_prices_fn),
        daemon=True,
        name="metals-llm-worker",
    )
    _llm_thread.start()
    _log("LLM background thread started (interval: {}s)".format(LLM_INTERVAL))


def stop_llm_thread():
    """Stop the background LLM inference thread."""
    _llm_stop.set()
    if _llm_thread:
        _llm_thread.join(timeout=15)


def get_llm_signals():
    """Get latest LLM signals (thread-safe read)."""
    with _lock:
        return dict(_llm_signals)


def get_llm_accuracy():
    """Get rolling accuracy stats (thread-safe read)."""
    with _lock:
        return dict(_llm_accuracy)


def get_llm_age():
    """Get seconds since last inference run."""
    with _lock:
        if _llm_last_run == 0:
            return None
        return time.time() - _llm_last_run


def get_llm_summary():
    """Get a compact summary dict suitable for metals_context.json."""
    signals = get_llm_signals()
    accuracy = get_llm_accuracy()
    age = get_llm_age()

    summary = {
        "age_seconds": round(age) if age else None,
        "models": ["ministral", "chronos_1h", "chronos_3h"],
        "accuracy": {},
        "predictions": {},
    }

    for key, stats in accuracy.items():
        summary["accuracy"][key] = {
            "hit_rate": stats["accuracy"],
            "samples": stats["total"],
        }

    for ticker, data in signals.items():
        pred = {
            "consensus": data.get("consensus", {}),
        }
        if data.get("ministral"):
            pred["ministral"] = {
                "action": data["ministral"]["action"],
                "reasoning": data["ministral"]["reasoning"][:100],
            }
        for h in ["1h", "3h"]:
            key = f"chronos_{h}"
            if data.get(key):
                pred[key] = data[key]
        summary["predictions"][ticker] = pred

    return summary
