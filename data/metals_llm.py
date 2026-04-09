"""
Local LLM inference for metals trading loop.

Runs Ministral-8B (metals-adapted prompt) every 5 minutes and Chronos (time-series
forecast) every 60 seconds in a background thread. Separate timers let Chronos
build accuracy samples ~5x faster. Tracks prediction accuracy at 1h and 3h
horizons. Weights model signals by their proven accuracy.

Usage from metals_loop.py:
    from metals_llm import start_llm_thread, get_llm_signals, get_llm_accuracy

    start_llm_thread()  # launch background inference thread
    signals = get_llm_signals()  # get latest cached results
    accuracy = get_llm_accuracy()  # get rolling accuracy per model
"""

import datetime
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback

os.chdir(r"Q:/finance-analyzer")
sys.path.insert(0, ".")

import requests

from portfolio.file_utils import atomic_append_jsonl

# --- CONFIG ---
LLM_INTERVAL = 300       # run Ministral every 5 minutes
CHRONOS_INTERVAL = 60    # run Chronos every 60 seconds (builds samples ~5x faster)

# Quiet hours: reduced intervals to cut CPU fan noise after market close
LLM_INTERVAL_QUIET = 1800       # Ministral every 30 minutes
CHRONOS_INTERVAL_QUIET = 300    # Chronos every 5 minutes
ACCURACY_WINDOW = 50     # rolling window for accuracy calculation
PREDICTION_LOG = "data/metals_llm_predictions.jsonl"
PREDICTION_OUTCOMES_LOG = "data/metals_llm_outcomes.jsonl"

# Binance endpoints for klines
FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"
SPOT_KLINES_BASE = "https://api.binance.com/api/v3/klines"

# Tracked symbols: metals (FAPI) + crypto (SPOT)
TRACKED_SYMBOLS = {
    "XAG-USD": "XAGUSDT",
    "XAU-USD": "XAUUSDT",
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}
# Crypto tickers use Binance SPOT, metals use FAPI
_CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}

# Backwards compat alias
METALS_SYMBOLS = TRACKED_SYMBOLS

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

# --- PERSISTENT MINISTRAL SERVER ---
_ministral_proc = None
_ministral_job = None  # Windows Job Object handle (auto-kill on parent death)
_ministral_lock = threading.Lock()  # serialize access to the persistent process

# --- PERSISTENT CHRONOS SERVER ---
_chronos_proc = None
_chronos_job = None  # Windows Job Object handle (auto-kill on parent death)
_chronos_lock = threading.Lock()
CHRONOS_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "chronos_server.py")

# Accuracy horizons in seconds
HORIZONS = {
    "1h": 3600,
    "3h": 10800,
}


def _log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [LLM] {msg}", flush=True)


def _is_quiet_hours():
    """True when outside EU/US market hours — reduce LLM frequency to cut CPU load.

    Quiet: weekdays 22:00-08:15 CET, all day weekends.
    Active: weekdays 08:15-22:00 CET (covers EU open through US close).
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now = datetime.datetime.now(ZoneInfo("Europe/Stockholm"))
    if now.weekday() >= 5:  # Saturday/Sunday
        return True
    h = now.hour + now.minute / 60.0
    return h < 8.25 or h >= 22.0


def _start_ministral_server():
    """Launch Ministral in persistent --server mode. Returns Popen or None."""
    global _ministral_proc, _ministral_job
    try:
        from portfolio.subprocess_utils import popen_in_job
        proc, job = popen_in_job(
            [MINISTRAL_PYTHON, MINISTRAL_SCRIPT, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Wait for MINISTRAL_READY on stderr (model loaded)
        deadline = time.time() + 60  # 60s timeout for model load
        while time.time() < deadline:
            if proc.poll() is not None:
                _log(f"Ministral server exited during startup (code {proc.returncode})")
                return None
            # Read stderr non-blocking (Windows doesn't have select on pipes)
            try:
                line = proc.stderr.readline()
                if "MINISTRAL_READY" in line:
                    _log("Ministral server ready (model loaded, persistent)")
                    _ministral_proc = proc
                    _ministral_job = job
                    return proc
            except Exception:
                time.sleep(0.5)
        _log("Ministral server startup timed out (60s)")
        proc.kill()
        return None
    except Exception as e:
        _log(f"Ministral server launch failed: {e}")
        return None


def _stop_ministral_server():
    """Stop the persistent Ministral server if running."""
    global _ministral_proc, _ministral_job
    if _ministral_proc is not None:
        try:
            _ministral_proc.stdin.close()
            _ministral_proc.wait(timeout=10)
        except Exception:
            try:
                _ministral_proc.kill()
            except Exception:
                pass
        _ministral_proc = None
        _log("Ministral server stopped")
    if _ministral_job is not None:
        try:
            from portfolio.subprocess_utils import close_job
            close_job(_ministral_job)
        except Exception:
            pass
        _ministral_job = None


def _query_ministral_server(context):
    """Send a request to the persistent Ministral server. Returns result dict or None."""
    global _ministral_proc
    with _ministral_lock:
        if _ministral_proc is None or _ministral_proc.poll() is not None:
            _ministral_proc = None
            _start_ministral_server()
        if _ministral_proc is None:
            return None  # server failed to start, caller falls back
        try:
            _ministral_proc.stdin.write(json.dumps(context) + "\n")
            _ministral_proc.stdin.flush()
            response_line = _readline_with_timeout(_ministral_proc.stdout, timeout_s=120)
            if not response_line:
                _log("Ministral server timed out or empty response, restarting")
                _stop_ministral_server()
                return None
            return json.loads(response_line.strip())
        except Exception as e:
            _log(f"Ministral server query failed: {e}")
            _stop_ministral_server()
            return None


def _start_chronos_server():
    """Launch Chronos in persistent server mode. Returns Popen or None.

    Checks VRAM before loading (Codex finding #6). Chronos uses ~673MB
    which coexists with llama-server, but we verify before loading.
    """
    global _chronos_proc, _chronos_job
    try:
        from portfolio.gpu_gate import get_vram_usage
        vram = get_vram_usage()
        if vram and vram["free_mb"] < 1000:
            _log(f"Chronos: insufficient VRAM ({vram['free_mb']}MB free, need 1000MB)")
            return None
    except ImportError:
        pass
    try:
        from portfolio.subprocess_utils import popen_in_job
        proc, job = popen_in_job(
            [MAIN_PYTHON, "-u", CHRONOS_SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        deadline = time.time() + 120  # Chronos model load can be slow
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read()
                _log(f"Chronos server exited during startup (code {proc.returncode}): {stderr[:200]}")
                return None
            line = proc.stderr.readline()
            if "CHRONOS_READY" in line:
                _log("Chronos server ready (model loaded, persistent)")
                _chronos_proc = proc
                _chronos_job = job
                return proc
            if "CHRONOS_FAILED" in line:
                _log(f"Chronos server failed: {line.strip()}")
                proc.kill()
                return None
        _log("Chronos server startup timed out (120s)")
        proc.kill()
        return None
    except Exception as e:
        _log(f"Chronos server launch failed: {e}")
        return None


def _stop_chronos_server():
    """Stop the persistent Chronos server if running."""
    global _chronos_proc, _chronos_job
    if _chronos_proc is not None:
        try:
            _chronos_proc.stdin.close()
            _chronos_proc.wait(timeout=10)
        except Exception:
            try:
                _chronos_proc.kill()
            except Exception:
                pass
        _chronos_proc = None
        _log("Chronos server stopped")
    if _chronos_job is not None:
        try:
            from portfolio.subprocess_utils import close_job
            close_job(_chronos_job)
        except Exception:
            pass
        _chronos_job = None


def _readline_with_timeout(stream, timeout_s=60):
    """Read a line from a pipe with a timeout (Codex finding #7).

    Uses a background thread since Python pipes don't support select() on Windows.
    Returns the line or None on timeout.
    """
    result = [None]

    def _read():
        result[0] = stream.readline()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return None  # timed out — caller should restart server
    return result[0]


def _query_chronos_server(close_prices, horizons=(1, 3)):
    """Send a request to the persistent Chronos server. Returns result dict or None."""
    global _chronos_proc
    with _chronos_lock:
        if _chronos_proc is None or _chronos_proc.poll() is not None:
            _chronos_proc = None
            _start_chronos_server()
        if _chronos_proc is None:
            return None
        try:
            req = json.dumps({"close_prices": close_prices, "horizons": list(horizons)})
            _chronos_proc.stdin.write(req + "\n")
            _chronos_proc.stdin.flush()
            response_line = _readline_with_timeout(_chronos_proc.stdout, timeout_s=60)
            if not response_line:
                _log("Chronos server timed out or empty response, restarting")
                _stop_chronos_server()
                return None
            result = json.loads(response_line.strip())
            if "error" in result:
                _log(f"Chronos server error: {result['error']}")
                return None
            return result
        except Exception as e:
            _log(f"Chronos server query failed: {e}")
            _stop_chronos_server()
            return None


def _fetch_fapi_klines(symbol, interval="1h", limit=200, ticker=None):
    """Fetch klines from Binance FAPI (metals) or SPOT (crypto).

    Args:
        symbol: Binance symbol (e.g. XAGUSDT, BTCUSDT)
        interval: candle interval (1h, 5m, etc.)
        limit: number of candles
        ticker: original ticker key (e.g. BTC-USD) — used to pick SPOT vs FAPI
    """
    try:
        # Crypto tickers use SPOT; metals use FAPI
        is_crypto = ticker in _CRYPTO_TICKERS if ticker else symbol in ("BTCUSDT", "ETHUSDT")
        base_url = SPOT_KLINES_BASE if is_crypto else FAPI_BASE

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        r = requests.get(base_url, params=params, timeout=15)
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


def _build_ministral8_prompt(context):
    """Build prompt for Ministral-8B (metals loop format)."""
    return f"""[INST]You are an expert cryptocurrency trader. Based on the following market data, provide a single trading decision: BUY, SELL, or HOLD.

Market Data:
- Asset: {context.get('ticker', 'XAG-USD')}
- Current Price: ${context.get('price_usd', 0):,.2f}
- 24h Change: {context.get('change_24h', 'N/A')}

Technical Indicators (1-hour candles):
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'} (gap: {context.get('ema_gap_pct', 'N/A')}%)
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}

Market Sentiment:
- Fear & Greed Index: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- News Sentiment: {context.get('news_sentiment', 'N/A')} (confidence: {context.get('sentiment_confidence', 'N/A')})

Multi-timeframe Analysis:
{context.get('timeframe_summary', 'N/A')}

Recent Headlines:
{context.get('headlines', 'N/A')}

Respond with EXACTLY one of: BUY, SELL, or HOLD.
Then give a one-sentence reason.
Format: DECISION: [BUY/SELL/HOLD] - [reason][/INST]"""


def _run_ministral_metals(context):
    """Run Ministral-8B with metals-adapted prompt via shared llama-server.

    Uses the unified llama-server (HTTP) with model swapping.
    Falls back to subprocess.run per-call if server fails.

    Returns {"action": "BUY/SELL/HOLD", "reasoning": "...", "confidence": 0.0-1.0}
    """
    try:
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

        # Try shared llama-server first
        try:
            from portfolio.llama_server import query_llama_server
            prompt = _build_ministral8_prompt(metals_context)
            text = query_llama_server("ministral8_lora", prompt,
                                      n_predict=100, temperature=0.1,
                                      stop=["[INST]", "\n\n"])
            if text is not None:
                decision = "HOLD"
                for word in ["BUY", "SELL", "HOLD"]:
                    if word in text.upper():
                        decision = word
                        break
                return {
                    "action": decision,
                    "reasoning": text[:200],
                    "confidence": 0.6,
                }
        except ImportError:
            pass  # portfolio not on path, fall through

        # Try stdin/stdout persistent server (legacy)
        result = _query_ministral_server(metals_context)
        if result is not None:
            return {
                "action": result.get("action", "HOLD"),
                "reasoning": result.get("reasoning", "")[:200],
                "confidence": 0.6,
            }

        # Fallback: one-shot subprocess (cold start)
        _log("All Ministral servers unavailable, falling back to subprocess")
        proc = subprocess.run(
            [MINISTRAL_PYTHON, MINISTRAL_SCRIPT],
            input=json.dumps(metals_context),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            _log(f"Ministral fallback failed: {proc.stderr[:200]}")
            return None
        result = json.loads(proc.stdout.strip())

        return {
            "action": result.get("action", "HOLD"),
            "reasoning": result.get("reasoning", "")[:200],
            "confidence": 0.6,
        }
    except Exception as e:
        _log(f"Ministral error: {e}")
        return None


def _run_chronos_metals(ticker, close_prices, horizons=(1, 3)):
    """Run Chronos forecast via persistent server (only ~673MB VRAM — coexists with llama-server).

    Falls back to subprocess if server unavailable.

    Returns dict of {horizon_key: {"direction": "up/down", "pct_move": float, "confidence": float}}
    """
    try:
        # Persistent server (fast, ~0.3s per query, minimal VRAM)
        result = _query_chronos_server(close_prices, horizons)
        if result is not None:
            if "error" in result:
                _log(f"Chronos error for {ticker}: {result['error']}")
                return None
            return result

        # Fallback: subprocess per-call
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
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
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
        atomic_append_jsonl(PREDICTION_LOG, entry)
    except Exception as e:
        _log(f"Prediction log error: {e}")


def _load_resolved_prediction_keys():
    """Load set of already-resolved prediction timestamps from outcomes file."""
    resolved = set()
    if not os.path.exists(PREDICTION_OUTCOMES_LOG):
        return resolved
    try:
        with open(PREDICTION_OUTCOMES_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    key = (entry.get("pred_ts", ""), entry.get("model", ""),
                           entry.get("ticker", ""), entry.get("horizon", ""))
                    resolved.add(key)
                except (json.JSONDecodeError, KeyError):
                    pass
    except Exception as e:
        _log(f"Prediction outcomes load error: {e}")
    return resolved


def check_prediction_accuracy(current_prices):
    """Check old predictions against current prices.

    Outcomes are written to a SEPARATE append-only file (metals_llm_outcomes.jsonl)
    so the prediction log itself is never rewritten. This eliminates the race
    condition where a concurrent _log_prediction() append could be lost during rewrite.

    Returns dict: {model_horizon: {correct: int, total: int, accuracy: float}}
    """
    now = time.time()
    accuracy = {}

    try:
        if not os.path.exists(PREDICTION_LOG):
            return accuracy

        # Load already-resolved keys
        resolved_keys = _load_resolved_prediction_keys()

        # Read predictions (append-only, never rewritten)
        entries = []
        with open(PREDICTION_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except (json.JSONDecodeError, ValueError) as e:
                    _log(f"Prediction log parse error: {e}")

        if not entries:
            return accuracy

        new_outcomes = []

        for entry in entries:
            pred_ts_str = entry.get("ts", "")
            model = entry.get("model", "?")
            ticker = entry.get("ticker", "")
            horizon_key = entry.get("horizon", "1h")

            # Skip if already resolved
            dedup_key = (pred_ts_str, model, ticker, horizon_key)
            if dedup_key in resolved_keys:
                continue

            # Parse timestamp
            try:
                pred_ts = datetime.datetime.fromisoformat(pred_ts_str)
                pred_epoch = pred_ts.timestamp()
            except (ValueError, KeyError):
                continue

            horizon_secs = HORIZONS.get(horizon_key, 3600)
            elapsed = now - pred_epoch
            if elapsed < horizon_secs * 0.9:
                continue

            if ticker not in current_prices:
                continue

            actual_price = current_prices[ticker]
            pred_price = entry.get("price_at_prediction", 0)
            if pred_price <= 0 or actual_price <= 0:
                continue

            pred_direction = entry.get("direction", "flat")
            if pred_direction == "flat":
                # Record as skipped
                outcome_entry = {
                    "pred_ts": pred_ts_str,
                    "model": model,
                    "ticker": ticker,
                    "horizon": horizon_key,
                    "outcome": "skipped",
                    "correct": None,
                    "resolved_at": datetime.datetime.now(datetime.UTC).isoformat(),
                }
                new_outcomes.append(outcome_entry)
                resolved_keys.add(dedup_key)
                continue

            actual_direction = "up" if actual_price > pred_price else "down"
            correct = (pred_direction == actual_direction)

            outcome_entry = {
                "pred_ts": pred_ts_str,
                "model": model,
                "ticker": ticker,
                "horizon": horizon_key,
                "outcome": actual_direction,
                "outcome_price": round(actual_price, 4),
                "pred_price": round(pred_price, 4),
                "correct": correct,
                "resolved_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            new_outcomes.append(outcome_entry)
            resolved_keys.add(dedup_key)

        # Append new outcomes (prediction log stays untouched)
        for outcome in new_outcomes:
            atomic_append_jsonl(PREDICTION_OUTCOMES_LOG, outcome)

        # Compute accuracy from outcomes file
        if os.path.exists(PREDICTION_OUTCOMES_LOG):
            with open(PREDICTION_OUTCOMES_LOG, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                    except (json.JSONDecodeError, ValueError):
                        continue

                    if rec.get("correct") is None:
                        continue

                    key = f"{rec.get('model', '?')}_{rec.get('horizon', '1h')}"
                    if key not in accuracy:
                        accuracy[key] = {"correct": 0, "total": 0, "accuracy": 0.0}

                    accuracy[key]["total"] += 1
                    if rec["correct"]:
                        accuracy[key]["correct"] += 1

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


def _run_inference_cycle(signal_data, underlying_prices, chronos_only=False):
    """Run one complete inference cycle for all tracked tickers (metals + crypto).

    Args:
        signal_data: dict from read_signal_data() in metals_loop
        underlying_prices: {"XAG-USD": float, "XAU-USD": float, "BTC-USD": float, ...}
        chronos_only: if True, skip Ministral and only run Chronos (faster cycle)
    """
    global _llm_signals, _llm_accuracy, _llm_last_run

    # Preserve existing signals in chronos_only mode (keep Ministral data)
    with _lock:
        results = dict(_llm_signals) if chronos_only else {}

    for ticker, binance_sym in TRACKED_SYMBOLS.items():
        current_price = underlying_prices.get(ticker, 0)
        if current_price <= 0:
            continue

        if chronos_only and ticker in results:
            # Preserve existing Ministral data, just update price/timestamp
            ticker_result = results[ticker]
            ticker_result["price"] = current_price
            ticker_result["ts"] = datetime.datetime.now(datetime.UTC).isoformat()
        else:
            ticker_result = {
                "ticker": ticker,
                "price": current_price,
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                "ministral": None,
                "chronos_1h": None,
                "chronos_3h": None,
                "consensus": {"direction": "flat", "confidence": 0, "weighted_action": "HOLD"},
            }

        # --- Ministral inference (skip in chronos_only mode) ---
        if not chronos_only:
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
            candles = _fetch_fapi_klines(binance_sym, interval="1h", limit=200, ticker=ticker)
            if candles and len(candles) >= 50:
                close_prices = [c["close"] for c in candles]

                chronos_result = _run_chronos_metals(ticker, close_prices, horizons=(1, 3))
                if chronos_result:
                    for h_key, h_data in chronos_result.items():
                        if isinstance(h_data, dict):
                            # forecast_chronos returns "action" (BUY/SELL/HOLD), map to direction
                            action = h_data.get("action", "HOLD")
                            direction = "up" if action == "BUY" else ("down" if action == "SELL" else "flat")
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
                                _log_prediction("chronos", ticker, direction,
                                                conf, current_price, log_horizon)

                    _log(f"Chronos {ticker}: {json.dumps({k: h_data.get('action', '?') for k, h_data in chronos_result.items() if isinstance(h_data, dict)})}")
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
    """Background worker that runs inference on separate schedules.

    Ministral runs every LLM_INTERVAL (5 min) — heavier GPU load, needs signal context.
    Chronos runs every CHRONOS_INTERVAL (60s) — fast subprocess, builds samples ~5x faster.

    Covers all tracked symbols: XAG, XAU (FAPI), BTC, ETH (SPOT).

    Args:
        signal_data_fn: callable that returns signal data dict
        underlying_prices_fn: callable that returns {"XAG-USD": float, ..., "BTC-USD": float, ...}
    """
    _log("LLM worker started")
    last_ministral = 0
    last_chronos = 0
    _was_quiet = None  # track transitions for logging

    while not _llm_stop.is_set():
        try:
            now = time.time()
            quiet = _is_quiet_hours()

            # Log transitions
            if quiet != _was_quiet:
                if quiet:
                    _log(f"Quiet hours — Ministral every {LLM_INTERVAL_QUIET}s, "
                         f"Chronos every {CHRONOS_INTERVAL_QUIET}s")
                else:
                    _log(f"Market hours — Ministral every {LLM_INTERVAL}s, "
                         f"Chronos every {CHRONOS_INTERVAL}s")
                _was_quiet = quiet

            ministral_interval = LLM_INTERVAL_QUIET if quiet else LLM_INTERVAL
            chronos_interval = CHRONOS_INTERVAL_QUIET if quiet else CHRONOS_INTERVAL

            prices = underlying_prices_fn()

            if not prices:
                _log("No underlying prices available, skipping")
            elif now - last_ministral >= ministral_interval:
                # Full cycle: Ministral + Chronos
                signal_data = signal_data_fn()
                _run_inference_cycle(signal_data, prices, chronos_only=False)
                last_ministral = time.time()
                last_chronos = time.time()  # Chronos ran too
            elif now - last_chronos >= chronos_interval:
                # Chronos-only fast cycle
                _run_inference_cycle({}, prices, chronos_only=True)
                last_chronos = time.time()
        except Exception as e:
            _log(f"LLM worker error: {e}")
            traceback.print_exc()

        # Poll every 10s (responsive to stop events, short enough for 60s Chronos interval)
        if not _llm_stop.is_set():
            time.sleep(10)

    _log("LLM worker stopped")


def _sweep_orphaned_servers():
    """Kill orphaned Chronos/Ministral processes from a previous crash.

    Safe to call at startup because the metals_loop singleton lock guarantees
    we are the only metals_loop — any matching processes are orphans.
    """
    try:
        from portfolio.subprocess_utils import kill_orphaned_by_cmdline
        killed = 0
        killed += kill_orphaned_by_cmdline("chronos_server.py")
        killed += kill_orphaned_by_cmdline("ministral_trader.py")
        if killed:
            _log(f"Swept {killed} orphaned LLM server(s) from previous session")
    except Exception as e:
        _log(f"Orphan sweep failed (non-fatal): {e}")


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

    # Kill any orphaned servers from a previous crash before starting new ones
    _sweep_orphaned_servers()

    _llm_stop.clear()
    _llm_thread = threading.Thread(
        target=_llm_worker,
        args=(signal_data_fn, underlying_prices_fn),
        daemon=True,
        name="metals-llm-worker",
    )
    _llm_thread.start()
    _log(f"LLM background thread started (Ministral: {LLM_INTERVAL}s, Chronos: {CHRONOS_INTERVAL}s)")


def stop_llm_thread():
    """Stop the background LLM inference thread and persistent servers."""
    _llm_stop.set()
    if _llm_thread:
        _llm_thread.join(timeout=15)
    _stop_ministral_server()
    _stop_chronos_server()


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
