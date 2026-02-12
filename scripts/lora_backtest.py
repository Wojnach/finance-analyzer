#!/usr/bin/env python3
"""Offline backtest: compare CryptoTrader-LM vs Custom LoRA on fresh candles.

Downloads 1h candles from Binance, computes indicators, labels with actual
12h outcomes, runs both models, and reports accuracy + confusion matrix.

Runs on herc2 in .venv-llm (has llama-cpp-python + numpy via torch).
Usage:
    Q:\\models\\.venv-llm\\Scripts\\python.exe scripts\\lora_backtest.py
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
TICKER_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

if platform.system() == "Windows":
    MODEL_PATH = r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    ORIGINAL_LORA = r"Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf"
    CUSTOM_LORA = r"Q:\models\custom-trading-lora.gguf"
else:
    MODEL_PATH = (
        "/home/deck/models/ministral-8b-gguf/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    )
    ORIGINAL_LORA = "/home/deck/models/cryptotrader-lm/cryptotrader-lm-lora.gguf"
    CUSTOM_LORA = "/home/deck/models/custom-trading-lora.gguf"

REPO_DIR = Path(__file__).resolve().parent.parent


def download_candles(ticker, days=14):
    symbol = TICKER_MAP[ticker]
    limit = days * 24
    candles = []
    end_time = None

    while len(candles) < limit:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "limit": min(1000, limit - len(candles)),
        }
        if end_time:
            params["endTime"] = end_time
        r = requests.get(BINANCE_KLINES, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        batch = []
        for k in data:
            batch.append(
                {
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        candles = batch + candles
        end_time = data[0][0] - 1
        if len(data) < params["limit"]:
            break

    candles = candles[-limit:]
    print(f"  {ticker}: downloaded {len(candles)} candles")
    return candles


def compute_indicators(candles):
    close = np.array([c["close"] for c in candles])
    n = len(close)

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1 / 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    rs = np.divide(avg_gain, avg_loss, out=np.ones(n), where=avg_loss > 0)
    rsi = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    def ema(data, span):
        a = 2 / (span + 1)
        out = np.zeros(len(data))
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = a * data[i] + (1 - a) * out[i - 1]
        return out

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal

    # EMA(9, 21)
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)

    # Bollinger Bands(20, 2)
    bb_mid = np.zeros(n)
    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    for i in range(19, n):
        window = close[i - 19 : i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=0)
        bb_mid[i] = m
        bb_upper[i] = m + 2 * s
        bb_lower[i] = m - 2 * s

    for i, c in enumerate(candles):
        c["rsi"] = float(rsi[i])
        c["macd_hist"] = float(macd_hist[i])
        c["ema9"] = float(ema9[i])
        c["ema21"] = float(ema21[i])
        c["bb_upper"] = float(bb_upper[i])
        c["bb_lower"] = float(bb_lower[i])
        c["bb_mid"] = float(bb_mid[i])

    return candles


def label_outcomes(candles, lookahead=12, threshold=0.02):
    n = len(candles)
    for i in range(n):
        if i + lookahead >= n:
            candles[i]["label"] = None
            continue
        future_return = candles[i + lookahead]["close"] / candles[i]["close"] - 1
        if future_return > threshold:
            candles[i]["label"] = "BUY"
        elif future_return < -threshold:
            candles[i]["label"] = "SELL"
        else:
            candles[i]["label"] = "HOLD"
    return candles


def build_context(candle, ticker):
    if candle["close"] < candle["bb_lower"]:
        bb_pos = "below lower band"
    elif candle["close"] > candle["bb_upper"]:
        bb_pos = "above upper band"
    else:
        bb_pos = "inside bands"

    return {
        "ticker": ticker,
        "price_usd": candle["close"],
        "rsi": round(candle["rsi"], 1),
        "macd_hist": round(candle["macd_hist"], 2),
        "ema_bullish": candle["ema9"] > candle["ema21"],
        "bb_position": bb_pos,
        "fear_greed": "N/A",
        "fear_greed_class": "",
        "news_sentiment": "N/A",
        "timeframe_summary": "N/A",
        "headlines": "N/A",
    }


def load_model(lora_path):
    from llama_cpp import Llama

    return Llama(
        model_path=MODEL_PATH,
        lora_path=lora_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )


def predict(model, context):
    prompt = f"""[INST]You are an expert cryptocurrency trader. Based on the following market data, provide a single trading decision: BUY, SELL, or HOLD.

Market Data:
- Asset: {context['ticker']}
- Current Price: ${context['price_usd']:,.2f}
- 24h Change: N/A

Technical Indicators (15-minute candles):
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'}
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}

Market Sentiment:
- Fear & Greed Index: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- News Sentiment: {context.get('news_sentiment', 'N/A')}

Multi-timeframe Analysis:
{context.get('timeframe_summary', 'N/A')}

Recent Headlines:
{context.get('headlines', 'N/A')}

Respond with EXACTLY one of: BUY, SELL, or HOLD.
Then give a one-sentence reason.
Format: DECISION: [BUY/SELL/HOLD] - [reason][/INST]"""

    response = model(prompt, max_tokens=100, temperature=0.1, stop=["[INST]", "\n\n"])
    text = response["choices"][0]["text"].strip()

    decision = "HOLD"
    for word in ["BUY", "SELL", "HOLD"]:
        if word in text.upper():
            decision = word
            break

    return decision


def run_model(contexts, lora_path, model_name):
    print(f"\n  Loading {model_name}...")
    t0 = time.time()
    model = load_model(lora_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    predictions = []
    total = len(contexts)
    t0 = time.time()
    for i, ctx in enumerate(contexts):
        pred = predict(model, ctx)
        predictions.append(pred)
        if (i + 1) % 50 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{total}  ({rate:.1f}/s, ETA {eta:.0f}s)")

    del model
    import gc

    gc.collect()
    time.sleep(2)
    return predictions


def evaluate(predictions, actuals, classes=("BUY", "SELL", "HOLD")):
    n = len(predictions)
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / n if n > 0 else 0

    matrix = {actual: {pred: 0 for pred in classes} for actual in classes}
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for p, a in zip(predictions, actuals):
        matrix[a][p] += 1
        if p == a:
            per_class[a]["tp"] += 1
        else:
            per_class[p]["fp"] += 1
            per_class[a]["fn"] += 1

    class_stats = {}
    for c in classes:
        tp = per_class[c]["tp"]
        fp = per_class[c]["fp"]
        fn = per_class[c]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_stats[c] = {"precision": precision, "recall": recall, "support": tp + fn}

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "confusion_matrix": matrix,
        "per_class": class_stats,
    }


def print_report(results):
    for model_name, model_results in results.items():
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        for ticker, data in model_results.items():
            stats = data["stats"]
            print(
                f"\n  {ticker}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})"
            )

            print(
                f"\n  {'':>8}  {'BUY':>6}  {'SELL':>6}  {'HOLD':>6}  | {'Prec':>6}  {'Recall':>6}"
            )
            print(
                f"  {'':>8}  {'----':>6}  {'----':>6}  {'----':>6}  | {'----':>6}  {'------':>6}"
            )
            for actual in ("BUY", "SELL", "HOLD"):
                row = stats["confusion_matrix"][actual]
                pc = stats["per_class"][actual]
                print(
                    f"  {actual:>8}  {row['BUY']:>6}  {row['SELL']:>6}  {row['HOLD']:>6}"
                    f"  | {pc['precision']*100:>5.1f}%  {pc['recall']*100:>5.1f}%"
                )

    # Agreement
    print(f"\n{'='*50}")
    print("  Agreement")
    print(f"{'='*50}")
    model_names = list(results.keys())
    if len(model_names) == 2:
        m1, m2 = model_names
        for ticker in results[m1]:
            p1 = results[m1][ticker]["predictions"]
            p2 = results[m2][ticker]["predictions"]
            agree = sum(1 for a, b in zip(p1, p2) if a == b)
            total = len(p1)
            print(f"  {ticker}: {agree}/{total} ({agree/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA backtest: compare original vs custom"
    )
    parser.add_argument(
        "--days", type=int, default=14, help="Days of data (default: 14)"
    )
    parser.add_argument("--tickers", default="BTC,ETH", help="Comma-separated tickers")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    output_path = (
        Path(args.output)
        if args.output
        else REPO_DIR / "data" / "lora_backtest_results.json"
    )

    print(f"LoRA Backtest — {args.days} days, tickers: {', '.join(tickers)}")
    print(f"Models: {MODEL_PATH}")
    print(f"  Original LoRA: {ORIGINAL_LORA}")
    print(f"  Custom LoRA:   {CUSTOM_LORA}")

    all_contexts = {}
    all_labels = {}

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        candles = download_candles(ticker, days=args.days)
        candles = compute_indicators(candles)
        candles = label_outcomes(candles, lookahead=12, threshold=0.02)

        valid = [c for c in candles[26:] if c["label"] is not None]
        print(f"  Valid candles: {len(valid)} (after indicators + lookahead trim)")

        label_counts = {}
        for c in valid:
            label_counts[c["label"]] = label_counts.get(c["label"], 0) + 1
        print(f"  Label distribution: {label_counts}")

        contexts = [build_context(c, ticker) for c in valid]
        labels = [c["label"] for c in valid]

        all_contexts[ticker] = contexts
        all_labels[ticker] = labels

    models = {
        "CryptoTrader-LM (original)": ORIGINAL_LORA,
        "Custom LoRA": CUSTOM_LORA,
    }

    results = {}
    for model_name, lora_path in models.items():
        results[model_name] = {}
        for ticker in tickers:
            preds = run_model(
                all_contexts[ticker], lora_path, f"{model_name} / {ticker}"
            )
            stats = evaluate(preds, all_labels[ticker])
            results[model_name][ticker] = {"predictions": preds, "stats": stats}

    print_report(results)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "tickers": tickers,
        "models": {},
    }
    for model_name in results:
        output["models"][model_name] = {}
        for ticker in tickers:
            stats = results[model_name][ticker]["stats"]
            output["models"][model_name][ticker] = stats

    # Agreement stats
    model_names = list(results.keys())
    if len(model_names) == 2:
        m1, m2 = model_names
        agreement = {}
        for ticker in tickers:
            p1 = results[m1][ticker]["predictions"]
            p2 = results[m2][ticker]["predictions"]
            agree = sum(1 for a, b in zip(p1, p2) if a == b)
            agreement[ticker] = {
                "agree": agree,
                "total": len(p1),
                "rate": agree / len(p1),
            }
        output["agreement"] = agreement

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
