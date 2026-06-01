#!/usr/bin/env python3
"""Offline A/B benchmark: current Qwen3 prompt vs FinCoT variant at 3h horizon.

Variant A = current production prompt verbatim (from portfolio/qwen3_trader.py).
Variant B = FinCoT: adds explicit chain-of-thought instruction, removes the
            HOLD-preference sentence "A confident HOLD is better than a
            low-confidence BUY/SELL." (documented in docs/LLM_FOLLOWUPS_20260518.md §3).

Scoring convention (matches repo directional-accuracy everywhere):
  - Denominator: windows where realized outcome is BUY or SELL *AND* the model
    predicted BUY or SELL (i.e. exclude HOLD predictions AND flat realized moves).
  - Numerator: correct directional predictions in that filtered set.

Usage:
    .venv/Scripts/python.exe scripts/qwen3_fincot_ab.py \\
        --tickers BTC,ETH --days 14 --max-windows 40

2026-06-01 (llm-stack): new standalone script, does NOT touch any live module.
See docs/LLM_FOLLOWUPS_20260518.md §3 for the hypothesis and the A/B design.
"""

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
TICKER_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

# ---------------------------------------------------------------------------
# Candle download + indicator helpers — mirrored from scripts/lora_backtest.py
# ---------------------------------------------------------------------------

def download_candles(ticker: str, days: int = 14) -> list[dict]:
    """Fetch 1h Binance candles for `days` days."""
    symbol = TICKER_MAP[ticker]
    limit = days * 24
    candles: list[dict] = []
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
            batch.append({
                "ts":     int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })
        candles = batch + candles
        end_time = data[0][0] - 1
        if len(data) < params["limit"]:
            break

    candles = candles[-limit:]
    print(f"  {ticker}: {len(candles)} candles downloaded")
    return candles


def compute_indicators(candles: list[dict]) -> list[dict]:
    """Compute RSI, MACD, EMA9/21, Bollinger Bands in-place."""
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

    def ema(data, span):
        a = 2 / (span + 1)
        out = np.zeros(len(data))
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = a * data[i] + (1 - a) * out[i - 1]
        return out

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_hist = (ema12 - ema26) - ema(ema12 - ema26, 9)
    ema9  = ema(close, 9)
    ema21 = ema(close, 21)

    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    bb_mid   = np.zeros(n)
    for i in range(19, n):
        window = close[i - 19: i + 1]
        m = float(np.mean(window))
        s = float(np.std(window, ddof=0))
        bb_mid[i]   = m
        bb_upper[i] = m + 2 * s
        bb_lower[i] = m - 2 * s

    for i, c in enumerate(candles):
        c["rsi"]       = float(rsi[i])
        c["macd_hist"] = float(macd_hist[i])
        c["ema9"]      = float(ema9[i])
        c["ema21"]     = float(ema21[i])
        c["bb_upper"]  = float(bb_upper[i])
        c["bb_lower"]  = float(bb_lower[i])
        c["bb_mid"]    = float(bb_mid[i])
        # Volume ratio vs rolling 20-bar mean
        if i >= 19:
            vol_window = np.array([candles[j]["volume"] for j in range(i - 19, i + 1)])
            mean_vol = float(np.mean(vol_window[:-1])) if np.mean(vol_window[:-1]) > 0 else 1.0
            c["volume_ratio"] = round(c["volume"] / mean_vol, 2)
        else:
            c["volume_ratio"] = 1.0
    return candles


def label_outcomes(candles: list[dict], lookahead: int = 3,
                   threshold: float = 0.002) -> list[dict]:
    """Label each bar with realized direction at lookahead bars ahead.

    threshold=0.002 (0.2%) chosen for 3h/1h crypto horizon.
    lora_backtest uses 2% for 12h — scale down proportionally.
    """
    n = len(candles)
    for i in range(n):
        if i + lookahead >= n:
            candles[i]["label"] = None
            continue
        ret = candles[i + lookahead]["close"] / candles[i]["close"] - 1
        if ret > threshold:
            candles[i]["label"] = "BUY"
        elif ret < -threshold:
            candles[i]["label"] = "SELL"
        else:
            candles[i]["label"] = "HOLD"
    return candles


def build_context(candle: dict, ticker: str) -> dict:
    """Build the context dict that qwen3_trader._build_prompt expects."""
    if candle["close"] < candle["bb_lower"]:
        bb_pos = "below lower band"
    elif candle["close"] > candle["bb_upper"]:
        bb_pos = "above upper band"
    else:
        bb_pos = "inside bands"

    ema_gap_pct = round(
        (candle["ema9"] - candle["ema21"]) / candle["ema21"] * 100
        if candle["ema21"] != 0 else 0.0, 2
    )

    return {
        "ticker":              ticker,
        "asset_type":          "cryptocurrency",
        "price_usd":           candle["close"],
        "rsi":                 round(candle["rsi"], 1),
        "macd_hist":           round(candle["macd_hist"], 2),
        "ema_bullish":         candle["ema9"] > candle["ema21"],
        "ema_gap_pct":         ema_gap_pct,
        "bb_position":         bb_pos,
        "volume_ratio":        candle.get("volume_ratio", "N/A"),
        "fear_greed":          "N/A",
        "fear_greed_class":    "",
        "news_sentiment":      "N/A",
        "sentiment_confidence":"N/A",
        "timeframe_summary":   "N/A",
    }


# ---------------------------------------------------------------------------
# Prompt builders — Variant A (current) and Variant B (FinCoT)
# ---------------------------------------------------------------------------

def _build_prompt_variant_a(context: dict) -> str:
    """Variant A: verbatim current production prompt from qwen3_trader._build_prompt.

    Source: portfolio/qwen3_trader.py _build_prompt() as of 2026-06-01.
    Quoted exactly — do NOT edit this without re-reading the source file.
    """
    ticker     = context.get("ticker", "UNKNOWN")
    asset_type = context.get("asset_type", "cryptocurrency")

    return f"""<|im_start|>system
You are an expert financial analyst specializing in {asset_type} trading.

Your job is to deeply analyze market data and produce a high-quality trading decision.
Think carefully before answering. Consider:
1. Are the technical indicators confirming each other or diverging?
2. Is the trend strengthening or weakening across timeframes?
3. Does volume support the move? Low volume signals are unreliable.
4. Is sentiment aligned with technicals, or is it contrarian?
5. What is the risk/reward setup? Is there a clear edge or is it ambiguous?

After your analysis, respond with a JSON object:
{{"action":"BUY|SELL|HOLD","confidence":0-100,"reasoning":"2-3 sentences explaining your logic"}}

Use HOLD when evidence is mixed, weak, or conflicting. A confident HOLD is better than a low-confidence BUY/SELL.
Confidence guide: 80+ = strong conviction, 60-79 = moderate, 40-59 = weak/mixed, <40 = default to HOLD.<|im_end|>
<|im_start|>user
Asset: {ticker} ({asset_type})
Current Price: ${context.get('price_usd', 0):,.2f}

Technical Indicators:
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'} (gap: {context.get('ema_gap_pct', 'N/A')}%)
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}
- Volume Ratio: {context.get('volume_ratio', 'N/A')}x avg

Market Context:
- Fear & Greed: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- Sentiment: {context.get('news_sentiment', 'N/A')} (conf: {context.get('sentiment_confidence', 'N/A')})

Multi-timeframe: {context.get('timeframe_summary', 'N/A')}

Analyze the data thoroughly, then provide your trading decision as JSON.<|im_end|>
<|im_start|>assistant
"""


def _build_prompt_variant_b(context: dict) -> str:
    """Variant B: FinCoT — removes HOLD-preference sentence, adds explicit CoT instruction.

    Two changes vs Variant A (2026-06-01):
      (1) Removed: "A confident HOLD is better than a low-confidence BUY/SELL."
          Rationale: identified in docs/LLM_FOLLOWUPS_20260518.md §3 as the
          primary driver of >95% HOLD rate in production. The model takes
          this literally as a tie-breaker in favour of HOLD.
      (2) Added explicit financial chain-of-thought instruction before the
          JSON schema: "Reason step-by-step about trend, momentum,
          mean-reversion, and risk factors FIRST, then give your decision."
          FinCoT hypothesis: explicit CoT elicitation improves directional
          reasoning without changing the output format.
    """
    ticker     = context.get("ticker", "UNKNOWN")
    asset_type = context.get("asset_type", "cryptocurrency")

    return f"""<|im_start|>system
You are an expert financial analyst specializing in {asset_type} trading.

Your job is to deeply analyze market data and produce a high-quality trading decision.
Think carefully before answering. Consider:
1. Are the technical indicators confirming each other or diverging?
2. Is the trend strengthening or weakening across timeframes?
3. Does volume support the move? Low volume signals are unreliable.
4. Is sentiment aligned with technicals, or is it contrarian?
5. What is the risk/reward setup? Is there a clear edge or is it ambiguous?

Reason step-by-step about trend direction, momentum signals, mean-reversion pressure, and risk factors FIRST, then commit to your best decision.

After your analysis, respond with a JSON object:
{{"action":"BUY|SELL|HOLD","confidence":0-100,"reasoning":"2-3 sentences explaining your logic"}}

Use HOLD when evidence is mixed, weak, or conflicting.
Confidence guide: 80+ = strong conviction, 60-79 = moderate, 40-59 = weak/mixed, <40 = default to HOLD.<|im_end|>
<|im_start|>user
Asset: {ticker} ({asset_type})
Current Price: ${context.get('price_usd', 0):,.2f}

Technical Indicators:
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'} (gap: {context.get('ema_gap_pct', 'N/A')}%)
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}
- Volume Ratio: {context.get('volume_ratio', 'N/A')}x avg

Market Context:
- Fear & Greed: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- Sentiment: {context.get('news_sentiment', 'N/A')} (conf: {context.get('sentiment_confidence', 'N/A')})

Multi-timeframe: {context.get('timeframe_summary', 'N/A')}

Analyze the data thoroughly, then provide your trading decision as JSON.<|im_end|>
<|im_start|>assistant
"""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _query_variant(prompt: str) -> tuple[str, float | None]:
    """Send prompt to qwen3 via llama_server. Returns (action, confidence)."""
    from portfolio.llama_server import query_llama_server
    from portfolio.qwen3_trader import _parse_response

    # 2026-06-01: use same params as production qwen3_signal._call_qwen3
    text = query_llama_server(
        "qwen3", prompt,
        n_predict=512,        # cap at 512 for speed — CoT reasoning is shorter
        temperature=0.6,
        top_p=0.95,
        stop=["<|endoftext|>", "<|im_end|>"],
    )
    if text is None:
        return "HOLD", None

    action, _reasoning, confidence = _parse_response(text)
    return action, confidence


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _directional_accuracy(predictions: list[str], labels: list[str]) -> dict:
    """Compute directional accuracy excluding HOLD predictions and flat labels.

    Denominator: rows where prediction in {BUY,SELL} AND label in {BUY,SELL}.
    Repo convention: matches accuracy_stats.py directional filter.
    """
    correct = total = buy_count = sell_count = hold_count = 0
    for pred, label in zip(predictions, labels):
        if pred == "HOLD":
            hold_count += 1
            continue
        if label == "HOLD":
            # flat outcome — excluded from denominator regardless of prediction
            hold_count += 1
            continue
        total += 1
        if pred == "BUY":
            buy_count += 1
        else:
            sell_count += 1
        if pred == label:
            correct += 1

    accuracy = correct / total if total > 0 else None
    hold_rate = hold_count / len(predictions) if predictions else None
    return {
        "accuracy":   accuracy,
        "correct":    correct,
        "scored_n":   total,
        "total_n":    len(predictions),
        "buy_count":  buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "hold_rate":  hold_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 FinCoT A/B benchmark at 3h horizon"
    )
    parser.add_argument("--tickers", default="BTC,ETH",
                        help="Comma-separated tickers (default: BTC,ETH)")
    parser.add_argument("--days", type=int, default=14,
                        help="Days of 1h candles (default: 14)")
    parser.add_argument("--horizon-bars", type=int, default=3,
                        help="Lookahead bars for labelling (default: 3 on 1h = 3h)")
    parser.add_argument("--threshold", type=float, default=0.002,
                        help="Flat-band threshold (default: 0.002 = 0.2%%)")
    parser.add_argument("--max-windows", type=int, default=40,
                        help="Cap labelled windows per ticker (default: 40). "
                             "Each window = 2 GPU calls; 40 windows ≈ 10-20 min.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: data/qwen3_fincot_ab.json)")
    args = parser.parse_args()

    tickers     = [t.strip().upper() for t in args.tickers.split(",")]
    output_path = (
        Path(args.output) if args.output
        else REPO_DIR / "data" / "qwen3_fincot_ab.json"
    )

    print("=" * 60)
    print("Qwen3 FinCoT A/B Benchmark")
    print(f"Tickers:      {', '.join(tickers)}")
    print(f"Days:         {args.days}")
    print(f"Horizon:      {args.horizon_bars} bars (1h candles = {args.horizon_bars}h)")
    print(f"Threshold:    {args.threshold*100:.1f}%")
    print(f"Max windows:  {args.max_windows} per ticker")
    print(f"Output:       {output_path}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Phase 1: Fetch candles + labels for all tickers
    # -----------------------------------------------------------------------
    all_contexts: dict[str, list[dict]] = {}
    all_labels:   dict[str, list[str]]  = {}

    for ticker in tickers:
        print(f"\n--- {ticker}: fetching candles ---")
        candles = download_candles(ticker, days=args.days)
        candles = compute_indicators(candles)
        candles = label_outcomes(candles, lookahead=args.horizon_bars,
                                 threshold=args.threshold)

        # Drop warm-up period (need 26 bars for MACD) and unlabelled tail
        valid = [c for c in candles[26:] if c["label"] is not None]
        print(f"  Valid labelled windows: {len(valid)}")

        label_dist: dict[str, int] = {}
        for c in valid:
            label_dist[c["label"]] = label_dist.get(c["label"], 0) + 1
        print(f"  Label distribution:     {label_dist}")

        # Cap to --max-windows, evenly spaced to cover the full time range
        if len(valid) > args.max_windows:
            step = len(valid) / args.max_windows
            idxs = [int(i * step) for i in range(args.max_windows)]
            valid = [valid[i] for i in idxs]
            print(f"  Capped to {len(valid)} windows (every ~{step:.0f} bars)")

        all_contexts[ticker] = [build_context(c, ticker) for c in valid]
        all_labels[ticker]   = [c["label"] for c in valid]

    # -----------------------------------------------------------------------
    # Phase 2: Inference — query qwen3 twice per window (A then B)
    # GPU lock is held per-call inside query_llama_server (llama_server.py
    # serialises via _thread_lock + file lock). We wrap the entire ticker
    # loop in gpu_gate so the live loop cannot interleave during the batch.
    # 2026-06-01: timeout=1800 (30 min) covers 40 windows × 2 calls × ~20s
    # -----------------------------------------------------------------------
    from portfolio.gpu_gate import gpu_gate

    variant_a_preds: dict[str, list[str]] = {t: [] for t in tickers}
    variant_b_preds: dict[str, list[str]] = {t: [] for t in tickers}

    total_windows = sum(len(all_contexts[t]) for t in tickers)
    done = 0
    t_start = time.time()

    print("\n--- Inference (A then B per window) ---")

    with gpu_gate("qwen3-fincot-ab", timeout=1800) as acquired:
        if not acquired:
            print("ERROR: GPU gate timed out — live loop may be holding GPU.")
            print("Reduce --max-windows or wait for the live loop to finish its LLM phase.")
            sys.exit(1)

        for ticker in tickers:
            contexts = all_contexts[ticker]
            n = len(contexts)
            print(f"\n  {ticker}: {n} windows")

            for i, ctx in enumerate(contexts):
                # Variant A
                prompt_a = _build_prompt_variant_a(ctx)
                action_a, _conf_a = _query_variant(prompt_a)
                variant_a_preds[ticker].append(action_a)

                # Variant B
                prompt_b = _build_prompt_variant_b(ctx)
                action_b, _conf_b = _query_variant(prompt_b)
                variant_b_preds[ticker].append(action_b)

                done += 1
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total_windows - done) / rate if rate > 0 else 0
                label = all_labels[ticker][i]
                print(
                    f"    [{done:>3}/{total_windows}] {ticker} w{i+1}: "
                    f"A={action_a:4s} B={action_b:4s} label={label:4s}  "
                    f"({rate:.2f}/s, ETA {eta:.0f}s)"
                )

    # -----------------------------------------------------------------------
    # Phase 3: Score
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    results: dict = {
        "timestamp":     datetime.now(UTC).isoformat(),
        "days":          args.days,
        "horizon_bars":  args.horizon_bars,
        "threshold":     args.threshold,
        "tickers":       tickers,
        "variants":      {},
        "verdict":       None,
        "verdict_detail": None,
    }

    all_a_stats: list[dict] = []
    all_b_stats: list[dict] = []

    for ticker in tickers:
        labels = all_labels[ticker]
        preds_a = variant_a_preds[ticker]
        preds_b = variant_b_preds[ticker]

        stats_a = _directional_accuracy(preds_a, labels)
        stats_b = _directional_accuracy(preds_b, labels)
        all_a_stats.append(stats_a)
        all_b_stats.append(stats_b)

        results["variants"].setdefault("A_current", {})[ticker]  = stats_a
        results["variants"].setdefault("B_fincot",  {})[ticker]  = stats_b

        acc_a_str = f"{stats_a['accuracy']*100:.1f}%" if stats_a["accuracy"] is not None else "N/A"
        acc_b_str = f"{stats_b['accuracy']*100:.1f}%" if stats_b["accuracy"] is not None else "N/A"
        delta_str = "N/A"
        if stats_a["accuracy"] is not None and stats_b["accuracy"] is not None:
            delta = stats_b["accuracy"] - stats_a["accuracy"]
            delta_str = f"{delta*100:+.1f}pp"

        print(f"\n  {ticker}")
        print(f"    Variant A (current): acc={acc_a_str}  "
              f"scored={stats_a['scored_n']}/{stats_a['total_n']}  "
              f"BUY={stats_a['buy_count']}  SELL={stats_a['sell_count']}  "
              f"HOLD-rate={stats_a['hold_rate']*100:.0f}%")
        print(f"    Variant B (FinCoT):  acc={acc_b_str}  "
              f"scored={stats_b['scored_n']}/{stats_b['total_n']}  "
              f"BUY={stats_b['buy_count']}  SELL={stats_b['sell_count']}  "
              f"HOLD-rate={stats_b['hold_rate']*100:.0f}%")
        print(f"    Delta B - A:         {delta_str}")

    # -----------------------------------------------------------------------
    # Aggregate across tickers (micro-average over all scored rows)
    # -----------------------------------------------------------------------
    total_correct_a = sum(s["correct"]   for s in all_a_stats)
    total_scored_a  = sum(s["scored_n"]  for s in all_a_stats)
    total_correct_b = sum(s["correct"]   for s in all_b_stats)
    total_scored_b  = sum(s["scored_n"]  for s in all_b_stats)
    total_hold_b    = sum(s["hold_count"] for s in all_b_stats)
    total_n_b       = sum(s["total_n"]   for s in all_b_stats)
    total_bs_b      = sum(s["buy_count"] + s["sell_count"] for s in all_b_stats)

    agg_acc_a = total_correct_a / total_scored_a if total_scored_a > 0 else None
    agg_acc_b = total_correct_b / total_scored_b if total_scored_b > 0 else None

    print(f"\n  AGGREGATE (micro-avg across {', '.join(tickers)})")
    agg_a_str = f"{agg_acc_a*100:.1f}%" if agg_acc_a is not None else "N/A"
    agg_b_str = f"{agg_acc_b*100:.1f}%" if agg_acc_b is not None else "N/A"

    if agg_acc_a is not None and agg_acc_b is not None:
        agg_delta = agg_acc_b - agg_acc_a
        delta_label = f"{agg_delta*100:+.1f}pp"
    else:
        agg_delta = None
        delta_label = "N/A"

    print(f"    A current:  {agg_a_str}  (n_scored={total_scored_a})")
    print(f"    B FinCoT:   {agg_b_str}  (n_scored={total_scored_b})")
    print(f"    Delta:      {delta_label}")

    results["aggregate"] = {
        "A_accuracy":  agg_acc_a,
        "B_accuracy":  agg_acc_b,
        "A_scored_n":  total_scored_a,
        "B_scored_n":  total_scored_b,
        "delta_pp":    agg_delta * 100 if agg_delta is not None else None,
    }

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    # Noise floor: ±1/sqrt(n) ≈ SE of a proportion. For n=40, SE ≈ 15.8pp.
    # We require delta > 5pp AND BUY+SELL recall must not collapse to near-zero
    # (< 5 directional predictions total across both tickers).
    # 2026-06-01: thresholds conservative because this is a prompt-change
    # decision, not a model swap.
    DELTA_THRESHOLD_PP   = 5.0   # minimum lift to consider shipping
    MIN_SCORED_N         = 10    # minimum directional predictions for significance

    verdict = "INCONCLUSIVE"
    reason  = ""

    if total_scored_b < MIN_SCORED_N:
        verdict = "INCONCLUSIVE"
        reason  = (
            f"Variant B produced only {total_scored_b} directional predictions "
            f"(need >= {MIN_SCORED_N}). Sample too small to draw conclusions."
        )
    elif agg_delta is None:
        verdict = "INCONCLUSIVE"
        reason  = "Could not compute accuracy for one or both variants (no scored windows)."
    elif agg_delta * 100 >= DELTA_THRESHOLD_PP:
        verdict = "SHIP"
        reason  = (
            f"FinCoT raises directional accuracy by {agg_delta*100:+.1f}pp "
            f"({agg_a_str} → {agg_b_str}) on {total_scored_b} scored windows. "
            f"BUY+SELL recall={total_bs_b} (above collapse floor). "
            f"Prompt change is directionally safe."
        )
    elif agg_delta * 100 <= -DELTA_THRESHOLD_PP:
        verdict = "DONT_SHIP"
        reason  = (
            f"FinCoT HURTS accuracy by {agg_delta*100:+.1f}pp "
            f"({agg_a_str} → {agg_b_str}). Do not change prompt."
        )
    else:
        # delta within noise floor
        sample_se = (1 / (total_scored_b ** 0.5)) * 100 if total_scored_b > 0 else 999
        verdict = "INCONCLUSIVE"
        reason  = (
            f"Delta {agg_delta*100:+.1f}pp is within noise floor "
            f"(approx ±{sample_se:.1f}pp at n={total_scored_b}). "
            f"Need larger sample or clearer signal to decide."
        )

    results["verdict"]        = verdict
    results["verdict_detail"] = reason

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict}")
    print(f"  {reason}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    from portfolio.file_utils import atomic_write_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(output_path), results)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
