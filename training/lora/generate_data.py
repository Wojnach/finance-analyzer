#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import pandas as pd

BUY_TEMPLATES = [
    "DECISION: BUY - RSI at {rsi:.1f} shows oversold conditions with bullish EMA crossover, suggesting upward momentum.",
    "DECISION: BUY - Price near lower Bollinger Band at extreme readings with RSI divergence indicating potential reversal.",
    "DECISION: BUY - Bearish exhaustion shown by RSI below {rsi:.1f} combined with MACD histogram turning positive at {macd_hist:.2f}.",
    "DECISION: BUY - Strong bullish divergence with EMA(9) crossing above EMA(21) and RSI recovering from {rsi:.1f}.",
    "DECISION: BUY - Price at ${price:,.2f} testing lower Bollinger Band with MACD histogram inflecting upward at {macd_hist:.2f}.",
    "DECISION: BUY - Momentum shift detected: MACD histogram at {macd_hist:.2f} turning positive while price holds support.",
    "DECISION: BUY - Oversold RSI at {rsi:.1f} with price compressing near lower Bollinger Band suggests accumulation phase.",
    "DECISION: BUY - Bullish EMA alignment confirmed with RSI at {rsi:.1f} bouncing off oversold territory and positive MACD crossover.",
    "DECISION: BUY - {ticker} showing capitulation signal with RSI at {rsi:.1f} and price below lower Bollinger Band, reversal likely.",
    "DECISION: BUY - Technical confluence: EMA bullish cross, MACD histogram flipping positive at {macd_hist:.2f}, RSI recovering from {rsi:.1f}.",
]

SELL_TEMPLATES = [
    "DECISION: SELL - RSI at {rsi:.1f} in overbought territory with bearish EMA crossover signaling distribution.",
    "DECISION: SELL - Price extended above upper Bollinger Band with negative MACD divergence at {macd_hist:.2f}.",
    "DECISION: SELL - Overbought RSI at {rsi:.1f} combined with MACD histogram declining to {macd_hist:.2f} indicates exhaustion.",
    "DECISION: SELL - Bearish EMA crossover with EMA(9) dropping below EMA(21) while RSI rolls over from {rsi:.1f}.",
    "DECISION: SELL - Price at ${price:,.2f} rejected at upper Bollinger Band with bearish momentum building.",
    "DECISION: SELL - Distribution pattern: RSI at {rsi:.1f} falling from overbought, MACD histogram negative at {macd_hist:.2f}.",
    "DECISION: SELL - {ticker} showing toppy price action above upper Bollinger Band with RSI divergence at {rsi:.1f}.",
    "DECISION: SELL - Momentum fading: MACD histogram at {macd_hist:.2f} declining sharply with bearish EMA alignment.",
    "DECISION: SELL - Bearish technical confluence with RSI at {rsi:.1f} rolling over and price losing EMA(21) support.",
    "DECISION: SELL - Overextended rally with RSI at {rsi:.1f} and price stretched above upper Bollinger Band, mean reversion expected.",
]

HOLD_TEMPLATES = [
    "DECISION: HOLD - Indicators show mixed signals with RSI neutral at {rsi:.1f} and no clear directional momentum.",
    "DECISION: HOLD - Price consolidating within Bollinger Bands, waiting for breakout confirmation.",
    "DECISION: HOLD - RSI at {rsi:.1f} in neutral territory with MACD histogram near zero at {macd_hist:.2f}, no edge.",
    "DECISION: HOLD - EMA signals conflicting with RSI and Bollinger Band readings, staying flat until clarity emerges.",
    "DECISION: HOLD - {ticker} range-bound at ${price:,.2f} with no decisive technical signal across indicators.",
    "DECISION: HOLD - MACD histogram at {macd_hist:.2f} shows indecision, RSI at {rsi:.1f} mid-range, no trade warranted.",
    "DECISION: HOLD - Price inside Bollinger Bands with flat EMAs, waiting for volatility expansion to trigger entry.",
    "DECISION: HOLD - Neutral RSI at {rsi:.1f} and tight Bollinger Band squeeze suggest imminent move but direction unclear.",
    "DECISION: HOLD - No conviction: EMA(9) and EMA(21) converging with RSI at {rsi:.1f} and muted MACD at {macd_hist:.2f}.",
    "DECISION: HOLD - Choppy price action with conflicting signals, preserving capital until trend establishes.",
]

PROMPT_TEMPLATE = """[INST]You are an expert cryptocurrency trader. Based on the following market data, provide a single trading decision: BUY, SELL, or HOLD.

Market Data:
- Asset: {ticker}
- Current Price: ${price:,.2f}
- 24h Change: {change_24h:+.2f}%

Technical Indicators (15-minute candles):
- RSI(14): {rsi:.1f}
- MACD Histogram: {macd_hist:.2f}
- EMA(9) vs EMA(21): {ema_direction}
- Bollinger Bands: Price is {bb_position}

Market Sentiment:
- Fear & Greed Index: N/A
- News Sentiment: N/A

Multi-timeframe Analysis:
N/A

Recent Headlines:
N/A

Respond with EXACTLY one of: BUY, SELL, or HOLD.
Then give a one-sentence reason.
Format: DECISION: [BUY/SELL/HOLD] - [reason][/INST]"""


def compute_indicators(df):
    close = df["close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()

    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    return df


def label_candles(df, threshold=0.02, lookahead=12):
    labels = []
    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(None)
            continue
        future_return = df["close"].iloc[i + lookahead] / df["close"].iloc[i] - 1
        if future_return > threshold:
            labels.append("BUY")
        elif future_return < -threshold:
            labels.append("SELL")
        else:
            labels.append("HOLD")
    df["label"] = labels
    return df


def build_prompt(row, ticker):
    change_24h = 0.0
    if pd.notna(row.get("change_24h")):
        change_24h = row["change_24h"]

    ema_direction = (
        "Bullish (9 > 21)" if row["ema9"] > row["ema21"] else "Bearish (9 < 21)"
    )

    if row["close"] < row["bb_lower"]:
        bb_position = "below lower band"
    elif row["close"] > row["bb_upper"]:
        bb_position = "above upper band"
    else:
        bb_position = "inside bands"

    return PROMPT_TEMPLATE.format(
        ticker=ticker,
        price=row["close"],
        change_24h=change_24h,
        rsi=row["rsi"],
        macd_hist=row["macd_hist"],
        ema_direction=ema_direction,
        bb_position=bb_position,
    )


def build_completion(row, ticker, label):
    templates = {"BUY": BUY_TEMPLATES, "SELL": SELL_TEMPLATES, "HOLD": HOLD_TEMPLATES}
    template = random.choice(templates[label])
    return template.format(
        rsi=row["rsi"],
        macd_hist=row["macd_hist"],
        price=row["close"],
        ticker=ticker,
    )


def process_asset(feather_path, ticker):
    print(f"Loading {feather_path.name}...")
    df = pd.read_feather(feather_path)

    if "close" not in df.columns:
        raise ValueError(
            f"No 'close' column in {feather_path.name}. Columns: {list(df.columns)}"
        )

    df = (
        df.sort_values("date").reset_index(drop=True)
        if "date" in df.columns
        else df.reset_index(drop=True)
    )

    df = compute_indicators(df)

    df["change_24h"] = (df["close"] / df["close"].shift(24) - 1) * 100

    df = label_candles(df)

    min_indicator_idx = 26
    valid = df.iloc[min_indicator_idx:].dropna(
        subset=[
            "rsi",
            "macd_hist",
            "ema9",
            "ema21",
            "bb_mid",
            "bb_upper",
            "bb_lower",
            "label",
        ]
    )
    print(f"  {ticker}: {len(valid)} valid candles out of {len(df)} total")

    examples = []
    for _, row in valid.iterrows():
        label = row["label"]
        prompt = build_prompt(row, ticker)
        completion = build_completion(row, ticker, label)
        examples.append(
            {
                "label": label,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            }
        )

    return examples


def balance_classes(examples, target_per_class=1000):
    by_class = {"BUY": [], "SELL": [], "HOLD": []}
    for ex in examples:
        by_class[ex["label"]].append(ex)

    print(f"\nClass distribution before balancing:")
    for label, items in by_class.items():
        print(f"  {label}: {len(items)}")

    min_count = min(len(v) for v in by_class.values())
    target = min(target_per_class, min_count)
    print(f"\nTarget per class: {target}")

    balanced = []
    for label, items in by_class.items():
        sampled = random.sample(items, target)
        balanced.extend(sampled)

    random.shuffle(balanced)

    for ex in balanced:
        del ex["label"]

    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feather-dir", required=True, help="Directory containing feather files"
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    random.seed(42)

    feather_dir = Path(args.feather_dir)
    output_path = Path(args.output)

    files = {
        "BTC": feather_dir / "BTC_USDT_USDT-1h-futures.feather",
        "ETH": feather_dir / "ETH_USDT_USDT-1h-futures.feather",
    }

    all_examples = []
    for ticker, fpath in files.items():
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping {ticker}")
            continue
        examples = process_asset(fpath, ticker)
        all_examples.extend(examples)
        print(f"  {ticker}: {len(examples)} examples generated")

    if not all_examples:
        print("ERROR: No examples generated. Check feather files.")
        return

    print(f"\nTotal examples before balancing: {len(all_examples)}")

    balanced = balance_classes(all_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in balanced:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(balanced)} examples to {output_path}")


if __name__ == "__main__":
    main()
