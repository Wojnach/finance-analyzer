#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import pandas as pd

FG_CLASSES = [
    (0, 20, "Extreme Fear"),
    (21, 40, "Fear"),
    (41, 60, "Neutral"),
    (61, 80, "Greed"),
    (81, 100, "Extreme Greed"),
]

SENTIMENTS = ["positive", "negative", "neutral"]

PROMPT_TEMPLATE = """You are an expert cryptocurrency trader. Based on the following market data, provide a single trading decision: BUY, SELL, or HOLD.

Market Data:
- Asset: {ticker}
- Current Price: ${price:,.2f}
- 24h Change: {change_24h:+.2f}%

Technical Indicators (1-hour candles):
- RSI(14): {rsi:.1f}
- MACD Histogram: {macd_hist:.2f}
- EMA(9) vs EMA(21): {ema_direction} (gap: {ema_gap_pct:.1f}%)
- Bollinger Bands: Price is {bb_position}

Market Sentiment:
- Fear & Greed Index: {fear_greed}/100 ({fear_greed_class})
- News Sentiment: {news_sentiment} (confidence: {sentiment_confidence:.2f})

Multi-timeframe Analysis:
{timeframe_summary}

Recent Headlines:
{headlines}

Respond with EXACTLY one of: BUY, SELL, or HOLD.
Then give a one-sentence reason.
Format: DECISION: [BUY/SELL/HOLD] - [reason]"""


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


def _sample_fear_greed():
    value = random.randint(5, 95)
    for lo, hi, cls in FG_CLASSES:
        if lo <= value <= hi:
            return value, cls
    return value, "Neutral"


def _sample_sentiment():
    sent = random.choice(SENTIMENTS)
    conf = round(random.uniform(0.25, 0.92), 2)
    return sent, conf


def _build_timeframe_summary(df, idx):
    horizons = [("12h", 12), ("2d", 48), ("7d", 168)]
    parts = []
    for label, offset in horizons:
        ref_idx = idx - offset
        if ref_idx < 26:
            continue
        ref_rsi = df["rsi"].iloc[ref_idx]
        if pd.isna(ref_rsi):
            continue
        if ref_rsi < 30:
            action = "BUY"
        elif ref_rsi > 70:
            action = "SELL"
        else:
            action = "HOLD"
        parts.append(f"{label}: {action} (RSI={ref_rsi:.0f})")
    return " | ".join(parts) if parts else "N/A"


def build_prompt(row, ticker, df_idx, df):
    change_24h = 0.0
    if pd.notna(row.get("change_24h")):
        change_24h = row["change_24h"]

    ema_direction = (
        "Bullish (9 > 21)" if row["ema9"] > row["ema21"] else "Bearish (9 < 21)"
    )
    ema_gap_pct = (
        abs(row["ema9"] - row["ema21"]) / row["ema21"] * 100 if row["ema21"] != 0 else 0
    )

    if row["close"] < row["bb_lower"]:
        bb_position = "below lower band"
    elif row["close"] > row["bb_upper"]:
        bb_position = "above upper band"
    else:
        bb_position = "inside bands"

    fg_value, fg_class = _sample_fear_greed()
    sentiment, sent_conf = _sample_sentiment()
    tf_summary = _build_timeframe_summary(df, df_idx)

    return PROMPT_TEMPLATE.format(
        ticker=ticker,
        price=row["close"],
        change_24h=change_24h,
        rsi=row["rsi"],
        macd_hist=row["macd_hist"],
        ema_direction=ema_direction,
        ema_gap_pct=ema_gap_pct,
        bb_position=bb_position,
        fear_greed=fg_value,
        fear_greed_class=fg_class,
        news_sentiment=sentiment,
        sentiment_confidence=sent_conf,
        timeframe_summary=tf_summary,
        headlines="N/A",
    )


def build_completion(row, ticker, label):
    rsi = row["rsi"]
    macd = row["macd_hist"]
    ema_bull = row["ema9"] > row["ema21"]

    reasons = []
    if label == "BUY":
        if rsi < 40:
            reasons.append(f"RSI at {rsi:.1f} approaching oversold territory")
        if macd > 0:
            reasons.append(f"MACD histogram positive at {macd:.2f} showing momentum")
        if ema_bull:
            reasons.append("bullish EMA(9) > EMA(21) alignment")
        if row["close"] < row["bb_lower"]:
            reasons.append("price near lower Bollinger Band suggesting accumulation")
        if not reasons:
            reasons.append(
                f"emerging bullish momentum with MACD at {macd:.2f} and RSI at {rsi:.1f}"
            )
    elif label == "SELL":
        if rsi > 60:
            reasons.append(f"RSI at {rsi:.1f} approaching overbought territory")
        if macd < 0:
            reasons.append(
                f"MACD histogram negative at {macd:.2f} showing bearish momentum"
            )
        if not ema_bull:
            reasons.append("bearish EMA(9) < EMA(21) alignment")
        if row["close"] > row["bb_upper"]:
            reasons.append("price above upper Bollinger Band suggesting overextension")
        if not reasons:
            reasons.append(
                f"bearish pressure building with MACD at {macd:.2f} and RSI at {rsi:.1f}"
            )
    else:
        reasons.append(f"RSI at {rsi:.1f} in neutral territory")
        if abs(macd) < 1:
            reasons.append(f"MACD histogram near zero at {macd:.2f}")
        reasons.append("no clear directional edge")

    reason_text = ", ".join(reasons[:3])
    return f"DECISION: {label} - {reason_text.capitalize()}."


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
    for idx, row in valid.iterrows():
        label = row["label"]
        prompt = build_prompt(row, ticker, idx, df)
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
