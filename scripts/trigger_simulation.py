#!/usr/bin/env python3
"""Trigger simulation — estimate how often "tradeable" moments occur
across 365 days of historical data, simulating multi-timeframe alignment
and sustained signal filters.

Answers: "How often would patient Claude Code approve a trade?"
"""

import pathlib
import pandas as pd
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
SYMBOLS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
WARMUP = 26


def load_klines(symbol):
    f = DATA_DIR / f"backtest_klines_{symbol}.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} — run signal_backtest.py first to download")
    return pd.read_csv(f)


def compute_indicators(df):
    c = df["close"]
    v = df["volume"]

    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["vol_avg20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_avg20"]
    df["price_change_3"] = c / c.shift(3) - 1
    return df


def vote_row(row):
    votes = []
    if row["rsi"] < 30:
        votes.append("BUY")
    elif row["rsi"] > 70:
        votes.append("SELL")

    if row["macd_hist"] > 0 and row["macd_hist_prev"] <= 0:
        votes.append("BUY")
    elif row["macd_hist"] < 0 and row["macd_hist_prev"] >= 0:
        votes.append("SELL")

    votes.append("BUY" if row["ema9"] > row["ema21"] else "SELL")

    if row["close"] <= row["bb_lower"]:
        votes.append("BUY")
    elif row["close"] >= row["bb_upper"]:
        votes.append("SELL")

    if row["vol_ratio"] > 1.5:
        if row["price_change_3"] > 0:
            votes.append("BUY")
        elif row["price_change_3"] < 0:
            votes.append("SELL")

    buys = votes.count("BUY")
    sells = votes.count("SELL")
    total = buys + sells
    if total < 2:
        return "HOLD", 0.0
    if buys > sells:
        return "BUY", buys / total
    if sells > buys:
        return "SELL", sells / total
    return "HOLD", 0.0


def resample_ohlcv(df, rule):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("dt")
    resampled = (
        df.resample(rule)
        .agg(
            {
                "open_time": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    return resampled.reset_index(drop=True)


def run_simulation():
    print("=" * 70)
    print("  TRIGGER SIMULATION — How often would Claude trade?")
    print("  365 days · 1h candles · multi-timeframe alignment")
    print("=" * 70)

    for symbol, label in SYMBOLS.items():
        df_1h = load_klines(symbol)

        # Build multi-timeframe views
        df_4h = resample_ohlcv(df_1h, "4h")
        df_1d = resample_ohlcv(df_1h, "24h")

        for d in [df_1h, df_4h, df_1d]:
            compute_indicators(d)

        # Compute signals per timeframe
        signals_1h = [vote_row(df_1h.iloc[i]) for i in range(len(df_1h))]
        signals_4h = [vote_row(df_4h.iloc[i]) for i in range(len(df_4h))]
        signals_1d = [vote_row(df_1d.iloc[i]) for i in range(len(df_1d))]

        df_1h["action"] = [s[0] for s in signals_1h]
        df_1h["conf"] = [s[1] for s in signals_1h]
        df_4h["action"] = [s[0] for s in signals_4h]
        df_1d["action"] = [s[0] for s in signals_1d]

        # Forward returns for outcome analysis
        df_1h["ret_1d"] = df_1h["close"].shift(-24) / df_1h["close"] - 1
        df_1h["ret_3d"] = df_1h["close"].shift(-72) / df_1h["close"] - 1

        # Map longer TF signals back to 1h index
        # For each 1h candle, find corresponding 4h and 1d signal
        df_1h["action_4h"] = "HOLD"
        df_1h["action_1d"] = "HOLD"

        for i, row in df_1h.iterrows():
            t = row["open_time"]
            # Find latest 4h candle <= this time
            mask = df_4h["open_time"] <= t
            if mask.any():
                df_1h.at[i, "action_4h"] = df_4h.loc[mask.values].iloc[-1]["action"]
            mask = df_1d["open_time"] <= t
            if mask.any():
                df_1h.at[i, "action_1d"] = df_1d.loc[mask.values].iloc[-1]["action"]

        # Skip warmup
        df = df_1h.iloc[WARMUP:].copy().reset_index(drop=True)

        # ── Analysis ─────────────────────────────────────────────

        total_candles = len(df)
        total_days = total_candles / 24

        # Count signal flips (trigger condition 1 — current behavior)
        flips = (df["action"] != df["action"].shift(1)).sum()

        # Sustained signals: action same as previous candle
        df["sustained"] = df["action"] == df["action"].shift(1)
        df["sustained_action"] = df["action"].where(df["sustained"])

        # Multi-timeframe aligned
        df["tf_aligned_2"] = (df["action"] == df["action_4h"]) & (
            df["action"] != "HOLD"
        )
        df["tf_aligned_3"] = (
            (df["action"] == df["action_4h"])
            & (df["action"] == df["action_1d"])
            & (df["action"] != "HOLD")
        )

        # "Claude would trade" = sustained + multi-TF aligned (2 of 3 TFs agree)
        df["tradeable"] = (
            df["sustained"] & df["tf_aligned_2"] & (df["action"] != "HOLD")
        )
        df["tradeable_strong"] = (
            df["sustained"] & df["tf_aligned_3"] & (df["action"] != "HOLD")
        )

        # Don't double-count consecutive tradeable candles — only count new entries
        df["tradeable_entry"] = df["tradeable"] & (
            ~df["tradeable"].shift(1, fill_value=False)
        )
        df["tradeable_strong_entry"] = df["tradeable_strong"] & (
            ~df["tradeable_strong"].shift(1, fill_value=False)
        )

        # ── Outcome analysis for tradeable moments ──────────────

        trades_2tf = df[df["tradeable_entry"]]
        trades_3tf = df[df["tradeable_strong_entry"]]

        def outcome_stats(subset, label_ret):
            if subset.empty:
                return 0, 0.0, 0.0
            ret = subset[label_ret].dropna()
            if ret.empty:
                return len(subset), 0.0, 0.0
            buys = subset[subset["action"] == "BUY"]
            sells = subset[subset["action"] == "SELL"]
            buy_ret = buys[label_ret].dropna()
            sell_ret = sells[label_ret].dropna()
            # For BUY: positive return = correct. For SELL: negative return = correct.
            buy_hits = (buy_ret > 0).sum() if len(buy_ret) else 0
            sell_hits = (sell_ret < 0).sum() if len(sell_ret) else 0
            total_hits = buy_hits + sell_hits
            total_n = len(buy_ret) + len(sell_ret)
            hit_pct = total_hits / total_n * 100 if total_n else 0
            avg_ret = (
                (buy_ret.mean() * len(buy_ret) - sell_ret.mean() * len(sell_ret))
                / total_n
                * 100
                if total_n
                else 0
            )
            return total_n, hit_pct, avg_ret

        # ── Print report ─────────────────────────────────────────

        print(f"\n{'─' * 70}")
        print(f"  {label}-USD  ({total_candles} candles = {total_days:.0f} days)")
        print(f"{'─' * 70}")

        print(f"\n  Raw signal distribution:")
        for act in ["BUY", "SELL", "HOLD"]:
            n = (df["action"] == act).sum()
            pct = n / total_candles * 100
            print(f"    {act:<6} {n:>6} candles ({pct:.1f}%)")

        print(f"\n  Trigger frequency (current system — every signal flip):")
        print(f"    Signal flips:       {flips:>6} ({flips / total_days:.1f}/day)")

        sustained_buys = ((df["sustained_action"] == "BUY")).sum()
        sustained_sells = ((df["sustained_action"] == "SELL")).sum()
        sustained_total = sustained_buys + sustained_sells
        print(f"\n  After sustained filter (2 consecutive same direction):")
        print(f"    Sustained BUY:      {sustained_buys:>6} candles")
        print(f"    Sustained SELL:     {sustained_sells:>6} candles")

        n_2tf = len(trades_2tf)
        n_3tf = len(trades_3tf)
        print(f"\n  Tradeable moments (sustained + multi-TF aligned):")
        print(
            f"    2-TF aligned (Now+4h):     {n_2tf:>5} entries ({n_2tf / total_days:.1f}/day)"
        )
        print(
            f"    3-TF aligned (Now+4h+1d):  {n_3tf:>5} entries ({n_3tf / total_days:.1f}/day)"
        )

        n2_buy = len(trades_2tf[trades_2tf["action"] == "BUY"])
        n2_sell = len(trades_2tf[trades_2tf["action"] == "SELL"])
        n3_buy = len(trades_3tf[trades_3tf["action"] == "BUY"])
        n3_sell = len(trades_3tf[trades_3tf["action"] == "SELL"])
        print(f"\n    2-TF: {n2_buy} BUY / {n2_sell} SELL entries")
        print(f"    3-TF: {n3_buy} BUY / {n3_sell} SELL entries")

        # Outcome analysis
        print(f"\n  Forward returns at tradeable moments:")
        print(f"    {'Scenario':<28} {'Count':>5} {'1d Hit%':>8} {'3d Hit%':>8}")
        print(f"    {'─' * 52}")

        for name, subset in [
            ("2-TF aligned", trades_2tf),
            ("3-TF aligned", trades_3tf),
        ]:
            n1, h1, _ = outcome_stats(subset, "ret_1d")
            n3, h3, _ = outcome_stats(subset, "ret_3d")
            print(f"    {name:<28} {n1:>5} {h1:>7.1f}% {h3:>7.1f}%")

        # BUY-only outcomes (most relevant for the user)
        for name, subset in [
            ("2-TF BUY only", trades_2tf[trades_2tf["action"] == "BUY"]),
            ("3-TF BUY only", trades_3tf[trades_3tf["action"] == "BUY"]),
        ]:
            ret1 = subset["ret_1d"].dropna()
            ret3 = subset["ret_3d"].dropna()
            h1 = (ret1 > 0).mean() * 100 if len(ret1) else 0
            h3 = (ret3 > 0).mean() * 100 if len(ret3) else 0
            avg1 = ret1.mean() * 100 if len(ret1) else 0
            avg3 = ret3.mean() * 100 if len(ret3) else 0
            print(f"    {name:<28} {len(ret1):>5} {h1:>7.1f}% {h3:>7.1f}%")
            print(f"      avg return:              {'':>5} {avg1:>7.2f}% {avg3:>7.2f}%")

        # Invocation estimate comparison
        print(f"\n  Estimated invocations (extrapolated to weekly):")
        flips_per_week = flips / total_days * 7
        tradeable_per_week = n_2tf / total_days * 7
        strong_per_week = n_3tf / total_days * 7
        print(f"    Current (every flip):     {flips_per_week:>7.0f}/week")
        print(f"    2-TF sustained:           {tradeable_per_week:>7.0f}/week")
        print(f"    3-TF sustained:           {strong_per_week:>7.0f}/week")

    print(f"\n{'=' * 70}")
    print("  Key: 'Hit%' = BUY correct if price went up, SELL correct if down")
    print("  '2-TF' = Now + 4h agree.  '3-TF' = Now + 4h + 1d all agree.")
    print("  'Sustained' = signal held for 2 consecutive checks (1h candles).")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation()
