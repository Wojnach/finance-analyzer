#!/usr/bin/env python3
"""Signal accuracy backtest — retroactively measure how well each signal
predicts price movement at 1d/3d/5d/10d horizons using 365 days of 1h data."""

import os
import time
import pathlib
import pandas as pd
import requests

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
SYMBOLS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
HOURS_PER_DAY = 24
HORIZONS = {"1d": 24, "3d": 72, "5d": 120, "10d": 240}
WARMUP = 26  # candles to skip for indicator convergence

# ── Data download ────────────────────────────────────────────────────


def download_klines(symbol, days=365):
    cache = DATA_DIR / f"backtest_klines_{symbol}.csv"
    if cache.exists():
        print(f"  cached: {cache}")
        return pd.read_csv(cache)

    print(f"  downloading {symbol} klines ({days}d)...")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        r = requests.get(
            f"{BINANCE_FAPI}/klines",
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for k in batch:
            all_rows.append(
                {
                    "open_time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        cursor = int(batch[-1][0]) + 1
        time.sleep(0.2)

    df = (
        pd.DataFrame(all_rows)
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  saved {len(df)} candles → {cache}")
    return df


def download_funding_rates(symbol, days=365):
    cache = DATA_DIR / f"backtest_funding_{symbol}.csv"
    if cache.exists():
        print(f"  cached: {cache}")
        return pd.read_csv(cache)

    print(f"  downloading {symbol} funding rates ({days}d)...")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        r = requests.get(
            f"{BINANCE_FAPI}/fundingRate",
            params={
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for entry in batch:
            all_rows.append(
                {
                    "time": int(entry["fundingTime"]),
                    "rate": float(entry["fundingRate"]),
                }
            )
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.2)

    df = (
        pd.DataFrame(all_rows)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  saved {len(df)} entries → {cache}")
    return df


# ── Indicator computation (vectorized, matches main.py exactly) ──────


def compute_all_indicators(df):
    close = df["close"]
    vol = df["volume"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # EMA(9, 21)
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()

    # Bollinger Bands(20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    # Volume: 20-period avg and 3-candle price change
    df["vol_avg20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_avg20"]
    df["price_change_3"] = close / close.shift(3) - 1

    return df


# ── Signal vote functions ────────────────────────────────────────────


def vote_rsi(row):
    if row["rsi"] < 30:
        return "BUY"
    elif row["rsi"] > 70:
        return "SELL"
    return None


def vote_macd(row):
    if row["macd_hist"] > 0 and row["macd_hist_prev"] <= 0:
        return "BUY"
    elif row["macd_hist"] < 0 and row["macd_hist_prev"] >= 0:
        return "SELL"
    return None


def vote_ema(row):
    return "BUY" if row["ema9"] > row["ema21"] else "SELL"


def vote_bb(row):
    if row["close"] <= row["bb_lower"]:
        return "BUY"
    elif row["close"] >= row["bb_upper"]:
        return "SELL"
    return None


def vote_volume(row):
    if row["vol_ratio"] > 1.5:
        if row["price_change_3"] > 0:
            return "BUY"
        elif row["price_change_3"] < 0:
            return "SELL"
    return None


SIGNALS = {
    "RSI(14)": vote_rsi,
    "MACD(12,26,9)": vote_macd,
    "EMA(9,21)": vote_ema,
    "BB(20,2)": vote_bb,
    "Volume": vote_volume,
}


# ── Funding rate signal (separate timeline, merge to nearest candle) ─


def merge_funding_votes(klines_df, funding_df):
    """Map each funding rate observation to the nearest hourly candle and
    produce a vote column.  Candles without a funding event get None."""
    votes = pd.Series([None] * len(klines_df), index=klines_df.index)
    if funding_df.empty:
        return votes

    for _, fr in funding_df.iterrows():
        rate = fr["rate"]
        idx = (klines_df["open_time"] - fr["time"]).abs().idxmin()
        if rate > 0.0003:
            votes.iloc[idx] = "SELL"
        elif rate < -0.0001:
            votes.iloc[idx] = "BUY"
    return votes


# ── Outcome + scoring ────────────────────────────────────────────────


def compute_outcomes(df):
    for label, offset in HORIZONS.items():
        df[f"future_{label}"] = df["close"].shift(-offset)
        df[f"ret_{label}"] = df[f"future_{label}"] / df["close"] - 1


def score_signal(votes, returns):
    """Given a Series of votes (BUY/SELL/None) and a Series of returns,
    compute hit rate: BUY correct when ret>0, SELL correct when ret<0."""
    mask = votes.notna() & returns.notna()
    v = votes[mask]
    r = returns[mask]
    if len(v) == 0:
        return 0, 0.0
    hits = ((v == "BUY") & (r > 0)) | ((v == "SELL") & (r < 0))
    return len(v), float(hits.sum()) / len(v) * 100


# ── Main ─────────────────────────────────────────────────────────────


def run_backtest():
    print("=== Signal Accuracy Backtest (365 days, 1h candles) ===\n")
    print("Downloading data...")

    for symbol, label in SYMBOLS.items():
        klines = download_klines(symbol)
        funding = download_funding_rates(symbol)

        print(f"\nComputing indicators for {label}...")
        df = compute_all_indicators(klines.copy())
        compute_outcomes(df)

        # Drop warmup rows
        df = df.iloc[WARMUP:].reset_index(drop=True)

        # Generate votes for each signal
        all_votes = {}
        for name, fn in SIGNALS.items():
            all_votes[name] = df.apply(fn, axis=1)

        # Funding rate (separate merge)
        all_votes["Funding Rate"] = merge_funding_votes(df, funding)

        # Print report
        print(f"\n{'─' * 65}")
        print(f"  {label}-USD ({len(df)} candles)")
        print(f"{'─' * 65}")
        header = f"  {'Signal':<18} {'Votes':>6}"
        for h in HORIZONS:
            header += f" {h+' Hit%':>9}"
        print(header)
        print(f"  {'─' * 60}")

        for name in list(SIGNALS.keys()) + ["Funding Rate"]:
            votes = all_votes[name]
            parts = [f"  {name:<18}"]
            total_votes = int(votes.notna().sum())
            parts.append(f"{total_votes:>6}")
            for h_label in HORIZONS:
                ret_col = f"ret_{h_label}"
                n, pct = score_signal(votes, df[ret_col])
                parts.append(f"{pct:>8.1f}%" if n > 0 else f"{'N/A':>9}")
            print("".join(parts))

        # Bonus: vote direction breakdown
        print(f"\n  Vote breakdown:")
        for name in list(SIGNALS.keys()) + ["Funding Rate"]:
            votes = all_votes[name]
            buys = int((votes == "BUY").sum())
            sells = int((votes == "SELL").sum())
            print(f"    {name:<18} BUY={buys:<5} SELL={sells}")

    print(f"\n{'═' * 65}")
    print("Done.")


if __name__ == "__main__":
    run_backtest()
