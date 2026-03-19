"""
Fetch 2026 YTD historical metals data from Binance FAPI.
Generates data/metals_history.json with:
- Daily OHLCV for XAG-USD and XAU-USD
- Key statistics: YTD high/low, ATR, max daily moves, trends
- Context for Layer 2 Claude to understand price behavior

Run: .venv/Scripts/python.exe data/metals_history_fetch.py
"""
import datetime
import json
import os
import time

import requests

os.chdir(r"Q:/finance-analyzer")

FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = {
    "XAG-USD": "XAGUSDT",
    "XAU-USD": "XAUUSDT",
}

# 2026 YTD start
YTD_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

def fetch_fapi_klines(symbol, interval, start_ts, end_ts=None, limit=1000):
    """Fetch klines from Binance FAPI with pagination."""
    all_klines = []
    current_start = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000) if end_ts else int(time.time() * 1000)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        try:
            resp = requests.get(FAPI_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching {symbol} {interval}: {e}")
            break

        if not data:
            break

        all_klines.extend(data)
        # Move start past last kline's close time
        last_close = data[-1][6]  # close_time ms
        current_start = last_close + 1

        if len(data) < limit:
            break
        time.sleep(0.3)  # rate limit

    return all_klines

def klines_to_daily(klines):
    """Convert raw klines to daily OHLCV dicts."""
    days = []
    for k in klines:
        ts = datetime.datetime.fromtimestamp(k[0] / 1000, tz=datetime.UTC)
        days.append({
            "date": ts.strftime("%Y-%m-%d"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return days

def compute_stats(days):
    """Compute key statistics from daily data."""
    if not days:
        return {}

    closes = [d["close"] for d in days]
    highs = [d["high"] for d in days]
    lows = [d["low"] for d in days]

    # True range for ATR
    true_ranges = []
    for i in range(1, len(days)):
        h = days[i]["high"]
        l = days[i]["low"]
        pc = days[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        true_ranges.append(tr)

    atr_14 = sum(true_ranges[-14:]) / min(14, len(true_ranges)) if true_ranges else 0

    # Daily changes
    daily_changes = []
    for i in range(1, len(days)):
        pct = ((closes[i] - closes[i-1]) / closes[i-1]) * 100
        daily_changes.append({
            "date": days[i]["date"],
            "change_pct": round(pct, 2),
            "close": closes[i],
        })

    # Biggest up/down days
    if daily_changes:
        biggest_up = max(daily_changes, key=lambda x: x["change_pct"])
        biggest_down = min(daily_changes, key=lambda x: x["change_pct"])
    else:
        biggest_up = biggest_down = {"date": "N/A", "change_pct": 0}

    # Streaks (consecutive up/down days)
    max_up_streak = max_down_streak = 0
    current_streak = 0
    for dc in daily_changes:
        if dc["change_pct"] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_up_streak = max(max_up_streak, current_streak)
        elif dc["change_pct"] < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_down_streak = max(max_down_streak, abs(current_streak))
        else:
            current_streak = 0

    # Recent trend (last 5 days)
    recent_5d = closes[-5:] if len(closes) >= 5 else closes
    recent_trend_pct = ((recent_5d[-1] - recent_5d[0]) / recent_5d[0]) * 100 if len(recent_5d) >= 2 else 0

    # YTD change
    ytd_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 2 else 0

    # Weekly ranges (for intraday context)
    avg_daily_range_pct = 0
    if days:
        daily_ranges = [((d["high"] - d["low"]) / d["low"]) * 100 for d in days if d["low"] > 0]
        avg_daily_range_pct = sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0

    return {
        "ytd_start": closes[0] if closes else 0,
        "ytd_current": closes[-1] if closes else 0,
        "ytd_change_pct": round(ytd_change_pct, 2),
        "ytd_high": round(max(highs), 2),
        "ytd_high_date": days[highs.index(max(highs))]["date"],
        "ytd_low": round(min(lows), 2),
        "ytd_low_date": days[lows.index(min(lows))]["date"],
        "atr_14d": round(atr_14, 2),
        "atr_14d_pct": round((atr_14 / closes[-1]) * 100, 2) if closes else 0,
        "avg_daily_range_pct": round(avg_daily_range_pct, 2),
        "biggest_up_day": biggest_up,
        "biggest_down_day": biggest_down,
        "max_up_streak_days": max_up_streak,
        "max_down_streak_days": max_down_streak,
        "recent_5d_trend_pct": round(recent_trend_pct, 2),
        "total_trading_days": len(days),
        "up_days": sum(1 for dc in daily_changes if dc["change_pct"] > 0),
        "down_days": sum(1 for dc in daily_changes if dc["change_pct"] < 0),
        "flat_days": sum(1 for dc in daily_changes if dc["change_pct"] == 0),
    }

def main():
    now = datetime.datetime.now(datetime.UTC)
    print(f"Fetching 2026 YTD metals data (Jan 1 - {now.strftime('%b %d')})...")
    print()

    result = {
        "generated": now.isoformat(),
        "period": f"2026-01-01 to {now.strftime('%Y-%m-%d')}",
        "metals": {},
    }

    for ticker, symbol in SYMBOLS.items():
        print(f"  Fetching {ticker} ({symbol}) daily klines...")
        klines = fetch_fapi_klines(symbol, "1d", YTD_START, now)
        days = klines_to_daily(klines)
        stats = compute_stats(days)

        print(f"    {len(days)} days | YTD: {stats.get('ytd_change_pct', 0):+.1f}% | "
              f"Range: ${stats.get('ytd_low', 0):.2f}-${stats.get('ytd_high', 0):.2f} | "
              f"ATR: ${stats.get('atr_14d', 0):.2f} ({stats.get('atr_14d_pct', 0):.1f}%)")

        # Also fetch 4h data for more granular recent context
        print(f"  Fetching {ticker} ({symbol}) 4h klines (last 30 days)...")
        thirty_days_ago = now - datetime.timedelta(days=30)
        klines_4h = fetch_fapi_klines(symbol, "4h", thirty_days_ago, now)
        days_4h = klines_to_daily(klines_4h)  # reuse the converter, it just extracts OHLCV

        result["metals"][ticker] = {
            "symbol": symbol,
            "stats": stats,
            "daily_ohlcv": days,  # full YTD daily
            "recent_4h": days_4h[-60:],  # last 60 4h candles (~10 days)
        }

    # Write to file
    outfile = "data/metals_history.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outfile}")

    # Print summary
    print(f"\n{'='*60}")
    print("  METALS 2026 YTD SUMMARY")
    print(f"{'='*60}")
    for ticker, data in result["metals"].items():
        s = data["stats"]
        print(f"\n  {ticker}:")
        print(f"    YTD: ${s['ytd_start']:.2f} -> ${s['ytd_current']:.2f} ({s['ytd_change_pct']:+.1f}%)")
        print(f"    Range: ${s['ytd_low']:.2f} (low {s['ytd_low_date']}) - ${s['ytd_high']:.2f} (high {s['ytd_high_date']})")
        print(f"    ATR(14): ${s['atr_14d']:.2f} ({s['atr_14d_pct']:.1f}% of price)")
        print(f"    Avg daily range: {s['avg_daily_range_pct']:.1f}%")
        print(f"    Biggest up: {s['biggest_up_day']['change_pct']:+.1f}% ({s['biggest_up_day']['date']})")
        print(f"    Biggest down: {s['biggest_down_day']['change_pct']:+.1f}% ({s['biggest_down_day']['date']})")
        print(f"    Up/Down days: {s['up_days']}/{s['down_days']} ({s['total_trading_days']} total)")
        print(f"    Max streak: {s['max_up_streak_days']} up, {s['max_down_streak_days']} down")
        print(f"    Recent 5d: {s['recent_5d_trend_pct']:+.1f}%")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
