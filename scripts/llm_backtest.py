"""Historical LLM backtest — replay indicator snapshots through local models.

Rebuilds the trading-prompt context at historical timestamps from Binance
1h klines, queries each llama_server model on identical prompts (temp 0),
and scores directional accuracy against the known +1d outcome.

Usage (on the GPU host):
    python scripts/llm_backtest.py --models ministral3,qwen3,phi4_mini,fin_r1 \
        --start 2026-02-01 --end 2026-07-11 --step-hours 8 --out data/llm_backtest_results.jsonl
    python scripts/llm_backtest.py --score data/llm_backtest_results.jsonl

Context is indicator-only (RSI/MACD/EMA/BB/volume + historical Fear&Greed);
news sentiment is fixed to "neutral" because it cannot be reconstructed.
Scores are comparable ACROSS models (same inputs), not with live accuracy
stats (different context mix).
"""

import argparse
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from portfolio.signal_utils import ema, rsi  # noqa: E402

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
FNG_HISTORY = "https://api.alternative.me/fng/?limit=0"
TICKERS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
FAPI_TICKERS = {"XAU-USD": "XAUUSDT", "XAG-USD": "XAGUSDT"}
HORIZONS_H = (1, 3, 24)
# Sweep support: horizons (hours) matched to candle interval, and how many
# candles ~ 24h for the change_24h context line.
INTERVAL_HORIZONS = {
    "15m": (0.25, 1, 3),
    "1h": (1, 3, 24),
    "4h": (4, 12, 72),
    "1d": (24, 72, 168),
}
INTERVAL_STEP_HOURS = {"15m": 2, "1h": 8, "4h": 24, "1d": 48}


def fetch_klines_1h(
    symbol: str,
    start_ms: int,
    end_ms: int,
    base_url: str = BINANCE_KLINES,
    interval: str = "1h",
) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while cur < end_ms:
        for attempt in range(3):
            try:
                r = requests.get(
                    base_url,
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": cur,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                    timeout=15,
                )
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][6] + 1
        time.sleep(0.15)
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qv",
            "n",
            "tbb",
            "tbq",
            "ig",
        ],
    )
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("ts")


def fetch_fng() -> dict:
    data = requests.get(FNG_HISTORY, timeout=15).json()["data"]
    return {
        dt.datetime.fromtimestamp(int(d["timestamp"]), dt.UTC).date(): int(d["value"])
        for d in data
    }


def build_context(
    df: pd.DataFrame,
    at: pd.Timestamp,
    ticker: str,
    fng: dict,
    interval: str = "1h",
) -> dict | None:
    hist = df[df.index <= at]
    if len(hist) < 120:
        return None
    close = hist["close"]
    price = float(close.iloc[-1])
    rsi_v = float(rsi(close, 14).iloc[-1])
    macd_line = ema(close, 12) - ema(close, 26)
    macd_hist = float((macd_line - ema(macd_line, 9)).iloc[-1])
    e9, e21 = float(ema(close, 9).iloc[-1]), float(ema(close, 21).iloc[-1])
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = float(sma20.iloc[-1] + 2 * std20.iloc[-1])
    lower = float(sma20.iloc[-1] - 2 * std20.iloc[-1])
    if price > upper:
        bb_pos = "above upper band"
    elif price < lower:
        bb_pos = "below lower band"
    else:
        bb_pos = "between bands, near middle"
    vol_ratio = float(
        hist["volume"].iloc[-1] / hist["volume"].rolling(20).mean().iloc[-1]
    )
    day_ago = close[close.index <= at - pd.Timedelta(hours=24)]
    chg24 = (price / float(day_ago.iloc[-1]) - 1) * 100 if len(day_ago) else 0.0
    return {
        "ticker": ticker,
        "timeframe": f"{interval} candles",
        "price_usd": price,
        "change_24h": f"{chg24:+.2f}%",
        "rsi": round(rsi_v, 1),
        "macd_hist": round(macd_hist, 2),
        "ema_bullish": e9 > e21,
        "ema_gap_pct": round((e9 / e21 - 1) * 100, 2),
        "bb_position": bb_pos,
        "volume_ratio": round(vol_ratio, 2),
        "fear_greed": fng.get(at.date(), 50),
        "sentiment": "neutral",
    }


def _interval_hours(interval: str) -> float:
    return {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}[interval]


def outcome_at(df: pd.DataFrame, at: pd.Timestamp, hours: int) -> float | None:
    fut = df[df.index >= at + pd.Timedelta(hours=hours)]
    hist = df[df.index <= at]
    if fut.empty or hist.empty:
        return None
    return (float(fut["close"].iloc[0]) / float(hist["close"].iloc[-1]) - 1) * 100


def run(args):
    from portfolio.llama_server import query_llama_server, stop_server
    from portfolio.qwen3_trader import _build_prompt, _parse_response

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    fng = fetch_fng()
    wanted = [t.strip() for t in args.tickers.split(",")]
    frames = {}
    for tick in wanted:
        if tick in TICKERS:
            sym, url = TICKERS[tick], BINANCE_KLINES
        elif tick in FAPI_TICKERS:
            sym, url = FAPI_TICKERS[tick], BINANCE_FAPI_KLINES
        else:
            raise SystemExit(f"unknown ticker {tick}")
        pad = start - pd.Timedelta(hours=200 * _interval_hours(args.interval))
        frames[tick] = fetch_klines_1h(
            sym,
            int(pad.timestamp() * 1000),
            int((end + pd.Timedelta(days=8)).timestamp() * 1000),
            base_url=url,
            interval=args.interval,
        )
        print(f"{tick}: {len(frames[tick])} {args.interval} candles", flush=True)

    horizons = INTERVAL_HORIZONS[args.interval]
    step_h = args.step_hours or INTERVAL_STEP_HOURS[args.interval]
    cases = []
    at = start
    while at <= end:
        for tick in frames:
            ctx = build_context(frames[tick], at, tick, fng, interval=args.interval)
            outs = {
                f"outcome_{h}h_pct": outcome_at(frames[tick], at, h)
                for h in horizons
            }
            main_h = f"outcome_{horizons[-1]}h_pct"
            if ctx is not None and outs.get(main_h) is not None:
                outs = {k: (round(v, 3) if v is not None else None) for k, v in outs.items()}
                cases.append({"at": at.isoformat(), "ctx": ctx, **outs})
        at += pd.Timedelta(hours=step_h)
    print(f"{len(cases)} cases x {len(args.models.split(','))} models", flush=True)

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["model"], r.get("interval", "1h"), r["at"], r["ticker"]))
        print(f"resume: {len(done)} results already present", flush=True)

    tripped = []
    with out_path.open("a") as fh:
        for model in args.models.split(","):
            todo = [
                c
                for c in cases
                if (model, args.interval, c["at"], c["ctx"]["ticker"]) not in done
            ]
            print(f"[{model}] {len(todo)} queries", flush=True)
            consec_err = 0
            for i, c in enumerate(todo):
                t0 = time.time()
                raw = None
                try:
                    raw = query_llama_server(
                        model, _build_prompt(c["ctx"]), n_predict=2048, temperature=0.0
                    )
                    vote, reason, conf = _parse_response(raw)
                except Exception as e:
                    vote, reason, conf = "ERROR", str(e)[:120], None
                row = {
                    "model": model,
                    "interval": args.interval,
                    "at": c["at"],
                    "ticker": c["ctx"]["ticker"],
                    "vote": vote,
                    "conf": conf,
                    "outcome_pct": c[f"outcome_{INTERVAL_HORIZONS[args.interval][-1]}h_pct"],
                    **{k: c[k] for k in c if k.startswith("outcome_")},
                    "secs": round(time.time() - t0, 1),
                }
                # Raw output kept on failures always (post-mortem), on all
                # rows with --keep-raw (pilot analysis).
                if args.keep_raw or vote == "ERROR" or vote is None:
                    row["raw"] = (raw or "")[:300]
                    if vote == "ERROR":
                        row["error"] = reason
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                # Circuit breaker: a dead server / broken model produces an
                # unbroken ERROR streak — abort the model phase instead of
                # burning hours writing garbage rows.
                consec_err = consec_err + 1 if vote == "ERROR" else 0
                if consec_err >= 6:
                    print(
                        f"[{model}] CIRCUIT BREAKER: 6 consecutive errors "
                        f"after {i + 1} queries — aborting model phase. "
                        f"Last error: {reason}",
                        flush=True,
                    )
                    tripped.append(model)
                    break
                if i % 25 == 0:
                    print(
                        f"[{model}] {i}/{len(todo)} last={vote} {time.time()-t0:.0f}s",
                        flush=True,
                    )
    stop_server()
    if tripped:
        print(f"BACKTEST RUN INCOMPLETE — breaker tripped: {tripped}", flush=True)
        sys.exit(2)
    print("BACKTEST RUN COMPLETE", flush=True)


def score(path):
    rows = [json.loads(line) for line in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r.get("interval", "1h"))].append(r)
    print(f"{'model':12s} {'interval':>8s} {'horizon':>8s} {'dir acc':>8s} {'votes':>6s} {'abstain%':>8s} {'err':>4s}")
    for (m, iv), rs in sorted(groups.items()):
        hkeys = sorted(
            {k for r in rs for k in r if k.startswith("outcome_") and k != "outcome_pct"},
            key=lambda k: float(k.split("_")[1].rstrip("h")),
        )
        holds = sum(1 for r in rs if r["vote"] not in ("BUY", "SELL", "ERROR"))
        errs = sum(1 for r in rs if r["vote"] == "ERROR")
        for hk in hkeys:
            hit = dirn = 0
            for r in rs:
                o = r.get(hk)
                if o is None or r["vote"] not in ("BUY", "SELL"):
                    continue
                dirn += 1
                hit += (r["vote"] == "BUY" and o > 0) or (r["vote"] == "SELL" and o < 0)
            acc = hit / dirn * 100 if dirn else 0.0
            label = hk.split("_")[1]
            print(f"{m:12s} {iv:>8s} {label:>8s} {acc:7.1f}% {dirn:6d} {holds/len(rs)*100:7.1f}% {errs:4d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="ministral3,qwen3,phi4_mini,fin_r1")
    p.add_argument("--start", default="2026-02-01")
    p.add_argument("--end", default="2026-07-11")
    p.add_argument("--step-hours", type=int, default=0)
    p.add_argument("--interval", default="1h", choices=list(INTERVAL_HORIZONS))
    p.add_argument("--tickers", default="BTC-USD,ETH-USD")
    p.add_argument("--out", default="data/llm_backtest_results.jsonl")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--score", metavar="RESULTS")
    a = p.parse_args()
    if a.score:
        score(a.score)
    else:
        run(a)
