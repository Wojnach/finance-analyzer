"""XGBoost walk-forward directional backtest — CPU baseline for the LLM matrix.

Reuses llm_backtest machinery (fetch, grid, outcome semantics) and writes
rows schema-compatible with llm_backtest results: one row per
(xgboost-<h>h, at, ticker), arm "A". One binary model per horizon,
retrained at every grid timestamp on an expanding window of labels that
are fully resolved before the timestamp.

Usage:
    python scripts/xgb_backtest.py --start 2026-02-01 --end 2026-07-11 \
        --interval 1h --step-hours 8 --tickers BTC-USD,ETH-USD \
        --out data/xgb_backtest_results.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_backtest as lb  # noqa: E402 — import is side-effect free (__main__ guard)
from portfolio.signal_utils import ema, rsi  # noqa: E402 — repo root on path via lb

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    eval_metric="logloss",
    random_state=0,
)
MIN_TRAIN_ROWS = 200
MIN_HIST = 120  # mirrors build_context's 120-candle minimum


def feature_frame(df: pd.DataFrame, fng: dict) -> pd.DataFrame:
    # NO-LEAKAGE: every column is causal — ewm/rolling/shift at candle t use
    # only candles with open_time <= t. Same information basis as
    # build_context, which likewise reads the as-of candle's close at t.
    close = df["close"]
    f = pd.DataFrame(index=df.index)
    f["rsi"] = rsi(close, 14)
    macd_line = ema(close, 12) - ema(close, 26)
    f["macd_hist"] = macd_line - ema(macd_line, 9)
    f["ema_gap_pct"] = (ema(close, 9) / ema(close, 21) - 1) * 100
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["bb_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20)
    f["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    # NO-LEAKAGE: last close with open_time <= t-24h, same asof lookup as
    # build_context's day_ago line
    p24 = df.index.searchsorted(df.index - pd.Timedelta(hours=24), side="right") - 1
    chg = (close.to_numpy() / close.to_numpy()[np.clip(p24, 0, None)] - 1) * 100
    chg[p24 < 0] = 0.0
    f["chg24"] = chg
    for n in (1, 3, 6, 12):
        f[f"lr{n}"] = np.log(close / close.shift(n))
    # F&G for date D is published at the start of day D — known intraday at t
    f["fng"] = [fng.get(d, 50) for d in df.index.date]
    return f


def label_arrays(df: pd.DataFrame, horizons, iv_h: float, sessions: bool) -> dict:
    # Label = future close > basis close, same outcome semantics as
    # llm_backtest (outcome_at for wall-clock, outcome_sessions candle
    # counts for yf tickers). resolve = CLOSE time of the future candle
    # (open_time + interval): the label only exists once that candle closed.
    close = df["close"].to_numpy()
    idx = df.index
    out = {}
    for h in horizons:
        if sessions:
            n = max(1, round(h / iv_h * 7 / 24)) if h >= 24 else max(1, round(h / iv_h))
            fpos = np.arange(len(df)) + n
        else:
            fpos = idx.searchsorted(idx + pd.Timedelta(hours=h), side="left")
        ok = np.flatnonzero(fpos < len(df))
        y = np.zeros(len(df), dtype=bool)
        y[ok] = close[fpos[ok]] > close[ok]
        resolve = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
        resolve.iloc[ok] = idx[fpos[ok]] + pd.Timedelta(hours=iv_h)
        out[h] = (y, resolve)
    return out


def run(args):
    from xgboost import XGBClassifier

    t_run = time.time()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    fng = lb.fetch_fng()
    horizons = lb.INTERVAL_HORIZONS[args.interval]
    step_h = args.step_hours or lb.INTERVAL_STEP_HOURS[args.interval]
    iv_h = lb._interval_hours(args.interval)
    wanted = [t.strip() for t in args.tickers.split(",")]
    # pad beyond llm_backtest's 200-candle warmup so the expanding training
    # window has resolved labels before the first grid stamp
    pad = start - max(
        pd.Timedelta(days=args.train_days), pd.Timedelta(hours=200 * iv_h)
    )
    frames = {}
    for tick in wanted:
        if tick in lb.TICKERS:
            sym, url = lb.TICKERS[tick], lb.BINANCE_KLINES
        elif tick in lb.FAPI_TICKERS:
            sym, url = lb.FAPI_TICKERS[tick], lb.BINANCE_FAPI_KLINES
        elif tick in lb.YF_TICKERS:
            frames[tick] = lb.fetch_klines_yf(
                lb.YF_TICKERS[tick],
                pad.strftime("%Y-%m-%d"),
                (end + pd.Timedelta(days=8)).strftime("%Y-%m-%d"),
                interval=args.interval,
            )
            print(f"{tick}: {len(frames[tick])} {args.interval} yf candles", flush=True)
            continue
        else:
            raise SystemExit(f"unknown ticker {tick}")
        frames[tick] = lb.fetch_klines_1h(
            sym,
            int(pad.timestamp() * 1000),
            int((end + pd.Timedelta(days=8)).timestamp() * 1000),
            base_url=url,
            interval=args.interval,
        )
        print(f"{tick}: {len(frames[tick])} {args.interval} candles", flush=True)

    feats, fmats, valid, labels = {}, {}, {}, {}
    for tick in frames:
        feats[tick] = feature_frame(frames[tick], fng)
        fmats[tick] = feats[tick].to_numpy(dtype=float)
        bv = np.isfinite(fmats[tick]).all(axis=1)
        bv[: MIN_HIST - 1] = False
        valid[tick] = bv
        labels[tick] = label_arrays(frames[tick], horizons, iv_h, tick in lb.YF_TICKERS)

    # grid construction mirrors llm_backtest.run
    cases = []
    at = start
    while at <= end:
        for tick in frames:
            pos = frames[tick].index.searchsorted(at, side="right") - 1
            if tick in lb.YF_TICKERS:
                # only timestamps that land on an actual session candle
                if pos < 0 or (at - frames[tick].index[pos]) > pd.Timedelta(hours=iv_h):
                    continue
            if pos + 1 < MIN_HIST:
                continue
            if tick in lb.YF_TICKERS:
                outs = {
                    f"outcome_{h}h_pct": lb.outcome_sessions(
                        frames[tick],
                        at,
                        (
                            max(1, round(h / iv_h * 7 / 24))
                            if h >= 24
                            else max(1, round(h / iv_h))
                        ),
                    )
                    for h in horizons
                }
            else:
                outs = {
                    f"outcome_{h}h_pct": lb.outcome_at(frames[tick], at, h)
                    for h in horizons
                }
            if outs[f"outcome_{horizons[-1]}h_pct"] is None:
                continue
            outs = {
                k: (round(v, 3) if v is not None else None) for k, v in outs.items()
            }
            cases.append({"at": at.isoformat(), "ticker": tick, "pos": pos, **outs})
        at += pd.Timedelta(hours=step_h)
    print(f"{len(cases)} cases x {len(horizons)} horizons", flush=True)

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["model"], r.get("interval", "1h"), r["at"], r["ticker"]))
        print(f"resume: {len(done)} results already present", flush=True)

    def train(at_ts, h):
        Xs, ys = [], []
        for tick in frames:
            y, resolve = labels[tick][h]
            # NO-LEAKAGE / WALK-FORWARD: expanding window — a row at candle t
            # trains only when its label's future candle CLOSED at or before
            # `at` (resolve <= at; NaT compares False). Features at t already
            # use only candles with open_time <= t.
            m = valid[tick] & (resolve <= at_ts).to_numpy()
            Xs.append(fmats[tick][m])
            ys.append(y[m])
        X, yv = np.concatenate(Xs), np.concatenate(ys)
        if len(yv) < MIN_TRAIN_ROWS or len(np.unique(yv)) < 2:
            return None
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X, yv)
        return clf

    written = 0
    with out_path.open("a") as fh:
        model_cache = {}
        cache_at = None
        for c in cases:
            if c["at"] != cache_at:
                model_cache, cache_at = {}, c["at"]
            at_ts = pd.Timestamp(c["at"])
            for h in horizons:
                name = f"xgboost-{h}h"
                if (name, args.interval, c["at"], c["ticker"]) in done:
                    continue
                t0 = time.time()
                if h not in model_cache:
                    model_cache[h] = train(at_ts, h)
                clf = model_cache[h]
                x = fmats[c["ticker"]][c["pos"]]
                if clf is None or not np.isfinite(x).all():
                    vote, conf = "ABSTAIN", None
                else:
                    p_up = float(clf.predict_proba(x.reshape(1, -1))[0, 1])
                    vote = "BUY" if p_up > 0.5 else "SELL"
                    conf = round(max(p_up, 1 - p_up), 3)
                row = {
                    "model": name,
                    "interval": args.interval,
                    "arm": "A",
                    "at": c["at"],
                    "ticker": c["ticker"],
                    "vote": vote,
                    "conf": conf,
                    "outcome_pct": c[f"outcome_{horizons[-1]}h_pct"],
                    **{k: c[k] for k in c if k.startswith("outcome_")},
                    "secs": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                written += 1
                if written % 100 == 0:
                    print(f"{written} rows, at={c['at']}", flush=True)
    print(
        f"XGB BACKTEST COMPLETE — {written} rows in {time.time() - t_run:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-02-01")
    p.add_argument("--end", default="2026-07-11")
    p.add_argument("--step-hours", type=int, default=0)
    p.add_argument("--interval", default="1h", choices=list(lb.INTERVAL_HORIZONS))
    p.add_argument("--tickers", default="BTC-USD,ETH-USD")
    p.add_argument("--out", default="data/xgb_backtest_results.jsonl")
    p.add_argument("--train-days", type=int, default=180)
    run(p.parse_args())
