"""Offline benchmark: amazon/chronos-2 (live) vs amazon/chronos-bolt-small.

WHY (2026-06-01): data/local_llm_report_latest.json has carried the
recommendation "Benchmark chronos-bolt-small against the current Chronos
model before any live swap" since March — but the live config already
moved to chronos-2 (forecast_signal._CHRONOS_MODEL), and the report's
trigger is gated on startswith("amazon/chronos-t5"), so it never fired
against the real production model. Meanwhile forecast accuracy on our own
data is poor and DECAYS with horizon (51% @3h -> 38% @1d, see
scripts/short_horizon_llm_report.py). This script finally runs the
head-to-head on OUR price data, scored at the intraday horizons that
actually matter (1h, 3h).

It is OFFLINE and read-only w.r.t. the trading system. It loads both
forecast models on the GPU under gpu_lock so it does not fight the live
loop's Chronos slot, rolls a walk-forward window over recent hourly
candles, and scores directional hit-rate (predicted median move sign vs
realized move sign, with a flat band) per model per horizon.

Decision rule: chronos-bolt-small is ~5x smaller/faster than t5 and a
peer of chronos-2 on public benchmarks. We swap ONLY if it matches or
beats chronos-2's directional hit-rate on our data at 1h/3h. If neither
clears ~50%, the real conclusion is "forecast is not an edge intraday",
not "pick the other model".

Usage (main venv has chronos-forecasting + torch+cuda):
    .venv/Scripts/python.exe scripts/chronos_bolt_benchmark.py
    .venv/Scripts/python.exe scripts/chronos_bolt_benchmark.py --tickers BTC,ETH --days 30 --windows 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LIVE_MODEL = "amazon/chronos-2"
BOLT_PATH_WIN = r"Q:\models\chronos-bolt-small"
BOLT_PATH_NIX = "/mnt/q/models/chronos-bolt-small"

# 2026-06-01 (firm-up pass): crypto via Binance spot, metals via Binance FAPI.
# Routed through the project's own collectors (binance_klines / binance_fapi_klines)
# so the benchmark uses the SAME candle source the live forecast path uses
# (forecast_signal._load_candles), not a parallel hand-rolled fetch. MSTR (Alpaca)
# is out of scope — equities have gaps/half-days that distort a continuous
# walk-forward; crypto+metals 24/7 series are the clean test for this model swap.
_SPOT = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
_FAPI = {"XAU": "XAUUSDT", "XAG": "XAGUSDT"}

FLAT_BAND = 0.001  # |move| < 0.1% counts as flat -> excluded from directional score


def fetch_hourly_closes(ticker: str, days: int) -> list[float]:
    """1h closes via the project collectors. limit honors Binance's 1500 cap."""
    limit = min(1500, days * 24 + 50)
    key = ticker.upper()
    if key in _SPOT:
        from portfolio.data_collector import binance_klines
        df = binance_klines(_SPOT[key], interval="1h", limit=limit)
    elif key in _FAPI:
        from portfolio.data_collector import binance_fapi_klines
        df = binance_fapi_klines(_FAPI[key], interval="1h", limit=limit)
    else:
        raise ValueError(f"no spot/fapi symbol for {ticker}")
    if df is None or len(df) < 60:
        raise ValueError(f"insufficient candles for {ticker}: {0 if df is None else len(df)}")
    return [float(x) for x in df["close"].values.tolist()]


def _load_chronos2():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(LIVE_MODEL, device_map="cuda")


def _load_bolt():
    from chronos import BaseChronosPipeline
    path = BOLT_PATH_WIN if sys.platform.startswith("win") else BOLT_PATH_NIX
    return BaseChronosPipeline.from_pretrained(path, device_map="cuda")


def _predict_dir_chronos2(pipe, context: list[float], h: int) -> int:
    """Return predicted sign of the h-ahead move (+1 up, -1 down, 0 flat)."""
    import pandas as pd
    n = len(context)
    ts = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")
    df = pd.DataFrame({"timestamp": ts, "target": context, "id": "x"})
    pred = pipe.predict_df(df, prediction_length=h, quantile_levels=[0.5],
                           id_column="id", timestamp_column="timestamp", target="target")
    median = float(pred.iloc[h - 1]["0.5"])
    return _sign(median, context[-1])


def _predict_dir_bolt(pipe, context: list[float], h: int) -> int:
    import torch
    # chronos 2.2.2 ChronosBoltPipeline.predict_quantiles takes `inputs`
    # positionally and expects a batched tensor [batch, context_len].
    ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    q, _mean = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=[0.5])
    # q shape: [batch, prediction_length, n_quantiles]
    median = float(q[0, h - 1, 0])
    return _sign(median, context[-1])


def _sign(pred_price: float, last_price: float) -> int:
    move = (pred_price - last_price) / last_price
    if abs(move) < FLAT_BAND:
        return 0
    return 1 if move > 0 else -1


def _realized_sign(closes: list[float], i: int, h: int) -> int:
    move = (closes[i + h] - closes[i]) / closes[i]
    if abs(move) < FLAT_BAND:
        return 0
    return 1 if move > 0 else -1


def benchmark(tickers: list[str], days: int, windows: int, ctx_len: int, horizons: list[int]) -> dict:
    from portfolio.gpu_gate import gpu_gate  # project GPU coordination (Q:/models/.gpu_lock)

    out: dict = {"live_model": LIVE_MODEL, "challenger": "chronos-bolt-small",
                 "ctx_len": ctx_len, "horizons": horizons, "flat_band": FLAT_BAND,
                 "tickers": {}, "scope": "crypto (spot) + metals (FAPI); MSTR/equities excluded (gappy series)"}

    series = {t: fetch_hourly_closes(t, days) for t in tickers}
    for t, c in series.items():
        print(f"  fetched {t}: {len(c)} hourly closes")

    # one GPU session: load both, score everything, release.
    with gpu_gate("chronos_benchmark", timeout=120):
        print("  loading chronos-2 ..."); t0 = time.time()
        c2 = _load_chronos2(); print(f"    loaded in {time.time()-t0:.1f}s")
        print("  loading chronos-bolt-small ..."); t0 = time.time()
        bolt = _load_bolt(); print(f"    loaded in {time.time()-t0:.1f}s")

        for t, closes in series.items():
            n = len(closes)
            max_h = max(horizons)
            # walk-forward indices: need ctx_len history before, max_h after
            idxs = list(range(ctx_len, n - max_h))
            if windows and len(idxs) > windows:
                step = len(idxs) // windows
                idxs = idxs[::step][:windows]
            tres = {f"{h}h": {"chronos2": {"correct": 0, "scored": 0},
                              "bolt": {"correct": 0, "scored": 0}} for h in horizons}
            for i in idxs:
                ctx = closes[i - ctx_len:i]
                for h in horizons:
                    realized = _realized_sign(closes, i, h)
                    if realized == 0:
                        continue  # flat realized -> not a directional test
                    for name, fn, pipe in (("chronos2", _predict_dir_chronos2, c2),
                                           ("bolt", _predict_dir_bolt, bolt)):
                        try:
                            pred = fn(pipe, ctx, h)
                        except Exception as e:
                            tres[f"{h}h"][name].setdefault("errors", 0)
                            tres[f"{h}h"][name]["errors"] += 1
                            continue
                        if pred == 0:
                            continue  # model abstained (flat) -> not scored
                        tres[f"{h}h"][name]["scored"] += 1
                        if pred == realized:
                            tres[f"{h}h"][name]["correct"] += 1
            for h in horizons:
                for name in ("chronos2", "bolt"):
                    d = tres[f"{h}h"][name]
                    d["accuracy"] = round(100 * d["correct"] / d["scored"], 1) if d["scored"] else None
            out["tickers"][t] = tres
            print(f"  scored {t}: {tres}")
    return out


def print_report(out: dict) -> None:
    print(f"\nChronos-2 (live) vs chronos-bolt-small — directional hit-rate")
    print("=" * 64)
    for t, tres in out["tickers"].items():
        print(f"\n{t}:")
        for h, m in tres.items():
            c2 = m["chronos2"]; b = m["bolt"]
            print(f"  {h:>4s}  chronos-2 {str(c2.get('accuracy')):>6}% (n={c2['scored']})   "
                  f"bolt {str(b.get('accuracy')):>6}% (n={b['scored']})")
    print("\nVerdict rule: swap to bolt only if it matches/beats chronos-2 at 1h/3h.")
    print("If both hover ~50%, forecast is not an intraday edge regardless of model.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default="BTC,ETH,XAU,XAG")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--windows", type=int, default=300, help="max walk-forward windows per ticker (0=all)")
    ap.add_argument("--ctx-len", type=int, default=168, help="context hours fed to the model (default 168=7d)")
    ap.add_argument("--horizons", default="1,3", help="comma hours")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",")]
    out = benchmark(tickers, args.days, args.windows, args.ctx_len, horizons)
    print_report(out)

    dst = Path(args.json) if args.json else (REPO / "data" / "chronos_bolt_benchmark.json")
    from portfolio.file_utils import atomic_write_json
    atomic_write_json(dst, out)
    print(f"\nWrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
