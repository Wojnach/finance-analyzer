"""Chronos-Bolt time-series foundation-model walk-forward backtest.

Same grid, outcome math and row format as scripts/llm_backtest.py so rows
interleave with the model-comparison matrix (model names "chronos-<h>h",
one row per horizon). Inference is IN-PROCESS via the official chronos
API (BaseChronosPipeline.from_pretrained + predict_quantiles — the same
call path scripts/chronos_bolt_benchmark.py validated), NOT the retired
live path (portfolio/signals/forecast.py), whose audited defects this
script deliberately avoids:
  - horizons are CANDLE COUNTS of an explicit --interval (no wall-clock
    mislabeling, no silent 15m fallback frame);
  - outcome labels come from the SAME candle series as the context
    (llm_backtest outcome math — no snapshot lookup, no cross-source
    baseline);
  - the context ends at the as-of candle, which is always a COMPLETED
    historical candle (grid requires the last horizon's outcome to
    exist), never an in-progress partial;
  - bolt takes a raw close tensor — no synthetic contiguous timestamps
    are fabricated;
  - no gates, no EMA fallback, no composite: raw model direction only,
    ABSTAIN on inference error;
  - a flat neutral band around the last close (default 0.1% =
    chronos_bolt_benchmark.FLAT_BAND; kronos_infer's own neutral
    threshold lives on herc2 and was not readable when this was
    written, so the band is explicit and sweepable via --neutral-band).

conf = fraction of the 9 decile quantiles strictly on the vote's side of
the last close — a direct read of the model's predictive CDF at the
current price (bolt's quantile head is trained on exactly these deciles),
so P(up) ~ conf for BUY. HOLD/ABSTAIN rows carry conf null.

Chronos-Bolt weights are frozen/pretrained — this script trains/fits
nothing, so the only leakage surface is the close window fed per call
(guarded in context_window: open_time <= at only).

Usage (on the GPU host):
    python scripts/chronos_backtest.py --start 2026-02-01 --end 2026-07-11 \
        --interval 1h --out data/chronos_backtest_results.jsonl
    python scripts/llm_backtest.py --score data/chronos_backtest_results.jsonl
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_backtest as lb  # noqa: E402 — __main__-guarded, import is side-effect free

if platform.system() == "Windows":
    DEFAULT_MODEL_DIR = Path(r"Q:\models\chronos-bolt-small")
else:
    DEFAULT_MODEL_DIR = Path.home() / "models" / "chronos-bolt-small"

# Bolt's quantile head is trained on exactly these deciles — predict_quantiles
# reads them directly, no interpolation. Index 4 = median.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def horizon_candles(h, iv_h, is_yf):
    # Horizons are candle periods; mirror llm_backtest's outcome candle math
    # (session-aware for equities) so the prediction span and the scored
    # outcome cover the same candles.
    if is_yf and h >= 24:
        return max(1, round(h / iv_h * 7 / 24))
    return max(1, round(h / iv_h))


def context_window(df, at, n_ctx):
    # WALK-FORWARD GUARD: only candles with open_time <= at reach the model;
    # the context window ends at the as-of candle, no future rows.
    return df[df.index <= at].tail(n_ctx)


def load_pipeline(model_dir, device):
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(str(model_dir), device_map=device)


def chronos_quantiles(pipe, closes, prediction_length):
    import torch

    ctx = torch.tensor(closes, dtype=torch.float32).unsqueeze(0)
    q, _mean = pipe.predict_quantiles(
        ctx, prediction_length=prediction_length, quantile_levels=QUANTILE_LEVELS
    )
    # q shape: [batch, prediction_length, n_quantiles]
    return q


def vote_from_qrow(qrow, last_close, band):
    median = qrow[len(qrow) // 2]
    move = (median - last_close) / last_close
    if abs(move) < band:
        return "HOLD", None
    if move > 0:
        return "BUY", round(sum(1 for v in qrow if v > last_close) / len(qrow), 2)
    return "SELL", round(sum(1 for v in qrow if v < last_close) / len(qrow), 2)


def build_cases(args, frames, fng, horizons, step_h, iv_h, start, end):
    cases = []
    at = start
    while at <= end:
        for tick in frames:
            if tick in lb.YF_TICKERS:
                near = frames[tick].index[frames[tick].index <= at]
                if len(near) == 0 or (at - near[-1]) > pd.Timedelta(hours=iv_h):
                    continue
            # ctx is used ONLY as the constructibility gate (>=120 hist
            # candles, computed from df.index <= at inside build_context) so
            # the grid matches llm_backtest exactly — bolt never sees it.
            ctx = lb.build_context(frames[tick], at, tick, fng, interval=args.interval)
            # Outcomes look into the future by construction, but they are
            # scoring labels written to the output row only — never fed to
            # the model and nothing is fitted on them.
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
            main_h = f"outcome_{horizons[-1]}h_pct"
            if ctx is not None and outs.get(main_h) is not None:
                outs = {
                    k: (round(v, 3) if v is not None else None) for k, v in outs.items()
                }
                cases.append({"at": at.isoformat(), "ticker": tick, **outs})
        at += pd.Timedelta(hours=step_h)
    return cases


def run(args):
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    fng = lb.fetch_fng()
    wanted = [t.strip() for t in args.tickers.split(",")]
    frames = {}
    for tick in wanted:
        # Same pad as llm_backtest (200 candles) so the grid is identical;
        # early timestamps get <512 context candles, bolt accepts any window.
        pad = start - pd.Timedelta(hours=200 * lb._interval_hours(args.interval))
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

    horizons = lb.INTERVAL_HORIZONS[args.interval]
    step_h = args.step_hours or lb.INTERVAL_STEP_HOURS[args.interval]
    iv_h = lb._interval_hours(args.interval)
    cases = build_cases(args, frames, fng, horizons, step_h, iv_h, start, end)
    models = {h: f"chronos-{h}h" for h in horizons}
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR

    if args.dry_payload:
        c = cases[0]
        at = pd.Timestamp(c["at"])
        n_by_h = {
            h: horizon_candles(h, iv_h, c["ticker"] in lb.YF_TICKERS) for h in horizons
        }
        win = context_window(frames[c["ticker"]], at, args.ctx)
        closes = [float(x) for x in win["close"]]
        print(
            json.dumps(
                {
                    "case": c,
                    "models": models,
                    "model_dir": str(model_dir),
                    "quantile_levels": QUANTILE_LEVELS,
                    "neutral_band": args.neutral_band,
                    "n_by_h": {str(h): n for h, n in n_by_h.items()},
                    "prediction_length": max(n_by_h.values()),
                    "n_context_closes": len(closes),
                    "first_ctx": {"ts": win.index[0].isoformat(), "close": closes[0]},
                    "last_ctx": {"ts": win.index[-1].isoformat(), "close": closes[-1]},
                    "no_leak": bool(win.index[-1] <= at),
                },
                indent=2,
            )
        )
        return

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["model"], r.get("interval", "1h"), r["at"], r["ticker"]))
        print(f"resume: {len(done)} results already present", flush=True)

    todo = []
    for c in cases:
        missing = [
            h
            for h in horizons
            if (models[h], args.interval, c["at"], c["ticker"]) not in done
        ]
        if missing:
            todo.append((c, missing))
    print(f"{len(cases)} cases, {len(todo)} need inference", flush=True)
    if not todo:
        print("CHRONOS BACKTEST COMPLETE", flush=True)
        return

    t0 = time.time()
    pipe = load_pipeline(model_dir, args.device)
    print(f"loaded {model_dir} on {args.device} in {time.time() - t0:.1f}s", flush=True)

    consec_err = 0
    with out_path.open("a") as fh:
        for i, (c, missing) in enumerate(todo):
            tick = c["ticker"]
            is_yf = tick in lb.YF_TICKERS
            n_by_h = {h: horizon_candles(h, iv_h, is_yf) for h in horizons}
            t0 = time.time()
            q, last_close, err = None, None, None
            try:
                win = context_window(frames[tick], pd.Timestamp(c["at"]), args.ctx)
                closes = [float(x) for x in win["close"]]
                last_close = closes[-1]
                q = chronos_quantiles(pipe, closes, max(n_by_h.values()))
            except Exception as e:
                err = str(e)[:160]
            secs = round(time.time() - t0, 1)
            vote = "ABSTAIN"
            for h in missing:
                conf = None
                if q is None:
                    vote = "ABSTAIN"
                else:
                    qrow = [float(x) for x in q[0, n_by_h[h] - 1]]
                    vote, conf = vote_from_qrow(qrow, last_close, args.neutral_band)
                row = {
                    "model": models[h],
                    "interval": args.interval,
                    "arm": "A",
                    "at": c["at"],
                    "ticker": tick,
                    "vote": vote,
                    "conf": conf,
                    "outcome_pct": c[f"outcome_{horizons[-1]}h_pct"],
                    **{k: c[k] for k in c if k.startswith("outcome_")},
                    "secs": secs,
                }
                if err:
                    row["error"] = err
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            consec_err = consec_err + 1 if q is None else 0
            if consec_err >= 6:
                print(
                    f"CIRCUIT BREAKER: 6 consecutive inference failures after "
                    f"{i + 1} cases — aborting. Last error: {err}",
                    flush=True,
                )
                sys.exit(2)
            if i % 25 == 0:
                print(f"{i}/{len(todo)} {tick} last={vote} {secs:.1f}s", flush=True)
    print("CHRONOS BACKTEST COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-02-01")
    p.add_argument("--end", default="2026-07-11")
    p.add_argument("--step-hours", type=int, default=0)
    p.add_argument("--interval", default="1h", choices=list(lb.INTERVAL_HORIZONS))
    p.add_argument("--tickers", default="BTC-USD,ETH-USD")
    p.add_argument("--out", default="data/chronos_backtest_results.jsonl")
    p.add_argument(
        "--ctx",
        type=int,
        default=512,
        help="context candles per call (chronos-bolt max context 2048)",
    )
    p.add_argument(
        "--neutral-band",
        type=float,
        default=0.001,
        dest="neutral_band",
        help="|median move| below this fraction of last close -> HOLD "
        "(0.001 = chronos_bolt_benchmark FLAT_BAND)",
    )
    p.add_argument("--model-dir", default=None, dest="model_dir")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dry-payload",
        action="store_true",
        help="build grid, print first case's model input summary, no inference",
    )
    run(p.parse_args())
