"""Kronos K-line foundation-model walk-forward backtest.

Same grid, outcome math and row format as scripts/llm_backtest.py so rows
interleave with the model-comparison matrix (model names "kronos-<h>h",
one row per horizon). Inference goes through the existing subprocess tool
kronos_infer.py (Windows Q:\\models\\kronos_infer.py, Linux
~/models/kronos_infer.py).

kronos_infer.py contract (read from herc2 2026-07-18): reads ONE JSON
request from stdin ({"candles": [...], "prices_close": [...], optional
temperature/top_p/sample_count}), horizons via --horizons as integer
CANDLE PERIODS, writes ONE JSON response to stdout ({"method":
"kronos"|"statistical_fallback"|"none"|"error", "results": {"<n>h":
{"direction": "up"|"down"|"neutral", "pct_move", "confidence", ...}}})
and exits. No request loop / batch array — every call pays the full
torch+CUDA init and 102M model load, so calls are per (at, ticker) with
all horizons in one shot (nothing finer to batch).

Kronos weights are frozen/pretrained — this script trains/fits nothing,
so the only leakage surface is the candle window fed per call (guarded
in candles_payload).

Usage (on the GPU host):
    python scripts/kronos_backtest.py --start 2026-02-01 --end 2026-07-11 \
        --interval 1h --out data/kronos_backtest_results.jsonl
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_backtest as lb  # noqa: E402 — __main__-guarded, import is side-effect free

if platform.system() == "Windows":
    KRONOS_INFER = Path(r"Q:\models\kronos_infer.py")
else:
    KRONOS_INFER = Path.home() / "models" / "kronos_infer.py"


def horizon_candles(h, iv_h, is_yf):
    # Kronos horizons are candle periods; mirror llm_backtest's outcome
    # candle math (session-aware for equities) so the prediction span and
    # the scored outcome cover the same candles.
    if is_yf and h >= 24:
        return max(1, round(h / iv_h * 7 / 24))
    return max(1, round(h / iv_h))


def candles_payload(df, at, n_ctx):
    # WALK-FORWARD GUARD: only candles with open_time <= at reach the model;
    # the context window ends at the as-of candle, no future rows.
    hist = df[df.index <= at].tail(n_ctx)
    return [
        {
            "timestamp": ts.isoformat(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
        }
        for ts, r in hist.iterrows()
    ]


def kronos_call(candles, n_candles, args):
    payload = json.dumps(
        {
            "candles": candles,
            "prices_close": [c["close"] for c in candles],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sample_count": args.samples,
        }
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(KRONOS_INFER),
            "--horizons",
            ",".join(str(n) for n in sorted(set(n_candles))),
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {proc.stderr[-200:]}")
    text = (proc.stdout or "").strip()
    # HF from_pretrained can leak prints before the JSON — brace-offset
    # fallback, same approach as portfolio/signals/forecast.py.
    idx = text.find("{")
    if idx < 0:
        raise RuntimeError(f"no JSON on stdout: {text[:120]!r}")
    return json.loads(text[idx:])


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
            # the grid matches llm_backtest exactly — Kronos never sees it.
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
        # early timestamps get <480 context candles, kronos_infer accepts
        # any window >=10.
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
    models = {h: f"kronos-{h}h" for h in horizons}

    if args.dry_payload:
        c = cases[0]
        n_by_h = {
            h: horizon_candles(h, iv_h, c["ticker"] in lb.YF_TICKERS) for h in horizons
        }
        payload = {
            "candles": candles_payload(
                frames[c["ticker"]], pd.Timestamp(c["at"]), args.ctx
            ),
            "prices_close": None,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sample_count": args.samples,
        }
        payload["prices_close"] = [x["close"] for x in payload["candles"]]
        print(
            json.dumps(
                {
                    "case": c,
                    "models": models,
                    "horizons_arg": ",".join(
                        str(n) for n in sorted(set(n_by_h.values()))
                    ),
                    "n_by_h": {str(h): n for h, n in n_by_h.items()},
                    "n_candles": len(payload["candles"]),
                    "first_candle": payload["candles"][0],
                    "last_candle": payload["candles"][-1],
                    "stdin_payload_head": json.dumps(payload)[:400],
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
    print(
        f"{len(cases)} cases, {len(todo)} need inference "
        f"(model reload per call — expect seconds each)",
        flush=True,
    )

    consec_err = 0
    with out_path.open("a") as fh:
        for i, (c, missing) in enumerate(todo):
            tick = c["ticker"]
            is_yf = tick in lb.YF_TICKERS
            n_by_h = {h: horizon_candles(h, iv_h, is_yf) for h in horizons}
            t0 = time.time()
            results, err = None, None
            try:
                candles = candles_payload(frames[tick], pd.Timestamp(c["at"]), args.ctx)
                res = kronos_call(candles, list(n_by_h.values()), args)
                if res.get("method") != "kronos":
                    # "statistical_fallback" is kronos_infer's linear-regression
                    # path when the model can't load — a vote here would score
                    # the wrong model. ABSTAIN instead.
                    raise RuntimeError(
                        f"method={res.get('method')} {str(res.get('error', ''))[:120]}"
                    )
                results = res["results"]
            except Exception as e:
                err = str(e)[:160]
            secs = round(time.time() - t0, 1)
            vote = "ABSTAIN"
            for h in missing:
                if results is None:
                    vote = "ABSTAIN"
                else:
                    d = (results.get(f"{n_by_h[h]}h") or {}).get("direction")
                    vote = {"up": "BUY", "down": "SELL", "neutral": "HOLD"}.get(
                        d, "ABSTAIN"
                    )
                row = {
                    "model": models[h],
                    "interval": args.interval,
                    "arm": "A",
                    "at": c["at"],
                    "ticker": tick,
                    "vote": vote,
                    # kronos_infer's "confidence" is min(|pct_move|*0.15, 0.7)
                    # — a magnitude heuristic, not a probability → conf null.
                    "conf": None,
                    "outcome_pct": c[f"outcome_{horizons[-1]}h_pct"],
                    **{k: c[k] for k in c if k.startswith("outcome_")},
                    "secs": secs,
                }
                if err:
                    row["error"] = err
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            consec_err = consec_err + 1 if results is None else 0
            if consec_err >= 6:
                print(
                    f"CIRCUIT BREAKER: 6 consecutive subprocess failures after "
                    f"{i + 1} cases — aborting. Last error: {err}",
                    flush=True,
                )
                sys.exit(2)
            if i % 25 == 0:
                print(f"{i}/{len(todo)} {tick} last={vote} {secs:.0f}s", flush=True)
    print("KRONOS BACKTEST COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-02-01")
    p.add_argument("--end", default="2026-07-11")
    p.add_argument("--step-hours", type=int, default=0)
    p.add_argument("--interval", default="1h", choices=list(lb.INTERVAL_HORIZONS))
    p.add_argument("--tickers", default="BTC-USD,ETH-USD")
    p.add_argument("--out", default="data/kronos_backtest_results.jsonl")
    p.add_argument(
        "--ctx",
        type=int,
        default=480,
        help="context candles per call (kronos-base max_context 512)",
    )
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument(
        "--dry-payload",
        action="store_true",
        help="build grid, print first case's stdin payload, no inference",
    )
    run(p.parse_args())
