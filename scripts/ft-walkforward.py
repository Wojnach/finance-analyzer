#!/usr/bin/env python3
"""Walk-forward validation for Freqtrade strategies.

Splits historical data into rolling train/test windows.
For each window: hyperopt on train period, backtest on test period.
Outputs summary table showing whether the strategy generalizes.

Usage:
    python3 scripts/ft-walkforward.py
    python3 scripts/ft-walkforward.py --epochs 200 --train-days 180 --test-days 60
    python3 scripts/ft-walkforward.py --data-start 20240601 --data-end 20260101
"""

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
USER_DATA = PROJECT_DIR / "user_data"
IMAGE = "docker.io/freqtradeorg/freqtrade:stable"
STRATEGY_JSON = USER_DATA / "strategies" / "ta_base_strategy.json"
BACKTEST_DIR = USER_DATA / "backtest_results"
WF_PARAMS_DIR = USER_DATA / "walkforward_params"


def run_ft(*args, timeout=3600):
    config = PROJECT_DIR / "config.json"
    if not config.exists():
        config = PROJECT_DIR / "config.example.json"
    cmd = [
        "podman",
        "run",
        "--rm",
        "--network=host",
        "--userns=keep-id",
        "-v",
        f"{USER_DATA}:/freqtrade/user_data",
        "-v",
        f"{config}:/freqtrade/config.json:ro",
        IMAGE,
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def fmt(dt):
    return dt.strftime("%Y%m%d")


def generate_windows(data_start, data_end, train_days, test_days):
    windows = []
    pos = data_start
    while pos + timedelta(days=train_days + test_days) <= data_end:
        train_end = pos + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        windows.append(
            {
                "train_start": pos,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
            }
        )
        pos += timedelta(days=test_days)
    return windows


def get_latest_backtest():
    last_result = BACKTEST_DIR / ".last_result.json"
    if not last_result.exists():
        return None
    ref = json.loads(last_result.read_text())
    zip_name = ref.get("latest_backtest", "")
    if not zip_name:
        return None
    zip_path = BACKTEST_DIR / zip_name
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if (
                name.endswith(".json")
                and "_config" not in name
                and "_market_change" not in name
            ):
                data = json.loads(z.read(name))
                if "strategy" in data:
                    return data["strategy"].get("TABaseStrategy", {})
    return None


def extract_metrics(strat_data):
    return {
        "trades": strat_data.get("total_trades", 0),
        "profit_pct": round(strat_data.get("profit_total", 0) * 100, 2),
        "profit_factor": round(strat_data.get("profit_factor", 0), 3),
        "max_dd_pct": round(strat_data.get("max_drawdown_account", 0) * 100, 2),
        "win_rate": round(strat_data.get("winrate", 0) * 100, 1),
        "sharpe": round(strat_data.get("sharpe", 0), 2),
    }


def run_window(window, idx, total, epochs, spaces):
    train_range = f"{fmt(window['train_start'])}-{fmt(window['train_end'])}"
    test_range = f"{fmt(window['test_start'])}-{fmt(window['test_end'])}"

    print(f"\n{'='*70}")
    print(f"  Window {idx+1}/{total}: Train {train_range} | Test {test_range}")
    print(f"{'='*70}")

    # 1. Hyperopt on train window
    print(f"  [1/3] Hyperopting ({epochs} epochs)...")
    r = run_ft(
        "hyperopt",
        "--config",
        "/freqtrade/config.json",
        "--strategy",
        "TABaseStrategy",
        "--strategy-path",
        "/freqtrade/user_data/strategies",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--spaces",
        *spaces.split(),
        "--epochs",
        str(epochs),
        "--timerange",
        train_range,
    )

    if r.returncode != 0:
        print(f"  HYPEROPT FAILED (exit {r.returncode})")
        if r.stderr:
            for line in r.stderr.strip().split("\n")[-3:]:
                print(f"    {line}")
        return {
            "window": idx + 1,
            "train": train_range,
            "test": test_range,
            "error": "hyperopt_failed",
        }

    # 2. Export best params (auto-updates ta_base_strategy.json)
    print(f"  [2/3] Exporting best params...")
    r = run_ft(
        "hyperopt-show",
        "--best",
        "--config",
        "/freqtrade/config.json",
        "--userdir",
        "/freqtrade/user_data",
    )

    if r.returncode != 0:
        print(f"  PARAM EXPORT FAILED (exit {r.returncode})")
        if r.stderr:
            for line in r.stderr.strip().split("\n")[-3:]:
                print(f"    {line}")
        return {
            "window": idx + 1,
            "train": train_range,
            "test": test_range,
            "error": "export_failed",
        }

    # Save a copy of this window's params
    if STRATEGY_JSON.exists():
        WF_PARAMS_DIR.mkdir(exist_ok=True)
        dest = WF_PARAMS_DIR / f"window_{idx+1:02d}_{train_range}.json"
        dest.write_text(STRATEGY_JSON.read_text())

    # 3. Backtest on test window
    print(f"  [3/3] Backtesting on {test_range}...")
    r = run_ft(
        "backtesting",
        "--config",
        "/freqtrade/config.json",
        "--strategy",
        "TABaseStrategy",
        "--strategy-path",
        "/freqtrade/user_data/strategies",
        "--timerange",
        test_range,
    )

    if r.returncode != 0:
        print(f"  BACKTEST FAILED (exit {r.returncode})")
        return {
            "window": idx + 1,
            "train": train_range,
            "test": test_range,
            "error": "backtest_failed",
        }

    strat = get_latest_backtest()
    if not strat:
        return {
            "window": idx + 1,
            "train": train_range,
            "test": test_range,
            "error": "no_results",
        }

    metrics = extract_metrics(strat)
    metrics["window"] = idx + 1
    metrics["train"] = train_range
    metrics["test"] = test_range
    print(
        f"  -> {metrics['trades']} trades | {metrics['profit_pct']:+.2f}% | PF {metrics['profit_factor']} | DD {metrics['max_dd_pct']}% | WR {metrics['win_rate']}%"
    )
    return metrics


def print_summary(results):
    print(f"\n\n{'='*70}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*70}\n")

    header = f"{'#':>3} | {'Test Period':>21} | {'Trades':>6} | {'Profit':>8} | {'PF':>6} | {'MaxDD':>6} | {'WinRate':>7} | {'Sharpe':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['window']:>3} | {r['test']:>21} | ERROR: {r['error']}")
        else:
            print(
                f"{r['window']:>3} | {r['test']:>21} "
                f"| {r['trades']:>6} "
                f"| {r['profit_pct']:>+7.2f}% "
                f"| {r['profit_factor']:>6.3f} "
                f"| {r['max_dd_pct']:>5.2f}% "
                f"| {r['win_rate']:>6.1f}% "
                f"| {r['sharpe']:>6.2f}"
            )

    ok = [r for r in results if "error" not in r]
    if ok:
        avg_pf = sum(r["profit_factor"] for r in ok) / len(ok)
        avg_profit = sum(r["profit_pct"] for r in ok) / len(ok)
        total_trades = sum(r["trades"] for r in ok)
        profitable = sum(1 for r in ok if r["profit_pct"] > 0)
        max_dd = max(r["max_dd_pct"] for r in ok)

        print(
            f"\n  Windows:            {len(ok)} completed ({len(results) - len(ok)} failed)"
        )
        print(f"  Profitable windows: {profitable}/{len(ok)}")
        print(f"  Avg profit factor:  {avg_pf:.3f}")
        print(f"  Avg profit/window:  {avg_profit:+.2f}%")
        print(f"  Total trades:       {total_trades}")
        print(f"  Worst drawdown:     {max_dd:.2f}%")
        verdict = "PASS" if avg_pf > 1.0 and profitable > len(ok) / 2 else "FAIL"
        print(
            f"  Verdict:            {verdict} (need avg PF > 1.0 + majority profitable)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for Freqtrade"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Hyperopt epochs per window (default: 100)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=180,
        help="Training window in days (default: 180)",
    )
    parser.add_argument(
        "--test-days", type=int, default=60, help="Test window in days (default: 60)"
    )
    parser.add_argument(
        "--data-start",
        default="20240210",
        help="Data start date YYYYMMDD (default: 20240210)",
    )
    parser.add_argument(
        "--data-end",
        default=datetime.now().strftime("%Y%m%d"),
        help="Data end date YYYYMMDD (default: today)",
    )
    parser.add_argument(
        "--spaces", default="buy sell", help="Hyperopt spaces (default: 'buy sell')"
    )
    args = parser.parse_args()

    data_start = datetime.strptime(args.data_start, "%Y%m%d")
    data_end = datetime.strptime(args.data_end, "%Y%m%d")
    windows = generate_windows(data_start, data_end, args.train_days, args.test_days)

    if not windows:
        print("ERROR: Not enough data for any walk-forward window.")
        print(
            f"  Need at least {args.train_days + args.test_days} days, have {(data_end - data_start).days}"
        )
        sys.exit(1)

    print(f"Walk-Forward Validation")
    print(
        f"  Train: {args.train_days}d | Test: {args.test_days}d | Step: {args.test_days}d"
    )
    print(
        f"  Data:  {fmt(data_start)} to {fmt(data_end)} ({(data_end - data_start).days}d)"
    )
    print(f"  Windows: {len(windows)} | Epochs: {args.epochs}/window")
    print(f"  Spaces: {args.spaces}")

    # Backup current params
    backup = None
    if STRATEGY_JSON.exists():
        backup = STRATEGY_JSON.read_text()
        print(f"  Backed up {STRATEGY_JSON.name}")

    results = []
    try:
        for i, w in enumerate(windows):
            result = run_window(w, i, len(windows), args.epochs, args.spaces)
            results.append(result)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Showing partial results...\n")
    finally:
        if backup is not None:
            STRATEGY_JSON.write_text(backup)
            print(f"\n  Restored original {STRATEGY_JSON.name}")

    if results:
        print_summary(results)
        results_file = USER_DATA / "walkforward_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        print(f"\n  Full results: {results_file}")
        if WF_PARAMS_DIR.exists():
            print(f"  Per-window params: {WF_PARAMS_DIR}/")


if __name__ == "__main__":
    main()
