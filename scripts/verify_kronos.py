"""Kronos verification script — test inference on a few tickers before enabling.

Runs on CPU only (no GPU, no fan spin), 1 thread, lowest priority.
Tests both crypto (Binance) and stock (Alpaca) candle loading paths.
"""

import json
import os
import sys
import time

# Throttle: CPU only, 1 thread, lowest priority
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
try:
    os.nice(19)
except (OSError, AttributeError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.signals.forecast import _load_candles_ohlcv


def verify_candle_loading():
    """Test that candle loading works for both Binance and Alpaca tickers."""
    test_tickers = {
        "BTC-USD": ("binance", "Crypto"),
        "XAU-USD": ("binance_fapi", "Metals"),
        "PLTR": ("alpaca", "Stock"),
        "NVDA": ("alpaca", "Stock"),
    }

    print("=" * 60)
    print("KRONOS VERIFICATION — Candle Loading")
    print("=" * 60)
    print(f"Mode: CPU only, 1 thread, nice=19\n")

    results = {}
    for ticker, (source, asset_class) in test_tickers.items():
        print(f"  {ticker} ({asset_class}, {source})...", end=" ", flush=True)
        t0 = time.time()
        try:
            candles = _load_candles_ohlcv(ticker, periods=100, interval="15m")
            elapsed = time.time() - t0
            if candles is not None and len(candles) > 0:
                print(f"OK — {len(candles)} candles in {elapsed:.1f}s")
                results[ticker] = {"status": "ok", "candles": len(candles), "time": elapsed}
            else:
                print(f"EMPTY — returned None/empty in {elapsed:.1f}s")
                results[ticker] = {"status": "empty", "time": elapsed}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL — {e} in {elapsed:.1f}s")
            results[ticker] = {"status": "error", "error": str(e), "time": elapsed}

    return results


def verify_kronos_inference():
    """Test Kronos model inference via subprocess (CPU only)."""
    import subprocess

    print("\n" + "=" * 60)
    print("KRONOS VERIFICATION — Model Inference")
    print("=" * 60)

    # Build synthetic candles for testing
    candles = []
    base_price = 3000.0
    for i in range(100):
        import random
        random.seed(42 + i)
        o = base_price + random.uniform(-5, 5)
        h = o + random.uniform(0, 10)
        l = o - random.uniform(0, 10)
        c = o + random.uniform(-5, 5)
        v = random.uniform(100, 1000)
        candles.append({"open": o, "high": h, "low": l, "close": c, "volume": v})

    # Find Kronos inference script
    kronos_script = None
    for path in ["/mnt/q/models/kronos_infer.py", "Q:/models/kronos_infer.py"]:
        if os.path.exists(path):
            kronos_script = path
            break

    if not kronos_script:
        print("  Kronos inference script not found!")
        return {"status": "not_found"}

    print(f"  Script: {kronos_script}")
    print(f"  Candles: {len(candles)}")
    print(f"  Running on CPU (CUDA_VISIBLE_DEVICES='')...", flush=True)

    payload = json.dumps({
        "candles": candles,
        "pred_len": 12,
        "num_samples": 3,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    t0 = time.time()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "1"

        # Use the LLM venv which has torch installed
        python_paths = [
            "/mnt/q/models/.venv-llm/Scripts/python.exe",
            "Q:/models/.venv-llm/Scripts/python.exe",
        ]
        python_exe = None
        for p in python_paths:
            if os.path.exists(p):
                python_exe = p
                break

        if not python_exe:
            print("  LLM venv python not found!")
            return {"status": "no_python"}

        result = subprocess.run(
            [python_exe, kronos_script],
            input=payload, capture_output=True, text=True,
            timeout=120, env=env,
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAIL — exit code {result.returncode} in {elapsed:.1f}s")
            print(f"  stderr: {result.stderr[:500]}")
            return {"status": "error", "returncode": result.returncode,
                    "stderr": result.stderr[:500], "time": elapsed}

        # Parse output
        stdout = result.stdout.strip()
        try:
            output = json.loads(stdout)
            method = output.get("method", "unknown")
            results_data = output.get("results", {})
            print(f"  OK — method={method}, {len(results_data)} result keys in {elapsed:.1f}s")
            if results_data:
                for k, v in list(results_data.items())[:3]:
                    print(f"    {k}: {v}")
            return {"status": "ok", "method": method, "time": elapsed,
                    "results": results_data}
        except json.JSONDecodeError:
            # Try to find JSON in stdout (HuggingFace noise)
            for line in stdout.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        output = json.loads(line)
                        print(f"  OK (extracted from noisy output) in {elapsed:.1f}s")
                        return {"status": "ok_noisy", "time": elapsed}
                    except json.JSONDecodeError:
                        continue
            print(f"  FAIL — no valid JSON in {elapsed:.1f}s")
            print(f"  stdout (first 500): {stdout[:500]}")
            return {"status": "no_json", "time": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {"status": "timeout", "time": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  EXCEPTION: {e} in {elapsed:.1f}s")
        return {"status": "exception", "error": str(e), "time": elapsed}


def main():
    print(f"Kronos Verification — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU only | 1 thread | nice=19\n")

    candle_results = verify_candle_loading()
    inference_result = verify_kronos_inference()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    for ticker, r in candle_results.items():
        status = "PASS" if r["status"] == "ok" else "FAIL"
        if r["status"] != "ok":
            all_ok = False
        print(f"  Candles {ticker}: {status}")

    inf_status = "PASS" if inference_result.get("status", "").startswith("ok") else "FAIL"
    if not inference_result.get("status", "").startswith("ok"):
        all_ok = False
    print(f"  Inference: {inf_status}")

    if all_ok:
        print("\n  ALL CHECKS PASSED — safe to enable kronos_enabled: true")
    else:
        print("\n  SOME CHECKS FAILED — review issues before enabling")

    return all_ok


if __name__ == "__main__":
    main()
