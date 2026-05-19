#!/usr/bin/env python3
"""Benchmark all GPU models to verify they fit within the 60s loop budget.

Simulates one full signal loop cycle:
1. Ministral-3 (2 crypto tickers via native binary)
2. Qwen3 (all 20 tickers — single mode vs batch mode)
3. Chronos-2 (all tickers via forecast pipeline)

Reports per-model timing and whether the budget fits.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LLAMA_CLI = r"Q:\models\llama-cpp-bin\cuda13\llama-completion.exe"
MINISTRAL_MODEL = r"Q:\models\ministral-3-8b-gguf\Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
QWEN3_PYTHON = r"Q:\models\.venv-llm\Scripts\python.exe"
QWEN3_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "portfolio", "qwen3_trader.py")

# Fake context for benchmarking
def make_context(ticker, asset_type="stock"):
    return {
        "ticker": ticker,
        "asset_type": asset_type,
        "price_usd": 100.0,
        "rsi": 55.0,
        "macd_hist": -2.0,
        "ema_bullish": False,
        "ema_gap_pct": 0.5,
        "bb_position": "inside",
        "fear_greed": 48,
        "fear_greed_class": "neutral",
        "news_sentiment": "neutral",
        "sentiment_confidence": 0.5,
        "volume_ratio": 0.9,
        "funding_rate": "N/A",
        "timeframe_summary": "Now=HOLD 12h=HOLD",
        "headlines": "",
    }


def benchmark_ministral(tickers):
    """Benchmark Ministral-3 via native binary."""
    if not os.path.exists(LLAMA_CLI) or not os.path.exists(MINISTRAL_MODEL):
        return None, "binary or model not found"

    results = []
    for ticker in tickers:
        prompt = f'[INST]Asset {ticker} at $100, RSI 55. JSON: {{"action":"BUY|SELL|HOLD","reasoning":"brief"}}[/INST]'
        pf = os.path.join(tempfile.gettempdir(), "bench_ministral.txt")
        with open(pf, "w") as f:
            f.write(prompt)

        t0 = time.time()
        proc = subprocess.run(
            [LLAMA_CLI, "-m", MINISTRAL_MODEL, "-ngl", "99", "-c", "4096",
             "-n", "60", "--temp", "0", "--no-display-prompt", "-f", pf],
            capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        results.append({"ticker": ticker, "time": elapsed, "ok": ok})
        print(f"  Ministral {ticker}: {elapsed:.1f}s {'OK' if ok else 'FAIL'}")

    return results, None


def benchmark_qwen3_single(tickers):
    """Benchmark Qwen3 in single-ticker mode (one subprocess per ticker)."""
    results = []
    for ticker in tickers[:3]:  # Only test 3 to avoid 5+ minute wait
        ctx = make_context(ticker)
        t0 = time.time()
        proc = subprocess.run(
            [QWEN3_PYTHON, QWEN3_SCRIPT],
            input=json.dumps(ctx), capture_output=True, text=True, timeout=120,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        results.append({"ticker": ticker, "time": elapsed, "ok": ok})
        print(f"  Qwen3-single {ticker}: {elapsed:.1f}s {'OK' if ok else 'FAIL'}")

    return results, None


def benchmark_qwen3_batch(tickers):
    """Benchmark Qwen3 in batch mode (one subprocess for all tickers)."""
    contexts = [make_context(t) for t in tickers]
    t0 = time.time()
    proc = subprocess.run(
        [QWEN3_PYTHON, QWEN3_SCRIPT],
        input=json.dumps(contexts), capture_output=True, text=True,
        timeout=30 + 20 * len(tickers),
    )
    elapsed = time.time() - t0
    ok = proc.returncode == 0

    count = 0
    if ok:
        try:
            results = json.loads(proc.stdout.strip())
            count = len(results) if isinstance(results, list) else 0
        except Exception:
            pass

    print(f"  Qwen3-batch {len(tickers)} tickers: {elapsed:.1f}s ({elapsed/len(tickers):.1f}s/ticker) {'OK' if ok else 'FAIL'} ({count} results)")
    return {"time": elapsed, "tickers": len(tickers), "ok": ok, "count": count}, None


def benchmark_chronos(tickers):
    """Benchmark Chronos-2 forecast."""
    try:
        from portfolio.forecast_signal import forecast_chronos, _get_chronos_pipeline
        import random
        random.seed(42)

        # Ensure pipeline is loaded (first call is slower)
        _get_chronos_pipeline()

        results = []
        for ticker in tickers[:5]:  # Test 5 tickers
            prices = [100 + random.uniform(-5, 5) for _ in range(168)]
            t0 = time.time()
            result = forecast_chronos(ticker, prices)
            elapsed = time.time() - t0
            ok = result is not None
            results.append({"ticker": ticker, "time": elapsed, "ok": ok})
            print(f"  Chronos-2 {ticker}: {elapsed:.1f}s {'OK' if ok else 'FAIL'}")

        return results, None
    except Exception as e:
        return None, str(e)


def main():
    crypto = ["BTC", "ETH"]
    all_tickers = ["BTC", "ETH", "XAU", "XAG", "PLTR", "NVDA", "AMD", "GOOGL",
                   "AMZN", "AAPL", "AVGO", "META", "MU", "SOUN", "SMCI",
                   "TSM", "TTWO", "VRT", "LMT", "MSTR"]

    print("=" * 60)
    print("GPU MODEL BENCHMARK — Simulating 1 loop cycle")
    print("=" * 60)

    total_start = time.time()

    # 1. Ministral-3 (crypto only)
    print(f"\n[1] Ministral-3-8B ({len(crypto)} crypto tickers)")
    m_results, m_err = benchmark_ministral(crypto)
    m_total = sum(r["time"] for r in m_results) if m_results else 0

    # 2. Wait for GPU to release Ministral VRAM
    print("\n  (waiting 2s for GPU VRAM release...)")
    time.sleep(2)

    # 3. Qwen3 batch mode (all 20 tickers — skip single, we know it's slow)
    q_single_avg = 0
    q_single_projected = 0
    print(f"\n[2] Qwen3-8B batch mode ({len(all_tickers)} tickers)")
    q_batch, _ = benchmark_qwen3_batch(all_tickers)
    q_batch_total = q_batch["time"] if q_batch else 0

    # Wait for GPU release
    print("\n  (waiting 2s for GPU VRAM release...)")
    time.sleep(2)

    # 3. Chronos-2 (sample 5 tickers)
    print(f"\n[3] Chronos-2 (5 tickers, extrapolate to {len(all_tickers)})")
    c_results, c_err = benchmark_chronos(all_tickers)
    c_avg = sum(r["time"] for r in c_results) / len(c_results) if c_results else 0
    c_projected = c_avg * len(all_tickers)

    total_elapsed = time.time() - total_start

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMinistral-3 ({len(crypto)} crypto): {m_total:.1f}s total ({m_total/len(crypto):.1f}s/ticker)")
    print(f"Qwen3 single mode (3 sampled): {q_single_avg:.1f}s/ticker → projected {len(all_tickers)}: {q_single_projected:.0f}s")
    print(f"Qwen3 batch mode ({len(all_tickers)} actual): {q_batch_total:.1f}s total ({q_batch_total/len(all_tickers):.1f}s/ticker)")
    print(f"Chronos-2 (5 sampled): {c_avg:.1f}s/ticker → projected {len(all_tickers)}: {c_projected:.0f}s")
    print(f"Batch savings: {q_single_projected - q_batch_total:.0f}s ({(1 - q_batch_total/q_single_projected)*100:.0f}%)")

    print(f"\n--- COLD START (all caches empty) ---")
    cold_single = m_total + q_single_projected + c_projected
    cold_batch = m_total + q_batch_total + c_projected
    print(f"Single mode: {cold_single:.0f}s ({cold_single/60:.1f} min)")
    print(f"Batch mode:  {cold_batch:.0f}s ({cold_batch/60:.1f} min)")

    print(f"\n--- STEADY STATE (1-2 cache misses/cycle) ---")
    # Average: 1.3 Qwen3 + 0.13 Ministral + 4 Chronos per 60s
    steady_qwen3 = 1.3 * (q_batch_total / len(all_tickers))
    steady_ministral = 0.13 * (m_total / len(crypto))
    steady_chronos = 4 * c_avg
    steady_total = steady_qwen3 + steady_ministral + steady_chronos
    print(f"Qwen3:     ~{steady_qwen3:.1f}s (1.3 misses × {q_batch_total/len(all_tickers):.1f}s)")
    print(f"Ministral: ~{steady_ministral:.1f}s (0.13 misses × {m_total/len(crypto):.1f}s)")
    print(f"Chronos:   ~{steady_chronos:.1f}s (4 misses × {c_avg:.1f}s)")
    print(f"TOTAL:     ~{steady_total:.1f}s / 60s budget {'✓ FITS' if steady_total < 50 else '✗ TOO SLOW'}")

    print(f"\nBenchmark took {total_elapsed:.0f}s total")


if __name__ == "__main__":
    main()
