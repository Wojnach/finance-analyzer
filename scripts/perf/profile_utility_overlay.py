"""One-shot profile of the utility_overlay phase in signal_engine.

Mirrors signal_engine.py:3401-3491 exactly:
  1. load_cached_regime_accuracy(horizon)
  2. fall through to signal_accuracy_by_regime(horizon) on miss
     (we do NOT write the recomputed value back — see harness note below)
  3. signal_utility(horizon)
  4. iterate accuracy_data applying boost

Measures wall time per sub-step, then dumps cProfile cumtime top-30.

Pure read-only. We deliberately skip write_regime_accuracy_cache so the
harness can run against a live system without racing the loop's writer
under _accuracy_write_lock or overwriting the disk cache mid-cycle.

Usage:
  .venv/Scripts/python.exe -m scripts.perf.profile_utility_overlay
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from concurrent.futures import ThreadPoolExecutor

import portfolio.accuracy_stats as A


HORIZONS = ["3h", "4h", "12h", "1d"]
TICKERS = ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD"]


def overlay_one(horizon: str) -> tuple[float, float, float, float]:
    """Run the utility_overlay block for one horizon. Returns timings (ms).

    Returns: (load_cached_ms, sig_utility_ms, boost_loop_ms, total_ms)
    """
    t_total = time.monotonic()

    t0 = time.monotonic()
    regime_acc = A.load_cached_regime_accuracy(horizon)
    if not regime_acc:
        # Mirror the production path's fall-through compute, but DO NOT
        # write back to disk — see module docstring. The live loop owns
        # the disk cache; this harness is a pure observer.
        regime_acc = A.signal_accuracy_by_regime(horizon)
    t_load = (time.monotonic() - t0) * 1000

    accuracy_data: dict = {}
    if regime_acc:
        for sig_name, rdata in regime_acc.get("ranging", {}).items():
            if rdata.get("total", 0) >= 30:
                accuracy_data[sig_name] = rdata

    t0 = time.monotonic()
    utility_data = A.signal_utility(horizon)
    t_util = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    for sig_name in list(accuracy_data.keys()):
        u = utility_data.get(sig_name, {})
        u_score = u.get("avg_return", 0.0)
        samples = u.get("samples", 0)
        if samples >= 30 and u_score > 0:
            boost = min(1.0 + u_score, 1.5)
            boosted = min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)
            accuracy_data[sig_name] = {**accuracy_data[sig_name], "accuracy": boosted}
    t_boost = (time.monotonic() - t0) * 1000

    return t_load, t_util, t_boost, (time.monotonic() - t_total) * 1000


def warm_phase(label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'horizon':<6} {'load_cached(ms)':<18} {'sig_utility(ms)':<18} {'boost(ms)':<12} {'total(ms)':<12}")
    for h in HORIZONS:
        load_ms, util_ms, boost_ms, total_ms = overlay_one(h)
        print(f"{h:<6} {load_ms:<18.1f} {util_ms:<18.1f} {boost_ms:<12.2f} {total_ms:<12.1f}")


def parallel_phase(label: str) -> None:
    """Simulate 4 ticker threads running utility_overlay simultaneously."""
    print(f"\n=== {label} (4 threads, all on horizon=3h) ===")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda _: overlay_one("3h"), TICKERS))
    wall = (time.monotonic() - t0) * 1000
    for tk, (load_ms, util_ms, boost_ms, total_ms) in zip(TICKERS, results):
        print(f"  {tk:<10} load={load_ms:7.1f}ms  util={util_ms:7.1f}ms  boost={boost_ms:6.2f}ms  total={total_ms:7.1f}ms")
    print(f"  WALL CLOCK (4 threads): {wall:.1f}ms")


def cprofile_phase(label: str, n_calls: int = 20) -> None:
    print(f"\n=== {label} (cProfile, {n_calls} calls horizon=3h) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_calls):
        overlay_one("3h")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    print("# utility_overlay profile probe")
    print(f"# Python: {__import__('sys').version.split()[0]}")
    print(f"# accuracy_stats: {A.__file__}")
    print(f"# REGIME_ACCURACY_CACHE_FILE: {A.REGIME_ACCURACY_CACHE_FILE}")
    print(f"# ACCURACY_CACHE_TTL: {A.ACCURACY_CACHE_TTL}s")
    print(f"# _SIGNAL_UTILITY_CACHE_TTL: {A._SIGNAL_UTILITY_CACHE_TTL}s")

    # Cold pass — caches empty
    warm_phase("PASS 1 (cold caches)")
    # Hot pass — caches populated
    warm_phase("PASS 2 (warm caches)")
    # Parallel: 4 simultaneous threads, simulating ticker pool
    parallel_phase("PASS 3 (parallel, warm)")
    # Bust caches and re-run parallel cold
    A._signal_utility_cache.clear()
    parallel_phase("PASS 4 (parallel, signal_utility cache cleared)")
    # Profile
    cprofile_phase("PASS 5", n_calls=20)
