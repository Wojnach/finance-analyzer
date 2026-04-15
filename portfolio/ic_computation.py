"""Information Coefficient (IC) computation for signal evaluation.

IC measures the rank correlation between signal predictions and actual returns,
providing a more nuanced quality metric than simple directional accuracy.
A signal can be 55% accurate but have IC of 0.15 if it's better at predicting
large moves, or 55% accurate with IC of 0.02 if it only catches noise.
"""

import logging
import math
import time
from pathlib import Path

from portfolio.file_utils import load_json, atomic_write_json, load_jsonl_tail
from portfolio.tickers import SIGNAL_NAMES, DISABLED_SIGNALS

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
IC_CACHE_FILE = DATA_DIR / "ic_cache.json"
IC_CACHE_TTL = 3600
SIGNAL_LOG_FILE = DATA_DIR / "signal_log.jsonl"

MIN_SAMPLES_FOR_IC = 30


def _spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation between two lists.

    Returns (rho, n) where rho is the correlation and n is the sample size.
    Uses the standard rank-correlation formula without scipy dependency.
    """
    n = len(x)
    if n < MIN_SAMPLES_FOR_IC:
        return 0.0, n

    def _rank(values):
        indexed = sorted(enumerate(values), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0, n

    return num / (den_x * den_y), n


def compute_signal_ic(horizon="1d", days=None, entries=None):
    """Compute Information Coefficient for each signal.

    IC = Spearman rank correlation between signal vote (+1 BUY, -1 SELL)
    and actual return (change_pct) over the given horizon.

    Returns:
        dict: {signal_name: {ic, ic_abs, samples, ic_buy, ic_sell, icir}}
    """
    if entries is None:
        entries = _load_entries(days=days)

    signal_data = {s: {"votes": [], "returns": []} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct")
            if change_pct is None:
                continue

            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                vote_num = 1.0 if vote == "BUY" else -1.0
                signal_data[sig_name]["votes"].append(vote_num)
                signal_data[sig_name]["returns"].append(change_pct)

    results = {}
    for sig_name in SIGNAL_NAMES:
        sd = signal_data[sig_name]
        votes = sd["votes"]
        returns = sd["returns"]
        n = len(votes)

        if n < MIN_SAMPLES_FOR_IC:
            results[sig_name] = {
                "ic": 0.0, "ic_abs": 0.0, "samples": n,
                "ic_buy": 0.0, "ic_sell": 0.0, "icir": 0.0,
            }
            continue

        ic, _ = _spearman_rank_correlation(votes, returns)

        buy_returns = [r for v, r in zip(votes, returns) if v > 0]
        sell_returns = [r for v, r in zip(votes, returns) if v < 0]
        ic_buy = sum(buy_returns) / len(buy_returns) if buy_returns else 0.0
        ic_sell = -sum(sell_returns) / len(sell_returns) if sell_returns else 0.0

        ic_list = _rolling_ic(votes, returns, window=50)
        if len(ic_list) >= 2:
            ic_mean = sum(ic_list) / len(ic_list)
            ic_std = math.sqrt(sum((x - ic_mean) ** 2 for x in ic_list) / len(ic_list))
            icir = ic_mean / ic_std if ic_std > 0.001 else 0.0
        else:
            icir = 0.0

        results[sig_name] = {
            "ic": round(ic, 4),
            "ic_abs": round(abs(ic), 4),
            "samples": n,
            "ic_buy": round(ic_buy, 4),
            "ic_sell": round(ic_sell, 4),
            "icir": round(icir, 4),
        }

    return results


def _rolling_ic(votes, returns, window=50):
    """Compute rolling IC values over a sliding window."""
    if len(votes) < window:
        return []
    ics = []
    for i in range(len(votes) - window + 1):
        v_win = votes[i:i + window]
        r_win = returns[i:i + window]
        ic, _ = _spearman_rank_correlation(v_win, r_win)
        ics.append(ic)
    return ics


def compute_signal_ic_per_ticker(horizon="1d", days=None, entries=None):
    """Compute IC for each signal per ticker.

    Returns:
        dict: {ticker: {signal_name: {ic, samples, ...}}}
    """
    if entries is None:
        entries = _load_entries(days=days)

    ticker_signal_data = {}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct")
            if change_pct is None:
                continue

            if ticker not in ticker_signal_data:
                ticker_signal_data[ticker] = {s: {"votes": [], "returns": []}
                                               for s in SIGNAL_NAMES}

            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                vote_num = 1.0 if vote == "BUY" else -1.0
                ticker_signal_data[ticker][sig_name]["votes"].append(vote_num)
                ticker_signal_data[ticker][sig_name]["returns"].append(change_pct)

    results = {}
    for ticker, sig_data in ticker_signal_data.items():
        results[ticker] = {}
        for sig_name in SIGNAL_NAMES:
            sd = sig_data[sig_name]
            votes = sd["votes"]
            returns = sd["returns"]
            n = len(votes)

            if n < MIN_SAMPLES_FOR_IC:
                results[ticker][sig_name] = {"ic": 0.0, "samples": n}
                continue

            ic, _ = _spearman_rank_correlation(votes, returns)
            results[ticker][sig_name] = {
                "ic": round(ic, 4),
                "ic_abs": round(abs(ic), 4),
                "samples": n,
            }

    return results


def _load_entries(days=None):
    """Load signal log entries, optionally limited to last N days."""
    from portfolio.accuracy_stats import load_entries
    entries = load_entries()
    if days and entries:
        import datetime
        cutoff = (datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(days=days)).isoformat()
        entries = [e for e in entries if e.get("ts", "") >= cutoff]
    return entries


def compute_and_cache_ic(horizon="1d"):
    """Compute IC data and write to cache file."""
    t0 = time.time()
    ic_global = compute_signal_ic(horizon=horizon)
    ic_per_ticker = compute_signal_ic_per_ticker(horizon=horizon)

    cache = {
        "time": time.time(),
        "horizon": horizon,
        "global": ic_global,
        "per_ticker": ic_per_ticker,
    }
    atomic_write_json(IC_CACHE_FILE, cache)
    dt = time.time() - t0
    logger.info("IC cache updated for %s in %.1fs", horizon, dt)
    return cache


def load_cached_ic(horizon="1d"):
    """Load cached IC data, returning None if stale."""
    cache = load_json(IC_CACHE_FILE)
    if cache is None:
        return None
    if time.time() - cache.get("time", 0) > IC_CACHE_TTL:
        return None
    if cache.get("horizon") != horizon:
        return None
    return cache


def get_signal_ic_ranking(horizon="1d", min_samples=30):
    """Get signals ranked by IC magnitude.

    Returns list of (signal_name, ic, samples) sorted by |IC| descending.
    """
    cache = load_cached_ic(horizon)
    if cache is None:
        cache = compute_and_cache_ic(horizon)
    ic_data = cache.get("global", {})
    ranked = []
    for sig_name, data in ic_data.items():
        if sig_name in DISABLED_SIGNALS:
            continue
        if data.get("samples", 0) < min_samples:
            continue
        ranked.append((sig_name, data.get("ic", 0.0), data.get("samples", 0)))
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    return ranked


def print_ic_report(horizon="1d"):
    """Print human-readable IC report."""
    ranked = get_signal_ic_ranking(horizon, min_samples=30)
    print(f"\n{'Signal':<24} {'IC':>8} {'|IC|':>8} {'Samples':>8}")
    print("-" * 52)
    for sig_name, ic, samples in ranked:
        tag = ""
        if ic < -0.02:
            tag = " [CONTRARIAN]"
        elif ic > 0.05:
            tag = " [STRONG]"
        print(f"{sig_name:<24} {ic:>8.4f} {abs(ic):>8.4f} {samples:>8}{tag}")
