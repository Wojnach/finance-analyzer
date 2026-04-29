"""Crypto autonomous swing-trading loop — BTC + ETH.

Mirrors `data/metals_loop.py` for the crypto subsystem. 60-second cycle:
  1. Acquire singleton lock (one process at a time).
  2. Fetch live BTC + ETH prices from Binance.
  3. Read the latest Layer 1 signal snapshot from data/agent_summary*.json.
  4. Run CryptoSwingTrader.evaluate_and_execute(prices, signal_data).
  5. Sleep CYCLE_SECONDS, with embedded fast-tick monitor every 10s for
     sharp-dip alerts (Telegram).

Ships in DRY_RUN=True via crypto_swing_config — the loop will log decisions
to data/crypto_swing_decisions.jsonl but place no Avanza orders. Wiring
into a scheduled task should wait until live warrant discovery has run
(via the loop itself: it'll call crypto_warrant_refresh.load_catalog_or_fetch
on first cycle when a Playwright page is available).

Run manually:
    .venv/Scripts/python.exe -u data/crypto_loop.py --loop

One-shot (single cycle, no sleep):
    .venv/Scripts/python.exe -u data/crypto_loop.py --once
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Ensure we can import portfolio.* and data.* when run directly
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from data import crypto_swing_config as cfg
from data.crypto_swing_trader import CryptoSwingTrader
from portfolio.file_utils import load_json

logger = logging.getLogger("crypto_loop")

CYCLE_SECONDS = 60
SIGNAL_SUMMARY_FILE = "data/agent_summary_compact.json"
SINGLETON_LOCK_FILE = "data/crypto_loop.lock"

# Binance public endpoints (no auth needed for prices)
_BINANCE_24HR = "https://api.binance.com/api/v3/ticker/24hr"


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


# ---------------------------------------------------------------------------
# Singleton lock (matches metals_loop pattern)
# ---------------------------------------------------------------------------
def acquire_singleton_lock(lock_path: str = SINGLETON_LOCK_FILE) -> Any:
    """Try to grab a per-process lock. Returns lock handle or None on conflict.

    Mirrors metals_loop.acquire_singleton_lock — file-based on Windows since
    fcntl/flock isn't available, falls back to PID-stale-check.
    """
    Path(os.path.dirname(lock_path) or ".").mkdir(parents=True, exist_ok=True)
    if os.path.exists(lock_path):
        try:
            with open(lock_path) as f:
                old_pid = int(f.read().strip() or "0")
        except (ValueError, OSError):
            old_pid = 0

        if old_pid > 0:
            # Check if pid still exists
            try:
                if os.name == "nt":
                    import subprocess
                    out = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {old_pid}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    alive = str(old_pid) in out.stdout
                else:
                    os.kill(old_pid, 0)
                    alive = True
            except (OSError, subprocess.SubprocessError):
                alive = False
            if alive:
                logger.warning("singleton lock held by pid %d", old_pid)
                return None

    try:
        with open(lock_path, "w") as f:
            f.write(str(os.getpid()))
        return lock_path
    except OSError as exc:
        logger.warning("acquire_singleton_lock: %s", exc)
        return None


def release_singleton_lock(lock_path: str | None) -> None:
    if not lock_path:
        return
    try:
        os.remove(lock_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Live price fetch
# ---------------------------------------------------------------------------
def fetch_live_prices() -> dict[str, float]:
    """Fetch BTC + ETH live prices from Binance public API.

    Returns dict mapping config tickers (BTC-USD, ETH-USD) to USD spot price.
    Failures land as missing keys — the caller treats them as "skip this
    instrument".
    """
    out: dict[str, float] = {}
    try:
        import requests
        for ticker in cfg.INSTRUMENTS:
            sym = cfg.DATA_SOURCES.get(ticker, {}).get("binance_symbol")
            if not sym:
                continue
            try:
                r = requests.get(_BINANCE_24HR, params={"symbol": sym}, timeout=10)
                if r.status_code == 200:
                    out[ticker] = float(r.json().get("lastPrice", 0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("price fetch %s: %s", ticker, exc)
    except ImportError:
        logger.warning("requests not available — no live prices")
    return out


# ---------------------------------------------------------------------------
# Signal data
# ---------------------------------------------------------------------------
def load_signal_snapshot() -> dict[str, Any]:
    """Read the latest Layer 1 signal snapshot.

    Tries `data/agent_summary_compact.json` first (small, fast); falls back
    to `data/agent_summary.json` if compact is missing or empty.
    """
    snap = load_json(SIGNAL_SUMMARY_FILE) or {}
    if not snap:
        snap = load_json("data/agent_summary.json") or {}
    return snap


# ---------------------------------------------------------------------------
# Fast-tick monitor (parallels metals _silver_fast_tick)
# ---------------------------------------------------------------------------
def fast_tick_check(reference_prices: dict[str, dict[str, float]],
                    notify: Any = None) -> dict[str, Any]:
    """Sub-poll for sharp-dip / velocity-flush alerts.

    Args:
        reference_prices: Per-ticker dict of {"price": ref, "ts": epoch}.
            Mutated in place: callers persist these between fast-tick calls
            so the reference window shifts on every full cycle.
        notify: optional callable(msg) for Telegram or stdout.

    Returns:
        Dict of alerts triggered this tick.
    """
    alerts: dict[str, Any] = {}
    live = fetch_live_prices()
    now = time.time()
    for ticker, price in live.items():
        ref = reference_prices.get(ticker)
        if not ref or "price" not in ref:
            reference_prices[ticker] = {"price": price, "ts": now}
            continue
        ref_price = ref["price"]
        ref_ts = ref.get("ts", now)
        change_pct = (price / ref_price - 1.0) * 100.0
        if change_pct <= cfg.FAST_TICK_DIP_ALERT_PCT:
            alert = {"type": "dip", "ticker": ticker,
                     "ref_price": ref_price, "current": price,
                     "change_pct": round(change_pct, 3)}
            alerts[ticker] = alert
            if notify:
                try:
                    notify(f"⚠️ {ticker} dip {change_pct:.2f}% "
                           f"(ref={ref_price:.2f} -> {price:.2f})")
                except Exception:  # noqa: BLE001
                    pass
            # Reset reference so we don't spam the same dip
            reference_prices[ticker] = {"price": price, "ts": now}
        elif (change_pct >= cfg.FAST_TICK_FLUSH_PCT
              and (now - ref_ts) <= cfg.FAST_TICK_FLUSH_WINDOW_SEC):
            alert = {"type": "flush", "ticker": ticker,
                     "ref_price": ref_price, "current": price,
                     "change_pct": round(change_pct, 3)}
            alerts[ticker] = alert
            if notify:
                try:
                    notify(f"🚀 {ticker} flush +{change_pct:.2f}% "
                           f"in {(now-ref_ts):.0f}s")
                except Exception:  # noqa: BLE001
                    pass
            reference_prices[ticker] = {"price": price, "ts": now}
    return alerts


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------
def run_one_cycle(trader: CryptoSwingTrader,
                  notify: Any = None) -> dict[str, Any]:
    prices = fetch_live_prices()
    signal_data = load_signal_snapshot()
    if not prices:
        return {"ok": False, "reason": "no live prices"}

    summary = trader.evaluate_and_execute(prices, signal_data)

    # Track value history for the dashboard
    try:
        from portfolio.file_utils import atomic_append_jsonl
        atomic_append_jsonl(cfg.VALUE_HISTORY_LOG, {
            "ts": _now_iso(),
            "cash_sek": trader.state.get("cash_sek"),
            "n_positions": len(trader.state.get("positions", {})),
            "prices": prices,
        })
    except Exception:  # noqa: BLE001
        pass

    return {"ok": True, "summary": summary, "prices": prices,
            "n_positions": len(trader.state.get("positions", {}))}


def run_loop(notify: Any = None) -> None:
    """Forever loop. Catches SIGINT/SIGTERM for graceful shutdown."""
    lock = acquire_singleton_lock()
    if lock is None:
        logger.error("crypto_loop: another instance is running. exiting.")
        return

    stop = {"flag": False}

    def _sigterm(_signum, _frame):
        stop["flag"] = True

    try:
        signal.signal(signal.SIGINT, _sigterm)
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except (AttributeError, ValueError):
            pass

        trader = CryptoSwingTrader(page=None, executor=None)
        logger.info("crypto_loop start. DRY_RUN=%s instruments=%s",
                    cfg.DRY_RUN, cfg.INSTRUMENTS)

        fast_tick_refs: dict[str, dict[str, float]] = {}

        while not stop["flag"]:
            cycle_started = time.time()
            try:
                result = run_one_cycle(trader, notify=notify)
                logger.info("cycle ok=%s n_pos=%s", result.get("ok"),
                            result.get("n_positions"))
            except Exception as exc:  # noqa: BLE001
                logger.exception("cycle error: %s", exc)

            # Fast-tick monitor — fill the remainder of the cycle window
            elapsed = time.time() - cycle_started
            remaining = max(0, CYCLE_SECONDS - elapsed)
            ticks = int(remaining // cfg.FAST_TICK_INTERVAL_SEC)
            for _ in range(ticks):
                if stop["flag"]:
                    break
                try:
                    fast_tick_check(fast_tick_refs, notify=notify)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fast_tick error: %s", exc)
                time.sleep(cfg.FAST_TICK_INTERVAL_SEC)
            # Mop up sub-second drift
            drift = CYCLE_SECONDS - (time.time() - cycle_started)
            if drift > 0:
                time.sleep(drift)
    finally:
        release_singleton_lock(lock)
        logger.info("crypto_loop exited cleanly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crypto swing-trading loop (BTC + ETH)")
    p.add_argument("--loop", action="store_true",
                   help="Run forever (60s cycle).")
    p.add_argument("--once", action="store_true",
                   help="Run a single cycle then exit.")
    p.add_argument("--report", action="store_true",
                   help="Print one-shot status without trading.")
    p.add_argument("--debug", action="store_true",
                   help="Verbose logging.")
    return p.parse_args()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    args = _parse_args()
    _setup_logging(args.debug)

    if args.report:
        # Just print snapshot of what would happen
        prices = fetch_live_prices()
        snap = load_signal_snapshot()
        print(json.dumps({
            "prices": prices,
            "n_signal_tickers": len((snap.get("per_ticker")
                                     or snap.get("tickers") or {})),
            "dry_run": cfg.DRY_RUN,
            "warrant_catalog_file": cfg.WARRANT_CATALOG_FILE,
            "state_file": cfg.STATE_FILE,
        }, indent=2))
        return 0

    if args.once:
        trader = CryptoSwingTrader(page=None, executor=None)
        result = run_one_cycle(trader)
        print(json.dumps({"ok": result.get("ok"),
                          "n_positions": result.get("n_positions"),
                          "actions": [a.get("type")
                                     for a in result.get("summary", {}).get("actions", [])]},
                         indent=2))
        return 0 if result.get("ok") else 1

    if args.loop:
        run_loop()
        return 0

    print("Specify --loop, --once, or --report", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
