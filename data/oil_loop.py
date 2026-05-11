"""Oil autonomous swing-trading loop — WTI.

Mirrors `data/crypto_loop.py` (which itself mirrors `data/metals_loop.py`)
for the oil subsystem. 60-second cycle:
  1. Acquire singleton lock (one process at a time).
  2. Fetch live WTI price via portfolio.price_source (CL=F → Binance FAPI
     real-time, with yfinance fallback).
  3. Read the latest Layer 1 signal snapshot from data/agent_summary*.json.
  4. Run OilSwingTrader.evaluate_and_execute(prices, signal_data).
  5. Sleep CYCLE_SECONDS, with embedded fast-tick monitor every 10s for
     sharp-dip alerts (Telegram).

Ships in DRY_RUN=True via oil_swing_config — the loop will log decisions
to data/oil_swing_decisions.jsonl but place no Avanza orders. Wiring
into a scheduled task should wait until live warrant discovery has run
(via the loop itself: it'll call oil_warrant_refresh.load_catalog_or_fetch
on first cycle when a Playwright page is available).

Run manually:
    .venv/Scripts/python.exe -u data/oil_loop.py --loop

One-shot (single cycle, no sleep):
    .venv/Scripts/python.exe -u data/oil_loop.py --once
"""
from __future__ import annotations

import argparse
import contextlib
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

from data import oil_swing_config as cfg
from data.oil_swing_trader import OilSwingTrader
from portfolio.file_utils import load_json

logger = logging.getLogger("oil_loop")

CYCLE_SECONDS = 60
SIGNAL_SUMMARY_FILE = "data/agent_summary_compact.json"
SINGLETON_LOCK_FILE = "data/oil_loop.lock"
HEARTBEAT_FILE = "data/oil_loop.heartbeat"
EXIT_LOCK_CONFLICT = 11        # Mirrors metals-loop.bat / crypto-loop.bat
                                # exit-code-11 contract: the .bat wrapper
                                # sees 11 and stops the restart loop instead
                                # of fork-bombing into the live instance.

# Live oil prices route through portfolio.price_source.fetch_klines —
# CL=F goes Binance FAPI (real-time) with yfinance fallback. Module-level
# constants kept for grep parity with crypto_loop / metals_loop.


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


# ---------------------------------------------------------------------------
# Singleton lock (matches metals_loop pattern)
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """Check if *pid* is running. Windows: tasklist; POSIX: kill(0).

    2026-05-01: subprocess imported at module top so the except clause
    below resolves on POSIX even when tasklist branch is skipped (codex
    review caught NameError on stale-lock cleanup in WSL/CI).
    """
    import subprocess  # noqa: PLC0415 — explicit local for the except clause
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def acquire_singleton_lock(lock_path: str = SINGLETON_LOCK_FILE) -> Any:
    """Try to grab a per-process lock. Returns lock handle or None on conflict.

    Uses O_CREAT|O_EXCL for atomic creation (no TOCTOU race). On stale lock
    (dead PID), removes and retries once.
    """
    Path(os.path.dirname(lock_path) or ".").mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock_path
        except FileExistsError:
            # Lock file exists — check if owner PID is still alive
            old_pid = 0
            with contextlib.suppress(ValueError, OSError), open(lock_path) as f:
                old_pid = int(f.read().strip() or "0")
            if old_pid > 0 and _pid_alive(old_pid):
                logger.warning("singleton lock held by pid %d", old_pid)
                return None
            # Stale lock — remove and retry once
            if attempt == 0:
                with contextlib.suppress(OSError):
                    os.remove(lock_path)
                continue
            return None
        except OSError as exc:
            logger.warning("acquire_singleton_lock: %s", exc)
            return None
    return None


def release_singleton_lock(lock_path: str | None) -> None:
    if not lock_path:
        return
    with contextlib.suppress(OSError):
        os.remove(lock_path)


# ---------------------------------------------------------------------------
# Live price fetch
# ---------------------------------------------------------------------------
def fetch_live_prices() -> dict[str, float]:
    """Fetch WTI/Brent live prices via the canonical price_source router.

    Returns dict mapping config tickers (OIL-USD) to USD spot price. The
    underlying yfinance symbol is taken from cfg.DATA_SOURCES. The router
    (`portfolio.price_source.fetch_klines`) prefers Binance FAPI for
    real-time CL=F (per oil_precompute.py 2026-04-14 routing), with
    yfinance fallback. Failures land as missing keys — the caller treats
    them as "skip this instrument".
    """
    out: dict[str, float] = {}
    try:
        from portfolio.price_source import fetch_klines
    except ImportError as exc:
        logger.warning("price_source unavailable: %s", exc)
        return out

    for ticker in cfg.INSTRUMENTS:
        sym = cfg.DATA_SOURCES.get(ticker, {}).get("yfinance_symbol")
        if not sym:
            continue
        try:
            # Pull recent 1m bars; last close is the freshest price the
            # router has. fetch_klines handles Binance/yfinance routing.
            hist = fetch_klines(sym, interval="1m", limit=5, period="1d")
            if hist is None or hist.empty:
                # Fall back to daily bar for cases where the 1m feed is
                # gapped (weekend, between sessions).
                hist = fetch_klines(sym, interval="1d", limit=2, period="5d")
            if hist is None or hist.empty:
                continue
            last = float(hist["close"].dropna().iloc[-1])
            if last > 0:
                out[ticker] = last
        except Exception as exc:  # noqa: BLE001
            logger.warning("price fetch %s: %s", ticker, exc)
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
                with contextlib.suppress(Exception):
                    notify(f"⚠️ {ticker} dip {change_pct:.2f}% "
                           f"(ref={ref_price:.2f} -> {price:.2f})")
            # Reset reference so we don't spam the same dip
            reference_prices[ticker] = {"price": price, "ts": now}
        elif (change_pct >= cfg.FAST_TICK_FLUSH_PCT
              and (now - ref_ts) <= cfg.FAST_TICK_FLUSH_WINDOW_SEC):
            alert = {"type": "flush", "ticker": ticker,
                     "ref_price": ref_price, "current": price,
                     "change_pct": round(change_pct, 3)}
            alerts[ticker] = alert
            if notify:
                with contextlib.suppress(Exception):
                    notify(f"🚀 {ticker} flush +{change_pct:.2f}% "
                           f"in {(now-ref_ts):.0f}s")
            reference_prices[ticker] = {"price": price, "ts": now}
    return alerts


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------
def run_one_cycle(trader: OilSwingTrader,
                  notify: Any = None) -> dict[str, Any]:
    # Refresh the oil signal feed that downstream consumers
    # (grid fisher in metals_loop) read each tick. Runs FIRST and
    # unconditionally so a fetch_live_prices() outage cannot leave a
    # stale BUY/SELL record on disk that the grid would treat as fresh
    # for up to 300s. compute_signal() returns a safe HOLD record on
    # its own fetch failures, so the file always reflects current
    # reality.
    with contextlib.suppress(Exception):
        from portfolio.oil_grid_signal import write_signal
        write_signal()

    prices = fetch_live_prices()
    signal_data = load_signal_snapshot()
    if not prices:
        return {"ok": False, "reason": "no live prices"}

    summary = trader.evaluate_and_execute(prices, signal_data)

    # Track value history for the dashboard
    with contextlib.suppress(Exception):
        from portfolio.file_utils import atomic_append_jsonl
        atomic_append_jsonl(cfg.VALUE_HISTORY_LOG, {
            "ts": _now_iso(),
            "cash_sek": trader.state.get("cash_sek"),
            "n_positions": len(trader.state.get("positions", {})),
            "prices": prices,
        })

    return {"ok": True, "summary": summary, "prices": prices,
            "n_positions": len(trader.state.get("positions", {}))}


def write_heartbeat(extra: dict | None = None) -> None:
    """Write a JSON heartbeat file each successful cycle.

    2026-05-04: thin shim over `portfolio.loop_health.write_heartbeat`.
    2026-05-04 codex P3-2 follow-up: ALL coercion + dict-handling
    happens inside try/except so a malformed `extra` cannot crash
    the loop (matches the pre-migration swallow contract).
    """
    try:
        e = dict(extra or {})
        try:
            cycle = int(e.pop("cycle", 0) or 0)
        except (TypeError, ValueError):
            cycle = 0
        ok = bool(e.pop("ok", True))
        try:
            n_positions = int(e.pop("n_positions", 0) or 0)
        except (TypeError, ValueError):
            n_positions = 0
        from portfolio.loop_health import write_heartbeat as _shared
        _shared(HEARTBEAT_FILE, cycle=cycle, ok=ok,
                n_positions=n_positions, extra=e or None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("heartbeat dispatch failed: %s", exc)


def run_loop(notify: Any = None) -> int:
    """Forever loop. Returns an int status code so main() can propagate it.

    Returns:
        0  on graceful shutdown (SIGINT/SIGTERM)
        EXIT_LOCK_CONFLICT (11) if another instance holds the singleton lock
    """
    lock = acquire_singleton_lock()
    if lock is None:
        logger.error("oil_loop: another instance is running. exiting.")
        return EXIT_LOCK_CONFLICT

    stop = {"flag": False}

    def _sigterm(_signum, _frame):
        stop["flag"] = True

    cycle_count = 0
    try:
        signal.signal(signal.SIGINT, _sigterm)
        with contextlib.suppress(AttributeError, ValueError):
            signal.signal(signal.SIGTERM, _sigterm)

        trader = OilSwingTrader(page=None, executor=None)
        logger.info("oil_loop start. DRY_RUN=%s instruments=%s",
                    cfg.DRY_RUN, cfg.INSTRUMENTS)
        if notify:
            with contextlib.suppress(Exception):
                notify(f"oil_loop start — DRY_RUN={cfg.DRY_RUN} "
                       f"instruments={','.join(cfg.INSTRUMENTS)}")

        fast_tick_refs: dict[str, dict[str, float]] = {}

        while not stop["flag"]:
            cycle_count += 1
            cycle_started = time.time()
            try:
                result = run_one_cycle(trader, notify=notify)
                logger.info("cycle ok=%s n_pos=%s", result.get("ok"),
                            result.get("n_positions"))
                write_heartbeat({
                    "cycle": cycle_count,
                    "ok": bool(result.get("ok")),
                    "n_positions": result.get("n_positions") or 0,
                })
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
        logger.info("oil_loop exited cleanly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oil swing-trading loop (WTI)")
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
            "n_signal_tickers": len(snap.get("per_ticker")
                                     or snap.get("tickers") or {}),
            "dry_run": cfg.DRY_RUN,
            "warrant_catalog_file": cfg.WARRANT_CATALOG_FILE,
            "state_file": cfg.STATE_FILE,
        }, indent=2))
        return 0

    if args.once:
        trader = OilSwingTrader(page=None, executor=None)
        result = run_one_cycle(trader)
        print(json.dumps({"ok": result.get("ok"),
                          "n_positions": result.get("n_positions"),
                          "actions": [a.get("type")
                                     for a in result.get("summary", {}).get("actions", [])]},
                         indent=2))
        return 0 if result.get("ok") else 1

    if args.loop:
        # Wire telegram notify if config + module both available; otherwise
        # run silently. Loop must never crash on missing config.
        notify = None
        try:
            from portfolio.file_utils import load_json
            from portfolio.telegram_notifications import send_telegram
            tg_cfg = load_json("config.json") or {}
            if tg_cfg.get("telegram", {}).get("token"):
                def notify(msg, _cfg=tg_cfg):  # noqa: E306
                    with contextlib.suppress(Exception):
                        send_telegram(msg, _cfg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("telegram notify wiring skipped: %s", exc)
        return run_loop(notify=notify)

    print("Specify --loop, --once, or --report", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
