"""Swedbank monitoring loop — prices the book, writes a snapshot. Never trades.

Follows the satellite-loop pattern of data/oil_loop.py: singleton lock, heartbeat,
fixed cycle, crash backoff.

What makes this one different:

* It CANNOT trade. The portfolio.swedbank package imports no order module and
  issues no api_post/api_delete; tests/test_swedbank_no_trading.py enforces both
  by AST inspection. The operator executes every order by hand.
* It shares the Avanza session with the real-money metals loop, so it sweeps
  SEQUENTIALLY (1.5s for 26 instruments, a 2.5% duty cycle at 60s) and never
  invokes browser recovery. On session errors it degrades to last-good prices
  and backs off, leaving recovery to the loops that actually trade.
* It reads no Layer-1 state. pf-dataloop may be stopped for weeks; consuming
  health_state.json or signal_log.jsonl would silently serve frozen data.

Run:
    .venv/bin/python -u data/swedbank_loop.py --loop
    .venv/bin/python -u data/swedbank_loop.py --once
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import logging
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio.file_utils import atomic_write_json, load_json  # noqa: E402
from portfolio.swedbank import book as bookmod  # noqa: E402
from portfolio.swedbank.pricing import CACHE_PATH, sweep  # noqa: E402

CYCLE_SECONDS = 60
SINGLETON_LOCK_FILE = "data/swedbank_loop.lock"
HEARTBEAT_FILE = "data/swedbank_loop.heartbeat"
SNAPSHOT_FILE = "data/swedbank_snapshot.json"
EXIT_LOCK_CONFLICT = 11

BACKOFF_MIN = 10
BACKOFF_MAX = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("swedbank_loop")

_stop = False


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def acquire_singleton_lock(lock_path=SINGLETON_LOCK_FILE):
    Path(os.path.dirname(lock_path) or ".").mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock_path
        except FileExistsError:
            old_pid = None
            with contextlib.suppress(ValueError, OSError), open(lock_path) as f:
                old_pid = int(f.read().strip() or 0)
            if old_pid and _pid_alive(old_pid):
                logger.warning("singleton lock held by pid %d", old_pid)
                return None
            with contextlib.suppress(OSError):
                os.remove(lock_path)
        except OSError as exc:
            logger.warning("acquire_singleton_lock: %s", exc)
            return None
    return None


def release_singleton_lock(lock_path):
    if lock_path:
        with contextlib.suppress(OSError):
            os.remove(lock_path)


def _heartbeat(status, extra=None):
    payload = {
        "loop": "swedbank",
        "ts": _now_iso(),
        "pid": os.getpid(),
        "status": status,
    }
    if extra:
        payload.update(extra)
    with contextlib.suppress(Exception):
        atomic_write_json(HEARTBEAT_FILE, payload)


def cycle():
    """One pass: load book, sweep prices, revalue, persist snapshot + cache."""
    b = bookmod.load()
    cache = (load_json(CACHE_PATH, default=None) or {}).get("quotes") or {}

    from portfolio.fx_rates import fetch_usd_sek

    s = sweep(keys=b.keys_held, cache=cache, fx_fn=fetch_usd_sek)
    val = bookmod.revalue(b, s)

    # The loop writes the price cache and the snapshot. It NEVER writes the book
    # itself — that belongs to the operator's sync path, and two writers on a
    # file of real positions is how a book gets corrupted (premortem P2-8).
    atomic_write_json(CACHE_PATH, s.to_dict())
    atomic_write_json(SNAPSHOT_FILE, val)

    live = sum(1 for q in s.quotes.values() if not q.degraded)
    logger.info(
        "cycle ok: %d/%d priced (%d live), %.2fs, value=%.2f %s%s",
        len(s.quotes),
        len(b.keys_held),
        live,
        s.duration_s,
        val["total"]["total_value"],
        val["base_currency"],
        f", {len(val['unpriced'])} unpriced" if val["unpriced"] else "",
    )
    if s.errors:
        # Structured degradation log, deliberately NOT critical_errors.jsonl —
        # a monitoring loop must not burn the fix-agent backoff budget when the
        # shared Avanza session is merely expired (premortem P2-6).
        logger.warning(
            "swedbank_session_degraded: %s", dict(list(s.errors.items())[:5])
        )
    return val


def run_loop():
    global _stop
    backoff = BACKOFF_MIN
    while not _stop:
        started = time.time()
        try:
            val = cycle()
            backoff = BACKOFF_MIN
            _heartbeat(
                "ok",
                {
                    "total_value": val["total"]["total_value"],
                    "unpriced": len(val["unpriced"]),
                    "degraded": len(val["degraded"]),
                },
            )
        except Exception as exc:
            logger.exception("cycle failed: %s", exc)
            _heartbeat("error", {"error": f"{type(exc).__name__}: {exc}"})
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue
        elapsed = time.time() - started
        for _ in range(int(max(0.0, CYCLE_SECONDS - elapsed))):
            if _stop:
                break
            time.sleep(1)


def _handle_signal(signum, _frame):
    global _stop
    _stop = True
    logger.info("signal %s received, shutting down", signum)


def main(argv=None):
    p = argparse.ArgumentParser(description="Swedbank monitoring loop (never trades)")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)
    if not (args.loop or args.once):
        p.error("specify --loop or --once")

    if args.once:
        cycle()
        return 0

    lock = acquire_singleton_lock()
    if not lock:
        logger.error("another swedbank loop is running; exiting")
        return EXIT_LOCK_CONFLICT
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logger.info(
        "swedbank loop starting (cycle=%ds, pid=%d)", CYCLE_SECONDS, os.getpid()
    )
    _heartbeat("starting")
    try:
        run_loop()
    finally:
        release_singleton_lock(lock)
        _heartbeat("stopped")
        logger.info("swedbank loop stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
