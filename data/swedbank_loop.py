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
# A signal pass is an OHLCV fetch plus indicator computation per instrument —
# orders of magnitude dearer than a quote sweep, and against the same shared
# Avanza session. Signals also move on a far slower clock than price, so
# recomputing every 60s would spend the session budget for no new information.
# Every 15th cycle (~15 min); in between the previous result is carried forward
# with its original signals_computed_at, so the UI shows real age instead of
# implying every value in the snapshot was computed at as_of.
SIGNAL_EVERY_N_CYCLES = 15
SINGLETON_LOCK_FILE = "data/swedbank_loop.lock"
HEARTBEAT_FILE = "data/swedbank_loop.heartbeat"
SNAPSHOT_FILE = "data/swedbank_snapshot.json"
EXIT_LOCK_CONFLICT = 11
EXIT_PEER_ACTIVE = 12

# The singleton lock is a PID in a local file, so it cannot see a loop on the
# other machine: herc's data/ is a different filesystem and a Deck PID means
# nothing on Windows. Both boxes now hold valid Avanza sessions for the SAME
# account, which the real-money metals loop also uses, so two concurrent loops
# would interleave requests on it.
#
# Only a persistent --loop is guarded. One-shot calls (--once, CLI, probes) stay
# free: herc is the testing bench, and one-shots against a live Deck loop were
# exercised repeatedly with no contention.
PEER_PRIMARY_HOST = "steamdeck"
PEER_URL_DEFAULT = "http://100.75.67.98:5055/api/swedbank"
PEER_HEARTBEAT_MAX_AGE_S = 180.0
PEER_TIMEOUT_S = 5.0

BACKOFF_MIN = 10
BACKOFF_MAX = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("swedbank_loop")

_stop = False
_cycle_n = 0


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


def _peer_config():
    cfg = (load_json("config.json", default=None) or {}).get("swedbank") or {}
    return (
        str(cfg.get("loop_primary_host") or PEER_PRIMARY_HOST),
        str(cfg.get("loop_peer_url") or PEER_URL_DEFAULT),
        cfg.get("loop_peer_guard", True),
    )


def is_primary_host(hostname=None, primary=None):
    """Primary never defers — it is the designated Avanza writer.

    Override by setting swedbank.loop_primary_host in config.json to the other
    machine, which inverts who yields without touching this file.
    """
    import socket

    name = (hostname or socket.gethostname() or "").strip().lower()
    want = (primary or _peer_config()[0]).strip().lower()
    return bool(want) and (name == want or name.startswith(want))


def peer_loop_alive(url=None, fetch_fn=None):
    """Whether the other machine's swedbank loop is running.

    Unreachable peer returns False on purpose: the Deck being off is exactly the
    case herc must still cover, so an unknown peer must not block us. That means
    a network partition can allow two writers — accepted, because the failure it
    prevents (silently no monitoring while the Deck is down) is the likelier one.
    """
    import json as _json
    import urllib.request

    target = url or _peer_config()[1]
    try:
        if fetch_fn is not None:
            payload = fetch_fn(target)
        else:
            token = (load_json("config.json", default=None) or {}).get(
                "dashboard_token"
            )
            req = urllib.request.Request(target)
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=PEER_TIMEOUT_S) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.info("peer check: %s unreachable (%s) — proceeding", target, exc)
        return False

    # /api/swedbank nests the heartbeat under "loop" and reports status "ok";
    # "heartbeat" is the shape of the raw data/swedbank_loop.heartbeat file, kept
    # so a direct file read works too.
    src = payload or {}
    hb = src.get("loop") or src.get("heartbeat") or {}
    status = str(hb.get("status") or "")
    if status not in ("starting", "running", "ok"):
        return False
    ts = hb.get("ts")
    if not ts:
        return False
    try:
        age = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.datetime.fromisoformat(ts)
        ).total_seconds()
    except (TypeError, ValueError):
        return False
    if age > PEER_HEARTBEAT_MAX_AGE_S:
        logger.info("peer check: heartbeat %.0fs stale — proceeding", age)
        return False
    logger.warning("peer check: peer loop alive (status=%s, %.0fs old)", status, age)
    return True


def peer_guard_blocks(ignore_peer=False):
    primary, _url, guard_enabled = _peer_config()
    if ignore_peer or os.environ.get("PF_SWEDBANK_IGNORE_PEER") == "1":
        logger.warning("peer guard overridden — starting anyway")
        return False
    if not guard_enabled:
        return False
    if is_primary_host(primary=primary):
        return False
    return peer_loop_alive()


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


def _carried_signals():
    """Previous signal pass, read from the snapshot we are about to replace.

    Read from the file rather than kept in memory so a loop restart mid-interval
    keeps showing the last real evaluation (with its true age) instead of a gap.
    """
    prev = load_json(SNAPSHOT_FILE, default=None) or {}
    return prev.get("signals"), prev.get("signals_computed_at")


def _evaluate_signals(keys):
    from portfolio.swedbank import signals as sigmod

    results = sigmod.evaluate_universe(keys=keys)
    sigmod.log_snapshot(results)
    return results


def cycle(with_signals=None):
    """One pass: load book, sweep prices, revalue, persist snapshot + cache."""
    global _cycle_n
    b = bookmod.load()
    cache = (load_json(CACHE_PATH, default=None) or {}).get("quotes") or {}

    from portfolio.fx_rates import fetch_usd_sek

    s = sweep(keys=b.keys_held, cache=cache, fx_fn=fetch_usd_sek)
    val = bookmod.revalue(b, s)

    if with_signals is None:
        with_signals = _cycle_n % SIGNAL_EVERY_N_CYCLES == 0
    _cycle_n += 1

    if with_signals:
        t0 = time.time()
        try:
            val["signals"] = _evaluate_signals(b.keys_held)
            val["signals_computed_at"] = _now_iso()
            n_err = sum(1 for v in val["signals"].values() if v.get("error"))
            logger.info(
                "signals: %d evaluated (%d error), %.1fs",
                len(val["signals"]),
                n_err,
                time.time() - t0,
            )
        except Exception as exc:
            # Pricing is the primary job. A broken signal pass must cost the
            # operator its own row and nothing else — never the valuation. The
            # error is stored rather than swallowed so the UI shows "no signal",
            # which is not the same thing as HOLD.
            logger.warning("signal evaluation failed: %s: %s", type(exc).__name__, exc)
            val["signals"] = {"error": f"{type(exc).__name__}: {exc}"}
            val["signals_computed_at"] = _now_iso()
    else:
        carried, carried_at = _carried_signals()
        if carried is not None:
            val["signals"] = carried
            val["signals_computed_at"] = carried_at

    # The loop writes the price cache and the snapshot. It NEVER writes the book
    # itself — that belongs to the operator's sync path, and two writers on a
    # file of real positions is how a book gets corrupted (premortem P2-8).
    atomic_write_json(CACHE_PATH, s.to_dict())
    atomic_write_json(SNAPSHOT_FILE, val)

    live = sum(1 for q in s.quotes.values() if not q.degraded)
    # No monetary values in logs — journald is not private and this repo is
    # public. Counts and timings only.
    logger.info(
        "cycle ok: %d/%d priced (%d live), %.2fs%s",
        len(s.quotes),
        len(b.keys_held),
        live,
        s.duration_s,
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
            # Never put monetary values or account labels in the heartbeat:
            # data/*.heartbeat is not covered by the data/swedbank_*.json
            # ignore rule, and this repo is public. Counts only.
            _heartbeat(
                "ok",
                {
                    "priced": len(val["accounts"])
                    and sum(len(a["holdings"]) for a in val["accounts"].values()),
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
    p.add_argument(
        "--ignore-peer",
        action="store_true",
        help="start --loop even if the primary machine's loop is running "
        "(also: PF_SWEDBANK_IGNORE_PEER=1, or swedbank.loop_peer_guard=false)",
    )
    args = p.parse_args(argv)
    if not (args.loop or args.once):
        p.error("specify --loop or --once")

    if args.once:
        # --once previously bypassed the lock, so running it while the service
        # was mid-cycle let the older cycle finish last and overwrite the newer
        # snapshot. Every writer takes the same lock.
        lock = acquire_singleton_lock()
        if not lock:
            logger.error("another swedbank loop holds the lock; exiting")
            return EXIT_LOCK_CONFLICT
        try:
            cycle()
        finally:
            release_singleton_lock(lock)
        return 0

    if peer_guard_blocks(ignore_peer=args.ignore_peer):
        logger.error(
            "the primary machine's swedbank loop is running — refusing to start a "
            "second writer against the shared Avanza session. Override with "
            "--ignore-peer if you really want both."
        )
        return EXIT_PEER_ACTIVE

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
