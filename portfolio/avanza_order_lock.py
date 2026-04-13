"""Cross-process advisory file lock guarding Avanza order placement.

Used across metals_loop, golddigger, fin_snipe_manager (and any future
Avanza-bound loop) to prevent two processes submitting overlapping orders
when they observe the same ``buying_power`` simultaneously. Without this,
the following sequence is possible:

    t=0.00  metals_loop reads buying_power=6000 SEK
    t=0.05  golddigger reads buying_power=6000 SEK
    t=0.10  metals_loop POSTs order for 5000 SEK
    t=0.15  golddigger POSTs order for 5000 SEK
    t=0.20  Avanza rejects one / or both fill and settlement overdraws

READ paths (``fetch_price``, ``fetch_positions``, ``buying_power``) are
NOT guarded — they're safe to run concurrently, and the whole point of
the resilience refactor is to keep those fast.

Usage:

    from portfolio.avanza_order_lock import avanza_order_lock

    with avanza_order_lock(op="place_order"):
        resp = api_post("/_api/trading-critical/rest/order/new", payload)

Design notes:

* ``filelock.FileLock`` is already in requirements (3.20.3 as of 2026-04-13).
* 2-second fail-fast default — long enough to ride through a normal order
  round-trip (~300ms), short enough that a hung peer doesn't block trading.
* Raises ``OrderLockBusyError`` on timeout so callers can log + retry next
  cycle instead of blocking the whole loop.
* Caller-provided ``op`` label threads through to log messages for
  diagnostics ("which loop hit the busy lock").
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import filelock

logger = logging.getLogger("portfolio.avanza_order_lock")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOCK_FILE = DATA_DIR / "avanza_order.lock"

DEFAULT_TIMEOUT_S = 2.0


class OrderLockBusyError(Exception):
    """Another process held the lock longer than the configured timeout."""


@contextmanager
def avanza_order_lock(
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    op: str = "order",
    lock_file: Path | None = None,
) -> Iterator[filelock.FileLock]:
    """Acquire the cross-process Avanza order lock for a short critical section.

    Fail-fast after ``timeout_s``. The lock is released automatically on exit.

    Args:
        timeout_s: Seconds to wait for the lock before raising
            :class:`OrderLockBusyError`. Default 2.0 — short enough to abort
            a stuck caller, long enough to ride through a normal order RTT.
        op: Short label for the operation, threaded into log messages.
        lock_file: Override the lock path (tests only). Defaults to
            ``data/avanza_order.lock``.

    Raises:
        OrderLockBusyError: If another process held the lock longer than
            ``timeout_s`` seconds.
    """
    target = Path(lock_file) if lock_file is not None else LOCK_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(target), timeout=timeout_s)
    try:
        lock.acquire()
    except filelock.Timeout as exc:
        logger.warning(
            "avanza_order_lock(%s): busy after %.1fs — another process holds the lock",
            op, timeout_s,
        )
        raise OrderLockBusyError(f"lock busy after {timeout_s}s (op={op})") from exc
    try:
        logger.debug("avanza_order_lock(%s): acquired", op)
        yield lock
    finally:
        try:
            lock.release()
            logger.debug("avanza_order_lock(%s): released", op)
        except Exception as exc:
            logger.warning("avanza_order_lock(%s): release failed: %s", op, exc)
