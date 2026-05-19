"""Entry point: python -m portfolio.mstr_loop

Starts the main loop. The PHASE is read from portfolio.mstr_loop.config
(can be overridden with MSTR_LOOP_PHASE env var). Default: shadow.

Singleton lock + atexit cleanup mirrors data/metals_loop.py and portfolio/main.py
(2026-05-19: previously this loop had no lock; duplicate-instance scenarios
silently corrupted mstr_loop_state.json and produced ghost processes that
held logs/mstr_loop_out.txt open, blocking subsequent restarts).
"""

from __future__ import annotations

import atexit
import contextlib
import datetime
import logging
import os
import sys

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import fcntl
except ImportError:
    fcntl = None

SINGLETON_LOCK_FILE = os.path.join("data", "mstr_loop.singleton.lock")
DUPLICATE_INSTANCE_EXIT_CODE = 11
_singleton_lock_fh = None


def _acquire_singleton_lock(lock_path: str = SINGLETON_LOCK_FILE) -> bool:
    """Acquire single-instance lock for the MSTR loop.

    Returns False if another instance already holds the lock. Caller should
    exit with DUPLICATE_INSTANCE_EXIT_CODE so the bat wrapper can stop
    looping (matches metals_loop + main_loop conventions).
    """
    global _singleton_lock_fh
    if _singleton_lock_fh is not None:
        return True
    if msvcrt is None and fcntl is None:
        logging.warning("mstr_loop: no file locking available — proceeding without singleton guard")
        return True

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fh.seek(0)
        if msvcrt is not None:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False

    with contextlib.suppress(Exception):
        fh.seek(0)
        fh.truncate()
        fh.write(
            f"pid={os.getpid()} started={datetime.datetime.now(datetime.UTC).isoformat()}\n"
        )
        fh.flush()

    _singleton_lock_fh = fh
    return True


def _release_singleton_lock() -> None:
    """Release single-instance lock if held — registered with atexit."""
    global _singleton_lock_fh
    if _singleton_lock_fh is None:
        return
    try:
        if msvcrt is not None:
            _singleton_lock_fh.seek(0)
            msvcrt.locking(_singleton_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:
            fcntl.flock(_singleton_lock_fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        with contextlib.suppress(Exception):
            _singleton_lock_fh.close()
        _singleton_lock_fh = None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    if not _acquire_singleton_lock():
        logging.info(
            "mstr_loop: another instance already holds %s — exiting (code %d)",
            SINGLETON_LOCK_FILE,
            DUPLICATE_INSTANCE_EXIT_CODE,
        )
        return DUPLICATE_INSTANCE_EXIT_CODE
    atexit.register(_release_singleton_lock)

    from portfolio.mstr_loop import loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("mstr_loop: stopped by KeyboardInterrupt")
        return 0
    except Exception:
        logging.exception("mstr_loop: fatal error — exiting")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
