"""Tests for portfolio.avanza_order_lock cross-process order lock."""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from portfolio.avanza_order_lock import (
    DEFAULT_TIMEOUT_S,
    OrderLockBusyError,
    avanza_order_lock,
)


# Use a repo-local lock file for tests to avoid WSL/Windows tmp perms issues
# (same class of issue seen in other test files using tmp_path). The lock
# file itself never stores data; we just need a path.
_TEST_LOCK_DIR = Path(__file__).resolve().parent / "tmp_locks"
_TEST_LOCK_DIR.mkdir(exist_ok=True)


def _lock_path(request) -> Path:
    # per-test isolated lock path
    safe_name = request.node.name.replace("/", "_").replace("[", "_").replace("]", "_")
    return _TEST_LOCK_DIR / f"test_{safe_name}.lock"


@pytest.fixture
def lock_path(request):
    p = _lock_path(request)
    if p.exists():
        p.unlink()
    yield p
    if p.exists():
        try:
            p.unlink()
        except OSError:
            # windows may hold the file briefly; ignore
            pass


def test_acquire_release_happy_path(lock_path):
    with avanza_order_lock(lock_file=lock_path, op="test") as lock:
        assert lock.is_locked
    # after exit, lock should be released (file may still exist, but unlocked)


def test_two_sequential_acquires(lock_path):
    with avanza_order_lock(lock_file=lock_path, op="first"):
        pass
    with avanza_order_lock(lock_file=lock_path, op="second"):
        pass


def _hold_lock_in_child(lock_path_str: str, hold_s: float, ready_marker: str):
    """Subprocess body: acquire lock, signal ready via file, hold, release."""
    from portfolio.avanza_order_lock import avanza_order_lock as child_lock

    with child_lock(lock_file=Path(lock_path_str), op="child-hold"):
        Path(ready_marker).touch()
        time.sleep(hold_s)


def test_second_process_times_out_when_held(lock_path, tmp_path_factory):
    """When one process holds the lock, another trying to acquire hits the timeout."""
    # Use a filesystem-based ready-marker instead of tmp_path to dodge
    # the WSL/Windows tmp permissions problem seen elsewhere.
    ready_marker = lock_path.parent / f"{lock_path.stem}.ready"
    if ready_marker.exists():
        ready_marker.unlink()

    proc = multiprocessing.Process(
        target=_hold_lock_in_child,
        args=(str(lock_path), 2.0, str(ready_marker)),
    )
    proc.start()
    try:
        # Wait for child to acquire (up to 2s).
        deadline = time.time() + 2.0
        while not ready_marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert ready_marker.exists(), "child never signalled ready"

        # Now try to acquire with a short timeout — must fail.
        start = time.time()
        with pytest.raises(OrderLockBusyError):
            with avanza_order_lock(lock_file=lock_path, timeout_s=0.25, op="parent-fail"):
                pass
        elapsed = time.time() - start
        # We should have waited ~timeout_s, not the child's full hold.
        assert elapsed < 1.5, f"took {elapsed:.2f}s — should fail fast after timeout"
    finally:
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        if ready_marker.exists():
            ready_marker.unlink()


def test_second_process_acquires_after_first_releases(lock_path):
    """Once the holder exits, the waiter acquires promptly."""
    ready_marker = lock_path.parent / f"{lock_path.stem}.ready"
    if ready_marker.exists():
        ready_marker.unlink()

    proc = multiprocessing.Process(
        target=_hold_lock_in_child,
        args=(str(lock_path), 0.3, str(ready_marker)),  # short hold
    )
    proc.start()
    try:
        deadline = time.time() + 2.0
        while not ready_marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert ready_marker.exists()

        # Wait with a generous timeout — should succeed once child releases.
        start = time.time()
        with avanza_order_lock(lock_file=lock_path, timeout_s=2.0, op="parent-wait"):
            elapsed = time.time() - start
            assert elapsed > 0.05  # must have waited for child
            assert elapsed < 1.5  # child holds ~0.3s
    finally:
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        if ready_marker.exists():
            ready_marker.unlink()


def test_default_timeout_constant():
    assert DEFAULT_TIMEOUT_S == 2.0


def test_exception_body_still_releases(lock_path):
    """If the body raises, the lock is still released (context-manager guarantee)."""

    class _BadThing(Exception):
        pass

    with pytest.raises(_BadThing):
        with avanza_order_lock(lock_file=lock_path, op="body-raises"):
            raise _BadThing("boom")

    # Should be able to acquire again.
    with avanza_order_lock(lock_file=lock_path, op="after-raise"):
        pass
