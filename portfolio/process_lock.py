"""Cross-platform helpers for non-blocking singleton process locks."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import IO
import contextlib

try:
    import msvcrt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def acquire_lock_file(
    lock_path: str | Path,
    *,
    owner: str = "",
    metadata: dict | None = None,
) -> IO[str] | None:
    """Acquire a non-blocking file lock and return the open handle.

    Returns None if another process already holds the lock.
    """
    path = Path(lock_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    fh = path.open("a+", encoding="utf-8")
    try:
        _lock_file(fh)
    except OSError:
        fh.close()
        return None

    _write_lock_metadata(fh, owner=owner, metadata=metadata)
    return fh


def release_lock_file(fh: IO[str] | None) -> None:
    """Release a previously acquired lock handle."""
    if fh is None:
        return
    try:
        _unlock_file(fh)
    except OSError:
        pass
    finally:
        with contextlib.suppress(Exception):
            fh.close()


def _lock_file(fh: IO[str]) -> None:
    fh.seek(0)
    if msvcrt is not None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return
    if fcntl is not None:  # pragma: no branch - platform-specific
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(fh: IO[str]) -> None:
    fh.seek(0)
    if msvcrt is not None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    if fcntl is not None:  # pragma: no branch - platform-specific
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _write_lock_metadata(
    fh: IO[str],
    *,
    owner: str = "",
    metadata: dict | None = None,
) -> None:
    payload = {
        "pid": os.getpid(),
        "started": datetime.now(UTC).isoformat(),
    }
    if owner:
        payload["owner"] = owner
    if metadata:
        payload.update({str(k): v for k, v in metadata.items() if v is not None})

    try:
        fh.seek(0)
        fh.truncate()
        fh.write(" ".join(f"{key}={value}" for key, value in payload.items()) + "\n")
        fh.flush()
    except Exception:
        pass
