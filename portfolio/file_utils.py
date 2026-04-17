"""Shared file I/O utilities."""
import json
import logging
import os
import tempfile
from collections import deque
from contextlib import suppress
from pathlib import Path

# Cross-platform file-locking primitives for `atomic_append_jsonl`.
# Same pattern as `portfolio/process_lock.py`.
try:
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - non-Windows
    _msvcrt = None  # type: ignore[assignment]
try:
    import fcntl as _fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]

logger = logging.getLogger("portfolio.file_utils")


def atomic_write_json(path, data, indent=2, ensure_ascii=True):
    """Atomically write JSON data to a file using tempfile + os.replace.

    Ensures the file is never left in a partially-written state.
    Fsyncs before replace to guarantee durability on power loss (H34).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def load_json(path, default=None):
    """Load a JSON file. Returns *default* if missing or unparseable.

    Uses try/except instead of exists() check to avoid TOCTOU race.
    Handles OSError (permission denied, locked files) gracefully on Windows.
    Logs WARNING on corrupt JSON so corruption is observable (H35).
    """
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except OSError:
        # BUG-139: PermissionError (file locked by antivirus/another process)
        # and other OS-level errors should degrade gracefully like missing files.
        logger.debug("load_json: OS error reading %s, returning default", path.name)
        return default
    except (json.JSONDecodeError, ValueError):
        # H35: Log corruption so it's observable — silent defaults hide data loss.
        logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
        return default


def require_json(path):
    """Load a JSON file, raising on corruption or missing file.

    Unlike load_json(), this function does NOT silently return defaults.
    Use for critical files where corruption must be surfaced (H35).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        OSError: If the file cannot be read.
    """
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path, limit=None):
    """Load entries from a JSONL file.

    Args:
        path: Path to the .jsonl file.
        limit: If set, keep only the *last* N entries (uses a deque).

    Returns:
        list of parsed dicts. Empty list if file missing or empty.
    """
    path = Path(path)
    container = deque(maxlen=limit) if limit else []
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return []
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                container.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.debug("Skipping malformed JSONL line in %s: %s", path.name, str(e)[:100])
                continue
    return list(container)


def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
    """Load the last N entries from a JSONL file by reading from the end.

    Much more efficient than load_jsonl(limit=N) for large files because
    it only reads the last `tail_bytes` bytes instead of the entire file.

    Args:
        path: Path to the .jsonl file.
        max_entries: Maximum entries to return.
        tail_bytes: How many bytes to read from the end of the file.
            Default 512KB ≈ ~1000 typical entries.

    Returns:
        list of parsed dicts (chronological order). Empty list if missing.
    """
    path = Path(path)
    try:
        file_size = path.stat().st_size
    except (FileNotFoundError, OSError):
        return []
    if file_size == 0:
        return []

    entries = []
    try:
        with open(path, "rb") as f:
            # Seek to near end of file
            offset = max(0, file_size - tail_bytes)
            f.seek(offset)
            data = f.read()
        # Decode and split into lines
        text = data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        # If we seeked mid-file, the first line is likely truncated — skip it
        if offset > 0 and lines:
            lines = lines[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
        return []

    # Return last max_entries in chronological order
    if len(entries) > max_entries:
        entries = entries[-max_entries:]
    return entries


def atomic_append_jsonl(path, entry):
    """Append a single JSON entry to a JSONL file with atomic semantics
    across threads and processes.

    Implementation: binary-append (``"ab"``) + an exclusive file-range
    lock held for the duration of the ``write + flush + fsync``
    sequence. Windows CRT does not guarantee ``O_APPEND`` atomicity (unlike
    POSIX), so without the lock heavy thread contention can produce
    torn lines (head bytes lost, tail bytes survive). The locking fix
    unxfails ``tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl``
    (2026-04-13 regression).

    The same primitive is used by ~20 JSONL writers across the codebase
    (signal_log, claude_invocations, critical_errors, telegram_messages,
    accuracy_snapshots, etc.) so this fix eliminates torn-line risk
    system-wide.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "ab") as f:
        fd = f.fileno()
        locked_win = False
        try:
            if _msvcrt is not None:
                # Windows: lock byte 0 (LK_LOCK is blocking). Writes in
                # append mode always go to EOF regardless of the file
                # pointer, so seeking to 0 just to lock is safe.
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
                    locked_win = True
                except OSError:
                    # Locking a non-existent byte 0 on a brand-new empty
                    # file can fail; extend by one byte then retry once.
                    f.write(b"")
                    f.flush()
                    os.lseek(fd, 0, os.SEEK_SET)
                    _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
                    locked_win = True
            elif _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            f.write(data)
            f.flush()
            os.fsync(fd)
        finally:
            if locked_win and _msvcrt is not None:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            # fcntl.flock releases automatically on close.


def atomic_write_jsonl(path, entries):
    """Atomically rewrite a JSONL file with the given entries.

    Uses tempfile + os.replace so the file is never left partially written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def last_jsonl_entry(path, field=None):
    """Return the last parsed JSON entry from a JSONL file (efficient tail read).

    Reads only the last 4KB of the file instead of scanning the entire file.

    Args:
        path: Path to the JSONL file.
        field: If set, return only this field's value from the last entry.

    Returns:
        The last entry (dict) or the value of *field*, or None if file is
        missing/empty/unreadable.
    """
    path = Path(path)
    try:
        file_size = path.stat().st_size
    except (OSError, FileNotFoundError):
        return None
    if file_size == 0:
        return None
    read_size = min(file_size, 4096)
    try:
        with open(path, "rb") as f:
            f.seek(max(0, file_size - read_size))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if field is not None:
                return entry.get(field)
            return entry
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


def prune_jsonl(path, max_entries=5000):
    """Prune a JSONL file to keep only the most recent *max_entries*.

    Reads the file, keeps the tail, and atomically rewrites it.
    Skips malformed lines (e.g., from partial writes) during read.
    No-op if the file has fewer entries than *max_entries*.

    Returns the number of entries removed, or 0 if no pruning was needed.
    """
    path = Path(path)
    lines = []
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return 0
    with f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Validate JSON to avoid preserving corrupt partial lines
            try:
                json.loads(stripped)
                lines.append(stripped)
            except json.JSONDecodeError:
                logger.warning("prune_jsonl: skipping malformed line in %s", path.name)
    if len(lines) <= max_entries:
        return 0
    removed = len(lines) - max_entries
    keep = lines[-max_entries:]
    # Atomic rewrite via tempfile
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in keep:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise
    logger.info("Pruned %s: removed %d entries, kept %d", path.name, removed, max_entries)
    return removed
