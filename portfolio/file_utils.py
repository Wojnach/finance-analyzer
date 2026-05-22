"""Shared file I/O utilities."""
import json
import logging
import os
import tempfile
from collections import deque
from contextlib import contextmanager, suppress
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


def _resolve_write_path(path):
    """Resolve symlinks so os.replace() targets the real file, not the link."""
    path = Path(path)
    if path.is_symlink():
        path = Path(os.path.realpath(path))
    return path


def atomic_write_text(path, text, encoding="utf-8"):
    """Atomically write text to a file using tempfile + os.replace.

    Same safety guarantees as atomic_write_json: fsync before replace,
    no partial writes on crash.
    """
    path = _resolve_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_json(path, data, indent=2, ensure_ascii=True):
    """Atomically write JSON data to a file using tempfile + os.replace.

    Ensures the file is never left in a partially-written state.
    Fsyncs before replace to guarantee durability on power loss (H34).
    """
    path = _resolve_write_path(path)
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
        logger.warning("load_json: OS error reading %s, returning default", path.name)
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
    except OSError as e:
        logger.warning("load_jsonl: cannot open %s: %s", path.name, e)
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
            # 2026-05-04 codex P3-1 follow-up: peek the byte just before
            # the seek point. If it's a newline, the seek lands exactly
            # at a line boundary and the first decoded line is intact.
            # Without this check, a happy-coincidence boundary would
            # cost us one valid entry on every read.
            seek_on_boundary = False
            if offset > 0:
                f.seek(offset - 1)
                prior = f.read(1)
                seek_on_boundary = prior == b"\n"
            f.seek(offset)
            data = f.read()
        # Decode and split into lines
        text = data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        # Drop the first line only when we landed mid-line. When seek
        # lands on a newline boundary, the first decoded line is
        # complete and should be kept.
        if offset > 0 and lines and not seek_on_boundary:
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


@contextmanager
def jsonl_sidecar_lock(path):
    """Yield while holding an exclusive sidecar lock keyed off *path*.

    Same locking primitive that ``atomic_append_jsonl`` uses, exposed as
    a context manager so other code (notably
    ``portfolio.log_rotation.rotate_jsonl``) can serialize against
    in-flight appends. Lock file is ``<path.parent>/.<path.name>.lock``;
    a single-byte range is locked exclusively.

    Pattern rationale:

    * **Sidecar (not target):** locking the target file itself is racy
      when it is brand-new / size 0 — two first-writers can both hit
      the empty-file ``msvcrt.locking(fd, LK_LOCK, 1)`` failure path
      and interleave. A pre-seeded sidecar guarantees a lockable byte
      always exists.
    * **Windows + POSIX:** ``msvcrt.locking`` blocks on contention on
      Windows; ``fcntl.flock`` blocks on POSIX. Both release on close.

    Callers MUST keep *all* mutations of the target file inside the
    ``with`` block — read, write, fsync, rename. Appends that arrive
    between rotation's "read all lines" and ``os.replace`` would
    otherwise be silently discarded (the divergence behind the
    ``signal_log_reconciliation`` contract invariant escalations of
    2026-05-11).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    if not lock_path.exists():
        try:
            with open(lock_path, "ab") as lf:
                if lf.tell() == 0:
                    lf.write(b"\0")
        except OSError as exc:
            logger.warning("sidecar lock creation failed for %s: %s", path, exc)

    with open(lock_path, "rb+") as lock_f:
        lfd = lock_f.fileno()
        win_locked = False
        try:
            if _msvcrt is not None:
                os.lseek(lfd, 0, os.SEEK_SET)
                _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
                win_locked = True
            elif _fcntl is not None:
                _fcntl.flock(lfd, _fcntl.LOCK_EX)
            yield
        finally:
            if win_locked and _msvcrt is not None:
                try:
                    os.lseek(lfd, 0, os.SEEK_SET)
                    _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            # fcntl.flock releases automatically on close.


def atomic_append_jsonl(path, entry):
    """Append a single JSON entry to a JSONL file with atomic semantics
    across threads and processes.

    Now built on :func:`jsonl_sidecar_lock` so the lock contract is
    shared with ``log_rotation.rotate_jsonl``. Without that contract,
    rotation's read → write-tmp → ``os.replace`` could discard any
    append that landed between read and replace — exactly the
    divergence the ``signal_log_reconciliation`` contract invariant
    detects.

    This primitive is used by ~20 JSONL writers across the codebase
    (signal_log, claude_invocations, critical_errors, telegram_messages,
    accuracy_snapshots, etc.) so the fix eliminates torn-line risk
    system-wide. Unxfails
    ``tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl``.
    """
    path = Path(path)
    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with jsonl_sidecar_lock(path):
        with open(path, "ab") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


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


def count_jsonl_lines(path):
    """Return the number of non-blank lines in a JSONL file.

    Counts lines, not parsed entries — malformed JSON still counts as a
    line. Returns 0 for missing/empty/unreadable files.

    Used as a robust write-detection primitive: comparing line counts
    before/after a subprocess catches genuine appends while ignoring
    spurious mtime/replace events that don't actually add data.
    """
    path = Path(path)
    try:
        with open(path, "rb") as f:
            count = 0
            for raw in f:
                if raw.strip():
                    count += 1
            return count
    except (OSError, FileNotFoundError):
        return 0


def prune_jsonl(path, max_entries=5000):
    """Prune a JSONL file to keep only the most recent *max_entries*.

    Reads the file, keeps the tail, and atomically rewrites it.
    Skips malformed lines (e.g., from partial writes) during read.
    No-op if the file has fewer entries than *max_entries*.

    Returns the number of entries removed, or 0 if no pruning was needed.
    """
    path = Path(path)
    with jsonl_sidecar_lock(path):
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
                try:
                    json.loads(stripped)
                    lines.append(stripped)
                except json.JSONDecodeError:
                    logger.warning("prune_jsonl: skipping malformed line in %s", path.name)
        if len(lines) <= max_entries:
            return 0
        removed = len(lines) - max_entries
        keep = lines[-max_entries:]
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
