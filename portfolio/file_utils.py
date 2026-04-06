"""Shared file I/O utilities."""
import json
import logging
import os
import tempfile
from collections import deque
from contextlib import suppress
from pathlib import Path

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
    """Append a single JSON entry to a JSONL file.

    Uses a write-then-append pattern so partial writes don't corrupt
    existing data. Flushes and fsyncs to ensure durability on crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
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
