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
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
        os.replace(tmp, str(path))
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def load_json(path, default=None):
    """Load a JSON file. Returns *default* if missing or unparseable.

    Uses try/except instead of exists() check to avoid TOCTOU race.
    """
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, ValueError):
        return default


def load_jsonl(path, limit=None):
    """Load entries from a JSONL file.

    Args:
        path: Path to the .jsonl file.
        limit: If set, keep only the *last* N entries (uses a deque).

    Returns:
        list of parsed dicts. Empty list if file missing or empty.
    """
    path = Path(path)
    if not path.exists():
        return []
    container = deque(maxlen=limit) if limit else []
    with open(path, encoding="utf-8") as f:
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


def prune_jsonl(path, max_entries=5000):
    """Prune a JSONL file to keep only the most recent *max_entries*.

    Reads the file, keeps the tail, and atomically rewrites it.
    Skips malformed lines (e.g., from partial writes) during read.
    No-op if the file has fewer entries than *max_entries*.

    Returns the number of entries removed, or 0 if no pruning was needed.
    """
    path = Path(path)
    if not path.exists():
        return 0
    lines = []
    with open(path, encoding="utf-8") as f:
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
