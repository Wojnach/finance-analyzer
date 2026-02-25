"""Shared file I/O utilities."""
import json
import os
import tempfile
from collections import deque
from pathlib import Path


def atomic_write_json(path, data, indent=2):
    """Atomically write JSON data to a file using tempfile + os.replace.

    Ensures the file is never left in a partially-written state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path, default=None):
    """Load a JSON file. Returns *default* if missing or unparseable."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
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
            except json.JSONDecodeError:
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
