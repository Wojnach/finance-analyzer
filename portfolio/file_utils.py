"""Shared file I/O utilities."""
import json
import logging
import os
import tempfile
import time
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path

# 2026-06-11: msvcrt.locking(LK_LOCK) is NOT truly blocking — per CPython it
# retries once/second and raises OSError after 10 failed attempts. Long
# rotations (log_rotation.rotate_jsonl holds the same sidecar lock across a
# read+gzip-archive-merge+rewrite that can exceed 10s on signal_log.jsonl)
# would otherwise make any concurrent atomic_append_jsonl on Windows raise.
# We wrap acquisition in a bounded blocking retry with a real wall-clock
# budget. The budget is deliberately well under the 300s heartbeat staleness
# threshold / loop-contract watchdog restart window (premortem hook 5): a
# lock wait that long means the loop has already stalled, so we surface a
# clear error rather than block to the watchdog timeout.
_MSVCRT_LOCK_BUDGET_S = 30.0  # total wall-clock wait before giving up
_MSVCRT_LOCK_WARN_S = 5.0     # log a WARNING once a wait exceeds this (hook 5 visibility)
_MSVCRT_LOCK_RETRY_SLEEP_S = 0.25  # poll interval between non-blocking attempts

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


def atomic_write_json(path, data, indent=2, ensure_ascii=False):
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


# Reused decoder for the concatenated-object recovery path below.
_JSONL_DECODER = json.JSONDecoder()


def _decode_jsonl_line(line):
    """Decode one physical JSONL line into a list of objects.

    Normal case: one object → ``[obj]`` (fast path, unchanged semantics).

    Recovery case (2026-06-01): some historical rows in append-only journals
    (e.g. ``critical_errors.jsonl`` lines 127/212) carry TWO JSON objects
    concatenated on a single physical line with no separating newline — the
    legacy append-race that predates the fsync+newline hardening in
    ``atomic_append_jsonl``. The old readers did ``json.loads(line)``, hit
    "Extra data", and dropped the WHOLE line — silently losing real data
    (resolution rows pointing at criticals, so the dashboard over-reported
    unresolved errors). This recovers every object on the line via successive
    ``raw_decode`` instead. Strictly more permissive: a valid single-object
    line still returns exactly one object through the fast path.

    Returns ``[]`` when nothing decodes (genuinely garbage line). Whatever
    prefix decodes before an unrecoverable point is kept.
    """
    try:
        return [json.loads(line)]
    except json.JSONDecodeError:
        pass
    objs = []
    idx, n = 0, len(line)
    while idx < n:
        # Skip inter-object whitespace (covers "}{", "} {", "}\t{").
        while idx < n and line[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = _JSONL_DECODER.raw_decode(line, idx)
        except json.JSONDecodeError:
            break  # keep the prefix we already recovered, drop the rest
        objs.append(obj)
        idx = end
    return objs


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
            objs = _decode_jsonl_line(line)
            if not objs:
                logger.debug("Skipping malformed JSONL line in %s", path.name)
                continue
            container.extend(objs)
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
            entries.extend(_decode_jsonl_line(line))
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
    * **Windows + POSIX:** ``msvcrt.locking(LK_LOCK)`` is NOT truly
      blocking (it retries 10x1s then raises OSError), so on Windows we
      poll ``LK_NBLCK`` against a bounded ~30s deadline (see
      ``_MSVCRT_LOCK_BUDGET_S``); ``fcntl.flock`` blocks indefinitely on
      POSIX. Both release on close.

    Lock ordering (2026-06-11): this sidecar lock is the OUTER lock of
    the system's two-level scheme; file-level msvcrt *byte* locks are
    inner. ``atomic_append_jsonl``, ``atomic_write_jsonl``, ``prune_jsonl``
    and ``log_rotation.rotate_jsonl`` all take this sidecar lock around
    their whole read/write/replace sequence. It is NOT reentrant — a
    caller already inside one ``jsonl_sidecar_lock(path)`` must not call
    another primitive that re-enters it for the *same* path (would
    deadlock under the Windows byte-lock). The B6 outcome_tracker backfill
    holds this lock around its snapshot-verify and calls only
    lock-free I/O inside; keep that discipline.

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
                # 2026-06-11: bounded blocking retry via non-blocking LK_NBLCK
                # polling (LK_LOCK is itself a bounded 10x1s retry that raises,
                # so we can't rely on it to block). Loop on a real deadline,
                # emit a WARNING once a wait exceeds _MSVCRT_LOCK_WARN_S so a
                # long-held rotation lock is visible (premortem hook 5), and
                # raise a clear error naming the contended file on budget
                # exhaustion instead of the opaque CPython OSError.
                deadline = time.monotonic() + _MSVCRT_LOCK_BUDGET_S
                warned = False
                start = time.monotonic()
                while True:
                    os.lseek(lfd, 0, os.SEEK_SET)
                    try:
                        _msvcrt.locking(lfd, _msvcrt.LK_NBLCK, 1)
                        win_locked = True
                        break
                    except OSError:
                        now = time.monotonic()
                        waited = now - start
                        if not warned and waited >= _MSVCRT_LOCK_WARN_S:
                            warned = True
                            logger.warning(
                                "sidecar lock on %s contended for %.1fs "
                                "(budget %.0fs) — possible long rotation hold",
                                path.name, waited, _MSVCRT_LOCK_BUDGET_S,
                            )
                        if now >= deadline:
                            raise OSError(
                                f"sidecar lock acquisition timed out after "
                                f"{_MSVCRT_LOCK_BUDGET_S:.0f}s for {path} "
                                f"(.{path.name}.lock); a holder (e.g. log "
                                f"rotation) did not release in time"
                            ) from None
                        time.sleep(_MSVCRT_LOCK_RETRY_SLEEP_S)
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
        # Open read+append so we can inspect the final byte. O_APPEND still
        # forces every write() to EOF, so the seek below is read-only and does
        # not change where `data` lands.
        with open(path, "a+b") as f:
            # 2026-06-06: self-heal a dangling final line. If a prior writer
            # left the file without a trailing newline — a process killed
            # mid-line, or an ad-hoc agent `echo >>` / one-liner that bypassed
            # this primitive (CLAUDE.md instructs agents to "append a
            # resolution line" to critical_errors.jsonl) — a plain append would
            # butt two JSON objects onto one physical line (the legacy
            # critical_errors.jsonl 127/212 corruption: `...}{"ts":...}`).
            # Under the lock, guarantee a '\n' terminator before appending.
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")  # O_APPEND -> lands at EOF
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_jsonl(path, entries):
    """Atomically rewrite a JSONL file with the given entries.

    Uses tempfile + os.replace so the file is never left partially written.

    2026-06-11: now takes ``jsonl_sidecar_lock`` around the write+replace so
    a concurrent ``atomic_append_jsonl`` cannot land on the doomed pre-replace
    inode and be discarded by ``os.replace`` (same class of loss the 2026-05-11
    rotation fix closed, commit 3b623129). The sidecar lock is the OUTER lock
    and is NOT reentrant for the same path — verified that no live caller
    (fin_evolve, forecast_accuracy, signal_history,
    scripts/backfill_accuracy_snapshots) already holds it, so taking it here is
    safe. A future caller that DOES hold the lock around a read-modify-rewrite
    must call a ``_locked`` variant instead of this function.

    NOTE: this closes the write-side race only. A caller doing
    ``load_jsonl -> mutate -> atomic_write_jsonl`` still reads outside the lock,
    so an append landing between the read and this call is overwritten. Such
    callers should hold ``jsonl_sidecar_lock(path)`` around the whole
    read→mutate→write sequence (and then must not re-enter via this function).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_sidecar_lock(path):
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
        objs = _decode_jsonl_line(line)
        if not objs:
            continue
        entry = objs[-1]
        # 2026-06-11: the 4KB tail window can truncate a >4KB final line
        # mid-object; raw_decode then happily returns a bare scalar (e.g.
        # the string "context" before a quoted key). A non-dict entry is
        # never a real journal row, so skip it and keep scanning backward
        # to the previous complete line rather than handing callers
        # (loop_contract, _write_fishing_context) a str/number where a dict
        # is expected.
        if not isinstance(entry, dict):
            continue
        if field is not None:
            return entry.get(field)
        return entry
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
    No-op if the file has fewer entries than *max_entries*.

    2026-06-11: validate via ``_decode_jsonl_line`` (the 2026-06-06 recovery
    decoder) instead of bare ``json.loads``. The old bare-loads validation
    permanently dropped legacy concatenated-object lines (``...}{"ts":...}``)
    that ``last_jsonl_entry``/``load_jsonl`` recover — silently deleting real
    data (e.g. critical_errors resolution rows) on the next prune. We now keep
    every line the decoder can recover, AND heal the file by re-emitting each
    recovered object on its own physical line. ``removed`` counts *objects*,
    matching the entry-count semantics of ``max_entries``.

    Returns the number of entries removed, or 0 if no pruning was needed.
    """
    path = Path(path)
    with jsonl_sidecar_lock(path):
        # Each element is a JSON-serialized single object (recovered lines are
        # split into one entry each, healing legacy concatenated rows).
        objs = []
        try:
            f = open(path, encoding="utf-8")
        except FileNotFoundError:
            return 0
        with f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                decoded = _decode_jsonl_line(stripped)
                if not decoded:
                    logger.warning("prune_jsonl: skipping malformed line in %s", path.name)
                    continue
                objs.extend(json.dumps(o, ensure_ascii=False) for o in decoded)
        if len(objs) <= max_entries:
            return 0
        removed = len(objs) - max_entries
        keep = objs[-max_entries:]
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
