# 2026-04-13 — `atomic_append_jsonl` torn lines under contention

**Status:** OPEN — codebase-wide bug discovered during fix-agent-dispatcher work.

## Symptom

`portfolio.file_utils.atomic_append_jsonl` produces TORN JSON lines under
sufficient thread contention on Windows. A line written like
`{"ts":"...","i":42}` may end up on disk as `42}` only — the head bytes
are dropped while the tail survives. Reproduced via
`tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl`
(currently xfail, strict=False).

## Why this matters

The same primitive is used by ~20+ JSONL writers across the codebase:

```
$ rg -l "atomic_append_jsonl" portfolio/
```

Includes high-stakes journals: `claude_invocations.jsonl`,
`signal_log.jsonl`, `metals_signal_log.jsonl`, `forecast_predictions.jsonl`,
`layer2_journal.jsonl`, etc. A torn line breaks any consumer that does
`json.loads(line)` without try/except — and most don't.

## Root cause

`open(path, "a", encoding="utf-8")` in Python uses text-mode buffered I/O.
On Windows, the `O_APPEND` semantic is provided by the C runtime, which
does NOT atomically combine seek-to-end + write across threads or
processes. Under contention, two writers can:

1. T1 seeks to end (offset N).
2. T2 seeks to end (offset N — same).
3. T1 writes 100 bytes; file extends to N+100.
4. T2 writes 100 bytes at offset N — overwriting T1's data.
5. Final file: T2's 100 bytes replace what should have been T1's, then
   T1's `flush` writes some leftover buffered content past N+100.

Net effect: torn line.

## Fix (proposed, not implemented)

Add platform-specific file locking around the append:

```python
import os, sys
if sys.platform == "win32":
    import msvcrt
    def _lock(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def _unlock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl
    def _lock(f):   fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def _unlock(f): fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def atomic_append_jsonl(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        _lock(f)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            _unlock(f)
```

Trade-offs:
- **Cross-process locking** works for both threads (within a process)
  and processes (e.g. dispatcher writing concurrently with main loop).
- **Performance**: one extra syscall per write. Not measurable for the
  dozens-per-minute write rate of the journals.
- **Cross-platform**: msvcrt and fcntl are stdlib. No new dependencies.
- **Existing callers**: signature unchanged, fully backward-compatible.

## Test plan

1. Re-enable the xfail marker in
   `tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl`
   and verify it passes after the fix.
2. Add a stress test in `tests/test_file_utils.py` directly: 8 threads
   × 500 writes, assert zero torn lines.
3. Cross-process test (Windows + POSIX): spawn 4 subprocesses, each
   writing 500 entries to the same file, assert zero torn lines on the
   parent.

## Implementation order

This is a tier-1 reliability fix but outside the scope of the
fix-agent-dispatcher branch (which only consumes the existing primitive).
Implement on its own branch with the test plan above. Estimate: small —
~50 LOC + tests.
