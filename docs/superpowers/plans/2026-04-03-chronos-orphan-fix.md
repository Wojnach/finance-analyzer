# Plan: Prevent Orphaned Chronos/Ministral Subprocess Accumulation

## Problem

When `metals_loop.py` crashes hard (kill -9, power loss, unhandled exception
bypassing cleanup), `stop_llm_thread()` → `_stop_chronos_server()` /
`_stop_ministral_server()` never runs. The orphaned `chronos_server.py` and
`ministral_trader.py --server` processes survive. On next restart, new instances
spawn alongside the orphans, accumulating over time.

## Root Cause

`_start_chronos_server()` and `_start_ministral_server()` in `data/metals_llm.py`
use raw `subprocess.Popen()` with no OS-level lifecycle binding to the parent.

## Solution: Two Layers

### Layer 1: Windows Job Object (automatic cleanup)

Add `popen_in_job()` to `portfolio/subprocess_utils.py` — a Popen wrapper that:
1. Creates a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
2. Starts the subprocess via `Popen`
3. Assigns the child to the Job Object
4. Returns `(proc, job_handle)` — caller stores both
5. When parent dies (crash, kill), OS closes the handle → kills the child automatically

Modify `_start_chronos_server()` and `_start_ministral_server()` to use
`popen_in_job()` instead of raw `Popen`. Store job handles in module globals.
Close handles explicitly in `_stop_*_server()`.

### Layer 2: Startup orphan sweep (safety net)

At `start_llm_thread()` entry, before spawning anything, scan for and kill any
running `chronos_server.py` or `ministral_trader.py --server` processes. Since
the metals_loop singleton lock ensures only one metals_loop runs, any such
processes are guaranteed orphans from a previous crash.

## Files to Modify

| File | Change |
|------|--------|
| `portfolio/subprocess_utils.py` | Add `popen_in_job()` function |
| `data/metals_llm.py` | Use `popen_in_job()`, add orphan sweep at startup |

## Files NOT Modified (no conflicts)

- `data/metals_loop.py` — FishEngine additions don't touch LLM startup/shutdown
- `portfolio/llm_batch.py` — new file, no overlap
- `data/chronos_server.py` — no changes needed (Job Object works from parent side)
- `portfolio/main.py` — singleton guard already merged, batch flush additions don't overlap

## What Could Break

- `popen_in_job()` ctypes calls could fail on unusual Windows configs → graceful
  fallback to raw Popen (same pattern as existing `run_safe()`)
- Orphan sweep could kill a legitimate process if metals_loop singleton lock is
  somehow bypassed → extremely unlikely, lock is msvcrt file lock

## Execution Order

1. Add `popen_in_job()` to `subprocess_utils.py`
2. Modify `metals_llm.py` to use it + add orphan sweep
3. Write tests for both
4. Run test suite, commit, merge, push
