# Improvement Backlog

Standing work items surfaced during sessions but intentionally deferred.
Each entry: title, reason-for-deferral, scope estimate, and any pointers
to prior triage docs.

---

## TEST-HYGIENE-1 — xdist module-state leak audit

**Discovered:** 2026-04-17 adversarial-review sessions.
**Prior triage:** `docs/plans/2026-04-17-pre-existing-tests.md`
**Scope estimate:** 1-2 days of dedicated work.

### Problem
`pytest -n auto` full-suite runs report 5-10 failures per run on main,
with a **different set each time**. Tests pass in isolation. Root cause
is module-level mutable state that leaks across test files when xdist
shards tests across workers in a different order than the last run.

Known-affected state (incomplete):

- `portfolio.agent_invocation._agent_proc / _agent_start / _agent_timeout
  / _agent_log / _agent_tier / _agent_reasons / _agent_log_start_offset`
- `data.metals_llm._chronos_proc / _chronos_job / _ministral_proc /
  _ministral_job`
- `portfolio.signal_engine._cached_or_enqueue` in-memory cache +
  `_last_phase_log_per_ticker`
- `portfolio.signals.forecast._FORECAST_MODELS_DISABLED` global +
  `_predictions_dedup_cache`
- `portfolio.accuracy_stats` TTL cache
- GPU-state reads (`portfolio.gpu_gate.get_vram_usage`) that read the
  REAL GPU and drift between tests

### Mitigations applied 2026-04-17 (4 tests)
- `test_consensus::test_stock_total_applicable` — stale assertion (not a
  flake, just a count update).
- `test_metals_llm_orphan::test_start_chronos_uses_popen_in_job` — autouse
  fixture resetting `_chronos_*` + `get_vram_usage` mock.
- `test_perception_gate::test_gate_skips_invocation` — inline reset of
  `_agent_*` state.

### Proposed scope
1. Produce a module-by-module catalogue of mutable module-scope state.
2. For each module, choose: (a) autouse reset fixture in the test file
   that imports it, (b) context manager for `with X.state_scope():`,
   or (c) refactor to a class instance injected into callers.
3. Run `pytest -n auto` 10× in a row, confirm failures = 0 every time.

### Why not solved now
Each cluster needs individual investigation + verification. It's hygiene
work across ~15 test files and ~8 production modules. Better as a
dedicated session than rolled into trade-logic fixes.

---

## TEST-HYGIENE-2 — `tests/test_llama_server_job_object.py` (untracked)

**Discovered:** 2026-04-17.
**Status:** untracked file in repo root, last seen 2026-04-17.
**Prior triage:** `docs/plans/2026-04-17-pre-existing-tests.md`.
**Scope estimate:** Either 1 day (implement the feature) or 1 line (delete file).

### Problem
`tests/test_llama_server_job_object.py` ships 11 regression tests for
Windows Job Object lifecycle support in `portfolio/llama_server.py`.
The tests expect:

- `popen_in_job` helper at `portfolio.subprocess_utils`
- `close_job` helper that closes the Job Object handle
- `_local_job_handle` global in `llama_server.py`
- `_sweep_done` flag + startup orphan sweep
- `kill_orphaned_llama_server` that targets `llama-server.exe`
- `_kill_orphaned_by_name(image_name)` helper factored from
  `kill_orphaned_llama`
- `atexit.register(stop_all_servers)` at module load

**None of these exist in production code.** The test file ships 20+
failures in any pytest run that collects it.

### Options

**A — Implement the feature.** Covers all 11 tests, justifies the file
in git. Needs:
- `popen_in_job(cmd, **popen_kwargs) → (Popen, job_handle)` using
  Windows `CreateJobObject` + `AssignProcessToJobObject` on Windows,
  no-op stub on POSIX.
- `close_job(handle)` that closes the Job Object.
- Integrate `popen_in_job` into `_start_server` + `_stop_server`.
- Startup orphan sweep guard.
- `atexit` handler.
- Refactor `kill_orphaned_llama` into `_kill_orphaned_by_name` +
  `kill_orphaned_llama` + `kill_orphaned_llama_server` callers.

**B — Delete the file.** One-line commit:
`git rm tests/test_llama_server_job_object.py`.
Accepts that Windows Job Object support is not planned.

### Why not solved now
The file is untracked in main, so it doesn't affect git history — but
it DOES poison `pytest -n auto` runs from the main repo (not
worktrees). Until the feature ships or the file is deleted, any
full-suite report from main includes 20 irrelevant failures that
obscure real regressions.

---

## Pattern for adding new backlog items

Append a new section with:
- Short ID (`TEST-HYGIENE-N`, `FEATURE-N`, `RISK-N`, etc.)
- Title
- Discovery session / date
- Prior triage doc (if any)
- Scope estimate
- What the problem actually is
- What the acceptance criteria look like
- Why it was deferred this time
