# Improvement Backlog

Standing work items surfaced during sessions but intentionally deferred.
Each entry: title, reason-for-deferral, scope estimate, and any pointers
to prior triage docs.

---

## ~~TEST-HYGIENE-1 — xdist module-state leak audit~~ RESOLVED

**Discovered:** 2026-04-17. **Resolved:** 2026-04-19 auto-session.

Global autouse fixture in `tests/conftest.py` (`_reset_module_state`)
resets all HIGH-risk module state (agent_invocation, signal_engine,
shared_state) before/after every test. Reset helpers in
`tests/_state_reset.py` also cover MEDIUM/LOW-risk modules (forecast,
logging_config, api_utils, trigger).

Result: 5+ random xdist flakes eliminated per run. Remaining 24
failures are all pre-existing infrastructure dependencies (freqtrade,
Ministral model).

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
