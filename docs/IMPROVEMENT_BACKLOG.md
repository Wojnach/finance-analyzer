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

## ~~TEST-HYGIENE-2 — `tests/test_llama_server_job_object.py` (untracked)~~ RESOLVED

**Discovered:** 2026-04-17. **Resolved:** 2026-05-01 cleanup session.
**Prior triage:** `docs/plans/2026-04-17-pre-existing-tests.md`.

### Resolution
The aspirational test file `tests/test_llama_server_job_object.py` is
no longer present in the working tree of `main` (was never tracked in
git history and has been cleaned out at some point between 2026-04-17
and 2026-05-01). Stale `__pycache__` artifacts (`.pyc` files) from
prior collection runs are also being cleaned up — pytest does not
collect from `.pyc` files but they are confusing residue.

Verified state on 2026-05-01:
- `git ls-files tests/ | grep llama_server` → only the legitimate
  `tests/test_llama_server.py` (model management + query
  serialization), not the job-object file.
- `pytest tests/ -k 'llama_server' --collect-only` from a fresh
  worktree of `main` collects 23 tests cleanly with 0 errors.
- Production code partially implements the feature anyway:
  `popen_in_job` and `close_job` now exist in
  `portfolio/subprocess_utils.py` (used by the metals subsystem).
  The remaining symbols (`_local_job_handle`, `_sweep_done`,
  `kill_orphaned_llama_server`, `_kill_orphaned_by_name`,
  `atexit.register(stop_all_servers)`) were never landed because
  `llama_server.py`'s lifecycle is solved differently — via PID
  files, file locks, and an external orphan reaper
  (`kill_orphaned_llama` in `subprocess_utils.py`).

### Why deletion was correct
The features the test file enumerated were aspirational. The
production solution chose a different shape (PID file + orphan
reaper). Implementing the test's vision would require ~300+ LOC of
restructuring `llama_server.py` for a feature that was never
prioritized. The lower-risk path (delete the file, accept the actual
production design) was the right call.

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
