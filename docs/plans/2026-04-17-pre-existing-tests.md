ultrathink

# Fix "Pre-Existing Test Failures" — 2026-04-17 (proper /fgl)

Branch: `fix/pre-existing-tests-20260417`
Worktree: `/mnt/q/finance-analyzer-tests`
Base SHA: `2067214f` (main after metals momentum-exit merge)

## Context
Following the metals-momentum-exit ship, a full `pytest -n auto` run on **main**
reported 23 failures. Triage via 3 parallel Explore agents + direct verification
in a clean worktree shows the real picture is much smaller:

- **20 failures** come from an **untracked** test file
  `tests/test_llama_server_job_object.py` that someone dropped into main's
  working tree without committing. It asserts features (Windows Job Object
  lifecycle, `popen_in_job`, `close_job`, `_kill_orphaned_by_name` helper)
  that **do not exist** in production `portfolio/llama_server.py` or
  `portfolio/subprocess_utils.py`. The tests fail in main because the file
  is collected by pytest but do NOT reproduce in the worktree (fresh
  checkout = no untracked file). These are NOT pre-existing regressions —
  they're shipped-before-production test artifacts.
- **3 failures are xdist isolation flakes** (pass in isolation, fail in
  parallel). Not real regressions.
- **1 failure is a real stale assertion** (off-by-one after signal count
  grew 41→43).

## Scope decisions

### In scope
1. Fix the real stale assertion (1 line).
2. Decide the untracked file: delete from main's working tree (out of scope
   here) OR write a short note documenting why it's untracked.
3. Mark the xdist flakes with a clarifying comment so future full-suite runs
   don't panic.

### Out of scope (needs own session)
1. Implementing Windows Job Object support in `llama_server.py` +
   `subprocess_utils.py` — substantial feature work (~300+ LOC, atexit
   registration, global state, cross-platform guards) to justify the
   pre-shipped test file's expectations.
2. The `test_start_chronos_uses_popen_in_job` test in
   `test_metals_llm_orphan.py` — same Job Object dependency.

## Verified failures reproducible in a clean worktree

| # | Test | Category | Fix |
|---|------|----------|-----|
| 1 | `test_consensus.py::TestStockSignalVoteCounts::test_stock_total_applicable` | Stale assertion | Change `== 26` to `== 27`. Current stock-applicable count after exclusions is 27 (SIGNAL_NAMES grew to 43). |
| 2 | `test_fish_engine.py::TestLayer2Staleness::test_exactly_4h_old_accepted` | xdist flake | Passes in isolation. Likely state-isolation with another test that patches a module-level constant. Add an autouse fixture that resets the state, or mark as `serial`. |
| 3 | `test_metals_llm_orphan.py::TestJobObjectIntegration::test_start_chronos_uses_popen_in_job` | Feature pre-shipped | SKIP with `pytest.mark.skip(reason="Job Object support not yet implemented — see plans/2026-04-17-pre-existing-tests.md")`. |
| 4 | `test_perception_gate.py::TestAgentInvocationIntegration::test_gate_skips_invocation` | xdist flake | Passes in isolation. Check what state another test is leaking. Likely an import-time patch to `portfolio.perception_gate.should_invoke`. |

## Execution plan

### Batch 1 — Mechanical fix + flake mitigation
Files (3):
- `tests/test_consensus.py` — update the `26 → 27` assertion (#1).
- `tests/test_metals_llm_orphan.py` — add skip marker to the Job Object test (#3).
- `tests/test_fish_engine.py` — investigate + add autouse reset fixture OR serial marker (#2).
- `tests/test_perception_gate.py` — same treatment (#4).

### Batch 2 — Verify + document
- Run full suite in worktree: `pytest tests/ -n auto --ignore=tests/integration`.
  Goal: 0 reproducible failures in the clean worktree. Any remaining failures
  must be documented as known-flaky with pattern.
- Update `docs/TESTING.md` with the new "Known xdist flakes" section if #2 and
  #4 don't yield a permanent fix.

### Batch 3 — Codex review → merge → push
- `codex review --base main`. Address P1/P2.
- Merge to main, push via cmd.exe, clean up worktree + branch.
- No loop restart needed (test-only changes).

## Why the untracked file is out of scope
Implementing Windows Job Object lifecycle in `llama_server.py` requires:
- `popen_in_job()` helper that creates a CreateProcess with Job Object association
- Cross-platform stubs (no-op on POSIX)
- Atexit registration to ensure child cleanup
- `_sweep_done` flag + orphan sweep at startup
- `kill_orphaned_llama_server` variant targeting `llama-server.exe` image name
- `_kill_orphaned_by_name` helper factored from existing `kill_orphaned_llama`
- Migration of existing `subprocess.Popen` call sites in `_start_server` + `_stop_server`

That's a feature, not a test fix. Covering it here would blur scope. The proper
next step is either:
1. A follow-up session that implements the feature and moves the untracked
   file into `git add`, OR
2. Delete the untracked file from main's working tree if the feature is not
   planned.

For now: leave untracked, document in this plan.
