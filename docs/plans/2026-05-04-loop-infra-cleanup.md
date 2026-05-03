# PLAN — loop-infra cleanup (2026-05-04)

**Date:** 2026-05-04
**Branch:** `feat/loop-infra-cleanup-2026-05-04`
**Worktree:** `.worktrees/loop-infra-cleanup`

**Goal:** Close three follow-ups flagged across yesterday's loop_health
work. All low-risk consolidation; no new behavior.

---

## Context

The 8-commit run from `7f70f52f` → `99115711` left three threads:

1. `crypto_loop` / `oil_loop` / `mstr_loop` each ship their own private
   `_write_heartbeat` wrapper (predates the shared `loop_health.write_heartbeat()`
   added in `b282b1a0`). The wrappers each duplicate the JSON schema +
   error-handling logic. A schema change today requires touching all four
   call sites; should require touching one.
2. `_last_dashboard_prewarm_ts` is in-process state. A loop restart
   resets it to 0, causing one extra cold-cache scan per restart. Cheap
   enough that we accepted it on initial ship; trivial fix is to persist
   it to `data/dashboard_prewarm_state.json`.
3. `CLAUDE.md` lists `/api/forecast` and `/api/prophecy` as "key
   endpoints" — both return 404 (removed in the dashboard mobile-redesign
   merge `62b4f4dd`). Doc drift; misleads future agents that grep CLAUDE.md
   for endpoint references.

## What this PR does

### Batch 1 — migrate heartbeat wrappers to shared helper

| File | Change |
|---|---|
| `data/crypto_loop.py` | `write_heartbeat(extra=...)` becomes a 5-line shim that delegates to `loop_health.write_heartbeat(HEARTBEAT_FILE, cycle=..., ok=..., n_positions=..., extra=...)`. Call sites unchanged. |
| `data/oil_loop.py` | Same shim. |
| `portfolio/mstr_loop/loop.py` | `_write_heartbeat(bot_state, cycle_count)` delegates same way; computes `n_positions` from `bot_state.positions` inline. |
| `tests/test_loop_health_write_heartbeat.py` | Add a contract test asserting all 5 default loops can call the shared helper and produce schema-compatible output. |

### Batch 2 — persistent prewarm timestamp

| File | Change |
|---|---|
| `portfolio/accuracy_stats.py` | `maybe_prewarm_dashboard_accuracy()` reads `data/dashboard_prewarm_state.json` on first call (lazy) to seed `_last_dashboard_prewarm_ts`, and atomic-writes back when it fires. Survives loop restarts. |
| `.gitignore` | Ignore `data/dashboard_prewarm_state.json` (runtime artifact). |
| `tests/test_accuracy_compute_lock.py` | New test: state persists across "restart" (simulated by re-importing module / resetting in-memory ts). |

### Batch 3 — CLAUDE.md endpoint list cleanup

| File | Change |
|---|---|
| `CLAUDE.md` | Replace the truncated, partly-stale endpoint list with the canonical 32-endpoint list grepped from `dashboard/app.py`. Drop `/api/forecast` and `/api/prophecy` (don't exist). |

### Out-of-scope (deferred)

- **Cycle-0 reporting guard** — would prevent the original `module_failures`
  trigger from happening at all. Touches `portfolio/reporting.py` and
  needs careful regression testing across all 18 reporting modules.
  Worth a separate PR.
- **Pre-existing 60 test failures** — config drift documented as
  26 in `docs/TESTING.md`. Reconciliation is its own session.

## Risk

- **Behavior change risk: zero.** Wrappers preserve their interface;
  the shared helper produces the same JSON schema. The persistent
  timestamp file is read-only on miss, so a corrupt or missing file
  falls back to the existing in-memory default.
- **Test risk: low.** Each batch has its own test; full suite must
  remain at the same failure count.

## Execution order

1. Worktree created ✅
2. Commit this plan
3. Batch 1 — heartbeat wrapper migration + tests
4. Batch 2 — persistent prewarm timestamp + tests
5. Batch 3 — CLAUDE.md cleanup
6. Run focused tests, then full suite
7. Merge into main, push, restart affected loops, verify
8. Clean up worktree
