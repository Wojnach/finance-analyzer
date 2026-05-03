# PLAN — extend loop health watchdog to metals + golddigger (2026-05-03)

**Date:** 2026-05-03
**Branch:** `feat/loop-health-metals-golddigger`
**Worktree:** `.worktrees/loop-health-metals-golddigger`

**Goal:** Close the remaining gaps in loop_health watchdog coverage. After
this morning's MSTR work (commit `7f70f52f`), three of five live loops emit
a heartbeat that the watchdog reads. This PR adds the remaining two:
**metals_loop** (silver/gold warrant trading, LIVE) and **golddigger**
(gold certificate trading).

---

## Context

The `portfolio/loop_health.py` rollup module ships with a TODO in its
docstring:

> "Add more loops here when they grow heartbeat support (metals, main loop)."

After this morning's MSTR addition, the rollup covers `crypto`, `oil`, and
`mstr`. Two production loops remain blind to the watchdog:

| Loop | Schedule | Live? | Current telemetry | Heartbeat? |
|------|----------|-------|-------------------|------------|
| `metals_loop` | PF-MetalsLoop (Running) | **YES — DRY_RUN=False** | `metals_swing_state.json`, `metals_loop_out.txt` | NO |
| `golddigger` | PF-GoldDigger (Running) | configurable | `golddigger_state.json`, `golddigger_log.jsonl` | NO |

**Main loop** (`PF-DataLoop`) deliberately stays out of scope: it already
has `data/health_state.json` + `loop_contract.py` invariant alerts; adding
a third mechanism would be redundant and risks alert fatigue.

## What this PR does

### Single batch

| File | Change |
|---|---|
| `data/metals_loop.py` | New `HEARTBEAT_FILE` const + module-level `_write_heartbeat()` helper + one call at cycle end (between `verify_and_act` at line 7630 and `_sleep_for_cycle` at line 7635). |
| `portfolio/golddigger/runner.py` | New `HEARTBEAT_FILE` const + helper + call between `verify_and_act` (line 360) and `time.sleep` (line 362). |
| `portfolio/loop_health.py` | Add `"metals"` and `"golddigger"` to `DEFAULT_HEARTBEAT_FILES`. Tighten docstring TODO. |
| `.gitignore` | Ignore `data/metals_loop.heartbeat`, `data/golddigger_loop.heartbeat`. |
| `tests/test_loop_health.py` | Extend default-files test to assert metals + golddigger keys. |
| `tests/test_metals_loop_heartbeat.py` (new) | Lightweight unit test of helper — write contract, error swallowing, end-to-end via `read_loop_health`. Avoid importing the full `metals_loop.py` (7K LoC + heavy deps). |
| `tests/test_golddigger_heartbeat.py` (new) | Same shape as MSTR heartbeat test. |

### Heartbeat file contract (matches existing crypto/oil/mstr)

```json
{
  "ts":           "2026-05-03T19:21:10.979681+00:00",
  "status":       "ok",
  "cycle":        N,
  "ok":           true,
  "n_positions":  K
}
```

Only `ts` is required by `loop_health.read_loop_status()`. Other fields
are operator-facing context.

## Why this design

1. **Minimal-touch for metals.** `metals_loop.py` is 7K+ lines, LIVE
   trading. Adding a single helper + one call site at the existing
   cycle-end keeps the blast radius tiny.
2. **Mirror of MSTR/crypto/oil.** Same pattern; no new mental model.
3. **Failure-mode aligned.** Heartbeat write inside try/except (best-effort
   telemetry; never crashes the loop). On `run_cycle` exception, heartbeat
   is skipped so the watchdog correctly sees staleness.
4. **Test-light by design.** Metals_loop tests historically import the
   full module which spins up llama-server / Avanza session; new
   heartbeat test stays isolated to the helper.

## Risk

- **Live-trading risk: zero.** Heartbeat write is additive telemetry.
- **Watchdog noise risk: zero.** Both loops are currently `Running`;
  after restart they write fresh heartbeats within one cycle.
- **Restart risk: low.** Per `docs/GUIDELINES.md` step 9.

## Out-of-scope (deferred)

- Main loop heartbeat (already covered by `health_state.json`).
- Per-loop watchdog cooldown tuning.
- Per-loop dashboard rendering (auto picks up via existing `/api/loop_health`).

## Execution Order

1. Create worktree `feat/loop-health-metals-golddigger` ✅
2. Commit this plan.
3. Batch 1 (single batch): all 7 file changes.
4. Run focused tests: heartbeat + loop_health + watchdog.
5. Merge into main, push, restart loops, verify, cleanup.
