# Adversarial Review: orchestration (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08

---

## CRITICAL

### CO1. classify_tier/update_tier_state double-read race — T3 can repeat every cycle [90% confidence]
**File**: `portfolio/trigger.py:283-318`, `portfolio/main.py`

`classify_tier(reasons_list)` reads trigger_state.json (no `state=` arg). Then
`update_tier_state(tier)` reads it AGAIN. If anything writes between reads,
`last_full_review_time` update may be lost. Result: T3 selected every cycle,
wasting Claude quota on repeated full reviews.

**Fix**: Pass the `state` dict already loaded in `check_triggers` through both calls.

### CO2. ThreadPoolExecutor `f.cancel()` doesn't stop running threads — zombie accumulation [88% confidence]
**File**: `portfolio/main.py:531-537`

`Future.cancel()` only prevents futures from starting. Already-running threads (blocking
in HTTP or GPU lock) continue alive inside the exited pool. Over dozens of timeout cycles,
zombie threads accumulate, consuming memory and holding rate-limiter state.

### CO3. Multi-agent path blocks main loop thread for 150s synchronously [85% confidence]
**File**: `portfolio/agent_invocation.py:241-266`

`wait_for_specialists(procs, timeout=150)` blocks the main loop. Combined with ticker
pool timeout (120s), one cycle can run 270s — far exceeding `MAX_CYCLE_DURATION_S = 180`.
During this window: no signals, no heartbeat, Layer 1 deaf to market.

---

## HIGH

### HO1. Post-restart trade detection: prev_count defaults to current_count — misses trades [90% confidence]
**File**: `portfolio/trigger.py:66-94`

On first call after restart, `prev_count = last_checked_tx.get(label, current_count)`.
Trades during outage are invisible. No post-trade reassessment triggers. Should default
to 0 with a warning, not current_count.

### HO2. Agent kill failure clears `_agent_proc` — next cycle spawns duplicate [85% confidence]
**File**: `portfolio/agent_invocation.py:164-202`

When `taskkill` fails: `_agent_proc = None` + `return False`. Next cycle: `_agent_proc
is None` → guard passes → new agent spawns alongside unkillable old one. The reference
to the unkillable process should be KEPT, not cleared.

(Confirms Claude's H4 finding with more precise mechanism)

### HO3. Digest timezone-naive comparison silently skips entries [88% confidence]
**File**: `portfolio/digest.py:72, 106, 128`

If any JSONL entry has naive timestamp (no `+00:00`), `fromisoformat` returns naive dt.
Comparison `naive >= aware` raises TypeError, caught by `except (ValueError, TypeError)`.
Entry silently skipped → digest under-counts invocations/analyses.

### HO4. Loop contract ticker count mismatch after pool timeout [85% confidence]
**File**: `portfolio/loop_contract.py:65`

On `as_completed` timeout, completed-but-unextracted futures fall into neither
`signals_ok` nor `signals_failed`. `total_processed < n_active` → spurious CRITICAL
invariant violation that looks worse than reality.

---

## MEDIUM

### MO1. US DST fall-back endpoint 06:00 UTC — should be 07:00 UTC [80% confidence]
**File**: `portfolio/market_timing.py:83`

US clocks fall back at 02:00 EST = 07:00 UTC. Using 06:00 causes 1-hour window once/year
where NYSE close time reported as 21:00 instead of 20:00. Stocks processed for extra hour.

### MO2. Agent state as module-level globals — latent thread safety risk [83% confidence]
**File**: `portfolio/agent_invocation.py:26-33`

8 globals (`_agent_proc`, `_agent_log`, etc.) with no lock. Currently single-threaded
but any future ThreadPoolExecutor access would corrupt state silently.

### MO3. Initial run crash swallows baseline — spurious T3 on first success [80% confidence]
**File**: `portfolio/main.py:884-900`

If initial `run()` throws, `trigger_state.json` never gets baseline. Next cycle's
`check_triggers()` compares against empty/stale prev → spurious T3 trigger.

### MO4. ViolationTracker resets consecutive count on single-cycle recovery [80% confidence]
**File**: `portfolio/loop_contract.py:480-498`

Intermittent violations (2 cycles fail, 1 passes, 2 fail) never reach
`ESCALATION_THRESHOLD = 3`. Need running violation rate, not strict consecutive count.

---

## LOW

### LO1. Digest accounting inconsistency with get_completion_stats [80% confidence]
### LO2. Agent window ignores Swedish market holidays [80% confidence]
### LO3. process_lock silent no-op when no locking module available [80% confidence]

---

## Cross-Critique: Claude vs Orchestration Agent

### Agent found that Claude missed:
1. **CO1**: classify_tier double-read race — total miss (subtle state file interaction)
2. **CO2**: ThreadPoolExecutor cancel no-op — mentioned BUG-178 generically but didn't
   analyze the cancel ineffectiveness or zombie thread accumulation
3. **CO3**: Multi-agent 150s synchronous block — total miss
4. **HO1**: Post-restart trade detection miss — total miss
5. **HO3**: Digest timezone comparison — total miss
6. **HO4**: Loop contract ticker count mismatch — total miss
7. **MO1**: US DST fall-back 1-hour error — total miss (I noted DST generically in M4)
8. **MO3**: Initial crash baseline loss — total miss
9. **MO4**: ViolationTracker gap reset — total miss

### Claude found that agent confirmed:
1. **H4/HO2**: Agent kill clears _agent_proc → duplicate spawn — both found
2. **H5**: Stack overflow counter never resets — agent didn't re-raise
3. **M3**: Trigger state set() not JSON serializable — agent didn't re-raise

### Net: 3 CRITICAL + 4 HIGH + 4 MEDIUM + 3 LOW = 14 issues, ~11 net-new.
