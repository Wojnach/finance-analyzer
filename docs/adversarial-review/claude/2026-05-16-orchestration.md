# Adversarial Review: Orchestration Subsystem (2026-05-16)

## [P1] Subprocess hangs on timeout-failure to drain stderr
**File:** portfolio/claude_gate.py:454-460

Subprocess tree-kill returns error code but child process does not actually terminate. Second communicate() call hangs indefinitely waiting for pipes to close, violating the stated 5-second recovery deadline.

**Fix:** Wrap second communicate() in try/except with hard timeout on individual stdout/stderr reads, or use threading with 5s deadline per thread.

---

## [P2] Race condition: trigger_state pruning loses coherent snapshot
**File:** portfolio/trigger.py:171-184 (_save_state)

Pruning logic uses `state["_current_tickers"]` set populated per-cycle. If signal processor crashes before completing cycle, set is incomplete. _save_state() then discards valid baseline entries. Next cycle, those tickers' BUY/SELL re-fire as "new consensus" even though triggered in crashed cycle.

**Fix:** Pass _current_tickers as explicit parameter; compute set once AFTER all signal processing completes, never mutate it after population.

---

## [P2] Market timing DST edge case: hour comparison allows off-by-one on transition days
**File:** portfolio/market_timing.py:340-342

Direct hour comparison against DST-aware open/close hours. On DST transition days (e.g., spring forward UTC 01:59→03:00), a loop cycle can straddle the boundary. Trigger fired at 01:55 UTC sees hour=1, suppresses Layer 2. Same ticker at 02:50 UTC sees hour=2, passes window. Hour comparison returns different results mid-cycle.

**Fix:** Capture DST result once at start of check_triggers() and pass through; do not recalculate per-hour.

---

## [P2] Tier downshift regex patterns miss multi-line fade-flip with appended conditions
**File:** portfolio/trigger.py:540-559

Pattern `_FADE_FLIP_RE = r'\bflipped (?:BUY|SELL)->HOLD \(sustained\)'` matches substring, not entire reason. If reason is "BTC flipped BUY->HOLD (sustained), volume spike 2.5x", regex matches and reason is downshiftable even though it contains a separate non-fade condition (volume spike). Downshifting T2→T1 loses the volume signal.

**Fix:** Use $ anchor or require match to be entire reason string. Or split on `;` and apply regex per sub-reason.

---

## [P2] Alert budget bucket reset has half-open race on prune
**File:** portfolio/alert_budget.py:32-45

Large deque pruning holds lock for milliseconds. Concurrent call to remaining_budget() also acquires lock and prunes independently, causing duplicate work. If max_per_hour changes between calls (config reload), reads/writes of max_per_hour race without sync, leading to inconsistent bucket state.

**Fix:** Cache pruned deque length and last-prune timestamp; only prune if >10s elapsed since last prune to avoid repeated work.

---

## [P2] Prophecy state corruption: save_beliefs does not validate belief_id uniqueness after update
**File:** portfolio/prophecy.py:105-125

update_belief() mutates entry and saves without verifying that mutation (e.g., changing belief["id"]) does not create duplicate IDs. If ID is accidentally duplicated, next add_belief() rejects NEW beliefs because existing_ids already contains the mutated duplicate. Silent corruption.

**Fix:** In update_belief(), verify that if "id" in updates and differs from original, no other belief has that new ID. Or forbid ID mutations.

---

## [P1] Autonomous fallback: _load_bold_state_safe uses wrong fallback default
**File:** portfolio/autonomous.py:835-846

Returns hardcoded default with `"initial_value_sek": 500000`. But real Bold portfolio may have different starting balance from config. PnL calculations wrong if actual initial != 500K. User makes trading decisions on fictitious returns.

**Fix:** Load actual initial_value_sek from config or from fallback read of portfolio_state_bold.json directly, not hardcoded 500K.

---

## [P2] Bigbet gate: L2 invocation timeout does not cascade back to check_bigbet loop
**File:** portfolio/bigbet.py:561-572

invoke_layer2_eval() times out after 30s, returns (None, ""). Gate skips that ticker. No rate-limiting on retries. If Claude times out, code retries same ticker again next cycle (10min later). Repeated timeouts cause thundering herd hitting overloaded Claude session.

**Fix:** Record last failed L2 invocation time per ticker; skip re-invocation for 5 minutes after timeout.

---

## [P2] Escalation gate runner timeout leaves thread orphaned
**File:** portfolio/escalation_gate.py:202-215

Ministral runner times out, executor shutdown with `wait=False, cancel_futures=True`. Only cancels tasks NOT started. Already-running task continues in background holding GPU VRAM. Orphaned threads accumulate over long session. After 100 timeouts, 100 threads × 500MB = 50GB lost. GPU eventually exhausts.

**Fix:** Use with-statement context manager or explicit executor.shutdown(wait=True). Or use subprocess-based runner with hard kill-on-timeout.

---

## [P2] Circuit breaker recovery timeout backoff does not validate monotonicity
**File:** portfolio/circuit_breaker.py:64-72

Doubles recovery_timeout and caps at max. No check that doubled value doesn't overflow or that persisted state doesn't creep toward infinity over long-lived session. Next HALF_OPEN→OPEN transition causes arithmetic error or saturates at float max, making breaker permanently open.

**Fix:** Clamp recovery_timeout to reasonable ceiling (e.g., 1 hour) independent of max_recovery_timeout. Use math.prod() with bounds checking.

---

## [P1] Agent invocation: PF_HEADLESS_AGENT passed but env not cleaned on Layer 2 startup
**File:** portfolio/agent_invocation.py:1042

Sets `agent_env["PF_HEADLESS_AGENT"] = "1"` but does NOT call _clean_env() from claude_gate to strip CLAUDECODE. Subprocess inherits parent's CLAUDECODE, causing Claude CLI to reject nested session error.

If Layer 2 agent calls invoke_claude_text() from iskbets/bigbet and they route through claude_gate, the env IS cleaned. But if they invoke claude directly, subprocess fails. Agent cannot invoke Claude for gate decisions, silently defaults to approved=True for iskbets.

**Fix:** Call `env = _clean_env()` from claude_gate in invoke_agent() to strip CLAUDECODE and CLAUDE_CODE_ENTRYPOINT before spawning.

---

## [P1] Main loop restart with modified trigger_state causes first-of-day T3 to fire twice
**File:** portfolio/trigger.py:609-610 (classify_tier)

classify_tier() returns T3 if last_trigger_date != today. Called after trigger is processed. If loop restarts mid-cycle, next cycle's check_triggers() updates last_trigger_date to today. But if restart happens between check_triggers() and classify_tier(), the two read different _today_str() values. Causes T3 to fire, but last_trigger_date not set (it was yesterday before restart), so next tick ALSO fires T3.

**Fix:** Move state["last_trigger_date"] update to AFTER classify_tier() in main.py, not inside check_triggers(). Or pre-compute date string once and pass through call chain.

---

## [P2] Iskbets stage1_hit does not update position price in state, breaking trailing stop math
**File:** portfolio/iskbets.py:400-404

When stage 1 hit, code sets stop_at_breakeven=True and updates stop_loss=entry_price. Does NOT update highest_price to current price at stage 1. Next cycle, check_exits() recomputes highest from pos.get("highest_price", entry_price), which is outdated. If price dips below stage 1 but above breakeven, trailing stop math uses wrong highest, allowing deeper loss than intended.

**Fix:** Update pos["highest_price"] = price when stage 1 is hit, before returning from check_exits().

---

## SUMMARY
P1 = 3, P2 = 8
