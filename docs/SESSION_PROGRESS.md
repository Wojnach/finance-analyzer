# Session Progress — Auto-Improve BUG-223/224/225 Fixes (2026-04-25)

**Session start:** 2026-04-25
**Status:** Implementation complete, merge pending

## What was done

### Phase 1: Deep Exploration
4 parallel agents explored: signal engine, portfolio/risk, infrastructure, metals/trading.
Manual verification of all P0/P1 findings against actual code. 32 false positives rejected.

### Phase 2: Plan
Wrote `docs/IMPROVEMENT_PLAN.md` with 3 batches targeting 3 confirmed bugs + doc updates.

### Phase 3: Implementation (3 commits)

**Batch 1: BUG-223 Stop-Loss sell_price Validation** (9f8dea6e)
- `avanza_session.py:748`: Non-trailing MONETARY stop-losses now reject sell_price <= 0.
  A zero sell_price on LESS_OR_EQUAL trigger would execute as market sell at worst price.
  Trailing stops (FOLLOW_DOWNWARDS/UPWARDS) with sell_price=0 remain allowed (by design).
- 2 new tests: rejects zero for monetary, allows zero for trailing.

**Batch 2: BUG-224 + BUG-225** (e8b390a1)
- `signal_engine.py:3137`: Added `_voters_post_filter` to extra_info after persistence
  filter. `_voters` was recording pre-filter count, inflating accuracy tracking downstream.
- `equity_curve.py:236`: Extracted mean before Sharpe std-dev generator (was O(n²), now O(n)).
- 2 new tests for post-persistence voter count logic.

**Batch 3: Documentation**
- Updated signal counts in CLAUDE.md and SYSTEM_OVERVIEW.md: 50→51 modules, 34→33 active, 16→18 disabled.
- Added smart_money and mahalanobis_turbulence to disabled signals list.

## Test Results
- All affected tests pass (6 stop-loss, 33 circuit breaker, 74 equity curve)
- 4 new tests, all pass
- Zero regressions

## Key Decisions
- BUG-223: safety-critical, could cause market sell at worst price. Simple guard.
- BUG-224: kept `_voters` for backwards compat, added `_voters_post_filter` alongside.
- BUG-225: performance only (O(n²)→O(n)), numerically identical result.
- BUG-226 deferred: exit optimizer cost model needs deeper investigation to avoid regression.

## What's next
- Merge worktree into main, push, restart loops
- BUG-226 (exit optimizer hold-to-close EV omits cost model) — deferred, needs investigation

---

# Previous: Auto-Improve BUG-219 + P1/P2 Fixes (2026-04-23)

**Session start:** 2026-04-23
**Status:** Merged

## What was done

### Phase 1: Deep Exploration
4 parallel agents explored: signal engine, portfolio/risk, infrastructure, test coverage.
Manual verification of all P0/P1 findings against actual code. 10+ false positives rejected
(Sortino formula, config wipe guard, dashboard auth, stack overflow counter).

### Phase 2: Plan
Wrote `docs/IMPROVEMENT_PLAN.md` with 4 batches targeting 5 confirmed bugs.

### Phase 3: Implementation (4 commits)

**Batch 1: BUG-219 Loss Escalation Fix** (CRITICAL — 23a2d152)
- `agent_invocation.py:720`: `_record_new_trades()` now extracts `pnl_pct` from
  transaction dicts and passes it to `record_trade()`. The consecutive-loss
  escalation system (1x→2x→4x→8x cooldown multiplier) was completely dead in
  production because `pnl_pct` was always None.
- 4 new tests for pnl_pct forwarding (loss, win, missing, backward compat).

**Batch 2: Rate Limiter + Drawdown NaN Guard** (4d76691d)
- `shared_state.py:256-269`: Rate limiter now reserves the next slot
  (`last_call = last_call + interval`) BEFORE releasing the lock and sleeping.
  Parallel threads see the reserved slot and calculate longer waits, preventing
  stampede.
- `risk_management.py:159`: NaN/Inf guard on drawdown calculation. Non-finite
  peak_value or current_value now returns fail-safe breached=True with 100%
  drawdown instead of silently passing all comparisons.
- 5 new tests (2 rate limiter slot reservation, 3 NaN/Inf guard).

**Batch 3: Cache None + Orphan Process Logging** (a8ce444c)
- `shared_state.py:94-95`: `_cached()` no longer stores None results. Transient
  API failures that return None instead of raising were cached for the full TTL,
  hiding the failure from retry logic.
- `subprocess_utils.py:132-133`: `contextlib.suppress(Exception)` replaced with
  try/except + logger.warning for Job Object assignment failures.
- 3 new tests for None-caching prevention.

## Test Results
- 325 tests pass across all affected files
- 12 new tests, all pass
- Zero regressions in existing test suites

## Key Decisions
- BUG-219 was the highest priority: safety-critical feature (loss escalation) was
  completely broken since the function was first wired.
- Rate limiter fix: chose slot reservation approach (deterministic) over
  post-sleep time.time() approach (non-deterministic with threads).
- NaN guard: fail-safe to breached=True (100% drawdown) rather than ignoring.
  Better to halt trading on corrupted data than to continue blind.
- Cache None: signals that legitimately have no data should return a HOLD dict,
  not None. None always indicates a failure path.

## What's next
- Merge worktree into main, push, restart loops
- Late-arriving findings not yet implemented: trigger.py:138 trade detection
  init, outcome_tracker.py:364 base_price handling

---

# Previous: Auto-Improve Safety Fixes (2026-04-21)

**Session start:** 2026-04-21 ~08:00 UTC
**Status:** Implementation complete, merged

## What was done

### Phase 1: Deep Exploration
4 parallel agents explored: signal system, metals subsystem, infrastructure, test suite.
Combined with the 2026-04-20 adversarial review synthesis (118 findings, 27 P1).

### Phase 2: Plan
Wrote `docs/IMPROVEMENT_PLAN.md` with 4 batches targeting 10 confirmed bugs from 3
consecutive adversarial reviews.

### Phase 3: Implementation (4 commits)

**Batch 1: Safety-Critical Core** (b7bbeb25)
- BUG-209: OHLCV zero/negative price validation in `indicators.py`
- BUG-210: Config wipe guard in `telegram_poller.py`
- BUG-211: Max order size limit (50K SEK) in `avanza_session.py`
- BUG-212: Rate limiter sleep-outside-lock in `shared_state.py`
- BUG-213: `_loading_timestamps` cleanup on success path
- Dashboard `hmac.compare_digest()` timing-safe token comparison

**Batch 2: Drawdown Circuit Breaker + I/O** (9b31afb8)
- BUG-214: `check_drawdown()` wired into `invoke_agent()` — first automated risk gate
- BUG-215: Thread-safe FX cache with `threading.Lock`
- BUG-216: Monte Carlo `seed=42` → `seed=None` (system entropy)
- Journal `write_text()` → `atomic_write_text()`
- Added `file_utils.atomic_write_text()` utility

**Batch 3: Signal + Metals** (7cba829b)
- BUG-217: `_execute_sell()` exception safety (per-position try/except)
- BUG-218: `econ_calendar` force-HOLD (structural SELL-only bias)

**Batch 4: Tests** (eccdc2ee)
- 21 new tests in `tests/test_safety_guards.py`

## Test Results
- 285 tests pass across affected files (indicators, shared_state, avanza_session, dashboard, risk_management, monte_carlo, journal, file_utils)
- 21 new tests all pass
- Full suite verification in progress
- 3 pre-existing failures in metals_swing_trader tests (not caused by this session)

## Key Decisions
- Drawdown circuit breaker: advisory at 20%, hard-block at 50% (per user's risk tolerance)
- Max order size: 50K SEK (~25% of 200K ISK account)
- econ_calendar: force-HOLD rather than rewriting (needs research for BUY capability)
- Persistence filter cold-start: NOT a bug — analyzed and confirmed correct behavior

## What's next
- Merge worktree into main, push, restart loops
- Adversarial review priority fixes still pending: claude_gate routing, browser idempotency, per-ticker accuracy filtering, DST hardcoding

---

# Auto-Improve Session: 2026-04-24

**Session focus:** Autonomous improvement — deep exploration (6 agents), bug fixes, doc updates.

## Changes (3 commits)
1. **BUG-220 fix** (9ad8b9ca): `outcome_tracker.py` — skip outcomes when `base_price` is None
   instead of storing phantom 0% change_pct entries that pollute accuracy stats. +4 tests.
2. **BUG-221 fix** (94b3d416): `daily_digest.py` — catch invalid timezone config strings
   (ValueError, KeyError, ZoneInfoNotFoundError), fall back to UTC. +2 tests.
3. **Doc updates**: Signal counts corrected across CLAUDE.md, SYSTEM_OVERVIEW.md (was 36/33,
   actual 50/34). Bug log updated with BUG-219/220/221.

## Verified Non-Bugs (5 false positives from agents)
- max_confidence caps correctly enforced (signal_engine.py:2706-2707)
- EWMA neutral weight correctly applied (signal_engine.py:280-287)
- trigger.py:138 first-run default is correct behavior
- metals_loop check_session_alive is properly imported
- fin_snipe_manager _critical_alert_last: single-threaded, no race condition
- BUG-222 (fin_snipe alert logging): already has logger.warning in except block

## What's next
- Merge worktree into main, push, restart loops
- Backlog: outcome_tracker JSONL→SQLite migration, test_llama_server cleanup

---

# Previous: Outcome-Tracking Repair (2026-04-20 afternoon)

**Session focus:** User reported MSTR signal accuracy wasn't tracked. Audit
revealed 3 broken paths in fin_evolve.py. Fixed, reviewed, merged, backfilled.

**Status:** SHIPPED to main (commit 486a631f).
