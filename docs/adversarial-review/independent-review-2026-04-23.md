# Independent Adversarial Review — 2026-04-23

**Date:** 2026-04-23 17:30 CET
**Reviewer:** Claude Opus 4.6 (independent stream, 1M context)
**Scope:** Full codebase, 8 subsystems, ~72K lines of Python
**Method:** Direct code reading of critical files, cross-referenced with
CLAUDE.md rules, memory files, and `.claude/rules/*.md` constraints.

---

## Findings

### MY-SC-001 — `_voters` count uses pre-accuracy-gating count; Stage 4 min_voters check is hollow
**Severity:** P1 | **Subsystem:** signals-core | **Confidence:** 88%
**File:** `portfolio/signal_engine.py:2857,3090,2096-2104`

`active_voters = buy + sell` at line 2857 counts all non-HOLD votes AFTER regime gating
and horizon disabling, but BEFORE the accuracy gate in `_weighted_consensus`. This count
is stored as `extra_info["_voters"]` at line 3090. Then `apply_confidence_penalties`
Stage 4 (line 2096-2104) checks `active_voters < dynamic_min` and forces HOLD if too few.

The problem: `_weighted_consensus()` (line 3061) accuracy-gates additional signals
internally. If 8 signals pass regime gating but 5 get accuracy-gated, only 3 signals
actually drive the consensus. But Stage 4 sees `_voters=8` and passes the check.
The consensus is built from a thin slate that the guard thinks is robust.

The circuit breaker (`_compute_gate_relaxation`) partially mitigates this by trying to
maintain 5+ active voters through gate relaxation. But when relaxation can't recover
enough voters, the weighted consensus proceeds with fewer voters than Stage 4 believes.

**Fix:** Expose the post-accuracy-gating voter count from `_weighted_consensus` and use
it in Stage 4, or move the min_voters check inside `_weighted_consensus`.

---

### MY-SC-002 — Utility boost can rescue accuracy-gated signals
**Severity:** P2 | **Subsystem:** signals-core | **Confidence:** 85%
**File:** `portfolio/signal_engine.py:3001-3024`

The utility overlay applies up to 1.5x boost to `accuracy_data[sig]["accuracy"]` BEFORE
passing to `_weighted_consensus`. A signal at 43% real accuracy (below 47% gate) with
utility boost 1.1x becomes 47.3%, clearing the gate. The gate check inside
`_weighted_consensus` runs on the already-boosted value.

Known issue from `memory/signal_engine_audit_findings.md` ("aggressive utility boost").
Means the accuracy gate can be systematically bypassed by signals that happen to catch
large moves — even if they're right less than half the time.

**Fix:** Apply utility as a weight multiplier, not an accuracy adjustment. Or apply the
accuracy gate on raw accuracy before the utility boost.

---

### MY-SC-003 — Persistence filter cold-start seeds cycles at minimum threshold
**Severity:** P2 | **Subsystem:** signals-core | **Confidence:** 82%
**File:** `portfolio/signal_engine.py:264-268`

On the first cycle for a new ticker, the persistence filter seeds all non-HOLD signals
with `cycles = _PERSISTENCE_MIN_CYCLES` (=2). On cycle 2, any signal voting the same
direction already meets the threshold (cycles >= 2). The filter's purpose is "require
2+ consecutive same-direction votes" but cold-start makes cycle 1->2 pass immediately.

After every system restart, signals bypass persistence for one cycle.

**Fix:** Seed with `cycles = 1` so the first real agreement on cycle 2 correctly
represents "seen twice."

---

### MY-ORCH-001 — Crashed cycles fire immediately after backoff sleep
**Severity:** P1 | **Subsystem:** orchestration | **Confidence:** 90%
**File:** `portfolio/main.py:1113-1145`

After a crash, `_crash_sleep()` runs (exponential backoff 10s-5min), then
`last_cycle_started = cycle_started` is set to the pre-crash timestamp. The next
`_sleep_for_next_cycle(last_cycle_started, sleep_interval)` computes
elapsed = (crash_sleep + work_duration), always > interval, so it returns immediately.

Result: after every crash backoff, the next cycle fires with zero additional sleep,
defeating the cadence and burning API rate limits.

**Fix:** Set `last_cycle_started = time.monotonic()` after crash sleep.

---

### MY-ORCH-002 — Stuck ticker threads leak on BUG-178 timeout
**Severity:** P2 | **Subsystem:** orchestration | **Confidence:** 88%
**File:** `portfolio/main.py:643-646`

After ticker pool timeout, `pool.shutdown(wait=False, cancel_futures=True)` doesn't
kill running threads — they persist for the process lifetime. After repeated BUG-178
timeouts, the process accumulates leaked threads with their file descriptors and
network connections.

---

### MY-RISK-001 — Drawdown circuit breaker silently uses stale prices
**Severity:** P1 | **Subsystem:** portfolio-risk | **Confidence:** 85%
**File:** `portfolio/risk_management.py:117-139`

`check_drawdown` loads `agent_summary.json` to price holdings. If the summary is stale
(written hours ago) but non-empty, it passes the `if summary:` check. The WARNING at
lines 132-138 only fires when the file is missing/empty. A stale summary during a flash
crash computes `current_value` too high, keeping drawdown below threshold and preventing
the circuit breaker from firing.

**Fix:** Check `summary.get("ts")` staleness. If older than 5 minutes with live
positions, trip conservatively.

---

### MY-RISK-002 — trade_guards.py has no lock around state file R/M/W
**Severity:** P1 | **Subsystem:** portfolio-risk | **Confidence:** 85%
**File:** `portfolio/trade_guards.py:31-42,97,229`

`_load_state()` / `_save_state()` have no lock. Under the 8-worker ThreadPoolExecutor,
two threads calling `record_trade` simultaneously can clobber each other's writes.
Cooldown records can be silently lost, allowing trades that should be blocked.

`portfolio_mgr.py` uses `update_state()` with per-path locking for exactly this reason.

**Fix:** Add a module-level `threading.Lock()` around the load-mutate-save sequence.

---

### MY-AVZ-001 — No 3% stop-proximity guard exists anywhere
**Severity:** P1 | **Subsystem:** avanza-api | **Confidence:** 90%
**File:** `portfolio/avanza_session.py:715+`, all stop-loss placement functions

`.claude/rules/metals-avanza.md` and memory state: "NEVER place a stop-loss within 3%
of current bid." No stop-loss placement function validates this. A stop within 3% of
bid on a 5x warrant triggers almost immediately on normal intraday volatility.

**Fix:** Add bid-proximity check in `place_stop_loss()`.

---

### MY-AVZ-002 — Confirmed orders use TOTP path, bypass order lock
**Severity:** P1 | **Subsystem:** avanza-api | **Confidence:** 85%
**File:** `portfolio/avanza_orders.py:17`

`avanza_orders.py` imports from `avanza_control`, which re-exports from `avanza_client`
(TOTP path). The BankID session functions in `avanza_session` have `avanza_order_lock`.
The TOTP functions don't. Confirmed orders bypass cross-process locking entirely.

**Fix:** Import from `avanza_session` instead of `avanza_control`.

---

### MY-INFRA-001 — telegram_poller uses raw file I/O for config
**Severity:** P2 | **Subsystem:** infrastructure | **Confidence:** 92%
**File:** `portfolio/telegram_poller.py:199-201`

Raw `open() + json.load()` for config.json. `PermissionError` propagates uncaught.
On Windows where config.json is a symlink, antivirus locking causes intermittent crashes.

**Fix:** Use `file_utils.load_json()`.

---

### MY-INFRA-002 — log_rotation doesn't fsync before os.replace
**Severity:** P2 | **Subsystem:** infrastructure | **Confidence:** 88%
**File:** `portfolio/log_rotation.py:243-249`

`rotate_jsonl` writes temp file then `os.replace()` without `f.flush() + os.fsync()`.
On power loss, the JSONL file can be replaced with truncated data.

**Fix:** Add `f.flush(); os.fsync(f.fileno())` before `os.replace()`.

---

### MY-INFRA-003 — weekly_digest loads entire signal_log.jsonl into memory
**Severity:** P2 | **Subsystem:** infrastructure | **Confidence:** 90%
**File:** `portfolio/weekly_digest.py:27-29`

The 4-hour digest correctly uses `load_jsonl_tail()` (BUG-109) for the 68MB+ file.
The weekly digest was never updated with the same fix.

**Fix:** Use `load_jsonl_tail()` with appropriate limits.

---

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| P1       | 6     | MY-SC-001, MY-ORCH-001, MY-RISK-001, MY-RISK-002, MY-AVZ-001, MY-AVZ-002 |
| P2       | 6     | MY-SC-002, MY-SC-003, MY-ORCH-002, MY-INFRA-001, MY-INFRA-002, MY-INFRA-003 |

**Top 3 most dangerous:**
1. **MY-AVZ-001**: No 3% stop-proximity guard — documented CRITICAL rule with zero enforcement
2. **MY-RISK-001**: Stale-price drawdown circuit breaker — fails silently during flash crash
3. **MY-ORCH-001**: Post-crash zero-sleep cycling — API rate limit exhaustion after every crash
