# Adversarial Code Review: Orchestration Subsystem (Claude Reviewer)

## Executive Summary

**5 P1 findings, 7 P2 findings, 6 P3 findings.**

The orchestration subsystem is mature with extensive bug-fix history visible in comments. However, critical issues remain: agent invocation bypasses the centralized `claude_gate.py`, a negative-hour bug in `session_calendar.py` crashes during summer DST, multi-agent specialist subprocess leaks, trigger state race conditions, and unprotected digest calls can crash post-cycle processing.

---

## P1 Findings (Critical)

### P1-1: `invoke_agent` bypasses `claude_gate.py` entirely
**File:** `portfolio/agent_invocation.py:451`
**Title:** Layer 2 main agent does not route through centralized gate

`claude_gate.py` states: "All callers MUST route through `invoke_claude()`. Direct `subprocess.Popen` calls are FORBIDDEN." Yet `invoke_agent()` calls `subprocess.Popen` directly, bypassing:
1. The `_invoke_lock` serializing all Claude invocations
2. The `CLAUDE_ENABLED` kill switch
3. The `_count_today_invocations()` rate-limit warning
4. The `_popen_kwargs_for_tree_kill()` process-group creation
5. The `_kill_process_tree()` cleanup

Without `CREATE_NEW_PROCESS_GROUP`, `_kill_overrun_agent` at line 248 uses `taskkill /T /PID` which on Windows doesn't reliably kill grandchild processes (MCP servers, Node.js helpers). These leak as zombies.

**Fix:** Refactor `invoke_agent` to use `claude_gate.invoke_claude()` or at minimum apply `_popen_kwargs_for_tree_kill()` and honor `CLAUDE_ENABLED` + `_invoke_lock`.

### P1-2: Negative UTC hour crash when CET hour < DST offset
**File:** `portfolio/session_calendar.py:85`
**Title:** `_make_session_end` produces negative hour in summer

```python
utc_hour = cet_hour - offset
```
In summer (offset=2), crypto session `open_cet = (0, 0)` → `0 - 2 = -2`. `now.replace(hour=-2)` raises `ValueError: hour must be in 0..23`. The crypto path is currently guarded by early return, but any future session with open/close < offset crashes.

**Fix:** `utc_hour = (cet_hour - offset) % 24` with date rollover handling.

### P1-3: Multi-agent specialist log file handles leak on Popen failure
**File:** `portfolio/multi_agent_layer2.py:166-168`

```python
log_fh = open(log_path, "w", encoding="utf-8")
proc = subprocess.Popen(...)  # If raises, log_fh leaks
proc._log_fh = log_fh
```
On Windows, leaked handles prevent file deletion/rotation.

**Fix:** Wrap in try/except that closes `log_fh` on failure.

### P1-4: Trigger state race between `check_triggers` and `update_tier_state`
**File:** `portfolio/trigger.py:312, main.py:800-804`

`check_triggers()` saves state → `classify_tier()` re-reads from disk → `update_tier_state()` re-reads and saves again. If `check_triggers` saved sustained_counts changes, `update_tier_state` overwrites them. The `state=` parameter exists but the caller doesn't use it.

**Fix:** Pass state object through: `state = _load_state(); tier = classify_tier(reasons_list, state=state); update_tier_state(tier, state=state)`.

### P1-5: `_maybe_send_digest` not wrapped in try/except in `_run_post_cycle`
**File:** `portfolio/main.py:281`

Every other post-cycle task uses `_track()`. This one is bare. If it raises, all subsequent tasks abort: daily digest, accuracy degradation, log rotation, JSONL pruning.

**Fix:** Wrap in `_track("digest", _maybe_send_digest, config)`.

---

## P2 Findings (High)

### P2-1: Agent globals not thread-safe
**File:** `portfolio/agent_invocation.py:27-42`

`_agent_proc`, `_agent_log`, `_agent_start` etc. are mutable globals with no lock. While currently single-threaded for agent ops, `_process_ticker` runs in ThreadPoolExecutor.

### P2-2: Stack overflow counter loaded at import, no auto-recovery
**File:** `portfolio/agent_invocation.py:68`

`_consecutive_stack_overflows` loaded once at import. If >= `_MAX_STACK_OVERFLOWS`, Layer 2 is permanently disabled with no automatic recovery even after weeks.

### P2-3: `wait_for_specialists` blocks main loop for up to 30s
**File:** `portfolio/agent_invocation.py:384-388`

Synchronous blocking call freezes heartbeat, crash detection, and trigger processing.

### P2-4: Session end doesn't handle day boundary crossing
**File:** `portfolio/session_calendar.py:82-88`

Even with P1-2 modulo fix, `now.replace(hour=23)` for a "yesterday 23:00 UTC" session gives wrong day.

### P2-5: Config not refreshed between cycles
**File:** `portfolio/main.py:1038, 1085`

Config loaded once outside while loop, stale for `_run_post_cycle`. Changes to config.json don't take effect until restart.

### P2-6: `process_lock.py` silently no-ops when neither msvcrt nor fcntl available
**File:** `portfolio/process_lock.py:60-66`

Lock acquisition succeeds without actually locking, allowing multiple processes.

### P2-7: Post-cycle maintenance skipped after crash
**File:** `portfolio/main.py:1090-1101`

JSONL pruning, log rotation, digest sending all skip after crash. If crash-looping, these tasks go hours without running.

---

## P3 Findings (Medium)

### P3-1: `_TICKER_PAT` regex misses post-trade and F&G trigger formats
**File:** `portfolio/main.py:243`

### P3-2: Bare `except Exception` masks type errors
**File:** Multiple locations in main.py

### P3-3: `_extract_ticker` falls back to "XAG-USD" hardcoded
**File:** `portfolio/agent_invocation.py:137`

### P3-4: `set()` stored in JSON-destined state dict
**File:** `portfolio/trigger.py:151`

### P3-5: Grace period flag resets on module reimport
**File:** `portfolio/trigger.py:54`

### P3-6: Crypto scheduler uses naive datetime in log
**File:** `portfolio/crypto_scheduler.py:310`
