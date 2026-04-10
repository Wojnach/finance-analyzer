# Agent Adversarial Review: orchestration

**Agent**: feature-dev:code-reviewer
**Subsystem**: orchestration (6,412 lines, 12 files)
**Duration**: ~254 seconds
**Findings**: 12 (3 P1, 7 P2, 2 P3)

---

## P1 Findings

### A-OR-1: Wrong TimeoutError Type Caught — Ticker Hang Crashes Loop [P1]
- **File**: `portfolio/main.py:575`
- **Description**: `as_completed(..., timeout=180)` raises `concurrent.futures.TimeoutError`, but the bare `except TimeoutError:` catches only the built-in `TimeoutError`. On Python <3.11, these are distinct types. The futures timeout propagates uncaught, triggering crash handler + exponential backoff + Telegram alert.
- **Impact**: Every ticker hang produces a false crash alert and wasted backoff time.
- **Fix**: `from concurrent.futures import TimeoutError as FuturesTimeoutError` and catch both.

### A-OR-2: classify_tier and update_tier_state Read trigger_state.json Independently [P1]
- **File**: `portfolio/main.py:711-714`, `portfolio/trigger.py:319-375`
- **Description**: Three separate disk reads of `trigger_state.json` within milliseconds. If state changes between reads, tier classification and state update see different data.
- **Impact**: Unnecessary T3 invocations (expensive full reviews burning Claude budget).
- **Fix**: Load state once, pass to both `classify_tier(state=)` and `update_tier_state(state=)`.

### A-OR-3: Multi-Agent Mode Blocks Main Loop for 30s Synchronously [P1]
- **File**: `portfolio/agent_invocation.py:256-259`
- **Description**: `wait_for_specialists(procs, timeout=30)` is synchronous inside `invoke_agent()`. No heartbeat, health check, or trigger update during the block.
- **Impact**: 5% of cycle budget wasted. Stale trigger data during fast-moving market events.
- **Fix**: Disable multi-agent until async TODO is implemented, or use background thread.

---

## P2 Findings

### A-OR-4: _maybe_send_digest Not Wrapped in _track — Crash Aborts All Post-Cycle Tasks [P2]
- **File**: `portfolio/main.py:280`
- **Description**: Every other post-cycle task uses `_track()` for error isolation. `_maybe_send_digest` is called directly. An exception in digest aborts JSONL pruning, health refresh, log rotation, etc.
- **Fix**: `_track("digest", _maybe_send_digest, config)`.

### A-OR-5: Stale Config in Post-Cycle — Config Changes Require Restart [P2]
- **File**: `portfolio/main.py:948/995`
- **Description**: `config = _load_config()` called once before the while loop. `_run_post_cycle(config)` always uses the stale value. `run()` loads its own fresh config.
- **Fix**: Move `config = _load_config()` inside the while loop body.

### A-OR-6: Loop Contract MAX_CYCLE_DURATION_S Equals Pool Timeout [P2]
- **File**: `portfolio/loop_contract.py:22`, `portfolio/main.py:554`
- **Description**: `MAX_CYCLE_DURATION_S = 180` = `_TICKER_POOL_TIMEOUT = 180`. Valid slow cycles always generate spurious warnings.
- **Fix**: Set `MAX_CYCLE_DURATION_S = 360`.

### A-OR-7: BUY↔SELL Direction Flips Poison Trigger Consensus [P2]
- **File**: `portfolio/trigger.py:177-181`
- **Description**: Rapid signal flips (BUY→SELL→BUY) silently update `triggered_consensus`, preventing section 1 trigger and requiring sustained flip wait. Rapid crypto oscillations miss signals.
- **Fix**: Only update `triggered_consensus` when a trigger actually fires.

### A-OR-8: Naive vs Aware Datetime in digest.py — Legacy Entries Silently Skipped [P2]
- **File**: `portfolio/digest.py:72,106,128`
- **Description**: `fromisoformat()` on naive timestamps compared with aware `cutoff` → TypeError caught and entry skipped. Digest undercounts Layer 2 invocations.
- **Fix**: Normalize: `if dt.tzinfo is None: dt = dt.replace(tzinfo=UTC)`.

### A-OR-9: Multi-Agent Log File Handle Leak [P2]
- **File**: `portfolio/multi_agent_layer2.py:155-163`
- **Description**: If `Popen` raises after log file open but before `procs.append`, the file handle is never closed. On Windows, prevents next cycle from opening the same file.

### A-OR-10: cleanup_reports() Never Called — Stale Specialist Reports [P2]
- **File**: `portfolio/multi_agent_layer2.py:206-216`
- **Description**: Specialist report files accumulate. Synthesis agent can read stale reports from previous invocations. `cleanup_reports()` is defined but never imported or called.

---

## P3 Findings

### A-OR-11: weekly_digest Loads Entire 68MB signal_log.jsonl [P3]
- **File**: `portfolio/weekly_digest.py:29,154`
- **Description**: Uses `load_jsonl(path)` with no limit. Should use `load_jsonl_tail(path, max_entries=10000)`.

### A-OR-12: weekly_digest P&L Ignores Unrealized Gains [P3]
- **File**: `portfolio/weekly_digest.py:52`
- **Description**: `pnl_sek = cash - initial` — ignores holdings value. Shows systematically wrong P&L to user.
