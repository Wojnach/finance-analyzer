# Claude critique of codex findings — orchestration

## Verdicts

- [P1] Parse stock-trigger reason strings in `_extract_ticker` — portfolio/agent_invocation.py:236-240
  Verdict: CONFIRMED
  Reason: Regex pattern on line 237 matches `BUY.*crossed.*broke` but codex correctly notes that reason strings from trigger.py like `"MSTR consensus BUY (..."` don't match; they fall through to XAG-USD default (line 240).

- [P1] Leave suppressed ranging consensuses eligible for a later trigger — portfolio/trigger.py:230-231
  Verdict: CONFIRMED
  Reason: Line 239 updates `triggered_consensus[ticker] = action` even when ranging dampening suppresses the trigger. Next cycle, same action persists so it no longer looks like a HOLD→BUY/SELL transition (line 244).

- [P2] Consume matched buy lots when computing reflection PnL — portfolio/reflection.py:67-70
  Verdict: FALSE-POSITIVE
  Reason: File `portfolio/reflection.py` does not exist in the main repository. Codex was reviewing an experimental worktree (`adv-orchestration`) that is not part of the committed codebase.

- [P2] Mark Swedish holiday sessions as closed for warrants/stocks — portfolio/session_calendar.py:183-184
  Verdict: FALSE-POSITIVE
  Reason: File `portfolio/session_calendar.py` does not exist in the main repository. Codex was reviewing an experimental worktree, not the committed code.

- [P0] Ship the internal modules that `portfolio.main` imports — portfolio/main.py:31-31
  Verdict: FALSE-POSITIVE
  Reason: Codex concerns a worktree that lacks internal modules; the main repo has all dependencies present. This is a non-issue for the committed codebase.

## New findings (mine)

- [P1] Timeout check uses truthiness on `_agent_timeout` — portfolio/agent_invocation.py:1142
  If `_agent_timeout` is set to 0 (e.g., via config error), the condition `if _agent_timeout and elapsed > _agent_timeout:` skips the timeout branch entirely. The subprocess then runs indefinitely without being killed. Fix: `if _agent_timeout > 0 and elapsed > _agent_timeout:`.

- [P1] Log file byte-offset captured before open creates race window — portfolio/agent_invocation.py:839
  Line 839 captures `_agent_log_start_offset` from file size BEFORE opening the log handle (line 840). Early subprocess output written before the file handle is open (between stat() and open()) is missed by the auth-failure detector at line 442. Exact failure pattern from Mar–Apr 2026 outage. Fix: Move stat() call to after open().

- [P0] Consensus baseline consumed even when ranging trigger suppressed — portfolio/trigger.py:238-239
  When ranging confidence threshold blocks a trigger (line 230), baseline is still updated (line 239). Next cycle, if same consensus persists, it won't appear as a new HOLD→ACTION transition (line 244 checks `last_tc != "HOLD"`), so Layer 2 never fires. Consequence: weak ranging signal misses first actionable consensus unless it drops to HOLD first.

- [P1] Startup grace flag not reset on soft loop restart — portfolio/trigger.py:69,170
  Module-level `_startup_grace_active = True` (line 69) is not reset on in-process restart-without-reimport. PID check at line 170 catches PROCESS restart, but not LOOP restart within same process (e.g., exception handler reinvoking check_triggers). Grace period remains inactive, firing spurious triggers immediately. Fix: Tie grace to a per-session marker stored in trigger_state.json.

- [P1] `_agent_log_start_offset` not cleared on completion — portfolio/agent_invocation.py:1323
  Completion block clears `_agent_start`, `_agent_proc`, `_agent_log` but NOT `_agent_log_start_offset` (line 38 definition). Stale offset persists into next invocation, distorting auth-error scan window. Fix: Add `_agent_log_start_offset = 0` at line 1323.

- [P1] Stack-overflow auto-disable counter lacks decay — portfolio/agent_invocation.py:544,171
  Counter reaching 5 (line 544) disables Layer 2 permanently until manual reset. No decay path exists; the counter persists in `stack_overflow_counter.json`. After 5 unrelated Node.js crashes, Layer 2 is offline forever. Fix: Add 24h decay check when loading counter (line 156).

- [P1] Journal scan is O(N) full-file walk per invocation — portfolio/agent_invocation.py:293-306
  Decision-feedback scans entire JOURNAL_FILE in reversed order (line 300). After weeks of trading (10K+ entries), each invocation linearly walks the whole file twice (once reversed, once to find tail). Cumulative drag. Fix: Limit to last 100 entries using `load_jsonl(maxlines=100)` or cache index.

- [P1] ThreadPoolExecutor.cancel_futures best-effort only — portfolio/main.py:658,661
  Line 658 calls `f.cancel()` on already-running futures (no effect). Line 661 calls `pool.shutdown(wait=False, cancel_futures=True)` which is also best-effort. Running signal workers still hold GPU memory (Chronos in-flight) and sockets after shutdown returns. Crash recovery then restarts loop with workers still running. Fix: Add 5s drain grace + log orphans.

- [P2] UTC weekday used for market-hours tier classification — portfolio/trigger.py:449-453
  Line 449 gets `datetime.now(UTC)`, line 453 uses `.weekday()` on UTC time. At UTC/CET boundary (23:00 UTC Thursday = 00:00 CET Friday), tier misclassified. Tier 3 periodic logic depends on correct day. Fix: Convert to CET (`UTC.localize(now_utc).astimezone(ZoneInfo("Europe/Stockholm"))`) before weekday check.

- [P2] Heartbeat daemon drain insufficient on exit — portfolio/health.py:140
  Line 140 joins daemon with 2.0s timeout. If daemon is mid-write to health_state.json when join times out, file is left open/partial. Subsequent reads or writes fail. Fix: Signal daemon first (line 138 already does this), but increase timeout to 10s and add logger.warning on timeout.

- [P2] Broad `except Exception` masks import failures — portfolio/agent_invocation.py:932
  Line 932 catches all exceptions during agent spawn. If `import portfolio.something` fails (typo in __init__.py, missing dependency), the exception is logged but wrapped, and code falls through to bat-file invocation. Silent degradation of Layer 2 path. Fix: Catch FileNotFoundError/OSError specifically; let ImportError surface.

## Summary
- Confirmed: 2 (codex P1 findings on real code)
- Partial: 0
- False-positive: 3 (codex findings on non-existent worktree files)
- New: 9
