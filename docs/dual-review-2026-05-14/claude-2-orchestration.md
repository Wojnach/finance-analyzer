# Adversarial Review — 2 orchestration (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/agent_invocation.py:572` — `_kill_overrun_agent` clears `_agent_proc = None` unconditionally even when `kill_ok=False`.
  ```python
  try:
      _agent_proc.wait(timeout=15)
  except subprocess.TimeoutExpired:
      if kill_ok:
          logger.error("Agent pid=%s did not exit after kill+15s wait", pid)
      kill_ok = False
  ...
  _agent_proc = None
  return kill_ok
  ```
  Docstring (line 506-511) promises: "the caller must NOT spawn a replacement in that case because the old process may still be holding resources." But the caller observes `_agent_proc is None` on the *next* cycle and the singleton gate in `invoke_agent` (line 581-587, only checks stack-overflow counter) does NOT re-check kill_ok. So a Layer 2 process that survives `taskkill /F /T` for 15+ s leaks while a fresh `claude -p` Popen starts, both appending to `data/agent.log` and racing on portfolio_state writes. Fix: keep `_agent_proc` set when kill_ok=False; only clear on confirmed exit.

## P1 — high-confidence bugs (should fix)

- `portfolio/agent_invocation.py:597-619` — auth-error cooldown fails open on I/O.
  ```python
  try:
      recent = load_jsonl(INVOCATIONS_FILE)
      for entry in reversed(recent[-50:]):
          ...
          if status == "auth_error" and ts:
              ...
              if age < 1800:
                  ...
                  return False
          break
  except Exception as e:
      ...   # logged, swallowed → invocation proceeds
  ```
  If `load_jsonl` raises (transient Windows file lock, partial write being replaced, etc.), we fall through and spawn the doomed `claude -p`. The whole point of the 30-min cooldown was to break the "8 spawns within 30 min on 2026-05-10" auth storm — failing open re-opens that hole. Either gate to ≤ N read failures per minute, or treat read-failure as cooldown-active (fail closed).

- `portfolio/journal.py:23-40` — `load_recent` uses raw `with open(JOURNAL_FILE) as f` and `json.loads` per line. Violates the project's atomic-I/O rule (CLAUDE.md: "use file_utils.atomic_write_json, load_json, atomic_append_jsonl"). On Windows, reading a file mid-`atomic_append_jsonl` rename can race; this caller is invoked from agent_invocation's `last_jsonl_entry` shim and from the dashboard. Switch to `load_jsonl_tail(JOURNAL_FILE, max_entries=max_entries*4)` filtered by cutoff.

- `portfolio/agent_invocation.py:562` — when `_agent_tier` is `None` (timeout path called without prior invoke), the label becomes the literal string `"layer2_tNone_timeout"`. The critical_errors entry's caller field then lies. Coerce to `0` or `"unknown"`.

- `portfolio/agent_invocation.py:946-953` — `subprocess.Popen(..., stdout=log_fh, stderr=subprocess.STDOUT)` with `log_fh = open(agent_log_path, "a", encoding="utf-8")`. The auth-failure detector relies on `_agent_log_start_offset = agent_log_path.stat().st_size` (line 913) being captured BEFORE Popen. That's correct. But the file is opened in text mode and the subprocess writes binary bytes via OS-level fd — on Windows, text-mode buffering can introduce CRLF translation that breaks the byte offset. Open in `"ab"` to keep the offset honest.

- `portfolio/trigger.py:111` — `_load_state` calls `load_json(STATE_FILE, default={})`. If `trigger_state.json` is corrupt (e.g. interrupted write before the atomic rename landed — only possible if someone bypassed atomic_write_json, but the dashboard or older code might), the entire trigger baseline silently resets. No alert. After a corruption, every subsequent run re-fires every "first observation" trigger storm. Fix: detect default={} return on a non-fresh install and write a critical_errors entry.

- `portfolio/main.py:608-616` — `pool = ThreadPoolExecutor(...)` opened without context manager (deliberate per OR-I-001 comment), but the manual `pool.shutdown(wait=False)` (presumably later in the function) leaks threads on timeout. Each timed-out cycle adds up to 8 zombie ticker threads that never get joined. Memory creeps and the next cycle's executor has thread contention. Use `as_completed(..., timeout=...)` then explicitly cancel pending + log thread IDs that don't terminate within a grace window.

- `portfolio/main.py:541` — log line uses `ind['rsi'], ind['macd_hist']` direct subscript; if either is missing (partial fetch from collect_timeframes), the log raises KeyError. Wraps in outer try/except at line 558, which converts to a generic error and returns None — silently skipping the ticker. The user-facing "%s: insufficient data" message at line 496 only triggers when ind is None. The KeyError path produces "%s: <KeyError repr>" to ERROR log without telling operators that 4/5 tickers passed but RSI key was missing on the 5th. Use `.get(..., 0)` for log formatting.

## P2 — concerns / smells (worth addressing)

- `portfolio/agent_invocation.py:1226` — `if _agent_timeout and elapsed > _agent_timeout` uses `_agent_timeout` from module global. After a completion, `_agent_timeout` is NOT cleared (search for the reset). If a Tier 1 (180s) completes successfully and a Tier 3 invocation immediately fails before `_agent_timeout = timeout` is reset at line 943, the watchdog could compare elapsed against the stale T1 budget. Cosmetic in current flow (assignment at 943 happens before Popen at 946), but a future refactor could trip it. Reset on completion.

- `portfolio/agent_invocation.py:907-913` — `_agent_log_start_offset` captured via `agent_log_path.stat().st_size`. If `data/agent.log` was rotated between cycles (PF-LogRotate runs daily) and the new file is empty, this is 0 — correct. But if rotation happens DURING this code (window: stat() and Popen), the new subprocess writes from offset 0 while we recorded N. Result: auth-error scan reads zero bytes ("file shorter than expected", IndexError suppressed). Coordinate with log_rotation or use a marker line.

- `portfolio/main.py:467-470` — `report = CycleReport(...)`. The CycleReport is built but I see no `report.finalize()` or `report.write()` in this excerpt. If finalize is skipped on exception paths (try/except inside the executor block but no outer try around CycleReport), invariant violations would be lost. Verify finalize is in a `finally`.

- `portfolio/digest.py` — every 4h digest send re-reads recent journal entries via `journal.load_recent` (same non-atomic open as P1 above). Sharing a fix here helps both.

- `portfolio/multi_agent_layer2.py:163-185, 210` — specialist Layer 2 subprocesses (bull/bear/synthesizer) bypass `claude_gate._invoke_lock` and `_kill_process_tree`. proc.kill() on Windows leaves child Node.js descendants. Auth-error scan only happens post-wait, so a stuck specialist can run past budget and orphan a process tree.

- `portfolio/autonomous.py` — fallback decision path runs when Layer 2 is disabled. Search for `DRY_RUN` or trade execution: this module currently produces "recommendations only" per CLAUDE.md but cannot independently verify; spot-check that no `place_*` call lurks.

- `portfolio/agent_invocation.py:1338-1342` — `with suppress(Exception): new_journal_entry = last_jsonl_entry(JOURNAL_FILE)` after journal_written check. If a different process (metals_loop or autonomous reflection) writes to JOURNAL_FILE between our check and read, we pick up its entry as if it were the agent's, then feed `_write_fishing_context` from foreign data. Add a uniqueness key (entry hash) and assert match.

## Did NOT find

1. **Silent failures**: covered — see P0/P1 above. Auth detector itself (claude_gate.detect_auth_failure) is correctly wired into both timeout and completion paths.
2. **Race conditions**: `_completion_lock` correctly serializes watchdog vs run() consumers (line 1202). Singleton gate exists but P0 kills it.
3. **Money-losing bugs**: no direct order math in this subsystem.
4. **State corruption**: `atomic_write_json` for trigger_state, health_state — correct. P1 finding on `journal.load_recent` is read-side though.
5. **Logic errors that pass tests**: P0 _agent_proc leak likely not caught by tests because tests usually mock subprocess wait().
6. **Resource leaks**: P1 thread-pool leak.
7. **Time/timezone bugs**: `datetime.now(UTC)` everywhere; `time.monotonic()` for elapsed measurements; `_safe_elapsed_s()` covers monotonic-vs-wallclock drift.
8. **API misuse**: claude CLI args `-p`, `--allowedTools`, `--max-turns` are valid. NODE_OPTIONS stack-size bump is appropriate.
9. **Trust boundary violations**: claude CLI prompt is a Python string built from internal data, not external — but `prompt` body content from agent_summary could contain `--allowedTools` look-alikes if any ticker name had special chars. ALL_TICKERS is hard-coded; safe today.
10. **Incorrect partial-state assumptions**: P1 finding `ind['rsi']` direct access.
