# Claude Review — orchestration

## P0 (money-losing or data-corrupting)

- `portfolio/journal.py:28-40` — `load_recent` reads JSONL with bare `open()` while Layer 2 atomic-renames over the same file
  ```python
  with open(JOURNAL_FILE, encoding="utf-8") as f:
      for line in f:
  ```
  `atomic_append_jsonl` does an atomic rename on Windows. If `load_recent()` holds an open read handle when the appender renames `.tmp` over `layer2_journal.jsonl`, the rename can fail with `PermissionError` on Windows. `write_context()` is called from `invoke_agent()` on every trigger and from `autonomous.py` every trigger when Layer 2 is disabled. Failure mode: journal entry NOT written → `check_agent_completion()` sees `journal_written=False` → loop contract fires CRITICAL violation → fix-agent dispatcher spawns. Project rule is explicit: never bare `open(...)`; use `file_utils.load_jsonl()`. Confidence 85.

- `portfolio/multi_agent_layer2.py:166-178` — specialist log file leaked when `Popen()` raises between `open()` and the `proc._log_fh = log_fh` assignment
  ```python
  log_fh = open(log_path, "w", encoding="utf-8")
  proc = subprocess.Popen(cmd, ..., stdout=log_fh, ...)
  proc._log_fh = log_fh  # attach for cleanup
  ```
  If `Popen()` raises (`PermissionError`, `OSError` on PATH lookup), `log_fh` never closes. `wait_for_specialists` closes `proc._log_fh` in `finally`, but if `Popen` failed `proc._log_fh` was never set. On Windows, leaked fd locks the log file exclusively. Three specialists × leaked handle per multi-agent invocation compounds across retries. Confidence 85.

- `portfolio/trigger.py:130-148` — `_check_recent_trade()` only catches `KeyError, AttributeError`; `JSONDecodeError`/`OSError` propagates
  ```python
  except (KeyError, AttributeError) as exc:
      logger.warning("Failed to parse portfolio file %s: %s", pf_file, exc)
  ```
  Windows atomic-rename on portfolio_state.json mid-read is a documented failure class. A `JSONDecodeError` or `OSError` crashes `check_triggers()`, caught at `run()` level — generic crash log, no specific diagnostic. `trigger_state.json` not updated → next cycle same crash. `trade_detected = False` default permanently suppresses the post-trade reassessment trigger until file is repaired. Confidence 80.

## P1 (high-confidence bugs)

- `portfolio/agent_invocation.py:739-741` — auth-scan offset captured BEFORE `open()`, vulnerable to log rotation between stat+open
  ```python
  _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
  log_fh = open(agent_log_path, "a", encoding="utf-8")
  ```
  `log_rotation.rotate_all()` runs hourly in `_run_post_cycle()`. If rotation runs between stat and open, the new `agent.log` starts at offset 0 but `_agent_log_start_offset` holds the old file's size (e.g. 8MB). `_scan_agent_log_for_auth_failure()` later seeks to 8MB on a 0-byte file — seek past EOF succeeds, read returns empty, **"Not logged in" in the new session is invisible**. This is exactly the failure class that caused the March-April 2026 outage. Fix: capture offset from the opened handle: `_agent_log_start_offset = log_fh.seek(0, 2)`. Confidence 80.

- `portfolio/multi_agent_layer2.py:193-210` — `wait_for_specialists` runs `proc.wait()` SERIALLY; first specialist's timeout consumes all remaining budget
  ```python
  for proc in procs:
      remaining = max(1, deadline - time.time())
      ...
      try:
          proc.wait(timeout=remaining)
  ```
  With 3 specialists and 30s `specialist_timeout_s`: if first specialist hangs 29s, second and third have 1s each — killed regardless of progress. **Multi-agent path effectively never produces specialist output**, silently falls back to synthesis with empty reports every invocation, wasting tokens. Confidence 85.

- `portfolio/bigbet.py:173-181` — `CLAUDECODE` env var not popped before `subprocess.run(["claude", ...])`
  ```python
  bigbet_env = os.environ.copy()
  bigbet_env["PF_HEADLESS_AGENT"] = "1"
  result = subprocess.run(
      ["claude", "-p", prompt, "--max-turns", "1"],
      ..., env=bigbet_env,
  )
  ```
  Both `agent_invocation.py:745` and `multi_agent_layer2.py:143` correctly pop `CLAUDECODE` and `CLAUDE_CODE_ENTRYPOINT`. Bigbet does not. Inheriting `CLAUDECODE` causes "nested session" error that exits 0 with error text on stdout — silent failure, parsed as `probability=None`. In production (PF-DataLoop scheduled task) typically unset; in interactive dev it's set. Bigbet always appears to work in scheduled-task mode and silently fails in dev — making the bug hard to catch. Fix: add `bigbet_env.pop("CLAUDECODE", None)` and `bigbet_env.pop("CLAUDE_CODE_ENTRYPOINT", None)`. Confidence 88.

- `portfolio/trigger.py:432-434` — `classify_tier` uses hour-only `now_utc.hour < close_hour`, misses NYSE :30 open
  ```python
  market_open = now_utc.weekday() < 5 and eu_open <= now_utc.hour < close_hour
  ```
  `is_us_stock_market_open()` in `market_timing.py` uses minute precision (NYSE opens :30). `classify_tier` uses only hours. From 13:00–13:29 UTC in summer, classify thinks US is open and `is_us_stock_market_open()` returns False. Causes T3 "full review" classification during pre-NYSE window — wastes T3 token budget. Confidence 80.

- `portfolio/main.py:865-885` — Sequential `heartbeat_keepalive()` blocks for bigbet+iskbets aren't measured by loop contract
  ```python
  with heartbeat_keepalive():
      check_bigbet(...)
  ...
  with heartbeat_keepalive():
      check_iskbets(...)
  ```
  `report.cycle_end` is set inside `run()` BEFORE bigbet/iskbets run. The contract measures `run()` only, not total wall-clock. Total post-cycle block: up to 300s+ (bigbet 150s + iskbets 150s) — invisible to monitoring. Not a contract violation but a monitoring blind spot for genuinely long cycles. Confidence 80.

## P2 (concerns / smells)

- `portfolio/trigger.py:64-95` — `_mono_start` (a `time.monotonic()` value) serialized to `trigger_state.json`; comment claims restart resets it but code uses stale disk value
  ```python
  state_dict[key] = {
      "value": value,
      "count": prev["count"] + 1,
      "_mono_start": prev.get("_mono_start", mono_now),
  }
  ```
  On restart, `time.monotonic()` resets near zero. Deserialized `_mono_start` could be a large float from prior run. `mono_now - _mono_start` becomes huge negative — duration_ok always False on first cycle. Safe direction by accident, but a long-running process accumulating large monotonic values means a freshly-restarted process could have `mono_now` already at several thousand seconds, and the stale disk value would appear valid. Comment lies about behavior. Confidence 82.

- `portfolio/journal.py:70-74` — `_entry_age_hours()` doesn't normalize timezone-naive entries
  ```python
  def _entry_age_hours(entry, now=None):
      if now is None:
          now = datetime.now(UTC)
      ts = datetime.fromisoformat(entry["ts"])
      return (now - ts).total_seconds() / 3600
  ```
  If `entry["ts"]` is a legacy naive timestamp (pre-2026 entries or any path that uses `datetime.now().isoformat()` without timezone), subtracting from aware `now` raises `TypeError`. Crashes `build_context()` → `layer2_context.md` not written → all subsequent Layer 2 reads stale context. Fix: `if ts.tzinfo is None: ts = ts.replace(tzinfo=UTC)`. Confidence 80.

## Did NOT find

1. CLAUDECODE inheritance in main Layer 2 path — `agent_invocation.py:745` correctly pops both vars.
2. Trigger storms / debouncing absent — `triggered_consensus` correctly prevents re-fire on same consensus state.
3. `_agent_proc` race between `invoke_agent` and `check_agent_completion` — both called serially from main loop.
4. `_consecutive_crashes` not resetting — `_reset_crash_counter()` correctly called after successful run.
5. Naive/aware mismatch in `market_timing.py` — all `datetime.now(UTC)` with imported UTC, comparisons aware.
6. heartbeat_keepalive daemon thread surviving past exit — Python daemon threads auto-killed.
7. Autonomous + Layer 2 journal write conflict — mutually exclusive by config gate at main.py:839/849.
8. Journal index BM25 unbounded growth — instantiated fresh per call.
