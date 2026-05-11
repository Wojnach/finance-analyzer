# Claude adversarial review: orchestration

## Summary

Orchestration code shows scar tissue from real outages (March-April 2026
auth blackout, GPU lock wedge, BUG-178, BUG-203, BUG-219) but several
silent-failure and race-condition classes are still live. The most
worrying are: (1) the prewarmer's GPU/thread-lock acquisition path,
(2) two production subprocess sites that still bypass `claude_gate` and
its tree-kill / auth-detection, (3) journal_index / prophecy /
trigger_state read-modify-write paths that are not protected against
concurrent writers, and (4) several telegram / message-store paths
where a Markdown parse failure or `r.text` access on a `None` result
can crash a notification handler. Several of these will manifest as
exact-zero-cost regressions of the same outage we already paid for.

## P0 — Blockers

- `portfolio/agent_invocation.py:846-851` — `agent_env["NODE_OPTIONS"] = "--stack-size=16384"`
  unconditionally OVERWRITES any inherited `NODE_OPTIONS`. If the user
  ever exports `NODE_OPTIONS="--max-old-space-size=8192 --enable-source-maps"`
  for unrelated tooling, those flags are silently dropped for the Layer 2
  subprocess — and on Windows / WSL `NODE_OPTIONS` is often inherited
  from the shell. Bigger: the same overwrite happens in
  `multi_agent_layer2.py:145`. Why it bites: the user reports Claude CLI
  stack overflows ≥5 → Layer 2 auto-disables (line 544). If a future
  Node release moves the stack flag, this hardcoded value silently
  becomes ineffective and the user has no signal that the workaround
  expired. Fix: `agent_env["NODE_OPTIONS"] = f"--stack-size=16384 {os.environ.get('NODE_OPTIONS','')}".strip()`.

- `portfolio/llm_prewarmer.py:299-301` — `query_llama_server(server_slot, "test", n_predict=1)`
  is invoked from inside `flush_llm_batch()` (called on the loop thread
  while still in the post-cycle path). `query_llama_server` grabs
  `_thread_lock` AND a 300s-timeout file lock and synchronously runs a
  full model swap (`_start_server` can sleep up to 30s for VRAM, 90s for
  llama-server startup). Why it bites: in the worst case one prewarm
  call blocks the main loop for **~120 seconds** AFTER the LLM batch has
  already cost 25-40s. That eats most of the 600s cadence budget and
  any subsequent `run_post_cycle` step (digests, prunes, prophecies)
  starves. The `try/except` wrapper at `llm_batch.py:310-314` catches
  exceptions but NOT a slow path. Fix: dispatch `prewarm_next_model`
  on a daemon thread with a strict 30s wallclock guard, or check
  `model_load_safe()` and skip if `_query_free_vram_mb()` < threshold.

- `portfolio/bigbet.py:175-181` — `subprocess.run(["claude", "-p", prompt, "--max-turns", "1"], ..., timeout=30)`
  bypasses `claude_gate.invoke_claude` ENTIRELY. No tree-kill on timeout
  (Claude CLI's grandchildren leak; documented at `claude_gate.py:282-291`),
  no `_invoke_lock` so the 8-worker ticker pool can fan out N parallel
  bigbet invocations, no daily rate-limit counter, no kill-switch
  honored, AND the `CLAUDECODE` env var is NOT cleaned (only
  `PF_HEADLESS_AGENT=1` is set at line 174). Why it bites: `claude_gate.py:9`
  literally says "Direct subprocess.Popen calls are FORBIDDEN".
  This is exactly the leak class that justifies the gate. Fix: route
  through `invoke_claude_text(prompt, caller="bigbet", model="haiku",
  timeout=30, ...)`.

- `portfolio/agent_invocation.py:824-830` — `pf-agent.bat` fallback
  ALSO bypasses `claude_gate`. The env-cleanup at lines 843-855 happens,
  but the bat path takes a `cmd /c` route that may not propagate the
  cleaned env on Windows depending on how the bat file calls `claude`.
  More critically: the fallback runs as Tier 3 (900s) regardless of
  what tier was requested — and `_agent_timeout` is set from the
  requested tier (line 869) at the call site. Result: a T1 fallback
  request via bat has `_agent_timeout=120` but the underlying claude
  invocation can take up to 900s. The completion watchdog will kill
  it at 120s, log "timeout", but the bat wrapper's grandchildren
  outlive the kill (no tree-kill). Fix: align bat-path timeout with
  tier-config, and add tree-kill to `_kill_overrun_agent`.

## P1 — High

- `portfolio/agent_invocation.py:1118-1330` — `_check_agent_completion_locked`
  reassigns module globals while holding `_completion_lock`, but
  `invoke_agent` (line 540-936) reads several of those same globals
  (`_agent_proc`, `_agent_log`, `_agent_log_start_offset`,
  `_journal_ts_before`, `_telegram_ts_before`, transaction counts)
  OUTSIDE the lock at lines 600-885. The big `with _completion_lock`
  at line 574 only covers the reentrancy check. Why it bites: the
  watchdog tick at line 100 can clear `_agent_proc=None` and reset
  txn counts at the exact moment `invoke_agent` is in the middle of
  initializing the next invocation — racy write-write on
  `_patient_txn_count_before`. Fix: widen the lock around lines 832-916.

- `portfolio/agent_invocation.py:541` — `_agent_log_start_offset` is
  declared `global` only inside the `try` at line 838 (`global _agent_log_start_offset`).
  The earlier read at line 428 (`f.seek(_agent_log_start_offset)`)
  in `_scan_agent_log_for_auth_failure` reads the module-level value
  without the lock. If a previous invocation crashed mid-cleanup, the
  offset is stale and we may scan the WRONG slice of agent.log —
  potentially re-flagging the same auth marker over and over (the
  exact pattern that triggered the 2026-04-16 echo-feedback bug in
  `claude_gate.py:120-159`). Fix: snapshot offset into a local before
  the file read.

- `portfolio/claude_gate.py:271-279` — `_count_today_invocations` does a
  full re-parse of `claude_invocations.jsonl` on EVERY invocation to
  check the daily-50 warn threshold. Under high invocation rate the
  file grows unbounded; `load_jsonl` reads the entire file (no tail).
  Why it bites: degrades silently as the file grows. The 5000-entry
  prune in `main.py:357-364` covers `claude_invocations.jsonl` so
  the practical cap is ~5000 lines, but the iteration is still O(N)
  for a check that should be O(1). Fix: maintain a daily counter in
  shared state, reset at UTC midnight.

- `portfolio/multi_agent_layer2.py:196-210` — `wait_for_specialists`
  iterates procs sequentially and uses `proc.wait(timeout=remaining)`.
  But `remaining = max(1, deadline - time.time())` means once the
  deadline passes, every still-running proc gets exactly `1s` to die,
  not 0s. More importantly: `proc.kill()` at line 207 does NOT tree-kill.
  If specialist Claude has a Node grandchild (mcp-server, etc.) it
  becomes a zombie holding the log file handle (line 211-213 then
  fails silently). Fix: use `_kill_process_tree` from `claude_gate`.

- `portfolio/journal.py:23-40` — `load_recent` reads `JOURNAL_FILE`
  with a plain `open()`, not `load_jsonl_tail`. The file grows
  unbounded (pruned at 5000 entries by `main.py:357-364`); read cost
  scales linearly. Called from `write_context` on EVERY Layer 2
  invocation. Fix: use `load_jsonl(..., limit=N)` or `load_jsonl_tail`.

- `portfolio/journal_index.py:370-383` — `retrieve_relevant_entries`
  also reads the ENTIRE journal file via plain `open()` and parses
  every line; called from `journal.write_context` which itself is
  called from `invoke_agent`. With smart_retrieval enabled this
  doubles the cost of every Layer 2 invocation.

- `portfolio/prophecy.py:70-73` — `save_beliefs` is a read-modify-write
  with NO file lock. Two paths can write concurrently:
  `evaluate_checkpoints` (called per cycle from the main loop) and
  any Layer 2 agent calling `add_belief`/`update_belief` via tools.
  `atomic_write_json` makes the WRITE atomic, but the read-modify
  sequence is not — a Layer 2 belief update CAN be silently
  overwritten by a concurrent checkpoint evaluation. Fix: add
  `threading.Lock` + file lock matching `agent_invocation._completion_lock`.

- `portfolio/health.py:152-166` — `check_staleness` uses
  `datetime.fromisoformat(hb)` then `(datetime.now(UTC) - last).total_seconds()`.
  If `hb` has no timezone, the subtraction raises `TypeError` because
  `datetime.now(UTC)` is aware. The except at line 164 catches
  `ValueError, TypeError` so it doesn't crash, but the function
  returns `(True, inf, state)` — falsely flagging the loop stale
  forever until someone fixes the file. Fix: normalize via
  `last.replace(tzinfo=UTC) if last.tzinfo is None else last`.

- `portfolio/main.py:1129-1136` — Singleton lock acquired BEFORE
  `setup_logging()` finishes registering atexit handlers. If
  `_acquire_singleton_lock` raises (not just returns False), the
  atexit handler at line 1137 is never registered and the lock file
  persists across the crash. The `try/except RuntimeError` inside
  `_acquire_singleton_lock` only covers the platform-availability
  branch. Fix: register atexit FIRST, then acquire.

- `portfolio/telegram_notifications.py:60-81` — `r.status_code` is
  accessed on `r` from `fetch_with_retry` after a `None` check, but
  the recursive retry at line 74 also returns `r2 is not None and r2.ok`
  without checking what `r2.text` looks like for a SECOND parse
  failure. Telegram sometimes returns 400 with no body on rate-limited
  bots — `r.json()` raises, gets caught, but `err_desc=""` means the
  next conditional at line 72 fails to match and the message is lost
  silently. Fix: always retry once without parse_mode on ANY 400.

- `portfolio/llm_prewarmer.py:262-271` — Skip-by-JSONL is OK but
  `_read_last_state` falls back to `read text() splitlines()` if the
  tail-block parse fails (line 147-152). For a corrupted JSONL that
  reaches that path, this loads the entire file. The state file is
  bounded only by external rotation — a long-running process that
  never rotates can produce arbitrarily large files. Fix: cap fallback
  read to N MB and treat oversize as "no state".

- `portfolio/llama_server.py:553-568` — `_thread_lock` AND file lock
  held for the whole HTTP query (up to 240s). If a query hangs (TCP
  silence to localhost), other in-process callers (chronos, ministral
  signals) block up to 240s + their own timeout. The 300s file-lock
  timeout was deliberately set higher than the 240s HTTP timeout
  (Codex review finding #1) but there's no escape hatch: if the
  HTTP request itself wedges below 240s on a connection-establish
  retry, `requests.post` does NOT honor the timeout for socket-level
  hangs cleanly on Windows. Fix: wrap in `concurrent.futures` with
  hard cancellation, or add a watchdog.

## P2 — Medium

- `portfolio/agent_invocation.py:1244-1247` — `atomic_append_jsonl(INVOCATIONS_FILE, log_entry)`
  is followed by completion alerts (lines 1271-1290). If the JSONL
  append fails on a full disk, the agent is still marked completed
  in module state but the row never lands — `get_completion_stats`
  permanently under-reports. Fix: retry once, then write to
  `critical_errors.jsonl` (which has its own dedup path).

- `portfolio/trigger.py:161-199` — Startup grace period gates on
  `os.getpid() != saved_pid`. If a subprocess inherits the pid via
  fork-exec (not happening on Windows but matters for WSL), the grace
  period mis-fires. More critical: the state load (line 162) and
  save (line 198) inside the grace branch are not under any lock, so
  a concurrent reader of trigger_state.json sees the half-built state.

- `portfolio/trigger.py:282` — `flip_cooldowns` dict grows unbounded
  per ticker. With ticker delisting / aliasing, stale keys accumulate.
  No prune logic.

- `portfolio/circuit_breaker.py:82-106` — `allow_request` returns True
  on the OPEN→HALF_OPEN transition AND records `_half_open_probe_sent
  = True` but only the OPEN check actually returns True — the
  HALF_OPEN branch at line 106 always returns False. A second caller
  arriving 1µs after the transition gets blocked even though the
  HALF_OPEN probe is in-flight. That's documented (line 102-106) and
  intended, but the comment claims the probe is "always sent via
  the OPEN→HALF_OPEN transition" — if the transition observer crashes
  between line 97 and line 99, state is HALF_OPEN with probe_sent=True
  but no probe ever fires. Fix: wrap the state mutation + early
  return in try/except that restores OPEN on exception.

- `portfolio/llama_server.py:179` — `logger.debug("Failed to kill server pid=%s", pid)`
  uses `pid` which may be unbound if the `if content:` branch was
  skipped. UnboundLocalError on logging — caught by the outer
  except so silent. Fix: capture in a local before reaching debug.

- `portfolio/digest.py:35-44` — `_get_last_digest_time` falls back to
  reading `trigger_state.json` on every call until the new file is
  written. After first invocation `_set_last_digest_time` writes the
  new file; works fine. But: `load_json(_DIGEST_STATE_FILE, default={})`
  returns `{}` instead of `None` on missing file, so the `if not state`
  branch ALWAYS runs (because `{}` is falsy). Means every digest call
  re-reads trigger_state.json until the timestamp is written —
  unnecessary I/O but not a bug.

- `portfolio/autonomous.py:88-90` — Bare `except Exception:` swallows
  errors and only logs via `logger.exception`. If the autonomous path
  is the only Layer 2 path (layer2.enabled=false), a chronic exception
  here means NO journal/decision logs and no telegram, but the loop
  reports `_log_trigger(..., "autonomous", ...)` as if it succeeded.
  Fix: include autonomous outcome in invocation log status field.

- `portfolio/health.py:182-196` — `check_agent_silence` reads
  `state["last_invocation_ts"]` from `load_health()` but the fallback
  path at line 187-196 writes back to `health_state.json` under
  `_health_lock`. The read-modify-write reads via `load_health()`
  (no lock) at line 194 then writes via `atomic_write_json` — a
  concurrent `update_health` between the unlocked load and the locked
  block can drop fields. Fix: read inside the lock.

- `portfolio/llm_batch.py:300` — `_ss._full_llm_cycle_count += 1` is
  not atomic across processes. Metals loop runs its own LLM batch in
  a separate process and bumps its own counter; the prewarmer trusts
  the main loop's counter only. Possible to prewarm the wrong slot
  if a multi-process integration ever lands. Document or add file-based
  counter.

- `portfolio/multi_agent_layer2.py:175` — `proc._log_fh = log_fh` /
  `proc._name = name` attach to a `subprocess.Popen` object using
  private attribute hack. Robust in CPython, but `_log_fh` could
  collide with future stdlib attributes. Use `getattr(proc, "_log_fh", None)`
  consumer-side (already done at line 211, good) but the attach is
  still hacky. Fix: use a wrapper dict.

- `portfolio/alert_budget.py:36-46` — No lock. `_sent_timestamps` and
  `_buffer` mutated without thread-safety. The class is module-level
  imported into telegram paths that can be hit from the 8-worker
  ticker pool. Race causes occasional double-increment of the budget.

- `portfolio/perception_gate.py:60-65` — Falls back to `agent_summary.json`
  if compact missing. On a fresh boot before the first cycle, BOTH
  may be missing → `summary is None` → returns `(True, "no summary
  available, pass through")`. That means the gate is *disabled*
  on cold start when it should be most conservative. Fix: require
  ≥1 cycle of data before applying gate logic.

- `portfolio/decision_outcome_tracker.py:43` — `decisions[-max_entries:]`
  truncates from the END of the file (latest), but outcomes for OLD
  entries are still missing. Once the file exceeds max_entries, old
  unfilled entries are never backfilled.

- `portfolio/telegram_poller.py:262-265` — `should_persist` includes
  `outcome["drop_reason"] in _SETTLED_DROP_REASONS` but excludes
  `processed=True` AND `drop_reason in ...` (both true never happens,
  but the OR makes it work). Fine logically. Risk: `_save_offset`
  failure leaves persistence behind in-memory offset; next restart
  re-fetches and applies the stale filter. The stale filter then
  drops because msg_date is older than startup_time - 60. Net effect:
  a transient disk error during ack creates a permanent loss of the
  user's queued command (the very bug the round-7 fix was for).

## P3 — Low

- `portfolio/main.py:1115-1126` — `_sleep_for_next_cycle` uses
  `time.monotonic()` deltas which can produce negative `remaining`
  if `elapsed > interval_s`; the `if remaining > 0` guard handles
  it, but the warning at line 1126 fires every cycle on a sustained
  overrun without any rate-limiting. Spam in agent.log.

- `portfolio/main.py:991` — `_consecutive_crashes = _load_crash_counter()`
  at module import time. If `data/crash_counter.json` has a stale
  count from days ago, the next crash triggers suppression
  immediately. Fix: also check a recency window.

- `portfolio/reflection.py:166-172` — `last_ts = datetime.fromisoformat(last["ts"])`
  followed by `datetime.now(UTC) - last_ts` raises if `last_ts` is
  naive; caught by `except (KeyError, ValueError)` which does NOT
  catch TypeError. Fix as health.py above.

- `portfolio/agent_invocation.py:1018` — `text = f"...CRASH LOOP SUMMARY..."`
  passes raw markdown to `send_or_store`; if the latest error message
  contains markdown specials they're not escaped. Could break the
  message render and silently fall through.

- `portfolio/digest.py:217-233` — Bold portfolio load uses bare `except
  (KeyError, TypeError, ZeroDivisionError)` — missing `OSError`, so
  a partial-file read (file removed between exists() check and load)
  crashes the digest entirely.

- `portfolio/llm_prewarmer.py:230-235` — Pytest detection via env
  var is brittle. If a subprocess inherits `PYTEST_CURRENT_TEST` from
  a wrapping pytest run (rare but possible in CI), the prewarmer
  never fires. Probably fine, but worth documenting.

- `portfolio/regime_alerts.py:39-43` — `_get_last_regime` iterates
  `load_jsonl(REGIME_HISTORY_FILE)` (full read). On a heavy regime
  ticker (XAG with high vol) the file grows fast. No tail-read.

- `portfolio/journal_index.py:64-65` — BM25 `_idf` returns 0 for
  unseen terms, so a query token that matches NO documents
  contributes nothing. But the math at line 86 has `numerator * 0 /
  denom = 0` always. Fine. But the BM25 fit at line 50 divides
  `sum(self.doc_lens) / self.doc_count if self.doc_count else 1` —
  if `doc_count == 0` the score loop at line 76-87 doesn't execute
  anyway. Defensive but unused.

## Tests missing

- No test covers the silent exit-0 auth-failure path with the
  `pf-agent.bat` fallback (bypass of `claude_gate`).
- No test covers `_kill_overrun_agent` failing to kill the
  grandchild (Claude CLI Node helpers).
- No test for concurrent `prophecy.save_beliefs` write-races.
- No test for `prewarm_next_model` blocking the main loop > 30s.
- No test that `_completion_watchdog` does NOT spawn duplicates
  under high `invoke_agent` reentry.
- No test for `bigbet.invoke_layer2_eval` race with `_invoke_lock`
  (it doesn't take the lock — uncovered).
- No assert that `NODE_OPTIONS` from environment isn't clobbered.

## Cross-cut observations

The orchestration layer has correctly internalised the "Claude CLI exits
0 on auth failure" pattern (the centralized `detect_auth_failure` +
critical_errors.jsonl pipeline is excellent) but only `agent_invocation`,
`multi_agent_layer2`, and the central `claude_gate` are routed through
it. `bigbet.py:175` is the smoking gun: a subprocess that prints "Not
logged in", exits 0, returns `probability=None`, and the bigbet path
treats it as a normal "no signal" outcome — exactly the silent failure
that motivated the audit. Either every Claude subprocess invocation
should be migrated to `invoke_claude_text`, or the gate should be
enforced by linting.

The locking discipline is uneven. `_completion_lock` was added
correctly for the watchdog/main-loop race but doesn't cover all the
globals it should. `_health_lock` covers writes but several read paths
operate outside it. `prophecy` and `trigger_state` have no locks at all
despite being accessed from multiple cycles. A single project-wide
"this dict is shared, here's its lock" doc would prevent the next
class-of-bug.

The CLAUDE.md rule "atomic I/O only" is mostly honored, but
`telegram_poller.py:373-383` calls `fetch_with_retry` and accesses
`r.text[:200]` at line 385 — if a 400 with empty body returns and
fetch_with_retry returned a real Response, `r.text` is fine, but
this is brittle.

Heartbeat keepalive (`heartbeat_keepalive` context manager in
`health.py:96-149`) is well-designed but only wraps the Layer 2 +
autonomous + bigbet + iskbets paths. The post-cycle housekeeping in
`_run_post_cycle` (digests, prunes, accuracy snapshots) can also
exceed 60s on a heavy day and is NOT wrapped — heartbeat goes stale
during big batch jobs.
