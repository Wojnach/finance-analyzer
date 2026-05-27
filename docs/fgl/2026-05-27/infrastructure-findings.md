# Infrastructure Subsystem — Adversarial Review (2026-05-27)

## Summary

**Counts:** P0 = 1, P1 = 7, P2 = 8, P3 = 4 (20 findings total)

**Top 3 themes:**
1. **Telegram delivery is best-effort with no end-to-end accounting.** Multiple paths re-send Markdown→plain on parse failure but never alert when a real loss happens; `send_telegram` returns True on `NO_TELEGRAM`/mute even though nothing was sent; `_send_reply` returns None silently; AlertBudget (the only true rate-limiter class) is dead code (never imported). The runaway-loop ban risk that motivated AlertBudget is real and not mitigated by `message_throttle` (which only throttles category=`analysis`).
2. **`atomic_append_jsonl` correctness depends entirely on the sidecar lock — and several writers bypass it.** The lock helper is correct, but `journal.py` reads `JOURNAL_FILE` with raw `open()` and journal entries are appended elsewhere (Layer 2 writers) with mixed mechanisms — combined with `prune_jsonl`/rotation holding the lock for hundreds of ms, this is the same data-loss vector the project hit on 2026-05-11 except now on the journal file. Several "atomic_write_json" writers also read-modify-write *without* any lock (`health.py`, `prophecy.py`, `digest.py`, `daily_digest.py`), losing last-write-wins races between threads/loops.
3. **Subprocess cleanup is only enforced through `claude_gate` / `subprocess_utils.run_safe`.** `qwen3_signal._call_qwen3_batch` calls `run_safe` (good), but `qwen3_signal._call_qwen3` and `ministral_signal._call_model` *do* go through `run_safe` (good). However, `llama_server._start_server` uses raw `subprocess.Popen` with no Job Object — when the main loop dies, llama-server orphans on Windows. `subprocess_utils.kill_orphaned_llama` only sweeps `llama-completion.exe` (the old subprocess name), NOT `llama-server.exe`.

**Biggest-risk one-liner:** `llama_server._start_server` (portfolio/llama_server.py:456) spawns `llama-server.exe` with **raw Popen, no Job Object** — on hard parent kill, an orphaned llama-server keeps port 8787 + 5 GB VRAM hostage indefinitely; the orphan reaper at `subprocess_utils.kill_orphaned_llama` only targets the legacy `llama-completion.exe` name and will never find it.

---

## Critical (P0)

### [P0] llama-server.exe spawned without Windows Job Object — orphans on parent crash
**File:** `portfolio/llama_server.py:456`
**Issue:** `_start_server` calls bare `subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`. Every other expensive subprocess in the codebase is wrapped in a Job Object via `subprocess_utils.popen_in_job()` / `run_safe()` so the OS auto-kills the child if the parent dies. llama-server is the heaviest child in the system (~5 GB VRAM, exclusive port 8787) and is precisely the one that *cannot* orphan, yet it's the only major Popen call that bypasses the protection. `subprocess_utils.kill_orphaned_llama()` only matches `llama-completion.exe` (a name the system no longer uses since the llama-server migration on 2026-04-09), so the safety-net reaper will never find an orphaned llama-server.
**Impact:** Parent loop crash (e.g. PF-DataLoop hard-killed by Task Scheduler restart, OS power blip, OOM) leaves llama-server.exe holding port 8787 + 5 GB VRAM + GPU lock. Next loop start: `_kill_server_by_pid` can find it via the pid file if that survived, but if pid file is gone or pid was reused, `_kill_by_port` falls back to netstat parsing which can fail under permission constraints. The 2026-05-02 25-hour GPU wedge documented in `gpu_gate.py` was a chronos pid 13152 dying with the GPU lock; the analogous failure here keeps llama-server hostage *and* its GPU lock, with no sweeper to recover.
**Fix:** Use `subprocess_utils.popen_in_job(cmd, stdout=..., stderr=...)` and stash the returned `job` handle in `_local_proc`'s slot for `close_job(job)` during `_stop_server`. Separately, update `kill_orphaned_llama` to also match `llama-server.exe`, or add a new `kill_orphaned_llama_server` that sweeps that name on every loop startup.
**Confidence:** 92

---

## Important (P1)

### [P1] AlertBudget is dead code — runaway-loop Telegram ban risk is unmitigated
**File:** `portfolio/alert_budget.py:22-67`
**Issue:** The `AlertBudget` class is correctly implemented (token-bucket with priority bypass, thread-safe, unbounded `_buffer` notwithstanding) but is **never instantiated, imported, or referenced** anywhere in the production code (`grep -rn AlertBudget portfolio/` returns only the class definition file and `tests/test_alert_budget.py`). `send_telegram` (telegram_notifications.py:35), `_do_send_telegram` (message_store.py:105), `_send_reply` (telegram_poller.py:371), `send_or_store` (message_store.py:168) — none of them consult a budget. `message_throttle` only throttles `category="analysis"` (3h cooldown), leaving trade/iskbets/bigbet/digest/error/regime/invocation/elongir/crypto_report categories uncapped. A runaway loop that triggers, say, the elongir signal every cycle (60s) can send 60 messages/h × N hours straight, and Telegram bot tokens get rate-limited at 30 msg/sec or 20 msg/min/chat — banned bot tokens take hours to recover.
**Impact:** A misbehaving signal can blast 1000+ messages and trip Telegram's anti-spam, killing the entire notification channel until manual intervention. The class that exists to prevent this is literally never called.
**Fix:** Wire `AlertBudget(max_per_hour=20)` into `send_or_store` ahead of the `should_send` check, with `PRIORITY_EMERGENCY` for category in {"trade","error","regime"} and `PRIORITY_NORMAL` for the rest. Add an unbounded-buffer guard: cap `_buffer` to last 100 with `deque(maxlen=100)` so a stuck flush doesn't leak memory forever.
**Confidence:** 90

### [P1] `journal.py:load_recent` reads `layer2_journal.jsonl` with raw `open()` — no sidecar lock
**File:** `portfolio/journal.py:28`
**Issue:** `with open(JOURNAL_FILE, encoding="utf-8") as f:` reads the journal in `load_recent()` while *other* code paths append to the same file via `atomic_append_jsonl` (which takes the sidecar lock) AND while `log_rotation.rotate_jsonl` does read→write-tmp→`os.replace` under the same lock. Reader holds neither the sidecar lock nor any FS-level lock — a concurrent rotation `os.replace` mid-iteration can either silently truncate the read (file handle pointing at the old inode on Linux; "the file has vanished" on Windows midway through iteration) or produce a partial last line. This is precisely the `signal_log_reconciliation` divergence pattern the lock contract was added to fix — but `journal.py` was missed.
**Impact:** Layer 2 occasionally sees a truncated journal context, hallucinates "fresh start" thesis, and forgets running theses. Symptoms: thesis-chain ruptures that look like Layer 2 forgetting, but are actually torn reads. Frequency: anytime rotation runs (~daily) while Layer 2 is reading context.
**Fix:** Wrap `load_recent` and the raw-`open` in `_load_journal_entries` (`vector_memory.py:272`), `journal_index.py:373`, and the equivalent reader paths in `digest.py` / `daily_digest.py` with `with jsonl_sidecar_lock(JOURNAL_FILE):` — OR migrate those readers to `load_jsonl()` / `load_jsonl_tail()` from `file_utils` which should be updated to take the lock for the read.
**Confidence:** 85

### [P1] `health.py:update_health` and friends — last-write-wins between main loop + metals loop
**File:** `portfolio/health.py:20-41, 64-86, 169-220, 240-298`
**Issue:** `_health_lock = threading.Lock()` is an in-*process* lock. The main loop AND the metals loop (`data/metals_loop.py`, separate process) both call into `health.py` to write `health_state.json`. The thread lock provides zero cross-process protection. Both processes do read→modify→write of the same file via `atomic_write_json`; `atomic_write_json` itself is atomic per call but the read-modify-write sequence isn't. So if metals loop reads `cycle_count=100`, main loop reads `cycle_count=101`, metals loop writes back `cycle_count=100+metals_data`, main loop writes back `cycle_count=101+main_data` — metals' update is lost. Same race applies to `signal_health`, `last_module_failures`, `errors[]`, and `update_signal_health_batch`.
**Impact:** Lost health updates from one or the other loop. The dashboard's `/api/health` sometimes shows stale module-failure data or undercounts cycle counts. Most insidious case: the recovery semantics in `update_module_failures` (clears `last_module_failures` on clean cycle) — if main loop clears it but metals loop's stale read writes back a 5-min-old failure, the dashboard shows a phantom failure for 60s.
**Fix:** Either (a) make `health.py` use a sidecar file lock (the `jsonl_sidecar_lock` from `file_utils.py` would work after a small adapter for non-JSONL targets), or (b) split health state per-process (`health_state_main.json` + `health_state_metals.json`) and have the dashboard merge.
**Confidence:** 88

### [P1] `prophecy.py:update_belief` / `add_checkpoint` / `evaluate_checkpoints` — read-modify-write race
**File:** `portfolio/prophecy.py:105-126, 170-198, 201-269`
**Issue:** All mutators do `data = load_beliefs(); ... mutate ...; save_beliefs(data)`. Layer 2 invocations can run concurrently (T2 + a metals-loop-triggered T1 overlap), and both can call `update_belief` / `evaluate_checkpoints`. No lock protects the read-modify-write. The second writer clobbers the first writer's update. `evaluate_checkpoints` is especially bad because it walks the entire belief tree and writes back the whole thing — a concurrent `add_belief` is silently dropped.
**Impact:** Belief updates silently lost. Checkpoints intermittently fail to fire (or fire twice). The `silver_bull` prophecy that drives metals trade conviction can drift silently. Auditable via "this belief's `updated_at` is older than the timestamp of the update I logged".
**Fix:** Wrap all mutators in `jsonl_sidecar_lock(PROPHECY_FILE)` (works fine on non-JSONL targets — the lock is on a sidecar file). Or introduce a dedicated `prophecy_lock` in a file beside `prophecy.json`.
**Confidence:** 87

### [P1] `telegram_poller._handle_mode_command` writes `config.json` directly — symlink loop risk
**File:** `portfolio/telegram_poller.py:312-369`
**Issue:** The mode command rewrites `config.json` via `atomic_write_json` + `_resolve_write_path` which follows the symlink. CLAUDE.md says `config.json` is a symlink to `C:\Users\Herc2\.config\finance-analyzer\config.json` containing API keys. The BUG-210 guard (`len(cfg) < 5`) is a good safety net but doesn't prevent the structural problem: a Telegram message — from anyone *who has the chat_id* (which is anyone who knows it) — can rewrite the live API-key config. The chat_id check (`telegram_poller.py:180`) is the *only* auth boundary — there is no command auth, no signature check, no rate limit on the `/mode` command path.
**Impact:** If the chat_id ever leaks (or another bot inadvertently joins the chat — Telegram does NOT enforce 1:1 chat ownership), arbitrary mode-flips can be triggered. More concerning: any future command added to `_parse_command` inherits the same trust model. Today's surface is small but the trust pattern is brittle.
**Fix:** Add a `sender.id` allow-list check in `_handle_update` (not just chat_id). Persist allowed user IDs in a config field. Also, the `/mode` handler should write a *delta* file (`config_override.json` merged on load) rather than the master `config.json` — preserves the principle that the master config is human-edited.
**Confidence:** 84

### [P1] `_acquire_file_lock` busy-loops with 1s sleep — no stampede protection
**File:** `portfolio/llama_server.py:488-526`
**Issue:** The cross-process lock uses `O_CREAT | O_EXCL` polling with `time.sleep(1)` between attempts. Timeout is 300s. Stale-PID check shells out to `tasklist` (Windows) on every iteration. Two concerns: (1) under contention from multiple processes (main loop ticker pool + metals loop + signal subprocesses), this becomes a 1-Hz tasklist storm — tasklist takes 100-500ms on a busy Windows box. (2) When the holder releases, multiple waiters wake up within the same 1s window and race for `O_EXCL`; the loser sleeps another second. Real-world lock acquire latency under contention can be 5-10s even when the holder finished promptly.
**Impact:** Cycle latency tail grows under contention. The user-visible symptom is "loop took 90s instead of 60s" because three threads were stacked behind each other on the file lock + tasklist polling overhead. The 2026-05-11 plex-vram-coord work added a 30s reclaim wait that compounds with this.
**Fix:** Use `msvcrt.locking(LK_NBLCK)` with a shorter sleep (250ms) for the busy-wait, and cache the tasklist output for 5s. Or switch to the proper `portfolio.process_lock.acquire_lock_file` pattern (msvcrt blocking lock), which the codebase already implements and proves correct.
**Confidence:** 82

### [P1] `gpu_gate._GPU_LOCK_FILE = Path("Q:/models/.gpu_lock")` — hardcoded Windows path, no fallback
**File:** `portfolio/gpu_gate.py:33-34`
**Issue:** The GPU lock file path is hardcoded to `Q:/models/.gpu_lock`. On a fresh Windows machine without the `Q:` drive (or a WSL test environment), `_GPU_LOCK_FILE.stat()` raises `OSError: [Errno 2]`. The fallback path is `_is_stale() → return True`, which means: on a system without Q:\, *every* GPU acquire treats the lock as stale and runs `_try_break_stale_lock()` which itself checks `_GPU_LOCK_FILE.exists() → False → return False`. So acquire goes through fine — but the lock file path is silently in the user-namespace (`Q:\models\.gpu_lock`) which is shared with model files, fragile if drive mapping changes.
**Impact:** Tests on machines without `Q:` drive (CI, dev laptops) cannot test the GPU lock path without mocking. More importantly, if `Q:\` becomes read-only or unmounted mid-loop, `gpu_gate` will silently fail to acquire and every LLM signal call goes to subprocess fallback (which is the documented "Plex-unsafe" path).
**Fix:** Move lock to `data/.gpu_lock` (repo-local, always writable) and stop using `Q:` for runtime coordination. `Q:` is for read-only model files.
**Confidence:** 80

---

## Important (P2)

### [P2] `send_telegram` returns True for muted/disabled sends, hiding "message not delivered"
**File:** `portfolio/telegram_notifications.py:35-46`
**Issue:** Three early-return paths all return `True`: `NO_TELEGRAM` env, `mute_all`, `layer1_messages=false`. Callers (`_maybe_send_alert`, others) treat True as "delivered". This means "fan out an alert if delivered" semantics break under mute — the system thinks it told the user, but it didn't. Same pattern in `_do_send_telegram` for `NO_TELEGRAM`.
**Impact:** Telegram-as-confirmation paths (e.g. iskbets `acknowledge_required` flows) treat muted state as ACK'd. Operationally rare today but the False-positive return is a footgun for future callers.
**Fix:** Return a tristate `("sent","muted","failed")` or at least keep True only for actual API 200; use a distinct sentinel for muted.
**Confidence:** 82

### [P2] `send_or_store` does not log the truncate event for Markdown re-send
**File:** `portfolio/message_store.py:147-165`
**Issue:** When the Markdown parse fallback to plain-text re-send is triggered, the function returns `r2.ok` but the JSONL log records only one entry with `sent=True/False` of the original. There's no record that the formatted message failed to render and the plain version was substituted. Debugging "why did this digest look broken" requires the operator to grep agent.log for the WARNING.
**Impact:** Observability gap. The same message logged as `sent=True` may have arrived stripped of formatting; users see degraded UX without a trace.
**Fix:** Add `fallback_used=True` field to the log entry when the plain re-send path was taken.
**Confidence:** 80

### [P2] `_loading_keys` set in `shared_state` is never bounded — memory creep on long uptimes
**File:** `portfolio/shared_state.py:28-29, 88`
**Issue:** `_loading_keys` and `_loading_timestamps` are pruned only inside `_cached()`/`_cached_or_enqueue()` via the stuck-key sweep (>120s). But on a system where the same N keys cycle constantly, the prune only runs for keys *currently being requested*. A key that was once loading and now isn't queried at all stays in the dict (covered by the stuck-eviction TTL but only triggered when `_cached` itself is called for *some* key — `_loading_timestamps` iteration runs unconditionally).
**Impact:** Looking at the code path more carefully, the `for k in stuck` loop runs every `_cached()` call so it does keep up. Low-impact. The real risk is `_loading_timestamps.pop(k, now)` — `now` was set at function entry, not at stuck-eviction time. Reported `stuck_duration` is off by the eviction loop's elapsed time. Cosmetic.
**Fix:** Cap `_loading_keys` to a sane upper bound (e.g. 1000) defensively, log+evict at high water mark.
**Confidence:** 75

### [P2] `vector_memory._load_journal_entries` reads full journal with raw `open()` — no lock, no tail
**File:** `portfolio/vector_memory.py:267-281`
**Issue:** Loads every line in `layer2_journal.jsonl` (even though the call site only embeds *new* entries). No size cap. Same rotation race as the journal.py P1 finding. As the journal grows, every Layer 2 invocation that has vector_memory enabled re-reads the full file.
**Impact:** Linear growth with journal size. At the 60d/10MB rotation policy that's ~10K reads of a 10MB file per Layer 2 invocation. Adds noticeable latency tail to the T3 path.
**Fix:** Maintain a `last_embedded_ts` watermark in `data/chromadb/state.json` and only read entries newer than that. Use `load_jsonl_tail` to read in chunks.
**Confidence:** 80

### [P2] `log_rotation.rotate_text` is not protected by any lock
**File:** `portfolio/log_rotation.py:467-541`
**Issue:** Text-file rotation (`agent.log`, `loop_out.txt`, `golddigger_out.txt`) does `shutil.copy2` → truncate-original. If the loop is actively writing to `agent.log` via `RotatingFileHandler` AND the standalone `python -m portfolio.log_rotation` runs concurrently (PF-LogRotate scheduled task vs. PF-DataLoop), the truncate can race with a `RotatingFileHandler` write. Worst case: `RotatingFileHandler` itself does size-based rotation, so two rotators can both decide "time to rotate" simultaneously.
**Impact:** Lost log lines around rotation events. Hard to detect because logs *look* fine post-rotation. Operationally rare (rotation is ~daily) but inevitable.
**Fix:** Either (a) make `log_rotation.rotate_text` skip files already managed by Python's `RotatingFileHandler`, or (b) use a sidecar lock per file.
**Confidence:** 80

### [P2] `_kill_by_port` (llama_server) shells out to `netstat -ano` — parses unstable output
**File:** `portfolio/llama_server.py:96-130`
**Issue:** `netstat -ano | grep :PORT LISTENING` is fragile: lines containing the port in a column other than the local-address column (e.g. remote endpoint `:PORT`) match falsely. Whitespace splitting `parts[-1]` assumes PID is always last — true on US-English Windows, but localized netstat outputs (e.g. Japanese, German) put the state column in different positions.
**Impact:** On a non-English Windows install the port-kill path either fails to find the PID or matches the wrong column. Low likelihood for this user (English Windows 11) but a portability landmine.
**Fix:** Use `psutil.net_connections()` and filter by `laddr.port == _PORT and status == 'LISTEN'` — cross-platform and column-independent.
**Confidence:** 80

### [P2] `ml_signal._pred_cache` is unbounded and unlocked
**File:** `portfolio/ml_signal.py:18-20, 105-162`
**Issue:** `_pred_cache = {}` keyed by ticker, no lock. Currently only 2 tickers (BTC, ETH) so size is trivially bounded. But the ThreadPoolExecutor pool can call `get_ml_signal("BTC-USD")` from multiple threads simultaneously, racing on `_pred_cache[ticker] = {"data": result, "time": now}`. CPython dict assignment is atomic so won't corrupt, but the model load `_load_model()` can also race — `_model_cache["model"] = joblib.load(MODEL_PATH)` runs twice concurrently on first call, double-loading a 50MB model.
**Impact:** Cosmetic. First two concurrent ML calls double-load the model (~200ms each). Steady state is fine.
**Fix:** Add a `_model_cache_lock = threading.Lock()` around `_load_model`.
**Confidence:** 78

### [P2] `digest.py:_get_last_digest_time` returns 0 on any JSON error — fires extra digest after corruption
**File:** `portfolio/digest.py:35-43`
**Issue:** `except (json.JSONDecodeError, OSError, ValueError): return 0`. A corrupted `digest_state.json` causes `last_digest_time = 0`, which means `time.time() - 0 > DIGEST_INTERVAL` always, so the next loop iteration sends a digest, AND fails to update the corrupted state (the `_set_last_digest_time` write happens), AND the next cycle is fine. Net effect: one extra digest sent. Not severe but the failure mode is "silently send another digest" not "alert operator about corrupt state".
**Impact:** One spurious digest per corruption event. Mostly fine; alerting would be better.
**Fix:** Log a WARNING on the error path so corruption is visible.
**Confidence:** 76

---

## Smell (P3)

### [P3] `process_lock` `_lock_file` raises RuntimeError on platforms without msvcrt OR fcntl
**File:** `portfolio/process_lock.py:71-74`
**Issue:** Defensive but the message "No file locking mechanism available" is unactionable — the user has no way to install one. Platform check `platform.system()` would let the caller pre-flight.
**Fix:** Add a module-level `LOCKING_AVAILABLE` boolean callers can check.
**Confidence:** 60

### [P3] `tickers.SIGNAL_NAMES` is in sync — verified
**File:** `portfolio/tickers.py:349-433`
**Issue:** None. Cross-checked against all 13 consumer modules (`signal_engine`, `accuracy_stats`, `outcome_tracker`, `ticker_accuracy`, `backtester`, `accuracy_degradation`, `signal_db`, `signal_history`, `ic_computation`, `meta_learner`, `weekly_digest`). All `_parse_disabled_reasons` regex paths handle the current list shape. The list is up-to-date through the 2026-05-26 additions (`kalman_trend_momentum`). The historic bug (8 missing signals) appears genuinely fixed. **No action needed.**
**Confidence:** 95

### [P3] `gpu_gate.py:_pid_alive` falls back to "assume dead" if psutil missing
**File:** `portfolio/gpu_gate.py:80-83`
**Issue:** "psutil not installed — assuming PID %d is dead". That's the opposite of safe-default for a stale-lock breaker: the safer assumption is "alive" (don't break a possibly-live lock). This was clearly a deliberate choice to prevent deadlocks but the comment is wrong about its safety direction.
**Fix:** Document the rationale (better-to-break than to-wedge) in the comment so the next reviewer doesn't "fix" it.
**Confidence:** 65

### [P3] `notification_text.py` is 64 lines and not reviewed in detail
**File:** `portfolio/notification_text.py`
**Issue:** Not actually a finding — flagging that I did not deep-read this file. No obvious red flags from the surface.
**Confidence:** 50

---

## Notes on items checked and found clean

- **`atomic_write_json` correctness:** Solid. tempfile in same dir → fsync → `os.replace`. Symlink resolution via `_resolve_write_path` is correct (writes to real file, not the link). Empty-write case is benign (writes empty file).
- **`atomic_append_jsonl`:** Correct via `jsonl_sidecar_lock` sidecar pattern. Cross-platform (msvcrt `LK_LOCK` blocking on Windows, `fcntl.flock LOCK_EX` on POSIX). Sidecar avoids the brand-new-empty-file race.
- **`process_lock`:** Correct non-blocking acquire pattern, metadata write under lock, release-on-close semantics. The "stale lock if process kill -9" concern is mitigated by FS-level lock release on file-handle close (OS does this even on kill).
- **`subprocess_utils.run_safe`:** Job Object pattern is correct, timeout actually kills tree on Windows. Linux fallback uses `start_new_session=True` + killpg.
- **`claude_gate.invoke_claude`:** Properly tree-killed on timeout via `_run_with_tree_kill`. `_invoke_lock` serializes in-process. The "stuck gate" risk is low — the in-process lock is released in `finally`.
- **HTTP retry:** Exponential backoff with jitter, capped at `retries` attempts. Telegram bot token redaction in URL is correct (`_redact_url`). 429 honors `retry_after` from Telegram response.
- **`telegram_poller` offset persistence:** Well-designed. Restart-bypass bounded to 1h (sensible). Stale-filter logic is correct. Dispatch failures don't ack the offset (Codex P1 round-7 fix correctly preserved).
- **`message_throttle`:** Correct queue-or-send logic, only affects category=analysis.
- **`forecast_signal.run_forecasts`:** Bounded ticker iteration, atomic append per entry. OK.
- **`shared_state._cached`:** Dogpile prevention is correct, stuck-key eviction works, KeyboardInterrupt path cleans up loading set.
