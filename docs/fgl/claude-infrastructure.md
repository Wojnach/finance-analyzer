# Adversarial Infrastructure Review — finance-analyzer

Reviewer: claude-infrastructure | Date: 2026-05-09
Files: file_utils, shared_state, log_rotation, logging_config, telegram_notifications,
telegram_poller, message_store, message_throttle, alert_budget, prophecy, journal,
journal_index, gpu_gate, process_lock, subprocess_utils, api_utils, http_retry,
dashboard/app.py, dashboard/auth.py

---

[P0] http_retry.py:31 — fetch_with_retry retries POST blindly with no idempotency check. Telegram sendMessage, Avanza order placement (when wired through this helper), and any other POST that times out at the server but already executed will be re-sent on 429/5xx/timeout. A 502 from Telegram + retry can dispatch the same trade alert twice; a Connection-reset mid-Avanza-order can place duplicate orders. There is no Idempotency-Key header, no nonce, no caller flag. | FIX: add `idempotent: bool = False` parameter; default False; only retry GETs and explicitly-flagged POSTs. For Telegram (idempotent at the bot-level via dedup) flip to True; never flag order POSTs.

[P0] file_utils.py:202-263 — atomic_append_jsonl uses `msvcrt.locking(fd, LK_LOCK, 1)` which on Windows blocks for ~10s then raises OSError, NOT indefinitely. Under heavy contention the lock acquisition fails after ~10s and the helper raises through the caller; comments claim "blocking" but Windows CRT semantics are bounded retry. Worse: the current code holds NO retry wrapper — a single contention failure bubbles to journal/telegram log loss. | FIX: wrap `_msvcrt.locking(fd, LK_LOCK, 1)` in a retry loop with bounded total wait (e.g. 30s with 100ms sleeps between OSError catches); document the Windows-specific bound.

[P0] gpu_gate.py:209-269 — Layer-1 thread lock is acquired BEFORE Layer-2 file lock; on file-lock timeout the thread lock is held for the full timeout (default 60s) blocking every other in-process consumer waiting for ANY model. If chronos holds the file lock and sweeper hasn't fired, a ministral acquirer parks the thread lock for 60s while spinning on file-lock retries, starving qwen3 even though qwen3 doesn't need chronos's resource. | FIX: invert order — acquire file lock first (cross-process), then thread lock (in-process); OR release thread lock during file-lock waits.

[P0] dashboard/auth.py:131-132 — Cloudflare Access path trusts ANY `Cf-Access-Authenticated-User-Email` header without verifying the request came through Cloudflare. A direct LAN connection to port 5055 bypassing the tunnel can spoof this header (request goes straight to Werkzeug, no CF strip). The dashboard binds dual-stack `[::]:5055`, accepting LAN traffic; comment claims "CF strips inbound Cf-Access-* headers at its edge" but that only protects CF-routed traffic, not direct binds. | FIX: require Cf-Access-Jwt-Assertion JWT validation against CF team public keys, OR bind only to localhost and route ALL traffic through CF tunnel, OR add an allow-list check (only trust header from configured CF tunnel source IP).

[P0] message_throttle.py:60-67 — TOCTOU race: `should_send_analysis` reads state, then `queue_analysis` re-reads state, mutates, writes. Two concurrent threads/processes both pass the cooldown check, both write `pending_text` (last writer wins, message lost), or both call `_send_now` (double-send). No lock. | FIX: wrap read+modify+write in a sidecar file lock (same pattern as atomic_append_jsonl) OR use atomic compare-and-swap on `last_analysis_sent`.

[P1] file_utils.py:53-59 — atomic_write_json fsyncs the file but NOT the parent directory after `os.replace`. On power loss between `os.replace` and a directory fsync, the rename can be lost on ext4/Linux and on some Windows configs (NTFS journal mostly handles this, but not guaranteed for all volume types). | FIX: on POSIX, open parent dir and fsync it after replace; on Windows os.replace + fsync of file is generally adequate but document the guarantee.

[P1] shared_state.py:37-124 — Race between cache eviction and concurrent reads: `_cached` mutates `_tool_cache` under lock during eviction (lines 54-66) but other readers calling `_cached` for a different key inside `func(*args)` (line 93) are NOT under lock when they touch the dict. If two threads both call `_cached("k1", ...)` and `_cached("k2", ...)` simultaneously and one triggers eviction, the eviction loop iterates a dict that another thread is about to read post-fetch. The `with _cache_lock` re-entry on line 94 protects the write but not the period between line 92 (release) and line 94 (re-acquire). Mostly safe because the dict mutation happens under the lock, but iteration over `_tool_cache.items()` on line 55 can OOM-allocate a list-comprehension copy that then becomes stale. | FIX: snapshot keys under lock (already done with list comprehension), but also reset eviction trigger so two simultaneous evictions don't both run; minor — primarily a correctness audit comment.

[P1] shared_state.py:89-90 — `_loading_keys.add(key)` happens after stale-data return path, but if `func(*args)` itself calls `_cached` recursively for the same key (improbable but possible via shared module imports), the recursive call sees the key in `_loading_keys` and returns None or stale, not deadlocking but masking the recursion. | FIX: document that `func` must not re-enter `_cached` for the same key; consider `threading.local` re-entry detection.

[P1] telegram_poller.py:262-265 — `_save_offset()` is called outside any lock. If two pollers run (race between PF-DataLoop scheduled task and a manual run), both write `telegram_poller_state.json`. atomic_write_json prevents corruption but the LATER write wins, potentially regressing offset. With only one poller this is fine, but the dispatcher comment in CLAUDE.md mentions auto-spawn fix agents — if any of those import this module the race exists. | FIX: file lock around the offset persistence, OR document single-instance assumption explicitly.

[P1] journal.py:28-40 — `load_recent` opens JOURNAL_FILE with no lock and iterates lines; concurrent atomic_append_jsonl writers can produce a torn final line under MSVCRT lock contention (P0 above). The except handler skips JSONDecodeError silently, so torn entries are LOST without warning, not just deferred. This is a journal — append-only contract — but the read path swallows malformed tails. | FIX: log.warning on JSONDecodeError so corruption is observable (matches the load_jsonl pattern in file_utils.py); upgrade severity if torn lines reappear.

[P1] journal_index.py:373-381 — Same silent-skip issue as journal.py:38: `except json.JSONDecodeError: continue` swallows torn lines. The retrieval system silently drops journal entries that crashed mid-write. | FIX: emit a logger.warning, count drops in a metric for observability.

[P1] gpu_gate.py:73-82 — `_pid_alive` falls back to `return True` (assume alive) when psutil is unavailable. With psutil missing AND a stale lock from a crashed process, the lock is NEVER broken — the comment "Fallback: assume alive if we can't check" makes the gate fail-CLOSED for stuck-lock recovery, not fail-OPEN. The 25-hour wedge of 2026-05-02 was supposedly solved by adding the sweeper, but if psutil is missing the sweeper degrades to never-break. | FIX: on Windows, use ctypes OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) as a psutil-free liveness check (already used in subprocess_utils.kill_orphaned_llama). On POSIX, `os.kill(pid, 0)`.

[P1] gpu_gate.py:218-242 — File-lock acquire loop sleeps `time.sleep(1.0)` inside the while-loop without checking deadline first; on a 60s timeout the loop sleeps up to 1s past deadline. Minor but means the contract `timeout=60` actually waits up to ~61s. | FIX: `time.sleep(min(1.0, max(0, deadline - time.time())))`.

[P1] message_store.py:36-49 — `_COMMON_MOJIBAKE_REPLACEMENTS` dict has DUPLICATE keys: `"â"` appears 6 times (lines 39, 41, 42, 44, 45, 46), so only the last assignment wins. The intended em-dash, single-quote, double-quote, arrow replacements are silently overridden by the last `"â": "↓"`. The mojibake repair is broken; only `↓` substitutions land. | FIX: distinct multi-byte mojibake patterns must be the actual broken bytes (e.g. "â" for "—"); rewrite the dict with bytes that don't all collapse to the same UTF-8 string in source.

[P1] log_rotation.py:319-327 — Active writers can append to `signal_log.jsonl` between the `keep_lines` read (line 242-264) and the `os.replace(tmp_path, filepath)` (line 327). Any entries appended in that window are LOST — they were written to the original file after the read but the replace overwrites them. Production rotation runs daily; if the loop appends a signal during the rewrite, that signal is gone. | FIX: hold the same sidecar lock that atomic_append_jsonl uses during the read+rewrite sequence; or rotate at a known-quiet time (the daily 03:00 task already tries this but isn't enforced).

[P1] log_rotation.py:402-403 — Truncating `agent.log` via `open(filepath, "w")` while another process has an open handle to it on Windows creates an empty file but the writer's offset stays at the prior position, so subsequent writes go into a sparse hole. | FIX: rotation should rename-then-create-fresh atomically; if rename is impossible due to share-mode, signal the producer to reopen its log handle (RotatingFileHandler in logging_config.py handles this for portfolio.log but NOT agent.log/loop_out.txt).

[P1] dashboard/app.py:1685-1733 — Avanza worker queue is unbounded; if the worker thread hangs (Playwright BankID flow stuck), every dashboard request piles up requests onto `_AVANZA_REQ_Q`. The `wait(timeout=25.0)` returns the 25s timeout error to clients but the queued request is NEVER drained — it sits behind a stuck worker forever, eventually causing MemoryError. | FIX: bound queue to a small size (e.g. 10), reject with 503 when full; OR drain orphaned futures on worker timeout.

[P1] dashboard/app.py:870-878, 882-890 — Non-cached, full-file scan reads of mstr_loop_poll.jsonl and mstr_loop_trades.jsonl on every `/api/mstr_loop` request. The "for line in f" loop reads the entire file just to keep the last entry. With dashboard polling at 5s and the JSONL growing, this scales O(file_size × request_rate). | FIX: use `last_jsonl_entry()` from file_utils.

[P1] alert_budget.py:36-46 — `should_send` mutates `_sent_timestamps` without locking. `AlertBudget` is consumed from multiple senders (notification call sites across digest, agent_invocation, etc.); concurrent calls can produce miscounts or deque mutation race ("deque mutated during iteration" or `_sent_timestamps[0]` returning a freed reference). | FIX: add `threading.Lock` around all deque/list operations.

[P1] subprocess_utils.py:215-220 — `wmic` is DEPRECATED in Windows 11 and removed in some 24H2 builds. The `kill_orphaned_by_cmdline` helper will silently return 0 on hosts where wmic is gone; orphaned subprocesses accumulate undetected. | FIX: switch to `Get-CimInstance Win32_Process` via PowerShell (already used in kill_orphaned_llama at line 268) — single PowerShell pattern, modern API.

[P1] prophecy.py:201-269 — `evaluate_checkpoints` is not concurrency-safe across processes. Two processes evaluating against the same `prophecy.json` both load, mutate `cp["status"] = "triggered"`, and write — last writer wins, the OTHER process's triggered checkpoint is silently reverted to pending. | FIX: file lock around load/modify/save in evaluate_checkpoints, add_belief, update_belief, add_checkpoint.

[P1] file_utils.py:13-19 — `_msvcrt` and `_fcntl` import-failure paths set the module to None; if BOTH are None (impossible on real systems but possible in restricted Python builds), `atomic_append_jsonl` opens the lockfile and writes to the target with NO lock — same torn-line problem this primitive was built to prevent. | FIX: explicit assert or warn at import time when neither lock primitive is available.

[P2] dashboard/auth.py:90-100 — Cookie has `secure=True` but the dashboard binds plain HTTP on port 5055 (line 2052-2061). On non-HTTPS access (LAN, localhost without TLS terminator), browsers REFUSE to send `secure=True` cookies — meaning the cookie auth path silently fails for any direct HTTP visitor. The query-param fallback works once but the cookie set in response is rejected, so EVERY request requires re-passing `?token=`. | FIX: detect `request.is_secure` (or X-Forwarded-Proto) and only set secure=True when behind TLS; otherwise omit the flag for local HTTP.

[P2] dashboard/auth.py:103-150 — No CSRF protection. The dashboard exposes POST /api/validate-portfolio (line 1039), with cookie-based auth. A malicious page that the user visits can submit a form to the dashboard host, the browser includes the auth cookie (samesite="Lax" allows top-level form POST), and the validation runs — leaking JSON body inspection results. Validate-portfolio is read-only-ish but the CSRF surface will grow. | FIX: require Bearer header for state-changing POSTs, OR add CSRF token to forms, OR stricter samesite="Strict".

[P2] http_retry.py:40-49 — Backoff jitter is `random.uniform(0, wait * 0.1)` which is only 10% — under thundering-herd retry-after-503, all clients retry within a ~10% jitter window. Standard recommendation is full jitter (`random.uniform(0, wait)`). | FIX: use full jitter or decorrelated jitter for AWS-style backoff.

[P2] http_retry.py:44-49 — `retry_after` from Telegram is taken as seconds and assigned to `wait` UNCAPPED. A misbehaving Telegram response (or attacker-spoofed via DNS hijack) can return retry_after=86400, parking the loop for 24h. | FIX: cap `retry_after` at a sane maximum (e.g. 60s).

[P2] gpu_gate.py:99-102 — `_write_lock` writes 4 fields (model|pid|ts|tid) but `_read_lock` parses only 3 (model, pid, ts). The thread-id is dropped on read, which means no observability of which thread actually held the lock; for debugging the 25h wedge this is the missing info. | FIX: parse the 4th field, log it on stale-break.

[P2] gpu_gate.py:220-227 — `os.O_CREAT | os.O_EXCL | os.O_WRONLY` is atomic CREATE but doesn't fsync. After write + close, a power-loss before the metadata is journaled can leave a zero-byte lock that `_read_lock` reads as model=unknown, pid=0, and `_pid_alive(0)` returns False so it gets reaped immediately on next acquire — actually a graceful behavior, but worth documenting. | FIX (optional): fsync the lockfile after write.

[P2] process_lock.py:78-99 — `_write_lock_metadata` writes pid/started/owner space-separated but the format `key=value` with potential spaces in values (e.g. owner names) is parsed by no consumer in this codebase, so it's effectively a write-only debug breadcrumb. If a consumer is added that splits on space, owner="my service" produces three k=v pairs. | FIX: emit JSON instead, or document that values must be space-free.

[P2] dashboard/app.py:84-94 — `_cached_read` releases the lock between cache-miss check and `read_fn()` (line 91). Two concurrent misses both call `read_fn()` (full JSONL scan, 80MB file) — same dogpile that shared_state.py:79-90 fixed. | FIX: hold lock across read_fn OR add loading-set pattern.

[P2] message_store.py:188-219 — `send_or_store` writes the message to JSONL twice on send-failure paths: log_message inside `should_send` block fires after the failed send, but if `_do_send_telegram` raises (not just returns False), the JSONL log is skipped and the message is lost from history. | FIX: wrap `_do_send_telegram` in try/except, log_message in finally.

[P2] telegram_notifications.py:54-60 — The "Layer 1 disabled" gate at line 45-46 returns True (success) without logging; an operational flip from layer1_messages=true to false silently drops the message with only a debug-level log. Combined with no metric, it's invisible. | FIX: log at INFO level on every drop.

[P2] log_rotation.py:298-309 — `gzip.open(gz_path, "rt")` then re-write opens the SAME path for write before the read handle is closed. On Windows, this raises PermissionError mid-rotation, leaving the archive in inconsistent state. | FIX: use a temp file + rename pattern; close read handle before opening write handle.

[P2] dashboard/app.py:101-103 — `_read_json` cache key is `f"json:{path}"` — `path` is a `Path` object stringified to absolute path. Different callers using relative vs absolute paths get different cache entries even when they read the same file. Minor cache-coherence issue. | FIX: normalize to `path.resolve()` in cache key.

[P2] message_throttle.py:107-111 — `_send_now` reads state, mutates, writes — same TOCTOU race as queue_analysis but in a different code path. | FIX: same lock fix as P0 above.

[P2] file_utils.py:202-263 — `atomic_append_jsonl` doesn't size-cap; an unbounded entry (e.g. 10MB compact summary mistakenly logged) causes a 10MB single line, breaking `last_jsonl_entry` (only reads last 4KB so it can't find any line boundary) and `load_jsonl_tail` defaults (512KB). | FIX: warn and truncate entries above a sane size (e.g. 64KB).

[P2] alert_budget.py:48-52 — `flush_buffer` returns the buffered messages but the caller has no way to mark them "sent later" — they're flushed and lost from internal state, but if the send fails after flush, the messages are gone. | FIX: pass a callback that confirms send before clearing buffer.

[P2] journal_index.py:198-207 — Stop-word list is hardcoded English-only; reasoning text occasionally contains Swedish (user is CET/Sweden, ISKBETS commands), Swedish stop words leak into BM25 tokens and skew relevance toward common-word matches. | FIX: extend stop list or use language detection.

[P2] subprocess_utils.py:280 — `shell=True` for a PowerShell command with embedded JSON+escapes is a code-smell; on Windows shell quoting through cmd.exe wrapping powershell.exe is fragile. The current escape (`\\\"`) works but a future maintainer adding a parameter could trivially break it. | FIX: use list args without shell=True; pass the script via -Command argv directly.

[P3] file_utils.py:307-310 — `last_jsonl_entry` reads only the last 4KB; if the last JSONL entry is larger than 4KB, the function returns the second-to-last entry or None. Edge case but quietly wrong. | FIX: grow read window if no newline boundary is found in the initial chunk.

[P3] shared_state.py:25-35 — `_LOADING_TIMEOUT = 120` is hardcoded; LLM rotation cycle is 3 × 900s TTL = 2700s (45 min). A legitimately slow Chronos forecast (>120s) is evicted from `_loading_keys` while still running, causing a duplicate enqueue. The comment claims "stuck loading keys older than _LOADING_TIMEOUT seconds" but doesn't account for slow-but-alive work. | FIX: add a heartbeat callback for long-running fetchers, OR raise timeout to 300s for LLM keys.

[P3] dashboard/app.py:84-94 — Cache has no max size; `_cache` grows unboundedly with every distinct path queried. With many JSONL paths and limit variants, memory grows over time. Minor for current scale. | FIX: cap with LRU.

[P3] dashboard/app.py:935-936 — `_API_ACCURACY_CACHE` is a module-level dict, no lock around the read+update at lines 952-1012. Concurrent dashboard requests can race during cache rebuild — both compute, both write, harmless waste but worth a lock. | FIX: lock or use the helper pattern from `_AVANZA_CACHE_LOCK`.

[P3] dashboard/app.py:807-817 — `/api/summary` reads 4 files synchronously with no overall TTL; 5s `_DEFAULT_TTL` per file means worst case each request hits 4 separate caches. With dashboard polling every 5s, the TTL exactly matches polling — nothing benefits. | FIX: align summary TTL or extend per-file TTLs.

[P3] gpu_gate.py:269 — `_release_lock()` uses `unlink(missing_ok=True)` swallowing all OSError. If unlink fails (e.g. another process took the file), the gate releases the thread lock anyway. Then the next acquirer does atomic-create which now fails (file still exists), falls into stale-check path — works but obscures the failure. | FIX: log.warning if unlink fails.

[P3] dashboard/auth.py:61-66 — `_read_config_uncached` swallows all OSError; on permission-denied (config.json briefly locked by another writer atomic-replacing it), returns {} — and then `_get_dashboard_token` returns None — and `require_auth` falls through to "no token configured = allow" path. A transient config read failure DISABLES auth entirely for the cache TTL window (60s). | FIX: distinguish FileNotFoundError (no config = allow) from OSError (transient = deny / use stale cached value).

[P3] message_store.py:65-69 — `_normalize_message_whitespace` mangles intentional indentation in code blocks except those exactly wrapped in single backticks at start AND end of line. Markdown code fences (triple-backtick blocks) lose their internal structure. | FIX: detect code fence blocks and skip whitespace normalization.

[P3] http_retry.py:33-34 — Other HTTP methods (PUT, DELETE, PATCH) bypass json_body — silent. Only GET/POST honor it. Stop-loss API in trading uses POST (covered) but future PUT calls would silently lose body. | FIX: extend the elif branches to handle PUT/DELETE/PATCH with json_body.

[P3] log_rotation.py:401-403 — After truncating the file, `with open(filepath, "w")` doesn't fsync; power-loss between truncate and any subsequent write leaves a zero-byte file with a metadata that looks rotated. | FIX: fsync after truncate.

[P3] alert_budget.py:30-33 — `_prune_old` is called inside `should_send` and `remaining_budget` but NOT in `flush_buffer`; the buffer can stuff stale messages from hours ago. | FIX: prune in flush_buffer.

[P3] prophecy.py:317-325 — Progress percentage calculation: `(current - entry_price) / (target - entry_price)`. If target == entry_price the calculation is skipped, but if target < entry_price (bearish belief, target lower than entry), the math returns a NEGATIVE progress for a price drop, which is correct math but the dashboard probably expects 0-100. | FIX: signed-aware progress, or document.

[P3] journal.py:75-149 — `build_context` uses fromisoformat without timezone fallback; entries written from Python before 3.11 may produce naive datetimes that compare-fail with cutoff (which is UTC-aware). The handler in `load_recent` already swallows ValueError from this, but build_context's `_entry_age_hours` will raise on naive-aware comparison — handled by the caller's broad except. | FIX: assume UTC if tzinfo is None.

[P3] gpu_gate.py:144-159 — Sweeper sleeps FIRST then sweeps, so on first launch there's a 30s window where nothing runs. If the loop boots after a crash with a stale lock, recovery is delayed 30s. | FIX: sweep once before the sleep-loop.

[MAYBE] dashboard/app.py:866-867 — `/api/mstr_loop` uses raw `import json as _json; with open(...)` instead of file_utils helpers. Violates CLAUDE.md rule 4 ("Atomic I/O only ... never raw json.loads(open(...))"). The file is read-only here so atomicity isn't strictly required, but the rule is unambiguous. | FIX: switch to `file_utils.last_jsonl_entry()` (also fixes the full-file-scan inefficiency P1 above).

[MAYBE] api_utils.py:30-31 — `with open(config_path, encoding="utf-8") as f: _config_cache = json.load(f)` is raw json — violates CLAUDE.md rule 4. Same comment as above — read-only path so atomicity not the issue, but the rule mandates load_json. | FIX: switch to `file_utils.load_json`.

[MAYBE] file_utils.py:55-58 — atomic_write_json uses default=str for non-serializable types. This silently coerces datetimes/Decimal/Path to strings without warning; if a caller accidentally passes a class instance, it gets stringified to `<MyClass object at 0x...>` and silently corrupts the file. | FIX: log.warning when default=str fires (requires custom JSONEncoder).

[MAYBE] gpu_gate.py:30 — `_THREAD_LOCK = threading.Lock()` not `RLock`. If a future code path inside a held gpu_gate calls another gpu_gate (e.g. forecast-during-ministral), it deadlocks instantly. Currently no such path exists, but the contract isn't enforced. | FIX: switch to RLock for safety, or document non-reentrancy.

[MAYBE] subprocess_utils.py:250-327 — `kill_orphaned_llama` doesn't grace-period kill — it TerminateProcess immediately on detection. A briefly-orphaned process (parent restarting) gets killed even though the new parent would otherwise reclaim it. With PF-DataLoop's auto-restart 30s delay this is plausible. | FIX: optional grace period (e.g. 60s) before terminating.

[MAYBE] dashboard/app.py:54-61 — CORS allow-list permits credentials=false but `Access-Control-Allow-Headers` permits Authorization. Combined with the 401-response cookie set, a CORS preflight that succeeds and a subsequent CORS-credentialed request will fail because credentials=false; the dashboard cookie won't be sent cross-origin from the allowed origins. Likely intentional (no CSRF surface) but worth confirming. | FIX: confirm intent, document.

[MAYBE] telegram_poller.py:108-121 — `_poll_loop` catches every exception and continues with `time.sleep(5)`. If the failure is permanent (config corruption, token revocation), the loop spins forever logging a warning every 5s — log noise is high. | FIX: exponential backoff on consecutive failures, or circuit-break after N failures.

---

COUNT: P0=5, P1=18, P2=18, P3=11, MAYBE=7, total=59
