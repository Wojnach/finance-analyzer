# Adversarial Review — 8 infrastructure (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/telegram_poller.py:361` — `atomic_write_json` on a symlinked
  `config.json` **destroys the symlink** and replaces it with a regular file.
  `config.json` is documented in CLAUDE.md as a symlink to
  `C:\Users\Herc2\.config\finance-analyzer\config.json` (an external file
  containing API keys, intentionally kept outside the repo since the
  Mar 15 leak). `file_utils.atomic_write_json` does
  `os.replace(tmp, str(path))`, which on both POSIX and Windows replaces
  the **link target** when path is a symlink — but since `tmp` is a real
  file in the same dir, the rename overwrites the symlink itself, severing
  the link. The very next `/mode` command from the user therefore:
    1. silently breaks the symlink (next process restart sees a stale config
       with whatever was in memory when the write happened),
    2. permanently embeds the API keys inside the repo's working tree
       — re-introducing the exact failure mode the Mar 15 leak retired.
  The BUG-210 size guard at line 350 protects against writing an empty
  config, but does nothing to preserve the symlink. Fix: `os.path.realpath`
  the target before atomic write, or refuse to write through symlinks at
  the `atomic_write_json` boundary.

- `portfolio/file_utils.py:281-284` — `atomic_append_jsonl` writes inside
  the sidecar lock, but the lock is **per-path**. `log_rotation.rotate_jsonl`
  (line 270 of log_rotation.py) acquires the same sidecar lock, ✓.
  However, **the sidecar lock is identified by the lockfile path
  `<parent>/.<name>.lock`**, computed from the caller's `path` argument.
  If two callers use different aliases of the same file (e.g. one passes
  `data/signal_log.jsonl`, another passes the resolved absolute path), they
  get **different** sidecar locks and the contract breaks. The codebase
  consistently uses absolute paths via `BASE_DIR / "data" / "..."` so this
  is latent rather than active, but a single drift would silently re-introduce
  the signal-log torn-write incident of 2026-05-11. Add `Path(path).resolve()`
  at lock-key derivation time so aliased paths collapse to the same lock.

- `portfolio/claude_gate.py:662 vs 777` — `invoke_claude_text` return-type
  mismatch. Signature declares `tuple[str, bool, int]` (3 values), docstring
  at line 670 says 4, and the body returns 4 values on every path:
  ```python
  return text, status == "invoked", exit_code, status        # line 777
  return "", False, -1, "blocked"                            # line 683
  return "", False, -1, "error"                              # line 688
  ```
  Any caller that follows the type hint and does
  `text, success, exit_code = invoke_claude_text(...)` raises
  `ValueError: too many values to unpack`. Worse, if a caller uses
  `tuple` indexing (`result[2]` for exit_code), they get the right value
  by accident — until someone "fixes" the signature to match the docstring
  and unpacking-callers break. Pick one shape and align signature + all
  call sites.

## P1 — high-confidence bugs (should fix)

- `portfolio/log_rotation.py:432-440` — `rotate_text` is **not atomic** and
  drops data on concurrent writers. The flow is:
  ```python
  if compress:
      _gzip_file(filepath, rotation_1_gz)  # read source → gzip dest
  ...
  with open(filepath, "w", encoding="utf-8") as f:
      f.write("")                          # TRUNCATE source
  ```
  Between the gzip copy and the truncate, any process that appends to
  `agent.log` / `loop_out.txt` / `golddigger_out.txt` (all of which are
  redirected stdout from running batch files) writes lines that survive
  the truncate window — those lines are then **lost** when `open(..., "w")`
  truncates. The classic Unix-rotation fix is rename-then-open-new, which
  costs an inode but preserves in-flight appends. As-is, the daily rotation
  scheduled task silently truncates whatever the loop emitted in the
  millisecond between gzip-copy completion and `open("w")`.

- `portfolio/llama_server.py:179-180` — `_kill_server_by_pid` references
  `pid` outside its assignment scope on the exception path:
  ```python
  try:
      if os.path.exists(_PID_FILE):
          with open(_PID_FILE, ...) as f:
              content = f.read().strip()
          if content:
              pid = int(content.split(":")[0])    # only assigned here
              ...
  except Exception:
      logger.debug("Failed to kill server pid=%s", pid)  # ← NameError if exception fires before line 167
  ```
  If `_PID_FILE` exists but is unreadable (locked, permission), `open()`
  raises before `pid` is bound, the bare `except` catches it, and
  `logger.debug("...", pid)` raises `NameError`. That NameError isn't
  caught — it escapes `_kill_server_by_pid` and breaks `_stop_server`,
  which would prevent any subsequent model swap. Initialize `pid = None`
  before the try.

- `portfolio/claude_gate.py:597-607` — Auth-failure detection runs over
  `_stdout`/`_stderr` from `_run_with_tree_kill`, but on the
  `Exception` path at line 608 those variables remain at their pre-init
  values (`_stdout = None` from line 567, `_stderr` never set). The next
  line through line 633 reaches `_log_invocation(...)` and the json-parse
  block at line 617 — the check `if status not in ("timeout",) and _stdout is not None`
  correctly skips parsing on `_stdout is None`, but a malformed Popen path
  that raises BEFORE assigning `_stdout` (e.g. `_clean_env()` raising,
  cwd missing) lands in the `except Exception as e` handler at line 608
  with status="error" and exit_code=-1 — no auth-failure check ever runs.
  This is the documented 3-week-outage pattern: an upstream failure
  bypasses the detector. Belt-and-braces fix: pre-compute `_stderr = None`
  next to `_stdout` at line 567 and explicitly call `detect_auth_failure`
  inside the exception handler against `str(e)` so even a Popen-time
  failure surfaces.

- `portfolio/subprocess_utils.py:214-225` — `kill_orphaned_by_cmdline`
  attempts to escape PowerShell wildcards with backtick prefixes:
  ```python
  safe_pattern = (
      pattern.replace("'", "''")
      .replace("[", "``[")
      .replace("]", "``]")
      .replace("*", "``*")
      .replace("?", "``?")
  )
  ps_cmd = (
      "...| Where-Object {{ $_.CommandLine -like '*{safe_pattern}*' }} ..."
  )
  ```
  PowerShell **single-quoted strings do not interpret backtick escapes** —
  inside `'...'` the backtick is a literal character. The backtick-escaping
  here is therefore inert, and the actual `-like` wildcard semantics
  (`*`, `?`, `[abc]`) apply unmodified to the input pattern. If a future
  caller passes a pattern containing `*` or `[`, this expands to a much
  broader match than intended — and `kill_orphaned_by_cmdline` does
  `taskkill /F /PID` against every matching PID. Worst case: a caller
  passes `pattern="python.exe"` thinking it's a literal, but if someone
  ever passes a pattern with wildcards, the function silently kills
  unrelated processes. Fix: use PowerShell `-cmatch` with a regex-escaped
  pattern, or pass the pattern through a parameter (`-ArgumentList`) so
  PowerShell sees it as data not code.

- `portfolio/feature_normalizer.py:35-40` — `_ensure_buffer` is racy across
  the 8-worker `ThreadPoolExecutor`:
  ```python
  def _ensure_buffer(ticker: str, indicator: str) -> deque:
      key = (ticker, indicator)
      if key not in _buffers:
          _buffers[key] = deque(maxlen=_DEFAULT_WINDOW)
      return _buffers[key]
  ```
  Two threads on the same `(ticker, indicator)` can both pass the
  `not in` check and both create a fresh deque — the second wins, the
  first thread's `update()` lands in an orphaned deque that's never
  consulted again. This is benign for the eventual z-score (the deque
  fills back up in <100 ticks) but it silently drops the first ~10-20
  samples after a fresh start, biasing the `_MIN_SAMPLES=20` gate's
  first crossover. Add a module-level lock around the `if key not in`
  check or switch to `_buffers.setdefault(key, deque(maxlen=...))`.

- `portfolio/file_utils.py:244-247` — Windows `msvcrt.locking(LK_LOCK, ...)`
  is **not pure-blocking**: it retries for ~10 s with 1 s sleeps and then
  raises `OSError` (`EDEADLOCK`/errno 36) if it still can't acquire. Under
  heavy log-rotation contention, `atomic_append_jsonl` can therefore raise
  from `jsonl_sidecar_lock` even though the docstring promises a blocking
  wait. The caller in `claude_gate._log_invocation` swallows it (line 339)
  so an invocation record silently goes missing, which feeds back into
  `_count_today_invocations` returning stale counts — the rate-limit warning
  threshold becomes lossy under contention. Either switch to `LK_NBLCK` with
  a manual retry loop the caller controls, or document the 10 s ceiling so
  callers can choose between "block forever, possibly stuck" and "10s and
  raise". Today neither contract is honored.

- `portfolio/http_retry.py:44-48` — `Retry-After` is honored only on 429
  responses, only via the **Telegram-specific** JSON body shape
  `parameters.retry_after`. The HTTP standard header `Retry-After:` (used
  by Alpaca, NewsAPI, FRED, Alpha Vantage, BGeometrics — every other API
  consumed by this codebase) is ignored. Even 429 from a non-Telegram
  endpoint falls back to exponential backoff:
  ```python
  if resp.status_code == 429:
      try:
          retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
      except Exception:
          retry_after = wait
      wait = retry_after
  ```
  Worst case: Alpaca returns 429 with `Retry-After: 60`, this code waits
  `backoff * 2^attempt` (1, 2, 4 s) and burns three of the retries hammering
  Alpaca, which extends the cooldown. Fix: read `resp.headers.get("Retry-After")`
  first, fall back to JSON body, fall back to exponential.

- `portfolio/http_retry.py:34` — Non-GET, non-POST methods drop the
  `json_body`:
  ```python
  resp = requester.request(method, url, headers=headers, params=params, timeout=timeout)
  ```
  No `json=json_body` argument. If a caller ever uses `PUT`/`PATCH`/`DELETE`
  with a JSON body (Avanza order-edit endpoints do exactly this) the body
  is silently dropped. Today no such caller exists in-tree, but the function
  signature accepts `json_body=` for any method, making this a future-bug
  trap.

- `portfolio/llama_server.py:419` — Plex-active swap abort path is racy.
  ```python
  plex_active = _plex_transcode_active()    # line 413
  ...
  _stop_server()                            # line 421 — KILLS THE EXISTING MODEL
  waited = _wait_for_vram_reclaim(...)
  if plex_active:
      free_now = _query_free_vram_mb() or 0
      if free_now < 7168:
          return False                      # ABORT — but server is already dead
  ```
  If the abort fires, `_stop_server()` has already killed whichever model
  was loaded. The next signal cycle's caller hits `_ensure_model` →
  `_start_server` → potentially fails again (Plex still active) →
  another kill cycle. Net effect: under sustained Plex transcoding the
  llama-server flaps off entirely, and `query_llama_server` returns None
  forever — the metals / crypto LLM signals silently fall back to subprocess
  (which the comment at 436 acknowledges is "slower than HTTP" — typo for
  "more likely to crash Plex"). The abort decision should run **before**
  `_stop_server()` so a Plex-busy state leaves the existing model loaded.

## P2 — concerns / smells (worth addressing)

- `portfolio/api_utils.py:30-35` — Violates the CLAUDE.md atomic-I/O rule:
  ```python
  with open(config_path, encoding="utf-8") as f:
      _config_cache = json.load(f)
  ```
  Uses raw `open()` + `json.load`, not `file_utils.load_json`. Mid-rename
  during a config update (which destroys the symlink — see P0 above) this
  can raise on Windows. Compounding the symlink issue, this path is hit
  early in startup so a transient unreadable config crashes the loop.
  Switch to `load_json(config_path)` with the missing-file guard.

- `portfolio/gpu_gate.py:202-203` — In-process `_THREAD_LOCK.acquire(timeout=...)`
  uses **wall-clock**-based remaining time:
  ```python
  remaining = deadline - time.time()
  thread_acquired = _THREAD_LOCK.acquire(timeout=max(0, remaining))
  ```
  `deadline = time.time() + timeout` on line 198. If the system clock
  jumps backwards (NTP correction, DST adjustment), `remaining` can
  briefly become much larger than the configured `timeout`, lengthening
  the wait beyond what the caller asked for. The fix is the standard
  `time.monotonic()` pattern. Same issue in `_acquire_file_lock` at
  llama_server.py:496-497.

- `portfolio/gpu_gate.py:155-176` — `_start_sweeper` creates a daemon
  thread on first `gpu_gate()` call. Daemon threads are killed abruptly
  at interpreter exit without unwinding context managers, so a long-running
  signal that holds the lock during shutdown can leave the lock file behind
  even though the sweeper would otherwise reap it on next start. The
  sweeper-thread reap logic at line 116-130 is fine, but the existence
  of a daemon-thread sweeper means a quick start/stop cycle within the
  sweeper's 30 s tick window won't reap stale locks until the next start.
  Acceptable but worth documenting.

- `portfolio/llama_server.py:515` — Tasklist-based PID liveness check uses
  substring matching:
  ```python
  result = subprocess.run(["tasklist", "/FI", f"PID eq {lock_pid}", ...], ...)
  if str(lock_pid) not in result.stdout:
      os.remove(_LOCK_FILE)
      continue
  ```
  Tasklist's PID filter narrows to exact matches, so this works today.
  But the cross-reference `if str(lock_pid) not in result.stdout` is
  a substring test — if Microsoft ever localizes tasklist output to
  include the PID elsewhere (header row, error message), the check could
  flip false. Use returncode + structured CSV parsing.

- `portfolio/subprocess_utils.py:325-335` — `kill_orphaned_llama` is
  vulnerable to **Windows PID reuse**:
  ```python
  handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(ppid))
  if handle:
      parent_alive = True
  ```
  If the original parent has died and another process has been assigned
  the same PID, this thinks the parent is alive and skips the kill. The
  llama-completion.exe is then never reaped. Windows kernel guarantees
  PID uniqueness only within a process's lifetime; reuse can happen
  within seconds on a busy machine. A `GetProcessTimes` cross-check
  against the orphan's start time vs the (alleged) parent's start time
  would close this — if the alleged parent started AFTER the orphan, it's
  not the real parent.

- `portfolio/shared_state.py:278` — Rate-limiter slot reservation:
  ```python
  self.last_call = self.last_call + self.interval if wait_time > 0 else now
  ```
  Works correctly for steady-state but reserves a slot **in the future**
  for the calling thread without verifying the thread actually sleeps.
  If the calling thread is killed/interrupted (e.g. by `KeyboardInterrupt`
  between `_lock` release and `time.sleep`), the slot is reserved and
  future calls wait extra time even though nobody consumed the slot. Minor;
  recovery is on the next call's `now > self.last_call` correction.

- `portfolio/health.py:194-196` — In `check_agent_silence`, the writeback
  reloads health state under the lock:
  ```python
  with _health_lock:
      wb_state = load_health()
      wb_state["last_invocation_ts"] = last_ts
      atomic_write_json(HEALTH_FILE, wb_state)
  ```
  This is a read-modify-write, but it runs only if the cache miss
  succeeded reading from `invocations.jsonl`. If two threads miss the
  cache simultaneously they both parse the JSONL (expensive) then both
  write back the same value (cheap, idempotent). Acceptable, but the
  parse is duplicated — the dogpile pattern in `shared_state._cached` is
  the right model.

- `portfolio/telegram_notifications.py:39-46` — Three layered gates
  (`NO_TELEGRAM`, `mute_all`, `layer1_messages`) all return `True` to
  the caller. From the caller's perspective "telegram sent successfully"
  is indistinguishable from "telegram silenced". Callers cannot detect
  whether a message actually went out. The `message_store.send_or_store`
  path has the same shape but at least logs `[mute_all]` at info level —
  this raw `send_telegram` does the same but the True-on-skip return value
  has bitten before (a "successful" send that nobody saw). Consider a
  3-tuple return (`sent`, `skipped_reason`, `truncated`).

- `portfolio/message_store.py:37-49` — The `_COMMON_MOJIBAKE_REPLACEMENTS`
  dict-literal contains duplicate visible keys (e.g. `"â": "—"`,
  `"â": "'"`, `'â': '"'`). Because Python silently retains only the
  last value for each unique-byte-sequence key, several mojibake repairs
  defined earlier in the dict are unreachable in practice. The bytes
  themselves likely differ (these look like distinct UTF-8 sequences
  that all render as the same glyph in some terminals), but a reader
  staring at source can't tell whether the dict has 11 distinct repairs
  or 4 — write the bytes explicitly (`"\xe2\x80\x94"` etc.) to make the
  intent unambiguous and verifiable.

- `portfolio/process_lock.py:65-69` — `msvcrt.locking(LK_NBLCK, 1)`
  locks 1 byte at the current file position (which is 0 after the
  `fh.seek(0)`). The metadata write at line 102-104 then does
  `fh.seek(0); fh.truncate(); fh.write(...)` — truncation while the byte
  is locked is undefined behavior on Windows for the locking semantics
  (msvcrt locks ranges, not the file itself). In practice this works
  because the lock byte is recreated by the subsequent write, but if the
  write fails between truncate and content-emit the lock byte is gone
  and the OS may release the lock. Better: open with `O_APPEND` semantics
  and never truncate the lock file.

## Did NOT find

1. **Silent failures**: looked at `claude_gate.detect_auth_failure` (the
   `--bare` 3-week-outage detector) — the start-of-line + fenced-code-block
   gating is conservative and correctly defends against echo-of-echo
   loops; the additional auth-failure scan against the Popen-time
   exception path is a P1 above, but the existing detector itself works
   as designed.
2. **Race conditions**: covered above (sidecar lock alias, `_ensure_buffer`,
   `_THREAD_LOCK` wall-clock).
3. **Money-losing bugs**: this subsystem is plumbing, not trading logic —
   no PnL math, fees, or stop-loss placement in scope.
4. **State corruption**: `atomic_write_json`/`atomic_append_jsonl`/the
   sidecar lock pattern are correctly fsync+rename. The remaining hazards
   (symlink replacement, alias-mismatched locks) are P0/P1 above.
5. **Logic errors that pass tests**: spot-checked the `is_llm_on_cycle`
   rotation math and the `_next_slot` worked-example reasoning in
   `llm_prewarmer.py:93` — math agrees with the comments.
6. **Resource leaks**: subprocess tree-kill in `claude_gate._run_with_tree_kill`
   correctly drains pipes post-kill; `gpu_gate` releases lock on yield;
   Job Object kwargs in `subprocess_utils._run_with_job_object` close the
   handle in `finally`. The orphan-llama detector has the PID-reuse issue
   (P2) but is itself a safety-net for leaks rather than a leak source.
7. **Time/timezone bugs**: wall-clock vs monotonic noted at gpu_gate /
   llama_server (P2). `health.check_outcome_staleness` uses `time.time()`
   for the "now" reference and `datetime.fromisoformat()` for entry
   timestamps — both naive-to-aware aware via the `_parse_ts` helper in
   log_rotation, but `health.check_outcome_staleness` parses directly with
   `datetime.fromisoformat` and `.timestamp()` which is correct for the
   UTC ISO-8601 timestamps the loop writes.
8. **API misuse**: this subsystem doesn't speak Avanza / Binance / Alpaca
   directly — `price_source.py` is a router and uses the documented
   `binance_klines` / `alpaca_klines` helpers. No `10m` interval, no
   stop-loss endpoints in scope.
9. **Trust boundary violations**: `kill_orphaned_by_cmdline` builds a
   PowerShell expression with caller-supplied pattern (P1 above — escaping
   is inert, not actively exploitable because all callers are in-tree).
   `telegram_poller._handle_mode_command` reads `mode_arg` from
   user-supplied Telegram text but allow-lists to `{"signals", "probability"}`
   before writing config — safe. No `eval` / `exec` / `shell=True` outside
   the `kill_orphaned_llama` case which is intentional.
10. **Incorrect assumptions about partial state**: `prophecy.evaluate_checkpoints`
    iterates `data["beliefs"]` with `for i, belief in enumerate(...)` while
    mutating `data["beliefs"][i]` — same-index mutation, not a mutation-during-
    iteration bug. The `between` comparison at prophecy.py:251-256 correctly
    guards on `isinstance(target, (list, tuple))` and `len(target) == 2`
    before indexing. `feature_normalizer.normalize` correctly handles
    cold-start and zero-variance cases.
