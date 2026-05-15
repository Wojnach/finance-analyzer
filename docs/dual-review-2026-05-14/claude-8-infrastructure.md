# Adversarial Review — 8 infrastructure (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/file_utils.py:228-238` — `jsonl_sidecar_lock` path normalization is missing.
  ```python
  path.parent.mkdir(parents=True, exist_ok=True)
  lock_path = path.parent / f".{path.name}.lock"
  ```
  `path = Path(path)` (line 226 implied) does NOT resolve relative paths. Caller A passes `Path("data/signal_log.jsonl")`, caller B passes `Path("/abs/data/signal_log.jsonl")`, caller C passes `Path("Q:/finance-analyzer/data/signal_log.jsonl")`. All three resolve to the SAME file on disk, but `path.parent / f".{path.name}.lock"` produces three DIFFERENT lock paths if the lock files end up in different directories due to CWD differences. The OS-level lock is keyed by the file descriptor of the *opened lock file*, so two callers opening different lock paths get independent locks. Torn-write hazard: simultaneous appends bypass the lock contract because they're locking *different lock files*. Fix: `path = Path(path).resolve()` at function entry.

- `portfolio/telegram_poller.py` (per subagent) — `atomic_write_json` on the **symlinked** config.json severs the symlink and writes the file inline in the repo. CLAUDE.md: "NEVER commit config.json — exposed API keys on Mar 15, 2026". If this path ever executes, the file becomes a regular file containing API keys, and a subsequent `git add data` could commit it (since the symlink target is outside the repo, git ignores it; but a regular file IS picked up). Verify by checking what telegram_poller writes to that path and whether the path is the symlinked config.

## P1 — high-confidence bugs (should fix)

- `portfolio/http_retry.py:44-49` — Retry-After header is NOT honored; only Telegram-style JSON `parameters.retry_after` is read.
  ```python
  if resp.status_code == 429:
      try:
          retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
      except Exception:
          retry_after = wait
      wait = retry_after
  ```
  Every API except Telegram (Binance, Alpaca, NewsAPI, FRED, BGeometrics, etc.) returns `Retry-After: <seconds>` as an HTTP header on 429. We ignore it and use our exponential-backoff `wait`. If Binance returns `Retry-After: 60` (typical IP-banned response), we wait 2-8 seconds and re-hit the same IP-ban window → escalates the ban. P1 because the 24/7 loop hits this scenario weekly during volatility spikes.

- `portfolio/http_retry.py:33-34` — non-GET/POST methods silently drop `json_body`.
  ```python
  else:
      resp = requester.request(method, url, headers=headers, params=params, timeout=timeout)
  ```
  No `json=json_body` passed. Avanza PUT/DELETE endpoints (if anyone uses them via this wrapper) would fire body-less requests. Quiet semantic bug.

- `portfolio/log_rotation.py:433-440` — `rotate_text` `copy2` then truncate.
  ```python
  if compress:
      _gzip_file(filepath, rotation_1_gz)
  else:
      shutil.copy2(filepath, rotation_1)
  # Truncate the original file
  with open(filepath, "w", encoding="utf-8") as f:
      f.write("")
  ```
  Race window between copy2 and the truncate at line 439: any writer appending to filepath in this window has its bytes in the source (which is about to be truncated) but NOT in the rotation_1 archive. For text-mode files where the writer flushes after every line (e.g., `subprocess.Popen(..., stdout=fh)` for `agent.log`), this loses lines that happened during the rotation tick. Take the sidecar lock during rotation, or use `os.rename` semantics for the archive (atomic move) rather than copy+truncate.

- `portfolio/claude_gate.py:662` — `def invoke_claude_text(...) -> tuple[str, bool, int]:` annotation says 3-tuple but body at line 777 returns 4-tuple `(text, status == "invoked", exit_code, status)`. Callers (iskbets line 324) unpack 4 — works because Python tuples don't enforce annotations. mypy strict mode would catch this. Update the annotation to `tuple[str, bool, int, str]`.

- `portfolio/file_utils.py:240-258` — `jsonl_sidecar_lock` uses `_msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)` which per Microsoft docs raises after retrying for 10 seconds. The comment claims "blocking" but it's actually a 10-second-timeout-blocking. Under heavy contention (8-worker pool, signal_log appender, rotation), one of the workers can hit the 10s wall, raise OSError, and the caller in `atomic_append_jsonl` propagates → ticker dropped that cycle. Fix: loop with retry or use a different Win32 lock primitive.

- `portfolio/feature_normalizer.py:35` (per subagent) — `_ensure_buffer` race drops first samples after fresh start. Feature engineering for ML signals can produce wrong training data.

- `portfolio/llama_server.py:419` (per subagent) — Plex-active swap aborts AFTER `_stop_server` kills the existing model. If swap target check fails after stop, no model is loaded → all LLM signals dark until next cycle.

- `portfolio/claude_gate.py:608-610` — `except Exception as e: status = "error"`. The catch is too broad — it swallows `BaseException` subclasses like `KeyboardInterrupt`. Use `Exception` only and let KeyboardInterrupt propagate.

- `portfolio/subprocess_utils.py:214` (per subagent) — PowerShell single-quoted strings don't interpret backticks. Wildcard escape is inert. Could kill unrelated processes during cleanup. P1.

## P2 — concerns / smells (worth addressing)

- `portfolio/file_utils.py:66-86` — `load_json` catches `json.JSONDecodeError, ValueError` and returns default. If the file content is `null` (valid JSON), `json.loads` returns Python `None`, which is then returned to caller. Many callers don't distinguish "file missing" from "file contains null". Document or use a sentinel.

- `portfolio/log_rotation.py:438-440` — `with open(filepath, "w", ...): f.write("")` truncates by re-opening in write mode. On Windows, if any other process (Notepad, antivirus) holds the file open, this raises and the rotation is partial: archive created but source not truncated. Caller catches? Verify the calling layer handles this.

- `portfolio/claude_gate.py:316-326` — `record_critical_error(category="auth_failure", ...)` writes to `critical_errors.jsonl`. The startup check at CLAUDE.md scans this file. Verify the schema matches `check_critical_errors.py` expectations — context dict shape, timestamp format.

- `portfolio/prophecy.py` — macro belief tracking. Per CLAUDE.md the user has "silver prophecy: $120 target, 0.8 conviction". If `prophecy.json` is updated mid-cycle by a Layer 2 invocation, the in-memory copy in the running loop is stale. Verify reload on every read.

- `portfolio/telegram_notifications.py` — per subagent / general: rate-limit, queue overflow. If 100+ alerts queue up (crash storm) and Telegram is down, where does the queue live? In-memory only? Disk-backed?

- `portfolio/ministral_signal.py`, `portfolio/qwen3_signal.py` — local LLM signals. GPU lock contention is documented (gpu_gate.py). If lock holder dies (process killed), the lock file may not get released. Verify gpu_gate handles stale lock cleanup.

- `portfolio/process_lock.py` — singleton lock for main.py loop. The lock file is held until process exit. If the process is killed -9 / power loss, the file remains. Verify the next startup detects stale lock by PID check, not just file presence.

- `portfolio/http_retry.py:39-41` — `wait = backoff * (backoff_factor ** attempt)` then `jitter = random.uniform(0, wait * 0.1)`. 10% jitter is too narrow — when many callers retry simultaneously after a 503, they cluster and re-hammer the API. Spread jitter to [0, wait] for "full jitter" pattern.

- `portfolio/log_rotation.py:430-431` — archive paths `rotation_1` AND `rotation_1_gz` both computed but only one is used per branch. Confusing variable allocation.

- `portfolio/gpu_gate.py` (not read) — verify the lock file path uses `Path(...).resolve()` like file_utils should (per P0 above).

## Did NOT find

1. **Silent failures**: claude_gate's auth detector is correctly wired (line 274). Other paths log on swallow.
2. **Race conditions**: P0 lock-path normalization, P1 rotation race, P1 jsonl_sidecar_lock 10s Win32 timeout.
3. **Money-losing bugs**: indirect via lock corruption or stale config.
4. **State corruption**: atomic_write_json correct. Lock-path P0 is the gateway to corruption.
5. **Logic errors that pass tests**: http_retry Retry-After miss not caught by mock tests that don't simulate the header.
6. **Resource leaks**: `_kill_process_tree` covers Node subprocess descendants. GPU gate lock is the unverified one.
7. **Time/timezone bugs**: log_rotation uses `datetime.now()` — check tz; `datetime.now(UTC)` in most places.
8. **API misuse**: claude CLI args correct (`-p` text mode, `--allowedTools`).
9. **Trust boundary violations**: P1 PowerShell single-quote escape (subprocess_utils).
10. **Incorrect partial-state assumptions**: load_json `null` ambiguity (P2).
