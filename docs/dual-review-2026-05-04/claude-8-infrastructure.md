# Claude Review — infrastructure

## P0 (money-losing or data-corrupting)

- `portfolio/log_rotation.py:320-327` — `rotate_jsonl` writes temp file without fsync, no cleanup on exception
  ```python
  tmp_path = filepath.with_suffix(".tmp")
  with open(tmp_path, "w", encoding="utf-8") as f:
      for line in keep_lines:
          f.write(line + "\n")
  # NO f.flush() / os.fsync() before os.replace
  os.replace(tmp_path, filepath)
  ```
  Two problems: (1) No `f.flush(); os.fsync(f.fileno())` before `os.replace` — power-loss between close and replace can leave file in unknown state. `signal_log.jsonl` can be 68 MB; a torn rotate destroys it. (2) If `os.replace` raises, `tmp_path` is left behind. Compare to `atomic_write_json` which catches `BaseException` and unlinks tmp. Confidence 92.

- `portfolio/local_llm_report.py:37` — raw `path.read_text()` violates atomic I/O rule on actively-written JSONL
  ```python
  for line in path.read_text(encoding="utf-8").splitlines():
  ```
  CLAUDE.md Critical Rule 4: never raw `json.loads(open(...).read())`. `forecast_predictions.jsonl` is concurrently written by main loop. Partial read produces truncated JSON lines, `json.loads` silently skips them — corrupts accuracy stats in the report. Fix: `load_jsonl(path)`. Confidence 88.

## P1 (high-confidence bugs)

- `portfolio/log_rotation.py:298-309` — `rotate_jsonl` decompresses + recompresses gz archive in-place with truncate
  ```python
  if gz_path.exists() and policy.get("compress", True):
      existing_lines = []
      with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
          for existing_line in gf:
              existing_lines.append(existing_line.rstrip("\n"))
      all_lines = existing_lines + lines
      with gzip.open(gz_path, "wt", encoding="utf-8") as gf:  # overwrites in-place
          for line in all_lines:
              gf.write(line + "\n")
  ```
  Reads existing archive into memory, concatenates, opens same path in `"wt"` (truncates first). Crash or disk-full mid-write **permanently destroys historical archive**. Fix: write `.tmp.gz` sibling and `os.replace`. Confidence 87.

- `portfolio/claude_gate.py:271-279` — `_count_today_invocations()` scans entire JSONL on every call, outside `_invoke_lock`
  ```python
  def _count_today_invocations() -> int:
      today_str = datetime.now(UTC).strftime("%Y-%m-%d")
      count = 0
      for entry in load_jsonl(INVOCATIONS_LOG):  # scans entire file
          ts = entry.get("timestamp", "")
          if ts.startswith(today_str):
              count += 1
      return count
  ```
  Called at line 454, before `_invoke_lock` is acquired. Two concurrent callers (metals + main loop subprocess) read stale counts. After months of operation this becomes a full file scan on every invocation. Fix: `load_jsonl_tail` or counter in health_state. Confidence 85.

- `portfolio/vector_memory.py:272-281` — raw `open()` on JSONL violates atomic I/O rule
  ```python
  with open(JOURNAL_FILE, encoding="utf-8") as f:
      for line in f:
          ...
          entries.append(json.loads(line))
  ```
  Same rule violation as P0-2. Truncated last line during concurrent append silently dropped from ChromaDB embedding batch. Fix: `load_jsonl(JOURNAL_FILE)`. Confidence 83.

- `portfolio/gpu_gate.py:98-102` — `_write_lock` function defined but never called; dead code creates maintenance trap
  ```python
  def _write_lock(model_name: str):  # NEVER CALLED
      _GPU_LOCK_FILE.write_text(
          f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}",
          ...
      )
  ```
  Real path uses `os.write(fd, ...)` after `os.open(O_CREAT|O_EXCL)` for atomic create. Future dev calling `_write_lock` would bypass atomic-create. Confidence 80.

## P2 (concerns / smells)

- `portfolio/config_validator.py:58-60` — uses raw `open()` + `json.load()` on config.json
  ```python
  with open(CONFIG_FILE, encoding="utf-8") as f:
      config = json.load(f)
  ```
  Symlink target rewrite races possible. Should use `file_utils.load_json()`.

- `portfolio/shared_state.py:104-108` — `KeyboardInterrupt` cleanup misses `_loading_timestamps.pop`
  ```python
  except KeyboardInterrupt:
      with _cache_lock:
          _loading_keys.discard(key)
  ```
  `_loading_keys.discard(key)` called but `_loading_timestamps.pop(key, None)` is not. Stale timestamp blocks key in dogpile path until 120s `_LOADING_TIMEOUT` evicts. Asymmetric vs `Exception` path which does both.

- `portfolio/http_retry.py:44-49` — Telegram 429 `retry_after` accepted with no cap
  ```python
  if resp.status_code == 429:
      try:
          retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
      except Exception:
          retry_after = wait
      wait = retry_after
  ```
  Telegram can return `retry_after=86400` during flood. Calling thread blocks 24h. No cap. Suggest 300s max.

- `portfolio/ministral_trader.py:45-46` — fixed-path temp file for prompt
  ```python
  prompt_file = os.path.join(tempfile.gettempdir(), "ministral3_prompt.txt")
  ```
  Cross-process race between metals + main loop subprocess fallback. `gpu_gate` file lock prevents in practice but design is fragile.

## Did NOT find

1. `atomic_write_json` temp-file leaks — pattern correct (fsync before replace, `BaseException` cleanup).
2. `claude_gate` silent-failure bypass — `detect_auth_failure` correctly scans both stdout and stderr with line-limit guard.
3. GPU lock leak on exception — `gpu_gate` `try/finally` releases both file lock and `_THREAD_LOCK`.
4. Process lock false-stale detection — OS-level `fcntl.LOCK_EX | LOCK_NB` / `msvcrt.LK_NBLCK`, no PID check, no false-stale risk.
5. HTTP retry runaway — backoff capped implicitly by `retries=3` (1s, 2s, 4s).
6. Backtester lookahead — uses historical cache, not current snapshot. Outcomes per-entry from stored `change_pct`.
7. JSONL append races — `atomic_append_jsonl` uses sidecar lock file with `msvcrt.LK_LOCK` (blocking).
8. `shared_state._cached` dict mutation during iteration — `_tool_cache` LRU eviction inside `_cache_lock` uses `sorted_keys` separate list.
9. `prophecy.py` non-atomic write — correctly uses `atomic_write_json`.
10. LLM model load races — `query_llama_server` holds both `_thread_lock` and file lock through entire swap+query.
