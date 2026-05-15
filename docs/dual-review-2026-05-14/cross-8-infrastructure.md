# Cross-Critique — 8 infrastructure

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/file_utils.py` `jsonl_sidecar_lock` path normalization (P0).** Both reviewers identify that the sidecar lock is keyed by lock-file path derived from caller's `path` argument; different aliases (relative vs absolute) → different locks → torn-write contract broken. Codex links it to the 2026-05-11 signal_log torn-write incident. **Independent rediscovery, very high confidence.** Action: `Path(path).resolve()` at lock-key derivation.

- **`portfolio/telegram_poller.py` `atomic_write_json` on the symlinked `config.json` (P0).** Both reviewers identify the symlink-severing bug. Codex's framing is sharper: "permanently embeds the API keys inside the repo's working tree — re-introducing the exact failure mode the Mar 15 leak retired." **Same finding, Codex's framing better.** Action: `os.path.realpath` before atomic write, OR refuse to write through symlinks at the `atomic_write_json` boundary.

- **`portfolio/claude_gate.py:662` vs `:777` return-type mismatch (P0/P1).** Claude flagged as P1, Codex flagged as P0 because docstring says 4 but type-hint says 3 — any caller following the type hint with 3-unpack hits `ValueError: too many values to unpack`. **Codex right to escalate.** Action: pick one shape, align signature + all call sites.

- **`portfolio/log_rotation.py:432-440` rotate_text race window (P1).** Both flag the copy2 → truncate window losing concurrent appends. Codex extends with explicit list of affected files (agent.log, loop_out.txt, golddigger_out.txt — all stdout-redirects from batch files). **Both right.** Action: rename-then-open-new pattern, or take sidecar lock during rotation.

- **`portfolio/file_utils.py:240-258` — Win32 `msvcrt.locking(LK_LOCK)` has 10s timeout, not pure-blocking (P1).** Both flag. Codex extends with explicit consumer (claude_gate._log_invocation swallows OSError → invocation record silently dropped → rate-limit warning lossy). **Both right.** Action: LK_NBLCK + retry loop OR document the ceiling.

- **`portfolio/http_retry.py:44-49` Retry-After header ignored (P1).** Both flag. Codex extends with explicit cost: Alpaca returns 429 + Retry-After: 60, our code does 1-4s backoff and burns three retries hammering the API. **Both right.** Action: read `resp.headers.get("Retry-After")` first.

- **`portfolio/http_retry.py:34` non-GET/POST drops json_body (P1).** Both flag. Latent today (no PUT/PATCH/DELETE callers) but signature accepts it. **Both right, easy fix.**

- **`portfolio/feature_normalizer.py:35-40` `_ensure_buffer` race (P1).** Claude says "per subagent". Codex shows exact code, frames it as drops-first-samples-after-fresh-start, biases `_MIN_SAMPLES=20` gate's first crossover. **Codex more concrete.** Action: `_buffers.setdefault(key, deque(maxlen=...))`.

- **`portfolio/llama_server.py:419` Plex-active swap abort race (P1).** Both flag. Codex extends with full failure chain: `_stop_server` already killed model → next signal cycle's `_ensure_model` → flap → llama-server off entirely → `query_llama_server` returns None forever → LLM signals fall back to subprocess. **Both right.** Action: abort decision BEFORE `_stop_server()`.

- **`portfolio/subprocess_utils.py:214-225` PowerShell single-quote backtick escapes inert (P1).** Both flag. Codex extends with concrete attack: pattern with `*` or `[` silently kills broader match. **Both right.** Action: regex-escaped pattern, or pass via `-ArgumentList`.

## Codex found, Claude missed

- **`portfolio/claude_gate.py:597-607` auth-failure detection bypassed on Popen-time exception (P1).** If `_clean_env()` raises or cwd is missing, exception bypasses `_stdout` assignment, lands at `except Exception as e` with status="error" → no auth check runs. **Documented 3-week-outage shape on the upstream-failure path.** Claude flagged exception-too-broad (line 608-610) but missed the auth-detector bypass on this specific path. **Codex stronger.** Action: pre-compute `_stderr=None`, call `detect_auth_failure(str(e))` inside exception handler.

- **`portfolio/llama_server.py:179-180` `_kill_server_by_pid` NameError on unbound `pid` (P1).** If `_PID_FILE` exists but is unreadable, `open()` raises before `pid` is bound; bare `except` catches it; `logger.debug("...", pid)` raises NameError. Escapes → breaks `_stop_server` → prevents subsequent model swap. **Concrete, narrow, real. Claude missed.**

- **`portfolio/api_utils.py:30-35` raw `open()` + `json.load` violates CLAUDE.md atomic-I/O rule.** Mid-rename on config update (which destroys the symlink — same P0 above) crashes startup. **Codex caught the integration with the symlink P0.**

- **`portfolio/gpu_gate.py:202-203` `_THREAD_LOCK.acquire(timeout=...)` uses wall-clock-based remaining time (P2).** NTP backwards jump lengthens wait beyond timeout. Same issue in `_acquire_file_lock` at `llama_server.py:496-497`. **Real P2.**

- **`portfolio/llama_server.py:515` tasklist substring match for PID liveness (P2).** Cross-locale risk. Real P2.

- **`portfolio/subprocess_utils.py:325-335` `kill_orphaned_llama` Windows PID reuse vulnerability (P2).** OpenProcess success on reused PID → orphan llama-completion never reaped. **Real P2.**

- **`portfolio/shared_state.py:278` rate-limiter slot reservation race on thread interrupt (P2).** Minor, self-corrects.

- **`portfolio/health.py:194-196` dogpile on cache miss in `check_agent_silence` (P2).** Acceptable but documented.

- **`portfolio/telegram_notifications.py:39-46` three-layered gates all return True (P2).** "Successful send" indistinguishable from "silenced". **Real P2, Claude missed.**

- **`portfolio/message_store.py:37-49` `_COMMON_MOJIBAKE_REPLACEMENTS` dict has duplicate visible keys.** Several mojibake repairs unreachable. **Codex caught visual ambiguity that source-reading can't disprove.**

- **`portfolio/process_lock.py:65-69` `msvcrt.locking(LK_NBLCK, 1)` + `truncate()` undefined behavior on Windows.** In practice works; failure between truncate and write loses lock byte. **Codex right.**

## Claude found, Codex missed

- **`portfolio/file_utils.py:66-86` `load_json` returns `null` (Python None) on valid-but-null file content.** Callers can't distinguish "file missing" from "file contains null". Codex didn't flag. **Real P2.**

- **`portfolio/log_rotation.py:438-440` `open(..., "w")` truncate fails if Notepad/antivirus holds the file open.** Partial rotation: archive created but source not truncated. Real P2.

- **`portfolio/claude_gate.py:316-326` `record_critical_error` schema vs `check_critical_errors.py` expectations.** Claude flagged the verification need. Codex didn't. P2.

- **`portfolio/prophecy.py` mid-cycle update vs in-memory cache.** Claude flagged the need to verify reload-on-every-read. Codex didn't dive in. P2.

- **`portfolio/telegram_notifications.py` rate-limit queue overflow disk-vs-memory backing.** Claude flagged. Codex didn't address.

- **`portfolio/ministral_signal.py`, `portfolio/qwen3_signal.py`** GPU lock holder dies → stale lock cleanup. Claude flagged. Codex didn't address.

- **`portfolio/process_lock.py` stale lock detection by PID check vs file presence.** Claude flagged. Codex P2 covered a different aspect of process_lock. Both relevant.

- **`portfolio/http_retry.py:39-41` 10% jitter too narrow → retry clustering.** Codex didn't address. Real P2 ("full jitter" pattern).

## Disagreements

**Severity on `claude_gate` return-type mismatch**: Claude P1, Codex P0. Codex's framing wins — any unpacking caller has a `ValueError` on every call. **Use Codex's severity.**

**Both reviewers concur on remaining priorities.**

## What BOTH missed (third pass)

- **`portfolio/file_utils.atomic_write_json` symlink protection.** Both flag it via telegram_poller path. Neither audited every caller of `atomic_write_json` for symlink risk; the `config.json` is the worst case but `data/portfolio_state*.json` could theoretically be symlinked.

- **`portfolio/gpu_gate.py` stale lock cleanup.** Codex covered sweeper-thread daemon issue (P2); Claude flagged it generically. Neither audited what happens if a *different* process loaded an LLM and crashed — the lock file PID is wrong-process's PID, sweeper logic may or may not handle.

- **`portfolio/telegram_poller.py` `/mode` command source authentication.** Codex briefly notes the allow-list on mode_arg, but neither reviewer asked whether the Telegram source chat is verified — anyone with the bot token can `/mode signals` and the bot would write config.

- **`portfolio/log_rotation.py` rotation lock interaction with file_utils sidecar lock alias bug.** Both flag the alias bug. Neither cross-checked whether `log_rotation.rotate_jsonl` uses a path consistent with the live writer's path — if a path mismatch occurs HERE, rotation steals the lock from a writer that isn't blocked.

- **`portfolio/claude_gate.py` invocations.jsonl writer.** Both flag the swallow-OSError → missed invocation. Neither checked whether the rate-limit threshold reads invocations.jsonl directly or from a derived cache; if cache is wrong, rate-limit can either over-restrict or under-restrict.

## Verdict

P0 list after cross: **3 confirmed** (sidecar lock path alias, telegram_poller symlink severing, claude_gate return-type mismatch).
P1 list after cross: **~9 confirmed** (rotate_text race, Win32 lock 10s, Retry-After miss, json_body drop on PUT/DELETE, feature_normalizer race, Plex swap abort, PowerShell escape inert, llama_server pid NameError, claude_gate auth on Popen exception).
P2 list after cross: ~12.

Infrastructure is the **cross-cutting failure-mode amplifier** — bugs here silently corrupt every dependent subsystem. The P0s read like a horror movie of the kind of issue that would re-trigger the 3-week silent-auth outage class.
