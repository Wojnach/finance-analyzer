# Infrastructure Subsystem Adversarial Review

Date: 2026-05-28. Reviewer: caveman:cavecrew-reviewer. Annotations `[ORCH: ...]` added by
the orchestrating agent after independent verification (some raw severities recalibrated).

Raw counts: 6 P0 | 4 P1 | 2 P2. **Recalibrated: 0–3 P0** (2 P0s disproven/downgraded).

## P0 (raw)

- `portfolio/file_utils.py:240`: data-corruption — sidecar lock creation race. Two writers
  both see lock_path missing, both `open(...,"ab")`; claim: lock file ends 1 byte, future
  callers see `tell()==0` False and lock a stale file.
  **[ORCH: FALSE POSITIVE — DROP.** Outcome is idempotent: any ≥1-byte lock file is lockable;
  the byte's content never matters, only that a lockable byte exists (the docstring states
  this is the whole point of the pre-seeded sidecar). 1-byte or 2-byte both work. No corruption.]

- `portfolio/process_lock.py:103`: data-corruption — lock metadata write not atomic.
  `fh.truncate()` then `fh.write()`; crash between leaves an empty lock file → next acquirer
  reads `{}`, loses PID, can't detect stale locks (BUG-182 needs PID). Fix: tempfile + os.replace.
  **[ORCH: plausible P1 — crash-window edge; verify before fixing.]**

- `portfolio/log_rotation.py:349`: data-loss — SQLite `-wal`/`-shm` not cleaned post-rotation
  (signal_log dual-writes signal_log.db). Concurrent writer holding the lock may corrupt the
  reopened DB. Fix: `PRAGMA wal_checkpoint(TRUNCATE)` or remove `-wal`/`-shm` before rotation.
  **[ORCH: plausible P1; `signal_log.db-wal`/`-shm` are present in the tree right now.]**

- `portfolio/log_rotation.py:510`: race — text-file rotation (`rotate_text`) shifts
  current→.1→.2 with no sidecar lock; a writer appending mid-rotation loses data. Fix: wrap in
  `jsonl_sidecar_lock` (adapt for the `.txt` pattern).
  **[ORCH: plausible P1.]**

- `portfolio/health.py:156`: silent-error — staleness check on naive ISO `last_heartbeat`
  yields a naive datetime; comparison with `datetime.now(UTC)` "fails silently / falsely
  reports not stale". Fix: if `dt.tzinfo is None`, set UTC before comparison.
  **[ORCH: OVERSTATED → P2.** The subtraction at `health.py:165` is NOT inside the try/except
  (which wraps only `fromisoformat`), so a naive ts RAISES `TypeError` to the caller — a crash,
  not a silent "not stale". And current `heartbeat()` writes tz-aware ISO, so it's latent.
  Still worth hardening: normalize tz / wrap the subtraction.]

- `portfolio/subprocess_utils.py:214`: PowerShell injection — `kill_orphaned_by_cmdline`
  escapes the pattern with backticks but passes it unquoted into a `Where-Object` filter;
  pattern with `$ ( ) {` executes. Fix: single-quote the filter.
  **[ORCH: low exploitability — pattern is an own-process cmdline, not external input → P2.
  Still quote it.]**

## P1 (raw)

- `portfolio/health.py:183`: silent-lag — `last_invocation_ts` cached in health_state can lag
  invocations.jsonl; `check_agent_silence` reads the stale cache → false "silent". Fallback
  re-parses but never invalidates the cache. Fix: invalidate on detected drift.
- `portfolio/file_utils.py:269`: liveness — concurrent `atomic_append_jsonl` serializes on the
  sidecar lock (~10ms/append w/ fsync); under the 8-worker pool queue depth can grow unbounded.
  No backpressure. Fix: lock-wait timeout + WARN if contended >50ms.
- `portfolio/log_rotation.py:549`: race — `.1`/`.2` rename can fail silently on Windows if a
  reader holds the file open (`os.replace` on open handle); exception swallowed, logs not
  preserved. Fix: explicit handler + retry + escalate CRITICAL.
- `portfolio/gpu_gate.py:66`: stale-lock window — `_is_stale()` uses >300s mtime; sweeper runs
  every 30s. A process dying between ticks makes the next acquirer wait ~300s → ~5 missed 60s
  cycles. Fix: verify reactive `_try_break_stale_lock()` is called synchronously in `acquire_lock_file()`.
- `portfolio/journal.py:28`: inefficiency — `load_recent` reads+parses the whole JOURNAL_FILE.
  Fix: `load_jsonl_tail(path, max_entries=...)`. **[ORCH: P2.]**

## P2

- `portfolio/file_utils.py:74`: `load_json` logs corruption but returns default like a missing
  file; caller can't distinguish. For critical files use `require_json`. **[ORCH: this is the
  Theme-1 "empty-as-valid" risk; also OSError→default masks a locked file as empty (IND-3).]**

## Verified-correct (no finding)

- `dashboard/auth.py` + `dashboard/cf_access.py`: NO auth bypass. CF-Access JWT verified
  (RS256 signature vs CF JWKs, aud/exp/iat, email-claim↔header match); cookie/query/bearer use
  `hmac.compare_digest`; fail-closed when config missing. The 2026-05-13 header-spoof P0 is fixed.
  **[ORCH: confirmed by direct read. Only minor: unset `dashboard_token` opens all endpoints
  incl. POST (IND-4, P3).]**
