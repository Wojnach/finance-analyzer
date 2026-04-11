# Agent Review: infrastructure — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 24 (file_utils, http_retry, shared_state, logging_config, log_rotation,
subprocess_utils, config_validator, api_utils, tickers, instrument_profile, reporting,
journal, journal_index, telegram_notifications, telegram_poller, message_store,
message_throttle, alert_budget, notification_text, digest, daily_digest, weekly_digest,
prophecy, dashboard/app.py)
**Duration**: ~235s

---

## Findings (7 total: 0 P0, 2 P1, 3 P2, 2 P3)

### P1

**IN-R5-1** journal.py:568,580 — Layer 2 context file written non-atomically
- write_text() truncates then writes — crash mid-write empties trading memory
- Layer 2 reads this file before every decision
- Fix: Use tempfile+os.replace pattern (add atomic_write_text to file_utils)

**IN-R5-2** log_rotation.py:235-242 — rotate_jsonl has no fsync + fixed .tmp suffix
- No fsync before os.replace → power-loss could replace journal with empty file
- Fixed .tmp suffix collides with atomic_write_json temp files
- Fix: Use tempfile.mkstemp + fsync

### P2

**IN-R5-3** shared_state.py:94-96 — _loading_timestamps not cleaned on success/interrupt
- After successful fetch, key removed from _loading_keys but NOT _loading_timestamps
- Stale timestamps accumulate, growing O(n) scan inside hot lock
- Fix: Add _loading_timestamps.pop(key, None) alongside discard

**IN-R5-4** [DANGEROUS] telegram_poller.py:151-159 — Raw open + empty fallback can wipe config
- If config.json is corrupt at read time, cfg = {} is used
- atomic_write_json then writes {"notification": {"mode": "..."}} — wiping ALL API keys
- Combined with IR-8 (symlink break), this is a two-step disaster
- Fix: Refuse to write if loaded config is empty

**IN-R5-5** message_throttle.py:57-58 — TOCTOU race allows duplicate Telegram sends
- No lock between should_send_analysis() check and _send_now()
- Digest thread + Layer 2 thread can both see cooldown elapsed
- Fix: Add threading.Lock around check-then-send

### P3

**IN-R5-6** dashboard/app.py:672 — Timing-vulnerable token comparison
- Uses == instead of hmac.compare_digest()
- LAN attacker could theoretically enumerate token

**IN-R5-7** config_validator.py:58-59 — Raw open() at startup
- PermissionError if antivirus locks config at boot
- Fix: Use file_utils.require_json()

---

## Regression Verification
- A-IN-2 (subprocess tree kill): Confirmed in subprocess_utils.py via Windows Job Objects
- H25 (rotate_all integrated): Confirmed
- H26 (http_retry 429 parsing): Confirmed — parses retry_after from response body
- Log rotation: PARTIAL — uses os.replace but lacks fsync (IN-R5-2)
