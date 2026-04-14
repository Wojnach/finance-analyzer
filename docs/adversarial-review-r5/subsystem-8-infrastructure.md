# Subsystem 8: Infrastructure — Round 5 Findings

## CRITICAL (P1)

**IN-R5-1** — process_lock.py msvcrt.locking locks only 1 byte. Duplicate loop possible.
`process_lock.py:63`. Two loop instances can run concurrently, interleaving writes.
Fix: Lock a larger byte range (1 << 30).

**IN-R5-2** — journal.py CONTEXT_FILE.write_text() non-atomic. Layer 2 reads partial context.
`journal.py:568,580`. Crash during write leaves corrupted context for LLM decisions.
Fix: Use tempfile + os.replace pattern (same as atomic_write_json but for text).

**IN-R5-3** — log_rotation.py uses fixed .tmp name. Concurrent rotations corrupt signal_log.
`log_rotation.py:244`. Two rotation runs write to same .tmp file.
Fix: Use tempfile.mkstemp for unique temp file names.

## HIGH (P2)

**IN-R5-4** — telegram_poller.py raw json.load(open()) on config.json. TOCTOU race.
**IN-R5-5** — message_throttle.py check-and-send race causes duplicate Telegram messages.
**IN-R5-6** — loop_contract.py raw json.load(open()) violating file_utils rules.
**IN-R5-7** — backup.py shutil.copy2 non-atomic. Corrupt backup on crash.

## MEDIUM (P3)

**IN-R5-8** — Dashboard no auth by default + 0.0.0.0 + CORS *. Portfolio data exposed on LAN.
**IN-R5-9** — market_timing.py hour comparison misses NYSE 09:30 open. 30-min early processing.
**IN-R5-10** — atomic_append_jsonl not thread-safe; 8-thread pool can interleave JSONL lines.
