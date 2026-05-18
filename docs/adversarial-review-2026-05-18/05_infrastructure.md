# Infrastructure Review — subagent result (caveman:cavecrew-reviewer)

Totals: 0 explicit P1, 24 P2 findings. (Reviewer marked all RISK; several would escalate to P1 after own review.)

## P1 (escalated from RISK by own review)

1. **shared_state.py:88-89** — `_loading_keys.add()` outside try; if `enqueue_fn` raises before try entry, key marked loading forever → deadlock on next cache miss for same key.
2. **health.py:23-41** — `update_health` read-modify-write without lock; concurrent heartbeat overwrites last update.
3. **journal.py:28** — `load_recent` no file lock; concurrent `atomic_append_jsonl` writes can present partially-flushed state on Windows.
4. **digest.py:47-52** — TOCTOU on `_set_last_digest_time` (load→modify→write); two digest workers both write stale timestamp.
5. **gpu_gate.py:216** — Lock file write bare `os.write` (no fsync); crash leaves incomplete marker → `_pid_alive` parses empty dict.
6. **regime_alerts.py:90** — `log_regime_change` appends without sidecar lock vs concurrent `_get_last_regime` scan → torn lines.
7. **message_throttle.py:44-66** — `queue_analysis` no lock; concurrent calls both pass should_send check + both _send_now.
8. **prophecy.py:72** — In-place metadata mutation before atomic_write_json; if write fails, in-memory state has timestamp file does not.

## P2 / 🟡

- file_utils.py:32 & 53 — tempfile.mkstemp dir=path.parent — fine on local fs but Windows network shares may still fail os.replace.
- shared_state.py:54-66 — Cache eviction during iteration; safe in 3.7+ but no len() recheck after first batch.
- shared_state.py:277 — Rate limiter off-by-one on `elapsed==0` back-to-back calls.
- shared_state.py:334 — `_newsapi_daily_reset` < today_start; init 0.0 forces first call always resets; clock forward jump double-resets.
- health.py:156 — `check_staleness` `fromisoformat` on corrupted ISO → ValueError after defensive try returned.
- journal.py:23-40 — Full file scan on 68MB+ JOURNAL_FILE per Layer 2 invocation; should use `load_jsonl_tail`.
- telegram_notifications.py:49 — Truncation at 4096 chars hardcoded; no test.
- telegram_notifications.py:130 — `load_json(BOLD_STATE_FILE)` no null guard; corrupt → AttributeError in portfolio_value.
- digest.py:140-147 — Accesses summary signals without null guard.
- gpu_gate.py:234 — `_try_break_stale_lock` spams logs every 30s if psutil missing.
- http_retry.py:40-42 — Jitter multiplied AFTER backoff_factor → unpredictable backoff.
- logging_config.py:27 — Root level changes don't propagate to already-configured children.
- telegram_poller.py:336 — `_newsapi_daily_reset` read outside lock; TOCTOU.
- telegram_poller.py:221 — Clock-backward jump revives old messages as "pending".
- journal.py:405-420 — `load_warrant_state` exception silently swallowed in build_context; incomplete memory.
- daily_digest.py:74-84 — Hour-only check; DST transition at digest hour → double-send window.
- shared_state.py:50-66 — Stale `_loading_keys` accumulate after batch failure (orphaned forever).
- health.py:283-296 — In-place `recent_results` mutation; torn list on concurrent read.
- shared_state.py:173-179 — Stale-while-revalidate can't distinguish "cache miss" from "explicit None".
- file_utils.py:115 — `load_jsonl` maxlen deque copied to list wastes memory.

## Own finding (not in subagent report)

- **file_utils.py:371-415 prune_jsonl** — Does NOT acquire `jsonl_sidecar_lock`. Concurrent appends during read→rewrite window get silently lost. Same class of bug that 2026-05-11 signal_log_reconciliation contract was supposed to detect. **P1**.
