# Infrastructure Review

Adversarial read-only review (caveman:cavecrew-reviewer) of the atomic-I/O,
locking, health, notification and dashboard primitives in worktree
`Q:/fa-rev-0531`. **Totals: 0 P0, 1 P1, 1 P2.** The foundational primitives
are in good shape (see "verified clean" below) — this is the strongest
subsystem in the review.

## P1
- `portfolio/health.py:64` — P1: `heartbeat()` updates `last_heartbeat` but NOT
  `cycle_count`/`signals_ok`/`signals_failed`. A keepalive tick publishes a
  fresh heartbeat every 60s even if the loop is hung mid-cycle; the dashboard
  staleness check uses only `last_heartbeat` age (max 300s), so a 59–299s stall
  reads "healthy" while no work is being done. This is the same *process-alive ≠
  working* blind spot that lets silent stalls hide. → distinguish keepalive-only
  ticks from real cycle completion (`last_real_work` timestamp or
  `last_heartbeat_was_keepalive` flag) and alert on its staleness.

## P2
- `portfolio/file_utils.py:26` — P2: `_resolve_write_path()` calls
  `os.path.realpath()` to follow the `config.json` symlink; on Windows a broken
  junction / transient network hiccup during resolve could yield a stale/wrong
  target, then `os.replace()` writes to the wrong location. → on realpath
  failure or non-existent result, log and fall back to the original path; never
  silently write to a potentially-wrong symlink destination.

## Verified clean (adversarial pass found no defect)
- `atomic_write_json/text/jsonl`: write tmp + fsync + `os.replace` on same fs — truly atomic. ✓
- JSONL append sidecar lock: prevents torn lines + rotation races. ✓
- Process locks: non-blocking, released on all paths, stale-lock recovery present. ✓
- `update_health()`: guarded by `_health_lock`, exception cleanup correct. ✓
- GPU gate: stale-lock sweeper + reactive break, handles dead PIDs. ✓
- Dashboard auth: `hmac.compare_digest` constant-time compare; CF JWT verified; cookie refresh OK. ✓
- HTTP retry: 4xx are FATAL (not retried) → no duplicate-order risk from retries on POST. ✓
- Message store: atomic append; Telegram truncation at line boundary. ✓
