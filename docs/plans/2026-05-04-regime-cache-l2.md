# PLAN — regime_accuracy_cache L1+L2 cache

**Date:** 2026-05-04
**Branch:** `feat/regime-cache-l2-2026-05-04`
**Worktree:** `.worktrees/regime-cache-l2-2026-05-04`

## Problem

Main loop cycles run ~595s (target 60s). `last_heartbeat` ages past the 300s
stale threshold on every cycle, flipping `/api/health` to "stale" mid-cycle.

Root cause from `data/portfolio.log`:

```
[SLOW-PHASE] MSTR/utility_overlay: 28.4s
[SLOW-PHASE] XAU-USD/utility_overlay: ~30s
[SLOW-PHASE] XAG-USD/utility_overlay: 35.9s
[SLOW-PHASE] ETH-USD/utility_overlay: 33.0s
[SLOW-PHASE] BTC-USD/utility_overlay: 28.8s
```

The `utility_overlay` named phase covers TWO independent caches:

1. `signal_utility_cache` — has L1 (in-memory) + L2 (disk) write-through. ✅ Done 2026-05-03.
2. `regime_accuracy_cache` — has L2 (disk) only. **No L1.** Every ticker call
   pays a JSON parse, and on any horizon TTL miss all 5 ticker threads
   cold-compute `signal_accuracy_by_regime` in parallel (50K-entry walks).

Hot path in `portfolio/signal_engine.py:3415-3419`:

```python
regime_acc = load_cached_regime_accuracy(acc_horizon)  # disk read every call
if not regime_acc:
    regime_acc = signal_accuracy_by_regime(acc_horizon)  # 30s walk
    if regime_acc:
        write_regime_accuracy_cache(acc_horizon, regime_acc)
```

## Fix

Mirror the `signal_utility` L1+L2 pattern exactly:

1. New module-level state in `portfolio/accuracy_stats.py`:
   - `_regime_accuracy_cache: dict[str, tuple[float, dict]]` — L1 keyed by horizon
   - `_regime_accuracy_cache_lock: threading.Lock` — guards L1 swap only
   - `_REGIME_ACCURACY_CACHE_TTL = 300` — same as `_SIGNAL_UTILITY_CACHE_TTL`

2. New wrapper `get_or_compute_regime_accuracy(horizon)`:
   - L1 hit → return immediately
   - L2 hit → populate L1 → return
   - L1+L2 miss → compute via existing `signal_accuracy_by_regime` outside lock
   - On compute success → populate L1 + write L2

3. Update call site in `portfolio/signal_engine.py:3415-3419` to use the wrapper.
   Remove the manual L2-only dance.

4. Existing `load_cached_regime_accuracy` + `write_regime_accuracy_cache` stay
   unchanged so `signal_postmortem.py` and external callers keep working.

5. The L2 file already merges horizons (existing `write_regime_accuracy_cache`
   uses load-modify-write under `_accuracy_write_lock`), so no new disk lock
   needed — it's reused as-is.

## Tests

`tests/test_regime_accuracy_cache.py` — new file:

- `test_l1_hit_returns_cached_dict_without_disk_read` — patch the disk loader
  to raise after first call; second call must not raise (L1 served).
- `test_l2_hit_populates_l1` — preload disk with fresh payload, first call
  reads disk + populates L1, second call hits L1.
- `test_l1_l2_miss_computes_writes_both` — clear caches, compute happens,
  both L1 and L2 are populated.
- `test_cross_horizon_writes_merge` — write 1d, then 3d, both readable from disk.
- `test_ttl_expiry_l1` — frozen time, expire L1, falls back to L2.
- `test_invalidate_clears_l1` — `invalidate_regime_accuracy_cache` removes L1
  entries so post-backfill cycles don't serve stale data.
- `test_compute_failure_returns_empty_dict` — wraps the compute in try/except
  so cache miss never crashes the pipeline.

## Risk + rollback

- `signal_postmortem.py:216` still uses `load_cached_regime_accuracy("1d")`
  directly. Unchanged behavior.
- The L1 cache can stale during a 5-min window after `outcome_tracker` runs.
  Same window as `signal_utility` — acceptable given outcomes update daily.
- Rollback: revert the commit; the existing L2-only path still works.

## Verification

After merge + restart:

```
schtasks /run /tn PF-DataLoop
# wait for cycle 2 (avoid first-cold-cycle artifacts)
# expect [SLOW-PHASE] *_utility_overlay times to drop from 28-36s to <5s
# expect last_heartbeat in /api/health to update every ~60-90s, not every 600s
```
