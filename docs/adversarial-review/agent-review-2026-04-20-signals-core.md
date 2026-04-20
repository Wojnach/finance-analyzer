# Agent Review: signals-core (2026-04-20)

## P1 Critical
1. **Neutral-outcome filtering MISSING in signal_db.py and ticker_accuracy.py** — +0.001% counts as correct BUY. Drives Mode B notifications with inflated accuracy.
2. **signal_history.py load-modify-write race** — No lock. Mitigated: offline-only callers now.
3. **SignalWeightManager (MWU) is pure dead code** — `signal_weights.py` entire file. Never imported.
4. **outcome_tracker.py signal_log.jsonl full-file rewrite race** — Backfill reads entire file, modifies, replaces. Concurrent appends lost.

## P2 High
1. SignalDB methods don't filter neutrals (ticker_accuracy drives real recommendations)
2. `_compute_dynamic_correlation_groups()` O(n²) blocks ticker thread on cold cache
3. `ticker_accuracy.accuracy_by_ticker_signal()` loads ALL entries per call (no caching)
4. ADX cache key collision theoretically possible (id(df) reuse)
5. `_load_accuracy_snapshots()` reads unbounded JSONL into memory
6. Utility boost compresses top-end signals to same 0.95 cap
7. `blend_accuracy_data` directional key uses winner-take-all (not blending)
8. IC zero-penalty can suppress good signals with volatile IC due to low stability threshold

## P3 Medium
1. signal_registry.py mutates entry dict (thread-safe under GIL but semantically racy)
2. signal_weight_optimizer.py entirely unused
3. Dynamic correlation TTL (2h) mismatched with signal utility cache (5m)
4. SignalDB._get_conn() not thread-safe (mitigated by per-call instantiation)
5. accuracy_degradation.py double-loads alert state
6. No upper bound on accuracy_data dict size in consensus

## Prior Finding Status
- Per-ticker accuracy inflated: **PARTIAL FIX** (accuracy_stats fixed, signal_db/ticker_accuracy not)
- Signal history race: **MITIGATED** (offline only)
- signal_log.jsonl rewrite race: **OPEN**
