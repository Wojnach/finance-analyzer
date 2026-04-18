# Adversarial Code Review: Signals-Core Subsystem (Claude Reviewer)

## Executive Summary

**5 P1 findings, 8 P2 findings, 5 P3 findings.**

Most critical: per-ticker accuracy inflated (no neutral filtering in `ticker_accuracy.py`/`signal_db.py`), signal history race condition under ThreadPoolExecutor, non-atomic snapshot reads, and MWU weight system is dead code.

---

## P1 Findings

### P1-1: `ticker_accuracy.py` and `signal_db.py` skip neutral-outcome filtering, inflating accuracy
**File:** `portfolio/ticker_accuracy.py:60-62`, `portfolio/signal_db.py:270-272`
Counts 0.01% moves as "correct" for BUY. Mode B probability notifications use these inflated numbers for real trading decisions.

### P1-2: `signal_history.update_history()` race condition with no locking
**File:** `portfolio/signal_history.py:53-82`
8 ThreadPoolExecutor workers can load-modify-write simultaneously. Second write clobbers first.

### P1-3: `SignalDB._get_conn()` creates non-thread-safe SQLite connection
**File:** `portfolio/signal_db.py:31-37`
No `check_same_thread=False`. Currently safe by accident (instance-per-call).

### P1-4: `_load_accuracy_snapshots()` uses raw `read_text()` instead of atomic I/O
**File:** `portfolio/accuracy_stats.py:1162-1175`
Can read empty file during `os.replace()` window. Disables degradation detection.

### P1-5: MWU SignalWeightManager is dead code
**File:** `portfolio/signal_weights.py:1-121`
Weights computed and written to disk but never read by signal_engine.py.

## P2 Findings (8)

P2-1: Silent signal registry overwrite -- `signal_registry.py:42-46`
P2-2: `_adx_cache` key collision possible -- `signal_engine.py:1847`
P2-3: `backfill_outcomes()` raw file read races with appends -- `outcome_tracker.py:300-313`
P2-4: `blend_accuracy_data` directional stats use max-sample-wins (ignores regime) -- `accuracy_stats.py:818-833`
P2-5: Regime accuracy cache shares timestamp across horizons (BUG-133 not applied) -- `accuracy_stats.py:1115-1116`
P2-6: `get_focus_probabilities` triggers 12 full signal log scans -- `ticker_accuracy.py:283-287`
P2-7: `train_signal_weights.py` loads raw JSONL bypassing SQLite -- `train_signal_weights.py:54`
P2-8: MWU eta handling incomplete -- `signal_weights.py:119-121`

## P3 Findings (5)

P3-1: Inconsistent accuracy computation across 4 modules
P3-2: Signal history rewrites full JSONL on every update -- `signal_history.py:31-37`
P3-3: WalkForwardResult tuple/list mismatch on JSON load -- `signal_weight_optimizer.py:170`
P3-4: `_compute_applicable_count` has stale ministral crypto-only guard -- `signal_engine.py:829-831`
P3-5: BUG-133 timestamp fix not applied to regime/ticker caches
