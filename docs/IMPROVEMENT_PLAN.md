# Improvement Plan — Auto-Session 2026-04-02

Updated: 2026-04-02
Branch: improve/auto-session-2026-04-02

Previous session (2026-04-01): Batches 5-7 completed (signal tracking, metals JSONL safety, doc consistency).

## 1. Bugs & Problems Found

### P1 — Critical (affects correctness)

#### BUG-165: llama_server.py — Model swap race condition
- **File**: `portfolio/llama_server.py:263-304`
- **Problem**: `query_llama_server()` acquires `_thread_lock` and `_file_lock` for the
  model swap check (lines 274-282), then **releases both locks** before sending the
  HTTP query (line 293). Another thread or process can swap the model between lock
  release and query, causing:
  1. The llama-server process is killed mid-query (model swap kills the old process)
  2. The querying thread gets a `ConnectionError`
  3. The signal engine treats this as a signal failure → votes HOLD
  4. Accuracy tracking records a false failure, degrading the signal's statistics
- **Impact**: Silent signal failures during the first cycle after cache expiry, when
  multiple threads simultaneously query the llama-server. Estimated 1-5% of cycles
  affected. The signal loss biases consensus toward HOLD during high-information moments.
- **Fix**: Hold both locks for the entire operation (model swap + HTTP query). This
  serializes all LLM queries, which is correct since only one model fits in VRAM at
  a time. The 240s timeout is the worst case; typical queries take 5-30s.
- **Risk**: Low — serialization is the intended behavior. The current parallel query
  attempts are a bug, not a feature.

### P2 — Important (code quality, performance)

#### BUG-166: shared_state._cached() — Thundering herd on TTL expiry
- **File**: `portfolio/shared_state.py:28-75`
- **Problem**: When a cached value expires, multiple threads can simultaneously detect
  the cache miss (line 36-37 checks under lock, line 53 calls func outside lock).
  All threads call the underlying function redundantly. For LLM signals, this means
  multiple threads compete to swap models, wasting the model swap time (60-90s per swap).
- **Impact**: Amplifies BUG-165 by increasing the number of concurrent model queries.
  Also wastes CPU/network for non-LLM signals (API calls duplicated across 8 threads).
- **Fix**: Add a per-key "loading" flag under the cache lock. When a thread sees a cache
  miss and no loading flag, it sets the flag and proceeds. Other threads that see the
  loading flag wait (or return stale data). After the function completes, clear the flag
  and store the result.
- **Risk**: Low — dogpile prevention is a standard cache pattern. Stale-while-revalidate
  is already partially implemented (line 73-74 returns stale data on error).

#### BUG-167: CORE_SIGNAL_NAMES vs _CORE_SIGNAL_SET divergence
- **File**: `portfolio/signal_engine.py:53-56, 378`
- **Problem**: `CORE_SIGNAL_NAMES` (frozenset) includes 10 signals:
  {rsi, macd, ema, bb, fear_greed, sentiment, volume, ministral, qwen3, claude_fundamental}.
  `_CORE_SIGNAL_SET` (plain set, line 378) includes those 10 PLUS {ml, funding}.
  They serve different purposes: CORE_SIGNAL_NAMES gates consensus (lines 1283-1285),
  _CORE_SIGNAL_SET is only used in `_compute_applicable_count()` to skip non-core
  signals... except it doesn't — `_compute_applicable_count` iterates `SIGNAL_NAMES`
  and `_CORE_SIGNAL_SET` is never referenced. This is dead code.
- **Impact**: Confusing — a maintainer might modify one thinking it affects both.
- **Fix**: Remove `_CORE_SIGNAL_SET` (dead code) or derive it from `CORE_SIGNAL_NAMES`.

#### REF-33: 18 unused imports in portfolio/ (F401)
- **Files**: agent_invocation.py (2), avanza/scanner.py (1), avanza_control.py (6),
  earnings_calendar.py (1), fish_monitor_smart.py (1), market_health.py (1),
  metals_cross_assets.py (1), microstructure.py (1), seasonality.py (1),
  signal_postmortem.py (1), train_signal_weights.py (2)
- **Fix**: Remove unused imports. Check avanza_control.py imports first — some may be
  intentional re-exports for metals_loop.py.

#### REF-34: 5 SIM105 try/except/pass in llama_server.py
- **File**: `portfolio/llama_server.py:142, 149, 240, 253, 257`
- **Fix**: Convert to `contextlib.suppress(Exception)`.

#### REF-35: 3 F541 f-strings without placeholders
- **Files**: `fin_fish.py:1351, 1372`, `memory_consolidation.py:406`
- **Fix**: Remove `f` prefix.

#### REF-36: 8 unsorted imports in portfolio/ (I001)
- **Fix**: `ruff check --fix --select I001 portfolio/`

#### REF-37: 1 unused variable in portfolio/ (F841)
- **File**: `fish_instrument_finder.py:149` — `updated` assigned but never used
- **Fix**: Remove or prefix with `_`.

### P3 — Minor (documentation, style)

#### REF-38: pyproject.toml description still says "32-signal" — matches reality (32)
- **Status**: Already correct. No change needed.

#### DOC-1: SYSTEM_OVERVIEW.md signal count discrepancy
- **File**: `docs/SYSTEM_OVERVIEW.md:141`
- **Problem**: Says "30 tracked + 2 untracked" but the 2026-04-01 session added all 3
  missing signals to SIGNAL_NAMES. Now 32 tracked + 0 untracked.
- **Fix**: Update overview.

---

## 2. Architecture Improvements

### ARCH-28: llama_server.py — Query-scoped locking (fixes BUG-165)
- **Scope**: Restructure `query_llama_server()` to hold locks during the entire
  model-swap + query operation. Remove the early lock release pattern.
- **Impact**: Fixes the race condition. Makes LLM queries correctly serialized.
- **Risk**: Low — serialization matches hardware constraint (single GPU).

### ARCH-29: shared_state._cached() — Dogpile prevention (fixes BUG-166)
- **Scope**: Add per-key loading flag to prevent thundering herd on cache expiry.
  Threads that find the key "loading" return stale data (if available) or wait briefly.
- **Impact**: Reduces redundant API calls and model swaps. Prevents amplification of
  BUG-165.
- **Risk**: Low — standard cache pattern. Must handle edge case where loading thread
  crashes (timeout on the loading flag).

---

## 3. Implementation Batches

### Batch 8: Ruff auto-fixes (portfolio/) — REF-33, REF-35, REF-36, REF-37
**Scope**: F401 (unused imports), F541 (f-string placeholders), I001 (import sort), F841 (unused var)
**Files**: ~15 files
**Test**: Run full test suite to verify no regressions
**Risk**: Zero for auto-fixes; manual review for avanza_control.py re-exports

### Batch 9: SIM105 conversions — REF-34
**Scope**: 5 try/except/pass → contextlib.suppress in llama_server.py
**Files**: 1 file
**Test**: Run llama_server tests (if any) + ruff check
**Risk**: Zero — behavioral equivalence

### Batch 10: llama_server race condition fix — BUG-165 + ARCH-28
**Scope**: Restructure query_llama_server() to hold locks during query
**Files**: portfolio/llama_server.py
**Test**: Write unit tests for the race condition scenario, run existing tests
**Risk**: Low — serialization matches hardware constraint

### Batch 11: Cache dogpile prevention — BUG-166 + ARCH-29
**Scope**: Add per-key loading flag to _cached()
**Files**: portfolio/shared_state.py
**Test**: Write unit tests for thundering herd scenario, run existing tests
**Risk**: Low — standard cache pattern, but must handle loading thread crash

### Batch 12: Dead code removal + doc updates — BUG-167, DOC-1
**Scope**: Remove _CORE_SIGNAL_SET, update SYSTEM_OVERVIEW.md
**Files**: portfolio/signal_engine.py, docs/SYSTEM_OVERVIEW.md
**Test**: Run signal_engine tests
**Risk**: Zero

---

## 4. Deferred Items (from prior sessions)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk)
- **ARCH-18**: metals_loop.py 4553-line monolith (risks live trading)
- **ARCH-19**: No CI/CD pipeline (needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption)
- **ARCH-21**: autonomous.py function decomposition (stable, low ROI)
- **ARCH-22**: agent_invocation.py class extraction (touches every caller)
- **ARCH-29-old**: Avanza package migration (needs manual staged rollout)
- **BUG-121**: news_event.py sector mapping hardcoded (low value)
- **BUG-132**: orb_predictor.py no caching (low priority)
- **BUG-149**: meta_learner orphaned — predict() never called
- **BUG-162**: metals_loop.py 4553-line monolith
- **BUG-164**: orb_predictor.py hardcodes UTC morning hours
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config)
- **FEAT-3**: Integrate meta_learner as signal #31

---

## 5. Dependency & Ordering

```
Batch 8 (ruff auto-fixes) → no dependencies, do first
Batch 9 (SIM105) → after Batch 8 (imports may shift)
Batch 10 (llama_server race fix) → independent of Batch 8/9
Batch 11 (cache dogpile) → independent, but enhances Batch 10's fix
Batch 12 (dead code + docs) → after Batch 10/11 (reflects final state)

Run full test suite after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 8 | ~15 (modify) | Zero — auto-fix | Zero |
| 9 | 1 (modify) | Zero — behavioral equiv | Zero |
| 10 | 1 (modify) | Low — serialization | Low — new tests needed |
| 11 | 1 (modify) | Low — cache pattern | Low — new tests needed |
| 12 | 2 (modify) | Zero — dead code + docs | Zero |
