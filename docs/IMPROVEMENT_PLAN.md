# Improvement Plan — Auto-Session 2026-04-04

Updated: 2026-04-04
Branch: improve/auto-session-2026-04-04

Previous session (2026-04-02): Batches 10-12 completed (llama_server race fix, cache dogpile, dead code).
Previous session (2026-04-01): Batches 5-7 completed (signal tracking, metals JSONL safety, doc consistency).

## Status of Previous Batches

| Batch | Target | Status |
|-------|--------|--------|
| 8 | Ruff auto-fixes (F401/F541/F841/I001) | **Partial** — F401/F541/F841 done, 2 I001 remain |
| 9 | SIM105 conversions (llama_server) | **Done** — only 1 SIM105 remains (main.py:84) |
| 10 | llama_server race fix (BUG-165) | **Done** — query-scoped locking implemented |
| 11 | Cache dogpile (BUG-166) | **Done** — `_loading_keys` pattern in shared_state |
| 12 | Dead code + docs (BUG-167) | **Done** — `_CORE_SIGNAL_SET` removed |

---

## 1. Bugs & Problems Found

### P2 — Important

#### BUG-168: llama_server.py — `_ensure_model()` missing global declaration
- **File**: `portfolio/llama_server.py:265-273`
- **Problem**: `_ensure_model()` assigns `_local_model = name` on line 270 without a
  `global _local_model` declaration. This creates a local variable that is immediately
  discarded. The module-level `_local_model` is never updated in the cross-process case
  (when another process started the correct model).
- **Impact**: Low — `_local_model` is only used by `_stop_server()` for cleanup, and
  if the server was started by another process, `_local_proc` would be None anyway.
  However, it's semantically wrong and could confuse maintainers.
- **Fix**: Add `global _local_model` or remove the dead assignment.
- **Risk**: Zero.

#### BUG-169: indicators.py — `_regime_cache` not thread-safe
- **File**: `portfolio/indicators.py:124-168`, `portfolio/shared_state.py:151-152`
- **Problem**: `_regime_cache` and `_regime_cache_cycle` are plain dict/int in
  `shared_state.py`, accessed without locks from 8 concurrent threads via
  `indicators.detect_regime()`. The check-then-clear pattern at lines 124-126 is racy:
  Thread A clears cache, Thread B computes and stores result, Thread C clears cache again.
- **Impact**: P3 in practice — worst case is redundant computation (same regime computed
  twice for the same ticker). Cannot produce wrong results. But it's a correctness gap
  that would become P1 if the cache stored mutable values or had side effects.
- **Fix**: Add a `_regime_lock = threading.Lock()` in shared_state.py. Wrap the
  check-and-clear and the read-or-compute patterns.
- **Risk**: Zero — adding a lock to a hot path adds ~1μs per call, negligible vs. the
  computation cost.

#### BUG-170: fear_greed.py — Non-atomic streak file write
- **File**: `portfolio/fear_greed.py:69`
- **Problem**: `_STREAK_FILE.write_text(json.dumps(data, indent=2))` is non-atomic.
  A crash during write could corrupt the streak file.
- **Impact**: P3 — streak data is non-critical. Loss would only affect the streak counter
  (not trading decisions).
- **Fix**: Replace with `atomic_write_json(_STREAK_FILE, data)`.
- **Risk**: Zero.

#### BUG-171: llm_batch.py — Ministral/Qwen3 parse result asymmetry
- **File**: `portfolio/llm_batch.py:89-115`
- **Problem**: Ministral parse wraps in `{"original": {...}, "custom": None}` while
  Qwen3 returns flat `{"action": ..., "reasoning": ..., "model": ...}`. These results
  are stored in the shared cache via `_update_cache()`. If callers expect a consistent
  format, one path will break.
- **Impact**: Depends on callers — needs verification. If both callers handle their own
  format, this is P3 cosmetic. If not, P2 bug.
- **Fix**: Verify caller expectations in `signal_engine.py` for both ministral and qwen3
  cache reads. Normalize if needed.
- **Risk**: Low — need to trace caller paths first.

### P3 — Minor (lint, style)

#### REF-39: 7 auto-fixable ruff issues
- 2 I001 (unsorted imports): `llm_batch.py:108`, `main.py:513`
- 1 UP015 (redundant open mode): `llama_server.py:124`
- 1 UP017 (datetime.UTC alias): `signals/news_event.py:92`
- 3 SIM114 (same-arms if): `indicators.py:163`, `signals/crypto_macro.py:150,154`

#### REF-40: 2 B007 unused loop variables in llm_batch.py
- **File**: `portfolio/llm_batch.py:47,57`
- **Problem**: `cache_key` (line 47) and `ctx` (line 57) are not used within their
  respective loop bodies.
- **Fix**: Prefix with `_` to indicate intentional disuse.

#### REF-41: 6 E741 ambiguous variable name `l`
- **Files**: `log_rotation.py:214,218,222`, `signals/heikin_ashi.py:79`,
  `signals/mean_reversion.py:106,294`
- **Fix**: Rename `l` to descriptive names (`line`, `low`, etc.).

#### REF-42: 1 SIM101 duplicate isinstance
- **File**: `signals/calendar_seasonal.py:381`
- **Fix**: Merge into single `isinstance(last_time, (datetime, str))`.

#### REF-43: 2 E731 lambda assignments
- **Files**: `avanza_control.py:270`, `avanza_session.py:304`
- **Fix**: Convert to def functions.

#### REF-44: 1 SIM105 remaining contextlib.suppress
- **File**: `main.py:84`
- **Fix**: Convert `try/except Exception: pass` to `contextlib.suppress(Exception)`.

---

## 2. Architecture Improvements

### ARCH-30: Regime cache thread safety (fixes BUG-169)
- **Scope**: Add `_regime_lock` to protect `_regime_cache` and `_regime_cache_cycle`
  in `shared_state.py`. Wrap access in `indicators.detect_regime()`.
- **Impact**: Eliminates last known thread-safety gap in the parallel ticker processing
  path. Completes the thread-safety story started by BUG-85/86.
- **Risk**: Zero — lock overhead is negligible.

---

## 3. Implementation Batches

### Batch 13: Ruff auto-fixes — REF-39
**Scope**: Run `ruff check --fix` for I001, UP015, UP017. Manual fix for SIM114 (unsafe-fix).
**Files**: 5 files
**Test**: `ruff check portfolio/` clean for targeted rules
**Risk**: Zero

### Batch 14: Bug fixes — BUG-168, BUG-170, REF-40, REF-44
**Scope**: Fix missing global in llama_server, atomic write in fear_greed, unused loop vars, contextlib.suppress
**Files**: 4 files (llama_server.py, fear_greed.py, llm_batch.py, main.py)
**Test**: Run relevant test files
**Risk**: Zero

### Batch 15: Regime cache thread safety — BUG-169 + ARCH-30
**Scope**: Add `_regime_lock` and wrap detect_regime() access patterns
**Files**: 2 files (shared_state.py, indicators.py)
**Test**: Write thread-safety tests, run existing indicator tests
**Risk**: Zero

### Batch 16: Variable naming + lint cleanup — REF-41, REF-42, REF-43
**Scope**: Rename ambiguous `l` vars, merge isinstance, convert lambdas
**Files**: 6 files
**Test**: Run ruff check, run affected test files
**Risk**: Zero

### Batch 17: LLM batch parse verification — BUG-171
**Scope**: Trace Ministral/Qwen3 cache read paths in signal_engine.py. Normalize if inconsistent.
**Files**: 1-3 files depending on findings
**Test**: Run signal_engine tests
**Risk**: Low — depends on findings

### Batch 18: Documentation update — SYSTEM_OVERVIEW.md, CHANGELOG.md
**Scope**: Update docs to reflect all changes from this session
**Files**: 2-3 doc files
**Risk**: Zero

---

## 4. Deferred Items (carried forward)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk)
- **ARCH-18/BUG-162**: metals_loop.py 5174-line monolith (risks live trading)
- **ARCH-19**: No CI/CD pipeline (needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption)
- **ARCH-21**: autonomous.py function decomposition (stable, low ROI)
- **ARCH-22**: agent_invocation.py class extraction (touches every caller)
- **BUG-121**: news_event.py sector mapping hardcoded (low value)
- **BUG-132**: orb_predictor.py no caching (low priority)
- **BUG-149**: meta_learner orphaned — predict() never called
- **BUG-164**: orb_predictor.py hardcodes UTC morning hours
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config)
- **FEAT-3**: Integrate meta_learner as signal #31
- **E402**: 42 module-import-not-at-top-of-file (intentional delayed imports)
- **SIM102**: 9 collapsible-if (subjective, most are readable as-is)
- **SIM115**: 5 open-file-with-context-handler (some intentional)
- ~10 silent `except Exception: pass` in non-critical paths (fishing, avanza scanner)

---

## 5. Dependency & Ordering

```
Batch 13 (ruff auto-fixes) → no dependencies, do first
Batch 14 (bug fixes) → after Batch 13 (line numbers may shift)
Batch 15 (regime lock) → independent of 13/14
Batch 16 (variable naming) → after 13 (line numbers may shift)
Batch 17 (LLM batch verify) → after 14 (depends on llm_batch.py state)
Batch 18 (docs) → last (reflects final state)

Run test suite after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 13 | 5 (modify) | Zero — auto-fix | Zero |
| 14 | 4 (modify) | Zero — trivial fixes | Zero |
| 15 | 2 (modify) | Zero — adding lock | Low — new tests |
| 16 | 6 (modify) | Zero — rename/style | Zero |
| 17 | 1-3 (modify) | Low — depends on findings | Low |
| 18 | 2-3 (docs) | Zero | Zero |
