# Improvement Plan — Auto-Session 2026-03-29

Updated: 2026-03-29
Branch: improve/auto-session-2026-03-29

## 1. Bugs & Problems Found

### P1 — Critical (affects accuracy or causes incorrect behavior)

#### BUG-143: Unanimity penalty uses pre-gated vote counts
- **File**: `portfolio/signal_engine.py:1095-1096, 650-664`
- **Problem**: `buy`/`sell` counts are computed from raw `votes` dict BEFORE `_weighted_consensus()` applies regime gating (line 328: `votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}`). These raw counts are stored in `extra_info["_buy_count"]` / `extra_info["_sell_count"]` and used by the unanimity penalty in Stage 5 of `apply_confidence_penalties()`.
- **Impact**: When signals are regime-gated (e.g., `trend` and `momentum_factors` in ranging regime), the unanimity ratio is computed including those gated signals. Example: raw votes 9 BUY / 1 SELL gives ratio 0.9 → 0.6x penalty. But after gating 2 signals, actual ratio is 7/8 = 0.875 → should be 0.75x penalty. The penalty is **33% too aggressive** in this case.
- **Fix**: Compute `buy`/`sell` counts AFTER regime gating, or pass the gated votes dict from `_weighted_consensus` back to the caller.

#### BUG-144: Forecast regime discount is dead code
- **File**: `portfolio/signals/forecast.py:864`, `portfolio/signal_engine.py:1030`
- **Problem**: `compute_forecast_signal()` reads `context.get("regime", "")` to apply regime-based confidence discount. But `generate_signal()` builds `context_data = {"ticker": ticker, "config": config or {}, "macro": macro_data}` (line 1030) — the `regime` key is never included. The forecast signal ALWAYS gets `regime=""` → `_REGIME_NEUTRAL` (1.0x multiplier).
- **Impact**: Chronos has a documented mean-reversion bias. In trending markets, its predictions should be discounted (0.8x) per `_REGIME_DISCOUNT_TRENDING`. This discount is never applied. Chronos predictions in trending markets get the same confidence weight as in ranging markets.
- **Fix**: Add `"regime": regime` to `context_data` dict at line 1030. Regime is already computed at line 683.

### P2 — Important (could cause incorrect behavior in edge cases)

#### BUG-145: meta_learner SQLite connection leak on exception
- **File**: `portfolio/meta_learner.py:97-109`
- **Problem**: `_load_data()` opens `conn = sqlite3.connect(...)` and closes it on line 109. But if `pd.read_sql_query()` throws (corrupt DB, query timeout, interrupted), `conn.close()` is never called.
- **Impact**: SQLite connection stays open until garbage collection. During weekly retraining, a DB error could leave WAL-mode connections dangling, preventing other processes from writing.
- **Fix**: Use try/finally: `try: ... finally: conn.close()`. Or use context manager.

#### BUG-146: meta_learner uses old datetime import style
- **File**: `portfolio/meta_learner.py:17`
- **Problem**: `from datetime import datetime, timezone` instead of `from datetime import UTC, datetime` (Python 3.11+). The rest of the codebase was modernized in REF-16.
- **Impact**: Style inconsistency. Functional equivalent (`timezone.utc` vs `UTC`), but ruff will flag it.
- **Fix**: `from datetime import UTC, datetime`, replace `timezone.utc` with `UTC`.

#### BUG-147: meta_learner duplicates SIGNAL_NAMES list
- **File**: `portfolio/meta_learner.py:34-41`
- **Problem**: Maintains its own copy of SIGNAL_NAMES instead of importing from `portfolio.tickers`. If signals are added/removed from `tickers.SIGNAL_NAMES`, the meta_learner's feature columns silently drift. The model would then be trained with wrong features.
- **Impact**: Medium — both lists are currently identical (30 signals). But any future signal change that updates `tickers.py` without updating `meta_learner.py` produces wrong features.
- **Fix**: `from portfolio.tickers import SIGNAL_NAMES`.

#### BUG-148: meta_learner.predict() loads model from disk on every call
- **File**: `portfolio/meta_learner.py:352`
- **Problem**: `model = joblib.load(model_path)` deserializes the model file on every prediction. If this module were ever integrated into the signal loop (20 tickers × 4 horizons = 80 calls per 60s cycle), it would read ~600KB × 80 = ~48MB from disk per cycle.
- **Impact**: Currently zero — predict() is never called from production code. But this is a blocker for integration.
- **Fix**: Add module-level model cache: `_model_cache: dict[str, Any] = {}`, load once per horizon, check mtime for staleness.

### P3 — Minor (code quality, observability)

#### BUG-149: meta_learner is orphaned (never imported)
- **File**: `portfolio/meta_learner.py`
- **Problem**: The module has `train()`, `train_all()`, and `predict()` functions but is never imported by any production code. It's trained weekly by a scheduled task (`PF-MLRetrain`) but the predictions are never used. The weekly retraining is essentially wasting CPU.
- **Impact**: No negative impact beyond wasted computation. But it represents dead functionality.
- **Fix**: Either integrate `predict()` into signal_engine.py as a 31st signal, or document it as experimental/orphaned. Integration would be a feature, not a fix.

---

## 2. Architecture Improvements

### ARCH-25: Pass regime through to enhanced signal context
- **File**: `portfolio/signal_engine.py:1030`
- **Problem**: Enhanced signals that need regime information (currently only forecast.py) can't access it because `context_data` doesn't include `regime`.
- **Fix**: Change line 1030 to include regime: `context_data = {"ticker": ticker, "config": config or {}, "macro": macro_data, "regime": regime}`.
- **Impact**: Minimal — only one consumer (forecast.py). No other enhanced signals use `context.get("regime")`.

### ARCH-26: Post-gated vote counts for penalty stages
- **File**: `portfolio/signal_engine.py`
- **Problem**: Vote counts are computed before _weighted_consensus applies regime gating, creating inconsistency between what signals actually voted and what the penalty system thinks voted.
- **Fix**: Apply regime gating to the `votes` dict BEFORE computing `buy`/`sell` counts (lines 1095-1096), so all downstream uses (core gate, min_voters, unanimity penalty) see consistent counts.
- **Impact**: Changes vote counts visible in `extra_info`. Penalty behavior changes for tickers in regimes that have gated signals. Needs careful test verification.

### ARCH-27: Meta-learner model caching for predict()
- **File**: `portfolio/meta_learner.py`
- **Problem**: If the meta-learner is ever integrated, the per-call disk I/O is prohibitive.
- **Fix**: Add a module-level dict cache keyed by horizon. Check model file mtime to detect retraining.
- **Impact**: Required prerequisite for FEAT-3 (meta-learner integration).

---

## 3. Improvements Implemented

### Batch 1: Signal accuracy fixes (2 files, P1) ✓ DONE
**Commit**: `87c3377` — `fix: apply regime gating before vote counts + pass regime to forecast context`

| # | Change | File | Bug | Status |
|---|--------|------|-----|--------|
| 1 | Apply regime gating to votes dict before computing buy/sell counts | `portfolio/signal_engine.py` | BUG-143 | ✓ |
| 2 | Pass regime through context_data to enhanced signals | `portfolio/signal_engine.py` | BUG-144, ARCH-25 | ✓ |

Also fixed 8 pre-existing test failures in `test_confidence_penalties.py` (unanimity penalty interacted with test fixtures using 83% agreement).

### Batch 2: Code quality + resource safety (1 file) ✓ DONE
**Commit**: `3e76954` — `fix: meta_learner SQLite leak, datetime modernize, canonical SIGNAL_NAMES, model cache`

| # | Change | File | Bug | Status |
|---|--------|------|-----|--------|
| 1 | Add try/finally to meta_learner._load_data() | `portfolio/meta_learner.py` | BUG-145 | ✓ |
| 2 | Modernize datetime import | `portfolio/meta_learner.py` | BUG-146 | ✓ |
| 3 | Import SIGNAL_NAMES from tickers | `portfolio/meta_learner.py` | BUG-147 | ✓ |
| 4 | Add model cache to predict() | `portfolio/meta_learner.py` | BUG-148/ARCH-27 | ✓ |

### Batch 3: Tests for new changes ✓ DONE
**Commit**: `8734b93` — `test: add coverage for regime gating, context regime, and meta_learner fixes`

| # | Change | File | Coverage | Status |
|---|--------|------|----------|--------|
| 1 | Test regime gating before vote counts (3 tests) | `tests/test_signal_engine_core.py` | BUG-143 | ✓ |
| 2 | Test regime in context_data (2 tests) | `tests/test_signal_engine_core.py` | BUG-144 | ✓ |
| 3 | Test meta_learner model cache (5 tests) | `tests/test_meta_learner.py` | BUG-148 | ✓ |
| 4 | Test meta_learner SQLite cleanup (2 tests) | `tests/test_meta_learner.py` | BUG-145 | ✓ |
| 5 | Test SIGNAL_NAMES import (2 tests) | `tests/test_meta_learner.py` | BUG-147 | ✓ |

**Total new tests**: 14 (all passing).

---

## 4. Deferred Items (from prior sessions + this session)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk)
- **ARCH-18**: metals_loop.py monolith (risks live trading)
- **ARCH-19**: No CI/CD pipeline (needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption)
- **ARCH-21**: autonomous.py function decomposition (stable, low ROI)
- **ARCH-22**: agent_invocation.py class extraction (touches every caller)
- **BUG-121**: news_event.py sector mapping hardcoded (low value)
- **BUG-132**: orb_predictor.py no caching (low priority)
- **BUG-140**: `_cached()` eviction under lock (negligible impact)
- **BUG-142**: `signal_best_horizon_accuracy()` O(E×T×H×S) (cached, 1h TTL)
- **BUG-149**: meta_learner orphaned — predict() never called (document or integrate)
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config)
- **FEAT-3**: Integrate meta_learner as signal #31 (requires ARCH-27 + accuracy evaluation)

---

## 5. Dependency & Ordering

```
Batch 1 (signal accuracy) → highest priority, changes trading behavior
Batch 2 (meta_learner quality) → independent of Batch 1
Batch 3 (tests) → depends on Batch 1 + 2

Run tests after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 1 file (modify) | Medium — changes consensus + penalty behavior | Medium — existing tests need updating |
| 2 | 1 file (modify) | Low — orphaned module, no production callers | Low — new test file |
| 3 | 2 files (add/modify) | None — test files only | None — new tests |
