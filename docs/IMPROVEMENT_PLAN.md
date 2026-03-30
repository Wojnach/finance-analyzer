# Improvement Plan — Auto-Session 2026-03-30

Updated: 2026-03-30
Branch: improve/auto-session-2026-03-30

## 1. Bugs & Problems Found

### P1 — Critical (affects accuracy or causes incorrect behavior)

#### BUG-150: Cross-horizon averaging bug in `_compute_dynamic_horizon_weights`
- **File**: `portfolio/signal_engine.py:311`
- **Problem**: When computing cross-horizon comparison accuracy, the code uses a running `(old + new) / 2` formula. For 3+ horizons, this gives disproportionate weight to later values instead of a true average.
  ```python
  cross_data[sig] = (cross_data[sig] + acc) / 2  # running average, NOT true mean
  ```
  Example with accuracies [0.60, 0.70, 0.80]:
  - Running: 0.60 → (0.60+0.70)/2=0.65 → (0.65+0.80)/2=0.725
  - True mean: (0.60+0.70+0.80)/3 = 0.700
  The last horizon's accuracy gets ~57% weight instead of 33%.
- **Impact**: Dynamic horizon weights are biased toward the last-processed horizon's accuracy. The cross-horizon baseline is wrong, so the ratio (this_horizon / cross_horizon) over- or under-weights signals. Currently 6 horizons (3h, 4h, 12h, 1d, 3d, 5d) are compared, so the 5d accuracy always dominates.
- **Fix**: Accumulate sum + count, then divide: `cross_sum[sig] = cross_sum.get(sig, 0) + acc; cross_count[sig] = cross_count.get(sig, 0) + 1` → `cross_data[sig] = cross_sum[sig] / cross_count[sig]`.

### P2 — Important (could cause incorrect behavior in edge cases)

#### REF-18: Duplicated LLM context building (~80 lines)
- **File**: `portfolio/signal_engine.py:1005-1043` (Ministral) vs `1081-1128` (Qwen3)
- **Problem**: The `tf_summary` construction (lines 1005-1019 ≡ 1081-1095) and `ema_gap` calculation (lines 1021-1025 ≡ 1097-1101) are identical between Ministral and Qwen3 signal generation. The `ctx` dict is nearly identical (Qwen3 adds `asset_type`).
- **Impact**: If either block is updated without the other, the two LLMs receive different context — a silent divergence. This has already happened: Qwen3 has `asset_type` but Ministral doesn't.
- **Fix**: Extract `_build_llm_context(ticker, ind, timeframes, extra_info) -> dict` that both consumers call, with Qwen3 adding `asset_type` afterward.

#### REF-19: Dead `funding_action`/`funding_rate` code in main.py
- **File**: `portfolio/main.py:341-342`
- **Problem**: These lines check for `funding_action` and `funding_rate` in `extra` dict, but the funding signal has been disabled since BUG-tracking began. The keys are never set in `extra_info` by `generate_signal()`.
- **Impact**: Dead code — never executes. Minor maintenance burden.
- **Fix**: Remove the 2-line block.

### P3 — Minor (code quality, observability)

#### REF-20: `outcome_tracker.py` uses function-local `import logging` (5 occurrences)
- **File**: `portfolio/outcome_tracker.py:161, 323, 396, 441, 453`
- **Problem**: Instead of a module-level `logger = logging.getLogger("portfolio.outcome_tracker")`, the module has 5 separate `import logging as _logging` statements inside function bodies, each creating an ad-hoc logger.
- **Impact**: No functional impact, but inconsistent with every other module in the codebase. Makes grep-based log analysis harder.
- **Fix**: Add module-level `import logging` + `logger = logging.getLogger("portfolio.outcome_tracker")` and replace all 5 function-local patterns.

---

## 2. Architecture Improvements

### ARCH-28: Extract LLM context builder for signal_engine.py
- **File**: `portfolio/signal_engine.py`
- **Problem**: Ministral and Qwen3 signal blocks build nearly identical context dicts. This is the single largest duplicated code block in the signal pipeline.
- **Fix**: Create `_build_llm_context(ticker, ind, timeframes, extra_info)` that returns the shared context dict. Qwen3 can extend it with `asset_type`.
- **Impact**: ~40 lines removed, single point of maintenance for LLM context.

---

## 3. Improvements to Implement

### Batch 1: Fix cross-horizon averaging (P1 bug)
| # | Change | File | Bug | Risk |
|---|--------|------|-----|------|
| 1 | Fix running average → true mean in `_compute_dynamic_horizon_weights` | `portfolio/signal_engine.py` | BUG-150 | Low — only changes edge case weights |

### Batch 2: Refactor duplicated LLM context + dead code cleanup
| # | Change | File | Bug | Risk |
|---|--------|------|-----|------|
| 1 | Extract `_build_llm_context()` helper | `portfolio/signal_engine.py` | REF-18, ARCH-28 | Medium — must preserve both callers' behavior |
| 2 | Remove dead funding_action/funding_rate code | `portfolio/main.py` | REF-19 | None — dead code removal |
| 3 | Add module-level logger to outcome_tracker | `portfolio/outcome_tracker.py` | REF-20 | None — logging cleanup |

### Batch 3: Tests for all changes
| # | Change | File | Coverage |
|---|--------|------|----------|
| 1 | Test `_compute_dynamic_horizon_weights` with 3+ horizons | `tests/test_signal_engine_core.py` | BUG-150 |
| 2 | Test `_build_llm_context` helper | `tests/test_signal_engine_core.py` | REF-18 |
| 3 | Test outcome_tracker uses module logger | `tests/test_outcome_tracker.py` | REF-20 |

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
- **FEAT-3**: Integrate meta_learner as signal #31 (requires accuracy evaluation)

---

## 5. Dependency & Ordering

```
Batch 1 (cross-horizon fix) → highest priority, changes signal weights
Batch 2 (refactor + cleanup) → independent of Batch 1
Batch 3 (tests) → depends on Batch 1 + 2

Run tests after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 1 file (modify) | Low — dynamic horizon weights are clamped + deadband | Low — existing tests should pass |
| 2 | 3 files (modify) | Low — refactor preserves behavior, dead code removal, logging change | Medium — need to verify Ministral/Qwen3 context unchanged |
| 3 | 2 files (add/modify) | None — test files only | None — new tests |
