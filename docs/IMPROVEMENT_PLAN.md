# Improvement Plan — Auto-Session 2026-04-15

Updated: 2026-04-15
Branch: improve/auto-session-2026-04-15
Status: **IN PROGRESS**

Previous session (2026-04-14): shipped signal gating (BUG-192–195), dogpile
exception safety (BUG-191), 22 accuracy tests, correlation realignment.

## Session Context

Deep exploration by 4 parallel agents analyzed: core orchestration (main.py,
agent_invocation.py, trigger.py, market_timing.py, signal_engine.py),
signal modules (31 files), data/portfolio/risk modules, metals subsystem,
dashboard, and reporting. Cross-referenced with live accuracy_cache.json.

Agent findings were verified — several false positives filtered:
- OBV integer overflow (not realistic, 5 orders of magnitude from int64 max)
- Knock-out distance formula (correct: (price-financing)/price = % drop to KO)
- Smart money missing try-except (already has them per sub-indicator)
- agent_invocation.py timestamp parsing bug (Python 3.12 handles +00:00/Z natively)

---

## 1. Bugs & Problems Found

### BUG-196: Relative path fragility in 6 modules (MEDIUM)
- **Files**:
  - `portfolio/microstructure_state.py:22` → `Path("data/microstructure_state.json")`
  - `portfolio/fear_greed.py:16` → `Path("data/fear_greed_streak.json")`
  - `portfolio/seasonality.py:21` → `Path("data/seasonality_profiles.json")`
  - `portfolio/linear_factor.py:27` → `Path("data/models/linear_factor_weights.json")`
  - `portfolio/signal_weight_optimizer.py:27` → `Path("data/models/walkforward_results.json")`
  - `portfolio/train_signal_weights.py:30` → `Path("data/signal_log.jsonl")`
- **Issue**: These use relative `Path("data/...")` instead of absolute `BASE_DIR / "data" / ...`.
  If CWD changes (e.g., subprocess, test runner), paths break silently.
  Most other modules (40+) correctly use `Path(__file__).resolve().parent.parent / "data"`.
- **Fix**: Add `BASE_DIR = Path(__file__).resolve().parent.parent` and use `DATA_DIR = BASE_DIR / "data"`.
- **Impact**: Prevents silent file-not-found in edge cases. Zero behavior change when CWD is correct.

### BUG-197: Dead timestamp cleaning code in agent_invocation.py (LOW)
- **File**: `portfolio/agent_invocation.py:691`
- **Issue**: `ts_str_clean` is computed (replacing `+00:00` → `+0000`, `Z` → `+0000`) but never used —
  line 693 passes original `ts_str` to `fromisoformat()`. In Python 3.12, `fromisoformat()` handles
  both `+00:00` and `Z` natively, so the code works but is misleading dead code.
- **Fix**: Remove `ts_str_clean` and the conditional, use `fromisoformat()` directly.
- **Impact**: Cleaner code, no behavior change.

### BUG-198: Signal registry re-imports on every failure (MEDIUM)
- **File**: `portfolio/signal_registry.py:78-89`
- **Issue**: `load_signal_func()` returns `None` on import failure but doesn't cache the failure.
  On next call, it re-attempts the import. With 5 tickers × 7 timeframes × per-cycle,
  a broken signal module logs ~35 warnings per cycle (every 60-600s).
- **Fix**: Cache `None` on import failure with a TTL so the warning is logged once, not 35× per cycle.
- **Impact**: Reduces log spam from broken signal modules. No behavior change (still returns None/HOLD).

### BUG-199: Trigger sustained gate logic duplicated (LOW)
- **File**: `portfolio/trigger.py:189-216` and `portfolio/trigger.py:242-278`
- **Issue**: Signal flip (section 2) and sentiment reversal (section 5) use identical
  count+duration debounce logic (SUSTAINED_CHECKS consecutive cycles OR SUSTAINED_DURATION_S
  wall-clock seconds). Copy-pasted, changes to one require changes to the other.
- **Fix**: Extract `_update_sustained(state_dict, key, value, now_ts)` helper returning
  `(count_ok, duration_ok)`.
- **Impact**: Maintenance improvement. Reduces risk of future divergence.

### BUG-200: Dashboard CORS allows all origins (LOW)
- **File**: `dashboard/app.py:42`
- **Issue**: `Access-Control-Allow-Origin: *` allows any website to make API requests.
  Combined with optional auth (no token configured = public access), portfolio data
  is readable from any origin.
- **Note**: Dashboard runs on local network only (port 5055), so practical risk is low.
  Flagging for awareness — not changing since user accesses from phone on LAN.

---

## 2. Architecture Improvements

### ARCH-3: Consistent path resolution pattern
Standardize all modules to use `BASE_DIR = Path(__file__).resolve().parent.parent`
pattern already used by 40+ modules. Fixes BUG-196.

### ARCH-4: Trigger debounce helper
Extract shared `_update_sustained()` from trigger.py to DRY the sustained gate logic.
Fixes BUG-199.

---

## 3. Test Coverage

### TEST-1: trigger.py sustained gate tests (HIGH VALUE)
trigger.py has extensive tests for consensus triggers but the sustained flip + sentiment
reversal debounce (count OR duration) paths are undertested. Adding 6-8 tests protects
the refactored helper.

### TEST-2: microstructure_state.py path resolution tests
Verify that the fixed absolute paths work correctly regardless of CWD.

### TEST-3: signal_registry.py failed import caching test
Verify that a broken signal module is only warned about once, not 35× per cycle.

---

## 4. Ordering — Batches

### Batch 1: Path hardening (BUG-196)
Files: `portfolio/microstructure_state.py`, `portfolio/fear_greed.py`,
`portfolio/seasonality.py`, `portfolio/linear_factor.py`,
`portfolio/signal_weight_optimizer.py`, `portfolio/train_signal_weights.py`
Tests: Existing tests should pass; add 2 path-resolution smoke tests.

### Batch 2: Dead code + DRY trigger (BUG-197, BUG-199)
Files: `portfolio/agent_invocation.py`, `portfolio/trigger.py`
Tests: `tests/test_trigger_core.py` — add 6-8 sustained gate tests

### Batch 3: Signal registry error caching (BUG-198)
Files: `portfolio/signal_registry.py`
Tests: `tests/test_signal_registry.py` or inline

---

## Deferred (not this session)

- Dashboard CORS restriction (BUG-200) — user accesses from phone on LAN, needs `*`
- Dashboard default-require-auth — would break existing setup
- IC-based signal weighting (P2 — needs validation framework)
- HMM regime detection (P2 — needs model tuning)
- MSTR-BTC proxy signal (P1 — needs mNAV data source)
- Graceful shutdown (KeyboardInterrupt handler) — low impact, 5-min heartbeat recovery exists
