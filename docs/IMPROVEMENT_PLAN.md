# Improvement Plan

Updated: 2026-03-16
Branch: improve/auto-session-2026-03-16

Previous sessions: 2026-03-05 through 2026-03-14.

## Session Plan (2026-03-16)

### Theme: Signal Health Tracking & Dynamic Applicable Count

Previous sessions completed the IO safety sweep (BUG-47 through BUG-50, REF-8). This
session addresses signal reliability infrastructure: persistent failure tracking,
dynamic `total_applicable` computation, and remaining non-atomic JSONL appends.

### 1) Bugs & Problems Found

#### BUG-51 (P1): Signal failure tracking is ephemeral — no persistent record

- **File**: `portfolio/signal_engine.py:693-696`
- **Issue**: When a signal raises an exception during `generate_signal()`, the failure
  is logged via `logger.warning()` and stored in `extra_info["_signal_failures"]`, but
  this dict is recreated each cycle. There is no persistent record of which signals
  fail, how often, or when they last succeeded. If a signal silently degrades (e.g.,
  Kronos at 0.5% success rate), the only way to discover it is manually checking logs.
- **Fix**: Add `update_signal_health(signal_name, success: bool)` to `health.py` that
  persists per-signal failure counts and timestamps to `data/health_state.json`. Call
  it from the signal execution loop in `signal_engine.py`.
- **Impact**: High. Enables automatic detection of degraded signals and surfaces
  failure rates in Layer 2 context.

#### BUG-52 (P2): `total_applicable` hardcoded, doesn't reflect actual signal availability

- **File**: `portfolio/signal_engine.py:723-730`
- **Issue**: `total_applicable` is hardcoded: 27 for crypto, 25 for stocks/metals.
  But 3 signals are disabled (ML, funding, custom_lora), and others may fail at
  runtime (Kronos ~0.5%, claude_fundamental requires API key). The hardcoded count
  inflates abstention rates and misrepresents signal coverage to Layer 2.
- **Fix**: Compute `total_applicable` dynamically from `SIGNAL_NAMES` minus
  `DISABLED_SIGNALS` minus signals that failed this cycle (from BUG-51 tracking).
  Also account for signals that don't apply to certain asset classes (e.g.,
  futures_flow only applies to crypto).
- **Impact**: Medium. More accurate confidence reporting.

#### BUG-53 (P2): 7 modules use non-atomic JSONL appends

- **Files**:
  - `portfolio/forecast_signal.py:296`
  - `portfolio/sentiment.py:398`
  - `portfolio/regime_alerts.py:98`
  - `portfolio/risk_management.py:433`
  - `portfolio/signals/forecast.py:175,801`
  - `portfolio/weekly_digest.py:287`
  - `portfolio/orb_postmortem.py:135`
- **Issue**: These files use raw `open("a") + write()` instead of
  `atomic_append_jsonl()` from `file_utils`. While less dangerous than non-atomic
  writes (append doesn't truncate), they lack flush+fsync guarantees.
- **Fix**: Replace with `atomic_append_jsonl()`.
- **Impact**: Low-medium. Consistency with established pattern.

#### BUG-54 (P3): `_compute_adx()` called repeatedly without caching

- **File**: `portfolio/signal_engine.py` (within confidence penalties)
- **Issue**: ADX computation iterates the full kline array each call. Not cached
  like other indicator values. Minor perf waste on each cycle.
- **Fix**: Cache ADX in the indicators cache alongside RSI/MACD/etc.
- **Impact**: Low. Minor performance improvement.

#### BUG-55 (P3): `fin_evolve.py` has dead fallback wrappers for file_utils

- **File**: `portfolio/fin_evolve.py:1-20`
- **Issue**: Has local `_load_json()` / `_load_jsonl()` wrappers with
  `ImportError` fallback to raw `json.loads`. Since `file_utils` is guaranteed
  to exist (it's in the same package), the fallback is dead code.
- **Fix**: Replace with direct `from portfolio.file_utils import load_json, load_jsonl`.
- **Impact**: Low. Code hygiene.

### 2) Architecture Improvements

#### ARCH-12: Signal failure tracking and health surfacing

- **Files**: `portfolio/health.py`, `portfolio/signal_engine.py`, `portfolio/reporting.py`
- **What**: Persistent per-signal health metrics: failure count, last success/failure
  timestamps, rolling success rate (7-day window). Surfaced in
  `agent_summary_compact.json → signal_health` section for Layer 2 awareness.
- **Why**: Currently no automated detection of signal degradation. Kronos was at
  0.5% success rate for weeks before manual discovery. Health tracking enables
  auto-alerting and dynamic signal weighting.
- **How**: `health.py` gets `update_signal_health()` + `get_signal_health()`.
  `signal_engine.py` calls it after each signal execution. `reporting.py` includes
  health summary in compact output.

#### ARCH-14: Dynamic `total_applicable` computation

- **Files**: `portfolio/signal_engine.py`, `portfolio/tickers.py`
- **What**: Replace hardcoded 27/25 with computed value from active signal list,
  accounting for disabled signals and per-asset-class applicability.
- **Why**: Hardcoded values become wrong when signals are added/removed/disabled.
  Currently 3 signals are disabled but still counted as applicable.
- **How**: `_compute_applicable_count(sector)` function using `SIGNAL_NAMES`,
  `DISABLED_SIGNALS`, and per-signal asset class restrictions.

### 3) Features

#### FEAT-2: Signal failure rate in accuracy reports

- **File**: `portfolio/accuracy_stats.py`
- **What**: Include signal failure rate (from ARCH-12 health data) alongside
  accuracy stats in the `--accuracy` CLI output and in `agent_summary_compact.json`.
- **Why**: A signal with 90% accuracy but 80% failure rate is effectively only
  voting 20% of the time — that context matters for Layer 2 decisions.

### 4) Refactoring

#### REF-9: Consolidate JSONL append pattern

- **Files**: 7 files listed in BUG-53
- **What**: Replace raw `open("a")` JSONL appends with `atomic_append_jsonl()`.
- **Why**: Consistency, flush+fsync guarantees, error handling.

#### REF-10: Remove dead `fin_evolve.py` fallback wrappers

- **File**: `portfolio/fin_evolve.py`
- **What**: Replace `_load_json` / `_load_jsonl` with direct `file_utils` imports.
- **Why**: Dead code that obscures actual dependencies.

### 5) Test Coverage

#### TEST-12: Signal health tracking tests

- **File**: `tests/test_signal_health.py`
- Tests for `update_signal_health()`, `get_signal_health()`, rolling window,
  persistence across restarts, integration with `generate_signal()`.

#### TEST-13: Dynamic applicable count tests

- **File**: `tests/test_signal_engine_core.py` (extend existing)
- Tests for `_compute_applicable_count()` with different sectors and disabled
  signal configurations.

### 6) Dependency/Ordering

#### Batch 1: Signal health infrastructure (BUG-51 + ARCH-12)
- Files: `portfolio/health.py`, `portfolio/signal_engine.py`
- Tests: `tests/test_signal_health.py`
- Adds `update_signal_health()` / `get_signal_health()` to health.py
- Wires signal_engine.py to call health tracking after each signal
- Zero-risk to existing behavior — additive only

#### Batch 2: Dynamic applicable count + JSONL consolidation (BUG-52 + BUG-53 + REF-9)
- Files: `portfolio/signal_engine.py`, `portfolio/tickers.py`,
  `portfolio/forecast_signal.py`, `portfolio/sentiment.py`,
  `portfolio/regime_alerts.py`, `portfolio/risk_management.py`,
  `portfolio/signals/forecast.py`, `portfolio/weekly_digest.py`,
  `portfolio/orb_postmortem.py`
- Tests: extend `tests/test_signal_engine_core.py`
- Depends on Batch 1 (uses health data for "actually available" count)

#### Batch 3: Reporting integration + cleanup (FEAT-2 + REF-10 + BUG-54 + BUG-55)
- Files: `portfolio/reporting.py`, `portfolio/accuracy_stats.py`,
  `portfolio/fin_evolve.py`, `portfolio/signal_engine.py`
- Depends on Batch 1+2 (surfaces health data in reports)

#### Batch 4: Final tests + documentation
- Files: `tests/test_signal_health.py` (verify all), `docs/SYSTEM_OVERVIEW.md`
- Run full test suite, verify no regressions

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 5 | BUG-51 through BUG-55 |
| Architecture | 2 | ARCH-12, ARCH-14 |
| Features | 1 | FEAT-2 |
| Refactoring | 2 | REF-9, REF-10 |
| Test Coverage | 2 | TEST-12, TEST-13 |
| Batches | 4 | Batch 1 health → Batch 2 dynamic+JSONL → Batch 3 reporting → Batch 4 tests |
