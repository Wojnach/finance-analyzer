# Improvement Plan — Auto-Session 2026-04-14

Updated: 2026-04-14
Branch: improve/auto-session-2026-04-14
Status: **IN PROGRESS**

Previous session (2026-04-13): shipped regime-universal signal gating, honest
meta-learner validation, fundamental correlation cluster, ETH qwen3 gate.

## Session Context

Deep exploration by 4 parallel agents analyzed: signal accuracy/gating, core loop
architecture, metals subsystem, and test coverage. Findings cross-referenced with
accuracy_cache.json and ticker_signal_accuracy_cache.json (live data).

---

## 1. Bugs & Problems Found

### BUG-191: Dogpile stuck key on enqueue exception (HIGH)
- **File**: `portfolio/shared_state.py:193-197`
- **Issue**: If `enqueue_fn()` raises, `key` stays in `_loading_keys` for 120s.
  All subsequent requests for that key return stale data instead of refetching.
  Affects: ministral, qwen3, sentiment (GPU signals that can fail on load).
- **Fix**: Wrap `enqueue_fn()` in try/except, discard key on failure.
- **Impact**: Prevents 120s stale-signal windows after GPU load failures.

### BUG-192: Per-ticker signal blacklist incomplete (HIGH)
- **File**: `portfolio/signal_engine.py:135-137`
- **Issue**: `_TICKER_DISABLED_SIGNALS` only has ETH-USD entries. Missing:
  - XAG-USD: ministral (18.9%, 95 sam) — catastrophic on silver
  - XAG-USD: credit_spread_risk (21.2%, 80 sam, SELL-only) — pure noise
  - MSTR: credit_spread_risk (6.5%, 31 sam) — actively harmful
  - XAU-USD: ministral (43.2%, 81 sam) — below gate for metals
- **Fix**: Expand `_TICKER_DISABLED_SIGNALS` with per-ticker accuracy data.
- **Impact**: Removes 4 actively harmful signal-ticker combinations.

### BUG-193: Oscillators globally below gate across all tickers (MEDIUM)
- **File**: `portfolio/tickers.py:61-80`
- **Issue**: oscillators accuracy at 1d: BTC 35.8%, ETH 36.3%, XAG 34.9%,
  XAU 40.2%, MSTR 42.6%. All below 45% gate. 5065 total samples — not
  small-sample. Already regime-gated in ranging but active in other regimes.
  At 3h: XAG 34.3%, ETH 41.6%, XAU 44.7%.
- **Fix**: Add to DISABLED_SIGNALS (like ml, forecast). The accuracy gate
  catches it dynamically but explicit disable is clearer and removes it
  from compute.
- **Impact**: Removes noise signal, reduces consensus pollution.

### BUG-194: Sentiment below gate at 3h for ALL tickers (MEDIUM)
- **File**: `portfolio/signal_engine.py:268-348`
- **Issue**: sentiment 3h_recent = 33.8% (3629 sam). Already regime-gated
  at 3h in trending-up/trending-down/high-vol but NOT in unknown regime.
- **Fix**: Gate sentiment at 3h in ALL regimes via REGIME_GATED_SIGNALS.
- **Impact**: Removes consistently harmful signal at short horizons.

### BUG-195: metals_cross_asset on XAG below gate (MEDIUM)
- **File**: signal_engine.py per-ticker accuracy
- **Issue**: XAG-USD metals_cross_asset = 39.8% (166 sam), 100% BUY bias.
  Bias penalty already applies but signal still votes.
- **Fix**: Add XAG-USD metals_cross_asset to _TICKER_DISABLED_SIGNALS.
- **Impact**: Removes 40% BUY-biased noise from XAG consensus.

---

## 2. Architecture Improvements

### ARCH-1: Dogpile exception safety (shared_state.py)
Wrap `enqueue_fn` call in try/except within `_cached_or_enqueue()`.
3-line fix that prevents 120s stale data windows.

### ARCH-2: Signal phase markers for BUG-178 diagnostics
Add `_set_last_signal(ticker, "__phase_name__")` calls around pre-dispatch,
dispatch loop, and post-dispatch phases. Helps identify hangs in phases
the current diagnostics can't see.

---

## 3. Useful Features

### FEAT-1: accuracy_stats test coverage (HIGH VALUE)
accuracy_stats.py: 1,401 LOC, 19 tests. Core functions untested.
Adding 15-20 tests protects the most critical calculation module.

---

## 4. Ordering — Batches

### Batch 1: Signal gating fixes (BUG-192, BUG-193, BUG-194, BUG-195)
Files: `portfolio/signal_engine.py`, `portfolio/tickers.py`
Tests: `tests/test_signal_engine.py`

### Batch 2: Dogpile exception safety (BUG-191) + phase markers (ARCH-2)
Files: `portfolio/shared_state.py`, `portfolio/signal_engine.py`
Tests: `tests/test_shared_state.py`, `tests/test_signal_engine.py`

### Batch 3: accuracy_stats test coverage (FEAT-1)
Files: `tests/test_accuracy_stats.py`

---

## Deferred (not this session)

- IC-based signal weighting (P2 — needs validation framework)
- HMM regime detection (P2 — needs model tuning)
- MSTR-BTC proxy signal (P1 — needs mNAV data source)
- Metals position sync on startup (HIGH — needs live testing)
- Cancel→sell→rearm atomicity (MEDIUM — needs Avanza API testing)
- prune_jsonl streaming rewrite (MEDIUM — performance, not correctness)
