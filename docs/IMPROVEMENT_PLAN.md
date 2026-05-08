# Improvement Plan — auto-session-2026-05-08

## Exploration Summary

Deep exploration via 4 parallel agents + direct reads. Coverage: core loop + signal engine,
risk + portfolio management, metals + dashboard + infra, signal modules + recent changes.

---

## 1. Bugs & Problems Found

### P0 — Critical

**B1: BUG-111 — RSI adaptive thresholds not used in outcome reconstruction**
- File: `portfolio/outcome_tracker.py:34-43`
- `_derive_signal_vote()` hardcodes RSI at [30, 70] but live `generate_signal()` uses
  adaptive `rsi_p20/rsi_p80` from rolling percentiles (capped 15-85).
- Impact: RSI accuracy tracking compares against wrong threshold. Every RSI outcome
  potentially misclassified, poisoning accuracy data that gates downstream signals.
- Fix: Read `rsi_p20`/`rsi_p80` from signal snapshot indicators. Fall back to [30, 70]
  for old snapshots missing these fields.

**B2: Silent Cholesky failure in monte_carlo_risk**
- File: `portfolio/monte_carlo_risk.py` — nearest-PSD fallback path
- When Cholesky fails, `_nearest_psd()` is used but no warning logged.
- Impact: Silent VaR/CVaR degradation with corrupted correlation matrix.
- Fix: Add `logger.warning()` on Cholesky fallback.

### P1 — High

**B3: outcome_tracker SQLite dual-write failure swallowed**
- File: `portfolio/outcome_tracker.py:160-167`
- SQLite write wrapped in bare `try/except` with no logging.
- Impact: signal_log.db can silently fall behind signal_log.jsonl. Accuracy queries
  from SQLite return incomplete data.
- Fix: Log at WARNING level with entry count context.

**B4: signal_registry failure sentinel doesn't expire properly**
- File: `portfolio/signal_registry.py:91`
- Import failures set `_fail_ts` for cooldown, but on process restart a previously
  failed signal never retries because the cooldown logic works correctly. The real
  issue: no periodic retry after initial cooldown expires. A signal that fails at
  startup due to a transient dependency remains dead for the entire process lifetime.
- Fix: Add a 5-minute retry window after each cooldown expiry.

**B5: ~15 bare `except Exception: pass` patterns across codebase**
- Files: `portfolio/accuracy_degradation.py:956`, `portfolio/agent_invocation.py:1053`,
  and others found via grep.
- Impact: Silent failure modes. The 3-week Layer 2 auth outage was partially caused
  by this pattern hiding errors.
- Fix: Replace bare `pass` with `logger.debug("...", exc_info=True)` minimum.
  For critical paths (journal writes, accuracy snapshots), use WARNING.

**B6: Signal sub-indicator exceptions silently swallowed**
- Files: All composite signal modules (momentum.py, oscillators.py, volatility.py, etc.)
- Pattern: `except Exception: sub_signals["x"] = "HOLD"` with no logging.
- Impact: Bugs in sub-indicators (IndexError, AttributeError) are invisible. Signal
  votes HOLD without anyone knowing computation failed.
- Fix: Add `logger.warning("sub-indicator %s failed: %s", name, e)` to catch blocks.

### P2 — Medium

**B7: FX rate hardcoded fallback (10.85 SEK/USD)**
- File: `portfolio/risk_management.py:125-184`
- Three-tier fallback chain ends at hardcoded 10.85. Current SEK/USD ~10.3 → 5% error.
- Impact: Could trigger false drawdown breach on extended API outage.
- Fix: Update to 10.50, add staleness counter.

**B8: equity_curve drops day when prev_val == 0**
- File: `portfolio/equity_curve.py:104`
- Zero previous value silently drops daily return instead of recording 0%.
- Impact: Biases Sharpe/Sortino during portfolio initialization.
- Fix: Treat zero prev_val as 0% return.

---

## 2. Architecture Improvements

### A1: Standardize DATA_DIR path resolution across signal modules
- Problem: Inconsistent path patterns. Some use `Path(__file__).resolve().parent.parent.parent / "data"`,
  others use `os.path.dirname()` chains, one (cot_positioning) was using relative paths (fixed 2026-05-02).
- Fix: Add `DATA_DIR` constant to `signal_utils.py` and import it.
- Impact: Prevents future CWD-relative path bugs. ~5 files need updating.

### A2: Extract common z-score helper to signal_utils
- Problem: 3+ signal modules independently compute z-scores with near-identical code.
- Fix: Add `zscore(series, window)` to `signal_utils.py`. Replace inline implementations.
- Impact: ~30 lines of deduplication. Lower maintenance burden.

---

## 3. Useful Features

### F1: Loop contract — SQLite/JSONL reconciliation check
- Add invariant to `verify_contract()` comparing signal_log.jsonl line count vs
  signal_log.db row count. WARNING if divergence > 100.
- Why: Makes B3 (dual-write gap) observable.
- Impact: ~20 lines in loop_contract.py.

---

## 4. Refactoring TODOs

### R1: Replace bare `except Exception: pass` with logging (~15 locations)
### R2: Add logging to signal sub-indicator catch blocks (~20 locations)
### R3: Standardize DATA_DIR in signal modules using relative path chains

---

## 5. Execution Batches

### Batch 1: Critical bug fixes (B1, B2) — 2 files
1. `portfolio/outcome_tracker.py` — B1: Fix RSI threshold reconstruction
2. `portfolio/monte_carlo_risk.py` — B2: Add Cholesky fallback warning

### Batch 2: Silent failure observability (B3, B5, B6) — ~10 files
1. `portfolio/outcome_tracker.py` — B3: Log SQLite write failures
2. `portfolio/accuracy_degradation.py` — B5: Replace bare pass with logging
3. `portfolio/agent_invocation.py` — B5: Replace bare pass with logging
4. Signal modules — B6: Add logging to sub-indicator catch blocks
   (momentum.py, oscillators.py, volatility.py, candlestick.py, structure.py,
   heikin_ashi.py, mean_reversion.py, macro_regime.py)

### Batch 3: Medium fixes + utility (B7, B8, A2) — 3 files
1. `portfolio/risk_management.py` — B7: Update FX fallback
2. `portfolio/equity_curve.py` — B8: Handle zero prev_val
3. `portfolio/signal_utils.py` — A2: Add zscore helper

### Batch 4: Path standardization (A1) + loop contract (F1) — ~6 files
1. `portfolio/signal_utils.py` — A1: Add DATA_DIR constant
2. Signal modules with non-standard paths — A1: Import DATA_DIR
3. `portfolio/loop_contract.py` — F1: Add SQLite/JSONL reconciliation

### Batch 5: Documentation + verification
1. `docs/SYSTEM_OVERVIEW.md` — Update
2. `docs/SESSION_PROGRESS.md` — Final session notes
3. Full test suite run
4. Merge, push

---

## Risk Assessment

- **Batch 1**: Low risk. outcome_tracker change adds field reads with fallback. monte_carlo_risk adds logging only.
- **Batch 2**: Very low risk. All changes add logging without changing behavior.
- **Batch 3**: Low risk. FX fallback is a constant update. equity_curve change affects edge case only. zscore is new utility function.
- **Batch 4**: Low risk. Path changes are mechanical. Loop contract check is additive.
