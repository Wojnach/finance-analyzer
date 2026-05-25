# Improvement Plan — Auto-Session 2026-05-25

**Branch:** `improve/auto-session-2026-05-25`
**Created:** 2026-05-25 10:00 CET

---

## 1. Bugs & Problems Found

### BUG-A: 5 signals missing from SIGNAL_NAMES → accuracy never tracked
**Severity:** HIGH
**Files:** `portfolio/tickers.py:66` (SIGNAL_NAMES + DISABLED_SIGNALS)
**Impact:** outcome_tracker.py:125-128 iterates only over SIGNAL_NAMES when recording votes to signal_log.jsonl. These 5 registered+disabled signals are computed each cycle (shadow mode) but their votes are silently dropped — accuracy can never accumulate, so they can never graduate from shadow to active.

Orphaned signals (in DISABLED_SIGNALS + registry, but NOT in SIGNAL_NAMES):
- `adx_regime_switch` (added 2026-05-19)
- `bocpd_regime_switch` (added 2026-05-24)
- `choppiness_regime_gate` (added 2026-05-22)
- `connors_rsi2` (added 2026-05-19)
- `gold_overnight_bias` (added 2026-05-11)

Additionally, 3 LLM signals are registered but in neither SIGNAL_NAMES nor DISABLED_SIGNALS:
- `cryptotrader_lm`
- `finance_llama`
- `meta_trader`

These may be handled via the shadow_registry LLM rotation system separately, but should still be in SIGNAL_NAMES for accuracy tracking consistency.

**Fix:** Add all 8 signals to SIGNAL_NAMES. The 5 shadow signals are already in DISABLED_SIGNALS, so they remain force-HOLD in consensus.

### BUG-B: SYSTEM_HEALTH_CONTRACT.md references 20 instruments (only 5 remain)
**Severity:** LOW (documentation only)
**File:** `docs/SYSTEM_HEALTH_CONTRACT.md:45-55`
**Impact:** Health check script may validate against stale expected ticker list. Misleading for operators.

### BUG-C: Dashboard lacks security headers
**Severity:** MEDIUM (security)
**File:** `dashboard/app.py`
**Impact:** No X-Frame-Options (clickjacking), no X-Content-Type-Options (MIME sniffing), no Strict-Transport-Security. Dashboard serves financial data behind auth — these are standard defense-in-depth headers.

---

## 2. Architecture Improvements

None proposed this session. Architecture is sound — event-driven orchestrator with plugin registry, atomic I/O, thread-safe caches, tiered agent dispatch. The system has survived 3+ months of continuous operation.

---

## 3. Useful Features

None proposed. The system is feature-complete for its current scope.

---

## 4. Refactoring & Cleanup

### CLEANUP-A: `register_signal()` decorator in signal_registry.py is dead code
**File:** `portfolio/signal_registry.py:16-35`
**Impact:** Never used. All 66 registered signals use `register_enhanced()`.

### CLEANUP-B: SYSTEM_OVERVIEW.md stale counts
**File:** `docs/SYSTEM_OVERVIEW.md`
**Impact:** Line counts, signal counts, and module counts drift from reality.

---

## 5. Dependency & Ordering

### Batch 1: Fix signal tracking bug (BUG-A)
**Files:** `portfolio/tickers.py`
**Risk:** Low — adding to SIGNAL_NAMES is additive. Tests that hardcode signal counts will need tripwire updates.

### Batch 2: Dashboard security headers (BUG-C)
**Files:** `dashboard/app.py`
**Risk:** Low — additive headers won't break existing functionality.

### Batch 3: Documentation fixes (BUG-B, CLEANUP-B)
**Files:** `docs/SYSTEM_HEALTH_CONTRACT.md`, `docs/SYSTEM_OVERVIEW.md`
**Risk:** None.

### Batch 4: Dead code removal (CLEANUP-A)
**Files:** `portfolio/signal_registry.py`
**Risk:** Very low — grep confirms no callers.
