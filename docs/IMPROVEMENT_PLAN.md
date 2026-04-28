# Improvement Plan — Auto-Improve Session 2026-04-28

## Summary

Findings from 4 parallel exploration agents + direct investigation. Focused on
real bugs, security issues, and lint violations. Excluding style-only changes,
deferred architecture work (ARCH-18+), and E402 violations (intentional lazy imports).

---

## 1. Bugs & Problems Found

### BUG-230 (P1): Dashboard CORS wildcard allows cross-origin data theft
- **File**: `dashboard/app.py:44-49`
- **Issue**: `Access-Control-Allow-Origin: *` with optional auth means any website
  can fetch portfolio data via XHR if user has dashboard open in browser.
- **Fix**: Restrict CORS to localhost origins. Add `Access-Control-Allow-Credentials: false`.
- **Impact**: Security-only. No functional change to dashboard behavior.

### BUG-231 (P2): Heartbeat uses non-atomic `.write_text()`
- **File**: `portfolio/main.py:1098, 1147`
- **Issue**: If process crashes mid-write, `heartbeat.txt` is corrupt. Next restart
  may fail parsing the truncated ISO timestamp.
- **Fix**: Use `atomic_write_text()` from file_utils.
- **Impact**: Startup reliability. Low risk — heartbeat is just a timestamp string.

### BUG-232 (P2): NaN fx_rate passes guard in portfolio_value()
- **File**: `portfolio/portfolio_mgr.py:162`
- **Issue**: Guard is `fx_rate <= 0` — NaN fails this check (`NaN <= 0` is False),
  so NaN propagates into portfolio value calculation, returning NaN.
- **Fix**: Add `math.isnan` check alongside the existing guard.
- **Impact**: Portfolio value becomes NaN instead of cash-only fallback. Affects
  reporting, risk management downstream.

### BUG-233 (P3): `CANCEL_HOUR`/`CANCEL_MIN` undefined in fish_monitor_live.py
- **File**: `scripts/fish_monitor_live.py:809`
- **Issue**: `NameError` at runtime if straddle mode is entered.
- **Fix**: Define constants (likely 21, 55 based on Avanza warrant hours).
- **Impact**: Script is deprecated (fish engine disabled 2026-04-17), but still crashes if run.

### BUG-234 (P3): Unused variable `recent_horizon` in signal_engine.py
- **File**: `portfolio/signal_engine.py:2919`
- **Issue**: Computed but never used (actual call uses `base_hz` on line 2922).
- **Fix**: Remove the variable.
- **Impact**: None — dead assignment.

### BUG-235 (P3): Dashboard 500 errors expose internal exception messages
- **File**: `dashboard/app.py` (multiple endpoints)
- **Issue**: `return jsonify({"error": str(e)}), 500` leaks file paths and internals.
- **Fix**: Log full traceback server-side, return generic error message to client.
- **Impact**: Information disclosure. Low risk on LAN-only dashboard.

---

## 2. Lint Violations (Ruff)

### Batch A: Auto-fixable (22 violations)
- 10 F401 unused imports (portfolio/)
- 8 I001 unsorted imports (portfolio/)
- 4 UP045 non-PEP604 Optional annotations (portfolio/)

### Batch B: Unused variables (9 violations)
- `portfolio/signal_engine.py:2919` — `recent_horizon`
- `portfolio/signals/complexity_gap_regime.py:108` — `n_assets`
- `portfolio/signals/crypto_evrp.py:224` — `recent_rv`
- `portfolio/signals/mahalanobis_turbulence.py:125` — `n_assets`
- `data/metals_risk.py:835,836,846,849` — 4 computed-but-unused vars
- `data/test_metals_swing_trader.py:61` — `api_type`

### Batch C: Unused imports in data/ and signals/ (12 violations)
- `data/metals_swing_trader.py:13` — `json`
- `portfolio/signals/hash_ribbons.py:31` — `majority_vote`
- `portfolio/signals/xtrend_equity_spillover.py:30` — `sma`, `ema`, `rsi`
- `portfolio/mstr_loop/*.py` — 6 unused imports across 4 files

### Batch D: SIM simplifications (11 violations in portfolio/)
- 4 SIM102 collapsible-if
- 2 SIM103 needless-bool
- 5 SIM115 context-manager for file open

### Batch E: Scripts cleanup (key items only)
- 12 E722 bare-except → `except Exception:`
- 3 F821 undefined names (CANCEL_HOUR/CANCEL_MIN in fish_monitor_live.py)
- 9 F401 unused imports
- 5 F541 f-strings without placeholders

---

## 3. Architecture Improvements

None proposed this session. Previous sessions addressed the major items (ARCH-10
through ARCH-27). Remaining ARCH items (18: metals monolith, 19: CI/CD, 20: mypy,
21: autonomous.py decomposition, 22: agent_invocation class) are all deferred and
require larger scope than a single auto-improve session.

---

## 4. Refactoring TODOs

### REF-50: Remove dead fish_engine references in metals_loop.py
- Lines 703-704 reference `_fish_engine` and `_reconcile_fish_engine_position`
- Engine permanently deprecated 2026-04-17. Safe to remove.
- **Deferred**: Touching metals_loop.py (7667 lines) is high-risk in an auto session.

---

## 5. Implementation Batches

### Batch 1: Security & Safety Fixes (3 files)
1. `dashboard/app.py` — CORS restriction (BUG-230), error message sanitization (BUG-235)
2. `portfolio/main.py` — Atomic heartbeat writes (BUG-231)
3. `portfolio/portfolio_mgr.py` — NaN fx_rate guard (BUG-232)

### Batch 2: Ruff auto-fix + unused variables (portfolio/, data/)
1. Run `ruff check --fix` for F401, I001, UP045 across portfolio/
2. Manually fix F841 unused variables (9 files)
3. Fix BUG-233 in scripts/fish_monitor_live.py

### Batch 3: Scripts & SIM cleanup
1. Bare-except fixes (E722) in scripts/
2. Auto-fix F401, F541, I001 in scripts/
3. SIM simplifications in portfolio/ (collapsible-if, needless-bool)

---

## 6. Risk Assessment

- **Batch 1**: Low risk. Heartbeat and NaN guard are defensive additions. CORS
  change is tightening, not loosening.
- **Batch 2**: Very low risk. Removing dead code/imports. No behavioral changes.
- **Batch 3**: Low risk. Style fixes in scripts (not production loop code).

All batches are additive/tightening. No behavioral changes to trading logic,
signal computation, or order execution.
