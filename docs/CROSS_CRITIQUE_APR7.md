# Cross-Critique: April 7 Review vs April 5 Review
**Date:** 2026-04-07
**Purpose:** Compare new adversarial findings against prior review, identify what's
new, what's been fixed, and what's been missed both times.

---

## Category 1: Fixed Since April 5 (6 of 10 CRITICALs)

These findings from the April 5 review have been addressed:

| Finding | Fix | Verification |
|---------|-----|--------------|
| C1: Self-heal Edit+Bash+Write | `allowed_tools="Read,Grep,Glob"` | loop_contract.py:657 |
| C2: place_order fails open | Strict `not in ("BUY","SELL")` check | avanza_control.py:323 |
| C5: Singleton lock WSL | Both msvcrt + fcntl | main.py:66-81 |
| C7: load_state silent regen | Rolling backups `_rotate_backups()` | portfolio_mgr.py:43-61 |
| C8: No concurrency safety | `update_state()` with per-file locks | portfolio_mgr.py:135-158 |
| C9: Monte Carlo identity | `norm.ppf(U)` replaces `t_dist.ppf` | monte_carlo_risk.py:291 |
| C10: Dead-signal trap | `raw_votes` captured pre-gate | signal_engine.py:1356 |

**Assessment:** Strong response. 7/10 critical findings fixed in 2 days. The remaining
3 (C3, C4, C6) are all still valid and still impactful.

---

## Category 2: Persistent Findings (Still Open from April 5)

### C3: No position-size / stop volume invariant
**Status:** STILL OPEN
**Why unfixed:** Requires a significant refactor — need to track outstanding stop-loss
volumes alongside position sizes across both `avanza_session.py` and `avanza/trading.py`.
**New context (April 7):** The `fin_snipe_manager.py` (1695 lines) manages exit ladders
with multiple stop-loss orders per position. Without volume invariant checking, the
ladder could exceed position size if partially filled.

### C4: `record_trade()` has zero callers
**Status:** STILL OPEN — THE MOST DANGEROUS OPEN FINDING
**Why unfixed:** Requires identifying all execution paths (Layer 2 CLI subprocess,
autonomous mode, metals loop, golddigger) and wiring `record_trade()` into each.
The Layer 2 execution happens inside a Claude CLI subprocess, making it hard to
add post-fill hooks.
**New context (April 7):** Confirmed by grepping: only golddigger's `record_trade_pnl()`
exists (different function, different module). The main trading paths have NO guard tracking.

### C6: MWU signal weights written, never read
**Status:** STILL OPEN
**Why unfixed:** Likely deprioritized since the weights system is vestigial. But it
still consumes CPU/disk I/O every cycle.

---

## Category 3: New Findings (Not in April 5 Review)

These were identified in the April 7 independent review:

| ID | Finding | Subsystem | Why April 5 missed it |
|----|---------|-----------|----------------------|
| N2 | `atomic_append_jsonl` not truly atomic | infrastructure | Focus was on `atomic_write_json` (correctly identified fsync issue) |
| N3 | Two parallel signal systems diverge | architecture | Noted briefly but not elevated to CRITICAL |
| N5 | `check_drawdown` O(n) scan | portfolio-risk | Performance issue, not a correctness bug |
| N6 | Fallback to entry price overstates value | portfolio-risk | Edge case analysis |
| N7 | Accuracy data mutation corrupts cache | signals-core | New code added since April 5 (per-ticker accuracy) |
| N8 | ADX cache keyed by id(df) | signals-core | Subtle GC interaction, hard to spot |
| N11 | FIFO order confirmation | avanza-api | Avanza orders were added post-April 5 review |

---

## Category 4: April 5 Findings That Were False Positives

From the prior meta-review, Claude had 7 false positives. Lessons:
1. Don't speculate about concurrent paths without verifying lock patterns
2. Don't claim DST bugs without checking the actual direction
3. Verify "missing functionality" claims by grepping for alternatives

---

## Category 5: Convergence (Found by Both Reviews)

Both the April 5 and April 7 reviews independently identified:
- Overtrading guards being advisory-only (severity: "warning", never "block")
- Hardcoded FOMC/CPI dates in econ_calendar.py
- Two parallel signal systems (metals vs main)
- Health state file lacking concurrency protection
- Potential for stale data in cached values

This convergence increases confidence that these are real, systemic issues.

---

## Agent Review Cross-Critique

### Agent 3: portfolio-risk (20 findings — COMPLETED)

**Agent-unique findings (not in independent review):**
- `check_drawdown()` never called — circuit breaker dead code (N22) ← CRITICAL MISS
- Sortino ratio denominator bug (N23) ← mathematical error
- Circuit breaker HALF_OPEN stuck state (N24) ← state machine bug
- Monte Carlo ATR-to-vol assumes fixed candle frequency
- Kelly metrics non-FIFO P&L computation

**Convergent findings (both found):**
- record_trade() zero callers (CRITICAL) — highest-confidence finding
- check_drawdown O(n) scan, fallback to entry price, concentration check

**Assessment:** Agent found check_drawdown dead code which independent reviewer missed entirely.
Strong convergence on the most critical finding (record_trade).

*(Remaining 7 agents still running)*

---

## Meta-Analysis

### Review Quality Improvements (April 7 vs April 5)
1. **More precise verification**: Every finding verified with grep + code read
2. **Delta-focused**: Explicitly checked which prior findings were fixed
3. **Financial impact emphasized**: Each finding rated for monetary risk
4. **Fewer false positives expected**: Learned from April 5's 16% false-positive rate

### Areas Still Under-Reviewed
1. **Dashboard (dashboard/app.py)**: Neither review deeply covered the Flask API
2. **Test suite quality**: 3168 tests exist but coverage gaps unknown
3. **GoldDigger bot**: `portfolio/golddigger/` has its own risk management
4. **Elongir bot**: Not reviewed in either pass
5. **External model files**: `Q:/models/` not inspected
