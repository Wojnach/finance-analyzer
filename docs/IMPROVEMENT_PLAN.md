# Improvement Plan — Auto-Session 2026-05-04

## Methodology

3 parallel exploration agents covering core loop/signals/triggers,
portfolio/risk/reporting, and metals/dashboard/golddigger. Manual verification
of all claims against source. False positives rejected (gpu_gate deadlock,
shared_state race — both correctly synchronized).

---

## 1. Bugs & Problems Found

### P1 — Critical (production impact)

**B1: Equity curve annualization uses 252 days for crypto (24/7 assets)**
- File: `portfolio/equity_curve.py:~228`
- Sharpe, Sortino, and volatility all annualize with sqrt(252). Crypto trades
  365 days/year. This understates crypto volatility by ~17% and overstates
  Sharpe ratio by the same factor.
- Impact: Risk metrics for BTC/ETH are too optimistic.
- Fix: Pass `trading_days` parameter based on asset class (252 for stocks,
  365 for crypto/metals).

**B2: Contract violation dedup ordering creates duplicate critical_errors**
- File: `portfolio/loop_contract.py:429-448`
- `check_layer2_journal_activity()` writes to critical_errors.jsonl (line 431)
  BEFORE persisting the dedup marker (line 446). If the dedup marker write fails
  (atomic_write_json error), subsequent cycles re-fire the same violation without
  dedup. The 7 near-identical contract_violation entries on 2026-05-03 demonstrate
  this pattern.
- Impact: Noise in critical_errors.jsonl, false escalation to fix agents.
- Fix: Persist dedup marker BEFORE writing critical error. If dedup write fails,
  skip the critical error write (violation still surfaces via Telegram alert).

**B3: Monte Carlo ATR default 2% is wrong for most instruments**
- File: `portfolio/monte_carlo.py:~281`
- When `extra.get("atr_pct")` is missing, defaults to 2.0%. Silver warrants
  have 5-8% daily ATR; BTC 3-4%. Underestimates tail risk.
- Impact: VaR/CVaR too optimistic when ATR data missing.
- Fix: Per-asset-class defaults: crypto=3.5, metals=4.0, stocks=2.0.

### P2 — Moderate

**B4: Stuck loading key eviction logged at DEBUG, invisible to operators**
- File: `portfolio/shared_state.py:75`
- When a cache key is stuck loading for >120s and force-evicted, only
  `logger.debug()` is emitted. Operators cannot see that a signal (e.g.,
  Ministral, Qwen3) was permanently stuck.
- Fix: Log at WARNING with key name and duration.

**B5: SYSTEM_OVERVIEW.md stale — 100+ lines of resolved bugs**
- Most bugs listed (BUG-15 through BUG-124) were fixed months ago.
- Signal/module counts outdated (says "16 enhanced disabled", actually 19).
- Fix: Archive resolved bugs, update counts.

### P3 — Low

**B6: Calmar ratio on mini equity curve excludes open positions**
- File: `portfolio/equity_curve.py:~528`
- Documented limitation, not a code bug.

---

## 2. Implementation Batches

### Batch 1: Fix B2 — Contract violation dedup ordering (reliability)
Files: `portfolio/loop_contract.py`
- Swap dedup marker write before critical_error write
- Add guard: if dedup write fails, skip critical error append
- Test: verify dedup-first semantics

### Batch 2: Fix B1 — Equity curve crypto annualization (correctness)
Files: `portfolio/equity_curve.py`, `tests/test_equity_curve.py`
- Add `trading_days` parameter to `compute_metrics()`
- Detect crypto/metals tickers → 365, stocks → 252
- Update Sharpe/Sortino/volatility formulas
- Write/update tests

### Batch 3: Fix B3 — Monte Carlo per-asset ATR defaults (risk accuracy)
Files: `portfolio/monte_carlo.py`, `tests/test_monte_carlo.py`
- Add ASSET_CLASS_ATR_DEFAULTS dict
- Use ticker→asset class mapping for fallback
- Write test

### Batch 4: Fix B4 + Documentation (B5)
Files: `portfolio/shared_state.py`, `docs/SYSTEM_OVERVIEW.md`
- Elevate stuck-key eviction to WARNING
- Archive resolved bugs from SYSTEM_OVERVIEW.md
- Update signal/module counts
