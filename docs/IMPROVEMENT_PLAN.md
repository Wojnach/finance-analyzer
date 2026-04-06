# Improvement Plan — Auto-Session 2026-04-06

Updated: 2026-04-06
Branch: improve/auto-session-2026-04-06

**Source:** Adversarial review synthesis (2026-04-05) — 10 critical, ~45 high findings.
Previous sessions fixed BUG-80 through BUG-170 + REF-16 through REF-44.
This session addresses the adversarial review findings in prioritized batches.

---

## 1. Bugs & Problems to Fix (from Adversarial Review)

### Batch 1: Safety-Critical Fixes (C2, H18, H34, H35)

#### C2: `place_order_no_page` fails open — non-BUY becomes live SELL
- **File**: `portfolio/avanza_control.py:313-325`
- **Problem**: Any non-"BUY" side (None, empty, "HOLD", typo) falls through to SELL.
- **Impact**: A bug in any caller could liquidate a position.
- **Fix**: Strict validation: `if normalized_side not in ("BUY", "SELL"): raise ValueError(...)`.
- **Risk**: Zero — adds a guard, doesn't change happy path.

#### H18: `delete_stop_loss_no_page` ignores API result, reports false success
- **File**: `portfolio/avanza_control.py:361-374`
- **Problem**: Always returns `True` for any non-exception response, even on API errors.
- **Fix**: Parse response for success indicator; return `False` on unexpected response.
- **Risk**: Zero — makes error reporting more honest.

#### H34: `atomic_write_json` doesn't fsync before os.replace
- **File**: `portfolio/file_utils.py:13-28`
- **Problem**: No `f.flush()` + `os.fsync()` before `os.replace()`. Data may be in OS
  buffer when rename happens. Power loss could result in zero-byte file.
- **Impact**: Portfolio state, accuracy cache, health state could all be lost.
- **Fix**: Add `f.flush(); os.fsync(f.fileno())` before `os.replace()`.
- **Risk**: Zero — adds durability guarantee. Marginal I/O cost (~1ms).

#### H35: `load_json` silent-default hides corruption
- **File**: `portfolio/file_utils.py:31-48`
- **Problem**: All errors (including JSONDecodeError on corrupt data) silently return
  `default`. Callers cannot distinguish "file missing" from "file corrupt".
- **Fix**: Add `require_json()` variant that raises on corruption. Keep `load_json()`
  for backward compat but add WARNING-level log on JSONDecodeError.
- **Risk**: Zero — additive, existing callers unchanged.

### Batch 2: Portfolio State Safety (C7, C8)

#### C7: `load_state` silently regenerates defaults on corrupt JSON
- **File**: `portfolio/portfolio_mgr.py:39-44`
- **Problem**: On load failure, returns fresh defaults. Next `save_state()` permanently
  destroys transaction history.
- **Fix**: On load failure, log CRITICAL + create `.bak` rolling backup on every
  successful save. If underlying load_json returns None (corruption), raise or
  return read-only state.
- **Risk**: Low — tested path, additive backup logic.

#### C8: Portfolio state has no concurrency safety
- **File**: `portfolio/portfolio_mgr.py:39-61`
- **Problem**: No lock between `load_state()` and `save_state()`. Concurrent callers
  can overwrite each other's mutations.
- **Fix**: Add `update_state(fn)` that holds a threading lock during read-modify-write.
  Existing `load_state/save_state` kept for backward compat.
- **Risk**: Low — lock is process-scoped, matches existing usage.

### Batch 3: Dead Code Giving False Security (C4, C6)

#### C4: `record_trade()` has ZERO production call sites
- **File**: `portfolio/trade_guards.py:171-219`
- **Problem**: Entire overtrading guard system is non-functional. Function works (tested)
  but nobody calls it from production code.
- **Fix**: Wire `record_trade()` calls in `portfolio_mgr.save_state/save_bold_state`.
  When `transactions` list grows, extract the new trade and call `record_trade()`.
- **Risk**: Medium — touches trade path. Wire recording only, defer enforcement.

#### C6: MWU signal weights written to disk but never read by engine
- **File**: `portfolio/signal_weights.py`, `portfolio/outcome_tracker.py`
- **Problem**: `SignalWeightManager.batch_update()` writes to `signal_weights.json`
  but `signal_engine.py` never reads it. Dead CPU + disk I/O.
- **Fix**: Remove the dead write path from `outcome_tracker.py`. Keep
  `SignalWeightManager` class for potential future use.
- **Risk**: Zero — removing dead code.

### Batch 4: Math & Signal Correctness (C9, C10, H26)

#### C9: Monte-Carlo t-copula is an identity transform
- **File**: `portfolio/monte_carlo_risk.py:270-290`
- **Problem**: `t_dist.cdf(T, df) → U → t_dist.ppf(U, df)` round-trips to identity.
  VaR/CVaR biased by ~sqrt(2).
- **Fix**: Replace `t_dist.ppf()` with `norm.ppf()` for proper t-copula + Gaussian GBM.
- **Risk**: Low — changes risk estimates to correct values.

#### C10: Regime-gated signals cannot recover through data
- **File**: `portfolio/signal_engine.py:1339-1355`, `portfolio/outcome_tracker.py`
- **Problem**: Votes rewritten to HOLD before logging. Accuracy for gated signals
  never accumulates — they can never be un-gated based on data.
- **Fix**: Log raw pre-gate vote as `raw_vote` alongside gated `vote` in signal
  snapshots. Accuracy computation can use `raw_vote` when present.
- **Risk**: Low — additive field, doesn't change gating behavior.

#### H26: Fear/greed streaks count fetches not days
- **File**: `portfolio/fear_greed.py:33-67`
- **Problem**: `update_fear_streak()` increments `streak_days` on every call (every
  60s loop cycle). After 24h, streak_days = 1440, not 1.
- **Fix**: Compare `last_updated` date to current date. Only increment on date change.
- **Risk**: Zero — streak data is informational only.

### Batch 5: System Safety (C1, C5, H47)

#### C1: Self-heal grants Edit+Bash+Write to Claude CLI in live loop
- **File**: `portfolio/loop_contract.py:625-653`
- **Problem**: On critical violations, spawns Claude with write+bash access synchronously.
  Blocks loop 180s + unreviewed code modification.
- **Fix**: Make self-heal read-only: `allowed_tools="Read,Grep,Glob"`. Diagnostic only.
- **Risk**: Zero — reduces permissions.

#### C5: Singleton lock no-ops on non-Windows
- **File**: `portfolio/main.py:39-55`
- **Problem**: On Linux/WSL, lock always returns True. No concurrent-writer protection.
- **Fix**: Add `fcntl.flock()` for non-Windows. Raise RuntimeError if neither available.
- **Risk**: Zero on Windows (no change). Adds safety on Linux/WSL.

#### H47: EU market open hardcoded to 07:00 UTC — no EU DST
- **File**: `portfolio/market_timing.py:8`
- **Problem**: In EU winter (CET), should be 08:00 UTC. Code uses 07:00 year-round.
- **Fix**: Add `_eu_market_open_hour_utc(dt)` with EU DST logic (last Sunday of
  March → last Sunday of October).
- **Risk**: Zero — only affects market state classification.

### Batch 6: Ruff + Lint Cleanup

- 1 F401 (unused import), 1 I001 (unsorted), 1 UP017, 1 UP035, 3 SIM114, 2 SIM105
- **Risk**: Zero

---

## 2. Deferred Items (NOT in this session)

### Critical (deferred — too risky for autonomous session)
- **C3**: Position-size / stop volume invariant — wrong impl could block valid trades
- **H11**: Warrant positions bypass cash accounting — structural portfolio model change
- **H12-H15**: Silver fast-tick bugs — touches live metals_loop.py monolith
- **H16**: Dual Avanza implementations — needs strategy decision

### Architecture (too broad)
- **A1-A4**: Cross-cutting themes from adversarial review
- **ARCH-18**: metals_loop.py decomposition
- **ARCH-19**: CI/CD pipeline

### Signal accuracy (requires data)
- **H2, H24, H49, H50**: Require accuracy data analysis

---

## 3. Implementation Order & Dependencies

```
Batch 1 (Safety)       → No dependencies, do first
  C2, H18              → avanza_control.py
  H34, H35             → file_utils.py

Batch 2 (Portfolio)    → After Batch 1 (file_utils changes)
  C7, C8               → portfolio_mgr.py

Batch 3 (Dead code)    → Independent of 1-2
  C4, C6               → trade_guards.py, outcome_tracker.py

Batch 4 (Math/Signal)  → Independent of 1-3
  C9, C10, H26         → monte_carlo_risk.py, signal_engine.py, fear_greed.py

Batch 5 (System)       → After Batch 4 (signal_engine changes)
  C1, C5, H47          → loop_contract.py, main.py, market_timing.py

Batch 6 (Lint)         → Last
  Ruff auto-fixes      → Multiple files
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 2 (avanza_control, file_utils) | Zero | Low (new tests) |
| 2 | 1 (portfolio_mgr) | Low | Low (new tests) |
| 3 | 2 (trade_guards, outcome_tracker) | Zero | Zero |
| 4 | 3 (monte_carlo_risk, signal_engine, fear_greed) | Low | Medium |
| 5 | 3 (loop_contract, main, market_timing) | Zero | Low |
| 6 | ~8 (lint) | Zero | Zero |
