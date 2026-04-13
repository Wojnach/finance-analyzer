# Improvement Plan — Auto-Session 2026-04-13

Updated: 2026-04-13
Branch: improve/auto-session-2026-04-13
Status: **IN PROGRESS**

## Session Context

Continuing from the 2026-04-12 after-hours research session which shipped 6 signal quality
improvements (forecast disabled, econ_calendar gated, tiered accuracy gate, cluster penalties).
Three P1 adversarial review findings remain unfixed. This session addresses them plus one P2
signal gating improvement.

---

## 1. Bugs & Problems Found

### P1: SC-I-001 — _weighted_consensus() re-gates BUG-158-exempt signals
- **File**: `portfolio/signal_engine.py` lines 768-769 vs 1778-1791
- **Bug**: `generate_signal()` exempts high-accuracy per-ticker signals from regime gating
  (BUG-158), but `_weighted_consensus()` unconditionally re-applies regime gating at line 769,
  negating the exemption.
- **Impact**: Per-ticker alpha recovery lost (e.g., fear_greed 93.8% on XAG-USD in ranging).
- **Fix**: Add `regime_gated_override` parameter to `_weighted_consensus()`. Pass
  `regime_gated_effective` from `generate_signal()`.
- **Risk**: Low — additive param with None default.

### P1: CROSS-001 — outcome_tracker reads post-gated votes for accuracy tracking
- **File**: `portfolio/outcome_tracker.py` line 122
- **Bug**: `extra.get("_votes")` reads post-gated votes. Regime-gated signals appear as HOLD,
  so accuracy never accumulates. This defeats the C10 raw_votes fix and creates a circular
  dependency: can't exempt without accuracy data → can't track accuracy because gated.
- **Impact**: Dead-signal trap for all regime-gated signals.
- **Fix**: `extra.get("_raw_votes", extra.get("_votes"))` — raw first, gated fallback.
- **Risk**: Low — _raw_votes is superset of information.

### P1: OR-I-001 — ThreadPoolExecutor with-block blocks main loop on hung threads
- **File**: `portfolio/main.py` lines 555-596
- **Bug**: `with ThreadPoolExecutor(...) as pool:` — `__exit__()` calls `shutdown(wait=True)`,
  blocking the loop even after the 180s timeout fires and futures are cancelled.
- **Impact**: Single stuck API call can hang the entire loop indefinitely.
- **Fix**: Manual executor lifecycle. `pool.shutdown(wait=False, cancel_futures=True)` after
  timeout (Python 3.9+). Add 5s grace period via `shutdown(wait=True)` with short timeout
  wrapper as best-effort cleanup.
- **Risk**: Medium — must prevent thread leaks. Mitigated by cancel_futures=True and grace period.

## 2. Signal Improvements

### P2: Gate news_event for ETH-USD
- **Issue**: 39.2% accuracy on ETH-USD, 100% SELL bias. Actively harmful.
- **Fix**: Add per-ticker signal disable dict: `_TICKER_DISABLED_SIGNALS = {"ETH-USD": {"news_event"}}`.
  Apply in `generate_signal()`.
- **Risk**: Low — single ticker-signal pair.

---

## 3. Implementation Batches

### Batch 1: SC-I-001 + CROSS-001 (2 files, tightly coupled)
1. `portfolio/signal_engine.py` — add `regime_gated_override` to `_weighted_consensus()`
2. `portfolio/outcome_tracker.py` — switch to `_raw_votes`
3. Tests for both fixes

### Batch 2: OR-I-001 (1 file)
1. `portfolio/main.py` — replace context manager with manual executor lifecycle
2. Test verifying loop continues after timeout

### Batch 3: news_event ETH gating (1 file)
1. `portfolio/signal_engine.py` — add `_TICKER_DISABLED_SIGNALS`
2. Test for per-ticker gating

### Batch 4: Full test suite + cleanup
1. Run full parallel test suite
2. Fix any regressions
3. Update SYSTEM_OVERVIEW.md

---

## 4. Risk Assessment

- **Batch 1**: Low risk — additive parameter, fallback behavior unchanged
- **Batch 2**: Medium risk — executor lifecycle change, but well-understood Python pattern
- **Batch 3**: Low risk — additive gating dict, no behavioral change for non-ETH tickers
