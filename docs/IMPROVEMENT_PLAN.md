# Improvement Plan — 2026-05-31 Auto Session

**Branch:** `improve/auto-session-2026-05-31`
**Sources:** FGL synthesis (2026-05-30, 60 findings) cross-referenced with 5 deep-exploration
agents on current main HEAD (`db2967be`). Verified against live code — no findings carried
from prior plans without source re-confirmation.

---

## 1. Verified Bugs (ordered by severity)

| ID | File | Line | Bug | Severity |
|----|------|------|-----|----------|
| B01 | warrant_portfolio.py | 100 | No knockout floor: 5× warrant past −20% underlying yields negative per-unit value → corrupts drawdown + VaR + circuit breaker | P0 |
| B02 | agent_invocation.py | 1622 | `status="failed"` sends Telegram but writes NO journal stub (unlike `incomplete` which does) → invisible to loop_contract | P0 |
| B03 | loop_contract.py | 372 | `"failed"` not in `_KNOWN_FAILURE_STATUSES` → double-invisible: no stub AND contract checker doesn't suppress | P0 |
| B04 | main.py | 1098 | IC cache refresh: `_run_cycle_id % 60 == 30` with 600s cadence = 10h, not "≈60 min" as comment says. Was correct at old 60s cadence | P1 |
| B05 | data_collector.py | ~168 | `fetch_vix` calls `yf.Ticker("^VIX")` without `_yfinance_lock` — thread-unsafe concurrent access with ticker workers | P1 |
| B06 | data_collector.py | ~160 | `alpaca_klines` records circuit-breaker failure on empty data (weekends) → false OPEN after 5 empty weekends | P1 |
| B07 | agent_invocation.py | 331 | `_extract_ticker` hardcoded fallback to "XAG-USD" for non-matching triggers → wrong ticker for BTC/MSTR decision feedback | P1 |
| B08 | market_timing.py | 244 | `_is_agent_window()` skips Swedish holidays → L2 invoked, Avanza orders fail silently on Midsummer Eve etc | P1 |
| B09 | dashboard/app.py | 1897 | `/api/market-health` handler logs "mstr endpoint error" (copy-paste from `api_mstr()`) | P1 |
| B10 | accuracy_stats.py | 962-976 | `blend_accuracy_data` directional: picks higher sample count instead of blending → masks recent directional degradation | P1 |
| B11 | risk_management.py | 465 | `compute_probabilistic_stops` uses 252 annualization for crypto/metals (should be 365) → understated stop probability | P1 |
| B12 | risk_management.py | 729 | Module-level `CORRELATED_PAIRS = get_correlated_pairs()` crashes entire import if correlation_priors fails | P1 |
| B13 | equity_curve.py | 419 | `_pair_round_trips` uses `list.pop(0)` — O(n²) on many partial fills. Should use deque | P2 |
| B14 | journal.py | 23-40 | `load_recent` reads entire JSONL with no tail optimization — O(n) on months of data | P2 |
| B15 | signal_engine.py | 1094 | `REGIME_GATE_ONLY_SIGNALS` is empty frozenset — feature is dead code | P2 |
| B16 | signal_engine.py | 1519 | `_CROSS_HORIZON_PAIRS` missing 12h entry → dynamic weights never fire for 12h timeframe | P2 |
| B17 | signal_engine.py | ~2829 | `_confluence_score()` return value unused — dead code | P2 |
| B18 | digest.py | 58 | Tail-500 limit can undercount in high-activity periods (~600 invocations/4h) | P2 |
| B19 | trade_guards.py | 381 | C4 wiring check fires every cycle on quiet days → log spam | P2 |
| B20 | price_source.py | 240 | yfinance fallback has no staleness metadata on returned DataFrame → consumers can't detect stale data | P2 |

## 2. Not In Scope (too risky, architectural, or deferred)

- P0-1/P0-2 Avanza session/stop-loss: requires Avanza BankID re-auth + real-money path testing
- Theme B1 cross-process atomic-RMW: architectural, needs dedicated session
- Theme B4 EOD-flat reconcile: complex metals logic, high blast radius
- Theme B5 reconstructed history: mstr_mnav_discount/stablecoin_supply_ratio methodology
- Avanza dashboard worker watchdog: complex threading, risk of making things worse
- trade_guards disk reads optimization: beneficial but not urgent, no functional bug

## 3. Batch Execution Order

### Batch 1 — P0 fixes + critical P1 (4 files, 5 bugs)

1. **warrant_portfolio.py** (B01): Add `max(0.0, ...)` clamp on `current_implied_sek`
2. **agent_invocation.py** (B02, B07): Write `failed` journal stub mirroring `incomplete` path; fix `_extract_ticker` to return `None` on no match and handle upstream
3. **loop_contract.py** (B03): Add `"failed"` to `_KNOWN_FAILURE_STATUSES`
4. **main.py** (B04): Fix IC cache refresh modulo (`% 6 == 3` for ~60min at 600s cadence)

**Impact:** warrant drawdown accuracy, Layer 2 failure visibility, IC cache freshness.
**Risk:** Low. All changes are additive guards or constant corrections.

### Batch 2 — P1 data/safety fixes (5 files, 5 bugs)

5. **data_collector.py** (B05, B06): Acquire `_yfinance_lock` in `fetch_vix`; don't circuit-break on empty Alpaca data
6. **market_timing.py** (B08): Add `is_swedish_market_holiday()` check to `_is_agent_window()`
7. **dashboard/app.py** (B09): Fix copy-paste log message in `/api/market-health`
8. **risk_management.py** (B11, B12): Fix annualization constant; lazy-init CORRELATED_PAIRS
9. **journal.py** (B14): Use `load_jsonl_tail` in `load_recent` with sensible limit

**Impact:** Thread safety, data-source resilience, Swedish holiday guard, dashboard logging.
**Risk:** Low-medium. `_yfinance_lock` change needs careful review of all yfinance call paths.

### Batch 3 — P1 accuracy + P2 performance (4 files, 3 bugs)

10. **accuracy_stats.py** (B10): Fix `blend_accuracy_data` to blend directional accuracy like overall
11. **equity_curve.py** (B13): Replace `list` with `collections.deque` in `_pair_round_trips`
12. **price_source.py** (B20): Add `_source` and `_stale` columns to yfinance-fallback DataFrame
13. **trade_guards.py** (B19): Guard C4 wiring check with `_wiring_confirmed` flag

**Impact:** Accuracy gate correctness, round-trip perf, data provenance, log noise.
**Risk:** Low. Accuracy blending change needs test verification.

### Batch 4 — Dead code + signal quality (2 files, 4 bugs)

14. **signal_engine.py** (B15, B16, B17): Remove `REGIME_GATE_ONLY_SIGNALS` dead set; add 12h to `_CROSS_HORIZON_PAIRS`; remove `_confluence_score`
15. **digest.py** (B18): Increase tail limit from 500 to 2000

**Impact:** Code cleanliness, 12h dynamic weights, digest accuracy.
**Risk:** Very low. Pure cleanup + constant change.
