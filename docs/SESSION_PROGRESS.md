# Session Progress — BUG-178 Timeout + Instrumentation (2026-04-15)

## Status: IN REVIEW

Telegram at 10:34 fired `LOOP ERRORS (884s cycle) 5 ticker(s) failed entirely` plus
`LOOP CONTRACT (main) — 1 critical violation: min_success_rate 0%`. Investigation traced
this to the 180s `_TICKER_POOL_TIMEOUT` (dropped 2026-04-09 after fingpt daemon retirement)
firing on legitimate slow work now that the ticker path has grown with vix_term_structure,
DXY intraday, per-ticker gating, directional accuracy, and fundamental correlation signals.
Zombie threads complete 330-525s into the cycle, all 5 within ~10s — shared-resource wait
pattern, not stuck work.

### Shipping (branch fix/bug178-instrumentation-and-timeout)

1. `e2ee124` docs(plans): instrumentation + timeout plan
2. `afe34ee` feat(bug178): phase-level timing inside generate_signal post-dispatch
   - New `_phase_log_per_ticker`, `_record_phase`, `get_phase_log`, `_reset_phase_log` in
     signal_engine.py with lock-guarded per-ticker list
   - Phases recorded: regime_gate, acc_load, utility_overlay, weighted_consensus,
     penalties, linear_factor, consensus_gate
   - `[SLOW-PHASE]` WARNING when any single phase > 2.0s (gated, zero noise on fast cycles)
   - BUG-178 pool-timeout handler and slow-cycle diagnostic both dump per-ticker phase
     breakdown so future failures show WHICH phase burned the time
   - 10 new tests in tests/test_phase_log.py (all green)
3. `3655c1d` perf(accuracy_stats): in-memory TTL cache for signal_utility
   - signal_utility walked the full signal log (~6.3K snapshots / ~92K ticker rows) on
     every ticker, every cycle. Cold cost: ~3.6s. With 5 parallel threads contending for
     the 108MB signal_log.db file cache, this legitimately blocked.
   - Split into public cache-wrapper + private `_compute_signal_utility` so explicit-
     entries callers (tests) bypass the cache. 300s TTL matches LLM rotation period.
   - Swap-outside-compute pattern: lock held only for the (time,value) swap, never for
     the 3.6s compute. At most one double-compute on TTL-boundary race.
   - 9 new tests in tests/test_signal_utility_cache.py (all green)
4. `f4719f0` fix(main): _TICKER_POOL_TIMEOUT 180 → 360 with full 2026-04-15 rationale
   - 2.8x p50-slow, 0.7x p95-slow; 240s margin inside 600s cadence
   - Comment rewrite preserves 120→500→180→360 timeline + why for each
   - Points to phase log + plan doc for future debugging

### Tests
- 41 test_accuracy_stats.py tests pass (covers signal_utility correctness)
- 10 new test_phase_log.py tests pass
- 9 new test_signal_utility_cache.py tests pass
- 119 non-tmp_path tests pass in the broader suite; 16 pre-existing Windows-tmp errors
  unrelated to this change
- Full-file `test_signal_engine.py` hangs pre-existing on main and on this branch;
  individual test classes that cover the changed code all pass

### Deferred
- Windows `tasklist /FI "PID eq X"` 5s subprocess timeouts in llama_server.py — real but
  orthogonal. Defensive fix proposed; not shipped with this PR to keep scope tight.

---

# Session Progress — Auto-Improvement Session (2026-04-14)

## Status: SHIPPED

Deep codebase exploration by 4 parallel agents. Cross-referenced findings with live
accuracy data (accuracy_cache.json, ticker_signal_accuracy_cache.json). Implemented
5 bug fixes, 1 architecture fix, and 37 new tests.

### What shipped (5 commits on improve/auto-session-2026-04-14)

1. `2884075` docs: improvement plan for auto-session 2026-04-14
2. `e45faa1` fix(signals): expand per-ticker blacklist + disable oscillators globally
   - BUG-192: Added XAG-USD (ministral 18.9%, credit_spread_risk 21.2%, metals_cross_asset 39.8%),
     XAU-USD (ministral 43.2%), MSTR (credit_spread_risk 6.5%) to `_TICKER_DISABLED_SIGNALS`
   - BUG-193: Added oscillators to global DISABLED_SIGNALS (35-43% across all tickers, 5065 samples)
   - BUG-194: Gated sentiment at 3h/4h in "unknown" regime (33.8%, 3629 samples)
   - 15 new tests in test_signal_engine.py (69 total, was 54)
3. `9bfafc5` fix(cache): release dogpile loading key on enqueue exception (BUG-191)
   - Prevents 120s stale-signal windows after GPU load failures
   - 2 new tests in test_shared_state_cache.py (15 total, was 13)
4. `0e76aa7` test(accuracy): add 22 tests for core accuracy calculation functions
   - TestVoteCorrect (7), TestSignalAccuracy (7), TestConsensusAccuracy (5), TestPerTickerAccuracy (3)
   - accuracy_stats.py now has 41 tests (was 19)
5. `aacbcd3` fix(signals): MSTR blacklist + correlation multi-group penalty fix
   - BUG-195: Added MSTR macro_regime, trend, volatility_sig to blacklist
   - Fixed correlation penalty: multi-group signals now get harshest (min) penalty, not last-wins
   - 4 new tests (TestMSTRSignalBlacklist, TestCorrelationPenaltyMultiGroup)

### Test impact
- 125 tests pass across 3 changed test files (69 + 41 + 15)
- 37 new tests added this session

### Deferred (documented in IMPROVEMENT_PLAN.md)
- IC-based signal weighting (P2)
- HMM regime detection (P2)
- MSTR-BTC proxy signal (P1)
- Metals position sync on startup
- Cancel-sell-rearm atomicity
- prune_jsonl streaming rewrite

### Previous session (2026-04-12)
Adversarial review round 5: 19 findings (3 P0, 10 P1, 7 P2).
Top priority: wire check_drawdown(), wire record_trade(), POSITIONS lock.

### Previous session (2026-04-10)
Per-ticker directional accuracy + raised directional gate to 40%.
