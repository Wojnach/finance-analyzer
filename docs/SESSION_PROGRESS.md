# Session Progress — Auth-Failure Bypass + Contract Tightening (2026-04-16)

## Status: MERGED (pending)

Three consecutive overnight Layer 2 outages (Apr 14-16): each day's 04:00-08:00 CET
window produced no Layer 2 invocations. Root cause: `claude -p` OAuth session expired,
claude returned exit 0 with "Not logged in" on stdout, and three direct `subprocess.run`
call sites bypassed `claude_gate.detect_auth_failure`. `iskbets._parse_gate_response`
additionally defaulted `approved=True` on parse miss — a real safety gap for warrant
trades, not just a detection gap. The `LAYER2_JOURNAL_GRACE_S = 60m` predated T3's
15-min subprocess cap, so the journal-activity contract didn't fire for 60+ minutes
post-trigger, losing the detection signal overnight.

### Shipped (branch improve/auto-session-2026-04-16)

1. `9722a0f docs: improvement plan for auto-session 2026-04-16`
2. `15ab78e fix(auth): route bigbet/iskbets/analyze through detect_auth_failure`
   - `bigbet.invoke_layer2_eval` → `(None, "")` + critical entry on auth fail
   - `iskbets.invoke_layer2_gate` → `approved=False` overrides default-approve
   - `analyze.run_analysis` → user-visible re-login hint + critical entry
   - 5 new tests in `tests/test_auth_failure_bypass.py`
3. `93a032f fix(contract): tighten LAYER2_JOURNAL_GRACE_S 60m -> 18m (BUG-202)`
   - 15m (T3 cap) + 3m slack. Pin test prevents silent widening.
4. `c4b3f45 chore: monotonic clock for elapsed, log silent excepts (BUG-203-205)`
   - `agent_invocation._agent_start` → `time.monotonic()` for elapsed math
   - qwen3 GPU reaper + dashboard market_health → `logger.debug(exc_info=True)`
5. `ef5f6ae style: ruff cleanup scripts/verify_kronos.py (SIM105, F541, E741)`

### Tests
- All 5 new auth-bypass tests pass
- 12 `test_layer2_journal_contract` tests pass (1 new pin + 11 existing)
- 61/62 `test_agent_invocation` pass (1 pre-existing fallback-to-bat failure unrelated)
- 97/97 `test_dashboard` pass
- 150/150 `test_bigbet + test_iskbets + test_analyze + test_claude_gate` pass

### What's next
- User must re-authenticate Claude CLI interactively (`claude` in terminal) — the
  code changes detect and surface auth failures but cannot refresh the OAuth token.
- After re-auth, restart `PF-DataLoop` via `schtasks /run /tn "PF-DataLoop"`.
- Monitor `data/critical_errors.jsonl` over the next 24h to confirm the journal
  contract fires at 18 min and auth failures from bigbet/iskbets/analyze paths
  record to the journal.

### Blockers
- None on the code side. Pending interactive OAuth re-login from the user.

---

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

# Session Progress — Auto-Improve 2026-04-15

## Status: COMPLETE (merged + pushed)

Autonomous improvement session: 5 phases, 8 commits, 28 files changed.

### What shipped
- **BUG-196: Absolute path resolution** — 6 modules (`microstructure_state`, `fear_greed`, `seasonality`, `linear_factor`, `signal_weight_optimizer`, `train_signal_weights`) used fragile `Path("data/...")` relative paths. All converted to `Path(__file__).resolve().parent.parent / "data"`.
- **BUG-197: DRY trigger sustained gate** — Duplicated sustained-debounce logic in `trigger.py` (signal flip + sentiment reversal) extracted into `_update_sustained()` helper.
- **BUG-198: Signal registry import caching** — Failed signal imports retried every 60s cycle (35 warnings/cycle). Added sentinel-based caching with 5-min TTL cooldown.
- **BUG-199: Dead timestamp code** — Removed unused `ts_str_clean` variable in `agent_invocation.py`; simplified to Python 3.12's native `fromisoformat()`.
- **12 new tests** — 8 for `_update_sustained` (count/duration gates, reset, independence), 4 for import caching (sentinel, cooldown, retry, clear).
- **Ruff cleanup** — 9 unused imports removed (F401), 8 violations fixed (UP035, SIM102, SIM105, SIM118, E731, I001). Violations reduced 67→59.

### Test results
- 139 targeted tests: all pass
- 7046 full suite: all pass (36 pre-existing failures unchanged)

### What's next
- IC-based dynamic signal weighting (highest impact from research session)
- MSTR BTC-proxy consensus
- HMM regime detection
- Per-ticker signal gating implementation

### 2026-04-15 10:57 UTC | fix/bug178-instrumentation-and-timeout
e2ee124 docs(plans): BUG-178 instrumentation + ticker pool timeout bump
docs/plans/2026-04-15-bug178-instrumentation-timeout.md

### 2026-04-15 11:16 UTC | fix/bug178-instrumentation-and-timeout
afe34ee feat(bug178): phase-level timing inside generate_signal post-dispatch
portfolio/main.py
portfolio/signal_engine.py
tests/test_phase_log.py

### 2026-04-15 11:18 UTC | fix/bug178-instrumentation-and-timeout
3655c1d perf(accuracy_stats): in-memory TTL cache for signal_utility
portfolio/accuracy_stats.py
tests/test_signal_utility_cache.py

### 2026-04-15 11:19 UTC | fix/bug178-instrumentation-and-timeout
f4719f0 fix(main): bump _TICKER_POOL_TIMEOUT 180 → 360 with 2026-04-15 rationale
portfolio/main.py

### 2026-04-15 11:20 UTC | fix/bug178-instrumentation-and-timeout
4811ce6 docs(session-progress): BUG-178 timeout + instrumentation session notes
docs/SESSION_PROGRESS.md

### 2026-04-15 11:21 UTC | fix/bug178-instrumentation-and-timeout
27e6dd2 style(tests): ruff B007 — use .values() instead of unused key in utility cache test
tests/test_signal_utility_cache.py

### 2026-04-15 11:24 UTC | fix/bug178-instrumentation-and-timeout
4ad689b fix(review): address 3 adversarial-review findings
portfolio/accuracy_stats.py
portfolio/signal_engine.py
tests/conftest.py
tests/test_phase_log.py

### 2026-04-15 11:25 UTC | fix/bug178-instrumentation-and-timeout
ced95ff docs(accuracy): correct cache-invalidation comment cadence (6h → daily)
portfolio/accuracy_stats.py
