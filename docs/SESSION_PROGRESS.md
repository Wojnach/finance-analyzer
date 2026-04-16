# Session Progress — Health Check + XAU Trigger Noise Triage (2026-04-16 afternoon)

## Status: READ-ONLY DIAGNOSTIC — NO CODE CHANGES

Session started ~13:45 CET; snapshot written ~15:25 CET (~1h35m elapsed). User prompt:
"can we check that everything is working as intended all the loops and that the system
is capable of trading". Crash-recovery snapshot per user request: "document everything
you are doing locally in case you or any subagents crash".

### Verified state (as of 13:20 UTC / 15:20 CET)

| Component | State | Evidence |
|---|---|---|
| PF-DataLoop | Running | heartbeat 09:13 UTC cycle 81, fresh <=6min |
| PF-MetalsLoop | Running | swing decisions every ~60s |
| Claude CLI | v2.1.107 | `C:\Users\Herc2\.local\bin\claude.exe` |
| Layer 2 journal | alive | 1,616 entries, last 13:20 UTC |
| SwingTrader | idle | 6,155 SEK cash, 0 positions, 12 session trades, correctly gating XAU conf 0.57 and XAG HOLD |
| Patient | ok | 497K SEK cash, ETH 3.47sh @ $2,332 (bought 10:13 UTC today) |
| Bold | ok | 224K SEK cash + BTC 0.201sh @ $74,556 + ETH 4.52sh @ $2,319 |
| PF-Dashboard | Disabled | port 5055 not listening (not trade-critical) |

### Diagnostic correction made mid-session

I initially mis-read file mtime on `data/layer2_journal.jsonl` and concluded Layer 2
was dead all day. WRONG. Layer 2 has been actively producing journal entries since the
10:32 UTC loop restart — 20+ entries written today. The mtime mislead was corrected
after seeing fresh Telegram analysis traffic starting 10:05 UTC.

### Outstanding unresolved items (not touched this session)

1. **Critical error `2026-04-16T05:58:09 contract_violation`** still marked unresolved
   in `data/critical_errors.jsonl`. Gap self-healed post-10:32 restart. Could append
   a resolution entry but I did not — user has not instructed.
2. **Overnight Layer 2 silent-failure pattern** (Apr 14 timeout, Apr 15 error, Apr 16
   silent) is a known recurring issue. Each morning recovers. Today's recovery was
   derailed by a fix-agent asking "proceed or investigate?" (see `data/agent.log`
   tail) — against the saved `feedback_be_decisive` memory. Not investigated further.
3. **PF-Dashboard scheduled task Disabled** — leave or re-enable? User has not asked.

### Open decision awaiting user input

User asked: "Want me to raise the XAU trigger-voter minimum to cut this down?"
Today: 10 XAU triggers, all returned HOLD. Raw confidence 30-48% with 5/28 voters
at floor (3B/2S). Five options presented with pros/cons:

| Opt | Change | Blast radius |
|---|---|---|
| A | Raise `MIN_VOTERS_STOCK` 3->6 for metals only | signal_engine.py; affects SwingTrader + autonomous + trigger |
| B | Add trigger-level confidence floor >=0.50 | trigger.py section 1 only; doesn't affect sustained flips |
| C | Per-ticker re-trigger cooldown (90 min) | trigger.py + trigger_state.json schema |
| D | Fix consensus: require `abs(buy-sell) >= max(2, voters/3)` | signal_engine.py global; tests + regression |
| E | Do nothing | zero |

**I recommended C or E.** User has NOT picked. No implementation has started.

Key risks to forewarn whichever option we take:
- XAU near ATH ($4,820) — any threshold raise delays breakout alerts
- F&G-23 extreme-fear BUY entries today were at 44% confidence — a 50% floor would
  have blocked Bold's BTC/ETH fills
- Semantic drift between `signal_log` (consensus) and `trigger_state` (triggered_consensus)
  could cause `layer2_journal_activity` contract_violation false positives

### Telegram traffic audit (today through 13:25 UTC)

- 51 messages logged total; 26 sent to user, 25 logged-only
- Breakdown: 21 invocation (logged), 19 analysis (sent), 3 error, 2 digest,
  2 crypto_report, 2 health, 1 trade, 1 daily_digest
- Cadence ~1 visible every 12 min is driven by **signal triggers**, not a fixed timer
- **Primary noise source today: XAU whipsawing BUY/HOLD/SELL 10 times in 5h**

### Files touched this session

None. Pure read-only diagnostic. No git changes. No worktree created.

### If this session crashes — pick up here

1. Read this block of `docs/SESSION_PROGRESS.md`
2. Re-check `data/critical_errors.jsonl` for new entries since 13:25 UTC
3. Ask user to pick between Option C (per-ticker cooldown) and Option E (do nothing)
4. DO NOT auto-implement — user has not confirmed
5. If user picks C: create worktree `fa-xau-cooldown` on branch `fix/xau-trigger-cooldown`,
   modify `portfolio/trigger.py` section 1 to add `state["triggered_consensus_ts"][ticker]` check
   against a new `CONSENSUS_RETRIGGER_COOLDOWN_S = 90 * 60` constant, add tests in
   `tests/test_trigger.py`, then codex review -> merge -> restart loops

---

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

## Session 2026-04-16 — Accuracy degradation tracker

### Context
W15/W16 Tier-1 1d consensus collapsed from 52-56% to 36-41% (see
memory/project_accuracy_degradation_20260416.md). The 11 main-loop
runtime contracts in loop_contract.py check execution health, not
decision quality, so the collapse went undetected for two weeks.
accuracy_stats.py shipped a save_accuracy_snapshot/check_accuracy_changes
pair months ago but they were never wired up.

### Shipped (branch feat/accuracy-degradation, /fgl protocol)
0. `33e5847 docs(plan): accuracy degradation tracker`
1. **Codex pre-impl adversarial review** — 4 valid findings (P1#1, P1#2,
   P2#3, P2#4). All addressed in `16c8128`.
2. `bb8bba5 feat(accuracy): batch 1 — snapshot infra`
   - `save_accuracy_snapshot(extras=...)` for arbitrary scope blocks
   - `consensus_accuracy(days=...)` recent-window variant
   - `cached_forecast_accuracy()` 1h-TTL wrapper
   - `econ_dates.recent_high_impact_events(hours)` backward window
3. `2381dc2 feat(accuracy): batch 2 — degradation tracker module + tests`
   - `portfolio/accuracy_degradation.py` with check_degradation,
     save_full_accuracy_snapshot, daily summary builder, severity
     classifier. Throttle replays cached violations so ViolationTracker
     escalation works (Codex P1#2).
4. `1637960 feat(loop): batch 3 — wire degradation tracker into the main loop`
   - loop_contract.verify_contract() invariant #12
   - main.py post-cycle daily snapshot + summary _track() entries
   - check_degradation() always writes last_full_check_time after
     passing throttle/blackout gates (was a real bug)

### Tests
- 4 batches added 38 tests, all green
- 191 in the affected neighborhood (loop_contract, accuracy_stats,
  accuracy_compute_lock, econ_dates, forecast_accuracy)
- E2E verified: scripts/_e2e_degradation_check.py injects 7-day-old
  baseline + stubs current state to a clear collapse, asserts CRITICAL
  violation + throttle replay + daily summary preview

### Telegram formats
Contract path: piggybacks the existing
  `*LOOP CONTRACT (main)* — N critical violation(s)` style.
Daily summary: new `*ACCURACY DAILY*` body via category=daily_digest,
  consensus + forecast + LLM split lines, top drops, top gains.

### Anti-noise gates
- 100/100 sample minimums (historical/current)
- 6-day baseline minimum
- 24h per-signal Telegram cooldown (Violation list still includes ALL
  alerts so ViolationTracker keeps escalation count alive)
- 55-min hourly compute throttle with cached-violation replay
- FOMC/CPI/NFP ±24h blackout (forward AND backward)


## Auto-commit log (parallel sessions, post-commit hook)

### 2026-04-16 10:31 UTC | fix/bug178-accuracy-thundering-herd
492d478 fix(BUG-178): serialize accuracy cache-miss compute, add pf-restart helper
portfolio/accuracy_stats.py
portfolio/signal_engine.py
scripts/win/pf-restart.bat
scripts/win/pf-restart.ps1
tests/test_accuracy_compute_lock.py

### 2026-04-16 10:32 UTC | main
95d6823 fix(pf-restart): replace em-dashes with ASCII for PS5 compatibility
scripts/win/pf-restart.ps1

### 2026-04-16 12:55 UTC | feat/accuracy-degradation
33e5847 docs(plan): accuracy degradation tracker
docs/plans/2026-04-16-accuracy-degradation-tracker.md

### 2026-04-16 13:05 UTC | feat/accuracy-degradation
16c8128 docs(plan): address Codex pre-impl adversarial findings
docs/plans/2026-04-16-accuracy-degradation-tracker.md

### 2026-04-16 13:10 UTC | feat/accuracy-degradation
bb8bba5 feat(accuracy): batch 1 — snapshot infra for degradation tracker
portfolio/accuracy_stats.py
portfolio/econ_dates.py
portfolio/forecast_accuracy.py
tests/test_accuracy_snapshot_extras.py

### 2026-04-16 13:15 UTC | feat/accuracy-degradation
2381dc2 feat(accuracy): batch 2 — degradation tracker module + tests
portfolio/accuracy_degradation.py
tests/test_accuracy_degradation.py

### 2026-04-16 13:16 UTC | fix/accuracy-gating-20260416
b88e3ed docs(plan): accuracy gating reconfiguration plan
docs/PLAN.md

### 2026-04-16 13:18 UTC | feat/accuracy-degradation
1637960 feat(loop): batch 3 — wire degradation tracker into the main loop
portfolio/accuracy_degradation.py
portfolio/loop_contract.py
portfolio/main.py
tests/test_loop_contract_accuracy.py

### 2026-04-16 13:21 UTC | fix/accuracy-gating-20260416
fd504d4 fix(signals): revert recency weights + trim MSTR horizon-mismatched blacklist
portfolio/accuracy_stats.py
portfolio/signal_engine.py
tests/test_accuracy_cache_timestamps.py
tests/test_signal_engine.py
tests/test_signal_engine_core.py

### 2026-04-16 13:22 UTC | feat/accuracy-degradation
007ddc6 docs: batch 4 — degradation tracker session notes + e2e verification
docs/SESSION_PROGRESS.md
scripts/_e2e_degradation_check.py

### 2026-04-16 13:26 UTC | fix/accuracy-gating-20260416
04e0ae2 feat(signals): circuit breaker - relax accuracy gate when voter count drops below floor
portfolio/signal_engine.py
tests/test_signal_engine_circuit_breaker.py

### 2026-04-16 13:32 UTC | fix/accuracy-gating-20260416
898c38e feat(signals): horizon-specific per-ticker blacklist - prevents horizon-mismatch regressions
portfolio/signal_engine.py
tests/test_horizon_specific_blacklist.py

### 2026-04-16 13:34 UTC | fix/accuracy-gating-20260416
c3f0916 tools(signals): counterfactual replay + session progress log
data/consensus_replay_20260416.json
scripts/replay_consensus.py

### 2026-04-16 13:35 UTC | fix/accuracy-gating-20260416
4b89214 docs(progress): overwrite SESSION_PROGRESS.md with accuracy-gating batch state
docs/SESSION_PROGRESS.md

### 2026-04-16 13:37 UTC | feat/accuracy-degradation
a7c8c08 fix(accuracy): post-impl review — perf, stale guard, blackout reset
docs/SESSION_PROGRESS_2026-04-16_degradation_in_flight.md
portfolio/accuracy_degradation.py
portfolio/accuracy_stats.py
portfolio/econ_dates.py
tests/test_accuracy_degradation.py
tests/test_accuracy_snapshot_extras.py
