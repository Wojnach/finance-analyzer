# Adversarial Review Synthesis — 2026-04-21

**Date**: 2026-04-21 (supersedes 2026-04-20 review)
**Methodology**: Dual review — independent Opus 4.6 deep-read + 8 parallel code-reviewer agents
**Scope**: Full codebase across 8 subsystems
**Duration**: ~8 minutes parallel agent execution + deep manual review

## Review Summary

| Metric | Value |
|--------|-------|
| Subsystems reviewed | 8 |
| Total raw findings | ~130 |
| After dedup + cross-critique | ~95 unique |
| P0 (Critical) | 21 |
| P1 (High) | 48 |
| P2 (Medium) | 44 |
| P3 (Low) | 14 |

---

## Top 15 P0 Findings (Ranked by Impact x Probability)

### 1. SIG-001: `_get_regime_gated` Returns Horizon-Only Set, Not Union with _default
**Subsystem**: signals-core | **Confirmed by**: Track A + Track B
**File**: `portfolio/signal_engine.py:699-706`

`_get_regime_gated(regime, horizon)` returns `regime_dict[horizon]` when a horizon key
exists, skipping the `_default` union. In ranging/3h, 11 signals that should be HOLD
(trend, ema, structure, etc. with 35-42% accuracy) are voting in every consensus.

**Every ranging-regime 3h/4h cycle is affected. Live production bug.**

### 2. IND-001 / SIG-012: Utility Boost Inflates Accuracy Above Gate Threshold
**Subsystem**: signals-core + cross-cutting | **Confirmed by**: Track A + Track B independently
**File**: `portfolio/signal_engine.py:2974-2985`

Utility boost modifies `accuracy_data[sig]["accuracy"]` by up to 1.5x BEFORE
`_weighted_consensus` checks the accuracy gate. A 45% signal with positive returns
gets boosted to 67.5% and sails past the 47% gate.

### 3. RISK-001: Main Loop save_state Overwrites Layer 2 Trade Decisions
**Subsystem**: portfolio-risk | **Confirmed by**: Track A
**File**: `portfolio/portfolio_mgr.py:107-112` + `main.py:767-768`

Read-modify-write gap of 200-360 seconds with no lock. Layer 2 trade decisions silently
rolled back by the next main loop cycle's `save_state`.

### 4. RISK-002: Trade Guards Are Permanently Inert (Zero Callers)
**Subsystem**: portfolio-risk | **Confirmed by**: Track A
**File**: `portfolio/trade_guards.py` (entire module)

`record_trade()` has zero production callers. All cooldown, loss escalation, and rate
limit guards are dead code.

### 5. IND-002: Drawdown Circuit Breaker Exception = No Protection
**Subsystem**: cross-cutting | **Found by**: Track B only
**File**: `portfolio/agent_invocation.py:396-397`

`except Exception: logger.warning("drawdown check failed (proceeding)")` — the single
most important safety gate fails OPEN on any exception.

### 6. ORCH-001: invoke_agent Bypasses claude_gate Kill Switch
**Subsystem**: orchestration | **Found by**: Track A
**File**: `portfolio/agent_invocation.py:492-498`

Layer 2 calls `Popen` directly, bypassing `claude_gate.py`'s kill switch, rate limiter,
and in-process serialization lock. `CLAUDE_ENABLED = False` does not stop Layer 2.

### 7. ORCH-003: taskkill /T May Kill Main Loop (No Process Group Isolation)
**Subsystem**: orchestration | **Found by**: Track A
**File**: `portfolio/agent_invocation.py:492`

`Popen` without `CREATE_NEW_PROCESS_GROUP` means `taskkill /F /T` on the Claude CLI
process can follow the parent chain and kill the main loop.

### 8. INFRA-001/002: JSONL Cross-Process Locking Is Broken on Windows
**Subsystem**: infrastructure | **Confirmed by**: Track A
**File**: `portfolio/file_utils.py:219-248`

`msvcrt.locking` provides per-CRT-handle advisory locks, NOT kernel-level cross-process
exclusion. Main loop + metals loop writing to same JSONL files can interleave.

### 9. MET-003: EOD Sell at bid=0 Sends Market Order at Price Zero
**Subsystem**: metals-core | **Found by**: Track A
**File**: `data/metals_loop.py:2059`

When `bid <= 0` (API failure), `emergency_sell(page, key, pos, 0)` executes a live
market sell. Real money at risk.

### 10. AVZ-001: CONFIRM Order Double-Execution on Crash
**Subsystem**: avanza-api | **Found by**: Track A
**File**: `portfolio/avanza_orders.py:122-142`

If `_execute_confirmed_order` raises after status=confirmed but before save, the next
cycle re-reads pending status and re-executes. Two fills possible.

### 11. AVZ-002: CONFIRM Has No Order ID Binding
**Subsystem**: avanza-api | **Found by**: Track A
**File**: `portfolio/avanza_orders.py:196-198`

Any "CONFIRM" message fires against the most-recent pending order regardless of which
order the user intended. Wrong order executed.

### 12. MET-001: 5x Leverage Stop-Loss 3x Too Tight
**Subsystem**: metals-core | **Found by**: Track A
**File**: `portfolio/fin_snipe_manager.py:61`

`HARD_STOP_CERT_PCT = 0.05` = 1% underlying move at 5x. Project rule says -15%+ for
5x certs. Normal intraday silver volatility triggers this every session.

### 13. MOD-002: structure.py SELL Near 52-Week Low (Inverted Semantics)
**Subsystem**: signals-modules | **Found by**: Track A
**File**: `portfolio/signals/structure.py:77`

Near-52-week-low producing SELL votes = persistent wrong-direction bias during every
recovery scenario. Active for all 5 instruments.

### 14. IND-003 / INFRA-010: Dashboard CORS * + Auth Token in URL
**Subsystem**: infrastructure + cross-cutting | **Confirmed by**: Track A + Track B
**File**: `dashboard/app.py:44-49, 675-676`

`Access-Control-Allow-Origin: *` + `?token=` in URL = portfolio data accessible from
any website, token leaked in logs/history.

### 15. DATA-001: Fear & Greed Crashes on Empty API Response
**Subsystem**: data-external | **Found by**: Track A
**File**: `portfolio/fear_greed.py:100`

`body["data"][0]` with no guard. alternative.me returns `{"data": []}` during
maintenance, crashing the F&G voter silently.

---

## Key P1 Findings (48 total, grouped by theme)

### Financial Safety (12 findings)
- RISK-004: Warrant P&L can go negative (should floor at 0)
- RISK-005: Drawdown blind when price feed stale (reads 0%)
- RISK-006: Position size uses cash denominator, not portfolio value
- RISK-008: ATR stop from entry price, not trailing
- AVZ-004: TOTP order path bypasses order lock
- AVZ-006: No max order size on TOTP path
- MET-005: Stop distance computed in cert-price space, not underlying
- MET-006: Knockout risk flags permanently disabled (financing_level=None)
- MET-010: Stale signal data used for fishing without age check
- ORCH-002: Layer 2 subprocess lacks stdin=DEVNULL
- ORCH-005: session_calendar ignores NYSE holidays
- DATA-003: Alpha Vantage never checks HTTP status codes

### Signal Accuracy (10 findings)
- SIG-005: Circuit-breaker relaxes high-sample gate incorrectly
- SIG-007: signal_history.py race between ticker threads
- SIG-010: Persistence filter cold-start seeds with MIN_CYCLES
- SIG-016: SignalDB omits neutral-outcome filter (inflated accuracy)
- MOD-004: claude_fundamental cache ts updated before results
- MOD-005: forecast prediction dedup TOCTOU race
- MOD-007: fibonacci permanent SELL above 161.8% extension
- MOD-008: orderbook spread_health HOLD voter dilutes confidence 17%
- IND-005: Regime exemption uses different accuracy source than gate
- IND-007: Single-link clustering produces transitive mega-clusters

### Concurrency (8 findings)
- IND-006: Agent state globals unprotected
- IND-004: Persistence filter leaks stale ticker state
- MET-004: _underlying_prices shared dict without lock
- INFRA-004: Dashboard cache thundering-herd
- INFRA-007: Job Object assignment failure silent
- INFRA-008: health.py write-back races under dashboard polling
- ORCH-004: Post-cycle exception increments crash counter for healthy loops
- ORCH-006: _check_recent_trade silently drops detection on file contention

### Project Rule Violations (6 findings)
- SIG-008: accuracy_snapshots uses raw read_text
- MET-008: fish_monitor raw file read
- INFRA-006: telegram_poller raw json.load on config
- INFRA-009: reporting imports private _atomic_write_json
- DATA-008: social_sentiment uses print() not logger
- DATA-010: crypto_macro raw open() for JSONL

---

## Recommended Fix Priority

### Immediate (before next trading session)
1. Fix `_get_regime_gated` to union `_default | horizon_specific` (SIG-001)
2. Change drawdown check to fail-closed (IND-002)
3. Guard `emergency_sell` against `bid=0` (MET-003)
4. Move utility boost AFTER accuracy gate (IND-001/SIG-012)
5. Add `stdin=DEVNULL` + `CREATE_NEW_PROCESS_GROUP` to invoke_agent (ORCH-002/003)

### This week
6. Wire `record_trade` into execution path (RISK-002)
7. Fix portfolio state race with `update_state` (RISK-001)
8. Replace `msvcrt.locking` with `LockFileEx` (INFRA-001/002)
9. Route invoke_agent through claude_gate (ORCH-001)
10. Fix `HARD_STOP_CERT_PCT` to 0.15 (MET-001)
11. Add order ID binding to CONFIRM flow (AVZ-002)
12. Persist status before executing confirmed orders (AVZ-001)

### Next sprint
13-25. Fix remaining P1 findings by category.

---

## Cross-Critique Notes

### Strengths of Agent Track
- Extremely thorough within-subsystem coverage (112 findings across 8 agents)
- Found subtle financial bugs (AVZ double-execution, MET bid=0) requiring deep API understanding
- Good at identifying project rule violations and dead code

### Strengths of Independent Track
- Found 3 cross-cutting P0s (IND-001, IND-002, IND-003) that no agent caught
- IND-005 (accuracy source ordering) requires understanding mutation order across 200+ lines
- Cross-subsystem interaction bugs are invisible to single-subsystem agents

### Combined advantage
- ~30% more unique findings than either track alone
- P0 findings from Track B (5) had zero overlap with Track A P0s (18) until cross-referencing
  revealed SIG-012 = IND-001 and INFRA-010 = IND-003
- The dual approach catches both "deep within-module" and "shallow cross-module" bugs

---

## Comparison with Previous Reviews

| Review Date | P0 | P1 | P2 | Total | Notes |
|------------|----|----|-----|-------|-------|
| 2026-04-10 | 4 | 17 | 9 | 30 | First adversarial review |
| 2026-04-17 | 12 | 23 | 18 | 53 | Codex rounds 1-13 |
| 2026-04-18 | 27 | 40 | 51 | 118 | First dual review |
| 2026-04-19 | 19 | 36 | 42 | 97 | Focused on fixes from 04-18 |
| 2026-04-20 | 22 | 41 | 38 | 101 | Post-fix validation |
| **2026-04-21** | **21** | **48** | **44** | **~130 raw / 95 unique** | **Opus 4.6 + 8 parallel agents** |

Persistent P0s across multiple reviews (still unfixed):
- SIG-001 (regime gating): **NEW** in this review
- RISK-001 (state overwrite): Identified 04-20, still open
- RISK-002 (trade guards inert): Identified 04-20, still open
- INFRA-001/002 (JSONL locking): Identified 04-17, still open
