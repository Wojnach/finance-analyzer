# Adversarial Review Synthesis — 2026-04-30

**Date:** 2026-04-30
**Review #:** 9
**Method:** Dual-stream: 8 parallel code-reviewer agents + independent Opus review.
Cross-critique and convergence analysis performed on all findings.
**Codebase:** ~75K lines Python, 142+ modules, 5 Tier-1 instruments
**Focus:** Code changed since 2026-04-23 + re-verification of prior unfixed P0/P1s
**Total raw findings:** Agent stream: 55, Independent stream: 12
**After dedup, cross-critique, & false-positive rejection:** 42 unique findings

---

## Executive Summary

This review found **42 unique issues** after dedup and cross-critique. Three prior P0
findings remain unfixed (3% stop-proximity guard, TOTP order lock bypass, stale drawdown
prices). The most impactful new finding is a regime-gating bug that silently lets 20+
signals bypass ranging/trending gates at the 3h/4h horizon. A newly added signal
(`crypto_cross_asset`) is completely dead due to a wrong return key.

| Severity | Count | Financial Risk? |
|----------|-------|-----------------|
| P0 CRITICAL | 5 | Yes — wrong trades, dead signals, bypassed safety gates |
| P1 HIGH | 14 | Yes — degraded accuracy, wrong weights, race conditions |
| P2 MEDIUM | 16 | Moderate — stale data, code quality, I/O violations |
| P3 LOW | 7 | Low — documentation, performance, minor inconsistencies |

**Key theme this round:** New code (crypto swing, new signals) was added rapidly and has
several integration bugs (wrong return keys, missing imports, bypassed infrastructure).
The core signal engine has accumulated enough complexity that regime-gating and accuracy
interactions have emergent bugs.

---

## False Positives Rejected

| Finding | Reason for rejection |
|---------|---------------------|
| RISK-007 (f.tell() after for-loop) | Empirically verified: CPython f.tell() correctly returns EOF after `for line in f` loop |
| DATA-001 (data.crypto_data import fails) | Verified: Python namespace packages allow `from data.crypto_data import ...` without `__init__.py` |

---

## Top 15 Critical + High Findings

### 1. `_get_regime_gated` drops `_default` set for named horizons [SC-001]
**Subsystem:** signals-core | **Agent stream** | **Severity: P0** | **Confidence: 97%**

When `horizon="3h"` or `"4h"`, the function returns ONLY the horizon-specific override set
(4 signals) and drops the entire `_default` set (20+ signals). Result: `oscillators` (34-39%),
`forecast` (36%), `volatility_sig` (35%), and many other signals that should be force-HOLD
in ranging/trending regimes vote freely at 3h/4h. These are the system's worst-performing
signals voting without any safety gate.

**Impact:** Structural wrong-direction voting at the most frequently-run horizon.
**Fix:** Return `default_set | regime_dict[horizon]` (union), not `regime_dict[horizon]` alone.
**File:** `signal_engine.py:829-836`

---

### 2. `crypto_cross_asset` returns `"signal"` key — engine reads `"action"` [SIG-001]
**Subsystem:** signals-modules | **Both streams** | **Severity: P0** | **Confidence: 100%**

`compute_crypto_cross_asset_signal()` returns `{"signal": "BUY", ...}` but the signal engine
at line 1061 reads `result.get("action")` → always `None` → forced to HOLD. The entire
crypto cross-asset signal is permanently dead. Every other signal module uses `"action"`.

**Impact:** New signal contributes zero information to BTC/ETH consensus.
**Fix:** Change all `"signal"` keys to `"action"` in `crypto_cross_asset.py`.
**File:** `signals/crypto_cross_asset.py:196,218,258`

---

### 3. 3% stop-proximity guard still absent [AVZ-001] — UNFIXED since review #5
**Subsystem:** avanza-api | **Both streams** | **Severity: P0** | **Confidence: 100%**

No stop-loss placement function validates that the trigger price is ≥3% below the current
bid. This rule is documented as CRITICAL in both MEMORY.md and the Book of Grudges. For
5x leveraged warrants, a stop within 3% triggers on normal intraday noise.

**Impact:** Real money lost on premature stop-loss triggers.
**Fix:** Add bid-fetch + distance check in `avanza_session.place_stop_loss()`.
**File:** `avanza_session.py:715-806`

---

### 4. `_rescued` flag leaks across loop iterations [SC-003]
**Subsystem:** signals-core | **Agent stream** | **Severity: P1** | **Confidence: 90%**

In the `_weighted_consensus` loop, `_rescued` is set to `True` when a signal passes
directional rescue. If the NEXT signal passes the accuracy gate outright (takes the
`else` branch), `_rescued` is never reset to `False` — it inherits `True` from the
previous iteration and gets a spurious 0.7x weight penalty.

**Impact:** Correct signals get wrongly penalized 30%, reducing consensus quality.
**Fix:** Add `_rescued = False` at the top of each loop iteration.
**File:** `signal_engine.py:1984,1989,2016`

---

### 5. TOTP order path bypasses cross-process lock AND max-order-size guard [AVZ-002]
**Subsystem:** avanza-api | **Both streams** | **Severity: P0** | **Confidence: 95%**

`avanza_client._place_order` has no `avanza_order_lock` and no 50K SEK cap. The BankID
path in `avanza_session._place_order` has both. Simultaneous TOTP + BankID orders can
race with no coordination, and TOTP orders have no upper bound on size.

**Impact:** Race conditions on order placement, potential for oversized orders.
**Fix:** Add `avanza_order_lock` and size guard to `avanza_client._place_order`.
**File:** `avanza_client.py:326-350`

---

### 6. Drawdown circuit breaker still uses stale prices [RISK-003] — UNFIXED since review #5
**Subsystem:** portfolio-risk | **Both streams** | **Severity: P1** | **Confidence: 85%**

`check_drawdown` reads `agent_summary.json` prices with no staleness check. A flash crash
could present stale high prices, preventing the circuit breaker from firing.

**Fix:** Check `summary.get("ts")` — if stale >180s, treat as breached (fail-safe).
**File:** `risk_management.py:144-154`

---

### 7. Trade guard state has no cross-process lock [RISK-001]
**Subsystem:** portfolio-risk | **Agent stream** | **Severity: P1** | **Confidence: 92%**

`trade_guards.py` `record_trade` does bare `_load_state()` / `_save_state()` with no
mutex. Two concurrent Layer 2 subprocesses can clobber each other's cooldown records.

**Fix:** Add cross-process file lock (matching `avanza_order_lock` pattern).
**File:** `trade_guards.py:260-310`

---

### 8. Regime accuracy cache uses shared timestamp [SC-002]
**Subsystem:** signals-core | **Agent stream** | **Severity: P1** | **Confidence: 98%**

`write_regime_accuracy_cache` sets a single `"time"` key for all horizons. Writing
"3h" makes "1d" appear fresh even if never written or hours stale.

**Fix:** Use per-horizon timestamps: `cache[f"time_{horizon}"]`.
**File:** `accuracy_stats.py:1125-1136`

---

### 9. Circuit breaker relaxes high-sample tier gate [SC-006]
**Subsystem:** signals-core | **Agent stream** | **Severity: P1** | **Confidence: 85%**

The 50% accuracy gate for signals with 10,000+ samples exists to filter proven coin-flips.
The circuit-breaker relaxation subtracts up to 6pp from this gate too, allowing 44% signals
to pass — contradicting the documented rule that the high-sample tier is strict.

**Fix:** Don't subtract relaxation from `_ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD`.
**File:** `signal_engine.py:1960-1964`

---

### 10. Decay reaches 1× without resetting loss counter [RISK-002]
**Subsystem:** portfolio-risk | **Agent stream** | **Severity: P1** | **Confidence: 88%**

After 72h, the trade guard multiplier decays to 1× (identical to zero losses), but
`consecutive_losses` stays at 4. First new loss instantly jumps back to 8× — creating
confusing oscillation between fully-open and fully-locked.

**Fix:** Reset `consecutive_losses` to 0 when decay reaches 1×.
**File:** `trade_guards.py:91-93`

---

### 11. SQL accuracy skips neutral-outcome filter [SC-004]
**Subsystem:** signals-core | **Agent stream** | **Severity: P1** | **Confidence: 92%**

`signal_db.signal_accuracy()` counts any positive change as correct BUY. The JSONL path
filters `abs(change_pct) < 0.05%`. Accuracy is systematically overstated in the SQL path.

**Fix:** Add `ABS(change_pct) >= 0.0005` to SQL WHERE clauses.
**File:** `signal_db.py:270-271`

---

### 12. Hardcoded Avanza close time in `fin_fish.py` [MTL-002]
**Subsystem:** metals-core | **Agent stream** | **Severity: P1** | **Confidence: 90%**

`AVANZA_CLOSE_H, AVANZA_CLOSE_M = 21, 55` violates the explicit rule "Check API for
todayClosingTime — do NOT hardcode." DST transition shifts the actual close time.

**Fix:** Call Avanza session API for `todayClosingTime`, fall back to 21:55.
**File:** `fin_fish.py:196-197`

---

### 13. OFI double-recording in persist_state [MTL-001] — UNFIXED since review #6
**Subsystem:** metals-core | **Agent stream** | **Severity: P1** | **Confidence: 85%**

`persist_state()` re-calls `get_microstructure_state()` which re-appends OFI to history.
Every cycle double-counts the same value, compressing z-score variance.

**Fix:** Read cached state instead of re-invoking the full pipeline.
**File:** `microstructure_state.py:205-213`

---

### 14. 5% cert stop too tight for 5x leverage [MTL-004]
**Subsystem:** metals-core | **Agent stream** | **Severity: P1** | **Confidence: 88%**

`HARD_STOP_CERT_PCT = 0.05` = 5% cert stop for 5x warrants = ~1% underlying move.
Rules say "5x leverage certificates need -15%+ stops." Normal silver ATR triggers this.

**Fix:** Raise `HARD_STOP_CERT_PCT` to at least 0.15.
**File:** `fin_snipe_manager.py:61`

---

### 15. Layer 2 subprocess spawned without tree-kill protection [ORCH-001]
**Subsystem:** orchestration | **Agent stream** | **Severity: P1** | **Confidence: 95%**

`invoke_agent()` uses plain `subprocess.Popen` without `CREATE_NEW_PROCESS_GROUP` or
Job Object flags. `claude_gate.py` has the correct infrastructure but it's not used by
the main invocation path. Grandchild processes may survive timeout kills.

**Fix:** Add `**_popen_kwargs_for_tree_kill()` to the Popen call.
**File:** `agent_invocation.py:498`

---

## Additional P1/P2 Findings (ranked by impact)

| # | ID | Sub | Sev | Issue |
|---|-----|-----|-----|-------|
| 16 | DATA-002 | data-ext | P1 | `fear_greed.py` crashes on malformed API response (no KeyError guard) |
| 17 | DATA-003 | data-ext | P1 | `onchain_data._load_onchain_cache` doesn't use `_coerce_epoch` — TypeError on old caches |
| 18 | SC-007 | sig-core | P2 | `blend_accuracy_data` blends overall acc but picks directional acc winner-take-all |
| 19 | SC-005 | sig-core | P2 | `_BIAS_THRESHOLD=0.85` not updated after 80%/10 bias detector tuning |
| 20 | RISK-004 | pf-risk | P2 | Zero ATR produces stop at entry price — spurious trigger on illiquid assets |
| 21 | RISK-005 | pf-risk | P2 | Portfolio validator Check 8 uses all-time BUY cost — false errors after partial sell+rebuy |
| 22 | RISK-006 | pf-risk | P2 | `min_order_sek` default 500 ≠ documented 1000 SEK minimum |
| 23 | RISK-008 | pf-risk | P2 | `!=0` share filter admits negative shares in Monte Carlo VaR |
| 24 | DATA-004 | data-ext | P2 | `crypto_precompute.py` bypasses `http_retry` and rate limiter — raw requests.get |
| 25 | DATA-005 | data-ext | P2 | `mstr_precompute.py` no market-hours guard — stale weekend data written as current |
| 26 | DATA-006 | data-ext | P2 | `forecast_signal.py` Chronos v2 uses string keys on float columns — KeyError |
| 27 | MTL-005 | metals | P2 | `fish_monitor_smart.py` raw read_text() on live JSONL — torn line risk |
| 28 | MTL-006 | metals | P2 | EV for daily certs vs MINIs incomparable — ranking bias toward daily certs |
| 29 | AVZ-003 | avanza | P2 | `metals_avanza_helpers.delete_stop_loss` missing order lock + wrong 404 handling |
| 30 | AVZ-005 | avanza | P2 | CONFIRM matches most-recent order regardless of user intent |
| 31 | AVZ-007 | avanza | P2 | Order state not persisted before execution — crash → duplicate order |
| 32 | AVZ-008 | avanza | P2 | Session expiry check ignores EXPIRY_BUFFER_MINUTES |
| 33 | ORCH-002 | orch | P2 | Specialist log files leak: `cleanup_reports()` never called |
| 34 | ORCH-005 | orch | P2 | `_invoke_lock` serializes all Claude calls for 60s — causes BUG-178 timeouts |
| 35 | INFRA-001 | infra | P2 | `_loading_timestamps` not cleaned on KeyboardInterrupt path |
| 36 | INFRA-002 | infra | P2 | `api_mstr_loop` reads JSONL with raw open() — violates atomic I/O |
| 37 | INFRA-003 | infra | P2 | `telegram_poller._handle_mode_command` reads config with raw json.load() |
| 38 | DATA-008 | data-ext | P2 | `fx_rates.py` hammers Frankfurter API on out-of-range rate (no backoff) |
| 39 | DATA-009 | data-ext | P2 | `sentiment._majority_label` 1e-9 epsilon causes false neutral on FP residue |
| 40 | SIG-004 | sig-mod | P3 | ETH/BTC sub-signal always HOLD for BTC — effective voter count 4 not 5 |
| 41 | DATA-007 | data-ext | P3 | `ml_signal._load_model` not thread-safe (disabled signal, low risk) |
| 42 | INFRA-004 | infra | P3 | `message_throttle._send_now` TOCTOU race → duplicate Telegram on crash |

---

## Cross-Critique Results

### Agent findings confirmed by independent review
- AVZ-001 (3% stop guard): Independently verified — `place_stop_loss` has no bid-distance check
- SIG-001 (wrong return key): Independently verified via grep — `"signal"` vs `"action"` mismatch confirmed
- RISK-001 (trade guard race): Confirmed — no lock in `record_trade`
- SC-001 (regime gate drops _default): Logic trace confirmed — union semantics needed

### Agent findings REJECTED by independent review
- RISK-007 (f.tell() offset bug): **FALSE POSITIVE** — CPython f.tell() returns correct position after for-loop (verified empirically)
- DATA-001 (crypto_data import fails): **FALSE POSITIVE** — Python namespace packages work without __init__.py, and both functions exist in data/crypto_data.py (verified with import test)

### Independent findings not covered by agents
- Chronos v2 string-vs-float column keys (DATA-006) — agents did not read forecast_signal.py deeply enough
- Confirmed post-crash zero-sleep fix (ORCH-004) — independently verified the fix is correct

### Disagreements resolved
- MTL-003 (OFI sign convention): Agent correctly determined the sign convention is NOW correct — previous review's finding was fixed

---

## Prior Unfixed P0/P1 Status

| Prior ID | Issue | Status |
|----------|-------|--------|
| AVZ-001 (review #5) | 3% stop-proximity guard | **STILL UNFIXED** |
| AVZ-002 (review #5) | TOTP order lock bypass | **STILL UNFIXED** |
| RISK-003 (review #5) | Drawdown staleness | **STILL UNFIXED** (comment added, but no freshness check) |
| MTL-001 (review #6) | OFI double-recording | **STILL UNFIXED** |
| SC-001 (review #6) | _voters count disconnect | **FIXED** (post_persistence_voters added in BUG-224/227) |
| ORCH-001 (review #5) | Post-crash zero-sleep | **FIXED** (verified: last_cycle_started correctly set) |
| MTL-004 (review #5) | OFI sign convention | **FIXED** (verified: Cont et al. formula now correct) |

---

## Recommendations

### Immediate (this week)
1. **Fix SC-001** (regime gate union) — highest impact, affects every 3h/4h cycle
2. **Fix SIG-001** (crypto_cross_asset action key) — dead signal, trivial fix
3. **Fix SC-003** (_rescued flag leak) — affects every consensus computation
4. **Fix AVZ-001** (3% stop guard) — real money protection, unfixed for 3+ reviews

### Next sprint
5. Fix RISK-001, RISK-002, RISK-003 — portfolio safety cluster
6. Fix SC-002, SC-004, SC-006 — accuracy integrity cluster
7. Fix MTL-002, MTL-004 — metals trading safety cluster

### Backlog
8. All P2 I/O violations (INFRA-002, INFRA-003, MTL-005) — consistency debt
9. All P2 data-external findings — robustness improvements
10. P3 findings — low priority

---

*Generated by Adversarial Review #9, 2026-04-30. Dual-stream methodology with
8 parallel code-reviewer agents + independent manual review + cross-critique.*
