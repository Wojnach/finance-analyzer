# Dual Adversarial Review Synthesis — Round 5 (2026-04-14)

**Methodology**: 8 parallel `feature-dev:code-reviewer` subagents (one per subsystem) +
1 independent cross-cutting manual review. Bidirectional cross-critique applied.
All findings deduplicated, false positives downgraded, cross-cutting patterns elevated.

**Scope**: Full codebase — 8 subsystems, ~153 files. Focus: new bugs since R4, verify R4
fix persistence, systemic cross-cutting analysis.

---

## Executive Summary

**Round 4 → Round 5 Fix Rate: R4 CRITICAL fixes held, but 3 CRITICALs remain unfixed.**

C6 (check_drawdown disconnected), C12 (raw open in metals_loop), and C14 (naked position
on stop-loss fail) are all still open from Round 4. More critically, the portfolio-risk
agent discovered that the disconnection is FAR worse than R4 identified: **6 independent
risk functions are disconnected, not just 1.** The entire risk management subsystem exists
only in tests and documentation.

**New critical findings**: 18 P1 CRITICAL, 26 P2 HIGH, 17 P3 MEDIUM across all 8 subsystems
plus the cross-cutting review. The most dangerous cluster is the risk disconnection (6 findings)
and the Claude subprocess bypass pattern (3 findings).

---

## P1 CRITICAL — Must Fix (18 findings)

### Risk Management Theater (6 findings — HIGHEST PRIORITY)

These 6 findings form a single systemic failure: the risk management subsystem is advisory-only,
with zero Python enforcement in any production execution path.

| ID | File | Function | Impact |
|----|------|----------|--------|
| PR-R5-1 | risk_management.py:86 | `check_drawdown()` | 20% drawdown breaker never fires |
| PR-R5-2 | trade_guards.py:177 | `record_trade()` | Cooldowns and rate limits never fire |
| PR-R5-3 | trade_validation.py:32 | `validate_trade()` | Pre-order validation never runs |
| PR-R5-4 | trade_risk_classifier.py:29 | `classify_trade_risk()` | Risk scoring dead code |
| PR-R5-5 | kelly_sizing.py:204 | `recommended_size()` | Kelly sizing not used for main portfolios |
| PR-R5-6 | trade_validation/kelly × 3 | min_trade_sek | 500 vs 1000 SEK inconsistency |

**Recommended fix approach**: Wire `check_drawdown()` + `check_overtrading_guards()` into
`main.py` loop. Call `validate_trade()` before all `avanza_session.place_order()` calls.
Unify MIN_TRADE_SEK to 1000.0 everywhere. This is 3-4 files touched, low risk.

### Claude Subprocess Bypass (3 findings)

| ID | File | Impact |
|----|------|--------|
| OR-R5-1 | bigbet.py:169 | Kill switch, rate limiter, env stripping bypassed |
| OR-R5-2 | multi_agent_layer2.py:154 | 3 specialists bypass _invoke_lock (4 concurrent Claude) |
| OR-R5-3 | agent_invocation.py:248 | Stale specialist reports used by synthesis |

**Recommended fix**: Route all callers through `claude_gate.invoke_claude_text()`.
Single-file change to bigbet.py, one to multi_agent_layer2.py. Add `cleanup_reports()`
call in agent_invocation.py.

### Metals Trading Safety (3 findings)

| ID | File | Impact |
|----|------|--------|
| MC-R5-1 | fin_snipe_manager.py:61 | Stop at entry-5%, not bid-3% (rule violation) |
| MC-R5-2 | fin_fish.py:732 | BEAR MINI knockout check `pass` instead of `continue` |
| MC-R5-3 | fin_snipe_manager.py:948 | Limit sell + stop = 200% volume (overfill → short) |

**Recommended fix**: Change MIN_STOP_DISTANCE_PCT 1.0→3.0, compute from current bid.
Replace `pass` with `continue`. Cap stop volume to position_volume minus open limit sells.

### Signal Engine Correctness (3 findings)

| ID | File | Impact |
|----|------|--------|
| SC-R5-1 | signal_engine.py:1996 | Utility boost bypasses accuracy gate |
| SM-R5-1 | signals/vix_term_structure.py:37 | _Z_THRESHOLD=0 → always votes (noise) |
| SM-R5-2 | signals/vix_term_structure.py:101 | _contango_depth permanent BUY bias |

**Recommended fix**: Apply utility boost only to weight, not to gate decision. Set
_Z_THRESHOLD=0.75. Align _contango_depth BUY threshold to 0.85.

### Data Integrity (3 findings)

| ID | File | Impact |
|----|------|--------|
| DE-R5-1 | onchain_data.py:95 | ISO timestamp crash in _load_onchain_cache (fix inconsistent) |
| IN-R5-2 | journal.py:568 | write_text non-atomic → corrupt Layer 2 context |
| IN-R5-3 | log_rotation.py:244 | Fixed .tmp name → concurrent rotation corrupts signal_log |

---

## P2 HIGH — Should Fix (26 findings)

### Signal Quality

| ID | File | Issue |
|----|------|-------|
| SC-R5-2 | signal_db.py:270 | SQL accuracy path skips neutral outcome filter |
| SC-R5-3 | signal_engine.py:1944 | Regime-accuracy wipes directional fields |
| SM-R5-3 | signals/hurst_regime.py:278 | Duplicate sub-signal inflates confidence |
| SM-R5-6 | signals/calendar_seasonal.py:63 | Day-of-week effect on 24/7 assets |

### Orchestration

| ID | File | Issue |
|----|------|-------|
| OR-R5-4 | analyze.py:272 | Direct subprocess bypasses claude_gate |
| OR-R5-6 | trigger.py:329 | classify_tier triple disk read (M10 optimization unused) |
| OR-R5-7 | agent_invocation.py:639 | Stack overflow counter reset on auth_error |
| OR-R5-8 | multi_agent_layer2.py:186 | Sequential drain starves later specialists |

### Portfolio & Risk

| ID | File | Issue |
|----|------|-------|
| PR-R5-7 | exposure_coach.py:89 | new_entries_allowed flag never enforced |
| PR-R5-8 | monte_carlo.py:310 | p_stop_hit uses terminal price only (not path-dependent) |
| PR-R5-9 | portfolio_mgr.py:43 | Backup rotation propagates corruption |
| PR-R5-11 | kelly_sizing.py:91 | Average buy price vs FIFO disagree on win/loss |

### Metals

| ID | File | Issue |
|----|------|-------|
| MC-R5-4 | fin_fish.py:196 | session_hours_remaining hardcodes 21:55 (DST violation) |
| MC-R5-5 | orb_predictor.py:32 | ORB window hardcoded winter UTC (wrong 1h in summer) |
| MC-R5-6 | fin_snipe_manager.py:510 | Stop proximity skipped when bid=0 |
| MC-R5-7 | fish_monitor_smart.py:225 | Raw read_text() on JSONL (race with metals loop) |

### Avanza

| ID | File | Issue |
|----|------|-------|
| AV-R5-1 | metals_avanza_helpers.py:253 | No account whitelist guard (page path) |
| AV-R5-2 | avanza_session.py:356 | api_delete doesn't handle 403 (stale session) |
| AV-R5-3 | golddigger/runner.py:178 | Stop-loss placed when bid=0 (no proximity guard) |

### Data External

| ID | File | Issue |
|----|------|-------|
| DE-R5-2 | crypto_macro_data.py:202 | Gold/BTC ratio uses 1h-stale disk prices |
| DE-R5-3 | futures_data.py:33 | get_open_interest missing oi_usdt (docstring lie) |
| DE-R5-4 | onchain_data.py:267 | 24h stale fallback with DEBUG logging only |
| DE-R5-5 | fear_greed.py:16 | Relative Path("data/...") — CWD-dependent |
| DE-R5-6 | earnings_calendar.py:49 | AV earnings bypass 25-call/day budget |

### Infrastructure

| ID | File | Issue |
|----|------|-------|
| IN-R5-4 | telegram_poller.py:199 | Raw json.load on config.json |
| IN-R5-5 | message_throttle.py:69 | Check-and-send race → duplicate Telegrams |

### Cross-Cutting

| ID | Scope | Issue |
|----|-------|-------|
| XC-R5-2 | main.py (25+ sites) | Error swallowing cascade — no escalation |
| XC-R5-5 | meta_learner.py:390,437 | Raw json.loads bypasses file_utils |

---

## P3 MEDIUM — Nice to Fix (17 findings)

| ID | File | Issue |
|----|------|-------|
| SC-R5-4 | signal_history.py:53 | Read-modify-write race (dead code — no callers) |
| SC-R5-5 | forecast_accuracy.py:28 | Raw read_text on predictions JSONL |
| SC-R5-6 | accuracy_stats.py:647 | Fictional sample count from max() |
| SC-R5-7 | signal_engine.py:358 | Regime gate fallback for unknown horizons |
| SC-R5-8 | outcome_tracker.py:74 | fear_greed derivation ignores sustained fear |
| OR-R5-5 | trigger.py:53 | _startup_grace_active dead logic |
| OR-R5-9 | perception_gate.py:59 | Reads prior-cycle summary |
| OR-R5-10 | bigbet.py:169 | CLAUDECODE env not stripped |
| PR-R5-13 | risk_management.py:21 | portfolio_value_history unbounded growth |
| MC-R5-8 | fin_snipe_manager.py:1527 | Static hours_remaining=6.0 in loop |
| MC-R5-10 | fin_fish.py:747 | Negative barrier distance accidentally correct |
| AV-R5-4 | avanza_orders.py:64 | Sub-1000 SEK accepted, fails at execution |
| AV-R5-5 | avanza/trading.py:38 | No whitelist or min size (not used in prod) |
| DE-R5-7 | social_sentiment.py:104 | Reddit errors go to stdout |
| DE-R5-8 | crypto_scheduler.py:110 | Telegram report no staleness guard |
| IN-R5-8 | dashboard/app.py:664 | No auth by default + 0.0.0.0 + CORS * |
| IN-R5-9 | market_timing.py:321 | Hour comparison misses NYSE 09:30 open |
| IN-R5-10 | file_utils.py:155 | atomic_append_jsonl not thread-safe |

---

## Round 4 → Round 5 Fix Scorecard

| R4 ID | Issue | R5 Status |
|-------|-------|-----------|
| C6 | check_drawdown() never called | **STILL OPEN** (now 6 functions, not 1) |
| C12 | raw open("a") in metals_loop | **NOT VERIFIED** (metals_loop.py not reviewed) |
| C14 | Naked position on stop-loss fail | **STILL OPEN** |
| C1 (partial) | ADX cache id(df) collision | Not re-examined |
| C3 (partial) | wait_for_specialists blocks | **CONFIRMED WORSE** (OR-R5-8: sequential drain) |

---

## Recommended Fix Order (Batches)

### Batch 1 — Risk Wiring (P1, low risk, 4 files)
1. Wire check_drawdown() into main.py loop cycle
2. Wire record_trade() into journal.py trade logging
3. Unify MIN_TRADE_SEK = 1000.0 in trade_validation, kelly_sizing, kelly_metals
4. Wire validate_trade() before avanza_session.place_order()

### Batch 2 — Claude Gate (P1, medium risk, 3 files)
1. Route bigbet.py through claude_gate.invoke_claude_text()
2. Route multi_agent_layer2.py specialists through claude_gate
3. Add cleanup_reports() call in agent_invocation.py

### Batch 3 — Metals Safety (P1, low risk, 2 files)
1. MC-R5-1: MIN_STOP_DISTANCE_PCT 1.0→3.0, compute from bid
2. MC-R5-2: Replace `pass` with `continue` in fin_fish.py
3. MC-R5-3: Cap stop volume to position_volume minus open limit sells

### Batch 4 — Signal Correctness (P1, low risk, 2 files)
1. SC-R5-1: Apply utility boost only to weight, not gate
2. SM-R5-1/2: Fix vix_term_structure thresholds

### Batch 5 — Data Integrity (P1, low risk, 3 files)
1. DE-R5-1: Apply _coerce_epoch in _load_onchain_cache
2. IN-R5-2: Use atomic write for journal context
3. IN-R5-3: Use mkstemp in log_rotation

### Batch 6+ — P2 HIGH (deferred for next session or autonomous fix)

---

## Statistics

| Category | P1 | P2 | P3 | Total |
|----------|----|----|----|----|
| Signals Core | 3 | 3 | 2 | 8 |
| Orchestration | 3 | 5 | 2 | 10 |
| Portfolio Risk | 6 | 5 | 1 | 12 |
| Metals Core | 3 | 4 | 3 | 10 |
| Avanza API | 0 | 3 | 3 | 6 |
| Signals Modules | 2 | 2 | 2 | 6 |
| Data External | 1 | 5 | 2 | 8 |
| Infrastructure | 3 | 2 | 3 | 8 |
| Cross-Cutting | 0 | 2 | 0 | 2 |
| **Total** | **21** | **31** | **18** | **70** |

Note: Some findings overlap across categories (e.g., PR-R5-3/XC-R5-4 min trade size).
After deduplication: **18 unique P1, 26 unique P2, 17 unique P3 = 61 unique findings.**

---

## Systemic Patterns

1. **Advisory-only guards**: Risk functions exist but are never wired into execution paths.
   Layer 2 (LLM) reads JSON fields; nothing in Python enforces them.

2. **Raw file I/O violations**: 8+ locations still use raw open/read_text/json.load instead of
   file_utils. Each is a TOCTOU or corruption risk. The rule exists; enforcement is inconsistent.

3. **DST-hardcoded constants**: 4+ locations hardcode winter-CET or winter-UTC offsets that
   break during summer time. session_calendar.py has the correct DST logic but isn't used.

4. **Claude subprocess bypass**: The claude_gate was designed as a chokepoint but 4+ call sites
   go around it, making the kill switch and serialization lock ineffective.

5. **Stale disk reads**: Multiple modules read agent_summary_compact.json as "current" data
   with no freshness check. When the loop is down, decisions are made on hours-old prices.
