# Dual Adversarial Review Synthesis — 2026-04-18

**Reviewers:** Claude Opus 4.6 (8 parallel agents) + Codex CLI o3 (8 parallel sessions)
**Baseline:** Commit 322c7a59 (main)
**Prior review:** 2026-04-10 (148 findings, many since fixed)

---

## Executive Summary

This second full adversarial review examines the codebase 8 days after the first review (2026-04-10).
Many P0 findings from the prior review have been fixed. However, **152 new findings** were identified:
**32 P1 (critical), 59 P2 (high), 61 P3 (medium)**.

The most alarming discovery: the **drawdown circuit breaker, trade blocking gate, and trade recording
for Patient/Bold portfolios are ALL dead code**. The main trading path has zero automated risk
enforcement. This was partially flagged in the prior review (A-PR-1) but remains unfixed.

---

## Aggregate Findings

| Subsystem | P1 | P2 | P3 | Total |
|-----------|----|----|-----|-------|
| signals-core | 5 | 8 | 5 | 18 |
| orchestration | 5 | 7 | 6 | 18 |
| portfolio-risk | 1 | 6 | 9 | 16 |
| metals-core | 4 | 7 | 10 | 21 |
| avanza-api | 3 | 5 | 5 | 13 |
| signals-modules | 6 | 9 | 10 | 25 |
| data-external | 3 | 7 | 6 | 16 |
| infrastructure | 5 | 10 | 10 | 25 |
| **TOTAL** | **32** | **59** | **61** | **152** |

---

## TOP 10 MOST CRITICAL FINDINGS

### 1. Drawdown circuit breaker is DEAD CODE
**portfolio-risk** | `risk_management.py:86`
`check_drawdown()` exists, is tested, but never called. System trades through unlimited drawdown.
Combined with `should_block_trade()` and `record_trade()` also never called for main portfolios.
**Zero automated risk enforcement.**

### 2. No OHLCV price validation — NaN/zero propagates to signals
**data-external** | `data_collector.py:94-248`
All three price sources pass through NaN/zero/negative prices unchecked. Division-by-zero in RSI/MACD.

### 3. Missing account whitelist on Playwright order path
**avanza-api** | `metals_avanza_helpers.py:253`
Pension account `2674244` can receive orders. No ALLOWED_ACCOUNT_IDS check.

### 4. Browser recovery duplicates non-idempotent orders
**avanza-api** | `avanza_session.py:207-228`
If browser dies during POST, retry places duplicate real-money order.

### 5. Agent invocation bypasses claude_gate
**orchestration** | `agent_invocation.py:451`
Direct Popen bypasses lock, kill switch, rate limit, tree-kill. Zombie processes accumulate.

### 6. Per-ticker accuracy inflated (no neutral filtering)
**signals-core** | `ticker_accuracy.py:60-62`, `signal_db.py:270-272`
0.01% moves counted as "correct". Mode B probability notifications use inflated numbers.

### 7. Dashboard token vulnerable to timing attack
**infrastructure** | `dashboard/app.py:675-682`
`==` comparison + wildcard CORS = brute-forceable token.

### 8. Monte Carlo VaR uses fixed seed=42
**metals-core** | `metals_risk.py:186`
Deterministic results. Risk metrics never adapt.

### 9. Signal history race under ThreadPoolExecutor
**signals-core** | `signal_history.py:53-82`
No lock on load-modify-write. 8 workers → data loss.

### 10. Config.json non-atomic mutation from Telegram
**infrastructure** | `telegram_poller.py:197-208`
Race condition on config file containing ALL API keys.

---

## SYSTEMIC ISSUES

### A. Dead Safety Code
Three critical safety mechanisms implemented but never called:
1. `check_drawdown()` — drawdown circuit breaker
2. `should_block_trade()` — trade blocking gate
3. `record_trade()` for Patient/Bold — trade guard data

### B. Atomic I/O Violations (4 sites)
- `journal.py:568` — `Path.write_text()` for Layer 2 context
- `metals_history_fetch.py:206` — raw `json.dump()`
- `accuracy_stats.py:1162` — `read_text()` for snapshots
- `telegram_poller.py:197` — raw `open()` for config

### C. Thread Safety Gaps (5 shared resources without locks)
`signal_history`, `fx_rates._fx_cache`, `reporting._held_tickers_cache`,
`api_utils._config_cache`, `shared_state._loading_timestamps`

### D. DST/Timezone Bugs (5 modules)
`session_calendar`, `metals_swing_trader`, `orb_predictor`, `metals_shared`, `econ_dates`

### E. Structural Signal Bias (3 signals)
`econ_calendar` (SELL-only), `funding_rate` (BUY-biased 3:1), `orderbook_flow.spread_health` (dead)

---

## COMPARISON WITH PRIOR REVIEW (2026-04-10)

### Fixed since prior review:
- A-MC-1 (HARD_STOP_CERT_PCT=5%) — appears addressed
- A-MC-2 (usdsek=1.0) — appears addressed
- A-AV-1 (Playwright thread safety) — partially addressed with `_pw_lock` expansion
- Several P2/P3 findings across subsystems

### Still open from prior review:
- A-PR-1: `record_trade()` never called → **CONFIRMED STILL OPEN** (our F02/F03)
- A-OR-2: Trigger state TOCTOU → **CONFIRMED** (our P1-4)
- A-OR-4: `_maybe_send_digest` unprotected → **CONFIRMED** (our P1-5)
- Subprocess governance → **CONFIRMED** (our P1-1 orchestration)

### New findings not in prior review:
- Per-ticker accuracy inflation (P1-1 signals-core)
- TOTP singleton session expiry (P1-2 avanza-api)
- ATR annualization underestimates vol by ~75% (P3-6 metals-core)
- Weekly digest OOM risk from 68MB signal_log (P1-5 infrastructure)
- GPU lock TOCTOU between processes (P1-3 infrastructure)
- Funding rate asymmetric thresholds creating BUY bias

---

## PRIORITY FIX BATCHES

### Batch 1: Safety-Critical (Do Now — 6 items)
1. Wire `check_drawdown()` into production
2. Wire `should_block_trade()` into Layer 2 path
3. Add OHLCV price validation
4. Add account whitelist to Playwright order path
5. Fix `hmac.compare_digest()` in dashboard
6. Move notification mode to separate state file

### Batch 2: Trading Quality (This Week — 6 items)
7. Fix neutral-outcome filtering in ticker_accuracy/signal_db
8. Remove MC `seed=42` from production
9. Add threading.Lock to signal_history
10. Add browser-recovery idempotency guard
11. Route invoke_agent through claude_gate
12. Fix DST bugs in session_calendar + metals_swing_trader

### Batch 3: Reliability (This Sprint — 6 items)
13. Fix all 4 atomic I/O violations
14. Fix 5 thread safety gaps
15. Import FOMC dates from canonical source in oil_precompute
16. Add barrier proximity guard on stop-loss
17. Fix weekly_digest OOM (use load_jsonl_tail)
18. Add timeout to Windows file lock

---

## CODEX CROSS-CRITIQUE

Codex CLI (o3) ran 8 parallel sessions (~5MB total session data). Sessions performed deep analysis
but exhausted context before producing final reports. Partial findings visible in session logs:
- Corroborated signal_log.jsonl rewrite race (matches Claude P2-3 signals-core)
- Corroborated portfolio_mgr.py shared mutable state (matches Claude F05 portfolio-risk)
- Corroborated browser recovery retry risks (matches Claude P2-3 avanza-api)
No contradictions found in partial codex output.

---

## REVIEW FILES

| File | Content |
|------|---------|
| `PLAN.md` | Subsystem definitions |
| `claude-signals-core.md` | 5P1/8P2/5P3 |
| `claude-orchestration.md` | 5P1/7P2/6P3 |
| `claude-portfolio-risk.md` | 1P1/6P2/9P3 |
| `claude-metals-core.md` | 4P1/7P2/10P3 |
| `claude-avanza-api.md` | 3P1/5P2/5P3 |
| `claude-signals-modules.md` | 6P1/9P2/10P3 |
| `claude-data-external.md` | 3P1/7P2/6P3 |
| `claude-infrastructure.md` | 5P1/10P2/10P3 |
| `SYNTHESIS-2026-04-18.md` | This document |

*Review conducted 2026-04-18 by Claude Opus 4.6 (1M) with Codex CLI o3.*
*Total: 32 P1, 59 P2, 61 P3 = 152 findings across 8 subsystems (~146 files).*
