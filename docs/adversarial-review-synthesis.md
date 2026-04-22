# Dual Adversarial Review — Synthesis Document
**Date:** 2026-04-22
**Methodology:** 8 parallel agent reviews + 1 independent review, cross-critiqued

## Executive Summary

Two independent review passes across 8 subsystems found **52 confirmed P1/P2 findings** and
**15 P3 quality issues**. The system is broadly well-engineered with extensive defensive coding,
but several classes of bug recur across subsystems:

1. **Silent data staleness** — Multiple paths fall back to stale data without logging (fx_rate=1.0, avg_cost_usd, stale VWAP, hardcoded econ dates)
2. **Concurrency gaps** — Thread-safe caching is excellent in shared_state.py but several modules (trade_guards, health, reporting) lack locks on read-modify-write cycles
3. **Gate/threshold inconsistency** — Constants that should be in sync aren't (ATR cap, fx_rate defaults, drawdown thresholds)
4. **Signal computation errors** — Fibonacci extensions 10x too small, VWAP cumulative instead of session-based, volatility annualization wrong for equities
5. **Avanza order safety** — CONFIRM flow lacks order-ID verification, account whitelist not enforced uniformly

---

## Cross-Critique: Agent Findings vs. Independent Review

### Agent found, Independent missed:
| Finding | Subsystem | Why missed |
|---------|-----------|------------|
| Fibonacci extensions 0.272 instead of 1.272 | signals-modules | Didn't read fibonacci.py deep enough |
| `record_trade` never receives `pnl_pct` → loss escalation dead | portfolio-risk | Didn't trace the call chain from agent_invocation |
| Regime accuracy cache single global timestamp across horizons | signals-core | Didn't read accuracy_stats.py's cache timing |
| `_check_telegram_confirm` no nonce/order-ID | avanza-api | **Agent confirmed independent finding** |
| Alpha Vantage "Information" rate-limit body not detected | data-external | Didn't read AV response handling |
| news_event "cut" keyword maps positive for "job cut" etc. | signals-modules | Didn't read keyword classification logic |
| `log_rotation.py` no fsync before replace | infrastructure | Didn't read log rotation implementation |
| `cancel_order` missing account whitelist check | avanza-api | Didn't read cancel_order path |
| `metals_avanza_helpers.place_order` no account whitelist | avanza-api | Didn't know about this file |
| Layer 2 unconditionally skipped on weekends for crypto | orchestration | Knew about market timing but didn't trace the gate |

### Independent found, Agent missed:
| Finding | Subsystem | Why missed |
|---------|-----------|------------|
| `_streaming_max` reads entire JSONL every cycle | portfolio-risk | Agent focused on logic bugs, not performance |
| Portfolio value doesn't handle SEK-denominated warrants | portfolio-risk | Narrow focus on USD holdings |
| metals_loop.py is a 3000+ line monolith | metals-core | Agents review code, not architecture |
| No integration test for signal→trigger→agent path | cross-cutting | Agent scope limited to single subsystem |

### Both found independently (high confidence):
| Finding | Subsystem |
|---------|-----------|
| Persistence filter cold-start double-permissive | signals-core |
| Orphaned agent process on loop restart | orchestration |
| Telegram CONFIRM lacks order-ID | avanza-api |
| Playwright browser context never closed | avanza-api |
| econ_dates.py hardcoded dates | data-external |
| `_cached` dogpile returns None without distinguishing from empty | infrastructure |

---

## Consolidated P1 Findings (Highest Priority)

### P1-01: Fibonacci extension multipliers 10x too small
**Source:** Agent (signals-modules)
**File:** `portfolio/signals/fibonacci.py:165-168`
**Impact:** Extension signals fire at 27.2% of swing range instead of 127.2%, producing premature SELL signals at routine pullback highs.
**Fix:** Change `0.272 → 1.272` and `0.618 → 1.618` for all four extension calculations.

### P1-02: `record_trade` called without `pnl_pct` → consecutive-loss escalation permanently dead
**Source:** Agent (portfolio-risk)
**File:** `portfolio/agent_invocation.py:720`
**Impact:** After multiple losing trades, cooldown multiplier stays at 1x — overtrading guards provide zero loss-streak protection.
**Fix:** Extract `pnl_pct` from transaction dict for SELL trades and pass to `record_trade`.

### P1-03: `ministral` excluded from `_compute_applicable_count` for non-crypto but runs for all tickers
**Source:** Agent (signals-core)
**File:** `portfolio/signal_engine.py:895 vs 2487`
**Impact:** Applicable signal count is wrong for metals/MSTR, affecting circuit breaker and voter diversity logic.
**Fix:** Sync the two code paths — either exclude ministral from non-crypto dispatch, or include in applicable count.

### P1-04: Regime accuracy cache uses single global timestamp across all horizons
**Source:** Agent (signals-core)
**File:** `portfolio/accuracy_stats.py:1136`
**Impact:** Writing 3h regime accuracy refreshes 1d TTL — stale regime accuracy used for 1d consensus.
**Fix:** Use per-horizon timestamp keys.

### P1-05: Fear & Greed sustained-fear gate contradicts stated intent
**Source:** Agent (signals-core)
**File:** `portfolio/signal_engine.py:2293-2300`
**Impact:** BUY allowed after 30d of extreme fear (when contrarian is least reliable) but blocked during first 30d (when it's most reliable).

### P1-06: `fear_greed.py` — unguarded `body["data"][0]` crashes on empty API response
**Source:** Agent (data-external)
**File:** `portfolio/fear_greed.py:100`
**Impact:** IndexError on maintenance window, entire Fear & Greed signal crashes.
**Fix:** Guard with `if not body.get("data"): return None`.

### P1-07: Alpha Vantage missing `raise_for_status()` — 4xx parsed as valid data
**Source:** Agent (data-external)
**File:** `portfolio/alpha_vantage.py:145`
**Impact:** Invalid API key or suspended account silently produces corrupt fundamentals data.

### P1-08: Avanza orders CONFIRM flow routes through TOTP path, bypassing 50K cap and order lock
**Source:** Agent (avanza-api)
**File:** `portfolio/avanza_orders.py:17`, `portfolio/avanza_control.py:36-45`
**Impact:** Confirmed order bypasses all safety guards in `avanza_session._place_order`.

### P1-09: `metals_avanza_helpers.place_order` accepts arbitrary account_id, no whitelist
**Source:** Agent (avanza-api)
**File:** `data/metals_avanza_helpers.py:253-327`
**Impact:** Config corruption or bug can send orders to wrong Avanza account.

### P1-10: `health.py` conflates trigger time with invocation time in `last_invocation_ts`
**Source:** Agent (infrastructure)
**File:** `portfolio/health.py:35`
**Impact:** Agent silence detector always sees fresh timestamp (trigger time, not completion time), defeating the 3-week-outage detection that was its purpose.

### P1-11: `log_rotation.py` no fsync before os.replace — data loss on power failure
**Source:** Agent (infrastructure)
**File:** `portfolio/log_rotation.py:242-249`
**Impact:** Signal history JSONL (weeks of data) can be truncated/zeroed on crash during rotation.

---

## Consolidated P2 Findings (Important)

| # | Subsystem | File | Finding |
|---|-----------|------|---------|
| P2-01 | signals-core | signal_engine.py:697 | `unknown` regime has no `_default` gate — funding/fear_greed vote freely at 1d |
| P2-02 | signals-core | accuracy_stats.py:841 | Directional accuracy not blended — bypasses directional gate for degrading signals |
| P2-03 | signals-core | signal_engine.py:1730 | Circuit breaker relaxation applies to high-sample tier, defeating its stricter floor |
| P2-04 | signals-modules | volume_flow.py:62 | VWAP cumulative across entire DataFrame, not session-based — permanent directional bias |
| P2-05 | signals-modules | calendar_seasonal.py:63 | Monday SELL / Friday BUY applied to 24/7 crypto/metals |
| P2-06 | signals-modules | news_event.py:255 | "cut" keyword maps positive for "job cut", "dividend cut" etc. |
| P2-07 | signals-modules | volatility.py:160 | `sqrt(365)` annualizer wrong for MSTR equity (should be 252) |
| P2-08 | signals-modules | cot_positioning.py:55 | Relative file path breaks outside repo-root CWD |
| P2-09 | orchestration | agent_invocation.py:249 | `_kill_overrun_agent` leaves stale module globals |
| P2-10 | orchestration | agent_invocation.py:425 | Multi-agent mode blocks main loop for 30s synchronously |
| P2-11 | orchestration | market_timing.py:253 | Layer 2 unconditionally skipped on weekends for all assets including crypto |
| P2-12 | orchestration | agent_invocation.py:135 | `_extract_ticker` defaults to XAG-USD for non-ticker triggers |
| P2-13 | portfolio-risk | risk_management.py:66 | fx_rate defaults to 1.0 vs 10.0 in monte_carlo_risk.py — 10x error |
| P2-14 | portfolio-risk | trade_guards.py:229 | No lock on load→mutate→save in `record_trade` |
| P2-15 | portfolio-risk | risk_management.py:225 | ATR 15% cap applied inconsistently across 3 stop functions |
| P2-16 | portfolio-risk | equity_curve.py:382 | `pnl_sek` excludes fees — all P&L metrics overstated |
| P2-17 | portfolio-risk | risk_management.py:79 | Per-ticker stale price fallback to avg_cost_usd with no warning |
| P2-18 | avanza-api | avanza_orders.py:122 | CONFIRM matches most-recent order, no order-ID verification |
| P2-19 | avanza-api | avanza_session.py:628 | `cancel_order` missing account whitelist check |
| P2-20 | avanza-api | avanza_orders.py:132 | Crash between status=confirmed and save allows duplicate execution |
| P2-21 | data-external | alpha_vantage.py:151 | "Information" rate-limit body not detected — circuit breaker misfires |
| P2-22 | data-external | http_retry.py:58 | Catches too-narrow exception set — ChunkedEncodingError bypasses retries |
| P2-23 | data-external | earnings_calendar.py:49 | AV earnings calls bypass daily budget counter |
| P2-24 | data-external | crypto_macro_data.py:275 | Raw `open()` for JSONL reads — violates atomic I/O rule |
| P2-25 | infrastructure | shared_state.py:256 | Rate limiter TOCTOU race allows bursting |
| P2-26 | infrastructure | log_rotation.py:318 | `rotate_text` truncates live log non-atomically — data loss window |
| P2-27 | infrastructure | api_utils.py:29 | Config mtime TOCTOU race during symlink update |

---

## Priority Remediation Plan

### Tier 1 — Fix this week (P1 with direct trading impact)
1. **P1-01** Fibonacci extensions (one-line fix, high impact)
2. **P1-02** Wire pnl_pct to record_trade (one-line fix, enables loss escalation)
3. **P1-08** Route Avanza CONFIRM through session path with safety guards
4. **P1-09** Add account whitelist to metals_avanza_helpers
5. **P1-06** Guard fear_greed empty response
6. **P1-07** Add raise_for_status to Alpha Vantage

### Tier 2 — Fix within 2 weeks (P1 infrastructure + P2 high-impact)
7. **P1-10** Fix health.py invocation timestamp source
8. **P1-11** Add fsync to log_rotation.py
9. **P1-03** Sync ministral applicable count
10. **P1-04** Per-horizon regime accuracy timestamps
11. **P2-04** Fix VWAP to session-based or rolling
12. **P2-11** Allow Layer 2 on weekends for crypto/metals
13. **P2-13** Sync fx_rate defaults (1.0 → 10.0)

### Tier 3 — Fix within month (P2 accuracy/safety)
14-27. Remaining P2 findings in priority order per subsystem

---

## Statistics

| Source | P1 | P2 | P3 | Total |
|--------|----|----|-----|-------|
| Independent review | 6 | 24 | 5 | 35 |
| Agent: signals-core | 5 | 5 | 3 | 13 |
| Agent: orchestration | 1 | 7 | 5 | 13 |
| Agent: portfolio-risk | 3 | 7 | 3 | 13 |
| Agent: avanza-api | 3 | 5 | 0 | 8 |
| Agent: signals-modules | 3 | 8 | 4 | 15 |
| Agent: data-external | 5 | 6 | 3 | 14 |
| Agent: infrastructure | 3 | 6 | 2 | 11 |
| **Deduplicated total** | **11** | **27** | **15** | **53** |

---

*Review generated 2026-04-22 by dual adversarial methodology: 8 parallel specialized agents + 1 independent review, with bidirectional cross-critique.*
