# Adversarial Code Review: Portfolio-Risk Subsystem (Claude Reviewer)

## Executive Summary

**1 P1 finding, 6 P2 findings, 9 P3 findings.**

The system has a well-implemented drawdown circuit breaker that literally nothing calls. The entire trade guard subsystem is wired for reporting but not enforcement. Together, the main portfolio trading path has **zero automated risk limit enforcement** despite having the code for it.

---

## P1 Findings

### F01: Drawdown circuit breaker is DEAD CODE
**File:** `portfolio/risk_management.py:86`
`check_drawdown()` is defined and tested but never called from any production code path. The system can trade through unlimited drawdown with no automated stop.

## P2 Findings

### F02: `should_block_trade()` never called in production
**File:** `portfolio/trade_guards.py:294`
Exists in module with tests. Never imported or called outside tests.

### F03: `record_trade()` never called for Patient/Bold
**File:** `portfolio/trade_guards.py:177`
Only called from golddigger/elongir. Main portfolios have zero trade guard data. Overtrading guards are NON-FUNCTIONAL.

### F04: Warrant portfolio has no locking -- race on concurrent read-modify-write
**File:** `portfolio/warrant_portfolio.py:197-245`
Metals loop + Layer 2 can both call `record_warrant_transaction()` concurrently.

### F05: Shallow copy of `_DEFAULT_STATE` shares mutable internals
**File:** `portfolio/warrant_portfolio.py:33`
If save fails, `_DEFAULT_STATE` itself is corrupted for process lifetime.

### F06: Drawdown fails open on missing price data
**File:** `portfolio/risk_management.py:126-139`
Falls back to cash-only value, underestimates drawdown, allows trading to continue.

### F07: Undocumented "CHECK" action bypass
**File:** `portfolio/risk_management.py:808-811`

## P3 Findings

F08: Silent cash-only fallback on bad fx_rate -- `portfolio_mgr.py:162`
F09: Monte Carlo terminal-price stop probability underestimates reality -- `monte_carlo.py:198`
F10: `_streaming_max` reads without lock -- `risk_management.py:37`
F11: Confusing vol/std dual computation -- `equity_curve.py:211`
F12: Calmar ratio uses synthetic equity curve -- `equity_curve.py:518`
F13: avg_cost check incorrect after partial sells -- `portfolio_validator.py:225`
F14: Concentration noted but not penalized in exposure coach -- `exposure_coach.py:104`
F15: Failure count not reset on OPEN->HALF_OPEN -- `circuit_breaker.py:79`
F16: No locking in trade guard state I/O -- `trade_guards.py:31`
