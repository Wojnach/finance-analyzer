# Adversarial Code Review: Metals-Core Subsystem (Claude Reviewer)

## Executive Summary

**4 P1 findings, 7 P2 findings, 10 P3 findings.**

The metals-core subsystem is well-hardened with post-mortem-driven fixes, but critical issues remain: double position deletion race between `_execute_sell` and `_check_exits`, hardcoded CET close time ignoring DST gap weeks, Monte Carlo VaR using fixed seed=42 (deterministic), and stop-loss inverted for SHORT positions.

---

## P1 Findings

### P1-1: Double position deletion between `_execute_sell` and `_check_exits`
**File:** `data/metals_swing_trader.py:2925, 2807-2812`
If `_execute_sell` succeeds but stop-loss cancellation fails, orphan stops remain on Avanza.

### P1-2: Hardcoded 21:55 CET close ignores DST gap weeks
**File:** `data/metals_swing_trader.py:2569`
During DST gaps (Mar 8-29, Oct 25-Nov 1), real Avanza close shifts to 21:00 CET.

### P1-3: Monte Carlo VaR uses fixed seed=42 -- not stochastic
**File:** `data/metals_risk.py:186`
VaR, CVaR, p_profit_5pct always return the same values. Risk metrics never adapt.

### P1-4: Stop-loss inverted for SHORT positions
**File:** `data/metals_swing_trader.py:2530-2533`
LESS_OR_EQUAL trigger fires on wins, not losses for BEAR warrants. Currently disabled (SHORT_ENABLED=False).

## P2 Findings

### P2-1: `fill_probability_buy` incorrect symmetry with non-zero drift
**File:** `portfolio/price_targets.py:106`

### P2-2: Re-reads config.json on every Telegram send
**File:** `data/metals_swing_trader.py:607-614`

### P2-3: Blocks loop with Playwright fetch_price for every position every cycle
**File:** `data/metals_swing_trader.py:2650`

### P2-4: Transient histories (MACD, RSI) persisted to disk causing unnecessary I/O
**File:** `data/metals_swing_trader.py:3006-3020`

### P2-5: Sells at stale bid price after lengthy exit evaluation
**File:** `data/metals_swing_trader.py:2866`

### P2-6: Reads entire signal log into memory for backfill
**File:** `data/metals_signal_tracker.py:343-349`

### P2-7: Entry rvol URL construction fragile, fail-open possible
**File:** `data/metals_loop.py:1192`

## P3 Findings

P3-1: CET fallback ignores CEST (off by 1h in summer) -- `metals_swing_trader.py:534`
P3-2: ORB morning window UTC offsets wrong during CEST -- `orb_predictor.py:33`
P3-3: pos_id collision possible within same second -- `metals_swing_trader.py:2392`
P3-4: weekday from UTC vs hours from CET mismatch -- `metals_shared.py:134`
P3-5: Relative config.json path fragile -- `metals_swing_trader.py:593`
P3-6: ATR annualization underestimates vol by ~75% -- `metals_risk.py:135`
P3-7: Non-atomic JSON write -- `metals_history_fetch.py:206`
P3-8: MINI_L_SILVER_AVA_301 barrier=75.03 appears wrong for LONG -- `metals_swing_config.py:27`
P3-9: Unnecessary HTTP call to timeapi.io every cycle -- `metals_shared.py:84`
P3-10: Reads full prediction logs every 60s -- `metals_llm.py:592`
