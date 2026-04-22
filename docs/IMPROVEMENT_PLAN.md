# Improvement Plan — Auto Session 2026-04-22

Based on exploration by 4 parallel agents (signals, portfolio/trading, infrastructure, metals/tests)
plus synthesis of findings from 4 consecutive adversarial reviews (Rounds 1-4).

## Exploration Summary

### Signal Engine (3,225 LOC)
- Operator precedence in `accuracy_stats.py:344` — correct by Python rules but needs explicit parens
- `_persistence_state` dict unbounded — memory leak for test/removed tickers
- 15/48 signals disabled pending validation (39% unvalidated code)

### Portfolio/Trading
- **CRITICAL (5th consecutive review)**: `record_trade()` in trade_guards.py has ZERO production callers
  - Overtrading prevention (cooldowns, loss escalation, rate limits) disconnected
  - Guards read state that is never populated — entire subsystem is paper-only
- Most prior review bugs (BUG-209 through BUG-218) confirmed fixed

### Infrastructure
- Rate limiter lacks jitter — synchronized wake under sustained load
- health_lock held during disk write — can block on antivirus
- yfinance_lock acquired at wrong level in data_collector.py

### Metals/Raw I/O
- Raw `open()` violations persist in 4 files (5th consecutive review):
  - `data/metals_swing_trader.py:596,612` — config.json read
  - `data/metals_loop.py:6732` — log file append
  - `portfolio/signals/credit_spread.py:285` — config.json read

---

## Batch 1: Overtrading Prevention Wiring (CRITICAL — 3 files, ~40 lines)

### 1.1 Wire `record_trade()` into journal parsing path

**Files:** `portfolio/agent_invocation.py`
**Problem:** `trade_guards.record_trade()` is never called from production code. The entire
overtrading prevention system (ticker cooldowns, consecutive-loss escalation, position rate
limits) is dead. This has been flagged in 5 consecutive adversarial reviews (C6, PR-R4-4).

**Fix:** In `check_agent_completion()`, after successfully parsing a journal entry that contains
a BUY or SELL decision, call `record_trade()` with the relevant ticker, direction, strategy,
and P&L data. The journal entry already contains all needed fields.

**Impact:** Activates the entire overtrading prevention subsystem for the first time.
**Risk:** MEDIUM — adds a new call on the completion path. If record_trade() raises, it must
be wrapped in try/except to avoid breaking the completion flow.

### 1.2 Add guard warnings to agent prompt context

**File:** `portfolio/agent_invocation.py`
**Problem:** Even with record_trade() wired in, Layer 2 never sees guard warnings because
`get_all_guard_warnings()` isn't called in the prompt-building path.

**Fix:** Call `get_all_guard_warnings()` in the T2/T3 prompt builder and include any blocking
warnings in the agent context.

**Impact:** Layer 2 makes informed decisions about cooldowns and rate limits.
**Risk:** LOW — additive only, doesn't change trading logic.

---

## Batch 2: Raw I/O Safety (4 files, ~25 lines)

### 2.1 metals_swing_trader.py — Replace raw open("config.json")

**File:** `data/metals_swing_trader.py`, lines 596 and 612
**Problem:** Two raw `open("config.json")` calls for Telegram config. Violates Rule 4.

**Fix:** Replace with `load_json()` from file_utils.

### 2.2 metals_loop.py — Replace raw open() for agent log

**File:** `data/metals_loop.py`, line 6732
**Problem:** `open("data/metals_agent.log", "a")` — raw append with relative path.

**Fix:** Use absolute path + proper error handling with context manager.

### 2.3 credit_spread.py — Replace raw open("config.json")

**File:** `portfolio/signals/credit_spread.py`, line 285
**Problem:** Raw `open("config.json")` for FRED API key.

**Fix:** Replace with `load_json()`.

---

## Batch 3: Signal Engine Hardening (2 files, ~20 lines)

### 3.1 Bound _persistence_state dict

**File:** `portfolio/signal_engine.py`, around line 260
**Problem:** `_persistence_state` grows unbounded. Each unique ticker adds an entry
that is never cleaned up.

**Fix:** Add size cap (32 tickers) with LRU eviction, matching `_phase_log_per_ticker` pattern.

### 3.2 Add explicit parentheses to cost-adjusted accuracy

**File:** `portfolio/accuracy_stats.py`, line 344
**Problem:** Missing explicit grouping. Correct by precedence but confusing.

**Fix:** Add parentheses for clarity.

---

## Batch 4: Tests (~40 lines)

### 4.1 Test record_trade() wiring
- Verify record_trade() called after journal parse with BUY/SELL
- Verify exception safety (record_trade failure doesn't break completion)
- Verify guard state populated after completion

### 4.2 Test persistence state bounds
- Verify dict doesn't grow beyond cap
- Verify LRU eviction preserves production tickers

---

## Dependency Order

Batch 1 → Batch 2 → Batch 3 → Batch 4

## What NOT to Implement (Deferred)

- **VWAP session reset (H17):** Low impact, one sub-signal in 5-signal composite
- **Metals POSITIONS lock (H31):** 7,667 line file, too risky for autonomous change
- **DST fallback fix (IC-R4-3):** Needs DST transition testing
- **Signal registry centralization:** Large refactor, not a bug fix
- **data/*.py → portfolio/data/ package:** Correct but too large for this session
- **exit_optimizer / microstructure integration:** Dead code, needs design decision first
