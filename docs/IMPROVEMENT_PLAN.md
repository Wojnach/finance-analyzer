# Improvement Plan — Auto-Session 2026-04-11

Updated: 2026-04-11
Branch: improve/auto-session-2026-04-11
Status: **COMPLETE**

## Session Context

Continuing from the 2026-04-10 after-hours session which shipped per-ticker directional
accuracy (6ec4be9) and raised the directional gate threshold to 0.40 (0d81282).

The previous session identified P0: **direction-specific weight scaling** as the next
highest-impact change. The data foundation is already in place — `buy_accuracy`/`sell_accuracy`
flow through accuracy_stats.py, ticker_accuracy.py, and the BUG-158 override in signal_engine.py.

---

## 1. Bugs & Problems Found

### BUG-182: Consensus weight uses overall accuracy, not directional accuracy (HIGH)
- **File**: `portfolio/signal_engine.py:842`
- **Issue**: `weight = acc if samples >= 20 else 0.5` uses overall accuracy (`acc`) for all
  signals. But if qwen3 has overall 59.8% but BUY=30.4% and SELL=74.3%, a BUY vote from
  qwen3 gets weight 0.598, massively overvaluing a direction where it's near coin-flip.
- **Fix**: When vote is BUY, use `buy_accuracy` as weight; when SELL, use `sell_accuracy`.
  Fall back to overall accuracy when directional samples are insufficient.
- **Impact**: Directly affects consensus quality. Signals voting in their weak direction
  get over-weighted, potentially flipping consensus incorrectly.
- **Risk**: Low — additive change, falls back to current behavior when directional data
  is unavailable.

### BUG-183: autonomous.py uses global throttle, suppresses multi-ticker signals (MEDIUM)
- **File**: `portfolio/autonomous.py:811-813`
- **Issue**: `_update_throttle()` stores a single `last_send` timestamp. If any ticker triggers
  a HOLD message, ALL tickers are throttled for 30 minutes — even if BTC-USD has a genuine
  BUY signal.
- **Fix**: Per-ticker throttle tracking.
- **Impact**: Missed signals during the throttle window.
- **Risk**: Low — only affects autonomous mode (Layer 3 fallback).

### BUG-184: trade_guards.py has no locking on state reads/writes (MEDIUM)
- **File**: `portfolio/trade_guards.py`
- **Issue**: `check_overtrading_guards()` and `record_trade()` both read/write to
  `trade_guard_state.json` without synchronization. In the ThreadPoolExecutor ticker
  processing, concurrent threads could read stale state and bypass cooldowns.
- **Fix**: Add a threading.Lock around state access.
- **Risk**: Low — additive. Worst case: slightly increased contention.

### INFO: Several agent-reported bugs verified as correct code
- **trigger.py:345** (first-of-day T3): Correct — `last_trigger_date` is only set when
  triggers fire, so first trigger of each day correctly returns T3.
- **trigger.py:167-181** (BUY↔SELL flip reasons): By design — section #1 handles HOLD→BUY/SELL,
  section #2 handles direction flips with sustained logic.
- **signal_registry.py:78-89** (None load): Handled at signal_engine.py:1637-1638.
- **autonomous.py:100-101** (empty journal): Has `if prev_entries else None` guard.

---

## 2. Architecture Improvements

### Direction-Specific Weight Scaling (P0 from previous session)
- Use `buy_accuracy`/`sell_accuracy` as the weight in `_weighted_consensus()` instead of
  overall accuracy, when the signal is voting in that specific direction.
- **Why**: The #1 finding from the 2026-04-10 signal audit: directional asymmetries of 15-44pp
  exist (qwen3 BUY 30.4% vs SELL 74.3%). Using overall accuracy masks these failures.
- **Enables**: More accurate consensus, fewer bad trades from signals voting in their weak
  direction.

### Per-Ticker Throttle in autonomous.py
- Replace global `last_send` with `{ticker: last_send_ts}` dict in throttle file.
- **Why**: The current global throttle can suppress valid signals for up to 30 minutes.
- **Enables**: Independent throttling per ticker.

---

## 3. Refactoring TODOs

### shared_state.py cache size increase
- **File**: `portfolio/shared_state.py:21`
- **Issue**: `_CACHE_MAX_SIZE = 256` may cause cache thrashing with 5 tickers × 7 timeframes.
- **Fix**: Increase to 512.

### volatility.py MIN_ROWS comment mismatch
- **File**: `portfolio/signals/volatility.py`
- **Issue**: Comment says "BB squeeze lookback (120) is binding constraint" but MIN_ROWS=50.
- **Fix**: Verify actual BB squeeze lookback and update comment or constant accordingly.

---

## 4. Ordering — Batches

### Batch 1: Direction-Specific Weight Scaling (2 files + tests) — P0
1. `portfolio/signal_engine.py` — modify `_weighted_consensus()` to use directional accuracy
2. `tests/test_signal_engine.py` — add tests for directional weight scaling

### Batch 2: Autonomous Per-Ticker Throttle (1 file + tests) — P1
1. `portfolio/autonomous.py` — replace global throttle with per-ticker dict
2. `tests/test_autonomous.py` — test per-ticker throttle

### Batch 3: Trade Guards Locking (1 file + tests) — P1
1. `portfolio/trade_guards.py` — add threading.Lock for thread safety
2. Add/update tests for concurrent access patterns

### Batch 4: Code Quality (2 files) — P2
1. `portfolio/shared_state.py` — increase _CACHE_MAX_SIZE to 512
2. `portfolio/signals/volatility.py` — fix comment mismatch

---

## 5. Risk Assessment

- **Batch 1** (direction-specific weights): Moderate impact — changes consensus computation
  but falls back to current behavior when directional data is insufficient. Full test
  coverage required.
- **Batch 2** (per-ticker throttle): Low risk — only affects autonomous mode, backwards
  compatible with existing throttle file format.
- **Batch 3** (trade guards lock): Low risk — additive synchronization.
- **Batch 4** (constants/comments): Zero risk — no behavioral change.
