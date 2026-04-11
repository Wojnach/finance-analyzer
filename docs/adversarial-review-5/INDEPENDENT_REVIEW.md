# Independent Adversarial Review — Round 5 (2026-04-11)

**Reviewer**: Claude Opus 4.6 (primary author, cross-cutting focus)
**Scope**: Full codebase cross-cutting analysis, focusing on money-losing bugs,
disconnected risk gates, and thread safety.
**Baseline**: commit 935c40c (main, post fix/queue-2026-04-11 merge)

---

## Executive Summary

Round 5 confirms that the **two most critical risk gates in the system remain disconnected**:

1. **check_drawdown()** (20% circuit breaker) — never called from production. Dead code since inception. Flagged in Rounds 3, 4, and now 5. Zero callers outside tests.
2. **record_trade()** (overtrading prevention) — never called from production. The cooldown guards depend on trade timestamps that are never recorded.

The 11 fixes from today's fix queue (fix/queue-2026-04-11) are **all correctly implemented**
and well-documented. The fixes for Playwright thread safety (A-AV-1), account whitelist
(A-AV-2), subprocess tree kill (A-IN-2), and drawdown peak streaming (A-PR-2) are solid.

**New findings** focus on an fx_rate fallback that would cause 10x portfolio miscalculation,
and continued thread-safety gaps in the metals subsystem.

---

## STILL-OPEN from Prior Rounds (Confirmed)

### IR-SO-1: check_drawdown() never called [P0 — 3 rounds open]
- **File**: `portfolio/risk_management.py:86` (definition)
- **Impact**: The 20% drawdown circuit breaker is completely inert. Portfolio can lose
  100% and the system won't stop trading. This is the single most critical finding
  across all 5 review rounds.
- **Evidence**: `grep -rn "check_drawdown" portfolio/ main.py` returns ZERO production callers.
  Only called in `tests/test_risk_management.py`.
- **Fix**: Call `check_drawdown()` in `main.py` or `reporting.py` once per cycle.
  Surface `breached=True` to Layer 2 context and agent_summary. Block new BUYs when breached.

### IR-SO-2: POSITIONS dict shared without lock [P1 — 2 rounds open]
- **File**: `data/metals_loop.py:662` (module-level dict), mutated throughout
- **Impact**: Main 60s cycle, 10s silver fast-tick, and fill handlers all mutate
  `POSITIONS` concurrently. Dict mutations are NOT atomic in Python when interleaved
  with `_save_positions()`. A fill handler can overwrite changes from a concurrent
  reconciliation pass, causing position state to drift from broker reality.
- **Fix**: Wrap all POSITIONS access in a `threading.Lock()`. Add a
  `_positions_lock = threading.Lock()` at module level.

### IR-SO-3: Stop-loss failure leaves naked position [P1 — 2 rounds open]
- **File**: `data/metals_loop.py:4109-4121` (_handle_buy_fill)
- **Impact**: When hardware trailing stop placement fails (API error, auth expired),
  the position is active and tracked but has NO broker-level stop protection.
  The code alerts via Telegram but takes no corrective action: no retry, no
  software-based trailing stop fallback, no flag to prevent additional buys.
  In a flash crash, the naked position could hit the warrant barrier.
- **Fix**: (a) Retry stop placement up to 3 times with backoff. (b) On persistent
  failure, set a `naked_position` flag that blocks new buys until resolved.
  (c) Add a software trailing stop fallback that triggers market sell at ATR-based level.

### IR-SO-4: metals_loop raw open() for agent log [P2 — 2 rounds open]
- **File**: `data/metals_loop.py:6051` — `log_fh = open("data/metals_agent.log", "a")`
- **Impact**: Non-atomic file I/O. If the process crashes mid-write, the log file
  could have a partial line. Also holds the file handle across the entire subprocess
  lifetime.
- **Fix**: Use a proper logging handler or `atomic_append_jsonl` for structured entries.

### IR-SO-5: VWAP cumulative from bar 0 [P2 — 2 rounds open]
- **File**: `portfolio/signals/volume_flow.py` (VWAP computation)
- **Impact**: VWAP is computed from the first bar in the DataFrame, not reset per
  trading session. For 24/7 crypto this is arguably correct, but for stocks and
  metals with session boundaries, the anchored VWAP drifts and becomes less
  meaningful as the session progresses.

---

## NEW Findings

### IR-1: fx_rate fallback 1.0 in risk_management [P1]
- **File**: `portfolio/risk_management.py:66`
- **Code**: `fx_rate = agent_summary.get("fx_rate", 1.0)`
- **Impact**: If agent_summary has no `fx_rate` key (early startup, corrupt file,
  missing data), the portfolio value is computed as if 1 USD = 1 SEK. The actual
  rate is ~10.85, so this undervalues the portfolio by ~10x. If check_drawdown()
  were actually called, this would trigger the circuit breaker immediately due to
  a phantom 90% drawdown.
- **Why it matters now**: If someone fixes IR-SO-1 (wiring check_drawdown), this
  latent bug becomes a P0 — the system would halt trading on every startup until
  fx_rate populates.
- **Fix**: Use `fetch_usd_sek()` as fallback instead of 1.0. Or require fx_rate
  to be present before computing drawdown.

### IR-2: record_trade() never called from production [P1]
- **File**: `portfolio/trade_guards.py:177` (definition)
- **Evidence**: grep shows callers only in `tests/test_trade_guards.py` and
  `portfolio/golddigger/risk.py` (separate bot, different function).
- **Impact**: The ticker_cooldown guard checks "time since last trade" — but
  since no trade is ever recorded, every check sees infinite elapsed time and
  always passes. The position rate limit similarly never fires because
  `new_position_timestamps` is never populated. The C4 warning at line 278
  detects this and logs a warning, but doesn't block trading.
- **Interaction**: This compounds with IR-SO-1. The system has ZERO automated
  risk protection: no drawdown circuit breaker AND no overtrading prevention.
- **Fix**: Call `record_trade()` from the Layer 2 journal-writing path (wherever
  BUY/SELL decisions are committed to portfolio state).

### IR-3: _streaming_max raw open() [P2]
- **File**: `portfolio/risk_management.py:37`
- **Code**: `with open(history_path, encoding="utf-8") as f:`
- **Impact**: The A-PR-2 fix introduced this to stream the full JSONL file for
  peak detection. Uses raw open() instead of file_utils, violating Rule 4.
  On Windows, concurrent writes could cause a partial last line, but json.loads
  catches DecodeError so it won't crash. Still a style/safety violation.
- **Fix**: Use a streaming variant of load_jsonl, or accept the raw open as
  intentional (document the exception).

### IR-4: Metals config loading via raw open() [P2]
- **File**: `data/metals_loop.py:688`
- **Code**: `with open(path, encoding="utf-8") as _cf: return json.load(_cf)`
- **Impact**: Bypasses file_utils.load_json(). If config.json is being written
  by another process simultaneously, a partial read could raise JSONDecodeError
  and crash the metals loop (the except block calls sys.exit(1)).
- **Fix**: Use `file_utils.load_json()` with graceful default.

### IR-5: _METALS_LOOP_START_TS import-time init [P2]
- **File**: `data/metals_loop.py:667`
- **Code**: `_METALS_LOOP_START_TS: float = time.time()`
- **Impact**: Set at import time, reassigned in main(). Any code that references
  it before main() runs sees the import-time value. In tests, this means
  session-relative drawdown calculations use the wrong baseline.
- **Status**: Flagged as "PARTIAL" in Round 4, unchanged.

### IR-6: _extract_ticker hardcoded default to XAG-USD [P3]
- **File**: `portfolio/agent_invocation.py:107`
- **Code**: `return "XAG-USD"  # default to silver`
- **Impact**: When trigger reasons don't mention a specific ticker (e.g., fear &
  greed extreme, macro event), the Layer 2 agent gets XAG-USD context regardless
  of what actually triggered. A crypto-specific trigger like "BTC funding rate
  extreme" would default to analyzing silver.
- **Fix**: Return None instead of hardcoded XAG-USD, and let the caller provide
  the most relevant ticker.

### IR-7: _handle_buy_fill POSITIONS race window [P1]
- **File**: `data/metals_loop.py:4032-4085`
- **Impact**: `_handle_buy_fill` reads POSITIONS, modifies it in multiple steps
  (lines 4056-4083), then calls `_save_positions(POSITIONS)` at line 4085. Between
  the read and the save, the 10s silver fast-tick could also modify POSITIONS. Since
  there's no lock (IR-SO-2), the save captures whichever thread wrote last,
  potentially losing the other thread's changes. Example: a fill updates units while
  a concurrent reconciliation marks a position inactive — one change is lost.
- **Fix**: Same as IR-SO-2 — a threading.Lock() around all POSITIONS mutations.

### IR-8: shared_state cache eviction under lock [P3]
- **File**: `portfolio/shared_state.py:54-66`
- **Impact**: Cache eviction (sorting 512+ entries, checking timestamps) runs
  while holding _cache_lock. All other threads are blocked from cache reads during
  eviction. In the worst case (512 entries, many expired), this could block signal
  computation for 10-100ms. Not a correctness bug, but could cause the BUG-178
  slow-cycle detection to fire unnecessarily.
- **Fix**: Collect keys to evict under the lock (O(n) scan), release lock, then
  do the eviction. Or reduce _CACHE_MAX_SIZE to prevent large eviction batches.

---

## Fix Queue Verification (11 fixes from today)

| Fix ID | Status | Verified Correct |
|--------|--------|-----------------|
| A-AV-1 (Playwright _pw_lock) | FIXED | Yes — RLock prevents deadlock on reentrant calls |
| A-AV-2 (Account whitelist) | FIXED | Yes — ALLOWED_ACCOUNT_IDS = {"1625505"} |
| A-PR-2 (Drawdown peak streaming) | FIXED | Yes — _streaming_max reads full file. But uses raw open() (IR-3) |
| A-PR-3 (Portfolio validator load_json) | FIXED | Yes — uses file_utils.load_json |
| A-IN-2 (Subprocess tree kill) | FIXED | Yes — kills process group on TimeoutExpired |
| A-IN-3 (In-process invoke lock) | FIXED | Yes — threading.Lock() around invoke_claude |
| A-MC-2 (fetch_usd_sek) | FIXED | Yes — replaces usdsek=1.0 |
| A-MC-4 (Real entry_ts) | FIXED | Yes — persists bought_ts from fill, not now() |
| A-DE-4 (yfinance MultiIndex) | FIXED | Yes — flattens MultiIndex columns |
| A-DE-5 (Onchain ISO ts coerce) | FIXED | Yes — coerces ISO strings to epoch |
| A-SM-1+2 (Gap-fill guard + GARCH schema) | FIXED | Yes — explicit guard on widening gap |
| Accuracy gate 0.45→0.47 | FIXED | Yes — ACCURACY_GATE_THRESHOLD = 0.47 |

---

## Summary

| Severity | Still-Open | New | Total |
|----------|-----------|-----|-------|
| P0 | 1 (check_drawdown) | 0 | 1 |
| P1 | 3 (POSITIONS lock, naked position, record_trade) | 3 (fx_rate, record_trade, fill race) | 4 unique |
| P2 | 2 (raw open, VWAP) | 4 (streaming open, config open, START_TS, cache eviction→P3) | 5 |
| P3 | 0 | 2 (extract_ticker, cache eviction) | 2 |
| **Total** | **6** | **8** | **12 unique** |

Note: IR-SO-2 and IR-7 are the same root cause (POSITIONS lock), and IR-SO-1 and IR-2
are complementary (both risk gates disconnected).

**The system currently has ZERO automated risk protection at runtime.**
Both the drawdown circuit breaker (check_drawdown) and overtrading prevention
(record_trade) are implemented but disconnected from production code paths.
