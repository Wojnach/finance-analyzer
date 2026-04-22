# Independent Adversarial Review — 2026-04-22

## Methodology

Deep code-level review of the finance-analyzer codebase, partitioned into 8 subsystems.
Each subsystem reviewed for: logic bugs, silent failures, concurrency issues, state corruption,
edge cases, security, and design problems. Severity: P1 (will cause wrong trades/data loss),
P2 (degraded accuracy/reliability), P3 (code quality/maintainability).

---

## 1. SIGNALS-CORE

### SC-P1-01: `_persistence_state` unbounded growth on cold-start seeding
**File:** `portfolio/signal_engine.py:260-268`
**What:** Cold-start path seeds `_persistence_state[ticker]` with cycles=`_PERSISTENCE_MIN_CYCLES`
for all non-HOLD votes and returns all votes unfiltered ("first cycle — trust all signals").
But the eviction logic (line 260-262) only fires when `ticker not in _persistence_state` AND
the dict is at cap. If the same 5 tickers cycle repeatedly, eviction never fires. However,
test probes or rogue callers with arbitrary ticker names grow the dict unboundedly until
the cap (32) is hit, at which point eviction removes half — but this is a dict of dicts of
dicts, so memory is proportional to 32 × 36 signals × 3 fields = ~3.5K entries. Low risk
in production (5 tickers), but **the cold-start seeding at full persistence is a logic bug**:
it means the first cycle after a restart lets ALL signals vote regardless of whether they've
been stable, defeating the purpose of the persistence filter.

**Impact:** After every loop restart, the first cycle's consensus includes unfiltered noise
signals. With the loop restarting via PF-DataLoop on logon + auto-restart, this happens
multiple times daily.

**Fix:** Cold-start should seed cycles=1 (not _PERSISTENCE_MIN_CYCLES) and return votes
unfiltered only when there's no alternative. Or: seed cycles=0 and let cycle 2 be the
first to filter. The current approach of seeding at MIN_CYCLES AND returning unfiltered
does double-permissive: the seed guarantees cycle 2 also passes.

### SC-P2-01: ADX cache key collision via `id(df)` reuse
**File:** `portfolio/signal_engine.py:1960`
**What:** The ADX cache key includes `id(df)` plus `len(df)` and last close. CPython can
reuse `id()` values for new objects at the same address. The mitigation (adding len + last
close) reduces but doesn't eliminate collisions — two DataFrames of the same length and
last close value at the same address will hit the cache incorrectly. This is unlikely but
not impossible in a GC-heavy threaded environment.

**Impact:** Wrong ADX value used for regime detection → wrong regime → wrong signal weights.

**Fix:** Use a hash of the DataFrame content (`hash(df.values.tobytes())`) or a cycle
counter as part of the key.

### SC-P2-02: `_weighted_consensus` accuracy_data sanitization drops fields silently
**File:** `portfolio/signal_engine.py:1523-1560`
**What:** The 13-round Codex sanitization logic is extremely complex (40 lines of nested
conditionals for field-pair drop semantics). The complexity itself is a bug risk — future
maintainers are unlikely to understand the drop-together semantics correctly. Additionally,
the comment on line 1550-1551 says "Accuracy present (clean), count field absent — keep acc"
which means a signal with accuracy=0.51 but no sample count passes the accuracy gate with
the downstream default of `samples=0` (line 1724), which is below ACCURACY_GATE_MIN_SAMPLES
(30), so it won't be gated. This is actually correct behavior (insufficient samples = no
gating), but it's fragile — the invariant "no count → no gating" is maintained by accident
across two separate code locations 200 lines apart.

**Impact:** Low immediate risk, but the sanitization complexity makes the consensus path
harder to reason about and more likely to harbor future regressions.

### SC-P2-03: Agreement rate treats BUY-BUY and SELL-SELL as equally "agreeing"
**File:** `portfolio/signal_engine.py:961-978`
**What:** `_compute_agreement_rate` counts `va == vb` as agreement, including SELL-SELL.
Two signals that both always say SELL will have 100% agreement and be clustered, which
correctly identifies redundancy. But if one signal's SELL is contrarian (mean-reversion)
and the other's is trend-following, they're agreeing for different reasons. The clustering
conflates agreement-by-coincidence with agreement-by-redundancy.

**Impact:** Potentially wrong correlation groups → wrong deduplication weights. The static
fallback groups are hand-curated and correct, so this only matters when dynamic groups
take over (2h cache, sufficient signal_log data).

### SC-P3-01: 20+ bare `except Exception:` blocks in signal_engine.py
**File:** `portfolio/signal_engine.py` (multiple locations)
**What:** At least 20 locations catch `Exception` with no logging or re-raise, just silent
fallback. While many are in diagnostic/optional paths, some (like line 1999 in ADX
computation) silently swallow computation errors that could indicate data corruption.

---

## 2. ORCHESTRATION

### OR-P1-01: Agent not killed when loop restarts during active invocation
**File:** `portfolio/main.py` + `portfolio/agent_invocation.py`
**What:** If the main loop process crashes and restarts (PF-DataLoop auto-restart),
`_agent_proc` is `None` in the new process — the previously spawned Claude CLI subprocess
is now orphaned. The singleton lock prevents two loops from running, but the orphaned
agent may still be writing to journal, portfolio state, or placing orders.

**Impact:** Two Layer 2 agents could be running simultaneously — one orphaned, one newly
spawned. Both could make conflicting trade decisions.

**Fix:** On startup, check for orphaned `claude` processes spawned by the previous loop
PID (recorded in agent.log or by convention). Or: use a PID file for the agent subprocess
that's checked on startup.

### OR-P2-01: ThreadPoolExecutor not shut down on exception path
**File:** `portfolio/main.py:595`
**What:** Comment says "OR-I-001: avoid context manager — __exit__ calls shutdown(wait=True)
which blocks the loop when threads hang past the timeout." The pool is created but
`pool.shutdown(wait=False)` is never called after the `as_completed` loop. If an exception
interrupts the `as_completed` iteration, worker threads become zombies.

**Impact:** Thread leak on exception → eventual resource exhaustion over many cycles.

### OR-P2-02: Massive re-export block in main.py
**File:** `portfolio/main.py:112-236`
**What:** main.py re-exports ~80+ symbols from 12+ modules. This creates a fragile
import graph where removing or renaming any symbol in a dependency breaks main.py's
public interface. Several re-exports are prefixed with `_` (private) but are nonetheless
imported by external code. This pattern makes refactoring extremely risky.

**Impact:** Technical debt. Every module change requires checking if main.py consumers
depend on the re-exported name.

### OR-P2-03: `_extract_triggered_tickers` regex misses some patterns
**File:** `portfolio/main.py:244`
**What:** The regex `r'^([A-Z][A-Z0-9]*(?:-[A-Z]+)?)\s+(?:consensus|moved|flipped)'`
requires the ticker to be at the start of the string. Trigger reasons like
"post-trade reassessment" or "F&G crossed extreme" don't match this pattern, so
`_extract_triggered_tickers` returns empty for those trigger types.

**Impact:** When the only trigger reason is non-ticker (e.g., post-trade reassessment),
the ticker set is empty. Downstream code that uses this set to filter processing
may process no tickers at all.

---

## 3. PORTFOLIO-RISK

### PR-P1-01: `_streaming_max` reads entire JSONL file on every drawdown check
**File:** `portfolio/risk_management.py:22-51`
**What:** Every call to `check_drawdown()` reads the ENTIRE `portfolio_value_history.jsonl`
file line-by-line to find the historical peak. This file grows by ~1440 entries/day
(one per 60s cycle). After 30 days, it's ~43K entries. After 1 year, ~525K entries.
The function runs every cycle (60s) in the agent_invocation path.

**Impact:** Increasing I/O overhead over time. Currently manageable but will degrade.

**Fix:** Cache the streaming max with a TTL (e.g., 5 minutes). The peak can only
increase, so caching is safe — a new-peak check only needs the recent tail.

### PR-P2-01: `check_drawdown` falls back to cash-only value when agent_summary is empty
**File:** `portfolio/risk_management.py:126-139`
**What:** When `agent_summary.json` is empty or missing, the function falls back to
`cash_sek` as the portfolio value. If the portfolio holds positions that are deeply
underwater, the cash-only value will look fine and the circuit breaker won't trip.
The warning log is good (added in the 2026-04-17 adversarial review), but the
fallback itself is dangerous.

**Impact:** Drawdown circuit breaker blind spot when agent_summary feed goes stale.
The 2026-03-27 → 2026-04-13 silent outage (--bare breaking OAuth) would have left
the summary stale for 3 weeks.

### PR-P2-02: `portfolio_value` doesn't handle SEK-denominated holdings
**File:** `portfolio/portfolio_mgr.py:162-179`
**What:** The function assumes all holdings are priced in USD and multiplies by `fx_rate`.
But Avanza warrants (MINI-SILVER, XBT-TRACKER) are priced in SEK. If these appear in
the portfolio state, their value would be incorrectly multiplied by fx_rate (~11x).

**Impact:** Overstated portfolio value for SEK-denominated holdings. May not affect
production if warrants are tracked in a separate state file.

### PR-P3-01: `_rotate_backups` suffix naming is confusing
**File:** `portfolio/portfolio_mgr.py:53-59`
**What:** Backup naming uses `.json.bak`, `.json.bak2`, `.json.bak3` but the iteration
starts from `_MAX_BACKUPS` down to 2, with special-casing for `i > 2`. This is fragile
and hard to verify by inspection.

---

## 4. METALS-CORE

### MC-P1-01: metals_loop.py is a 3000+ line monolith
**File:** `data/metals_loop.py`
**What:** The entire metals trading loop — data fetch, signal computation, Avanza
integration, LLM inference, stop-loss management, fast-tick monitoring — is in a
single file. Any bug in any component can crash the entire metals trading system.
Error handling is inconsistent (mix of `except Exception`, bare excepts, and specific
catches).

**Impact:** Single point of failure. Any unhandled exception in the fast-tick monitor
(10s cycle) can crash the 60s main cycle.

### MC-P2-01: GPU lock contention between metals_loop and main loop
**File:** `portfolio/gpu_gate.py` + `data/metals_loop.py`
**What:** Both the main loop (Ministral/Qwen3 signals via signal_engine) and the
metals loop (local LLM inference) compete for the GPU via `gpu_lock.py`. If the
metals loop holds the GPU lock during a main loop cycle, the main loop's GPU signals
time out and return stale/None data.

**Impact:** Intermittent signal degradation for GPU-dependent signals.

### MC-P2-02: `fin_snipe_manager.py` at 1737 lines is the second-largest module
**File:** `portfolio/fin_snipe_manager.py`
**What:** Complex order management logic concentrated in a single large file.
High cognitive load for review and maintenance. Order placement bugs here directly
affect real money.

---

## 5. AVANZA-API

### AV-P1-01: `_check_telegram_confirm` doesn't prevent replay attacks
**File:** `portfolio/avanza_orders.py:146-200`
**What:** A CONFIRM message is processed if it arrives in the Telegram getUpdates
response since the last offset. If an attacker (or the user accidentally) sends
multiple CONFIRM messages, only one order is confirmed per cycle, but on the NEXT
cycle, the remaining CONFIRMs could confirm DIFFERENT orders. The offset advances
past all updates, but `found_confirm` is set to True for any CONFIRM — it doesn't
check message timestamps against order creation time.

**Impact:** Low risk (requires access to the Telegram chat), but a stale CONFIRM
from a previous order could auto-confirm a new unrelated order.

**Fix:** Include a nonce or order ID in the CONFIRM message (e.g., "CONFIRM abc123").

### AV-P2-01: Playwright browser context never explicitly closed on loop shutdown
**File:** `portfolio/avanza_session.py:131-148`
**What:** `_get_playwright_context()` creates `_pw_instance`, `_pw_browser`, `_pw_context`
as module-level globals but there's no cleanup function. On loop shutdown, the Chromium
process becomes orphaned.

**Impact:** Resource leak — orphaned Chromium processes accumulate over restarts.

### AV-P2-02: RLock on `_pw_lock` allows recursive entry within same thread
**File:** `portfolio/avanza_session.py:49`
**What:** The upgrade from Lock to RLock (A-AV-1) allows the same thread to re-enter
the lock. This is intentional for api_get/api_post calling _get_playwright_context,
but it also means that if api_get is called recursively (e.g., a retry within a retry),
the lock doesn't protect against self-interference within the same thread.

### AV-P2-03: `ALLOWED_ACCOUNT_IDS` is hardcoded, not enforced at order time
**File:** `portfolio/avanza_session.py:38`
**What:** `ALLOWED_ACCOUNT_IDS = {"1625505"}` is defined but I don't see enforcement
at the order placement level in avanza_orders.py. The actual enforcement may be in
avanza_control.py, but if any code path calls the Avanza API directly (bypassing
avanza_control), orders could go to unintended accounts.

---

## 6. SIGNALS-MODULES

### SM-P2-01: 141 `.iloc[-1]` accesses across 20 signal modules without empty-check
**What:** Signal modules extensively use `df[col].iloc[-1]` to get the latest value.
If the DataFrame is empty (0 rows), this throws `IndexError`. Most modules have a
`len(df) < N` guard at the top, but the guard values vary (30, 50, 200) and some
internal sub-functions skip the check.

**Impact:** A single bad API response returning 0 rows could crash the signal for
that ticker. The outer `try/except` in signal_engine catches it, but the signal
silently becomes HOLD with no diagnostic of what went wrong.

### SM-P2-02: Signal modules don't validate DataFrame columns
**What:** Signal modules assume columns like `open`, `high`, `low`, `close`, `volume`
exist. If a data source returns a DataFrame with different column names (e.g., `Open`
vs `open`), the module crashes.

**Impact:** Data source changes could break all signals simultaneously.

### SM-P2-03: `forecast.py` (964 lines) mixes I/O with computation
**File:** `portfolio/signals/forecast.py`
**What:** The forecast signal module directly loads model files, runs inference, and
handles caching internally. This makes it impossible to test in isolation and creates
a tight coupling between the signal interface and the model loading infrastructure.

### SM-P3-01: Inconsistent return formats across signal modules
**What:** Some modules return `{"action": X, "confidence": Y, "sub_signals": {}}`,
others return additional fields. The `_validate_signal_result` function in
signal_engine.py normalizes this, but the inconsistency makes debugging harder.

---

## 7. DATA-EXTERNAL

### DE-P1-01: `econ_dates.py` has hardcoded FOMC/CPI/NFP dates
**File:** `portfolio/econ_dates.py`
**What:** Economic calendar dates are hardcoded. When dates expire, the econ_calendar
signal silently stops producing risk-off signals near major events.

**Impact:** Missing risk-off signals near unscheduled/unupdated economic events could
lead to trades during volatile periods.

### DE-P2-01: API rate limiter state not persisted across restarts
**File:** `portfolio/shared_state.py` (rate limiters)
**What:** Rate limiters (Binance, Alpha Vantage, yfinance) are in-memory. On loop
restart, the limiter resets and the first cycle may burst past API limits, triggering
bans or 429 errors.

**Impact:** Post-restart API ban risk, especially for Alpha Vantage (25/day hard limit).

### DE-P2-02: `data_collector.py` silent fallback to yfinance
**What:** When Binance or Alpaca fail, the collector may fall back to yfinance with
different data quality, different timestamps, and different candle semantics. The
downstream signal computation doesn't know which source provided the data.

**Impact:** Signals computed on yfinance data may have different accuracy characteristics
than those computed on Binance data, but the accuracy tracking treats them identically.

### DE-P2-03: `fx_rates.py` caching may serve stale rates during high volatility
**What:** USD/SEK rate is cached with a TTL. During high-volatility events (central
bank decisions, market crises), the rate can move significantly within the cache window.
Portfolio value calculations use the stale rate.

**Impact:** Incorrect portfolio valuations during FX volatility events. The drawdown
circuit breaker may not trip or may trip spuriously.

---

## 8. INFRASTRUCTURE

### IF-P2-01: `atomic_append_jsonl` sidecar lock file can accumulate
**File:** `portfolio/file_utils.py:187+`
**What:** The sidecar lockfile pattern creates `.lock` files alongside JSONL files.
If a process crashes while holding the lock (e.g., between lock acquire and release),
the lock file remains but is not stale-checked. On Windows with msvcrt, this is
handled (lock is released on process exit), but on unexpected power loss, the lock
file could become permanently orphaned.

**Impact:** Low risk on Windows (OS cleans up file locks on process death).

### IF-P2-02: `_cached` dogpile prevention returns None when no stale data exists
**File:** `portfolio/shared_state.py:88`
**What:** When a cache miss occurs and another thread is already loading the key,
if there's no stale data available, the function returns `None`. The caller must
handle `None` correctly. In signal_engine.py, many callers use patterns like
`data = _cached(...) or {}` which silently converts None to empty dict, losing
the signal that the data is unavailable vs. empty.

**Impact:** Signals may compute on empty data (treated as "no signal") rather than
properly recognizing "data unavailable" as a condition requiring HOLD.

### IF-P2-03: Health monitoring can appear healthy while Layer 2 is broken
**File:** `portfolio/health.py`
**What:** `update_health()` updates heartbeat on every Layer 1 cycle. Layer 1 runs
independently of Layer 2. So `check_staleness()` will show a fresh heartbeat even
if Layer 2 is completely broken (as happened during the 3-week silent outage). The
`check_agent_silence()` function addresses this, but it's a separate check that
must be explicitly called.

**Impact:** Dashboard/monitoring that only checks heartbeat will miss Layer 2 failures.

### IF-P3-01: `load_json` returns `default` for corrupt JSON with only WARNING log
**File:** `portfolio/file_utils.py:84-86`
**What:** When a JSON file is corrupt, `load_json` logs WARNING and returns the
default value. For non-critical files this is fine, but for portfolio_state.json,
this means a corrupt portfolio state silently becomes the default (500K SEK, no
holdings) — effectively losing the entire trade history. The `_load_state_from`
in portfolio_mgr.py has backup recovery for this case, but other callers of
`load_json` on critical files may not.

---

## Cross-Cutting Findings

### XC-P2-01: No integration test for the full signal→consensus→trigger→agent path
**What:** The system has ~6000 unit tests, but no end-to-end test that verifies a
signal change triggers Layer 2 invocation and produces a journal entry. The 3-week
silent outage would have been caught by such a test.

### XC-P2-02: Magic numbers scattered across signal_engine.py
**File:** `portfolio/signal_engine.py`
**What:** At least 40+ named constants for thresholds, gates, penalties, etc. While
individually documented, the interaction between them is complex and untested. A
change to any single constant may have cascading effects on consensus behavior.

### XC-P3-01: Inconsistent error handling philosophy
**What:** Some modules use strict error handling (require_json, raise on failure),
others use lenient handling (load_json, return default). The choice between these
approaches seems arbitrary rather than following a consistent policy based on
criticality.

---

## Summary Statistics

| Subsystem | P1 | P2 | P3 |
|-----------|----|----|-----|
| signals-core | 1 | 3 | 1 |
| orchestration | 1 | 3 | 0 |
| portfolio-risk | 1 | 2 | 1 |
| metals-core | 1 | 2 | 0 |
| avanza-api | 1 | 3 | 0 |
| signals-modules | 0 | 3 | 1 |
| data-external | 1 | 3 | 0 |
| infrastructure | 0 | 3 | 1 |
| cross-cutting | 0 | 2 | 1 |
| **TOTAL** | **6** | **24** | **5** |

## Top Priority Fixes

1. **SC-P1-01**: Fix persistence filter cold-start seeding (double-permissive)
2. **OR-P1-01**: Detect and kill orphaned agent processes on loop restart
3. **PR-P1-01**: Cache streaming_max to avoid reading full JSONL every cycle
4. **AV-P1-01**: Add nonce/order-ID to Telegram CONFIRM flow
5. **DE-P1-01**: Switch econ_dates from hardcoded to API-sourced dates
6. **MC-P1-01**: Extract metals_loop.py into smaller modules
