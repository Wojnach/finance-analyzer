# Independent Adversarial Review — 2026-04-12

## Methodology

Reviewed 8 subsystems of the finance-analyzer codebase by reading source code directly,
searching for anti-patterns, and tracing data flow across module boundaries. This review
was conducted independently from the "codex" agent reviews to enable cross-critique.

Focus: bugs, logic errors, race conditions, data corruption risks, and security vulnerabilities
in a system that trades real money.

---

## Subsystem 1: signals-core

### SC-I-001 — P1: Double regime gating nullifies BUG-158 per-ticker exemptions

**File**: `portfolio/signal_engine.py` lines 1743-1766 (outer gate) vs 749 (inner gate)

**Bug**: `generate_signal()` applies regime gating with per-ticker accuracy exemptions
(BUG-158 fix, lines 1751-1759): signals with ≥60% accuracy on a specific ticker are
exempted from regime gating. But `_weighted_consensus()` (called at line 1971) independently
applies its OWN regime gating at line 749:
```python
votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}
```
This uses `_get_regime_gated(regime, horizon)` which returns the FULL gate set WITHOUT
per-ticker exemptions.

**Impact**: The entire BUG-158 exemption mechanism is silently nullified. For example,
`fear_greed` has 93.8% accuracy on XAG-USD but is globally gated in ranging regime.
The outer code exempts it, restoring it to its non-HOLD vote. Then `_weighted_consensus`
re-gates it to HOLD, throwing away the alpha.

**Evidence**: Lines 749 vs 1754-1759: the inner function has no access to `regime_gated_effective`.

**Fix**: Pass `regime_gated_effective` to `_weighted_consensus()` or skip inner regime
gating when the outer caller has already applied it.

### SC-I-002 — P2: Dynamic horizon weights cache stale static fallback

**File**: `portfolio/signal_engine.py` line 475, 489

**Bug**: `_compute_dynamic_horizon_weights()` returns static `HORIZON_SIGNAL_WEIGHTS`
when the accuracy cache is temporarily unavailable (file locked, I/O error). This static
result gets cached by `_cached()` for 1 hour (`_DYNAMIC_HORIZON_WEIGHT_TTL = 3600`).
When the accuracy cache becomes available again (next cycle, ~60s later), the stale
static weights persist for up to 59 more minutes.

**Impact**: One I/O glitch pins horizon weights to static values for an hour.

**Fix**: Return `None` or sentinel from the compute function and have `_cached()` not
cache None results. Or use a shorter TTL for the fallback case.

### SC-I-003 — P3: Unanimity penalty applied AFTER weighted consensus

**File**: `portfolio/signal_engine.py` lines 1200-1211

**Observation**: The unanimity penalty (Stage 5 of `apply_confidence_penalties`) uses
`extra_info["_buy_count"]` and `extra_info["_sell_count"]` which are raw unweighted
vote counts (line 1769-1770), not the weighted buy/sell proportions from
`_weighted_consensus`. So a scenario where 10 signals BUY but 9 of them are
accuracy-gated (leaving 1 real voter) still triggers the unanimity penalty because
raw_buy_count=10.

**Impact**: Spurious unanimity penalties when most signals are gated. Low severity
because the weighted consensus action would already be HOLD in that case.

### SC-I-004 — P3: Correlation group merge is greedy, not transitive-optimal

**File**: `portfolio/signal_engine.py` lines 619-661

**Observation**: The greedy clustering algorithm for dynamic correlation groups depends
on the iteration order of the correlation matrix. Signals A-B (corr 0.8) and B-C
(corr 0.75) get clustered together, but if we iterate C-D (corr 0.71) before B-C,
the groups may differ. This is a minor issue since the clustering is recomputed
every 2 hours.

---

## Subsystem 2: orchestration

### OR-I-001 — P2: ThreadPoolExecutor timeout doesn't kill stuck threads

**File**: `portfolio/main.py` lines 554-595

**Bug**: When `as_completed(futures, timeout=180)` fires, `f.cancel()` is called on
remaining futures. But `cancel()` only prevents not-yet-started tasks; already-running
threads continue executing indefinitely. If a signal makes a blocking network call
(e.g., Binance API timeout set too high), the thread persists.

**Impact**: After multiple timeout events, zombie threads accumulate. Each holds a thread
in the pool, potentially exhausting the 8-worker limit. The ThreadPoolExecutor is
created fresh each cycle (`with` block), so the old pool's `__exit__` calls
`shutdown(wait=True)` which BLOCKS until stuck threads complete — meaning the main
loop itself hangs.

**Severity upgrade**: This is actually P1 because the `with` block's `shutdown(wait=True)`
will block the entire main loop until stuck threads complete or are killed.

**Fix**: Use `shutdown(wait=False, cancel_futures=True)` (Python 3.9+), or use a separate
process pool with kill capability.

### OR-I-002 — P3: main.py re-exports 50+ internal symbols

**File**: `portfolio/main.py` lines 112-234

**Observation**: main.py re-exports over 50 internal symbols from various modules for
backwards compatibility. This creates a fragile dependency web where tests and other
modules import private symbols like `_prev_sentiment` through `main.py`.

**Impact**: Any refactoring of signal_engine.py or shared_state.py risks breaking imports
in tests and external code.

### OR-I-003 — P2: No timeout on `fetch_usd_sek()` call

**File**: `portfolio/main.py` line 417

**Bug**: `fetch_usd_sek()` is called outside the thread pool, before parallel ticker
processing. If the FX rate API hangs, the entire cycle hangs before any signal
processing starts.

---

## Subsystem 3: portfolio-risk

### PR-I-001 — P2: Drawdown uses fallback avg_cost when live prices missing

**File**: `portfolio/risk_management.py` lines 79-82

**Bug**: When `agent_summary` doesn't have live prices for a holding, the portfolio
value falls back to `avg_cost_usd` (entry price). This can drastically misrepresent
current value — if the price has dropped 50% since entry, the drawdown calculation
sees the original entry price instead, missing the real drawdown.

**Impact**: Circuit breaker may not fire during a genuine drawdown if agent_summary is
stale or incomplete.

### PR-I-002 — P3: _streaming_max reads entire JSONL every drawdown check

**File**: `portfolio/risk_management.py` lines 21-51

**Observation**: `_streaming_max` reads the entire `portfolio_value_history.jsonl` file
line-by-line on every call to `check_drawdown()`. For a system running every 60s,
this means reading a growing file 1440 times per day. The file grows by ~1 line per
minute, so after a year it would be ~525K lines read 1440 times per day.

**Fix**: Cache the peak value in a separate file and only scan entries newer than
the last cached peak timestamp.

### PR-I-003 — P2: portfolio_mgr doesn't validate cash_sek is non-negative

**File**: `portfolio/portfolio_mgr.py` lines 64-74

**Bug**: `_validated_state()` merges defaults but doesn't validate that `cash_sek >= 0`.
If a bug in the trading logic causes cash to go negative (double-deduction from a
concurrent trade), the system continues operating with negative cash.

---

## Subsystem 4: metals-core

### MC-I-001 — P2: metals_execution_engine is advisory-only, real execution path unclear

**File**: `data/metals_execution_engine.py` line 1-6

**Observation**: The module header says "intentionally advisory-only: it scores candidate
BUY and SELL limit prices, but it does not place or queue orders." The actual order
execution happens through `avanza_session.py` and `metals_loop.py` directly.

The execution flow is: metals_loop → avanza_session.place_buy_order() with no
intermediate validation step from the execution engine's advisory scores.

### MC-I-002 — P2: ZoneInfo fallback hardcodes UTC+1

**File**: `data/metals_execution_engine.py` lines 47-53, `data/metals_loop.py`

**Bug**: When `zoneinfo` is unavailable, the fallback adds 1 hour to UTC:
```python
now_cet = utc_now + _dt.timedelta(hours=1)
```
This ignores CET/CEST (summer time = UTC+2). During summer months (late March to
late October), this gives the wrong time by 1 hour, affecting trading window
calculations.

**Impact**: During CEST, `hours_to_metals_close()` reports 1 extra hour of trading
time, potentially allowing trades after actual market close.

### MC-I-003 — P2: MIN_BARRIER_DISTANCE_PCT imported from try/except with different default

**File**: `data/metals_execution_engine.py` lines 22-36

**Observation**: When `metals_swing_config` can't be imported, fallback values are used.
The fallback `MIN_BARRIER_DISTANCE_PCT = 15.0` may differ from the production value
in `metals_swing_config.py`. If the import fails silently (e.g., Python path issue),
the system operates with potentially wrong safety parameters.

---

## Subsystem 5: avanza-api

### AV-I-001 — P1: Account whitelist only in avanza_client.py, not enforced in avanza_session.py

**File**: `portfolio/avanza_client.py` line 31

**Bug**: `ALLOWED_ACCOUNT_IDS = {"1625505"}` is defined in `avanza_client.py`, and the
code at line 182 filters positions by this whitelist. However, the Playwright-based
`avanza_session.py` (which is the primary trading path for metals warrants) may not
have the same account filtering.

**Impact**: If a code path through `avanza_session.py` doesn't filter by account ID,
trades could be placed on the pension account (2674244).

**Note**: This was partially addressed (comment says "Mirror the ALLOWED_ACCOUNT_IDS
pattern from avanza_session.py"), but the dual-path architecture means both paths
must be checked independently.

### AV-I-002 — P3: Telegram CONFIRM order flow has no order-specific matching

**File**: `portfolio/avanza_orders.py` lines 121-143

**Bug**: When a user sends "CONFIRM" via Telegram, line 122 sorts pending orders by
timestamp descending and confirms the MOST RECENT one. If two orders are pending
simultaneously, there's no way to confirm a specific order — the user can only
confirm the latest.

**Impact**: In a rapid-fire scenario with multiple pending orders, confirming the
wrong order could lead to an unwanted trade.

### AV-I-003 — P2: CORS wildcard on dashboard

**File**: `dashboard/app.py` line 42

**Observation**: `Access-Control-Allow-Origin: *` allows any website to make requests
to the dashboard API. Combined with the optional auth (no token = open access),
any malicious website visited by the user could read portfolio data.

**Impact**: Information disclosure of portfolio holdings, trade history, and signal data.
No trade execution risk (dashboard is read-only + validate-portfolio POST).

---

## Subsystem 6: signals-modules

### SM-I-001 — P2: social_sentiment uses raw json.loads(resp.read())

**File**: `portfolio/social_sentiment.py` lines 34, 67

**Bug**: Uses `json.loads(resp.read())` instead of the project's `file_utils.load_json()`.
While this is for HTTP responses (not files), it violates the project convention and
the error handling may differ.

### SM-I-002 — P3: Signal modules silently return HOLD on any exception

**File**: `portfolio/signal_engine.py` lines 1689-1692

**Observation**: Any exception in an enhanced signal module results in a HOLD vote.
While this is correct fail-safe behavior, it means a persistently broken signal
(e.g., API key expired) silently stops voting rather than alerting that it's broken.

**Mitigation**: The system tracks `_signal_failures` (line 1692) and logs warnings
when >3 signals fail (line 1696). The health tracking at lines 1703-1710 also records
failures. This is adequate.

---

## Subsystem 7: data-external

### DE-I-001 — P2: Alpha Vantage 25/day rate limit enforced only by caller discipline

**File**: `portfolio/alpha_vantage.py`

**Observation**: The 25/day Alpha Vantage rate limit is enforced by the
`_alpha_vantage_limiter` in `shared_state.py`, not in the API module itself.
Any code that directly calls the API without going through the rate limiter
bypasses the protection.

### DE-I-002 — P2: FX rate hardcoded fallback

**File**: `portfolio/fx_rates.py`

**Observation**: If all FX rate sources fail, the system uses a hardcoded fallback
(typically 1.0 or a stale cached value). All portfolio valuations depend on FX rate,
so a wrong FX rate cascades to wrong drawdown calculations, wrong position sizing,
and wrong P&L.

---

## Subsystem 8: infrastructure

### IN-I-001 — P2: atomic_append_jsonl is not truly atomic

**File**: `portfolio/file_utils.py` lines 155-167

**Bug**: Despite the name, `atomic_append_jsonl` opens the file in append mode and
writes a line. If the process crashes between `f.write(line)` and `os.fsync()`,
the file could have a partial JSON line at the end. The next `load_jsonl()` call
would skip that line (handled at line 96), but the data is lost.

**Impact**: Signal log entries, journal entries, or Telegram messages could be lost
on crash. The partial-line handling in `load_jsonl` prevents corruption propagation.

### IN-I-002 — P3: Dashboard cache has no max-size bound

**File**: `dashboard/app.py` lines 62-77

**Bug**: The `_cache` dict grows unboundedly. Every unique (path, limit) combination
creates a cache entry that persists for the lifetime of the process. Since limit
is user-controlled (via query parameters), an attacker could create many cache
entries with different limit values.

**Impact**: Memory growth. Low severity since the dashboard is on a LAN.

### IN-I-003 — P3: Journal JSONL entries not validated before append

**File**: `portfolio/journal.py` (not read in detail)

**Observation**: `atomic_append_jsonl` accepts any dict. If the caller passes a dict
with missing required fields (e.g., no timestamp, no action), the entry is written
but may break downstream readers expecting specific fields.

---

---

## Cross-Subsystem Findings

### CROSS-001 — P1: C10 dead-signal-trap fix is incomplete — raw_votes never consumed

**Files**: `portfolio/signal_engine.py` line 2002, `portfolio/outcome_tracker.py` line 122-124

**Bug**: The C10 fix captures `_raw_votes` (pre-gating signal votes) at signal_engine.py
line 2002 to break the "dead signal trap" — regime-gated signals that can't accumulate
accuracy data because they're always HOLD. However, `_raw_votes` is NEVER used by any
downstream consumer.

The outcome_tracker.py at line 122-124 uses `extra.get("_votes")` — these are the
POST-gating votes where regime-gated signals are already HOLD. The accuracy_stats.py
system thus never sees votes from gated signals, so they can never prove themselves
and get un-gated.

**Evidence**: `grep -r "_raw_votes" portfolio/` returns only 2 hits in signal_engine.py.
No other module reads this field.

**Impact**: Regime-gated signals are permanently trapped — they can never accumulate enough
accuracy data to demonstrate that they've improved, because their votes are always
logged as HOLD. This affects ~13 signals in ranging regime, ~7 in trending-up, and ~8
in trending-down.

**Fix**: Update `outcome_tracker.py:log_signal_snapshot()` to use `_raw_votes` instead of
`_votes` for the signals dict, or add a parallel accuracy tracking path that uses raw votes.

### CROSS-002 — P2: AV-I-001 revision — account whitelist IS enforced in both paths

**Files**: `portfolio/avanza_client.py` line 31, `portfolio/avanza_session.py` line 35, 522, 665

**Revision**: Both `avanza_client.py` and `avanza_session.py` enforce
`ALLOWED_ACCOUNT_IDS = {"1625505"}` with explicit ValueError raises.
The original AV-I-001 finding is a false positive. Both trading paths are protected.

---

## Summary by Severity

| Severity | Count | Notable |
|----------|-------|---------|
| P1       | 3     | SC-I-001 (double regime gating), OR-I-001 (pool shutdown blocks loop), CROSS-001 (dead signal trap) |
| P2       | 10    | SC-I-002, OR-I-003, PR-I-001, PR-I-003, MC-I-002, MC-I-003, AV-I-003, SM-I-001, DE-I-001, DE-I-002, IN-I-001 |
| P3       | 7     | SC-I-003, SC-I-004, OR-I-002, AV-I-002, SM-I-002, IN-I-002, IN-I-003 |

## Top 5 Critical Findings

1. **SC-I-001**: Per-ticker regime gating exemptions (BUG-158) are completely nullified by
   double gating in `_weighted_consensus`. Active alpha is being thrown away on every cycle
   for every ticker in every non-trending regime.

2. **CROSS-001**: The C10 dead-signal-trap fix captured `_raw_votes` but NO downstream module
   uses them. 13+ regime-gated signals permanently cannot accumulate accuracy data.

3. **OR-I-001**: ThreadPoolExecutor `with` block calls `shutdown(wait=True)`, which blocks the
   entire main loop if any ticker thread hangs on network I/O. A single hung API call can
   stall the 24/7 trading system indefinitely.

4. **PR-I-001**: Drawdown circuit breaker uses entry price as fallback when live prices are
   missing, potentially missing real drawdowns during API outages.

5. **MC-I-002**: CET/CEST timezone fallback hardcodes UTC+1, giving wrong trading hours
   during summer months (March-October = UTC+2).
