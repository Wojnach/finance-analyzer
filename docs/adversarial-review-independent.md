# Independent Adversarial Review — 2026-04-25

## Methodology

Deep code-level review of the finance-analyzer codebase, partitioned into 8 subsystems.
Each subsystem reviewed for: logic bugs, silent failures, concurrency issues, state corruption,
edge cases, security, and design problems. Severity: P1 (will cause wrong trades/data loss),
P2 (degraded accuracy/reliability), P3 (code quality/maintainability).

Focus: changes since review #7 (commit 84dc913d) plus re-verification of previously
reported unfixed issues.

---

## 1. SIGNALS-MODULES

### SM-P1-01: `mahalanobis_turbulence._fetch_multi_asset_closes()` wrong `_cached()` call signature
**File:** `portfolio/signals/mahalanobis_turbulence.py:99`
**What:** The call is:
```python
return _cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)
```
But `_cached` in `shared_state.py:37` has signature `_cached(key, ttl, func, *args)`.
This passes `_do_fetch` (a callable) as the `ttl` parameter (position 2), then also
passes `ttl=_CACHE_TTL` as a keyword argument. Python will raise:
`TypeError: _cached() got multiple values for argument 'ttl'`

**Impact:** When mahalanobis_turbulence is enabled (currently disabled pending validation),
every call to `compute()` will crash with TypeError. The signal will never produce a vote
and will be caught by the signal dispatch's `except Exception`, appearing as a silent failure.
The system will log a warning but the signal will be permanently dead until the call is fixed.

**Fix:** Reorder arguments: `return _cached("mahalanobis_turb_closes", _CACHE_TTL, _do_fetch)`

**Note:** Agent review also found the SAME bug in `portfolio/signals/complexity_gap_regime.py:92`.
Both disabled signals have identical `_cached()` arg-swap bugs.

### SM-P2-01: `smart_money` disabled per-ticker but not in DISABLED_SIGNALS — wasteful computation
**File:** `portfolio/signal_engine.py:372-398` (per-ticker blacklist)
**What:** Commit 8fe0be35 disabled smart_money by adding it to `_TICKER_DISABLED_BY_HORIZON["_default"]`
for ALL five Tier-1 tickers. However, smart_money is NOT in the `DISABLED_SIGNALS` frozenset
in `tickers.py`. The signal module is still loaded and `compute()` is called every cycle.
The vote is then immediately force-HOLD'd by the per-ticker blacklist.

**Impact:** Wasted CPU (~50-100ms per ticker per cycle x 5 tickers) for a signal that can never vote.

**Fix:** Add `"smart_money"` to `DISABLED_SIGNALS` in `tickers.py`.

---

## 2. SIGNALS-CORE

### SC-P2-01: `_get_horizon_weights()` returns None during dogpile -> TypeError in consensus
**File:** `portfolio/signal_engine.py:901-909` (getter) and `1526, 1826` (consumer)
**What:** `_get_horizon_weights(horizon)` delegates to `_cached()`. During a dogpile event
(two threads hit simultaneously with no stale data), `_cached` returns None (shared_state.py:87).
The caller at line 1526 stores `horizon_mults = None`. At line 1826, the code does
`if signal_name in horizon_mults:` which raises `TypeError: argument of type 'NoneType' is not iterable`.

**Impact:** On cold-start with concurrent ticker processing, one ticker's consensus computation
crashes for that cycle. Subsequent cycles recover from cache. Non-trivial probability on
every loop restart (5 tickers, 8 workers).

**Fix:** Guard the return in `_get_horizon_weights`:
```python
result = _cached(cache_key, TTL, lambda: _compute_dynamic_horizon_weights(horizon))
return result if result is not None else HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
```

### SC-P2-02: Persistence filter cold-start still double-permissive (unfixed since review #7)
**File:** `portfolio/signal_engine.py:258-268`
**What:** On cold-start, the persistence filter seeds `cycles=_PERSISTENCE_MIN_CYCLES` (=2)
for non-HOLD signals AND returns all votes unfiltered. Cycle 2 also passes because the
seeded count already meets the threshold. Two unfiltered cycles instead of one.

**Impact:** Every loop restart produces 2 unfiltered cycles. With auto-restart on logon,
this happens multiple times daily.

**Fix:** Seed `cycles=1` instead of `_PERSISTENCE_MIN_CYCLES`.

### SC-P2-03: ADX cache key still uses `id(df)` — collision risk on GC address reuse
**File:** `portfolio/signal_engine.py:2005`
**What:** Key is `(id(df), len(df), last_close)`. Two DataFrames of identical length and
last close at the same address (after GC) produce a cache hit with wrong data.

**Impact:** Wrong ADX -> wrong regime -> wrong signal weights. Low probability in production.

### SC-P3-01: `_weighted_consensus` accuracy_data sanitization is 80+ lines of nested logic
**File:** `portfolio/signal_engine.py:1528-1605`
**What:** The 13-round Codex sanitization logic is extremely complex. While each round fixed
a real bug, the accumulated complexity makes the consensus path nearly impossible to reason about.

---

## 3. ORCHESTRATION

### OR-P2-01: Multi-agent specialist timeout blocks main loop synchronously
**File:** `portfolio/agent_invocation.py:428-429`
**What:** `wait_for_specialists()` blocks for up to 30s. Comment says "TODO: run in background
thread" but this hasn't been implemented.

**Impact:** 30s main loop stall during multi-agent invocations.

### OR-P2-02: `_extract_ticker` falls back to "XAG-USD" when no ticker found in reasons
**File:** `portfolio/agent_invocation.py:151`
**What:** When no ticker pattern is found in trigger reasons, the function defaults to
"XAG-USD". This biases multi-agent specialist analysis toward silver when the trigger
was for a different asset. Affects the specialist prompt context and ticker-specific
data loading.

**Impact:** Multi-agent specialists may analyze XAG-USD when the trigger was for BTC-USD
if the trigger reason format doesn't match the regex patterns.

---

## 4. PORTFOLIO-RISK

### PR-P2-01: `_streaming_max` re-reads entire JSONL on every `check_drawdown` call
**File:** `portfolio/risk_management.py:21-51`
**What:** Called on every `invoke_agent()`. Reads the full `portfolio_value_history.jsonl`.

**Impact:** Increasing latency. Low now (file in OS cache), degrades over months.

### PR-P2-02: `_compute_portfolio_value` silently falls back to `avg_cost_usd` for missing prices
**File:** `portfolio/risk_management.py:79-81`
**What:** When agent_summary doesn't contain a ticker's price, the function uses entry price.
The drawdown reading becomes inaccurate in either direction.

**Impact:** Stale/missing price feed makes drawdown circuit breaker unreliable for that position.

### PR-P2-03: trade_guards `_load_state`/`_save_state` has no locking
**File:** `portfolio/trade_guards.py:31-42`
**What:** `check_overtrading_guards` calls `_load_state()` and `record_trade` calls `_save_state()`.
These can run concurrently (main loop calls check, Layer 2 calls record) with no lock
protecting the read-modify-write cycle. The file I/O is atomic (atomic_write_json), so
no corruption, but a trade record written by Layer 2 can be overwritten by a stale read
in the main loop's next check.

**Impact:** Trade cooldowns could be reset by a race condition, allowing overtrading.

---

## 5. METALS-CORE

### MC-P2-01: `metals_loop.py` sys.path manipulation creates import collision risk
**File:** `data/metals_loop.py:200-208`
**What:** Both `BASE_DIR` and `DATA_DIR` are inserted into `sys.path`. Any `.py` file
in `data/` shadows stdlib modules of the same name.

**Impact:** Currently safe (no name collisions). Adding `data/json.py` or `data/logging.py`
would break the system silently.

### MC-P3-01: metals_loop.py is 5000+ lines (monolith)
**File:** `data/metals_loop.py`
**Impact:** Extremely difficult to review, test, or refactor safely.

---

## 6. AVANZA-API

### AV-P2-01: Playwright browser context has no atexit cleanup
**File:** `portfolio/avanza_session.py:129-172`
**What:** `close_playwright()` exists but isn't registered as atexit handler. On normal
process exit, Chromium processes may be orphaned.

### AV-P2-02: CONFIRM flow still matches any CONFIRM to most recent order (no order-ID)
**File:** `portfolio/avanza_orders.py:130-135`
**What:** Previously reported in reviews #6 and #7. Still unfixed.

---

## 7. DATA-EXTERNAL

### DE-P2-01: `fx_rates.py` defaults to 1.0 on API failure
**What:** Reported in multiple reviews. FX rate of 1.0 instead of ~10.5 makes portfolio
value calculations off by ~10x during outages.

### DE-P2-02: yfinance MultiIndex column handling fragile
**File:** `portfolio/data_collector.py:179`
**What:** `hist.columns.get_level_values(0)` flattening may lose data if MultiIndex has
duplicate level-0 names.

---

## 8. INFRASTRUCTURE

### IF-P2-01: `log_rotation.rotate_jsonl()` reads entire file into memory
**File:** `portfolio/log_rotation.py:164-186`
**What:** For signal_log.jsonl at 68MB+, this is a significant memory spike. If the
process crashes between truncating the original and writing archives, data is lost.

**Fix:** Write archives first, then atomically replace main file.

### IF-P2-02: lockfile accumulation in data/ directory
**File:** `portfolio/file_utils.py:200+`
**What:** Sidecar `.lock` files for JSONL append are never cleaned up. Dozens accumulate.

---

## Cross-Cutting: `_cached()` dogpile None-return pattern

Multiple callers of `_cached()` don't handle the None return from the dogpile code path.
The `or` fallback pattern works for some callers (e.g., `_get_correlation_groups() or STATIC`),
but `_get_horizon_weights()` lacks this guard and can propagate None into consensus.

**Recommendation:** Systematically audit all `_cached()` callers for None handling, or
make `_cached()` never return None (return a sentinel default instead).

---

## Summary

| Severity | Count | Subsystems |
|----------|-------|------------|
| P1       | 1     | signals-modules |
| P2       | 14    | All subsystems |
| P3       | 2     | signals-core, metals-core |
| **Total** | **17** | |

The most critical new finding is **SM-P1-01**: the `mahalanobis_turbulence` signal's
`_cached()` call has swapped arguments. Currently harmless (signal disabled) but will
crash when enabled.

The most systemic finding is the **`_cached()` dogpile None-return pattern** affecting
`_get_horizon_weights()` and potentially other callers.
