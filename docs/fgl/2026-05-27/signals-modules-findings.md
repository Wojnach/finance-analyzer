## Summary

Audited 17 critical-active + 4 per-ticker-override + ~10 disabled signal modules under `portfolio/signals/` (71 files total). Verified `signal_registry.py`, `_validate_signal_result()` (signal_engine.py:1615), and the dispatch loop (signal_engine.py:3656+).

Counts: 1 P0, 7 P1, 6 P2, 4 P3.

Top 3 themes:
1. **`crypto_evrp.py` contradicts its own docstring** — emits SELL on conditions the comments call bullish, and the percentile sub-signal is partially dead-code (parameter ignored).
2. **Module-level caches without locks** — pattern repeated across `credit_spread`, `crypto_evrp`, `hash_ribbons`, `copper_gold_ratio`, `btc_etf_flow`, `sentiment_extremity_gate`. With 8 worker threads, last-writer-wins races leak HTTP calls (harmless) but the `sentiment_extremity_gate` single-cell cache also crosses tickers.
3. **External HTTP/yfinance inside signal compute** — CLAUDE.md §"Critical Rules #2" says "Search before writing code, reuse... http_retry"; several modules use raw `requests.get` (cot_positioning) or yfinance (`treasury_risk_rotation`, `network_momentum`, `copper_gold_ratio`, `ovx_metals_spillover`), each blocking the per-ticker dispatch worker for whatever yfinance/FRED returns. The `_cached` wrapper masks repeated cost but cold start still blocks. Per-ticker fan-out × N signals × cold cache ⇒ multi-second tail latency.

Biggest-risk one-liner: **`crypto_evrp.py` documents "high eVRP = bullish" but the constants emit SELL on high eVRP** (P0 below) — this signal is one of the 16 active voters for BTC/ETH and any directional contribution it makes today is opposite of its stated thesis.

---

### Critical (90-100)

### [P0] `crypto_evrp` direction contradicts its docstring; percentile sub-signal is partially dead code
**File:** `Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:12-19,41-45,195-201,204-242`
**Issue:** The module-level docstring says:
- "When eVRP is very high (>10), implied vol far exceeds realized vol... historically this precedes mean-reversion downward in IV, often coinciding with bullish price action"
- "When eVRP is very negative (<-10), realized vol exceeds implied... can signal contrarian BUY"

But the constants/`_evrp_level_signal` (lines 41-45, 197-201) do the opposite:
```python
EVRP_BUY_THRESHOLD = -10.0  # eVRP below this → BUY
EVRP_SELL_THRESHOLD = 10.0  # eVRP above this → SELL
```
So eVRP > 10 (the "bullish price action" case per docstring) emits SELL.

Separately, `_evrp_percentile_signal` (lines 204-242) accepts a `current_evrp` parameter that is never used after line 207. Lines 215-217 compute `rv_hist` from `rv_series` interpreted as if it were realized-vol but it's the close-price series — the computation is `log(price/price.shift(1)).std()` which is unrelated to the function's purpose, and the result is only used to gate `len(rv_hist) < PCTILE_WINDOW`. Line 236 then ranks `current_dvol` against historical DVOL — using DVOL, not eVRP. The docstring (line 207) says "eVRP percentile rank" but the implementation ranks DVOL.

**Impact:** Crypto eVRP is one of the 16 active voters on BTC-USD and ETH-USD (CLAUDE.md signal table, accuracy 55.5% on 366 samples). Either (a) the docstring is wrong and the 55.5% measurement IS the inverted-vs-docstring behavior — fine — but then the docstring is misleading future maintainers; or (b) the constants are wrong vs the original research and the signal has been emitting opposite-direction votes since registration on 2026-04-25. Without a unit test pinning intended direction, a refactor that "fixes" either the docstring or the constants flips the live signal's polarity. Also the unused `current_evrp` parameter is a smell that the percentile sub-signal isn't doing what its name says.
**Fix:**
1. Reconcile docstring vs constants. Check the Zarattini/Mele/Aziz (2025) paper cited at line 24 — eVRP > 0 (IV > RV) is typically the "volatility risk premium" that compensates option sellers and is associated with calm markets; on equities the sign is usually `dvol > rv → BUY` (calm). On crypto Zarattini's paper specifically finds the opposite, so verify before flipping.
2. Either (a) implement `_evrp_percentile_signal` to actually rank the eVRP series (`dvol - rv` computed historically), or (b) rename it `_dvol_percentile_signal` and remove the unused `current_evrp` parameter.
3. Add a regression test that pins `_evrp_level_signal(evrp=20.0) == "SELL"` (current behavior) so the next reviewer doesn't flip it accidentally.
**Confidence:** 92

---

### Important (80-89)

### [P1] `btc_etf_flow` is registered nowhere, has wrong signature, returns wrong schema
**File:** `Q:\finance-analyzer\portfolio\signals\btc_etf_flow.py:19-25,53,84-88`
**Issue:** Module is fully implemented but never registered in `signal_registry._register_defaults()` — verified by Grep. The function signature is `compute(ticker, indicators, context=None)` while every other signal uses `(df, context=None)`. The return dict has `action / confidence / indicators` but NO `sub_signals` key. The TODO at line 19-25 says "currently discovered but not registered".
**Impact:** Dead code today. If anyone wires it up by following the registry pattern, it will crash at first invocation because dispatch passes `(df, context=context_data)` but the function expects `(ticker, indicators)` — so `ticker` would receive a DataFrame and `ticker not in APPLICABLE_TICKERS` (line 59) would silently HOLD without crashing… unless `flow_data.get(...)` etc. is later called on indicators (a dict) that's actually a kwargs object. Latent footgun.
**Fix:** Either delete the file with a note in IMPROVEMENT_BACKLOG, or rename `compute` → `compute_btc_etf_flow_signal(df, context=None)`, register in `signal_registry.py`, and adjust to pull `ticker` from `context["ticker"]`. Add a `sub_signals` field so `extra_info[f"{sig_name}_sub_signals"]` in signal_engine.py:3788 receives the expected dict.
**Confidence:** 92

### [P1] `cot_positioning._fetch_cot_historical` uses raw `requests.get` and runs inside the per-ticker hot loop
**File:** `Q:\finance-analyzer\portfolio\signals\cot_positioning.py:87-125,361-367`
**Issue:** Uses raw `requests` (line 94 import, line 102 `requests.get(url, timeout=15)`) instead of `portfolio.http_retry.fetch_with_retry` like every other FRED/CFTC consumer in this folder. No retry on 429, no circuit breaker. CLAUDE.md §"Critical Rules #2" mandates reuse of `http_retry`.

Worse: the fetch fires whenever the local jsonl history has <20 entries (line 363). On a cold install or after a `cot_history.jsonl` rotation, every metals invocation hits CFTC. The CFTC SOCRATA API does enforce limits and a 15s timeout per request × XAU + XAG × N timeframes is multi-second tail latency on the signal engine dispatch.
**Impact:** Loop stalls during cold-cache moments, and a CFTC outage blocks the metals signal dispatcher with no retry/backoff. Matches the BUG-178 "150+s silent gaps" pattern documented at signal_engine.py:3661.
**Fix:** Replace `requests.get` with `fetch_with_retry` (same wrapper used in `credit_spread.py:74`, `crypto_evrp.py:76`, `metals_cross_asset.py:125`). Add a 24h `_cached` wrapper on the historical fetch since CFTC publishes weekly — caching for a few hours is safe.
**Confidence:** 90

### [P1] `sentiment_extremity_gate` module-level F&G cache shares state across all tickers
**File:** `Q:\finance-analyzer\portfolio\signals\sentiment_extremity_gate.py:34-51`
**Issue:** `_fg_cache = {"value": None, "ts": 0.0}` is a single dict keyed by nothing. `_get_fg_value(ticker)` is called per-ticker but writes/reads the same cell. Comment on line 49-50 suggests the underlying `get_fear_greed(ticker)` may be ticker-aware (passes `ticker` argument). If BTC returns one value, then within 60s ETH gets BTC's cached value regardless of what ETH would have returned, AND if ETH is the first call cold, BTC gets ETH's value.

Today this happens to be benign because `portfolio.fear_greed.get_fear_greed` returns the single alternative.me crypto F&G index regardless of ticker — but the API surface invites a per-ticker fix that would silently cross-pollinate.
**Impact:** Latent bug if `fear_greed.py` is ever updated to be ticker-aware (e.g., adding stocks F&G via VIX). The signal also lacks a lock, so 8 worker threads racing on `_fg_cache["value"] = ...` are technically a TOCTOU window, though Python's GIL makes the dict assignment atomic in practice.
**Fix:** Either (a) document on the cache that it intentionally stores the single global crypto F&G index and the `ticker` arg is decorative, or (b) key the cache by ticker: `_fg_cache: dict[str | None, tuple[int, float]] = {}` and look up by `ticker`.
**Confidence:** 85

### [P1] `intraday_seasonality` and `gold_overnight_bias` silently fall back to wall-clock when df has RangeIndex
**File:** `Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:79-92`, `Q:\finance-analyzer\portfolio\signals\gold_overnight_bias.py:44-57`
**Issue:** Both `_get_utc_hour_and_dow` and `_get_utc_time` check `hasattr(df.index, "hour")`. Verified via Grep that `signal_engine` never calls `set_index("time")` before dispatch — `df` arrives with `RangeIndex` (no `hour` attr). So both signals always hit the fallback path that returns `datetime.now(UTC)`.

For LIVE invocations this is benign (the loop runs every 60s and the last bar is at most a few minutes old, so wall clock ≈ last bar hour). For BACKTESTS or any code that replays historical data through these signals, the hour will be "now" instead of "the historical bar's hour" — every backtest row gets the same hour, which destroys the seasonality signal's edge.
**Impact:** Backtest results for `intraday_seasonality` and `gold_overnight_bias` are meaningless because they all read the same wall-clock hour. Walk-forward validation passes through this code path. Confidence numbers in `accuracy_cache.json` for these two signals (54.x% range per CLAUDE.md) may be measured from live runs (OK) or backfilled outcomes via replay (broken).
**Fix:** Either (a) set the index to `df['time']` once in `signal_engine.py` before dispatch (cleanest, touches all signals), or (b) accept a `now_ts` kwarg from context and have these two signals consume it. Option (b) requires no engine change. Also: change the `hasattr(...)` check to `isinstance(df.index, pd.DatetimeIndex)` — `RangeIndex.hour` doesn't exist but `MultiIndex` could have it accidentally.
**Confidence:** 88

### [P1] `news_event._thesis_alignment_vote` is structurally bias-confirming
**File:** `Q:\finance-analyzer\portfolio\signals\news_event.py:424-504,610-613`
**Issue:** When prophecy is bullish and news is bullish, registers BUY. When prophecy is bullish but news is bearish, function returns `HOLD` (line 496-499 with the comment "Don't vote against belief"). Symmetric on the bearish side (line 500-502).

In the dispatch logic at line 611-612, this vote is only appended to `votes` if `thesis_action != "HOLD"`. Combined with the asymmetric branches, this means the thesis-aligned direction can ONLY ever add to its consensus — counter-evidence is structurally suppressed.
**Impact:** Once a prophecy belief is activated (e.g., the current `silver_bull` 0.8 conviction per MEMORY.md), the news_event signal gains a one-way amplifier toward BUY for that ticker. The CLAUDE.md key principle "Data-driven, not speculative" is violated — the signal's prior literally shapes the posterior. Config-gated by `prophecy.news_alignment`, so impact depends on whether that flag is on.
**Fix:** Either (a) emit the contradicted direction at reduced confidence (e.g., explicit SELL when belief=bullish and news=bearish, with a `belief_contradicted` flag in indicators), or (b) remove the asymmetric branches entirely so the sub-signal is symmetric news-vs-belief regardless of which way the belief points.
**Confidence:** 83

### [P1] `credit_spread._fetch_hy_oas` and several other FRED-using signals write the module cache without a lock
**File:** `Q:\finance-analyzer\portfolio\signals\credit_spread.py:53,113-115`, `Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:51-54,107`, `Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:51,107-108`, `Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:43`
**Issue:** Pattern: module-level mutable dict `_oas_cache = {}` / `_DVOL_CACHE = {}` etc., written from inside the fetch function without a lock. 8 worker threads (`ThreadPoolExecutor` per CLAUDE.md §Architecture/Layer 1) can call the same signal concurrently for different tickers.

For comparison, `metals_cross_asset.py:88` and `gold_real_yield_paradox.py:40` DO use `threading.Lock()`. So the pattern is known to be needed but inconsistently applied.
**Impact:** Worst case is benign — last-writer-wins. Best case: 1-3 extra HTTP calls per cache window if 8 threads hit cold cache simultaneously. The real risk is the inconsistency: a future refactor that switches to a richer cache structure (e.g., adding timestamps, or appending to a series) will see torn writes.
**Fix:** Move all module-level caches behind `threading.Lock()` or — preferred — refactor onto `portfolio.shared_state._cached` which is already used by `crypto_macro.py`, `network_momentum.py`, `ovx_metals_spillover.py`, `treasury_risk_rotation.py`, and `btc_gold_correlation_regime.py`. That utility already handles concurrency.
**Confidence:** 82

### [P1] Disabled signals in `_SHADOW_SAFE_SIGNALS` still execute their external API calls every cycle
**File:** `Q:\finance-analyzer\portfolio\signal_engine.py:3683-3711`
**Issue:** The dispatch loop intentionally runs shadow-safe disabled signals so their predictions can be logged for accuracy tracking (3684-3686). But this means `crypto_evrp`, `hash_ribbons`, `gold_real_yield_paradox`, `cot_positioning`, `credit_spread`, etc. ALL execute their full compute (including the HTTP fetches at lines `crypto_evrp.py:304/319`, `hash_ribbons.py:252`, `gold_real_yield_paradox.py:275`, `cot_positioning.py:342`) on every loop iteration, even if their vote will be force-HOLD'd anyway.

For a signal like `cot_positioning` whose data updates weekly, hitting CFTC every 60s × tickers × timeframes is wasteful and noisy. The signals do cache (4h-24h), so steady-state is fine — but cold-start, cache eviction, or an outage forces them to redo the fetch.
**Impact:** Wasted CPU/network on signals whose votes don't count, plus log noise. Measured impact is bounded by the cache TTLs (4h FRED, 24h hashrate, 24h CFTC) but cold-start still pays the full cost. The 2026-04-09/10 BUG-178 incident at `signal_engine.py:3657-3666` was exactly this class of failure mode.
**Fix:** For shadow-safe signals, fetch from cache only — pass a `read_only=True` hint to the cache helper or wrap the signal in a `compute_shadow()` variant that returns the last cached prediction without firing fresh HTTP. Alternatively, gate the HTTP fetches behind `if not shadow_mode:` inside each signal. The current behavior keeps the spirit of the BUG-178 fix only partially.
**Confidence:** 82

---

### Medium (P2)

### [P2] `calendar_seasonal._pre_holiday_effect` uses hardcoded approximate US holiday dates
**File:** `Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:210-220`
**Issue:** Holidays whose dates shift annually (MLK=3rd Monday, Memorial=last Monday, Labor=1st Monday, Thanksgiving=4th Thursday) are hardcoded to one specific calendar date that only matches ~1 year in 7. Comment line 209 acknowledges this. Also missing Good Friday entirely. Crypto runs 24/7 so a misplaced "pre-holiday BUY" fires on the wrong day.
**Impact:** Stale signal value for 5-6 out of 7 holidays per year — emits BUY on dates that aren't actually pre-holiday. Calendar signal's confidence is capped at 0.6 so impact is bounded, but it's wrong much of the time.
**Fix:** Replace with `pandas_market_calendars` or a precomputed yearly holiday list. The `econ_calendar.py` module already does proper date arithmetic via `econ_dates.py` — same pattern can be used here.
**Confidence:** 80

### [P2] `calendar_seasonal._day_of_week_effect` applies US-stock seasonality to 24/7 crypto and metals
**File:** `Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:50-70`
**Issue:** "Monday SELL, Friday BUY" is a US equity weekend-effect heuristic. Crypto and metals (XAU/XAG via Binance FAPI) trade 24/7 with no weekend. The signal applies this rule blindly regardless of asset class. With `df["time"]` likely in UTC and bars closing 23:00 UTC every day, the "Friday" determination depends on which timezone the bar timestamp lives in.
**Impact:** Adds a steady BUY/SELL bias to crypto and metals on Mon/Fri that the underlying empirical edge doesn't support. Confidence is capped at 0.6 so contribution is bounded.
**Fix:** Add `context["asset_class"]` gating — only emit the day-of-week vote for `"stocks"`. The `econ_calendar` and `claude_fundamental` signals already accept `context` for similar gating.
**Confidence:** 80

### [P2] `metals_cross_asset` Sub 5 (SPY momentum) creates asymmetric votes for gold
**File:** `Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:385-395`
**Issue:** SPY positive → silver BUY, gold HOLD. SPY negative → gold BUY, silver SELL. So gold gets a BUY-only contribution from this sub-signal (no SELL branch), and silver gets symmetric BUY/SELL. The asymmetry creates a permanent gold-bias when SPY is volatile around zero.
**Impact:** Inflates gold's BUY confidence vs silver in regimes where SPY oscillates. Not necessarily wrong (gold IS a safe haven and benefits from both directions in some framings) but structurally one-sided. Confidence is capped at 0.7.
**Fix:** Make the gold branch symmetric: SPY up → gold SELL (risk-on, no safe haven demand), SPY down → gold BUY. Document the asymmetry intent if intentional.
**Confidence:** 80

### [P2] `mean_reversion` seasonality detrending is dead code (never triggers)
**File:** `Q:\finance-analyzer\portfolio\signals\mean_reversion.py:463-489`
**Issue:** `if context and context.get("seasonality_profile") and hasattr(df.index, "hour")` — but `signal_engine` passes df with `RangeIndex` (no `hour` attr, verified). The detrending branch never executes. The seasonality_profile is loaded at `signal_engine.py:3637-3644` but never consumed for mean_reversion.
**Impact:** No functional impact (signal works in non-detrended mode) but ~30 lines of complex code including a P1-6 bug-fix comment from 2026-05-02 protect a path that doesn't run. Future readers will assume it's active.
**Fix:** Either remove the dead path with a note, or fix the dispatch loop to set `df.index = df["time"]` (which would also fix the intraday_seasonality/gold_overnight_bias issue above). Recommend fixing dispatch and keeping the detrend code live.
**Confidence:** 82

### [P2] `cot_positioning._sub_cot_index` includes the current sample in its own min/max range
**File:** `Q:\finance-analyzer\portfolio\signals\cot_positioning.py:173-194`
**Issue:** `nc_net_history = [nc_net]` then prepends historical values (line 173-177). Then `_compute_cot_index` uses `nc_net_history[0]` as "current" but `min(nc_net_history)` and `max(nc_net_history)` include `nc_net` itself. When the current value is a new extreme, the percentile is automatically 0 or 100 (which then maps to BUY/SELL respectively).
**Impact:** When `nc_net` exceeds any historical value, the COT index reads 100 → contrarian SELL fires automatically. This is technically the correct rank (it IS the max now) but bypasses the calibration: a true 100th-percentile reading in a 156-week window has different meaning than a value that's merely the new highest in your local sample, especially when local history is <20 entries and `_fetch_cot_historical` is forced (line 364-365).
**Fix:** Exclude the current sample from the range: `hist_min = min(nc_net_history[1:])`, `hist_max = max(nc_net_history[1:])`. Or compute against the historical window only, then check whether `current > hist_max` separately.
**Confidence:** 80

### [P2] `crypto_macro.OPTIONS_TTL` defined AFTER first use; relies on import-order side effect
**File:** `Q:\finance-analyzer\portfolio\signals\crypto_macro.py:228,281`
**Issue:** Line 228 references `OPTIONS_TTL` inside `compute_crypto_macro_signal`. The constant is defined at line 281 (last line). Works at runtime because function bodies bind names at call time, not def time. But this is fragile: any refactor that moves the function call earlier (e.g., evaluated at module top level for testing) breaks. Also confuses readers — typical Python convention is constants at top.
**Impact:** No runtime bug today. Smell + maintainability risk.
**Fix:** Move `OPTIONS_TTL = 900` to the top of the file alongside the other module constants.
**Confidence:** 80

---

### Smells (P3)

### [P3] Schema inconsistency: `connors_rsi2` and `adx_regime_switch` return `sub_signals` as dict-of-dicts, others return dict-of-strings
**File:** `Q:\finance-analyzer\portfolio\signals\connors_rsi2.py:144-152`, `Q:\finance-analyzer\portfolio\signals\adx_regime_switch.py:152-159`
**Issue:** Most signals: `sub_signals = {"name": "BUY"}`. These two: `sub_signals = {"name": {"value": ..., "signal": "BUY"}}`. `_validate_signal_result` (signal_engine.py:1641) only checks `isinstance(sub_signals, dict)` so both pass, but downstream consumers (reporting.py heatmap, dashboard) that do `if sub_signals[name] == "BUY"` would treat the nested dict as not-equal-to-"BUY" silently.
**Impact:** Display/audit only. No vote correctness impact because the parent `action`/`confidence` is what gets voted.
**Fix:** Normalize to `{"name": "BUY"}` with indicators going into the separate `indicators` dict (the standard pattern used by every other signal in the folder).
**Confidence:** 80

### [P3] `_get_fred_key` ternary chain is hard to parse and silently returns empty when cfg is non-dict object without expected attrs
**File:** `Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:100-102`, `Q:\finance-analyzer\portfolio\signals\credit_spread.py:134-136`, `Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:51-53`
**Issue:** `return getattr(cfg, "fred_api_key", "") or getattr(getattr(cfg, "golddigger", None), "fred_api_key", "") if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""`. Python parses this as `(X) if (Y) else Z` correctly, but the chain `getattr(None, "fred_api_key", "")` is a footgun if `golddigger` is missing (returns `""` fine, but only after the `hasattr` check). Three copies of the same code.
**Impact:** Works today. Three duplicate copies = future divergence risk.
**Fix:** Extract to `portfolio.config_utils.get_fred_key(context)` and reuse. Also rewrite as explicit if/elif rather than the inline ternary chain.
**Confidence:** 80

### [P3] `forecast.py:88-93` reads `config.json` at import time (`_init_kronos_enabled`)
**File:** `Q:\finance-analyzer\portfolio\signals\forecast.py:76-100`
**Issue:** Module import does disk I/O to load config. Tests that import this module incur the I/O cost. The "config.json" path is computed from `__file__` which works but couples test layout to file location.
**Impact:** Test setup slowdown. Possible test isolation issue if a test mutates the global `_KRONOS_ENABLED` flag and another test imports the module fresh.
**Fix:** Lazy-load on first compute, or make `_KRONOS_ENABLED` a function instead of a module global.
**Confidence:** 80

### [P3] Several signals catch broad `Exception` and silently log at DEBUG, hiding API failures
**File:** `Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:80-88`, `Q:\finance-analyzer\portfolio\signals\network_momentum.py` (similar pattern in many places)
**Issue:** Pattern `except Exception as e: logger.warning(...)` is reasonable, but in several places the catch is `except Exception: logger.debug(...)` which makes failures invisible. CLAUDE.md MEMORY notes "logging to silent exception swallowers" was explicitly worked on (commit `b7d82290` in git log).
**Impact:** Silent degradation when an upstream API breaks. The recent commit b7d82290 fixed 4 of these — there are more.
**Fix:** Audit pass: any `except Exception` in the signals folder that swallows to `logger.debug` should be promoted to `logger.warning` with the exception included.
**Confidence:** 80
