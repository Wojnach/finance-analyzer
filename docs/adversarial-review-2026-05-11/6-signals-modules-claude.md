# Claude adversarial review: signals-modules

## Summary

Worst-5 by severity (mix of P0/P1 bugs and structural risk):

1. **`mahalanobis_turbulence.py`** — `_cached()` signature drift; passes the fetcher
   function as the positional `ttl` argument and `ttl=` as kwarg. Calling this signal
   would TypeError on the very first invocation. Disabled today, but registered, and
   will instantly break the loop the moment someone flips it on.
2. **`claude_fundamental.py`** — Per-tier cache `ts` is set *before* the background
   refresh thread runs. If the Claude CLI fails (timeout, gate kill, network),
   `_cache[tier]["results"]` stays at its previous (possibly empty `{}`) state until
   the next cooldown window — silent permanent HOLD with no diagnostic. This is the
   exact "silent fallback" failure-mode the project guards against.
3. **`smart_money.py`** — Supply/demand zone proximity math is broken: the
   `proximity_pct / _ZONE_PROXIMITY_PCT` ratio cancels (both equal 0.005), so the
   "expand zone by X%" knob is dead. Plus the same module is globally disabled in
   `DISABLED_SIGNALS`, hiding the bug from production but waiting to bite re-enable.
4. **`news_event.py`** — `_persist_headlines()` writes to a single shared file
   `data/headlines_latest.json` with no ticker key. With 8 ThreadPoolExecutor workers
   and 5 tickers running news_event concurrently, fish-monitor reads whichever ticker
   wrote last — effectively random.
5. **`calendar_seasonal.py`** — Hardcoded US holiday dates (MLK on Jan 20, Memorial
   Day on May 26, Labor Day on Sep 1) are wrong every year because those holidays
   are floating Mondays. The "pre-holiday BUY" sub-signal fires on the wrong day at
   least 6 times a year.

Honorable mention: `intraday_seasonality.py` falls back to wall-clock `now()` when
the DataFrame has no datetime index — silent live-time leak on historical data, a
subtle look-ahead vector for any backtest that doesn't set `df.index` to bars.

---

## P0 — Blockers

- **`portfolio/signals/mahalanobis_turbulence.py:99`** — Why it bites:

  ```python
  return _cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)
  ```

  `shared_state._cached` signature is `_cached(key, ttl, func, *args)`
  (`portfolio/shared_state.py:37`). This call passes `_do_fetch` as the **second
  positional** (`ttl`), then `ttl=_CACHE_TTL` as kwarg — `TypeError: _cached() got
  multiple values for argument 'ttl'` (or `func` arg missing). Today the signal is in
  `DISABLED_SIGNALS` so the import/registration path doesn't reach it, but the
  registry has `requires_context=True, max_confidence=0.7` and signal_engine will
  dispatch the moment it leaves the disable list. Fix: `_cached("mahalanobis_turb_closes", _CACHE_TTL, _do_fetch)`.

- **`portfolio/signals/claude_fundamental.py:929`** — Why it bites:

  ```python
  if _needs_refresh(tier, cooldowns):
      with _lock:
          if _needs_refresh(tier, cooldowns):
              _cache[tier]["ts"] = time.time()      # marked refreshed
              t = threading.Thread(target=_bg_refresh, args=(tier, context), ...)
              t.start()
  ```

  `ts` is bumped before the thread runs. If `_bg_refresh` raises (CLI timeout,
  claude_gate kill switch, JSON extract returns `{}`, network), the thread silently
  fails and `_cache[tier]["results"]` keeps the prior value (often `{}` from
  initialization). Next cycle, `_needs_refresh()` is still False because `ts` says
  "we just refreshed", so the failure invisibly persists for the full cooldown
  (5min/30min/2h). Compounded across all three tiers and you get the entire
  fundamental signal returning HOLD with zero log signal for the operator. Fix:
  bump `ts` inside `_refresh_tier` *only on successful parse*, or set a separate
  `_in_flight` flag.

## P1 — High

- **`portfolio/signals/news_event.py:46-52, 96`** — `_HEADLINES_PATH` is a single
  process-wide file with no ticker key, written via `atomic_write_json` from every
  per-ticker signal invocation in parallel. With 5 tickers x 7 TFs in the same
  cycle, the file is overwritten dozens of times per cycle; `fin_snipe.py` and any
  other reader sees an unpredictable ticker's headlines tagged with another
  ticker's `"ticker"` field — and the rename-atomicity doesn't help because the
  payload itself is per-ticker. Fix: key by ticker (`headlines_latest_{ticker}.json`)
  or move into a per-ticker subdir.

- **`portfolio/signals/intraday_seasonality.py:91-92`** — `_get_utc_hour_and_dow`
  silently falls back to `datetime.datetime.now(datetime.timezone.utc)` when the
  DataFrame index has no `.hour` attribute. In backtesting (where `df` is loaded
  from JSON/CSV with integer index or `time` column) this fetches **live wall-clock
  time**, applying today's seasonality multipliers to historical bars — implicit
  look-ahead disguised as "the bar's time". Fix: require an explicit time column
  and HOLD with `error: "no_timestamp"` when missing.

- **`portfolio/signals/forecast.py:206-220`** (`_oi_acceleration` in
  `futures_flow.py` has the same shape) — `current_price = close_prices[-1]` plus
  forecast over horizon 1h/24h, but `close_prices` come from Binance which by
  default includes the **currently-forming bar**. The forecast is conditioning on
  an incomplete bar whose close will change. Not catastrophic at 1h, but the
  Kronos/Chronos "shadow accuracy backfill" will compare to the final close of that
  same bar — guaranteed correlation that overstates accuracy.

- **`portfolio/signals/calendar_seasonal.py:210-220`** — `_US_HOLIDAYS` is hardcoded
  to fixed (month, day) tuples for floating-Monday holidays:
  - MLK Day `(1, 20)` — actually 3rd Monday of January (Jan 19 in 2026, 18 in 2027)
  - Presidents' Day `(2, 17)` — 3rd Monday of Feb (Feb 16 in 2026)
  - Memorial Day `(5, 26)` — last Monday of May (May 25 in 2026)
  - Labor Day `(9, 1)` — 1st Monday of Sept (Sept 7 in 2026)
  - Thanksgiving `(11, 27)` — 4th Thursday of Nov (Nov 26 in 2026, 25 in 2027)
  The pre-holiday BUY fires the day before the wrong date and silently misses the
  real day. Fix: use `pandas.tseries.holiday.USFederalHolidayCalendar` or a proper
  holiday lib.

- **`portfolio/signals/futures_flow.py:33-36`** — Thresholds drift vs reality:
  `_LS_EXTREME_LOW = 0.7` is documented as "crowd overleveraged short", but Binance's
  `longShortRatio = longAccount/shortAccount` with value 0.7 means longs 41% / shorts
  59% — that's a mild lean, not the "overleveraged shorts" extreme the contrarian
  rule presumes. Fires false BUYs in routine sentiment. Recalibrate against the
  observed distribution per ticker (`btc` and `eth` LS ratios sit in very different
  bands).

- **`portfolio/signals/claude_fundamental.py:828-883`** — Cascade Opus>Sonnet>Haiku
  is plausible, but the bias-suppression logic uses `load_jsonl_tail(_CF_LOG,
  max_entries=400)` on every invocation — that's 5 tickers x N TFs reading + tail-
  parsing the JSONL on every signal call (file grows unbounded). With log rotation
  uncertain and a 400-entry tail, this is a hot path doing disk I/O inside the
  ThreadPoolExecutor.

- **`portfolio/signals/smart_money.py:374, 384`** — Zone-expansion math:

  ```python
  margin = (z_high - z_low) * proximity_pct / _ZONE_PROXIMITY_PCT if z_high > z_low else 0
  expand = max(current_close * proximity_pct, margin * 0.1)
  ```

  `proximity_pct == _ZONE_PROXIMITY_PCT == 0.005` is the only call site, so
  `margin = (z_high - z_low)`. The intended "expand by proximity_pct" knob does
  nothing. The whole supply/demand sub-signal is using a default-only proximity
  rule. Disabled globally so non-fatal — but anyone re-enabling per ticker per the
  Apr-24 blacklist will inherit this dead knob.

## P2 — Medium

- **Module-level un-locked caches** in at least:
  `copper_gold_ratio.py:43` (`_CACHE`), `credit_spread.py:53` (`_oas_cache`),
  `claude_fundamental.py:42, 148` (`_cache`, `_earnings_cache` — the first has a
  `_lock`, the second does not). With ThreadPoolExecutor read-modify-write races
  these can corrupt or duplicate-fetch. The shared_state `_cached` helper exists for
  exactly this — these signals should use it, not roll their own.

- **`portfolio/signals/structure.py:60-83`** — `_highlow_breakout` uses
  `df.iloc[-252:]` *including* the current bar. `period_high` then equals
  `max(highs)` which often is just the current bar's high during the bar's life
  (intraday). `pct_from_high <= 0.02` then trivially fires BUY whenever price is
  near today's session high — basically a momentum-of-current-bar signal masked as
  a 52-week breakout. Donchian sub-signal does it right (`iloc[-(period+1):-1]`) —
  Fix: same exclusion here.

- **`portfolio/signals/realized_skewness.py:33`** — `SKEW_LOOKBACK = 252` is labeled
  "~1 year of daily data", but the signal engine runs this across all 7 timeframes
  (Now, 1h, 3h, 4h, 12h, 1d, 3d). On a 3h ticker 252 bars is ~31 days; on a "Now"
  timeframe it's whatever last 252 of the live-bar mix is. The fixed thresholds
  (`Z_BUY = -1.5`, `Z_SELL = 1.5`) and the "Sharpe 0.79 from Fernandez-Perez
  (commodity futures DAILY data)" paper backing both assume daily bars. Same
  threshold applied to 3h bars predicts something else. Same issue would apply to
  most newly-added papers' lookback windows.

- **`portfolio/signals/forecast.py:495-498`** — `_REGIME_DISCOUNT_TRENDING = 0.5`
  hard-coded contradicts the doc-string assumption that confidence should be
  *boosted* in trending regimes for momentum-style models. Chronos is described as
  mean-reverting so this is okay for *Chronos*, but the same factor applies to the
  Kronos sub-signal — different model class, same discount. Per-model regime
  scaling needed.

- **`portfolio/signals/news_event.py:280-295`** — `_sentiment_shift` defaults
  unmatched "cut" to bearish, but the keyword scanner sees the headline title in
  lowercase, and headline content from NewsAPI can include words like "shortcut",
  "haircut", "cuts" embedded mid-token (substring search). "Bitcoin Layer 2
  shortcut launched" → `"cut" in title` → bearish neg vote on news that has nothing
  to do with cutting. Use whole-word boundary check (`\bcut\b`).

- **`portfolio/signals/credit_spread.py:125-136`** — `_get_fred_key` is a tortured
  ternary that depends on `cfg.golddigger`. When config is loaded as dict (the
  normal path), the dict branch handles it. When config is a `SimpleNamespace`,
  `hasattr(cfg, "golddigger")` is True but `cfg.golddigger` could be None, and
  `getattr(None, "fred_api_key", "")` returns `""` — wrapped in `or` chain returns
  `""` even when the dict branch would have returned the key. Untested cross-branch
  inconsistency. Pick one config shape and validate at registry time.

## P3 — Low

- **`portfolio/signals/vwap_zscore_mr.py:124`** — Bare `except Exception: return
  HOLD` with no `logger.warning`. Silent fallback violates the project's
  log-everything rule. Same pattern appears in `dxy_cross_asset.py` (catches
  ImportError silently), `cot_positioning.py`, `intraday_seasonality.py:89`.

- **`portfolio/signals/dxy_cross_asset.py:78-82`** — confidence formula
  `min(abs(change_1h)/0.5, 1.0)` returns up to 1.0, but registry caps the signal at
  `max_confidence=0.8`. Cleaner to clamp in the function (defense in depth) so
  unit tests don't pass with values that signal_engine will silently reduce.

- **`portfolio/signals/momentum.py:46-77`** — `_rsi_divergence` compares
  first-half-low vs second-half-low using `.min()` on raw bar lows. The min in the
  first half can land on bar index 0 (oldest) and in the second half on bar -1
  (newest); the resulting "swing point comparison" sometimes has 0-bar separation
  between the two "swings". A swing isn't a swing if you didn't have the chance
  to confirm direction. The window-split heuristic should require swing points to
  be at least N bars apart.

- **`portfolio/signals/calendar_seasonal.py:256-257`** — When all FOMC dates are in
  the past, the function only `logger.warning(...)` once per call. There's nothing
  surfaced through the signal output that would let signal_engine downgrade
  confidence or mark the sub-signal as stale. Add a `_stale: True` to indicators
  so downstream knows.

- **`portfolio/signals/copper_gold_ratio.py:43`** — `_CACHE` keyed only by the
  string `"ratio_df"` — fine for a global pair, but the data is fetched per-call
  from yfinance via `price_source.download`. yfinance multi-call rate limiting will
  bite during loop hot-restarts after the 5-min cache expires.

## Tests missing

- No tests that exercise `_cached` signature for the newly-registered signals
  (mahalanobis_turbulence, crypto_evrp, etc.). A trivial `pytest.mark.parametrize`
  loop over `get_enhanced_signals()` calling each with a synthetic OHLCV would have
  caught the mahalanobis TypeError.
- No tests for the claude_fundamental cooldown-on-failure path. A test injecting
  `_call_claude_cli` failure should verify that `_cache[tier]["ts"]` is *not* bumped.
- No test that news_event's `_persist_headlines` doesn't race when run with two
  tickers in parallel — would catch the shared-file pattern.
- No look-ahead bias test. Standard pattern: run the signal on a DataFrame, then on
  the same DataFrame truncated by one bar — the result for bar N should be
  identical between the two runs. structure._highlow_breakout would fail this.
- No timeframe-resolution test for realized_skewness, drift_regime_gate,
  vwap_zscore_mr — all use fixed lookback constants that mean different absolute
  windows on 3h vs 1d.
- No test that disabled signals don't side-effect on import (per the "force-HOLD
  silently side-effects" risk).

## Cross-cut observations

1. **Cascade of bare `except: return HOLD`** is the dominant error idiom across
   ~15 signal modules. Combined with `DISABLED_SIGNALS` force-HOLD, the system
   cannot distinguish a deliberately-disabled signal from one that's silently
   raising every cycle. Add a `last_exception` field to `result["indicators"]` and
   require all `except` blocks to log at WARN level on first occurrence per process.

2. **Shared mutable module state without locks** in copper_gold_ratio,
   credit_spread, claude_fundamental (`_earnings_cache`), is duplication of work
   that `shared_state._cached` is built to handle. Migrate.

3. **Hardcoded thresholds calibrated on a single asset class get applied
   universally**: futures_flow LS thresholds (calibrated on BTC), drift_regime_gate
   fractions (paper used SPY 20yr daily), realized_skewness 252-bar lookback
   (paper used 1y daily), credit_spread `_CRISIS_LEVEL = 5.0` (defined for HY-OAS
   crisis bands, not crypto). Either gate per asset_class or document that they're
   noise on non-paper assets.

4. **Cache key naming inconsistency**: futures_flow uses
   `f"futures_flow_data_{ticker}"`, forecast uses `f"forecast_candles_{ticker}"`,
   news uses `f"news_headlines_crypto_{short}"` — but some signals share dict caches
   keyed by string `"ratio_df"`/`"mahalanobis_turb_closes"`/`"ratio_df"` with no
   ticker scoping. The latter is fine when the underlying data really is
   ticker-agnostic (cross-asset, macro), but easy to mis-extend later. Document
   per-cache scope in a comment.

5. **Backfill log files (`forecast_predictions.jsonl`, `claude_fundamental_log.jsonl`,
   `headlines_latest.json`, etc.) grow unbounded.** No rotation policy referenced
   in the modules themselves. `_bias_rate_from_entries` reads tail-only, but the
   tail-read pass still has to scan the file. With months of accumulated logs the
   I/O cost rises silently inside the per-cycle hot path.

6. **Look-ahead is the biggest silent risk** — at minimum the `structure._highlow_breakout`
   and `intraday_seasonality._get_utc_hour_and_dow` fallback warrant fixing, and a
   property-style test ("bar N's signal must not change when bar N+1 is appended")
   should be the new gate for any added signal.

7. **The 50-file plug-in plus DISABLED_SIGNALS pattern means that the static
   surface is much bigger than the active surface**. A registry health check that
   runs every disabled signal in a sandbox (synthetic DF, captured stdout/stderr,
   no real network) on every CI run would catch the Mahalanobis-style bugs before
   re-enable. Disabled signals are not stress-free dead code; they're booby traps
   armed to fire the day someone removes their name from `DISABLED_SIGNALS`.
