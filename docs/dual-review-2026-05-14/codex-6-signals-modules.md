# Adversarial Review — 6 signals-modules (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.
>
> Note: 50 modules — sampled ~14 in depth + systemic grep across all of them.
> In-depth sample: breakeven_inflation_momentum, cubic_trend_persistence,
> gold_overnight_bias, intraday_seasonality, metals_vrp, treasury_risk_rotation,
> vwap_zscore_mr, dxy_cross_asset, metals_cross_asset, orderbook_flow,
> williams_vix_fix, copper_gold_ratio, mahalanobis_turbulence, crypto_evrp,
> hurst_regime, drift_regime_gate, realized_skewness, structure, futures_basis,
> credit_spread, futures_flow, shannon_entropy, forecast, calendar_seasonal,
> cot_positioning. Skimmed: trend, volatility, momentum, mean_reversion,
> heikin_ashi, oscillators, fibonacci, smart_money, candlestick, news_event,
> complexity_gap_regime. Coverage gaps: macro_regime (only first 100 lines),
> hash_ribbons, statistical_jump_regime, ovx_metals_spillover, vix_term_structure,
> vol_ratio_regime, xtrend_equity_spillover, residual_pair_reversion,
> volume_flow, econ_calendar, gold_real_yield_paradox, momentum_factors,
> cross_asset_tsmom — not read.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/signals/mahalanobis_turbulence.py:99` — `_cached` invocation passes
  args in wrong order; raises `TypeError` on every call.
  ```python
  return _cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)
  ```
  `_cached` is defined as `_cached(key, ttl, func, *args)` in
  `portfolio/shared_state.py:37`. This call passes `_do_fetch` as the `ttl`
  positional and supplies `ttl=_CACHE_TTL` as a keyword — but `_cached` has
  no `ttl` keyword arg. Result: `TypeError: _cached() got an unexpected
  keyword argument 'ttl'`. Neither `_fetch_multi_asset_closes` nor
  `compute_mahalanobis_turbulence_signal` wraps this in try/except, so the
  exception propagates into the signal engine. The signal is in the
  "Enhanced Disabled" list per CLAUDE.md so this hasn't bitten production,
  but the moment anyone re-enables it the entire ticker cycle for that
  signal will explode (and depending on caller behaviour may abort the
  remaining sub-signals on the affected timeframe).

- `portfolio/signals/complexity_gap_regime.py:92` — identical `_cached`
  argument-order bug as Mahalanobis.
  ```python
  return _cached("complexity_gap_closes", _do_fetch, ttl=_CACHE_TTL)
  ```
  Same `TypeError` failure mode. Also currently disabled per CLAUDE.md but
  will crash on re-enable. Same root cause; presumably copy-paste from
  Mahalanobis.

- `portfolio/signals/copper_gold_ratio.py:251-252` — `action` flipped AFTER
  the sub-signals dict is built and AFTER confidence is computed; the
  recorded sub_signals therefore CONTRADICT the recorded action for any
  metals ticker.
  ```python
  if is_metals and action != "HOLD":
      action = "SELL" if action == "BUY" else "BUY"
  ```
  The dict on lines 260-265 returns `ratio_zscore`, `ratio_trend` etc. with
  their UN-INVERTED votes alongside the inverted final action. Two
  downstream consequences:
  (1) Any per-sub-signal accuracy backfill that compares sub-signal vote
      to outcome will measure the OPPOSITE of what was actually voted —
      sub-signal accuracy metrics for this module on XAU/XAG are
      systematically inverted.
  (2) The composite confidence (`majority_vote(votes, count_hold=False)`)
      is computed from un-inverted votes and then attached to the inverted
      action. If 3/4 votes were BUY (conf 0.75) the metals SELL inherits
      that 0.75 — a high-conviction SELL derived from a high-conviction
      BUY. Semantically the confidence number is fine (it measures
      agreement strength) but combined with (1) the audit trail is
      misleading and any future "diff sub_signals between modules"
      consistency check breaks on metals.

## P1 — high-confidence bugs (should fix)

- `portfolio/signals/hurst_regime.py:283-285, 301-302, 333-334` —
  same-vote double-counting in majority_vote.
  ```python
  sub_signals["hurst_regime"] = trend_vote       # e.g. "BUY"
  sub_signals["trend_direction"] = trend_vote    # also "BUY"
  ```
  Then line 333: `votes = list(sub_signals.values())` — the same trend
  vote enters the majority pool twice. In MR regime, `hurst_regime` and
  `mr_extreme` are similarly twinned (lines 301-302). Effect: the active
  sub-signal gets 2 votes against `hurst_momentum`'s 1, so a single
  trend_direction BUY at 0.3% EMA spread beats a HOLD from
  hurst_momentum. The intended composition (regime + direction +
  momentum) is collapsed to (direction × 2 + momentum × 1). Either omit
  the duplicate (`hurst_regime` should be a regime classifier sub-signal
  that votes HOLD, with `trend_direction` doing the BUY/SELL) or pass a
  deduplicated vote list to `majority_vote`.

- `portfolio/signals/vwap_zscore_mr.py:124-125` — bare
  `except Exception: ... return HOLD` swallowing every error.
  ```python
  except Exception:
      return {"action": "HOLD", "confidence": 0.0,
              "sub_signals": {}, "indicators": {}}
  ```
  No logging, no health metric, no `exc_info=True`. This is precisely the
  silent-failure pattern called out in the adversarial prompt (the 3-week
  Layer 2 outage was the same shape). If `_rolling_vwap` starts returning
  NaN-only series after a yfinance schema change, the signal silently
  becomes "HOLD on every ticker forever" and operators have no way to
  notice from logs. Other modules (e.g. forecast.py) use
  `logger.warning(..., exc_info=True)` plus `_log_health` — mirror that.

- `portfolio/signals/futures_flow.py:118, 135, 162, 287, 293-300` —
  direct dictionary key access on Binance FAPI response with no `.get()`
  fallback or KeyError handler.
  ```python
  latest = ls_ratio[-1]["longShortRatio"]          # line 118
  top_ls = top_position_ratio[-1]["longShortRatio"]  # line 135
  latest_rate = funding_history[-1]["fundingRate"]   # line 162
  ```
  If Binance changes a field name (they've done so before, e.g. the
  `10m` interval removal noted in MEMORY.md), this raises `KeyError`. The
  caller in `compute_futures_flow_signal` does not catch — the exception
  bubbles all the way up to whoever invoked the signal. Either guard with
  `.get(..., 0.0)` and skip the sub-signal, or wrap the whole compute in
  try/except with logging.

- `portfolio/signals/williams_vix_fix.py` (entire file, ~lines 67-167) —
  asymmetric vote pool: 3 of 4 sub-indicators can only ever vote BUY or
  HOLD. Only `_wvf_complacency` can vote SELL, and only when
  `low_count >= 8 of 10 bars` AND `rsi > 70` simultaneously.
  Consequence: under the `count_hold=False` majority rule (line 210), any
  non-trivial WVF reading will produce BUY with near-certain frequency
  because SELL requires a rare double-confirm. On a long uptrend the
  signal will mark BUY on every fear spike but never close, producing
  systematic long-only bias in the metals ensemble. This is by design per
  the docstring ("bottom/capitulation detection") but it's worth
  documenting that the sub-signal SHOULD only contribute BUY votes and
  the SELL branch is essentially decorative. If accuracy is being
  measured on overall {BUY, SELL, HOLD} this signal will look great in
  bull markets and terrible in bears purely from the structural bias.

- `portfolio/signals/metals_cross_asset.py:220, 224` — "≥ 3 of 4 sources
  healthy" gate silently drops one intraday source on degraded sessions,
  but the gated source still injects HOLD into the 8-vote tally.
  ```python
  use_intraday = intraday_ok >= 3
  ...
  result["copper_change_pct"] = copper["change_3h_pct"] if copper else 0.0
  ```
  When (say) `oil` is missing, `oil_change_pct = 0.0` → oil sub-signal
  votes HOLD. The 7 remaining sub-signals are unchanged but oil's HOLD
  dilutes confidence calculation. With `count_hold=False` (default in
  `majority_vote`) HOLD doesn't enter the denominator — but the logger
  warning fires only AT WARNING level and there's no aggregate per-source
  health tracking, so repeated single-source outages won't trigger any
  alert. Either gate to "≥ 4 of 4" or vote DROP instead of HOLD for the
  degraded source.

- `portfolio/signals/realized_skewness.py:43-46, 61-69` — rolling
  skewness window equals the available-data length, producing only one
  valid rolling value when input is short.
  ```python
  def _compute_rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
      return returns.rolling(window=window, min_periods=max(window // 2, 20))...
  ```
  Then at line 55: `lookback = min(SKEW_LOOKBACK, len(returns))` —
  passed in as `window`. With 80 rows of returns, `window=80`,
  `min_periods=40` → the rolling skew series has up to 41 valid points
  but `recent = rolling_skew.iloc[-NORM_WINDOW:]` (line 62, NORM_WINDOW=60)
  with `.std()` over 41 points is computed, then `(skew_val - mean_skew)
  / std_skew` is the z-score. The z-score's std uses an overlapping
  rolling window of the SAME data points that overlap heavily — adjacent
  rolling values share ~79/80 of their underlying returns, so the std is
  artificially small. The signal will look very confident even when the
  underlying skew didn't actually change. Effect: aggressive
  Z_BUY/Z_SELL crossings on short-history tickers.

- `portfolio/signals/cot_positioning.py:213-217` — sub-signal labelled
  `commercial_change` is computed from `noncomm_net_change` (negated).
  ```python
  change = cot_data.get("noncomm_net_change")
  ...
  indicators["comm_net_change"] = -change  # Commercial change is inverse
  ```
  Indicator field `comm_net_change` and sub-signal name `commercial_change`
  imply this is commercial-hedger data, but the actual source is the
  non-commercial (speculator) change with a sign flip. This works as a
  proxy only if the OI is closed (every contract has a counter-party),
  which is roughly true at aggregate but breaks when `commercial_total`
  ≠ `-noncommercial_total`. If actual `comm_net_change` field exists in
  upstream `cot_data`, use it directly; otherwise rename the sub-signal
  to `speculator_change_inverse` so the audit trail is honest.

- `portfolio/signals/gold_overnight_bias.py:118-140` — `_fix_proximity_vote`
  is unidirectional: it ALWAYS votes BUY when within 90 minutes of either
  fix.
  ```python
  if min_dist == dist_pm:
      return "BUY", 0.3 * proximity_strength
  else:
      return "BUY", 0.2 * proximity_strength
  ```
  Combined with `_session_phase_vote` which can vote SELL (London PM
  session), the proximity vote can never align with a SELL. So at 14:00
  UTC (in London PM session → SELL with confidence ~0.7), if proximity
  is within 90 min of 15:00, the BUY-only proximity vote cancels one of
  the SELL votes in the majority pool. The intended SELL during London
  PM is structurally weakened near the fix boundary. Either let proximity
  vote in the SAME direction as `_session_phase_vote`, or have it abstain
  (HOLD) near the boundary.

## P2 — concerns / smells (worth addressing)

- `portfolio/signals/structure.py:79-82` — `_highlow_breakout` votes BUY
  when current_close is within 2% of the 52-week HIGH and SELL when
  within 2% of the LOW. This is trend-following labelled as "breakout" —
  there's no actual breakout check (close > prior_high), just proximity.
  At a 52w peak the signal flashes BUY perpetually until price drops 2%,
  potentially feeding into Layer 2 decisions near tops.
  ```python
  if pct_from_high <= 0.02:
      return "BUY", indicators
  if pct_from_low <= 0.02:
      return "SELL", indicators
  ```

- `portfolio/signals/forecast.py:914, 921` — health booleans
  `kronos_ok = kronos is not None and bool(kronos.get("results"))` and
  `chronos_ok = chronos is not None` use different definitions. Chronos
  is "healthy" if the result is non-None even if it's an empty dict or
  has missing horizon keys. The `_health_weighted_vote` at line 470
  reads `sub_signals.get("chronos_1h", "HOLD")` which masks this — a
  result with missing 1h key becomes a HOLD vote silently. Cleaner:
  define `chronos_ok = chronos is not None and ("1h" in chronos or "24h"
  in chronos)`.

- `portfolio/signals/copper_gold_ratio.py:79-86, 80-89` — bare
  `except Exception` around `price_source.download` then ANOTHER bare
  `except Exception` around `yfinance.download` fallback, all logged at
  WARNING. If both fail, the warning is logged once but the underlying
  reason (network? auth?) is swallowed — `exc_info` is not passed.

- `portfolio/signals/intraday_seasonality.py:110-118` — `_hour_alpha_vote`
  has a structural issue: BUY is returned for `mult >= 1.2` regardless of
  trend direction (the hour says "high alpha" → BUY only). For metals at
  hour 13-15 UTC this fires BUY even during a confirmed downtrend on the
  same bar. The composite at lines 188-198 partially compensates by
  requiring trend alignment, but `_hour_alpha_vote`'s "BUY" is then
  passed into sub_signals["hour_alpha"] — recorded as a directional vote
  despite being a time-of-day classifier. Use HOLD for high-alpha hours
  and let trend determine direction.

- `portfolio/signals/calendar_seasonal.py` — eight sub-signals where
  most can only vote one direction (DOW: Mon=SELL, Fri=BUY; Month-end=BUY;
  Sell-in-May=SELL May-Oct/BUY Nov-Apr; January=BUY; Pre-holiday=BUY;
  FOMC-drift=BUY; Santa=BUY). Out of 8 sub-signals, six can vote BUY
  and only three can vote SELL — same asymmetry as williams_vix_fix.
  Conservative `_MAX_CONFIDENCE = 0.6` partially mitigates but the
  long-bias is structural.

- `portfolio/signals/treasury_risk_rotation.py:187-188` — index math is
  off-by-one safe but obscure:
  `ief.iloc[-min(_SLOPE_LOOKBACK, len(ief) - 1) - 1]`. If `len(ief) ==
  _SLOPE_LOOKBACK = 65`, then `-min(65, 64) - 1 = -65`. The data
  contract upstream guarantees ≥66 rows (line 68) so this works, but the
  expression reads as "65 days ago" while computing 64 days ago in the
  edge case. Use `min(_SLOPE_LOOKBACK + 1, len(ief))` for clarity.

- `portfolio/signals/metals_vrp.py:153-176` — `current_rv` and
  `current_gvz` come from independent sources (yfinance OHLCV vs FRED
  daily index). They are not date-aligned — `close.iloc[-1]` is the
  latest OHLCV bar (which on a weekend is Friday-close for some sources)
  while `gvz_data[0]` is the latest FRED publication. Off-by-one-day
  VRP can flip sign near regime boundaries. Align by date before
  subtraction.

- `portfolio/signals/breakeven_inflation_momentum.py:60-63` — convoluted
  config lookup uses a single-line ternary with `getattr` chains and
  `hasattr` checks:
  ```python
  return getattr(cfg, "fred_api_key", "") or getattr(
      getattr(cfg, "golddigger", None), "fred_api_key", ""
  ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
  ```
  The operator-precedence intent is ambiguous; if `cfg` is a non-dict
  object lacking both attributes the `else ""` branch runs, but adding
  a third config schema would require parsing this carefully. The same
  pattern is duplicated in `metals_vrp.py:120-122`,
  `metals_cross_asset.py:100-102`, `credit_spread.py:134-136`. Factor
  into a shared helper.

- `portfolio/signals/futures_flow.py:54, 89, 196-197` — divide-by-zero
  guard uses `if recent_oi[0]` which treats both `0` and `None` as
  falsy → returns 0 momentum. Fine semantically, but combined with the
  `.get("oi", 0) or 0` defaults on line 53 means a Binance response
  containing `{"oi": null}` silently produces a momentum of 0 → HOLD,
  with no observability that the data was actually missing.

- `portfolio/signals/credit_spread.py:154` — `history = values[:lookback]`
  where `values` is newest-first AND `current = values[0]`. So `current`
  is included in the rolling mean/std used to z-score `current`. Minor
  in-sample bias (1/252 weight on itself), but if you're trying to match
  published z-scores (e.g. against FRED's own normalization) you'll be
  ~0.1 std off near regime boundaries.

- `portfolio/signals/copper_gold_ratio.py:43-44` — single module-level
  dict `_CACHE` used without a lock; if the signal engine runs tickers
  in parallel via ThreadPoolExecutor (it does — CLAUDE.md notes 8
  workers) two threads can race on
  `_CACHE["ratio_df"] = (now, combined)`. Last-write-wins on a dict
  set is atomic in CPython for one key, so no torn read, but stale data
  could be over-written by an in-flight computation from another
  thread. `metals_vrp.py` does this correctly with `_gvz_lock`;
  `copper_gold_ratio.py` should mirror.

- `portfolio/signals/cubic_trend_persistence.py:142` — `phi_threshold`
  for trend exhaustion uses formula `sqrt(-b / (3*c))`. With the
  configured `B_DAILY=0.0129, C_DAILY=-0.0062` this yields ~0.83. But
  phi is clipped to [-2.5, 2.5] on line 50, so the exhaustion sub-signal
  fires whenever `|phi| > 0.83` — about 50% of the time on a strongly
  trending asset. The exhaustion vote will dominate `trend_direction`
  during sustained moves, producing a contrarian bias that contradicts
  the module's stated "weak trends persist" intent.

## Did NOT find

1. Silent failures: WIDE pattern found — vwap_zscore_mr (P1 above) and
   silent yfinance fetch failures across treasury_risk_rotation,
   mahalanobis_turbulence, network_momentum, copper_gold_ratio. Logged
   in findings above.
2. Race conditions: No file-write races (signal modules are pure
   computation; persistence is delegated to caller). One unlocked
   module cache in copper_gold_ratio (P2 above). `metals_vrp` and
   `metals_cross_asset` correctly use `threading.Lock` on their FRED
   caches.
3. Money-losing bugs: No direct order placement or position sizing in
   this subsystem — signals only output BUY/SELL/HOLD votes consumed by
   downstream portfolio logic. Sub-signal/action mismatch in
   `copper_gold_ratio` flagged P0 as data-corrupting.
4. State corruption: No JSONL/JSON writes in any sampled module. (The
   forecast.py atomic_append_jsonl path is correctly locked.)
5. Logic errors that pass tests: Did not inspect tests — many modules
   compute on rolling windows that include the current bar (futures_basis,
   credit_spread z-scores include `values[0]` in their own mean); these
   are not look-ahead bias (they only use past bars) but they DO bias
   z-score magnitudes. Likely passes tests because tests probably feed
   the same data shape.
6. Resource leaks: forecast.py uses ThreadPoolExecutor as context
   manager (correctly closed). No bare socket / file handle leaks
   spotted in sampled modules.
7. Time/timezone bugs: `gold_overnight_bias._get_utc_time` correctly
   converts `tzinfo`-aware timestamps to UTC and falls back to
   `datetime.now(timezone.utc)` for tz-naive index. `intraday_seasonality`
   uses the same pattern. `calendar_seasonal` not deeply inspected.
8. API misuse: `futures_flow.py` direct-key access flagged P1.
   `futures_basis.py` uses `premiumIndexKlines` correctly. `metals_vrp`
   FRED params are correct (`sort_order=desc`, `limit=300`). No
   Binance `10m` interval misuse seen in sampled files.
9. Trust boundary violations: `cot_positioning._fetch_cot_historical`
   builds the SOCRATA URL via f-string with `commodity_name` which is
   passed in as `"GOLD"` or `"SILVER"` (hard-coded literal in the caller
   at line 364). Not user-controlled. No `eval/exec/shell=True` seen.
10. Incorrect partial-state assumptions: Multiple modules call
    `oi_history[-1]["longShortRatio"]` etc. without checking the key
    exists (P1 in futures_flow). `intraday_seasonality._classify_asset`
    falls back to "crypto" when ticker is empty — could mis-classify a
    pre-context-init invocation. Hurst/realized_skewness assume
    sufficient rolling-window samples (P1 above).
