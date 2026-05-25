# Signals-Modules Adversarial Review

Scope: 68 modules under `portfolio/signals/*.py` in the worktree
`Q:/finance-analyzer-worktrees/review-signals-modules/`. Focus on ACTIVE
voters, recently added (Apr–May 2026) signals, and shadow-track signals.

The single most important environmental fact for grading these findings:
`generate_signal()` is invoked with `df = _fetch_klines(..., interval="15m", limit=100)`
(see `portfolio/main.py:492` and `portfolio/main.py:502-504`). Many of the
"new" signals were obviously authored with a *daily-bar* assumption — their
lookback constants are sized in "trading days" (252, 60, 50, …) but get
applied to 15-minute bars, silently shrinking the lookback by ~96×. This
re-occurs across enough modules to warrant a cross-cutting fix.

## P0 findings

`portfolio/signals/gold_real_yield_paradox.py:282-285,290-291`
  🔴 Timeframe mismatch — daily yield change vs intraday gold price.
  `gold_30d_return = close.iloc[-1] / close.iloc[-min(30, len(close))] - 1`
  reads `close.iloc[-30]` from a 15-minute-bar DataFrame (= 7.5h, not 30
  days) while `yield_30d_change` correctly reads 30 daily FRED
  observations. `_compute_baseline_correlation()` (line 293) then
  correlates hourly gold returns against daily yield diffs. The paradox
  spread, correlation break, and momentum split sub-signals are all
  computed on apples-to-oranges series. Currently disabled / not
  shadow-tracked, but the signal will produce statistically meaningless
  votes the moment it is re-enabled. Fix: resample df to daily bars
  before computing 30d return, or fetch its own daily price series like
  the rest of the macro signals do.

`portfolio/signals/cross_asset_tsmom.py:107-127,191`
  🔴 Same TF mismatch on the "own TSMOM" sub-indicator.
  `_compute_own_tsmom(close)` does `close.iloc[-lookback - 1]` with
  `_TSMOM_LOOKBACK = 252`. On a 15-minute-bar df this is 63 hours of
  data, not 12 months. The other three sub-indicators (cross-pair, bond,
  equity) correctly use 252-day yfinance daily closes. The composite
  vote then mixes one intraday-momentum vote with three multi-month
  momentum votes. Indicator label `own_ret_252d` is therefore wrong by
  the same factor.

`portfolio/signals/network_momentum.py:107-148,166`
  🔴 Same TF mismatch. `_compute_own_tsmom(close)` and
  `_compute_network_divergence` use `_MOM_WINDOWS = [5, 20, 60]` and a
  20-bar own-momentum window on a 15-minute df (75min / 5h / 15h),
  while peer momentum is 20-day yfinance-daily. The "divergence" between
  own and network momentum is therefore computed on incompatible time
  horizons; a strong peer 20d trend is being compared against the
  target's last 5 hours.

`portfolio/signals/bocpd_regime_switch.py:227-232`
  🔴 Sub-signal contract violation that will silently corrupt any
  downstream consumer that assumes BUY/SELL/HOLD:
  ```
  sub_signals = {
      "changepoint_detector": "BREAK" if is_changepoint else "STABLE",
      "trend_follower": trend_action,
      "mean_reverter": mr_action,
      "regime_classifier": regime,   # e.g. "changepoint_mr"
  }
  ```
  `_validate_signal_result` only checks that `sub_signals` is a dict, so
  these escape the validator and land in `extra_info`. Today nothing
  reads sub_signals downstream, but the dashboard's signal-card renderer
  (`/api/signals`, `/api/signal-log`) and any future accuracy-by-sub-signal
  analysis will choke on the non-canonical strings. Strict P0 if anyone
  downstream ever uses `votes.count("BUY")` or equivalent.

## P1 findings

`portfolio/signals/credit_spread.py:283-288`
  🟠 Relative-path config fallback re-introduces the exact CWD bug that
  cot_positioning.py:27-33 documented and fixed two months ago. When
  `_get_fred_key(context)` returns "" (config object misshapen), the
  fallback reads `load_json("config.json", default={})` with no absolute
  path resolution. PF-DataLoop runs under `C:\Windows\System32` on a
  fresh task-scheduler start; the fallback silently returns `{}` and the
  signal returns `empty` for the rest of the session. Mirror the
  `_DATA_DIR = Path(__file__).resolve().parent.parent.parent` pattern
  cot_positioning uses.

`portfolio/signals/gold_real_yield_paradox.py:262-269`
  🟠 Same relative-path config fallback (`load_json("config.json")` on
  line 265). Same CWD bug, same fix.

`portfolio/signals/hurst_regime.py:283-285,300-302`
  🟠 Double-counted vote — the hurst regime sub-signal is set to
  `trend_vote` AND `trend_direction` is set to the same value when the
  Hurst regime is "trending":
  ```
  sub_signals["hurst_regime"]    = trend_vote
  sub_signals["trend_direction"] = trend_vote
  ```
  Same pattern for mean-reverting regime with `mr_vote`. `majority_vote`
  then sees the same vote twice and gives it 2/3 of the weight. Either
  pick one canonical slot or count distinct contributions only.

`portfolio/signals/copper_gold_ratio.py:251-252,260-264`
  🟠 Action / sub_signal divergence after inversion. The action is
  inverted for metals ("falling ratio = gold strength = BUY metals") but
  the sub_signals dict is built from the un-inverted votes (`zscore_vote`
  etc are still "SELL" while the returned `action` is now "BUY"). Anyone
  joining the per-sub-signal log against the headline action gets random
  noise on metals tickers. Either invert sub_signals symmetrically, or
  store both raw and inverted with explicit names.

`portfolio/signals/treasury_risk_rotation.py:182-198`
  🟠 Same inversion / sub_signal mismatch as copper_gold_ratio. Inverts
  the headline action for safe havens but leaves `slope_direction`,
  `slope_momentum`, `slope_zscore`, `regime_persistence` un-inverted.

`portfolio/signals/xtrend_equity_spillover.py:226-252`
  🟠 Same inversion / sub_signal mismatch for safe havens.

`portfolio/signals/crypto_evrp.py:204-243`
  🟠 The `evrp_percentile` sub-signal does not compute an eVRP percentile.
  Despite the name and docstring, the implementation falls back to
  ranking *DVOL alone* against the recent DVOL distribution ("Use just
  the DVOL percentile as proxy"). The vote direction is then inverted
  vs the level signal interpretation: high DVOL pctile ⇒ SELL
  ("compression coming") while `_evrp_level_signal` uses high eVRP ⇒
  SELL. These look identical but are computing different statistics on
  different inputs, leaving the operator to debug a confusing accuracy
  curve. Either implement the eVRP percentile correctly, or rename the
  sub-signal `dvol_percentile`.

`portfolio/signals/crypto_evrp.py:248-261`
  🟠 `_evrp_momentum_signal` and `_evrp_level_signal` carry contradictory
  semantic models. Level signal says high eVRP ⇒ SELL (vol compression
  preceding price decline). Momentum signal says rising DVOL ⇒ SELL
  ("more risk priced in = bearish"). These are conflicting theses
  about what high implied vol implies; they will systematically fight
  each other in the composite vote.

`portfolio/signals/metals_vrp.py:125-153`
  🟠 Annualisation factor assumes daily bars on intraday data.
  `_compute_realized_vol(close, _RV_WINDOW)` multiplies the rolling
  20-bar std by `sqrt(252)`, but the upstream df is 15-minute bars (~96
  per trading day). The correct annualisation is `sqrt(252*96)` for
  15m data — current code underestimates RV by ~9.8×, so `vrp = gvz -
  rv` is dominated by GVZ and `vrp_z` becomes mostly a noisy GVZ-z proxy.
  The downstream gold/silver direction calls are derived from this bias.

`portfolio/signals/cubic_trend_persistence.py:58-70,114-121`
  🟠 Auto-detection of timeframe via index spacing reads "hourly" for
  15-minute bars (`median_hours < 20 → hourly`) and then applies
  `B_HOURLY = 0.00132 / C_HOURLY = -0.00039` — coefficients fit on
  *hourly* data, not 15-minute. The cubic model E[R] = b·φ + c·φ³ is
  calibration-sensitive; using hourly coefficients on quarter-hour
  returns rescales the expected-return threshold inappropriately.

`portfolio/signals/hash_ribbons.py:158-179,256-258`
  🟠 Timeframe mismatch in the price filter. Hashrate is daily by
  construction (blockchain.info); price filter `_price_momentum_filter`
  uses PRICE_FAST=10, PRICE_SLOW=20 on df.close (15m bars → 2.5h vs
  5h MAs, not the documented "10d vs 20d"). The hash-ribbon recovery
  detection on hashrate is correct daily, but the gating condition is
  effectively a noisy intraday momentum check.

`portfolio/signals/trend_slope_momentum.py:97-100,28-35`
  🟠 Same intent-vs-implementation gap: MOMENTUM_LOOKBACK=50 documented
  as "50-day momentum confirmation", Z_LOOKBACK=252 as a "252-day
  rolling z-score window". Both run on the 15m df, so the actual
  windows are 12.5h and 63h. Probabilities and z-scores carry the
  daily-bar interpretation in the indicator labels (`p_bull`, `z_score`)
  but the units are completely different.

`portfolio/signals/residual_pair_reversion.py:276-296`
  🟠 Index alignment bomb. `df["close"]` carries a RangeIndex (the
  binance fetcher writes the timestamp into `df["time"]`, not the
  index — see `data_collector.py:96`). Line 277 then does
  `target_close.index = pd.to_datetime(df.index)` which converts integer
  range positions into datetimes near 1970-01-01. Joined with
  `driver_close` (daily DatetimeIndex), the inner-join `.dropna()`
  produces an empty DataFrame on every call, so the signal returns the
  empty hold result for the lifetime of the process. Today the signal
  is disabled (not shadow-safe — yfinance call), but the moment it gets
  flipped on it will deliver zero votes silently.

`portfolio/signals/futures_basis.py:30-44,156-157,73-89`
  🟠 Docstring/threshold drift between code and comments:
  `_Z_THRESHOLD_BUY = -1.5 / _Z_THRESHOLD_SELL = 1.5`, while both the
  module docstring (line 11–13 and `_basis_z_extreme` docstring at
  line 70–73) say "extreme backwardation (z < -1.8)". `_basis_acceleration`
  requires `len(basis_values) >= 2 * _VELOCITY_WINDOW + 1 = 49`, but
  `_MIN_KLINES = 48` is the global gate; for the boundary case where
  exactly 48 klines are returned, three of the four sub-indicators
  fire while the fourth always returns HOLD.

`portfolio/signals/vix_term_structure.py:37,109-123`
  🟠 `_Z_THRESHOLD = 0.0` means the ratio_zscore sub-indicator can
  never return HOLD — every cycle it casts either BUY or SELL (strict
  `>` and `<` over zero). Combined with the always-active
  backwardation_flag / contango_depth sub-signals, the composite vote
  is structurally biased toward producing a directional action almost
  every cycle. The "z=0.0 >> z=1.0" backtest comment justifies the
  threshold choice but the design no longer matches "z-score sub-
  indicator" intent — at this point it's just a sign-of-ratio voter
  and should be renamed accordingly.

## P2 findings

`portfolio/signals/cot_positioning.py:138-194,226-265`
  🟡 Look-back-includes-current-bar bias. `_compute_cot_index`
  prepends the current `nc_net` value to the history before computing
  min/max for the percentile, so when the current value IS the
  extreme, the percentile snaps to 100 (or 0) by construction.
  `_sub_managed_money` does the same with mean/std for the z-score.
  With N≈156 the bias is modest but it weakens contrarian extremes
  exactly when they should be sharpest.

`portfolio/signals/connors_rsi2.py:142-152`
  🟡 Sub-signal shape contract violation: returns dict-of-dicts
  (`{"value": …, "signal": …}`) instead of dict-of-strings, while every
  other module returns dict-of-strings. Doesn't crash because
  `_validate_signal_result` only checks the outer dict type, but any
  consumer that does `votes.count("BUY")` over `sub_signals.values()`
  will get garbage. Same pattern in `adx_regime_switch.py:152-159`.

`portfolio/signals/bocpd_regime_switch.py:227-232`,
`portfolio/signals/sentiment_extremity_gate.py:181-209,224-237`,
`portfolio/signals/choppiness_regime_gate.py:106-138`,
`portfolio/signals/tsi_chop_mr.py:138-165,167-181`,
`portfolio/signals/vol_ratio_regime.py:255-261`
  🟡 Sub-signal value contract violation — these emit regime-label
  strings ("PASS", "TREND", "trending", "ranging", "neutral",
  "transition", "STABLE", "BREAK", "HOLD (trending)") rather than the
  canonical {BUY, SELL, HOLD}. Same downstream risk as above.

`portfolio/signals/intraday_seasonality.py:110-118,176-194`
  🟡 The `hour_alpha` sub-signal returns "BUY" when the hour
  multiplier is high — but a "favorable hour" is not a direction. The
  actual direction comes from the trend sub-signal; meanwhile this
  vote labels every favorable-hour as BUY (or every unfavorable as
  HOLD). Any per-sub-signal accuracy tracking is meaningless because
  the label has no relationship to expected price direction.

`portfolio/signals/gold_overnight_bias.py:71-97,118-141,164-198`
  🟡 Structural BUY bias. `_session_phase_vote` never returns HOLD
  (always BUY or SELL with confidence based on session depth) and
  `_fix_proximity_vote` only ever returns BUY or HOLD. Across 24h the
  session-phase voter spends ~1170 min/day voting BUY and only
  270 min voting SELL, and the proximity voter is BUY-only. With
  `majority_vote(..., count_hold=False)`, the signal will lean BUY
  even when the trend sub-signal disagrees.

`portfolio/signals/news_event.py:222-247`
  🟡 `_keyword_severity_vote` starts with `max_sev = "normal"` and
  `max_weight = 1.0`; the code only updates `max_sev` when
  `weight > max_weight`, so a "moderate" headline (weight == 1.0
  per news_keywords default) never updates max_sev. The function's
  return branches only support "critical"/"high" → SELL, "normal" →
  HOLD. Despite the docstring claim "Multiple moderate positive →
  BUY", this sub-signal CANNOT emit BUY — structurally SELL/HOLD only.

`portfolio/signals/ovx_metals_spillover.py:104-110,147-163`
  🟡 `_ovx_level_signal` and `_ovx_reversion_signal` produce mutually
  contradictory votes in the high-pctile reversal scenario.
  When OVX is >80th pctile AND falling >5%, sub1=SELL, sub4=BUY,
  cancelling out. This is by design but produces frequent ties that
  silently degrade the signal to HOLD with high conviction warranted.

`portfolio/signals/crypto_macro.py:228,281`
  🟡 `OPTIONS_TTL` is referenced inside the function body (line 228)
  but defined at module bottom (line 281). Works at runtime because
  module-level names are resolved at call time, but it's a fragile
  pattern — any function-level reordering that creates closures or
  import shadowing will produce NameError silently.

`portfolio/signals/crypto_macro.py:100-102`,
`portfolio/signals/credit_spread.py:125-136`,
`portfolio/signals/breakeven_inflation_momentum.py:53-63`,
`portfolio/signals/metals_vrp.py:112-122`,
`portfolio/signals/metals_cross_asset.py:91-102`,
`portfolio/signals/gold_real_yield_paradox.py:43-53`
  🟡 The `_get_fred_key` helper is duplicated verbatim across six
  modules with a convoluted ternary at the end:
  ```
  return getattr(cfg, "fred_api_key", "") or getattr(
      getattr(cfg, "golddigger", None), "fred_api_key", ""
  ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
  ```
  If `cfg.golddigger` is itself a dict instead of an object, the
  `getattr(dict, "fred_api_key", "")` quietly returns `""` (dicts
  don't have attributes). The dict-vs-object handling is inconsistent
  with the first `isinstance(cfg, dict)` branch and silently degrades
  on mixed config. Factor into a single helper in `portfolio.config`.

`portfolio/signals/metals_cross_asset.py:169-180`,
`portfolio/signals/credit_spread.py:148-174`
  🟡 `_compute_zscore` (and `_oas_zscore_signal`) compute mean / std
  over a window that INCLUDES the current value (`values[0]` is the
  newest, and `history = values[:n]` is `values[0:n]`). The "current"
  observation contaminates its own baseline, biasing z-scores toward
  zero (modestly with N=252, more so for shorter windows). Use
  `history = values[1:n+1]` to exclude the current bar.

`portfolio/signals/mean_reversion.py:107-122,295-333`
  🟡 `_internal_bar_strength` and `_ibs_rsi2_combined` derive IBS
  from the LAST bar of the df. In the 60s main loop, the last bar
  is the currently-forming bar — high/low/close are not finalised
  until the bar closes. IBS will jitter as the bar evolves, producing
  early-bar BUY/SELL votes that may reverse before the bar closes.
  Standard practice for bar-completion-dependent features is to read
  `iloc[-2]` (last completed bar) and accept the 5-15 min lag.

`portfolio/signals/mahalanobis_turbulence.py:193-204`
  🟡 `_turbulence_z_vote` has two thresholds (`_Z_HIGH = 2.0` and
  `_Z_ELEVATED = 1.5`) but both produce the same vote (BUY for safe
  havens, SELL for risk assets). One of the thresholds is dead code;
  the conf-boost at line 310-311 uses `_Z_HIGH` separately.

`portfolio/signals/amihud_illiquidity_regime.py:75-81`
  🟡 Z-score thresholds are asymmetric (`> 2.0` ⇒ SELL, `< -1.0` ⇒
  BUY). Illiquidity is normally interpreted as a *regime gate* (high
  illiq ⇒ suppress directional votes from other voters), not a
  directional signal. Voting SELL on high illiquidity hard-codes a
  prior that may not hold across crypto / metals / MSTR.

`portfolio/signals/drift_regime_gate.py:35-59`
  🟡 The paper this references (arxiv:2511.12490) treats drift
  regimes as a *gate* for value+reversal signals, not as a standalone
  directional signal. Implementing >60% positive days as "overextended
  ⇒ SELL" inverts the paper's mechanism. The OOS Sharpe >13 claim in
  the docstring cannot reasonably be expected from this
  implementation.

`portfolio/signals/cryptotrader_lm.py:150-159` and
`portfolio/signals/finance_llama.py:204-214`
  🟡 When the parser can't extract a confidence, both modules default
  `confidence = 0.50`. If `_parse_response` returns `decision="HOLD"`,
  the result is `action=HOLD, confidence=0.50` — a "confident HOLD"
  that contradicts the project convention (see
  `signal_utils.majority_vote`: "HOLD confidence is always 0.0 — it's
  the absence of a signal, not a directional vote"). Clamp confidence
  to 0 whenever the parsed action is HOLD.

## P3 findings

`portfolio/signals/cross_asset_tsmom.py:228`
  🔵 Indicator `gld_ret_63d` references "GLD" but `_YF_TICKERS` only
  contains GC=F. Indicator always reports None.

`portfolio/signals/ttm_squeeze.py:197`
  🔵 Dead conditional: `"squeeze_state": "HOLD" if currently_squeezing
  else "HOLD"` — both branches return "HOLD".

`portfolio/signals/signal_utils.rsi` (`signal_utils.py:32-43`)
  🔵 When `avg_loss` is zero (all gains in the window), the function
  returns NaN. Standard interpretation is RSI=100. Affects every
  caller of `rsi()` — mean_reversion, momentum, hurst_regime,
  williams_vix_fix, etc.

`portfolio/signals/trend_slope_momentum.py:38-44`
  🔵 `_ema_smooth` uses a Python loop with `iloc[i]` access. Could
  be a one-liner: `series.ewm(alpha=1-EMA_LAMBDA, adjust=False).mean()`.
  100-row df makes the perf hit irrelevant but the code is misleading.

`portfolio/signals/autotune_adaptive_cycle.py:23-104`
  🔵 Three Ehlers DSP filters implemented as Python loops. On a
  100-row df it's fine; if ever applied to larger dfs (e.g. backtester
  on the full daily history) this will be 30-100× slower than the
  pandas-native equivalents.

## Cross-cutting observations

1. **Daily-bar assumption applied to 15m bars** — the biggest
   recurring class of bug. main.py feeds `interval="15m", limit=100`,
   yet `gold_real_yield_paradox`, `cross_asset_tsmom`,
   `network_momentum`, `cubic_trend_persistence`,
   `trend_slope_momentum`, `metals_vrp`, `hash_ribbons`,
   `realized_skewness` (less severe), all interpret lookback constants
   in "trading days". Either resample to daily inside each signal, or
   document the time-unit mismatch and rename indicators. The fact
   that these all *register* successfully without crashing means the
   error is silent until you look at the math.

2. **Sub-signal value contract is unenforced** — at least seven
   modules emit non-{BUY,SELL,HOLD} strings as sub-signal values
   ("PASS", "TREND", "trending", "ranging", "neutral", "transition",
   "STABLE", "BREAK", "HOLD (trending)") or dict-of-dicts
   (connors_rsi2, adx_regime_switch). `_validate_signal_result` only
   validates the outer dict type, not contents. This is a
   reliability landmine — the moment anyone writes
   `votes = [v for v in sub_signals.values()]; majority_vote(votes)`,
   the calculation silently rolls "trending" into the "neither BUY
   nor SELL" bucket. Recommendation: extend `_validate_signal_result`
   to validate each sub-signal value or coerce to HOLD.

3. **Inversion / sub-signal divergence on safe-haven invert** —
   `copper_gold_ratio`, `treasury_risk_rotation`,
   `xtrend_equity_spillover` all invert the headline `action` for
   metals tickers but leave the per-sub-signal vote slots un-inverted.
   Per-sub-signal accuracy tracking on those tickers is meaningless.

4. **Look-back-includes-current-bar z-score bias** —
   `metals_cross_asset._compute_zscore`,
   `credit_spread._oas_zscore_signal`, `cot_positioning._sub_managed_money`,
   `cot_positioning._compute_cot_index`,
   `mahalanobis_turbulence` (percentile only). All include the latest
   value inside the baseline window, contaminating mean/std/min/max
   with their own observation. Statistically modest with N=252; sharp
   with shorter windows; in all cases the fix is one slice change
   (`values[1:n+1]` instead of `values[:n]`).

5. **Relative-path config reads** — `credit_spread.py:285` and
   `gold_real_yield_paradox.py:265` both do
   `load_json("config.json")` as a fallback. cot_positioning.py
   explicitly documented this exact bug being fixed on 2026-05-02
   (SM-P1-4) using absolute `_DATA_DIR`. Two newer modules
   reintroduced the same hazard.

6. **Module-level mutable caches under ThreadPoolExecutor** — many
   signals use top-level `_xxx_cache: dict = {}` with manual ts/key
   tracking (`futures_basis`, `metals_cross_asset`, `credit_spread`,
   `crypto_evrp`, `metals_vrp`, `gold_real_yield_paradox`,
   `breakeven_inflation_momentum`, `hash_ribbons`, `ovx_metals_spillover`,
   `vix_term_structure`, `copper_gold_ratio`). Only `metals_vrp`,
   `gold_real_yield_paradox`, `metals_cross_asset`, and `news_event`
   use a `threading.Lock`. The 8-worker pool will race on dict
   assignment in the other modules. CPython GIL makes the
   single-statement writes atomic in practice, but stale reads and
   inconsistent (key, data, time) triplets are possible during cache
   refresh — manifesting as a momentarily stale cache for a different
   API key. Recommend: standardise on `portfolio.shared_state._cached`,
   which uses a lock internally.

7. **Code duplication in FRED fetching** — `_get_fred_key` and
   `_fetch_*` patterns are duplicated across six modules with subtle
   divergence (some use `fetch_with_retry`, some inline `requests`,
   some swallow ImportError silently). Factor into
   `portfolio.fred_client`.

8. **"Always-active" sub-signals defeat the abstain-with-HOLD
   convention** — `vix_term_structure._ratio_zscore` (threshold 0),
   `gold_overnight_bias._session_phase_vote` (always BUY or SELL),
   `cubic_trend_persistence.trend_exhaustion` (threshold dependent on
   c≠0 but very low). These structurally force directional votes and
   distort the composite when other voters legitimately HOLD.

## Files reviewed

In detail (read full file or load-bearing sections):

- `portfolio/signals/__init__.py` (141 B)
- `portfolio/signals/momentum.py` (14 KB)
- `portfolio/signals/mean_reversion.py` (20.5 KB)
- `portfolio/signals/news_event.py` (21.9 KB)
- `portfolio/signals/econ_calendar.py` (10.2 KB)
- `portfolio/signals/cot_positioning.py` (14 KB)
- `portfolio/signals/credit_spread.py` (11.2 KB)
- `portfolio/signals/crypto_macro.py` (9.2 KB)
- `portfolio/signals/metals_cross_asset.py` (17.4 KB)
- `portfolio/signals/statistical_jump_regime.py` (10 KB)
- `portfolio/signals/futures_basis.py` (9.5 KB)
- `portfolio/signals/hurst_regime.py` (11.6 KB)
- `portfolio/signals/shannon_entropy.py` (10.2 KB)
- `portfolio/signals/vix_term_structure.py` (5.9 KB)
- `portfolio/signals/gold_real_yield_paradox.py` (10.4 KB)
- `portfolio/signals/cross_asset_tsmom.py` (7.2 KB)
- `portfolio/signals/copper_gold_ratio.py` (9.5 KB)
- `portfolio/signals/network_momentum.py` (13.1 KB)
- `portfolio/signals/ovx_metals_spillover.py` (7.1 KB)
- `portfolio/signals/xtrend_equity_spillover.py` (8.1 KB)
- `portfolio/signals/complexity_gap_regime.py` (8.8 KB)
- `portfolio/signals/realized_skewness.py` (8.5 KB)
- `portfolio/signals/mahalanobis_turbulence.py` (10.6 KB)
- `portfolio/signals/crypto_evrp.py` (11.8 KB)
- `portfolio/signals/hash_ribbons.py` (10.4 KB)
- `portfolio/signals/drift_regime_gate.py` (6.3 KB)
- `portfolio/signals/vol_ratio_regime.py` (9.3 KB)
- `portfolio/signals/residual_pair_reversion.py` (13.2 KB)
- `portfolio/signals/williams_vix_fix.py` (7.8 KB)
- `portfolio/signals/treasury_risk_rotation.py` (6.7 KB)
- `portfolio/signals/intraday_seasonality.py` (7.1 KB)
- `portfolio/signals/cubic_trend_persistence.py` (4.9 KB)
- `portfolio/signals/vwap_zscore_mr.py` (4.3 KB)
- `portfolio/signals/gold_overnight_bias.py` (6.6 KB)
- `portfolio/signals/metals_vrp.py` (7.6 KB)
- `portfolio/signals/breakeven_inflation_momentum.py` (7.2 KB)
- `portfolio/signals/trend_slope_momentum.py` (5.2 KB)
- `portfolio/signals/ttm_squeeze.py` (8.2 KB)
- `portfolio/signals/finance_llama.py` (9.9 KB)
- `portfolio/signals/cryptotrader_lm.py` (6.8 KB)
- `portfolio/signals/meta_trader.py` (2.0 KB)
- `portfolio/signals/tsi_chop_mr.py` (6.7 KB)
- `portfolio/signals/amihud_illiquidity_regime.py` (3.6 KB)
- `portfolio/signals/absorption_ratio_regime.py` (7.5 KB)
- `portfolio/signals/connors_rsi2.py` (4.8 KB)
- `portfolio/signals/adx_regime_switch.py` (5.6 KB)
- `portfolio/signals/sentiment_extremity_gate.py` (7.5 KB)
- `portfolio/signals/choppiness_regime_gate.py` (4.3 KB)
- `portfolio/signals/autotune_adaptive_cycle.py` (6.2 KB)
- `portfolio/signals/bocpd_regime_switch.py` (7.8 KB)
- `portfolio/signals/dxy_cross_asset.py` (3.1 KB)
- `portfolio/signals/macro_regime.py` (head only; 14 KB total)
- `portfolio/signals/orderbook_flow.py` (head only; 7.2 KB total)
- `portfolio/signals/futures_flow.py` (head only; 11 KB total)
- `portfolio/signals/oscillators.py` (head only; 19 KB total)
- `portfolio/signals/trend.py` (head only; 18.9 KB total)
- `portfolio/signals/forecast.py` (head only; 40.4 KB total)
- `portfolio/signals/momentum_factors.py` (head only; 17.7 KB total)
- `portfolio/signal_engine.py` (sampled — dispatch loop, validator, function map)
- `portfolio/signal_registry.py` (full)
- `portfolio/signal_utils.py` (full)
- `portfolio/data_collector.py` (head — to verify df shape)
- `portfolio/main.py` (sampled — to verify interval/limit passed to generate_signal)

Not opened (skipped per scope — disabled signals with no recent change and
no obvious accuracy gating problem from the registry):
`absorption_ratio_regime` (read in detail above — listed for completeness),
`calendar_seasonal`, `candlestick`, `claude_fundamental`, `fibonacci`,
`heikin_ashi`, `smart_money`, `structure`, `volatility`, `volume_flow`.
