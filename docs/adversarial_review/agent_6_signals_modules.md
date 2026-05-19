# Adversarial Review — Signals Modules Subsystem

**Subsystem:** `portfolio/signals/`
**Files enumerated:** 61 (60 signal modules + `__init__.py`)
**Reviewer:** agent_6 — full subsystem adversarial pass
**Scope:** Depth on top-12 highest-impact active signals listed in CLAUDE.md;
lighter on disabled and pending-validation signals.

## Severity counts

| Severity | Count |
|----------|------:|
| P0 (bad vote → bad trade) | 4 |
| P1 (real bug) | 16 |
| P2 (latent) | 14 |
| P3 (minor) | 8 |
| **Total** | **42** |

---

## P0 — bad vote will cause bad trade

### `portfolio/signals/credit_spread.py:285`
P0 (data correctness / silent failure): `cfg = load_json("config.json", default={})` uses a
RELATIVE path. The loop runs from variable CWDs (scheduled task PF-DataLoop launched from
`C:\Windows`). When `_get_fred_key(context)` returns `""`, this fallback silently fails to
find the FRED key — signal then early-returns `empty` (HOLD) on every cycle even though the
key exists at the symlinked `config.json`. Mirrors the SM-P1-4 fix already applied to
`cot_positioning.py` (lines 33, 67, 83) but never propagated here. This signal is in the
active list (54.2% acc per CLAUDE.md) but votes HOLD instead of contributing real direction.

### `portfolio/signals/gold_real_yield_paradox.py:265`
P0 (data correctness): Same relative-path bug — `cfg = load_json("config.json")`. Silent
HOLD instead of using the FRED key from the symlinked config. `forecast.py:340` is the
correct pattern (`Path(__file__).resolve().parent.parent.parent / "config.json"`).

### `portfolio/signals/futures_basis.py:209-212`
P0 (applicability): The signal applies to **any** ticker in
`{**BINANCE_FAPI_MAP, **BINANCE_SPOT_MAP}` — which includes `XAU-USD` and `XAG-USD` because
metals trade on Binance FAPI. Premium index on metals perp futures is far thinner and not
the contango/backwardation signal calibrated in the academic source. Yet the signal votes
metals tickers with no per-asset class restriction. With max_confidence=0.7, a bad metals
vote dominates a tight consensus on XAG (MIN_VOTERS=2).

### `portfolio/signal_engine.py:3405-3410` (on-chain BTC inline voter)
P0 (no confidence + tie handling): The BTC on-chain voter sets `votes["onchain"] = "BUY"|"SELL"`
but never sets a corresponding confidence anywhere. Downstream `_weighted_consensus` either
defaults to 1.0 or drops the entry from the weight calculation. With 4 sub-votes and a tie
(2-2), votes["onchain"] silently stays `"HOLD"` (its default at line 3355) even when 4 of 4
metrics fired — `total >= 2` passes but `buy_count == sell_count` falls through without
emitting any indicator. Combined with the 100% accuracy line in CLAUDE.md, this is the
likely surface where the small-sample tautology is silently being scored — the function
NEVER votes HOLD-when-data-exists, only HOLD-when-tied or HOLD-when-no-data, and those two
indistinguishable cases collapse outcome attribution.

---

## P1 — real bugs

### `portfolio/signals/cot_positioning.py:268` and `:393-396`
P1 (applicability mismatch with docstring): `_sub_real_yield` docstring says
"Sub-indicator 4: Real yield direction (gold-specific)." but the function is called
unconditionally at line 393 for BOTH XAU-USD AND XAG-USD via the iteration. The vote (BUY
on falling yield, SELL on rising) is then mixed into the silver vote pool. Silver does not
share gold's real-yield mechanic — silver is more industrial. The "100% acc on 5 samples"
ticker stat may be poisoned by this asymmetric sub-signal that's only theoretically valid
for gold being applied to silver as well.

### `portfolio/signals/cot_positioning.py:309-396` (overall)
P1 (data resolution offset / tautology suspicion):
COT data publishes weekly (Friday for Tuesday data per docstring). The outcome backfill
correlates the signal at time `signal_ts` against price moves N hours/days later. But because
the COT data only updates weekly, the signal output is IDENTICAL across multiple consecutive
cycle invocations for ~7 days. Each invocation creates a separate `signal_log.jsonl` entry
with the same vote — outcome backfill counts each entry separately. So one weekly-correct
COT vote becomes N=hundreds-of-cycles "correct" entries. This is the most likely explanation
for the suspicious "100% accuracy on 5 samples" CLAUDE.md notes — the outcome lookup uses
signal_ts as key but COT data doesn't change between cycles. Needs deduplication by `report_date`.

### `portfolio/signals/news_event.py:47-50` and `:103-104`
P1 (data loss): `_HEADLINES_PATH` is a SINGLE file `data/headlines_latest.json` (no
`{ticker}` suffix). 8 worker threads race to persist headlines for different tickers.
The P1.10 lock added at line 103 prevents file corruption but NOT data loss — the last
writer wins. The fish monitor consuming `headlines_latest.json` therefore only sees
whichever ticker happened to write last on the previous cycle. If "BTC" is last, XAG headlines
silently vanish.

### `portfolio/signals/news_event.py:280-303` `_sentiment_shift`
P1 (asymmetric sentiment classification): default "cut" → bearish (line 297-303). But the
`_POSITIVE_KEYWORDS` only contains generic uplift words. Headlines like "Fed cuts rates and
stocks rally" hit `has_bearish_cut=False, has_positive=True ("rally")` → POS. But "rate cut"
itself is in `_BULLISH_CUT_PHRASES` — duplicate path. Headlines with both bullish AND bearish
phrases produce non-deterministic vote depending on order. Subtle but pervasive.

### `portfolio/signals/finance_llama.py:204-214`, `cryptotrader_lm.py:150-158`
P1 (confidence masquerade): When `_parse_response` returns `decision="HOLD"` (because the
regex fallback found no BUY/SELL token) AND `confidence=None`, the code defaults confidence
to 0.50. The composite returns `action="HOLD", confidence=0.50`. Per
`_validate_signal_result` at `signal_engine.py:1571` confidence is capped to `max_confidence`
(0.7), so 0.50 survives. Downstream consumers receive HOLD with non-zero confidence — but a
genuine abstention should be `confidence=0.0`. The `_abstain` helper handles failure paths
correctly (conf=0), but the `_parse_response` non-result path leaks through. The comment
acknowledges this is intentional for "BUY/SELL argmax" preservation, but doesn't catch the
HOLD fallback case where there is no argmax to preserve.

### `portfolio/signals/statistical_jump_regime.py:204`
P1 (false SELL bias in low-vol regime): `vol_vote = "BUY" if slope > 0 else "SELL" if slope < 0 else "HOLD"`
votes SELL on ANY negative SMA slope in low-vol regime, without a magnitude threshold. Sub-4
trend_confirm at line 217-220 correctly uses 0.005 threshold around zero. Sub-3 (vol_vote)
inconsistently has no deadband. In persistent low-vol environments this fires SELL on every
tiny pullback.

### `portfolio/signals/statistical_jump_regime.py:171`
P1 (numerical instability): `log_returns = np.log(close / close.shift(1)).dropna()` —
division by zero (or near-zero close, possible on stale/illiquid feeds) yields ±inf, and
`log(inf)=inf`. `dropna()` does NOT remove infs. Subsequent rolling std propagates the inf,
breaking jump detection. Should be `replace([inf, -inf], nan).dropna()`. Same pattern in
`cubic_trend_persistence.py:85`, `residual_pair_reversion.py:290-291`.

### `portfolio/signals/vwap_zscore_mr.py:74-79`
P1 (subsignal contradicts thesis): `vwap_slope` votes BUY when slope > 0.1% — a rising
VWAP. But the master signal is mean-reversion: when price is way above VWAP (sub-1 votes
SELL), VWAP is rising (sub-2 votes BUY). Sub-1 and sub-2 then cancel under `majority_vote`,
producing HOLD. This makes the signal rarely fire net direction in trending markets where
the MR thesis applies most. Three-vote tie collapses too often.

### `portfolio/signals/vwap_zscore_mr.py:124-125`
P1 (silent except returning HOLD): `except Exception: return HOLD/0.0`. Catches ALL
exceptions including programming bugs (KeyError on missing column, AttributeError on None).
Per user criteria #4: "Silent except: pass that returns HOLD" — masquerades real failures
as legitimate HOLD votes. Should at minimum log.

### `portfolio/signals/forecast.py:797`
P1 (data fallback in core signal): When `candles` fetch fails AND `df` is provided,
`close_prices = df["close"].values.tolist()` is used. But `df` is the per-cycle OHLCV
DataFrame at the loop's primary timeframe; `candles` are 1h candles loaded for forecast.
If df is at 5m or 1m granularity, Chronos receives 50 minute-bars and predicts a 1-hour-ahead
move that doesn't match its training distribution. Type/timeframe mismatch silently produces
junk predictions for tickers where candle fetch transiently fails.

### `portfolio/signals/residual_pair_reversion.py:281-284`
P1 (timeframe-alignment silent loss): `target_close` likely has hourly/intraday DatetimeIndex
while `driver_close` from yfinance (`_fetch_driver_closes`) is DAILY. `pd.DataFrame({"target":..., "driver":...}).dropna()` aligns on index — only keeps rows where BOTH have data. Result is
near-empty (just daily timestamps that happen to match intraday bar starts, if any). Combined
with `MIN_ROWS=200`, the signal might silently always return empty when intraday driver data
is requested. The `.intersection(common_idx)` at line 294 compounds the issue. Effective
behavior: signal works for tickers with daily input bars only.

### `portfolio/signals/network_momentum.py:81-83`
P1 (timeframe mismatch): `yf.download(tickers, period="4mo", progress=False)` returns DAILY
data, but the signal is called every 60s for tickers with intraday cadence. Daily peer
returns vs hourly own returns produce meaningless correlations and momentum spillover. No
timeframe-mismatch guard.

### `portfolio/signals/trend_slope_momentum.py:99-100`
P1 (no zero guard + bfill artifact): `past_price = float(raw_close.iloc[-MOMENTUM_LOOKBACK])`
followed by `mom_ratio = current_price / past_price`. No protection against `past_price == 0`.
And `df["close"].ffill().bfill()` at line 98 means a backfilled price can be used as the
"past", silently injecting a forward-looking value as historical reference.

### `portfolio/signals/trend_slope_momentum.py:142-150`
P1 (vote double-counting): Sub-4 `slope_momentum_agreement` votes BUY when `current_slope > 0 AND momentum > 0.5`. But sub-1 (`trend_slope`) ALSO votes BUY when `z_clipped > 0.5` (≈ slope positive) and sub-2 (`momentum_50d`) ALSO votes BUY when `momentum > 0.5`. So sub-4 votes BUY iff
sub-1 and sub-2 both vote BUY. Three of four sub-signals collapse to the same condition,
inflating confidence by counting the same evidence twice.

### `portfolio/signals/cubic_trend_persistence.py:117 + 121` (`_detect_timeframe`)
P1 (timeframe misclassification): `if median_hours >= 20 → daily else hourly`. A signal
running on 4h or 2h candles is classified as "hourly" — but the parameters `B_HOURLY`,
`C_HOURLY` are calibrated to 1h cadence per the academic source. 4h cadence with hourly
parameters underestimates phi by 4×, suppressing the signal when it should fire.

### `portfolio/signals/copper_gold_ratio.py:251-252`
P1 (hardcoded inversion vs CLAUDE.md rule): `if is_metals: action = SELL if BUY else BUY`.
Per `.claude/rules/signals.md`: "NEVER invert sub-50% signals — gate them as HOLD instead.
Inversion causes whiplash." This signal hardcodes a theory-based inversion for metals. If
the un-inverted theory is correct at 55% for risk assets but the inversion mapping is wrong
for metals (e.g., gold sometimes leads copper, not the other way around), inverted accuracy
becomes 45% and the signal pollutes consensus. No per-direction accuracy validation.

### `portfolio/signals/treasury_risk_rotation.py:185`
P1 (same hardcoded inversion concern): `_invert(action)` for safe-haven tickers. Same risk
pattern as copper_gold_ratio above. Plus: ticker is checked AT line 182, AFTER `votes` are
computed — if `context` is None or missing `ticker`, `is_safe_haven=False`, action stays
risk-on-direction. Metals tickers without a context get the OPPOSITE direction silently.

### `portfolio/signals/metals_cross_asset.py:335-340`
P1 (silver direction asymmetry without validation): GVZ (GOLD volatility index) is treated
as opposite-direction for silver: "High GVZ = fear → BUY gold, SELL silver." But silver is
higher-beta than gold — when gold fear rises, silver typically rises MORE. The hardcoded
inversion `if not is_silver else SELL` is unjustified. Should be uniform BUY for both metals,
or per-asset calibrated.

---

## P2 — latent issues

### `portfolio/signals/breakeven_inflation_momentum.py:53-63`
P2 (inconsistent fallback): Unlike `credit_spread.py` and `gold_real_yield_paradox.py`, this
signal does NOT fall back to `load_json("config.json")` when context lacks the FRED key. So
when the context-passed key is missing but the file has it, signal silently HOLDs while
its siblings recover the key. Inconsistent across the FRED-using cluster.

### `portfolio/signals/breakeven_inflation_momentum.py:184`
P2 (silent lookback truncation): `bei_values[min(_BEI_CHANGE_LOOKBACK, len(bei_values) - 1)]`
— if `len(bei_values) <= _BEI_CHANGE_LOOKBACK`, the `min` uses the OLDEST element. "20-day
change" silently becomes "full-history change" without warning. The earlier `if not bei_values
or len(...) < _BEI_CHANGE_LOOKBACK + _BEI_Z_WINDOW + 1` gate (line 176) covers the worst
case but not all sub-cases.

### `portfolio/signals/credit_spread.py:274`
P2 (MSTR in risk_assets is debatable): MSTR is mostly a BTC proxy on equity venue. FRED
HY OAS is a daily fundamental measure; using it to vote intraday MSTR (which has 60s
cycles) creates a daily-data → intraday-vote mismatch. Should at least mark vote staleness.

### `portfolio/signals/forecast.py:781-794`
P2 (cache key collision risk): Two different cache keys for the same ticker:
`forecast_candles_{ticker}` (1h) and `forecast_candles_{ticker}_{kronos_interval}`. If
kronos_interval is "1h" too (default), keys differ but data is the same — cache duplication.
If interval is e.g. "5m" only when `_KRONOS_ENABLED` is True, but flag toggles between
cycles, stale cached 1h data may be used for Kronos.

### `portfolio/signals/hash_ribbons.py:248`
P2 (empty-ticker bypass): `if ticker and ticker.upper() not in _BTC_TICKERS: return hold` —
only filters when `ticker` is truthy. If `context` is missing or `ticker == ""`, the BTC-only
filter is silently bypassed; the function proceeds and computes BTC hashrate-based vote that
gets returned as if applicable to the (unknown) caller ticker.

### `portfolio/signals/hash_ribbons.py:282`
P2 (hardcoded confidence): Returns `confidence = 0.7` as a fixed value when all conditions
align — no scaling by signal strength (e.g., days since crossover, hash ratio magnitude).
Two genuine recoveries 10 years apart with very different hashrate dynamics produce identical
confidence.

### `portfolio/signals/vix_term_structure.py:119-122` and `:165-166`
P2 (sub-signal collinearity): Sub-3 zscore with `_Z_THRESHOLD = 0.0` votes SELL on any
positive deviation, BUY on any negative. Sub-1 backwardation_flag votes SELL when ratio ≥ 1.0
which is ~always when z > 0 above the long-run mean. Sub-1 and sub-3 effectively duplicate.
Sub-2 contango_depth duplicates sub-1 for BUY side (depth > 0.10 ⟺ ratio < 0.90). The 4-vote
composite is really 2 independent votes — but `majority_vote` treats them as 4. Inflated
confidence under unanimous noise.

### `portfolio/signals/vix_term_structure.py:141`
P2 (no ticker applicability gate): Despite docstring acknowledging "Weaker on BTC/equities
(~40-50%)," the signal computes for all callers without ticker check. Per-ticker accuracy
gate eventually disables but for the first 30 samples the signal pollutes consensus.

### `portfolio/signals/amihud_illiquidity_regime.py:107-112`
P2 (RVOL inconsistent with ILLIQ thesis): The signal is ILLIQ-based but volume_confirm
sub-3 votes BUY/SELL based on raw RVOL alone, with no interaction with the ILLIQ regime
detected in sub-1/sub-2. High RVOL + high ILLIQ (sub-1 SELL) creates contradictory votes
that cancel — defeating the signal's purpose.

### `portfolio/signals/amihud_illiquidity_regime.py:76-81`
P2 (asymmetric thresholds): `cur_z > 2.0 → SELL`, `cur_z < -1.0 → BUY`. z=2 fires ~2%
of the time; z=-1 fires ~17%. No comment justifies asymmetric calibration. BUY pivots fire
~8x more than SELL pivots — structural BUY bias on a signal designed to be symmetric.

### `portfolio/signals/intraday_seasonality.py:114-118` and `:121-129`
P2 (never-SELL sub-signals): `_hour_alpha_vote` returns only "BUY" or "HOLD"; `_dow_vote`
likewise. The only path to SELL is through the trend_context sub-signal. So during low-alpha
hours, the signal CANNOT vote SELL even if trend says SELL (gated to HOLD via combined_mult
check at line 188). Asymmetric SELL exposure embedded in the gate logic.

### `portfolio/signals/intraday_seasonality.py:194`
P2 (no-op multiplier): `confidence = min(base_conf * (combined_mult / 1.0), 0.7)`. The `/ 1.0`
is a no-op. Code smell — original intent was likely `/ some_normalization` but normalization
constant was removed.

### `portfolio/signals/gold_overnight_bias.py:118-140` (`_fix_proximity_vote`)
P2 (never-SELL sub-signal): The proximity vote always returns BUY or HOLD, never SELL. Even
when 90 min before the PM fix (mid-London-PM, SELL session per sub-1), proximity votes BUY.
Sub-1 SELL + sub-3 BUY = wash; signal rarely fires SELL during the documented London-PM
underperformance window.

### `portfolio/signals/vol_ratio_regime.py:181-184`
P2 (no deadband around SMA): `current_price < sma_20_val → BUY` in ranging regime fires 50%
of the time on random bars near the SMA. No threshold around the mean. Pure noise contribution.

### `portfolio/signals/ttm_squeeze.py:197`
P2 (useless ternary): `"squeeze_state": "HOLD" if currently_squeezing else "HOLD"` — both
branches return "HOLD". Either intentional placeholder or copy-paste artifact. Also at
line 198-199, sub_signals report `momentum_direction` and `momentum_acceleration` votes
that may be BUY/SELL while the composite action is forced HOLD at line 194 — logs show
sub-votes that don't match the composite, confusing operators.

### `portfolio/signals/ttm_squeeze.py:111-115`
P2 (lookahead-bias smell): `dev_shifted = close.shift(1) - avg_mid.shift(1)` then `mom_prev`
uses linreg over the shifted series. The `dropna()` at line 113 drops the first NaN entry,
shifting the indices in `mom_prev` so it's NOT comparable bar-for-bar with `mom_current`.
mom_delta has a subtle 1-bar misalignment. Not strictly lookahead but the comparison is off.

### `portfolio/signals/orderbook_flow.py:142-147`
P2 (sub-3 piggyback on sub-2): VPIN sub-3 votes the SAME direction as TIR (sub-2 input).
When VPIN is high and trade flow imbalance is positive, both sub-2 and sub-3 vote BUY. The
4-sub-signal composite effectively has 3 independent votes — double-counting toxic flow.

### `portfolio/signals/orderbook_flow.py:175-176`
P2 (permanent HOLD vote in pool): Sub-5 spread_health always votes HOLD by design but is
INCLUDED in `votes`. Since `majority_vote(count_hold=False)` uses `active = buy + sell` as
denominator, the HOLD vote doesn't affect majority — but the comment says it does via a
0.3x penalty applied somewhere. Sub-signal that never votes should not be appended to votes.

### `portfolio/signals/crypto_macro.py:228 + 281`
P2 (variable forward-reference smell): `OPTIONS_TTL` referenced at line 228 inside function,
defined at line 281 (module-level after the function). Python resolves at call time so this
works at runtime, but is fragile — any future test that calls during module loading would
NameError. Refactor to define before use.

### `portfolio/signals/copper_gold_ratio.py:206-215`
P2 (duplicate threshold branches): `if zscore < -2.0: SELL elif zscore < -1.5: SELL`. Both
branches produce identical output — the `-2.0` branch is dead code. Same for the BUY side
lines 210-213. Code suggests intent to differentiate "strong" vs "moderate" but they were
collapsed to the same vote.

### `portfolio/signal_engine.py:3393-3400` (on-chain BTC, line `netflow == 0`)
P2 (HOLD inflates sub_votes denominator): When `netflow == 0`, `sub_votes.append("HOLD")`.
The downstream `total = buy_count + sell_count` is unaffected — good. But the
`extra_info["onchain_sub_votes"]` reports `"2B/1S"` which doesn't reveal the HOLD count, so
the operator can't tell whether 0 or 2 sub-metrics returned None/zero. Observability gap.

### `portfolio/signals/intraday_seasonality.py:81-92` (`_get_utc_hour_and_dow`)
P2 (silent timezone error fallback): If `last_ts` has no tzinfo, the code uses it as if it
were UTC (line 88). For tickers fed with non-UTC bars, returns wrong hour. The hour-of-day
multipliers are then misaligned with actual UTC time.

### `portfolio/signals/futures_basis.py:71-73`
P2 (docstring vs constant mismatch): Docstring at lines 71-72 says "z < -1.8" but constants
at lines 38-39 are `±1.5`. Indicator of calibration uncertainty / spec drift.

---

## P3 — minor / style

### `portfolio/signals/finance_llama.py:135` and `cryptotrader_lm.py:81`
P3 (defensive duplication): The `_abstain` reason-string contract is duplicated verbatim
across `meta_trader.py`, `finance_llama.py`, `cryptotrader_lm.py`. Should be extracted to
a shared helper to ensure schema parity.

### `portfolio/signals/breakeven_inflation_momentum.py:53-63`
P3 (copy-paste): `_get_fred_key` function is duplicated VERBATIM across `credit_spread.py:125`,
`metals_cross_asset.py:91`, `gold_real_yield_paradox.py:43`, `breakeven_inflation_momentum.py:53`.
Refactor into `portfolio/fred_utils.py` or similar.

### `portfolio/signals/momentum.py:294`
P3 (no `context` parameter despite signal_engine convention): `compute_momentum_signal(df: pd.DataFrame)`
lacks a `context` parameter. Other plugins (mean_reversion, momentum_factors) take it.
`signal_registry.py` line 99 doesn't declare requires_context — works, but inconsistent
with sibling signals. Future extensions need refactoring.

### `portfolio/signals/heikin_ashi.py:530-550`
P3 (default indicator non-NaN values): `default_result["indicators"]["ha_color"] = "green"`
on insufficient-data path. Should be `None` or `"unknown"` — claiming a default color is
misleading when there's no data.

### `portfolio/signals/momentum_factors.py:317`
P3 (`_apply_seasonality` only "metals" eligible via `hasattr(df.index, 'hour')`): No
explicit metals ticker check, relies on profile being None for non-metals. Works but
brittle.

### `portfolio/signals/forecast.py:776-794`
P3 (config override at signal level): `set_chronos_model(chronos_model)` mutates a
global at signal-compute time. Concurrent calls for different tickers with different
chronos_model values race. Today not exploited but a footgun.

### `portfolio/signals/news_event.py:611-612`
P3 (conditional vote inclusion): `thesis_alignment` is appended to votes only when
`thesis_ind.get("enabled") and thesis_action != "HOLD"` — but `enabled` is in `indicators`
not `sub_signals`. The composite confidence then varies based on the number of "active"
sub-signals, which is misleading: same headlines + same beliefs produce one confidence
when prophecy.news_alignment=true, different when false. Hidden config dependency in
confidence reporting.

### `portfolio/signals/__init__.py`
P3 (file is empty / placeholder): Only 141 bytes. Could re-export the signal compute
functions for cleaner imports, or at least mark it as namespace-package intentionally.

---

## Cross-cutting observations (not file:line — system-level)

### Composite confidence inflation under sparse activation
`signal_utils.majority_vote(count_hold=False)` computes `confidence = winner / active_voters`.
With 5 sub-signals where 1 fires BUY and 4 are HOLD, confidence = 1/1 = 1.0 (capped to
max_confidence). This treats "1 sub-signal fired" identically to "5 sub-signals unanimous."
Affects nearly every composite — momentum (8 subs), mean_reversion (8), momentum_factors (7),
news_event (7), econ_calendar (5), credit_spread (4), metals_cross_asset (8), cot_positioning (4),
forecast (4), futures_basis (4), hash_ribbons (3), residual_pair_reversion (3),
treasury_risk_rotation (4), vix_term_structure (4), williams_vix_fix (4), copper_gold_ratio (4),
metals_vrp (~3), cubic_trend_persistence (3), trend_slope_momentum (4), gold_real_yield_paradox (3),
breakeven_inflation_momentum (3), crypto_evrp (3), shannon_entropy (~4), vol_ratio_regime (2-3),
network_momentum (3), realized_skewness (4), mahalanobis_turbulence (~4), drift_regime_gate (~3),
ovx_metals_spillover (4), statistical_jump_regime (3), hurst_regime (4), amihud_illiquidity_regime (3),
ttm_squeeze (3), vwap_zscore_mr (3), intraday_seasonality (3), gold_overnight_bias (3),
trend (7), oscillators (8), heikin_ashi (7), candlestick (~), structure (~), smart_money (~),
volume_flow (~), volatility (~), claude_fundamental (~), futures_flow (~).

**This is the single most important fix point.** Use `count_hold=True` for the denominator
or apply an activation-fraction penalty.

### Disabled signal verification (DISABLED_SIGNALS in `tickers.py:65-215`)
Spot-checked — `signal_engine.py:3592` correctly force-HOLDs disabled signals before
invocation. No regressions found. The shadow-safe whitelist at `_SHADOW_SAFE_SIGNALS` and
the per-ticker override at `_DISABLED_SIGNAL_OVERRIDES` add complexity but the gate is
intact.

### News event P1.10 fix verified
`_headlines_lock` is present at `news_event.py:52`, used at line 103. No regression.
(But see P1 finding above about single-file overwrite.)

### Schema drift check
`_validate_signal_result` at `signal_engine.py:1547-1582` is robust:
- action defaulted to HOLD on invalid
- confidence forced to float, clamped to [0, max_confidence]
- sub_signals forced to dict
- non-finite confidence → 0.0
No signal-side bypass found. The cap to `max_confidence` is consistently applied.

### Live prices requirement
Most signals correctly fetch live data. The 4-hour FRED caches (`credit_spread`,
`metals_cross_asset`, `gold_real_yield_paradox`, `breakeven_inflation_momentum`) are
acceptable given FRED's daily cadence. The 1-hour yfinance cache (`network_momentum`,
`residual_pair_reversion`, `treasury_risk_rotation`) is a separate concern: combined with
the daily-data-into-intraday-signal pattern (P1 findings above), the staleness multiplies.

### LLM signal output parsing robustness
`finance_llama.py` and `cryptotrader_lm.py` reuse `ministral_trader._parse_response`. The
fallback chain (JSON → regex action → regex confidence → confidence=0.50) is documented and
intentional. The masquerade issue (P1 finding) is the only real concern. Model timeout
fallback is via `query_llama_server` returning None → `_abstain("server_unavailable")` —
correct.

---

## Top 5 priority fixes

1. **`credit_spread.py:285` + `gold_real_yield_paradox.py:265`** — fix relative `config.json`
   path. Two active signals silently HOLD when run from scheduled task CWD. (P0 × 2)
2. **`signal_engine.py:3405-3410`** — on-chain BTC voter set confidence + tie/HOLD logic.
   Likely root cause of "100% accuracy on 5 samples" tautology in CLAUDE.md. (P0)
3. **`futures_basis.py:209-212`** — restrict to crypto-only tickers. Currently votes on
   metals with garbage premium-index data. (P0)
4. **`signal_utils.majority_vote`** — confidence-inflation fix system-wide. Affects 30+
   composite signals. (cross-cutting)
5. **`cot_positioning.py` outcome resolution** — verify backfill dedupes weekly COT entries
   by `report_date`, not by `signal_ts`. The suspicious 100% accuracy CLAUDE.md note points
   directly at this. (P1, possibly P0 depending on outcome tracker impl)
