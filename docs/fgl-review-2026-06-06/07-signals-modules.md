# Adversarial review — portfolio/signals/*.py (2026-06-06)

Scope: 79 plugin signal modules. Depth focused on ACTIVE signals (per signal_engine
DISABLED_SIGNALS + _DISABLED_SIGNAL_OVERRIDES). Confirmed-active set reviewed:
momentum, mean_reversion, econ_calendar, cot_positioning, metals_cross_asset,
crypto_macro, crypto_evrp, statistical_jump_regime, adx_regime_switch,
amihud_illiquidity_regime, choppiness_regime_gate, bocpd_regime_switch,
vol_ratio_regime, drift_regime_gate, realized_skewness (XAU override), signal_credibility_filter.
Note: williams_vix_fix and credit_spread overrides were REMOVED 2026-05-31/05-26 — both now
force-HOLD (disabled). mstr_mnav_discount, gold_platinum_ratio_risk, stablecoin_supply_ratio,
gold_btc_vol_spillover are all in DISABLED_SIGNALS (force-HOLD).

## P1

signals/crypto_evrp.py:195-201: (active) `_evrp_level_signal` votes SELL when eVRP>10 and
BUY when eVRP<-10 — the OPPOSITE of this module's own docstring (lines 12-19): "eVRP very
high (IV>>RV)... coinciding with bullish price action (vol compression = uptrend)" and "eVRP
very negative... contrarian BUY opportunities." Code and thesis disagree on direction. The
percentile/momentum sub-signals follow the code's (risk-off) convention, so either the
docstring is stale or the level sub-signal is inverted vs the documented edge. Confirm which
direction was actually validated; if the docstring reflects the backtested edge, the level vote
is wrong-direction. → Reconcile: pick one convention and make level/percentile/momentum +
docstring all agree.

## P2

signals/mstr_mnav_discount.py:175-182: (disabled) `historical_ratios` recomputes mNAV for every
historical MSTR close using the SINGLE current `btc_price` (held constant across all bars). So
`_mnav_velocity` and `_discount_depth_zscore` collapse to pure MSTR-price velocity / z-score —
they do NOT measure mNAV discount dynamics as documented (BTC price variation is erased). Only
`_mnav_level` uses live BTC. Two of three sub-votes mislabel MSTR momentum as mNAV momentum.
→ Build the historical ratio series from a BTC price history aligned to each MSTR bar (fetch BTC
OHLCV, not a scalar), or drop velocity/z-score sub-signals.

signals/realized_skewness.py:228-239: (active, XAU) returns `confidence` straight from
`majority_vote(..., count_hold=False)` with NO 0.7 cap, unlike every sibling signal. A single
firing sub-signal out of 4 (e.g. only skew_zscore=BUY, rest HOLD) yields confidence=1.0 because
majority_vote with count_hold=False ignores HOLD voters in the denominator. → Cap at 0.7
(`confidence = min(confidence, 0.7)`) for consistency, or feed the per-sub confidences instead of
the lone-vote ratio.

signals/signal_credibility_filter.py / shared majority_vote convention: (active, multiple)
`majority_vote(['BUY','HOLD','HOLD'], count_hold=False)` returns ('BUY', 1.0). Across the
regime/composite signals a lone non-HOLD sub-signal produces max confidence pre-cap. Most
signals clamp to 0.7 afterward so impact is bounded, but the pattern means "1 of 3 voters" and
"3 of 3 voters" are indistinguishable below the cap. → Consider denominator = total sub-signals
(count_hold=True) for confidence, or document that count_hold=False is intentional.

signals/choppiness_regime_gate.py:65-80: (active) `_chop_roc` maps the rate-of-change of the
Choppiness Index to a directional BUY/SELL vote (falling CHOP→BUY, rising→SELL). Choppiness has
no inherent price direction; this attaches a directional vote to a non-directional quantity. It
only counts when the gate already passed (non-choppy), limiting damage, but it's a spurious
directional voter. → Make chop_roc a confidence modifier / gate, not a BUY/SELL voter.

signals/crypto_evrp.py:204-220: (active) dead code — `_evrp_percentile_signal` computes `rv_hist`
(lines 216-220) and never uses it; the percentile is taken purely from DVOL, not eVRP, despite
the function name. Harmless but misleading and wastes a rolling computation each call. → Remove
the unused rv_hist block or actually rank eVRP.

signals/amihud_illiquidity_regime.py:76-81 + 107-112: (active) directional mapping is arbitrary:
high illiquidity z (>2)→SELL, low (<-1)→BUY, and volume rvol>1.3→BUY / <0.6→SELL. Illiquidity
itself is non-directional; the asymmetric thresholds (+2 vs -1) plus a volume voter that fires
BUY on any high-volume bar give this a structural BUY bias in liquid uptrends. Not a crash, but
the edge claim (68% 1d, 225 sam) rests on a directional reading of a non-directional regime
metric — re-verify per-ticker before trusting confidence. → Validate sign per asset; consider
making illiq a gate that scales other signals rather than voting.

## P3

signals/mstr_mnav_discount.py:85-97: (disabled) `_mnav_level` has redundant branches —
`ratio <= MNAV_STRONG_BUY` and `ratio <= MNAV_BUY` both return BUY; same for SELL. The
STRONG_* thresholds only matter via the separate +0.15 confidence bump at lines 198-201, so
the level function never distinguishes strong vs normal. Dead distinction. → Collapse branches
or use strong tiers to set a higher per-vote confidence.

signals/statistical_jump_regime.py:53-70: (active) `_classify_vol_regime` builds a full
rolling-percentile-rank via `.rolling(...).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])`
every cycle — O(n·window) Python-level apply. Fine at current bar counts but the slowest path in
this module; consider `expanding`/vectorized percentile if bar history grows.

signals/cot_positioning.py:197-223: (active, metals) `_sub_commercial_change` is named/
documented as a commercial-hedger signal but actually keys off `noncomm_net_change` and negates
it; `comm_net` is read into indicators but never drives the vote. Logic is internally consistent
(contrarian on spec longs) but the naming will mislead future maintainers. → Rename to
`_sub_speculator_change` or wire actual commercial-net delta.

## Risk summary
The active core (momentum, mean_reversion, the regime gates, econ_calendar, cot) is sound on
look-ahead and NaN/division handling, and all external-data fetches in active signals are
cached with timeouts (no 60s-loop blocking risk found). The one item worth resolving before
trusting accuracy numbers is the crypto_evrp level-vote direction (P1) which contradicts its own
documented edge; the remaining findings are confidence-calibration (uncapped/lone-vote=1.0) and
arbitrary directional mappings on non-directional regime metrics that inflate or bias votes
within the 0.7 cap rather than crash the loop.
