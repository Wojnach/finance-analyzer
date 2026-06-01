# FGL Review — signals-modules

Scope: 12 signal modules under `portfolio/signals/` (mean_reversion, momentum,
momentum_factors, news_event, oscillators, metals_cross_asset, cot_positioning,
macro_regime, residual_pair_reversion, calendar_seasonal, volatility,
candlestick). Reviewed against the engine contract: `compute_*_signal(df, …)
-> {action, confidence, sub_signals, indicators}`, where `df` is the **"Now"
timeframe = 15m candles, limit=100, positional RangeIndex with a separate
`time` column** (`portfolio/data_collector.py:45`, `portfolio/main.py:498-514`).
The engine validates/clamps every result (`signal_engine.py:1658`
`_validate_signal_result` — confidence forced finite and clamped to
`[0, max_confidence]`, invalid action → HOLD), which downgrades several
otherwise-scary issues (out-of-range confidence, NaN confidence) below P1.

Verdict: directions are correct across all 12 modules (no vote inversion
found). One module — `residual_pair_reversion` — is effectively dead in
production (always HOLD) due to an index/timeframe mismatch. Remaining findings
are intra-bar instability and minor consistency/maintainability items.

---

## Critical (90-100)

_None._ No systematic vote inversion and no pipeline-killing crash found — each
sub-signal is wrapped in per-signal try/except in the engine dispatch, and bad
returns are clamped by `_validate_signal_result`.

---

## Important (P1 / 80-89)

- **[P1] residual_pair_reversion.py:41 + 271-296** — The signal is **dead on
  its only production path: it always returns HOLD.** Two independent causes,
  both verified empirically:
  1. **Row-count gate vs. the timeframe it is fed.** `MIN_ROWS = 200`
     (line 41) and `compute_…` returns the empty HOLD when `len(df) < 200`
     (line 260-261). But the engine passes `now_df` = the "Now" 15m frame with
     **limit=100** rows (`main.py:503`, `data_collector.py:45`). 100 < 200 →
     immediate HOLD every cycle.
  2. **Target/driver index mismatch wipes the join even if rows were enough.**
     The target `df` carries a **positional RangeIndex** (Binance klines build a
     `time` *column* then `reset_index(drop=True)`, `data_collector.py:96,249`).
     Line 277-278 does `target_close.index = pd.to_datetime(df.index)` — which
     converts integers `0..99` into **1970-epoch-nanosecond** timestamps. The
     driver (`_fetch_driver_closes`, line 81) is fetched as **daily** bars with
     real 2026 `DatetimeIndex`. `aligned = pd.DataFrame({...}).dropna()`
     (line 281-284) therefore intersects 1970-ns timestamps with 2026 daily
     dates → **0 surviving rows** (reproduced: aligned rows = 0). Even the
     correct dates wouldn't help: a 15m intraday grid can't be inner-joined onto
     a daily grid.

     Causal chain: registry intends this signal to specifically rescue ETH-USD
     and XAG-USD BUY/SELL accuracy (`signal_registry.py:228-235`), but it can
     never emit a directional vote → it contributes nothing while appearing
     "registered and active." No test exercises the non-HOLD path
     (`tests/test_signal_residual_pair_reversion.py` never mocks
     `_fetch_driver_closes` and `_make_df` uses a RangeIndex), so the defect is
     invisible to the suite.

     **Fix:** (a) Drive alignment off real timestamps, not the positional index
     — use the `time` column: `target_close.index = pd.to_datetime(df["time"])`.
     (b) Resample the daily driver and the intraday target to a common cadence
     (e.g. fetch the driver at the same interval as `df`, or resample target to
     daily) before the join. (c) Lower `MIN_ROWS`/`_OLS_WINDOW` to fit the ~100
     bars actually delivered, OR have the engine pass a longer-history frame to
     this signal. Add a regression test that mocks the driver with an aligned
     DatetimeIndex and asserts a directional vote is produced.

---

## Minor / Edge cases (P2) and Maintainability (P3)

- **[P2] calendar_seasonal.py:73-108 (`_turnaround_tuesday`)** — On the 15m
  "Now" frame the "prior bar" is the candle 15 minutes ago, not Monday. The
  function reads `df["close"].iloc[-2]`/`df["open"].iloc[-2]` and treats a red
  *15-minute* candle as "red Monday," voting BUY on any Tuesday after a red
  15m bar. The intended Monday→Tuesday reversal semantics don't hold on
  intraday data. Low impact (signal capped at 0.6, in the disabled "Calendar"
  set, and the day-of-week sub-signal is HOLD on Tuesdays anyway).
  **Fix:** resample to daily before the prior-bar test, or gate this sub-signal
  to daily-cadence frames; on intraday, vote HOLD.

- **[P2] candlestick.py:95-132** — Patterns are computed on the last 3 bars
  including the **still-forming current bar** (`c2 = tail.iloc[-1]`). Hammer/
  engulfing/star shapes flip as the live candle forms, so the same bar can vote
  BUY then HOLD then SELL within one 15-minute window. This is the classic
  intra-bar instability rather than future leakage. Tolerated by codebase
  convention (engine notes "candlestick strong path abstains ~87%",
  `signal_engine.py:343`) but worth a HOLD-until-bar-close guard for stability.
  **Fix:** evaluate patterns on `iloc[:-1]` (last *closed* bar), or require the
  current bar to be within N seconds of close before voting.

- **[P2] cot_positioning.py:158-265** — Three of the four sub-signals
  (`_sub_cot_index`, `_sub_commercial_change`, `_sub_managed_money`) are all
  derived from speculator net positioning (`noncomm_net` / `noncomm_net_change`
  / `managed_money_net`, which co-move). When specs are extremely long, all
  three fire SELL from effectively one independent observation, producing
  3/3 = 1.0 → 0.7 (capped) confidence. Inflated confidence under realistic
  inputs. Directions are individually correct (contrarian-on-specs). Mitigated
  by the engine's correlation-group gate, but the within-module confidence is
  overstated. **Fix:** treat the speculator-derived trio as one weighted vote,
  or down-weight 2 of the 3.

- **[P3] cot_positioning.py:197-223 (`_sub_commercial_change`)** —
  Misleadingly named: documented as "Commercial hedger net change" but actually
  reads the **non-commercial** `noncomm_net_change` and inverts it. The logic
  is internally correct (specs adding longs → contrarian SELL), but `comm_net`
  is fetched (line 206) and never used for the vote, and `comm_net_change` is
  reported as `-change`. Rename/clarify to avoid a future "fix" reintroducing a
  sign error.

- **[P3] volatility.py:160 vs 264** — Annualization factor is inconsistent
  between sub-signals: `_historical_volatility` uses `np.sqrt(365)` (line 160)
  while `_garch_signal` uses `np.sqrt(252)` (lines 264, 268). Does **not**
  change any vote (HV compares current vs prev — same scalar cancels; GARCH uses
  a garch/realized ratio — both ×√252 cancel), so it only affects the displayed
  `hist_vol`/`garch_vol` indicator magnitudes. **Fix:** pick one factor
  (365 for 24/7 crypto/metals) for consistency in reported indicators.

- **[P3] volatility.py:233-285 (`_garch_signal`)** — Vol-regime direction is
  price-agnostic (expanding vol → SELL, compressing vol → BUY) while
  `_atr_expansion` and `_historical_volatility` gate the same expansion by price
  direction. On a strong up-move with rising vol, GARCH=SELL contradicts
  ATR=BUY within the same composite. Defensible as a de-risk bias, but the
  internal disagreement quietly suppresses the module to HOLD in exactly the
  trending-vol regimes it claims to detect. Worth a deliberate decision, not a
  bug per se.

---

## Per-module direction check

| Module | Intended direction (derived) | Code matches? | Notes |
|---|---|---|---|
| mean_reversion | RSI2/3 low→BUY, IBS<0.2→BUY, 3+ down→BUY, gap-up-filling→SELL, %B<0→BUY/%B>1→SELL, OU z<−1.5→BUY | Yes | Gap-widening guard explicit; OU θ<0 required. Divzero guarded. |
| momentum | Stoch X-up oversold→BUY, StochRSI<0.2→BUY, CCI<−100→BUY, **Williams %R<−80→BUY** (neg-scale handled), ROC accel→BUY, PPO X-up→BUY, Bull/Bear>0→BUY | Yes | Williams %R sign correct (no inversion). divzero via replace(0,nan). |
| momentum_factors | TSMOM>0→BUY, vol-scaled ROC z>1.5→BUY, high-proximity≥0.95→BUY, low-reversal+3green→BUY, 4+ green→BUY, accel→BUY, vol+price→BUY | Yes | Proportional lookback indices valid; pct_change divzero guarded. |
| news_event | negative/critical→SELL, positive→BUY, dissemination amplifies dominant | Yes | "cut" phrase whitelist/blacklist correct; confidence capped 0.7. Sub-signals share keyword heuristic (correlated). |
| oscillators | AO/TRIX/KST/PPO X-up→BUY, Aroon>50→BUY, Vortex VI+>VI−→BUY, CMO>50→BUY, STC<25→BUY/>75→SELL, Coppock up-from-below-0→BUY | Yes | `.dropna().iloc[-2]` consistent with non-NaN iloc[-1]; divzero guarded. |
| metals_cross_asset | copper↑→BUY, GVZ high→BUY gold/SELL silver, G/S z high→BUY silver, G/S vel falling→BUY silver, SPY↑→BUY silver, oil↑→BUY, EPU high→BUY, TIPS falling→BUY | Yes | Gold vs silver deliberately differentiated; z-score not lookahead. Keys verified against metals_cross_assets.py. |
| cot_positioning | COT idx>80→SELL/<20→BUY (contrarian), specs adding→SELL, MM z high→SELL, real yield falling→BUY | Yes | Live/historical key names verified. P2 correlation + P3 naming. |
| macro_regime | price>SMA→BUY, strong DXY→SELL, curve inverted→SELL, 10Y↑→SELL, ≤2d to FOMC→SELL, golden cross→BUY | Yes | Adaptive SMA period=n is full-sample mean, not lookahead. |
| residual_pair_reversion | residual z<−2→BUY, z>2→SELL, beta unstable/HL slow→HOLD | Direction OK; **never reached** | P1: always HOLD (row-count + index/timeframe mismatch). |
| calendar_seasonal | Mon→SELL/Fri→BUY, month-end→BUY, Jan→BUY/Dec→SELL, pre-FOMC→BUY, Santa→BUY | Yes | Calendar from current bar's date = correct. P2: turnaround-Tuesday prior-bar wrong on intraday. |
| volatility | squeeze-release+price>upper→BUY, BB/Keltner/Donchian breakout up→BUY, ATR/HV expansion gated by price | Yes | Donchian confirms vs prior-bar channel (no lookahead). P3: GARCH price-agnostic; annualization mismatch. |
| candlestick | hammer/inv-hammer after downtrend→BUY, hanging-man/shooting-star after uptrend→SELL, bull engulf→BUY, doji+trend→reverse, morning star→BUY/evening star→SELL | Yes | P2: evaluated on the still-forming current bar (intra-bar instability). |

---

## Summary

No vote-direction inversions and no division-by-zero / NaN-mis-vote crashes
survive into a wrong live vote — every module guards denominators
(`replace(0, np.nan)`, explicit `== 0` checks), returns HOLD on short series,
and the engine clamps confidence and validates action at the boundary.

The one material finding is **P1: `residual_pair_reversion` always returns
HOLD** because (a) `MIN_ROWS=200` exceeds the 100-row "Now" frame the engine
feeds it, and (b) it aligns a positional-RangeIndex intraday target against a
daily-DatetimeIndex driver, which `.dropna()` reduces to zero rows
(reproduced). The signal is registered and counted as active but contributes
nothing — silently misleading on the very tickers (ETH, XAG) it was added to
help. No test covers its directional path.

The remaining items are intra-bar instability on the two pattern/calendar
sub-signals that read the live bar (P2, tolerated by convention but worth a
bar-close guard), a confidence-inflation correlation among the three
speculator-derived COT sub-signals (P2), and consistency/naming cleanups (P3).
Recommend fixing the residual-pair alignment + row-count gate (or excluding it
from "active" counts until fixed) before relying on it.
