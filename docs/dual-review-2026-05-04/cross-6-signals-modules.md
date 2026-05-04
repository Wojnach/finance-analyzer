# Cross-critique — signals-modules

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `calendar_seasonal.py:157-163` — When `month == 1`, `_sell_in_may()` votes BUY here AND `_january_effect()` votes another BUY at lines 183-184. Composite needs only 2 aligned votes → **standing January-long BUY bias on every ticker at 0.6 confidence**. | Claude flagged the asset-class issue (P1 — calendar applied to crypto/metals) but missed this within-stocks systematic bias. The double-vote is subtle until you trace what actually fires in January. |
| `calendar_seasonal.py:222-223` — Pre-holiday BUY only checks `last_date + 1 day`. For Friday-before-Monday-holiday cases (MLK, Presidents', Memorial, Labor Day) `next_day` is **Saturday** → pre-holiday BUY never fires. Misses the exact long-weekend setup it's named for. | Claude reviewed the calendar effect categories but didn't verify the scanning window matches the documented intent. |
| `futures_flow.py:286-287` — Sub-signals use `d.get("oi", 0)` but reporting block at 286 uses `d["oi"]`. Sparse OI history → `KeyError` after vote computed → entire `futures_flow` returns failed/HOLD instead of degraded. (Codex confirmed by runtime probe — log shows `KeyError 'oi'`.) | Claude flagged the related `ls_ratio[-1]["longShortRatio"]` KeyError (P1) but missed this second instance for `oi` in the reporting block. |
| `mean_reversion.py:460` — `hasattr(df.index, "hour")` is False for the standard OHLCV shape (timestamps in `df["time"]` column, RangeIndex). **Metals seasonality profile silently never applied in production.** Same assumption duplicated in `momentum_factors._apply_seasonality()`. | Claude reviewed mean_reversion for entry/exit logic but didn't check the seasonality dispatch condition. **Silent feature death** class of bug — the seasonality has been doing nothing for these tickers. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `econ_calendar.py:137` — `if evt is None or evt["hours_until"] > 24: return "BUY"`. **`evt is None` means calendar exhausted (running past last hardcoded date) → silent BUY on data staleness.** | Codex didn't check the post-event-relief return-on-None path. Same staleness-as-bullish class as Claude's data-external econ_dates 14:00 UTC finding. |
| `credit_spread.py:285` — `cfg = load_json("config.json", default={})` — relative path fails when CWD ≠ repo root (PF-DataLoop scheduled task can launch from `C:\Windows`). FRED key empty → silent HOLD. Same bug fixed in `cot_positioning.py` (SM-P1-4). | Codex didn't check CWD-dependence. Project-specific (Windows scheduled task quirk). |
| `volume_flow.py:323-324` — NaN price change defaults `price_up=True` → biases volume RSI toward BUY. Comment says "neutral bias" but BUY is directional. | Codex didn't audit the NaN fallback direction. |
| `volatility.py:160, 264` — `_historical_volatility` uses `sqrt(365)` but `_garch_signal` uses `sqrt(252)` — inconsistent annualization within the same composite signal. Indicator values to logs/consumers on different scales. | Codex didn't compare the two annualization conventions. |
| `cot_positioning.py:212-221` — `_sub_commercial_change` HOLD guard checks `comm_net is None` but vote requires `noncomm_net_change`. Wrong field gate. | Codex didn't trace the HOLD-guard vs vote-data field consistency. |
| `calendar_seasonal.py` — applies stock-market calendar effects to 24/7 crypto/metals (no asset-class guard). **Active at 3h horizon for crypto/metals.** | Codex's findings on calendar_seasonal were within-stock bugs; Claude added the cross-asset-class concern. |

## Disagreements

None directly. Both reviewers found independent bugs. Notably:

- **Calendar signal**: Both flagged it with completely different bugs. Codex found within-stocks January double-vote bias and Saturday pre-holiday miss. Claude found cross-asset-class misapplication and the bigger "Did NOT find" insight that the per-ticker accuracy data already shows the signal has 100% BUY bias on crypto/metals. **Together they paint a much grimmer picture: calendar_seasonal is broken multiple ways.**
- **Futures flow**: Both flagged different KeyError opportunities (Claude: `ls_ratio`, Codex: `oi` in reporting block). Both real.

## What both missed (likely)

- **`structure.py` Donchian breakout band** — Claude's "Did NOT find" notes correct exclusion of current bar at `high.iloc[-(period+1):-1]`. Neither asked whether this matches the band used at signal-fire time vs the one logged.
- **`smart_money.py`** — DISABLED globally per memory. Both skimmed. If re-enabled, the BOS/CHoCH state machine deserves its own review.
- **Forecast signal Chronos vote weight under regime conditioning** — Claude flagged the `_cached`+circuit-breaker stale return but neither asked whether Chronos's weight is correctly downweighted in ranging regimes.
- **`heikin_ashi.py`** — neither flagged anything. HA Trend has known regime-dependence issues.

## Reconciled verdict

**P0 (must fix):**
1. **(Claude)** `econ_calendar.py:137` returns BUY on stale calendar (`evt is None`). Production silently produces BUYs once 2026 dates are exhausted (or earlier with future dates).
2. **(Claude)** `credit_spread.py:285` relative `config.json` path → silent HOLD when scheduled task CWD wrong.
3. **(Codex)** `mean_reversion.py:460` + `momentum_factors._apply_seasonality()` — `df.index.hour` False for standard shape → metals seasonality silently dead.

**P1:**
4. (Codex) `calendar_seasonal.py:157-163` January double-vote BUY bias.
5. (Codex) `calendar_seasonal.py:222-223` pre-holiday BUY misses Saturday gap → MLK/Presidents'/Memorial/Labor day pre-holiday setups never fire.
6. (Claude) `calendar_seasonal.py` no asset-class guard (active at 3h for crypto/metals despite per-ticker 1d disabling).
7. (Codex) `futures_flow.py:286-287` `d["oi"]` KeyError aborts entire signal.
8. (Claude) `futures_flow.py:118` `ls_ratio[-1]["longShortRatio"]` similar KeyError.
9. (Claude) `volume_flow.py:323-324` NaN→BUY bias.
10. (Claude) `volatility.py:160, 264` inconsistent `sqrt(365)` vs `sqrt(252)`.

**P2:**
11. (Claude) `cot_positioning.py:212-221` HOLD guard checks wrong field.
12. (Claude) `volatility.py:244-251` dead `len(returns) < 20` guard.
13. (Claude) `forecast.py:847-848` cached result bypasses circuit breaker for 5 min.
