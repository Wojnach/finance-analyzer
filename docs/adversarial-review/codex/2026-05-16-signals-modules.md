## [P1] Turbulence Signal Always Crashes
**File:** portfolio/signals/mahalanobis_turbulence.py:99
**Bug:** `_cached` is called as `_cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)`, passing `ttl` both positionally and by keyword.
**Why it matters:** Any call with enough rows raises `TypeError: _cached() got multiple values for argument 'ttl'`, disabling the signal or crashing the caller.
**Fix:** Call `_cached("mahalanobis_turb_closes", _CACHE_TTL, _do_fetch)`.

## [P2] Timezone Conversion Relabels Local Time As UTC
**File:** portfolio/signals/econ_calendar.py:44
**Bug:** `.replace(tzinfo=UTC)` relabels the timestamp instead of converting it.
**Why it matters:** `2026-05-16 00:30 Europe/Stockholm` becomes `2026-05-16 00:30 UTC` instead of `2026-05-15 22:30 UTC`, shifting event windows and even the event date near midnight/DST.
**Fix:** Use `tz_convert(UTC)` for aware timestamps and explicitly localize naive timestamps before conversion.

## [P2] Post-Event Relief Ignores Data Timestamp
**File:** portfolio/signals/econ_calendar.py:127
**Bug:** `recent_high_impact_events(24)` uses wall-clock time instead of the dataframe-derived `ref_date`.
**Why it matters:** Backtests or delayed feeds can emit live-time BUY/HOLD decisions unrelated to the bar being evaluated.
**Fix:** Pass the dataframe timestamp as `ref_time=ref_date` and keep all calendar calculations anchored to the same reference time.

## [P2] Seasonality Detrending Compounds Returns
**File:** portfolio/signals/momentum_factors.py:356
**Bug:** The loop reconstructs each adjusted close from the previously adjusted close.
**Why it matters:** Flat prices with a repeated `-0.1%` seasonality adjustment drift geometrically across the whole window, fabricating momentum.
**Fix:** Preserve the original close series and reconstruct each adjusted bar from `original_close.iloc[i - 1]`.

## [P2] Invalid Half-Life Math Can Still Vote
**File:** portfolio/signals/mean_reversion.py:356
**Bug:** `_half_life_mr` takes `np.log(prices.values)` without rejecting zero, negative, or non-finite prices.
**Why it matters:** A bad zero tick can produce `nan`/`inf` half-life, then the later z-score checks can still return BUY/SELL with an invalid model.
**Fix:** Reject non-positive and non-finite prices before log regression, and require finite `theta`, `half_life`, and `zscore` before voting.

## [P2] Alternating Jumps Create False Regimes
**File:** portfolio/signals/statistical_jump_regime.py:101
**Bug:** Neutral-regime persistence increments the same `current_count` for both positive and negative jumps.
**Why it matters:** A `+1, -1, +1` jump sequence reaches count 3 and becomes a bull regime despite no consecutive same-direction evidence.
**Fix:** Track pending direction and reset the persistence count whenever jump sign changes.

## [P2] Hurst Momentum Votes Without Direction
**File:** portfolio/signals/hurst_regime.py:203
**Bug:** Rising Hurst returns BUY and falling Hurst returns SELL without checking price trend or current regime.
**Why it matters:** In a random-walk regime, the other sub-signals can be HOLD and this one auxiliary vote becomes the entire composite BUY/SELL because HOLDs abstain.
**Fix:** Make Hurst momentum a confidence modifier or vote only in the confirmed trend direction.

## [P2] News Signal Swallows Sub-Signal Failures
**File:** portfolio/signals/news_event.py:549
**Bug:** Sub-signal exceptions are converted to `("HOLD", {})` with no logging.
**Why it matters:** A keyword parser or sector-impact failure silently neutralizes news risk, allowing trades through market-moving headlines.
**Fix:** Log exceptions with `exc_info=True` and expose an error indicator so the caller can suppress trading on degraded news state.

## [P2] Headline Persistence Is Last-Writer-Wins
**File:** portfolio/signals/news_event.py:96
**Bug:** Every ticker writes to the same `headlines_latest.json`.
**Why it matters:** Concurrent or sequential multi-ticker runs overwrite each other; the monitor can show BTC headlines while NVDA is being traded, or lose severe headlines entirely.
**Fix:** Persist per ticker or atomically merge into a ticker-keyed JSON object under a lock.

## [P2] Credit Spread Config Fallback Uses Relative Path
**File:** portfolio/signals/credit_spread.py:285
**Bug:** `load_json("config.json")` depends on the process working directory.
**Why it matters:** If the scheduler launches outside the repo, the FRED key is missed and the credit-spread risk signal silently disables itself.
**Fix:** Resolve config from a repo-root absolute path or require it through `context`.

## [P2] Expired OAS Cache Trades Indefinitely
**File:** portfolio/signals/credit_spread.py:122
**Bug:** On fetch failure, `_fetch_hy_oas` returns `_oas_cache["data"]` without checking whether it is older than `_CACHE_TTL`.
**Why it matters:** If FRED is down for days, stale credit-spread data can keep producing BUY/SELL votes as if it were current.
**Fix:** Reject cached OAS data older than a bounded max-stale window and include data age in indicators.

## [P2] Metals FRED Cache Ignores Staleness After Failure
**File:** portfolio/signals/metals_cross_asset.py:166
**Bug:** `_fetch_fred_values` returns cached values after failed fetches with no max-age check.
**Why it matters:** EPU/TIPS can drive metals trades from arbitrarily old values during API outages.
**Fix:** Enforce max stale age before returning cached FRED data and emit a degraded-data indicator.

## [P2] Oil Fallback Treats Zero As Missing
**File:** portfolio/signals/metals_cross_asset.py:314
**Bug:** `oil_ctx.get("change_1d_pct") or oil_ctx.get("change_5d_pct")` treats a valid `0.0` one-day move as absent.
**Why it matters:** If 1d oil is flat but 5d oil is +3%, the signal can fabricate an inflation BUY vote.
**Fix:** Check key presence / `is not None` instead of using truthiness for numeric values.

## [P2] COT Report Date Is Never Validated
**File:** portfolio/signals/cot_positioning.py:354
**Bug:** Any `cot_data` dict from deep context is accepted without checking `report_date` age.
**Why it matters:** If precompute stops updating, the system can trade metals from weeks-old positioning while still reporting normal confidence.
**Fix:** Parse `report_date`, reject stale COT data beyond the expected weekly publication lag, and expose stale status in indicators.

## SUMMARY
P1=1 P2=13 P3=0