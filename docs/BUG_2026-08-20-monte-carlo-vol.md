# Monte Carlo annualized volatility is understated ~7-12x

**Found:** 2026-08-20 · **Severity:** high (risk model) · **Status:** FIXED for the
Monte Carlo family (`47a49c67`, `aeb0f020`). Order-placing callers deliberately
left on the old function — see "Deliberately not migrated" at the bottom.

## Symptom

`data/agent_summary.json` reports `monte_carlo.volatility_annual = 0.05` for BOTH
BTC-USD and ETH-USD. Identical values are the tell: 0.05 is `MIN_VOLATILITY`, the
floor at `portfolio/monte_carlo.py:32`. The computed value is below the floor, so
the floor is masking the defect.

Measured realized volatility over the same window (30d log returns, annualized):

| Ticker  | reported | realized 30d | realized 1h-based |
| ------- | -------- | ------------ | ----------------- |
| BTC-USD | 5.0%     | 34.3%        | 38.8%             |
| ETH-USD | 5.0%     | 62.7%        | 62.8%             |
| XAU-USD | —        | 28.7%        | 21.4%             |
| XAG-USD | —        | 43.3%        | 34.8%             |

## Root cause — two compounding errors

`portfolio/monte_carlo.py:60-74`

```python
def volatility_from_atr(atr_pct, period=14, trading_days=365):
    atr_frac = atr_pct / 100.0
    annual_factor = math.sqrt(float(trading_days) / period)   # <-- BUG 1
    vol = atr_frac * annual_factor
    return max(vol, MIN_VOLATILITY)
```

### Bug 1: the `/ period` term (understates by sqrt(14) = 3.74x)

The formula treats `ATR(14)` as a _14-period cumulative range_, and so divides the
per-period volatility by `sqrt(14)`. But ATR(14) is the **mean of 14 one-bar true
ranges** — it is itself a ONE-bar measure. Annualizing a one-bar quantity requires
`sqrt(trading_days)`, not `sqrt(trading_days / period)`.

Verification using daily-bar ATR:

| Ticker | daily ATR% | current formula | fixed `sqrt(365)` | realized |
| ------ | ---------- | --------------- | ----------------- | -------- |
| BTC    | 2.133%     | 10.9%           | **40.7%**         | 34.3%    |
| ETH    | 2.919%     | 14.9%           | **55.8%**         | 62.7%    |

The fixed formula lands within ~20% of realized. The current one is off by ~3.7x.

### Bug 2: the `atr_pct` actually supplied is intraday, not daily

`simulate_all` receives `atr_pct = 0.54` (BTC) / `0.53` (ETH). Measured ATR% is
2.13% on daily bars and 0.98% on hourly bars — so the supplied value is on an
intraday (sub-hourly) scale while the annualization assumes one bar = one day.

`portfolio/fin_fish.py:344-348` shows this was partially known:

```python
# Hourly ATR path — volatility_from_atr assumes hourly candles
vol = volatility_from_atr(atr_pct)
# (volatility_from_atr uses sqrt(252/14) which is wrong for daily data)
```

Note the function is wrong for hourly input too: correct annualization of an
hourly sd is `× sqrt(24*365) = 93.6`, not `× sqrt(365/14) = 5.1`.

## Consequences (all observed live in agent_summary.json)

- `portfolio_var.drawdown_1pct_prob = 0.0` and `drawdown_5pct_prob = 0.0`.
  At BTC's real 34% vol a 1% daily drawdown is roughly a coin flip. The VaR
  model asserts it is impossible.
- `p_stop_hit_1d = 0.0` for BTC — the risk layer believes a 2xATR stop can never
  be hit intraday. Combined with `stop_price = price*(1 - 2*atr_pct/100)`, that
  stop sits 1.07% below spot on a 34%-vol asset: it would be hit constantly.
- `price_bands_1d` for BTC spans 71,508–72,120 (p5–p95) — a ±0.4% band where the
  true 1d 90% interval is roughly ±3%.
- `price_targets` XAG 6h band spans 66.733–66.866 (0.2%) where 43% vol implies
  roughly ±1.2%. Fill probabilities and EV on the metals ladder are therefore
  wrong, and every rung is placed far too close to spot.
- `p_up` becomes drift-dominated (vol term vanishes), so MC `p_up` of 0.709 (BTC)
  / 0.20 (ETH) carries no distributional information.

## Blast radius

Consumers of `volatility_from_atr` / `monte_carlo`:

- `portfolio/monte_carlo_risk.py:435` — portfolio VaR/CVaR
- `portfolio/price_targets.py:333` — structural price targets
- `portfolio/fin_fish.py:345` — metals bid/exit ladder (**real money via Avanza**)
- `portfolio/reporting.py:523` — writes `agent_summary.monte_carlo`
- downstream readers: `crypto_scheduler.py:118`, `mstr_loop/data_provider.py:169`,
  `fish_monitor_smart.py:173`, `exit_optimizer.py` (has its own vol floor)

Current live exposure is limited: the patient book shows `n_positions: 1` /
$1,796 exposure, and the metals loop is not running on this host. The defect
matters the moment metals or grid-fisher restarts.

## Why this must not be a one-line fix

`fin_fish.py` and `grid_fisher.py` may have been empirically tuned against the
understated vol. Correcting `volatility_from_atr` in place would widen every
ladder rung and every stop simultaneously, changing real-money order placement.

Recommended approach:

1. Add a new correctly-scaled function (e.g. `annualized_vol_from_atr(atr_pct,
bar_interval, trading_days)`) that takes the bar interval explicitly, so the
   caller cannot silently supply intraday ATR where daily is assumed.
2. Unit-test it against realized vol for BTC/ETH/XAU/XAG — the fixed formula
   should land within ~25% of realized, as verified above.
3. Migrate callers one at a time, starting with the ones that do NOT place
   orders (`monte_carlo_risk`, `reporting`), and validate the VaR output against
   the equity curve's own `volatility_annual_pct` (`equity_curve.py:233`), which
   is computed independently from returns and can serve as a cross-check.
4. Migrate `price_targets` and `fin_fish`/`grid_fisher` last, with the metals loop
   stopped, and re-tune ladder spacing deliberately rather than inheriting it.
5. Consider removing or lowering `MIN_VOLATILITY = 0.05`; its only effect here was
   to hide the bug behind a constant.

---

## Resolution (2026-08-20)

Commits `7682514d` (fix) · `c59777e5` (rounding) · merged `47a49c67`, `aeb0f020`.

**New API.** `annualized_vol_from_atr(atr_pct, trading_days, atr_to_sd)` in
`portfolio/monte_carlo.py`. Requires a **daily** ATR and says so in the
docstring. Divides by `ATR_TO_SD = 1.2` — both the expected range of a Gaussian
bar and the measured ratio (1.24 / 1.02 / 1.25 / 1.33 across BTC/ETH/XAU/XAG,
mean 1.21).

**Why the interval had to change too.** The measured ATR-to-realized-vol ratio
is stable on daily bars but not intraday:

| bars | BTC | ETH | XAU | XAG |
|---|---|---|---|---|
| daily | 1.24 | 1.02 | 1.25 | 1.33 |
| 1h | 2.79 | 2.98 | 1.14 | 1.33 |
| 15m | 3.20 | 2.06 | 0.97 | 1.13 |

So rescaling the 15m input was not an option — microstructure and 24/7 trading
break sqrt-of-time intraday. Instead the daily ATR is lifted from the **"7d"
horizon**, which already fetches `interval="1d"` every cycle
(`data_collector.TIMEFRAMES`); `compute_indicators` computed `atr_pct` from it
and reporting was discarding it. **Zero new network calls.** Published as
`signals[tkr].daily_atr_pct`, rounded to 3dp.

Fallback is `_ATR_DEFAULT_BY_CLASS` (already daily-scale), never the intraday
value — falling back to that would silently reinstate the bug.

**Migrated:** `simulate_ticker`, `compute_portfolio_var`.

### Live before/after, verified in agent_summary.json

| | before | after | realized |
|---|---|---|---|
| BTC `volatility_annual` | 0.05 (floor) | **0.429** | ~0.34 |
| ETH | 0.05 (floor) | **0.618** | 0.626 |
| XAU | 0.05 | **0.285** | 0.287 |
| XAG | 0.05 | **0.515** | 0.433 |
| `drawdown_1pct_prob` | **0.0** | **0.665** | — |
| `drawdown_5pct_prob` | **0.0** | **0.135** | — |
| VaR95 as % of exposure | 0.27% | **6.3%** | — |
| 2xATR stop distance | ~1.1% | **3.6-7.8%** | — |

ETH and XAU land within 2% of realized. XAG runs ~19% high and BTC ~26% high
because ATR's 14-day window weights the recent shock differently than a 30-day
stdev — expected for a proxy, and far inside the ±25% the tests assert.

### Deliberately not migrated

`fin_fish.py:345`, `price_targets.py:333`, and through them `grid_fisher.py`
still call the deprecated `volatility_from_atr`. Those size **real Avanza limit
ladders and stop levels** and may have been empirically tuned against the
understated number; correcting it underneath them would widen every live rung
and stop simultaneously. `volatility_from_atr` is therefore left in place,
behaviour unchanged, with a docstring stating it is wrong and why it survives.

To migrate: stop the metals loop, switch the call, and **re-tune ladder spacing
deliberately** rather than inheriting whatever the wrong vol produced.

### Separate limitation noticed, NOT fixed

`p_stop_hit_*` is computed as `mc.probability_below(stop_price)` on the
**terminal** simulated price. A stop can be touched intraday and recover, so
this understates the true probability of a stop being hit. It is now at least
vol-sensitive (0.021 XAU, 0.044 XAG) rather than identically zero, but a
path-minimum barrier calculation would be the correct model. Distinct from the
volatility defect above.

---

## Stage 2 — order-placing callers migrated (2026-08-20)

Commit `0fdfd7f6`, merged `eaf2919d`. The metals loop was verified stopped
first: `pf-metalsloop` / `pf-silvermonitor` / `pf-golddigger` all
inactive+disabled, no matching processes, heartbeats 796h stale, last metals
decision 2026-07-17, `grid_fisher_state.json` untouched since 2026-07-17, zero
order POSTs in the journal.

### Migrated

| call site | was | now |
|---|---|---|
| `price_targets.compute_targets` | `volatility_from_atr(15m)` | `annualized_vol_from_atr(daily)` |
| `price_targets.compute_all_targets` | `extra.atr_pct` | `daily_atr_pct` |
| `fin_fish._compute_vol_and_drift` | `volatility_from_atr(15m)` | `annualized_vol_from_atr(daily)` |
| `fin_fish.load_signal_data` | — | surfaces `daily_atr_pct` |
| `metals_ladder` (x2) | `atr_pct`, default 0.3 | `_daily_atr_pct()`, default 3.0 |
| `metals_execution_engine` (2 atr sources, 2 drift calls, `_warrant_vol_from_underlying`) | 15m | daily |

`swedbank/signals.py` needed **no change**: it builds indicators from
`("one_year", "day")` bars, so its `atr_pct` was already daily and simply became
correct. Live values 2.4–9.6% confirm the scale.

**`volatility_from_atr` is REMOVED.** After the migration it had zero
production callers. Leaving mathematically wrong code importable is precisely
what let this bug persist — `fin_fish` carried a comment saying the function was
wrong for daily data and used it anyway.

### Measured effect on the live XAG ladder (5x warrant, 6h to close)

| | buy rung | exit target |
|---|---|---|
| before (15m ATR) | −0.140% | +0.078% |
| after (daily ATR) | **−1.605%** | **+0.909%** |

**The exit target is the material finding.** +0.078% underlying x5 leverage is
**+0.39% on the warrant against a 0.50% spread** on `BULL_SILVER_X5_AVA_3` —
the old ladder's exit sat **inside the spread** and could not profit even on a
perfect fill (net −0.11%). Corrected: +4.55% gross, **+4.05% net**. Instruments
with 0.07–0.20% spreads squeaked by; the 5x AVA certs the grid actually trades
did not.

**Operational expectation:** materially fewer fills. A 1.6% dip is much rarer
in a 6h window than a 0.14% one. The ladder's character changes from
noise-scalping to genuine dip-fishing, with a viable edge per fill instead of a
high fill rate at negative expectancy.

### Two interaction bugs surfaced by the correction

**1. Flash reserve collapsed onto the working rung.** `flash_underlying =
min(working_underlying, spot*(1-flash_drop))` was implicitly safe while vol was
understated and the working rung always shallow. With correct vol,
`extremes.p25` can already be deeper than the historical post-open drop, so that
`min()` returned `working_underlying` and the ladder emitted
`flash_price == working_price` — **two identical live orders**. Fixed: the flash
rung must clear the working rung by `_FLASH_MIN_EDGE_PCT` (0.15%) or it is
disabled. It arms at daily ATR <=2% and disarms above ~3%.

**2. A regime-dependent test assumption.** `test_metals_execution_engine`
asserted a positive Chronos drift always *raises* `expected_close_underlying`.
That held only while vol was floored: `drift_from_probability` scales with vol,
so at the old 0.22 vol the signal drift was ~2.3 against a Chronos drift of
~2.86 (Chronos pulled up). At the corrected ~0.70 vol the signal drift is ~7.3,
making +18%/24h the *less* bullish view, which correctly tempers the expectation
downward. Rewritten to the durable invariant: the 0.7/0.3 blend must land
strictly between the signal-only and chronos-only expectations — a form that
also passes under the old vol.

### Verification

10 new tests, 4 stale expectations corrected, 216 tests green across every
affected suite. Full suite shows **zero regressions**: the 5 deltas versus the
prior worktree baseline are 4 herc2-reachability-dependent applicable-count
tests (all reproduce with the change stashed) plus one known hash-order
nondeterminism — `signal_engine.py:2718` does `max()` over a **set**, which
passes at `PYTHONHASHSEED=0` and fails at 1-5. That one is a real, separate,
unfixed bug: on an exact accuracy tie the correlation-group leader is chosen by
string-hash order and the loser eats a 0.3x follower penalty that can flip
consensus direction. One-line fix: `key=lambda s: (_leader_accuracy_key(s), s)`.

### Still not migrated

Nothing. `grid_fisher.py` consumes `price_targets` / `metals_ladder` output
rather than calling the vol function directly, so it inherits the correction.
Re-tune its ladder spacing against the new, wider rungs before the metals loop
is restarted.
