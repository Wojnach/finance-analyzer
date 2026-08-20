# Monte Carlo annualized volatility is understated ~7-12x

**Found:** 2026-08-20 · **Severity:** high (risk model) · **Status:** diagnosed, NOT fixed

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
