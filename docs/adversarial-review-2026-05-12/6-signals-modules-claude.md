# Claude adversarial review: signals-modules (2026-05-12)

## Summary

Reviewed all 50 modules in `portfolio/signals/`. The dispatcher in
`portfolio/signal_engine.py` (lines 3495, 3554-3562) provides robust
defense-in-depth — DISABLED_SIGNALS are intercepted before invocation,
`_validate_signal_result` normalizes outputs, and outer try/except converts
any module crash to a HOLD. So most bugs surface as quality / accuracy
problems rather than blocking incidents.

That said, several issues warrant attention:

1. **Race-condition risk** on module-level caches in newly-added signals
   (`hash_ribbons._hash_cache`, `crypto_evrp._DVOL_CACHE`,
   `crypto_evrp._DVOL_HISTORY_CACHE`, `credit_spread._oas_cache`,
   `copper_gold_ratio._CACHE`). All are read/written without a lock
   while the loop runs 8 ThreadPool workers in parallel across tickers.
   `gold_real_yield_paradox._yield_cache`, `metals_cross_asset._epu_cache`,
   `forecast._forecast_lock` use locks correctly — the new modules diverged
   from that established pattern.

2. **Silent except → HOLD anti-pattern** still pervasive in `momentum.py`
   (8 sub-signal blocks, lines 358–425), `vwap_zscore_mr.py` (top-level
   wrapper, line 124), and `news_event.py` (7 sub-signal blocks,
   lines 549–581). Earlier sweep (commit `b1587646`) added debug logging
   to 47 such blocks but missed these.

3. **Relative `config.json` loads** in `credit_spread.py:285`,
   `gold_real_yield_paradox.py:265`. SM-P1-4 fixed exactly this pattern
   in cot_positioning a few commits ago. These signals will silently
   return HOLD when launched from Task Scheduler / any CWD that is not
   the repo root.

4. **Look-ahead bias absent** in the modules I read — the new VWAP/cubic
   trend / drift / williams_vix_fix detectors all use only data up to
   `iloc[-1]` for the verdict. No bar is consumed from the future.

5. **Asset-class scoping** is correctly enforced in
   `futures_flow.compute_futures_flow_signal:233` (crypto-only),
   `metals_cross_asset:285` (metals-only), `cot_positioning:336`
   (metals-only), `gold_overnight_bias:152` (metals-only),
   `crypto_evrp:283` (crypto-only), `hash_ribbons:248` (BTC-only),
   `dxy_cross_asset:54` (metals+crypto allow-list), `residual_pair_reversion`
   (PAIR_MAP gates by ticker). **But:** see P2 below for sub-signals
   whose direction is hardcoded for one class.

6. **Hardcoded confidence cap = 0.7** is the de-facto convention.
   `vwap_zscore_mr.py` was registered with `max_confidence=0.85` but
   the module body caps internally at 0.85 — slightly higher than
   the rest of the post-2026-04 cohort. Minor inconsistency.

## P0 — Blockers

None. The dispatch-layer guardrails neutralise the worst module-level
bugs before they reach the consensus vote.

## P1 — High

### P1-1 — `intraday_seasonality` hour/dow tables emit BUY-only, never SELL

`portfolio/signals/intraday_seasonality.py:110-129`

`_hour_alpha_vote` and `_dow_vote` only ever return `"BUY"` or `"HOLD"`.
The composite's direction is taken from `_trend_direction` (EMA9/EMA21),
so the structural BUY bias propagates whenever the EMA pair is rising —
combined with `combined_mult >= 1.1` it amplifies long-only confidence
and **suppresses SELL confidence** because no sub-signal contributes
a SELL vote during favourable seasonal hours.

This is the same structural-BUY-only failure mode that killed `calendar`
(2026-05-09: 29.3% recent accuracy → DISABLED_SIGNALS). Signal is
currently in DISABLED_SIGNALS, but graduating to live without rebalancing
the hour/dow tables to also emit SELL on `mult <= 0.7` will repeat
the calendar pattern.

### P1-2 — Direct `requests.get` + no `_cached()` in `cot_positioning._fetch_cot_historical`

`portfolio/signals/cot_positioning.py:102`

```python
resp = requests.get(url, timeout=_CFTC_TIMEOUT)
```

No `http_retry.fetch_with_retry`, no rate limiting, no cache. Called
from `compute_cot_positioning_signal` whenever local history has <20
entries — which is the bootstrap state. Each cycle, for each metals
ticker × each timeframe (XAU/XAG × 7), this fires up to 14 CFTC
network round-trips. If CFTC rate-limits or the bootstrap path persists
(load_jsonl returns less than threshold), the loop pays the latency cost
every cycle. Wrap with `_cached("cot_history_<commodity>", 86400, ...)`.

### P1-3 — Module-level caches without locks (8-worker race)

The Layer-1 ThreadPoolExecutor runs 8 ticker workers concurrently. The
following module-level dicts are read AND written without a lock:

- `portfolio/signals/hash_ribbons.py:51` — `_hash_cache: dict = {}`
  (write at lines 107-108, read at 64-65)
- `portfolio/signals/crypto_evrp.py:51,53` — `_DVOL_CACHE`,
  `_DVOL_HISTORY_CACHE` (write at 107, 171; read at 71-72, 121-122)
- `portfolio/signals/credit_spread.py:53` — `_oas_cache`
  (write at 113-115; read at 62-67)
- `portfolio/signals/copper_gold_ratio.py:43` — `_CACHE`
  (write at 113; read at 71-72)

Compare with the correct pattern in `metals_cross_asset.py:88`
(`_fred_cache_lock`) and `gold_real_yield_paradox.py:40`
(`_yield_cache_lock`). The Python GIL makes pure dict reads/writes
atomic in CPython, but the **read-decide-write** ordering is not:
two workers can both pass the TTL check, both fetch, then both write,
producing duplicate API calls (mild) — and one can write a partially
populated dict mid-read by the other. The worst observable symptom
would be brief stale-or-None reads triggering early `return empty`.

### P1-4 — Relative `config.json` loads in shadow-mode signals

`portfolio/signals/credit_spread.py:285`,
`portfolio/signals/gold_real_yield_paradox.py:265`:

```python
cfg = load_json("config.json", default={})
```

These are CWD-relative. PF-DataLoop has CWD = repo root, but
PF-DataLoop has historically been launched from `C:\Windows` and other
locations (commit `97eb05f0` fixed exactly this in cot_positioning).
When the relative load fails, both signals return `empty` and silently
emit HOLD, masking accuracy data. Mirror the
`Path(__file__).resolve().parent.parent.parent / "config.json"` pattern
used in `forecast.py:90` and `cot_positioning.py:33`.

### P1-5 — `momentum.py` has 8 silent except → HOLD blocks with no logging

`portfolio/signals/momentum.py:358, 367, 377, 386, 395, 404, 414, 425`

```python
except Exception:
    sub_signals["rsi_divergence"] = "HOLD"
```

No `logger.debug(...)`, no `logger.exception(...)`. Commit `b1587646`
added debug logging to 47 such blocks across 8 modules but missed
momentum (and mean_reversion's sub-signal blocks at 504, 514, 524, 534,
547, 562, 573, 585 already log — the inconsistency is just in momentum).
This is `momentum` — an enabled signal contributing to live consensus.
If any sub-indicator crashes (pandas API drift, scipy edge case), the
operator never knows.

### P1-6 — `vwap_zscore_mr.py` top-level except swallows everything

`portfolio/signals/vwap_zscore_mr.py:124-125`

```python
except Exception:
    return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
```

This is the top-level handler for the entire compute function (not just
one sub-signal). Any indexing error, dtype issue, or pandas operation
drift inside the body silently produces HOLD across **all tickers**
forever. Signal was added two days ago (2026-05-10) so the empirical
accuracy will mask the silent-fail mode until the signal is enabled.
Wrap individual sub-signals like every other module in this directory.

## P2 — Medium

### P2-1 — `vol_ratio_regime` returns non-protocol sub_signals values

`portfolio/signals/vol_ratio_regime.py:255-261`

```python
"sub_signals": {
    "gk_cc_regime": "ranging" if gk_cc > 2.0 else ("trending" if ... else "neutral"),
    ...
    "composite_regime": regime,
    "directional_vote": action,
},
```

Per `.claude/rules/signals.md`, each sub-signal vote must be one of
`BUY`/`SELL`/`HOLD`. Downstream consumers of `sub_signals`
(`accuracy_stats.py`, `signal_log.db`, dashboard tooltips) parse these
values; "ranging"/"trending"/"neutral" will be silently misclassified
as HOLD by every consumer that compares against the standard set.
Signal is in DISABLED_SIGNALS so the live impact is zero, but the
shadow-mode accuracy table will report wrong activation rates.

`portfolio/signals/shannon_entropy.py:137,140` does the same
(`"trending_strengthening"`, `"noise_increasing"`, `"stable"`) but
remaps to `BUY`/`SELL`/`HOLD` inside the composite — only the
intermediate `indicators["entropy_momentum"]` carries the non-standard
string, which is fine. Vol_ratio_regime is the strict violation.

### P2-2 — `hash_ribbons` hardcoded `confidence = 0.7` regardless of agreement

`portfolio/signals/hash_ribbons.py:282`

```python
if hash_fires and price_vote == "BUY":
    action = "BUY"
    confidence = 0.7  # High conviction when all conditions align
```

Confidence is identical whether the recovery crossover is fresh (1 day
old) or stale (14 days old), and whether the SMA spread is +10% or
+0.1%. The cap is the registered `max_confidence=0.7`, which is fine,
but the value should at least scale with `days_since_recovery /
hash_ratio` magnitude.

### P2-3 — `cubic_trend_persistence` ignores the documented Bouchaud
coefficient sign for short horizons

`portfolio/signals/cubic_trend_persistence.py:23-26`

```python
B_DAILY = 0.0129
C_DAILY = -0.0062
B_HOURLY = 0.00132
C_HOURLY = -0.00039
```

The cubic-curve model is calibrated for daily returns on developed
equity indices in Bouchaud et al. The hourly coefficients in the module
have no citation and the threshold `phi > 0.3` / `e_r > 0.0005` is also
unsourced — meaning the "trend persistence" sub-signal applies the
same regime classifications regardless of asset class. Crypto's hourly
return distribution has fat tails that violate the AR(1) assumption.
Signal is in DISABLED_SIGNALS pending live validation; before
graduation, document or recalibrate per asset class.

### P2-4 — `residual_pair_reversion._PAIR_MAP` uses futures symbols for
crypto pairs that may have spot↔futures basis drift

`portfolio/signals/residual_pair_reversion.py:65-66`

```python
"XAG-USD": "GC=F",      # gold futures (yfinance symbol)
"XAU-USD": "SI=F",      # silver futures (inverse pair)
```

Target is XAG-USD (Binance FAPI perpetual price), driver is GC=F
(NYMEX August/October gold front-month futures). Roll dates and
storage-cost basis make the OLS residual jumpy each contract roll.
Same for XAU-USD ↔ SI=F. Either switch to spot XAU/XAG via Binance
or apply a roll-adjusted futures continuous series.

The function comment ("inverse pair") is also misleading — the
direction is taken from the z-score sign without any explicit
inversion. Comment vs code drift.

### P2-5 — `treasury_risk_rotation._invert()` violates the "never
invert sub-50% signals" rule by inverting deterministically per asset

`portfolio/signals/treasury_risk_rotation.py:145-150,183-185`

```python
if is_safe_haven:
    action = _invert(action)
```

Per `.claude/rules/signals.md`: *"NEVER invert sub-50% signals — gate
them as HOLD instead. Inversion causes whiplash."* The rule is about
**accuracy-based** inversion (i.e., "this signal is 35% accurate, so
invert it to get 65%"). Treasury_risk_rotation inverts based on
asset-class polarity (steepening → BUY risk-on, SELL safe-haven). That
is semantically correct intermarket logic, NOT the inversion the rule
warns about. **But:** if the underlying composite vote already has
sub-50% accuracy on risk-on assets, applying `_invert` makes the
safe-haven side spuriously high-accuracy without fixing the underlying
model. Same anti-pattern exists in `copper_gold_ratio.py:251-252` and
`xtrend_equity_spillover` (via `_SAFE_HAVEN_TICKERS` invert).
Recommend: when a signal-class is below the directional gate threshold
on its primary class, force-HOLD on **both** sides rather than letting
the inverted side appear to work.

### P2-6 — `forecast.py` 24h accuracy not enforced per-horizon at the
sub-signal layer

`portfolio/signals/forecast.py:721`

```python
scaled_conf = base_conf * acc * r_discount
```

`acc` is fetched via `get_all_ticker_accuracies(horizon="24h", days=14)`
(line 542) — a single per-ticker accuracy for 24h. But the composite
vote combines 1h AND 24h Chronos predictions, where the 1h horizon
historically has worse accuracy than 24h (per the comments in the
module header — 1h=45.4%, 24h=52.4%). Scaling confidence by 24h
accuracy alone masks the 1h horizon being structurally weaker.
`_gate_subsignal_votes_by_accuracy` (line 575) does gate per-sub-signal,
but the final confidence scaling does not.

### P2-7 — `news_event` fetches "stock" headlines for metals tickers

`portfolio/signals/news_event.py:115`

```python
if _is_crypto(short):
    articles = _cached(..., _fetch_crypto_headlines, short, ...)
else:
    articles = _cached(..., _fetch_stock_headlines, short, ...)
```

For `XAU-USD` → `short = "XAU"`, which is not a stock ticker; the
NewsAPI/Yahoo query will return junk or empty. Add a metals branch
(query "gold news", "silver news") or skip the sub-signal for metals
tickers.

### P2-8 — `_validate_signal_result` does not validate `sub_signals`
values against `_VALID_ACTIONS`

`portfolio/signal_engine.py:1505-1507`

```python
sub_signals = result.get("sub_signals")
if not isinstance(sub_signals, dict):
    sub_signals = {}
```

The dispatcher checks the top-level `action` field against
`_VALID_ACTIONS` (line 1489) but does not check individual sub_signal
values. This is what lets `vol_ratio_regime`'s non-protocol values
("ranging", "trending", "neutral") through to downstream consumers.

## P3 — Low

### P3-1 — `volatility._historical_volatility` annualizes by `sqrt(365)` for stocks

`portfolio/signals/volatility.py:160`

```python
hv = log_returns.rolling(window=20, min_periods=20).std() * np.sqrt(365)
```

365 calendar days is right for crypto, wrong for stocks/metals (252
trading days). Reported `hist_vol` indicator is overstated by ~20%
for non-crypto. Vote direction is unaffected (just compares HV now
vs HV-lookback). Cosmetic.

### P3-2 — `calendar_seasonal._US_HOLIDAYS` hardcoded for 2026 only

`portfolio/signals/calendar_seasonal.py:208-220`

```python
_US_HOLIDAYS = [
    (1, 20),   # MLK Day (approx — 3rd Monday)
    (2, 17),   # Presidents' Day (approx — 3rd Monday)
    ...
    (11, 27),  # Thanksgiving (approx — 4th Thursday)
```

Floating-Monday holidays change each year. 2027-01-19 will be MLK Day,
not 2027-01-20. Signal is disabled, but if re-enabled it produces
wrong pre-holiday signal for years other than 2026.

### P3-3 — `intraday_seasonality` only ever emits BUY-side multipliers

(see P1-1 — listing here as P3 secondary impact on confidence inflation
in addition to the direction bias.)

### P3-4 — `cubic_trend_persistence._detect_timeframe` median-based heuristic

`portfolio/signals/cubic_trend_persistence.py:67`

```python
median_hours = diffs.median().total_seconds() / 3600
if median_hours >= 20: return "daily"
```

15-minute, 30-minute, and 4-hour timeframes all bucket to "hourly"
coefficients. The 4h calibration would be closer to daily than to 1h,
but the heuristic uses `>= 20h → daily, else hourly`. Should at least
have a 3-bucket classifier (intraday < 8h, mid 8-20h, daily ≥ 20h).

### P3-5 — `crypto_evrp._evrp_percentile_signal` computes "rv_hist" via
volatility-of-volatility — likely a logic bug

`portfolio/signals/crypto_evrp.py:216-217`

```python
rv_hist = np.log(rv_series / rv_series.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(365) * 100
```

`rv_series` is the close-price series passed in by the caller (named
`close` at line 297). Despite the variable name, this computes the
realized vol of the price series, not the historical RV time series.
The result is then **not used** (the function falls back to ranking
the current eVRP against DVOL percentiles at line 236). Dead code,
but the variable name suggests the author intended to do something
else and left the wrong expression behind.

### P3-6 — `metals_cross_asset._get_fred_key` config traversal is brittle

`portfolio/signals/metals_cross_asset.py:91-102`

The conditional expression at line 100-102 returns `""` when neither
`hasattr` check passes, but a `golddigger` attribute that happens to
exist without a `fred_api_key` attribute will trigger an AttributeError
via the inner `getattr(getattr(cfg, "golddigger", None), ...)` chain
when `getattr(cfg, "golddigger", None)` returns a non-None object
without the attribute. The same brittle pattern appears in 4 other
signal modules. Pre-extract `cfg.golddigger` once and check `is None`.

### P3-7 — `econ_calendar` post_event_relief BUYs regardless of asset class

`portfolio/signals/econ_calendar.py:114+`

After a high-impact event passes, the signal emits BUY universally.
For metals during FOMC tightening, the post-event "relief" can be
SELL (gold drops). For crypto during CPI surprise, it can go either
way. Currently the same BUY sub-signal vote propagates to all 5
Tier-1 instruments — works for risk-on stocks/crypto but
underperforms for metals.

## Modules sampled

Deep inspection (full file or substantial section):

- `portfolio/signals/metals_cross_asset.py`
- `portfolio/signals/dxy_cross_asset.py`
- `portfolio/signals/gold_overnight_bias.py`
- `portfolio/signals/vwap_zscore_mr.py`
- `portfolio/signals/cubic_trend_persistence.py`
- `portfolio/signals/intraday_seasonality.py`
- `portfolio/signals/futures_flow.py`
- `portfolio/signals/trend.py`
- `portfolio/signals/mean_reversion.py`
- `portfolio/signals/orderbook_flow.py`
- `portfolio/signals/macro_regime.py`
- `portfolio/signals/calendar_seasonal.py`
- `portfolio/signals/news_event.py`
- `portfolio/signals/williams_vix_fix.py`
- `portfolio/signals/treasury_risk_rotation.py`
- `portfolio/signals/residual_pair_reversion.py`
- `portfolio/signals/cot_positioning.py`
- `portfolio/signals/crypto_evrp.py`
- `portfolio/signals/credit_spread.py`
- `portfolio/signals/forecast.py`
- `portfolio/signals/momentum.py`
- `portfolio/signals/volume_flow.py`
- `portfolio/signals/vol_ratio_regime.py`
- `portfolio/signals/drift_regime_gate.py`
- `portfolio/signals/hash_ribbons.py`
- `portfolio/signals/copper_gold_ratio.py`
- `portfolio/signals/realized_skewness.py`
- `portfolio/signals/gold_real_yield_paradox.py`
- `portfolio/signals/volatility.py`
- `portfolio/signals/shannon_entropy.py`
- `portfolio/signals/cross_asset_tsmom.py`
- `portfolio/signals/xtrend_equity_spillover.py`
- `portfolio/signals/mahalanobis_turbulence.py`
- `portfolio/signals/crypto_macro.py`
- `portfolio/signals/network_momentum.py`
- `portfolio/signals/futures_basis.py`
- `portfolio/signals/hurst_regime.py`
- `portfolio/signals/econ_calendar.py`
- `portfolio/signals/smart_money.py`
- `portfolio/signals/claude_fundamental.py`

## Modules NOT sampled

Brief structural / header scan only — no deep adversarial inspection:

- `portfolio/signals/momentum_factors.py`
- `portfolio/signals/oscillators.py`
- `portfolio/signals/heikin_ashi.py`
- `portfolio/signals/fibonacci.py`
- `portfolio/signals/structure.py`
- `portfolio/signals/candlestick.py`
- `portfolio/signals/ovx_metals_spillover.py`
- `portfolio/signals/statistical_jump_regime.py`
- `portfolio/signals/vix_term_structure.py`
- `portfolio/signals/complexity_gap_regime.py`

## Tests missing

(Inferred from module risk; not exhaustive — actual `tests/` listing not
re-checked here.)

1. **No test asserts that the `_validate_signal_result` would reject
   `vol_ratio_regime`'s non-`BUY/SELL/HOLD` sub_signals values** —
   either tighten the validator (see P2-8) and add a test that
   enforces it, or add a per-module conformance test.

2. **No test for concurrent-access safety of module-level caches**
   in `hash_ribbons`, `crypto_evrp`, `credit_spread`, `copper_gold_ratio`.
   A stress-test that spins 8 workers calling these compute functions
   for the same ticker would surface the missing locks.

3. **No test that signals fall back gracefully when run from a
   non-repo-root CWD** — `credit_spread.compute_credit_spread_signal`
   and `gold_real_yield_paradox.compute_gold_real_yield_paradox_signal`
   would currently silently return `empty`. Add a test that chdir's
   to `tmp_path` and asserts the signal still functions.

4. **No test that BUY-biased composite signals (calendar,
   intraday_seasonality) emit SELL under bearish conditions** — the
   structural bias check would catch a repeat of the calendar 29.3%
   regression.

5. **No test that the dispatch-layer DISABLED_SIGNALS interception
   prevents compute calls** for disabled modules. The 19 disabled
   modules should be assertable as never-invoked when in
   `DISABLED_SIGNALS` and not in `_SHADOW_SAFE_SIGNALS`.

6. **No test for cot_positioning's CFTC API rate-limit** — wrap the
   `requests.get` call in `_cached()` (P1-2) and add a test that
   asserts only one network call per cache TTL.
