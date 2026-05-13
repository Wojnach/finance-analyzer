# Claude adversarial review: signals-modules

## Summary

Read all 52 `portfolio/signals/*.py` files (~18.6k LOC). The composite-vote
architecture is largely consistent, but there are several **directional
inversions** in active modules, several **structural directional biases**
(BUY-only or SELL-only sub-signal families), a **broken `_cached(...)`
signature in two disabled cross-asset modules** that would crash on enable,
and one **silent dead-code branch** in `crypto_evrp.py`. The Claude-cascade
fundamental signal has **multiple correctness/safety problems** including
unconditional cache-timestamp poisoning before background refresh completes
(stalls refresh on errors), unused alpha_vantage cache hits during
market-closed windows that still call `_get_fundamentals_data()` for both
caching only on subsequent refresh, and a silent same-key on bias detection
that runs the journal scan TWICE per ticker per tier.

The deepest concrete bugs by file:

- `mahalanobis_turbulence.py:99` and `complexity_gap_regime.py:92` — both
  call `_cached(key, fn, ttl=...)` against a `_cached(key, ttl, func, *args)`
  signature → would raise `TypeError` on first invocation. Disabled, but
  any re-enable will crash the cycle for the calling ticker.
- `crypto_evrp.py:216` — computes `rv_hist = np.log(rv_series/rv_series.shift(1))...`
  then **never uses `rv_hist`**; the eVRP percentile sub-signal becomes a
  pure DVOL percentile and ignores realized vol entirely. Direction also
  inverted vs the docstring's stated thesis (`> +10` returns SELL but
  docstring says "high eVRP → bullish price action").
- `cross_asset_tsmom.py:148-171` — `_compute_bond_momentum` and
  `_compute_equity_momentum` return the **same direction regardless of
  target asset class**. TLT up = BUY for everyone, including BTC/ETH/MSTR
  (where rising bonds = risk-off = SELL). Disabled signal, but the bug
  predates the disable flag.
- `claude_fundamental.py:929` — `_cache[tier]["ts"] = time.time()` is set
  to *now* BEFORE the background refresh thread starts. If the thread
  raises, the timestamp is still updated, so `_needs_refresh()` returns
  False for the entire cooldown window. This pre-marks success on failure.
- `structure.py:79-83` — `_highlow_breakout` votes **BUY when price is
  within 2% of the period HIGH** and SELL near the LOW. That's
  resistance-as-support / floor-as-ceiling. Real breakout signals fire on
  *exceeding* the high, not *approaching* it. Active signal.
- `fibonacci.py:457-459, 504-507` — Pivot Points use the **previous BAR**
  H/L/C rather than the previous DAY's. On an hourly DataFrame this
  computes "1-hour pivots", which is non-standard and effectively noise.
  Active signal.
- `econ_calendar.py` + `news_event.py` + `calendar_seasonal.py` —
  structural directional bias (SELL-only or BUY-only sub-signal families)
  documented per-file below.
- `metals_cross_asset.py:210` — assumes `get_all_cross_asset_intraday()`
  returns a dict; no None-guard before `intraday.get(...)`. Active signal.

The remainder are correctness drifts, look-ahead-on-current-bar issues
(every BB/RSI/MACD computed at bar close uses the in-progress bar as a
closed bar — same flaw across all 33 active detectors but accepted as
project convention; flagged once at the cross-cutting level), and
structural biases that will systematically bias the consensus.

## P0 — Blockers

### P0-1 — `crypto_evrp.py` eVRP percentile is dead code

`portfolio/signals/crypto_evrp.py:216` computes `rv_hist` from a return-of-
returns transform of `rv_series` (the close price series — wrong input,
should be RV time-series), then **never references `rv_hist` again**. The
sub-signal silently degrades to a pure DVOL-only percentile. Combined with
the directional inversion below this is the worst inversion bug in scope.

Lines 216-242: `rv_hist` is computed, then `len(rv_hist) < PCTILE_WINDOW`
is checked; if it passes, the function continues but uses `dvol_history`
only.

Module is in DISABLED_SIGNALS, but the bug must be fixed before
re-enabling.

### P0-2 — `crypto_evrp.py` direction inversion vs docstring

`portfolio/signals/crypto_evrp.py:42-43` and `:195-201`:

```python
EVRP_BUY_THRESHOLD = -10.0  # eVRP below this → BUY (vol expansion)
EVRP_SELL_THRESHOLD = 10.0  # eVRP above this → SELL (vol compression)
```

Docstring lines 12-15 say: "*When eVRP is very high (>10), implied vol
far exceeds realized vol — historically precedes mean-reversion downward
in IV, often coinciding with bullish price action (vol compression =
calm = uptrend)*."

So per the documented thesis, high eVRP → BULLISH (BUY), but the code
returns SELL. Direction is inverted from the stated logic. Disabled
module.

### P0-3 — `mahalanobis_turbulence.py:99` wrong `_cached` arg order

```python
return _cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)
```

`_cached` signature (`portfolio/shared_state.py:37`) is `(key, ttl, func,
*args)`. The call passes `_do_fetch` as the `ttl` positional and
`_CACHE_TTL` as a keyword `ttl=` — `_cached` doesn't accept a `ttl=`
kwarg, so this is `TypeError: _cached() got an unexpected keyword
argument 'ttl'`. Module is disabled but will crash on enable.

### P0-4 — `complexity_gap_regime.py:92` identical `_cached` arg order bug

```python
return _cached("complexity_gap_closes", _do_fetch, ttl=_CACHE_TTL)
```

Same TypeError as P0-3. Both modules were apparently written against an
older `_cached` signature. Disabled but will crash on enable.

### P0-5 — `cross_asset_tsmom.py` bond/equity sub-signals ignore target asset class

`portfolio/signals/cross_asset_tsmom.py:148-171`:

```python
def _compute_bond_momentum(yf_data: dict) -> str:
    ret = yf_data["TLT"]["ret_63d"]
    if ret > 0.005:
        return "BUY"          # <-- BUY for all tickers
    if ret < -0.005:
        return "SELL"
    ...
def _compute_equity_momentum(yf_data: dict) -> str:
    ret = yf_data["SPY"]["ret_63d"]
    if ret > 0.005:
        return "BUY"          # <-- BUY for all tickers
```

TLT (long bonds) up = yields fall = risk-off = bullish for gold/silver,
bearish for risk-on assets. SPY up = risk-on = bullish for BTC/ETH/MSTR,
neutral or bearish for gold (rotation out of safe-haven). The function
returns the same direction for all target tickers, so 2 of the 4
sub-signals are systematically miscalibrated for either safe havens or
risk assets, no matter what. Disabled but it's a structural inversion.

## P1 — High

### P1-1 — `structure.py:79-83` `_highlow_breakout` direction inverted

```python
if pct_from_high <= 0.02:
    return "BUY", indicators   # within 2% of 52w HIGH → BUY
if pct_from_low <= 0.02:
    return "SELL", indicators  # within 2% of 52w LOW → SELL
```

This is the opposite of a breakout signal: a breakout fires on *exceeding*
the high (i.e., `close > prior_high`), and being within 2% of the high
means approaching resistance — often a SELL setup. Being within 2% of the
low is often a mean-reversion BUY setup, not a SELL breakdown.

This is sub-1 of the `structure` composite (active signal), so it biases
the entire `structure` vote in the wrong direction whenever price is near
extremes.

### P1-2 — `fibonacci.py` pivot points use last BAR not last DAY

`portfolio/signals/fibonacci.py:457-459` and the pivot subroutines at
`:301-366`:

```python
high_prev = float(high.iloc[-2])
low_prev = float(low.iloc[-2])
close_prev = float(close.iloc[-2])
```

Standard daily/Camarilla pivots take the *previous trading day's* H/L/C.
On a 1h or 5m DataFrame this becomes "the previous bar's" H/L/C, which
makes R1/S1 and R3/S3 indistinguishable from current-bar noise. Both
`_pivot_standard_signal` and `_pivot_camarilla_signal` are affected.
Active signal; biases two of fibonacci's five sub-signals.

### P1-3 — `news_event.py` keyword/severity sub-signals are SELL-only

`portfolio/signals/news_event.py:235-239` (`_keyword_severity_vote`):

```python
if max_sev == "critical":
    return "SELL", indicators
if max_sev == "high":
    return "SELL", indicators
return "HOLD", indicators
```

This function NEVER returns BUY. The corresponding `_sentiment_shift`
also has a structural neg-default at line 295: any moderate-severity
headline without an explicit positive keyword defaults to negative
(`neg += 1`). Combined with the additional dissemination + headline
velocity sub-signals all leaning the same way, news_event is
near-impossible to fire BUY without a very specific keyword soup.

### P1-4 — `calendar_seasonal.py` _sell_in_may never emits SELL

`portfolio/signals/calendar_seasonal.py:160-164`:

```python
if is_weak_period:
    return "HOLD", indicators    # <-- May-Oct: documented "SELL" → returns HOLD
if is_strong_month:
    return "BUY", indicators
return "HOLD", indicators
```

The function name and docstring promise SELL bias in May-Oct, but the
code only emits BUY in strong months and HOLD otherwise. The "sell in
May" half is never expressed as a SELL vote. Net effect: the
calendar_seasonal composite has structural BUY bias (most subs are
BUY-only: month_end, sell_in_may, pre_holiday, fomc_drift,
santa_claus_rally are all BUY-or-HOLD; only january_effect and
day_of_week can emit SELL).

### P1-5 — `econ_calendar.py` 4 SELL-only vs 1 BUY-only sub-signals

`portfolio/signals/econ_calendar.py:48-173`:

- `_event_proximity` → SELL or HOLD
- `_event_type_info` → SELL or HOLD
- `_pre_event_risk` → SELL or HOLD
- `_sector_exposure` → SELL or HOLD
- `_post_event_relief` → BUY or HOLD (only BUY-capable sub)

Even with the 2026 "post_event_relief" fix (line 122 docstring "BUG-218"),
the signal is structurally biased toward SELL. The quorum logic in
`majority_vote` will tend to surface SELL on any active event day.

### P1-6 — `claude_fundamental.py:929` cache TS poisoned before refresh succeeds

`portfolio/signals/claude_fundamental.py:923-934`:

```python
if not skip_refresh:
    for tier in ("haiku", "sonnet", "opus"):
        if _needs_refresh(tier, cooldowns):
            with _lock:
                if _needs_refresh(tier, cooldowns):
                    # Mark as refreshing to prevent duplicate spawns
                    _cache[tier]["ts"] = time.time()
                    t = threading.Thread(target=_bg_refresh, ...)
                    t.start()
```

The `ts` is updated to `time.time()` BEFORE the background refresh runs.
If `_bg_refresh` raises (CLI timeout, JSON parse error, rate limit,
disabled-CLI-token), the cache results stay stale but `_needs_refresh`
returns False until the cooldown elapses again. So a failed refresh
locks out re-attempts for the full cooldown period (5 min haiku /
30 min sonnet / 2 h opus). The intent (de-dup spawn) and the
side-effect (de-dup retry on failure) are conflated. Should track an
in-flight flag separately and only update `ts` on successful completion
in `_refresh_tier`.

### P1-7 — `claude_fundamental.py:128` `_get_fred_key` complicated ternary

`portfolio/signals/claude_fundamental.py:97-110` (also present in
`credit_spread.py:125-136`, `metals_cross_asset.py:91-102`,
`gold_real_yield_paradox.py:43-53`):

```python
return getattr(cfg, "fred_api_key", "") or getattr(
    getattr(cfg, "golddigger", None), "fred_api_key", ""
) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
```

The grouping is `(getattr ... or getattr ...) if (hasattr or hasattr) else ""`.
For an object cfg lacking both attributes the ternary is False → "". But
for an object with `golddigger` attribute that itself lacks
`fred_api_key`, the inner `getattr` falls back to `""` which is falsy,
short-circuits the `or`, returning `""`. Functionally OK but fragile —
trivial to misread, easy to break on refactor. Not strictly a bug but
this pattern is duplicated across 4 files and should be a shared helper.

### P1-8 — `forecast.py:454-466` Kronos 1h gets 2x weight in composite

`portfolio/signals/forecast.py:463-472`:

```python
if kronos_ok and not _KRONOS_SHADOW:
    alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
    alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))   # doubled
    alive_votes.append(sub_signals.get("kronos_24h", "HOLD"))
if chronos_ok:
    alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
    alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))   # doubled
    alive_votes.append(sub_signals.get("chronos_24h", "HOLD"))
```

The 1h sub-signal gets 2x weight intentionally per the comment, but
this double-vote bypasses the `majority_vote(count_hold=False)` accuracy
gating logic and the recency-weighted accuracy table — accuracy is
tracked per sub-signal independently, but the vote pool now counts 1h
twice. The accuracy gate at `_gate_subsignal_votes_by_accuracy` does
the right thing, but the composite vote pool double-counts the same
vote. If 1h has 35% accuracy and gets held to HOLD, both copies become
HOLD (correctly skipped). If 1h passes the gate, both copies vote
together. Hidden but functional weight inflation.

### P1-9 — `smart_money.py:218-236` FVG fill loop INCLUDES current bar

`portfolio/signals/smart_money.py:207-236`:

```python
for i in range(start, n - 2):
    ...
    if candle3_low > candle1_high:
        gap_low = candle1_high
        gap_high = candle3_low
        # Check if gap has been filled by any subsequent bar
        filled = False
        for j in range(i + 3, n):                    # <-- includes last bar (n-1)
            if float(lows[j]) <= gap_low:
                filled = True
                break
        if not filled:
            unfilled_bullish.append((gap_low, gap_high))
```

The inner `for j in range(i+3, n)` includes `j = n-1` (the current bar).
If the current bar's low has reached the gap_low, the gap is marked as
filled and removed from `unfilled_bullish`. Then the later check
`filling_bullish = any(gap_low <= current_close <= gap_high)` cannot
match it because it's been excluded. So the signal can never fire BUY
on a freshly filling FVG — only on stale unfilled ones that haven't
been touched yet. The signal as documented can never emit BUY in its
intended use case. Module is disabled in 2026-04-24 blacklist, but
this is a major directional defect.

### P1-10 — `volatility.py:278-285` GARCH conflates volatility with direction

`portfolio/signals/volatility.py:279-285`:

```python
# Rising GARCH above realized = expanding risk = SELL (caution)
if current_garch_vol > prev_garch_vol and ratio > 1.2:
    return "SELL", indicators
# Falling GARCH below realized = compression = BUY (breakout setup)
if current_garch_vol < prev_garch_vol and ratio < 0.8:
    return "BUY", indicators
```

Volatility *expansion* is not a directional signal — markets can rip up
or crash down with expanding vol. Treating rising vol as "SELL caution"
overrides the directional intent of the composite. The other 6 sub-
signals in `volatility.py` are correctly directional; GARCH is the
oddball that maps risk to direction. Active signal.

### P1-11 — `network_momentum.py:300, 357` corr regime vote without basis

`portfolio/signals/network_momentum.py:296-304`:

```python
if avg_abs_corr > 0.5:
    return "BUY", avg_abs_corr  # direction determined by other sub-signals
if avg_abs_corr < 0.2:
    return "HOLD", 0.0
return "HOLD", 0.0
```

The "BUY" return is a placeholder ("direction determined by other
sub-signals"), then at line 357:

```python
if corr_regime == "BUY" and net_div != "HOLD":
    corr_regime = net_div  # same direction as divergence
```

If `net_div == "HOLD"` the corr_regime stays `"BUY"` without
justification. So whenever average abs correlation > 0.5 and the
divergence sub returns HOLD, this signal contributes a phantom BUY vote.
Module disabled but this is a phantom-direction bug.

### P1-12 — `hurst_regime.py:174-207` `_hurst_momentum` direction inversion

`portfolio/signals/hurst_regime.py:174-207`:

```python
# Strong rising Hurst = trend strengthening -> vote with trend
# Strong falling Hurst = trend weakening -> favor caution
if roc > 0.05:
    return safe_float(roc), safe_float(h_now), "BUY"   # trend strengthening
if roc < -0.05:
    return safe_float(roc), safe_float(h_now), "SELL"   # trend weakening
```

Hurst rising means trend persistence increasing — but this says NOTHING
about direction. Returning "BUY" because Hurst is rising is non-
sequitur. Should be HOLD or scaled by the actual trend direction (e.g.,
ema9-ema21 sign). Module disabled.

### P1-13 — `vix_term_structure.py:93-138` direction ignores asset class

`portfolio/signals/vix_term_structure.py:93-98`, `:101-138`:

```python
def _backwardation_flag(ratio: float) -> str:
    if ratio >= _BACKWARDATION_THRESHOLD:
        return "SELL"
    if ratio < _DEEP_CONTANGO:
        return "BUY"
    return "HOLD"
```

`compute_vix_term_structure_signal` (line 141) takes a `context` arg but
**never reads `ticker` or asset_class**. Backwardation (VIX>VIX3M, stress)
is bearish for risk assets and bullish for safe havens (gold). Returning
the same direction for all asset classes is the same kind of structural
inversion as P0-5. Module disabled.

### P1-14 — `metals_cross_asset.py:210` no None-guard

`portfolio/signals/metals_cross_asset.py:210-213`:

```python
intraday = get_all_cross_asset_intraday()
intraday_ok = sum(
    1 for key in ("copper", "gold_silver_ratio", "spy", "oil")
    if intraday.get(key) is not None
)
```

`get_all_cross_asset_intraday()` can return `None` (yfinance failure,
network outage). `None.get(...)` would raise AttributeError. Active
signal; one network hiccup will crash this signal for the cycle.

### P1-15 — `treasury_risk_rotation.py` docstring vs implementation

`portfolio/signals/treasury_risk_rotation.py:3-5`:

> *Steepening curve (TLT outperforms) signals risk-on; flattening/inverting
>  (IEF outperforms) signals risk-off.*

But TLT (20Y+) outperforming IEF (7-10Y) means long-end yields fell more
than mid-end yields — that's curve *flattening* (bull flattener), often
a recession signal, not a steepening. The implementation correctly uses
TLT_return - IEF_return; the bug is in the documented direction.
Re-reading the code under the corrected mechanic suggests the inversion
flag for `_SAFE_HAVENS` may be wrong direction. Active signal — worth a
deeper audit before trusting it.

## P2 — Medium

### P2-1 — Z-score includes current observation in mean/std

Multiple files (`credit_spread.py:155-163`, `cot_positioning.py:138-155`,
`metals_cross_asset.py:167-178`, `mahalanobis_turbulence.py:296`)
compute z-score as `(current - mean) / std` where mean/std were
computed *including the current value*. This dampens the z-score and
makes extremes look less extreme. Standard practice is to exclude the
current observation. Minor bias.

### P2-2 — `credit_spread.py:285` relative `config.json` path

`portfolio/signals/credit_spread.py:285`:

```python
cfg = load_json("config.json", default={}) or {}
```

Relative path — breaks when the loop's CWD differs from repo root
(e.g., PF-DataLoop launched from `C:\Windows`). The cot_positioning
module fixed this in `SM-P1-4` (line 27-33 comment). credit_spread and
`gold_real_yield_paradox.py:265` still use the relative path.

### P2-3 — `gold_real_yield_paradox.py:130-138` asymmetric paradox

`portfolio/signals/gold_real_yield_paradox.py:120-144` `_paradox_spread`:

```python
both_positive = gold_returns_30d > 0 and yield_change_30d > 0
if both_positive:
    ...
    action = "BUY"
elif gold_returns_30d < 0 and yield_change_30d < 0:
    ...
    action = "HOLD"           # <-- both negative also a paradox, but HOLD
else:
    ...
    action = "HOLD"
```

If gold and yields are inversely correlated by baseline, "both rising"
is paradox A (gold up with yields up = anomalous bull). "Both falling"
is paradox B (gold down with yields down = anomalous bear). The function
treats only paradox A as a signal; paradox B is silently HOLD.
Asymmetric.

### P2-4 — `gold_real_yield_paradox.py:168-173` correlation break direction

```python
if corr_30d > baseline_corr + 0.3:
    action = "BUY"
elif corr_30d < baseline_corr - 0.3:
    action = "SELL"
```

`baseline_corr` is typically ~-0.45 (line 237). `corr_30d > baseline+0.3`
means 30d correlation is *less negative* than baseline (paradox forming
= BUY ✓). `corr_30d < baseline-0.3` means correlation is *more negative*
than baseline (stronger inverse = NORMAL regime). Returning SELL in a
normal regime makes no thesis sense. Should be HOLD.

### P2-5 — `metals_cross_asset.py:99-102` `getattr` chain fragile

```python
return getattr(cfg, "fred_api_key", "") or getattr(
    getattr(cfg, "golddigger", None), "fred_api_key", ""
) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
```

The same complicated ternary as P1-7. Listed P2 since metals already
has the fallback dict path; the obj-path is unlikely to be hit.

### P2-6 — `news_event.py:289-295` "cut" defaults to negative

`portfolio/signals/news_event.py:282-295` `_sentiment_shift`:

```python
elif has_positive or has_bullish_cut:
    pos += 1
elif has_bare_cut:
    # Unmatched "cut" — default to bearish.
    neg += 1
else:
    neg += 1
```

The non-cut else branch (line 295) defaults moderate-severity headlines
without positive keywords or "cut" terms to NEGATIVE. So a headline
like "Apple unveils new model" with moderate severity would default to
neg + 1. Strong structural negative bias for news ambiguity.

### P2-7 — `news_event.py:488-494` thesis alignment never votes against belief

`_thesis_alignment_vote`:

```python
elif direction == "bullish" and neg_count > pos_count and neg_count >= 2:
    indicators["alignment"] = "contradicted"
    return "HOLD", indicators       # never SELL even when news contradicts
elif direction == "bearish" and pos_count > neg_count and pos_count >= 2:
    indicators["alignment"] = "contradicted"
    return "HOLD", indicators
```

When news contradicts the prophecy belief, the signal abstains rather
than voting against. This is a confirmation-bias amplifier — the signal
only ever confirms the existing belief direction.

### P2-8 — `econ_calendar.py:138, 142` redundant `next_event` calls

`portfolio/signals/econ_calendar.py:136-145`:

```python
# Check that no new event is imminent (would negate the relief)
evt = next_event(...)
if evt is None or evt["hours_until"] > 24:
    return "BUY", indicators

# Event-free calm window: next event >72h away
evt = next_event(...)         # <-- called again, same result
if evt is not None and evt["hours_until"] > 72:
    ...
```

Minor — perf only. Caches not used.

### P2-9 — `econ_calendar.py:142` BUY only when event >72h

When there's no upcoming event in the calendar at all, `next_event`
returns `None`, the first BUY branch fires only if there's also a
recent relief event. Otherwise the function falls through past line 142
(`evt is not None and ...`) because `evt is None`, and returns HOLD.
So a permanently calm calendar (no upcoming events) defaults to HOLD,
not BUY, contradicting the "event-free calm window" branch's stated
intent.

### P2-10 — `calendar_seasonal.py:208-220` US holidays are approximate

`_pre_holiday_effect` hardcodes a fixed (month, day) list for US
holidays including "approx 3rd Monday" entries. On years where MLK Day
falls on Jan 19 (not 20), this misses. Cosmetic.

### P2-11 — `cot_positioning.py:172-178` current value included in history

`_sub_cot_index`:

```python
nc_net_history = [nc_net]           # include current first
for h in historical:
    val = h.get("nc_net")
    ...
```

`_compute_cot_index` then uses `current = nc_net_history[0]` and
min/max over the full list. Same z-score-with-itself issue as P2-1.

### P2-12 — `claude_fundamental.py:194` yfinance `Earnings Average`

The earnings calendar parsing at `:178-198` accesses
`cal.get("Earnings Average")` (dict) or `cal.loc["Earnings Average"]`
(DataFrame). yfinance has changed the calendar schema multiple times
over the past year; both keys may be missing on current versions. Silent
empty result. Should add explicit logging.

### P2-13 — `momentum.py:80-117` Stochastic uses crossover gate

`_stochastic` line 110: requires K-crosses-D AND D < 20 for BUY. That's
restrictive — Stochastic OB/OS thresholds (20/80) are more commonly used
on K alone or on the K/D pair without requiring a fresh cross. As coded,
this fires rarely and depends on the exact bar of the cross.

### P2-14 — `oscillators.py:425-428` Coppock SELL is invented

```python
# Classic Coppock BUY: curve turns up from below zero
if val < 0 and val > prev:
    return val, "BUY"

# Symmetric SELL: curve turns down from above zero
if val > 0 and val < prev:
    return val, "SELL"
```

Classic Coppock is a BUY-only indicator (its inventor explicitly noted
it doesn't generate sell signals). Inventing a symmetric SELL is an
overfit risk — there's no academic basis for it. Module disabled.

### P2-15 — `forecast.py:524-530` crypto-ticker test by string

```python
def _is_crypto_ticker(ticker: str) -> bool:
    try:
        from portfolio.tickers import CRYPTO_SYMBOLS
        return ticker in CRYPTO_SYMBOLS
    except ImportError:
        return ticker in {"BTC-USD", "ETH-USD"}
```

The import failure case hardcodes only BTC/ETH — fragile if the project
ever adds SOL-USD etc. Minor.

### P2-16 — `forecast.py:323` Kronos GPU gate vs Chronos GPU gate

`_run_kronos` requests `gpu_gate("kronos", timeout=90)` (line 321),
`_run_chronos` requests `gpu_gate("chronos", timeout=120)` (line 400).
These are separate names — so they cannot block each other on the lock
file (different lock keys). The forecast comment at line 836+ says
Chronos runs first to avoid Kronos holding the lock; that's only true if
both use the *same* lock key. Possibly the intent vs. implementation
mismatch.

### P2-17 — `crypto_evrp.py:159-160` rate limit sleep inside loop

`_fetch_dvol_history` line 160 `time.sleep(0.2)` while holding nothing
— OK pattern, but called inside the cache miss path of a hot signal
function. Multiple ticker calls collide. Per-call latency penalty.

### P2-18 — `vol_ratio_regime.py` no tests evident (TODO file)

Module looks correct, but only the GK formula has a clamp for anomalous
ticks. VR test computes `var(log_ret_k) / (k * var(log_ret_1))` where
`log_ret_k = ln(c/c.shift(k))` — that's a k-period log return, not the
sum of k 1-period returns. Lo-MacKinlay's VR uses overlapping k-period
returns vs. (k × 1-period variance). The formula is correct under that
definition. OK.

### P2-19 — `intraday_seasonality.py:110-118` hour vote = BUY ignoring trend

`_hour_alpha_vote` returns "BUY" for high-multiplier hours regardless of
trend direction. Then the composite uses `trend_vote` for actual
direction at line 192. But the reported `sub_signals["hour_alpha"]` is
"BUY" — misleading if downstream tooling treats sub-signals as
independent direction votes. Cosmetic but confusing.

### P2-20 — `williams_vix_fix.py` structural BUY bias

3 of 4 sub-signals are BUY-or-HOLD only (bb_spike, percentile,
rsi_confirm), 1 is SELL-or-HOLD (complacency). The composite will lean
BUY in any volatility spike, never SELL until extreme complacency.
Documented as bottom-detector, so it's by-design — but accuracy gating
will need to recognize this structural asymmetry.

### P2-21 — `gold_overnight_bias.py:118-141` fix proximity is BUY-only

`_fix_proximity_vote` returns BUY for both AM-fix-proximity and PM-fix-
proximity. Combined with `_session_phase_vote` which switches direction
between sessions, the proximity vote frequently conflicts with the
session vote during transition. Net effect: spurious BUY bias near fix
times.

### P2-22 — `forecast.py:946` Lock + cache TS race window

`_last_prediction_ts[ticker] = now_mono` inside `_forecast_lock`. Between
the `should_log = ...` check and the entry write, the cache could be
flushed by another thread. Mostly fine because `should_log` is the
gating decision; missing a dedup window is at worst a duplicate log
entry, not data corruption.

### P2-23 — `futures_basis.py:13` says applies to metals

Docstring claims `BTC-USD, ETH-USD, XAU-USD, XAG-USD` applicability,
but Binance FAPI does not list spot-vs-perp basis for XAU-USD / XAG-USD
the way it does for BTCUSDT/ETHUSDT. The premium index klines for
metals will likely return empty or unusable data. Active signal but
metals branch is dead.

## P3 — Low

### P3-1 — `trend.py:418-422` ADX threshold

`_adx_di` uses ADX > 25 for BUY/SELL. Standard ADX overbought is often
40+. This makes adx_di fire on relatively weak trends. Tunable, not a
bug.

### P3-2 — `momentum.py:223-256` PPO crossover w/ deadband

PPO uses signal crossover but no deadband — bars that cross by 0.001%
fire signals. Sensitive to noise on low-vol assets.

### P3-3 — `mean_reversion.py:127-167` `_consecutive_days`

The "flat day breaks the streak" logic at line 158 is reasonable for
daily but for hourly bars flat closes occur on low-volume hours, breaking
otherwise valid trend streaks. Active signal.

### P3-4 — `volume_flow.py:324` price_up default

```python
price_up = price_change > 0 if not np.isnan(price_change) else True
```

When `price_change` is NaN, defaults to True (BUY direction). Should be
HOLD or skip the volume_rsi sub-signal entirely.

### P3-5 — `candlestick.py:131-138` consecutive bars colors counts

Single-bar dojis often have body_pct < 10% by construction even on
trending bars (high volatility but small body). Doji at the end of a
trending session is treated as reversal — but it's often just a pause.

### P3-6 — `metals_cross_asset.py:425` EPU FRED daily 4h cache

EPU is published daily but with a 1-2 day delay. The 4h cache means
fresh-cache hits return up to 4h old data, but underlying FRED data may
be 24h+ old. Stale-data risk; should log freshness.

### P3-7 — `momentum_factors.py:184-205` low_reversal needs 4 bars

`_low_reversal` requires 3 green bars + ratio<=1.05 for BUY. Three
consecutive green bars near a 52w low on hourly bars happens
non-trivially during normal trading — this fires on noise.

### P3-8 — `realized_skewness.py:60` `_compute_rolling_skewness` perf

Recomputes 252-bar rolling skew per sub-signal call (called from 2
sub-signals). For 5 active tickers × 7 timeframes × every cycle = many
calls. CPU cost. Disabled module.

### P3-9 — `news_event.py:96` `_persist_headlines` writes on every call

Persists headlines to `data/headlines_latest.json` on every signal
invocation. With 5 tickers × every cycle, this is 5 writes per cycle.
The disk I/O is atomic, but on slow disks could pile up.

### P3-10 — `claude_fundamental.py:179` yfinance Earnings Date parsing

`dates[0] if isinstance(dates, list) else dates` — assumes dates is
sortable when it's a list but yfinance returns datetime objects in
mixed order across versions. Should explicit-sort.

### P3-11 — `forecast.py:415` ThreadPoolExecutor inside GPU gate

`_run_chronos_inner` creates a ThreadPoolExecutor inside the GPU gate
to enforce timeout. The executor spawns a new thread per call — over
many cycles this is many short-lived threads. Acceptable but
non-ideal.

### P3-12 — `cot_positioning.py:51` CFTC URL hardcoded

```python
_CFTC_LEGACY_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
```

CFTC has previously moved endpoints. No fallback. Module already-degrades
to HOLD on fetch failure, so OK.

### P3-13 — `ovx_metals_spillover.py` OVX may be discontinued

CBOE discontinued OVX in late 2024 (per public records). yfinance
`^OVX` may return empty silently. Module degrades to HOLD correctly,
but should log explicit "OVX discontinued" warning. Disabled module.

### P3-14 — `intraday_seasonality.py:42-58` hardcoded multipliers

The hour-of-day multipliers are static literals. Markets shift over
time (e.g., crypto activity hours have shifted toward US session). No
auto-recalibration. Cosmetic.

### P3-15 — `forecast.py:807-820` `kronos_candles` fallback iterates df.iterrows()

For df length ~500 this is fine, but for an unusual large df this could
be slow. Use `.to_dict('records')` instead. Perf only.

### P3-16 — `forecast.py:48-50` `_FORECAST_MODELS_DISABLED = False` as module global

Used as a kill-switch but not configurable at runtime — must edit code
to flip. Inconsistent with the `kronos_enabled` config read at line 86.

### P3-17 — `claude_fundamental.py:418-428` portfolio holdings filter

```python
held = ", ".join(
    f"{t} {h.get('shares', 0):.2f}sh@${h.get('avg_cost_usd', 0):.2f}"
    for t, h in holdings.items()
    if h.get("shares", 0) > 0
)
```

If `shares` is a float very close to 0 (e.g., 0.0001 from rounding), it
still appears. Cosmetic.

### P3-18 — `volume_flow.py:88-92` VWAP `idx.tz_convert` on tz-aware

```python
if idx.tz is not None:
    session_id = idx.tz_convert("UTC").date
else:
    session_id = idx.date
```

`idx.tz_convert("UTC").date` returns a numpy array of dates. Works but
fragile to pandas API changes.

### P3-19 — `dxy_cross_asset.py:67` no fallback

If `get_dxy_intraday` returns None, returns HOLD immediately. No daily
fallback. Active signal — silent degradation on yfinance hiccup.

### P3-20 — `heikin_ashi.py:159-166` floating-point wick tolerance

```python
tol = 1e-10
...
if not is_green or abs(row["ha_low"] - row["ha_open"]) > tol * ha_range + tol:
    all_green_no_lower_wick = False
```

`tol * ha_range + tol` mixes absolute and relative tolerance. For
ha_range ~ 100, this becomes 1e-8 — fine. For ha_range ~ 0.0001 (e.g.,
on a forex-style scaled instrument), becomes 1e-14, effectively zero.
Inconsistent tolerance across asset scales.

## Tests missing

Across the 52 detector files I see correctness gaps that are not
covered by `tests/test_signals_*.py` (judging from the file list):

- **Direction-class invariants**: there is no test that exercises each
  detector on a SAFE-HAVEN ticker (XAU-USD) and a RISK-ON ticker
  (BTC-USD) and asserts the directions are opposite where the docstring
  says they should be. P0-5, P1-13, P2-3, P2-4 would all have been
  caught by such a test.
- **Disabled-module enable-smoke-test**: P0-3, P0-4 (the `_cached` TypeError)
  would be caught by a test that imports every signal module and
  verifies `compute_*_signal(df_with_50_rows, context={"ticker": "BTC-USD"})`
  doesn't raise.
- **Look-ahead boundary test**: feed each detector the same df twice
  (with and without the last 5 bars) and assert that historical sub-
  signals on the truncated df match the same bars on the full df.
  Catches drop-in look-ahead.
- **Structural bias detector**: feed each detector 1000 random OHLCV
  series and count the BUY/SELL/HOLD distribution. Anything with >70%
  one-direction in random data has structural bias (P1-3, P1-4, P1-5,
  P2-20, P2-21 are all candidates).
- **FVG sub-signal current-bar test** (P1-9): feed smart_money a df
  where the last bar's low has just touched a gap_low, assert that
  fvg sub-signal == "BUY" (currently fails — always HOLD).
- **`structure._highlow_breakout` near-extreme test** (P1-1): feed a df
  where close is at the 52w high and assert this is NOT a fresh BUY
  breakout (currently fires BUY).
- **`fibonacci` pivot points timeframe test** (P1-2): assert pivot
  calculation uses 24-bar (or daily resample) reference rather than
  the previous bar.
- **`claude_fundamental` failed-refresh recovery test** (P1-6): force
  `_bg_refresh` to raise, verify next call within cooldown still
  retries (currently doesn't — TS is poisoned).

## Cross-cutting observations (patterns across detectors)

1. **`iloc[-1]` on current-bar everywhere**. Every TA detector reads
   `df.close.iloc[-1]` as if it were a closed bar. In Layer 1's 60s
   loop, the bar being read may be the in-progress current minute, hour,
   or day depending on the timeframe and source. This causes "current
   bar look-ahead" — the signal acts as if the in-progress bar is
   final. Project convention accepts this since the loop reads on a
   fixed cadence, but it does mean every signal's reported sub-signal
   value will flicker as the bar evolves, especially for
   crossover-based signals (MACD zero-line cross, supertrend flip,
   golden cross). Single mention here rather than per-file.

2. **Z-score includes current value in mean/std**. Pattern repeated in
   credit_spread, cot_positioning, metals_cross_asset, mahalanobis_turb,
   futures_basis. Standard practice in finance literature is to compute
   z = (current - mean_ex_current) / std_ex_current. The bias is small
   for n=252 but real for n=20 (5% dampening). Worth a shared utility.

3. **Same direction for all asset classes**. Multiple cross-asset
   detectors (cross_asset_tsmom bond/equity subs, vix_term_structure,
   futures_basis docstring for metals) emit the same direction
   regardless of whether the target is a safe-haven or risk asset. The
   pattern in copper_gold_ratio (apply, then invert for metals) is
   the correct fix — but it's inconsistently applied.

4. **Structural BUY-only or SELL-only sub-signal families**. Calendar,
   news_event, econ_calendar, williams_vix_fix, hash_ribbons — each has
   sub-signals that can never emit one direction. The composite vote
   then has a structural lean that the accuracy gate must learn to
   recognize, but the gate only sees the *final* action, not the
   per-sub-signal one-sidedness. This bias propagates upstream.

5. **`_cached` signature inconsistency**. `mahalanobis_turbulence.py:99`
   and `complexity_gap_regime.py:92` use `_cached(key, fn, ttl=...)`
   while every other file uses `_cached(key, ttl, fn, *args)`. Both
   would TypeError. Suggests these two files were ported from a
   different code generation.

6. **Relative `config.json` path**. `credit_spread.py:285` and
   `gold_real_yield_paradox.py:265` still use relative paths after
   `cot_positioning.py:33` fixed it with `_DATA_DIR = Path(__file__)
   .resolve().parent.parent.parent / "data"`. Half-applied fix.

7. **`_get_fred_key` duplication**. The complex object/dict ternary
   appears in 4 files verbatim. Should live in `portfolio.config_utils`
   or similar.

8. **No regime/asset-class enforcement in registry**. The signal
   registry doesn't enforce that a "metals-only" detector actually
   declines stocks/crypto. Each detector self-gates via
   `if ticker not in _METALS_TICKERS: return empty`. Easy to forget.

9. **Background refresh thread can poison cache TS on failure** —
   pattern in claude_fundamental.py. Same pattern would emerge in
   forecast.py if Kronos/Chronos refresh threads existed (they don't —
   forecast runs synchronously per call, so this is unique to
   claude_fundamental).

10. **Confidence cap at 0.7 inconsistent**. Most signals cap confidence
    at 0.7 (`_MAX_CONFIDENCE = 0.7`). A few use 0.6 (calendar_seasonal),
    0.85 (vwap_zscore_mr line 105), or no cap (trend, momentum,
    volatility — return raw majority_vote confidence which is bounded
    by 1.0). Should be standardized.

11. **`np.argmax` for "periods since high" assumes contiguous data**.
    `oscillators.py:111-112` `_aroon_oscillator` uses
    `np.argmax(high_window.values)` — fine for contiguous bars but if
    there are gaps in the data (missing minutes/hours), the periods
    since high is miscounted.

12. **Most signals ignore `context["regime"]`**. Only forecast.py and a
    few others read it. Despite CLAUDE.md noting "regime penalties:
    ranging 0.75x, high-vol 0.80x", individual detectors don't apply
    these.

## Per-detector verdicts

| file | active? | verdict | top finding |
| --- | --- | --- | --- |
| `__init__.py` | n/a | OK | docstring only |
| `calendar_seasonal.py` | active | medium concern | P1-4: `_sell_in_may` never returns SELL despite name; 5 of 8 sub-signals are BUY-only → structural BUY bias |
| `candlestick.py` | active | OK | hammer/engulfing logic correct; star pattern gap detection conservative |
| `claude_fundamental.py` | active | medium concern | P1-6: cache TS poisoned before refresh succeeds — failed refresh blocks retries for full cooldown |
| `complexity_gap_regime.py` | disabled | broken on enable | P0-4: `_cached(key, fn, ttl=...)` raises TypeError on first call |
| `copper_gold_ratio.py` | disabled | OK | Direction inversion correctly applied only for metals; raw signal otherwise |
| `cot_positioning.py` | active | OK | Z-score includes current value (minor); direction logic correct |
| `credit_spread.py` | active | minor concern | P2-2: relative `config.json` path can break CWD-sensitive callers |
| `cross_asset_tsmom.py` | disabled | broken | P0-5: bond/equity sub-signals ignore target asset class — same direction for gold and BTC |
| `crypto_evrp.py` | disabled | broken | P0-1 + P0-2: dead-code rv_hist, direction inverted vs docstring |
| `crypto_macro.py` | active | OK | options gravity, P/C, gold rotation logic consistent with thesis |
| `cubic_trend_persistence.py` | active | OK | Cubic regression for trend persistence, well-implemented |
| `drift_regime_gate.py` | active | by-design contrarian | Mean-reversion at extremes will SELL into uptrends by design |
| `dxy_cross_asset.py` | active | OK | DXY inverse correlation correctly implemented for metals+crypto |
| `econ_calendar.py` | active | medium concern | P1-5: 4 SELL-only sub-signals vs 1 BUY-only → structural SELL bias |
| `fibonacci.py` | active | medium concern | P1-2: pivot points use previous bar not previous day |
| `forecast.py` | active | minor concern | P1-8: kronos_1h doubled in vote pool; P2-16: kronos/chronos different gpu_gate keys |
| `futures_basis.py` | active | OK | crypto-correct, but metals docstring claim (P2-23) misleading |
| `futures_flow.py` | active | OK | OI/funding/LS logic consistent; asymmetric oi_divergence is acceptable |
| `gold_overnight_bias.py` | active | minor concern | P2-21: fix_proximity always BUY creates bias near transitions |
| `gold_real_yield_paradox.py` | disabled | broken | P2-3, P2-4: asymmetric paradox + likely-inverted SELL direction |
| `hash_ribbons.py` | active | OK | BTC-only BUY-only by design |
| `heikin_ashi.py` | active | OK | HA trend/doji/color/HMA/Alligator/Elder/TTM all reasonable |
| `hurst_regime.py` | disabled | broken | P1-12: hurst_momentum returns BUY for rising Hurst with no direction basis |
| `intraday_seasonality.py` | active | minor concern | P2-19: hour_alpha sub-signal returns "BUY" but actual direction is from trend |
| `macro_regime.py` | active | OK | DXY threshold (0.3%) assumes change_5d_pct is in percent — verify upstream |
| `mahalanobis_turbulence.py` | disabled | broken on enable | P0-3: `_cached` arg order TypeError |
| `mean_reversion.py` | active | OK | RSI(2/3), IBS, gap-fill (with explicit widening guard A-SM-1), BB %B, half-life all consistent |
| `metals_cross_asset.py` | active | minor concern | P1-14: no None-guard on `get_all_cross_asset_intraday()` return |
| `metals_vrp.py` | active | OK | GVZ-RV spread contrarian, metals-only |
| `momentum.py` | active | OK | RSI div, Stochastic (P2-13 restrictive gate), CCI, Williams %R, ROC, PPO, BBP all standard |
| `momentum_factors.py` | active | OK | TSMOM, ROC-vol-scaled, high prox, low reversal, consec bars, acceleration, vol-weighted |
| `network_momentum.py` | disabled | broken | P1-11: corr_regime votes BUY without direction basis when net_div is HOLD |
| `news_event.py` | active | medium concern | P1-3: keyword_severity SELL-only, P2-6: cut defaults negative, P2-7: thesis alignment never contradicts |
| `orderbook_flow.py` | disabled | OK | per-sub-signal logic clean, OFI z-score vs absolute fallback acceptable |
| `oscillators.py` | disabled | minor concern | P2-14: Coppock SELL is invented; otherwise OK |
| `ovx_metals_spillover.py` | disabled | likely-dead-data | P3-13: OVX may be discontinued by CBOE |
| `realized_skewness.py` | disabled | minor concern | P3-8: 252-bar rolling skew recomputed twice; otherwise OK |
| `residual_pair_reversion.py` | active | OK | OLS rolling regression + z-score; pair map sensible |
| `shannon_entropy.py` | active | OK | Entropy-based composite with reasonable confidence assembly |
| `smart_money.py` | disabled | broken | P1-9: FVG fill check includes current bar, so BUY on freshly filling gap never fires |
| `statistical_jump_regime.py` | disabled | OK | jump detection + persistence; regime change logic clean |
| `structure.py` | active | broken | P1-1: high/low_breakout votes BUY near 52w HIGH (resistance) and SELL near 52w LOW (support) |
| `treasury_risk_rotation.py` | active | medium concern | P1-15: docstring direction (steepening vs flattening) appears inverted from TLT/IEF mechanics |
| `trend.py` | active | OK | Golden cross, ribbon, vs MA200, supertrend, SAR, ichimoku, ADX — all standard |
| `vix_term_structure.py` | disabled | broken | P1-13: direction ignores asset class — same SELL for gold and BTC in backwardation |
| `vol_ratio_regime.py` | active | OK | GK/CC + VR + ER regime classifier; directional logic in main function |
| `volatility.py` | active | minor concern | P1-10: GARCH sub-signal conflates volatility expansion with directional SELL |
| `volume_flow.py` | active | OK | OBV/VWAP/AD/CMF/MFI/Volume-RSI; VWAP session reset relies on DatetimeIndex |
| `vwap_zscore_mr.py` | active | OK | VWAP z-score MR with slope context and volume confirmation |
| `williams_vix_fix.py` | active | by-design BUY-biased | 3 of 4 sub-signals are BUY-or-HOLD only; calibrate downstream |
| `xtrend_equity_spillover.py` | disabled | OK | SPY/QQQ RSI/MACD/trend with safe-haven inversion |
