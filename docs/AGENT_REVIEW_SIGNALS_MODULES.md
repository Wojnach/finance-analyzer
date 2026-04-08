# Adversarial Review: signals-modules (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08

---

## HIGH

### HS1. structure.py: 52-week high/low uses ENTIRE dataset, not a windowed period [85% confidence]
**File**: `portfolio/signals/structure.py:64-80`

`_highlow_breakout` does `high.max()` and `low.min()` over the entire DataFrame.
If 2000 candles are passed, the "period high/low" is the all-time extremes. For crypto
with a prior ATH, the BUY condition (within 2% of period high) becomes nearly impossible.
This permanently SELL-biases the sub-signal.

**Fix**: Cap lookback: `window = df.iloc[-252:]` before computing period_high/low.

### HS2. smart_money.py: `dropna(how="all")` allows partial-NaN rows through [85% confidence]
**File**: `portfolio/signals/smart_money.py:462`

`df.dropna(subset=required_cols, how="all")` only drops rows where ALL OHLCV cols are NaN.
A row with `close=NaN` but valid `high/low` passes through, propagating NaN into BOS,
FVG, and supply/demand calculations. `NaN > value` evaluates False silently.

**Fix**: Change `how="all"` to `how="any"`.

### HS3. volume_flow.py: NaN price direction defaults to BUY — long bias on bad data [83% confidence]
**File**: `portfolio/signals/volume_flow.py:288-289`

`price_up = price_change > 0 if not np.isnan(price_change) else True` — defaults to `True`
when price direction is unknown. Volume RSI emits BUY on strong volume regardless of
actual direction.

**Fix**: Default to `None`, return HOLD when direction unknown.

### HS4. trend.py: golden cross uses positional `iloc[-2]` without NaN guard [82% confidence]
**File**: `portfolio/signals/trend.py:55-61`

`sma50.iloc[-2]` may be NaN when data has gaps. The `valid.sum() >= 2` guard ensures 2
non-NaN values exist somewhere, but doesn't guarantee `iloc[-2]` is one of them.
`NaN > NaN` evaluates False, making `prev_above = False` → spurious golden cross signal.

**Fix**: Use `sma50[sma50.notna()].iloc[-2]` for prior-bar comparison.

### HS5. volatility.py: `_empty_result` omits "garch" from sub_signals [80% confidence]
**File**: `portfolio/signals/volatility.py:420-444`

Success path returns 7 sub_signals (including garch). Fallback `_empty_result()` returns
only 6 (no garch). Downstream code doing `result["sub_signals"]["garch"]` will KeyError
on the fallback path.

### HS6. volatility.py: `_bb_breakout` votes momentum (not mean-reversion) inside "volatility" module [80% confidence]
**File**: `portfolio/signals/volatility.py:86-90`

BUY when price > upper BB, SELL when < lower BB. This is momentum interpretation, conflicting
with `_bb_squeeze` in the same module. Both sub-signals vote in the composite, creating
contradictory votes that average out and reduce signal strength.

---

## MEDIUM

### MS1. calendar_seasonal.py: FOMC drift warning fires every loop after Dec 2027 [80% confidence]
**File**: `portfolio/signals/calendar_seasonal.py:253`

After `max(FOMC_ANNOUNCEMENT_DATES)` = `date(2027, 12, 8)`, warning fires once per ticker
per 60s cycle indefinitely. FOMC drift sub-signal permanently returns HOLD with no error.

### MS2. fibonacci.py: Pivot points use prior candle, not prior session [75% confidence]
**File**: `portfolio/signals/fibonacci.py:443-453`

Standard pivot points use prior session's H/L/C. `iloc[-2]` is just the prior candle.
For 15m candles, this produces meaningless intraday pivots with very tight levels.

### MS3. news_event.py: "cut" keyword misclassifies "job cut"/"dividend cut" as positive [72% confidence]
**File**: `portfolio/signals/news_event.py:256-264`

"cut" keyword triggers positive path first. "rate cut" (positive) is special-cased but
"job cut", "cost cut", "dividend cut" fall through as positive sentiment. Only "guidance cut"
is correctly classified as negative.

### MS4. mean_reversion.py: Seasonality detrending feedback loop can propagate NaN [60% confidence]
**File**: `portfolio/signals/mean_reversion.py:447-461`

Each step's detrended close feeds into the next step. A NaN at any step corrupts all
subsequent values silently. The `if np.isfinite()` guard skips the current bar but still
uses the (potentially corrupted) previous bar for the next iteration.

### MS5. econ_calendar.py: Timezone relabeling instead of conversion [65% confidence]
**File**: `portfolio/signals/econ_calendar.py:35`

`.replace(tzinfo=UTC)` relabels timezone without converting. CET timestamps get marked
as UTC, making event proximity off by 1-2 hours.

### MS6. heikin_ashi.py: Alligator shift delays signal by 3-8 bars [50% confidence]
**File**: `portfolio/signals/heikin_ashi.py:300-338`

Williams Alligator uses `rma().shift(N)` — by design, but means "current" values are
actually 3-8 bars old. Reduces responsiveness for intraday signals.

---

## LOW

### LS1. candlestick.py: Engulfing uses `<=` / `>=` — exact equality passes [50% confidence]
**File**: `portfolio/signals/candlestick.py:293-302`

Two identical bars would pass as bullish engulfing. Extremely rare with float prices.

### LS2. calendar_seasonal.py: US holiday dates are hardcoded month/day pairs [50% confidence]
**File**: `portfolio/signals/calendar_seasonal.py:207-218`

Memorial Day (5, 26) is correct for ~1 year. Floating holidays shift year to year.
Low impact since signal is capped at 0.6 confidence.

### LS3. forecast.py: Circuit breaker TTL of 30s may be too short for GPU recovery [40% confidence]
**File**: `portfolio/signals/forecast.py`

GPU OOM retried in 30s → hits same error. With 60s cycle and 27 tickers, some retry
before 30s expires. Minor since health logging captures the pattern.

---

## Cross-Critique: Claude Direct vs Signals-Modules Agent

### Agent found that Claude missed:
1. **HS1**: structure.py all-history high/low — complete miss (major SELL bias)
2. **HS2**: smart_money dropna(how="all") — complete miss
3. **HS3**: volume_flow NaN defaults to BUY — complete miss (long bias on bad data)
4. **HS4**: trend.py golden cross NaN — Claude noted "NaN propagation" generically but
   agent pinpointed the exact mechanism
5. **HS5**: volatility garch missing from empty result — complete miss
6. **HS6**: BB breakout momentum vs mean-reversion conflict — complete miss
7. **MS2**: Fibonacci pivot using single candle — complete miss
8. **MS3**: news "cut" keyword misclassification — complete miss

### Claude found that agent confirmed:
1. **M10/MS1**: Hardcoded calendar dates — both found, agent added FOMC warning spam detail
2. **H14**: Smart money swing detection blind spot — agent didn't re-raise this specific issue
3. **M9**: Silent HOLD on short data — agent found this implicitly in multiple modules

### Net assessment:
The agent found **6 HIGH + 6 MEDIUM + 3 LOW = 15 net-new issues** in signals-modules.
The structure.py all-history high/low (HS1) and volume_flow NaN-to-BUY (HS3) are the
most impactful — they create persistent directional bias in the signal output.

Agent was significantly stronger for this subsystem since individual indicator math
requires line-by-line reading that the broad review skipped.
