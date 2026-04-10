# Agent Adversarial Review: signals-modules

**Agent**: feature-dev:code-reviewer
**Subsystem**: signals-modules (10,949 lines, 24 files in portfolio/signals/)
**Duration**: ~253 seconds
**Findings**: 16 (2 P0, 8 P1, 6 P2)

---

## P0 Findings

### A-SM-1: Gap-Fill Signal Fires BUY on Continuing Gap-Down — Inverted Logic [P0]
- **File**: `portfolio/signals/mean_reversion.py:212-226`
- **Description**: `_gap_fill()` fires BUY on gap-down when `fill_pct < 0.3`. But for a gap-down where price CONTINUES DOWN (not recovering), `fill_amount > 0` and `gap_distance < 0`, so `fill_pct < 0` (negative). Negative is always < 0.3, so the gate passes, and `gap_pct < 0` returns BUY — a false BUY during a crash.
- **Impact**: During gap-down events (bad news on metals/crypto), the signal fires BUY when price is falling, not recovering. Opposite of mean-reversion intent.
- **Fix**: Add `if fill_pct < 0: return ..., "HOLD"` to reject cases where price moved further against the gap.

### A-SM-2: GARCH Missing From _empty_result Schema [P0]
- **File**: `portfolio/signals/volatility.py:420-445`
- **Description**: `_empty_result()` returns sub_signals with 6 keys but no `"garch"`. Happy path adds it at line 401. Schema inconsistency between normal and fallback paths.
- **Impact**: Downstream schema assertions or diagnostic tools get inconsistent data.
- **Fix**: Add `"garch": "HOLD"` to `_empty_result()`.

---

## P1 Findings

### A-SM-3: FOMC Drift vs FOMC Proximity Direct Contradiction [P1]
- **File**: `portfolio/signals/calendar_seasonal.py:274` vs `portfolio/signals/macro_regime.py:234`
- **Description**: Calendar says BUY 1-2 days pre-FOMC (drift theory). Macro_regime says SELL ≤2 days pre-FOMC (risk-off). They cancel each other 32 days/year.
- **Fix**: Decide which theory to follow. Gate one off for FOMC proximity.

### A-SM-4: Historical Volatility Uses sqrt(365), GARCH Uses sqrt(252) — Ratio Meaningless [P1]
- **File**: `portfolio/signals/volatility.py:159,263`
- **Description**: `_historical_volatility` annualizes with sqrt(365), `_garch_signal` with sqrt(252). The ratio comparison (`garch_vol / hist_vol`) is biased by the sqrt(365/252) ≈ 1.204 factor, causing systematic compression/BUY bias.
- **Fix**: Standardize both to sqrt(252) or make annualization factor a module constant.

### A-SM-5: FOMC Dates Unsorted — First-Match Short-Circuit May Skip Nearest Date [P1]
- **File**: `portfolio/signals/calendar_seasonal.py:259-280`
- **Description**: Iterates `_FOMC_ANNOUNCEMENT_DATES` and returns on first match. If unsorted, a farther date could match before the nearest FOMC date.
- **Fix**: Pre-sort dates or use `min()` to find nearest upcoming date.

### A-SM-6: Donchian Upper Includes Current Bar — Lookback Bias [P1]
- **File**: `portfolio/signals/volatility.py:186-225`
- **Description**: `high.rolling(20).max().iloc[-1]` includes the current bar's high. Comparing `current_high >= current_upper` is trivially true when current bar IS the high. Structure.py correctly excludes current bar.
- **Fix**: Use `high.iloc[-21:-1].max()` (prior 20 bars only).

### A-SM-7: "cut" Keyword Counted as Positive — Job/Budget Cuts Score as BUY [P1]
- **File**: `portfolio/signals/news_event.py:255-265`
- **Description**: Positive keyword list includes "cut". "Rate cut" → pos, "guidance cut" → neg, but "job cut", "budget cut", "production cut" → all score as positive. During market stress, negative headlines generate false BUY signals.
- **Fix**: Remove "cut" from positive list. Handle "rate cut" as a separate two-word phrase.

### A-SM-8: Golden Cross NaN Guard Missing for iloc[-2] [P1]
- **File**: `portfolio/signals/trend.py:55-62`
- **Description**: Verifies `sma50.iloc[-1]` is non-NaN but NOT `sma50.iloc[-2]`. NaN comparison returns False, causing false crossover detection on data with gaps.
- **Fix**: Check `pd.isna(sma50.iloc[-2])` before crossover comparison.

### A-SM-9: BB Breakout + Keltner Structural Double-Counting [P1]
- **File**: `portfolio/signals/volatility.py:85-148`
- **Description**: Both vote BUY when price breaks above their upper bands. Since KC is wider than BB (TTM squeeze condition), price above BB is almost always above KC too. Same event counted twice.
- **Fix**: Make mutually exclusive or collapse into a single "band breakout" sub-signal.

### A-SM-10: futures_flow._oi_trend Uses Truthiness for Zero Guard [P1]
- **File**: `portfolio/signals/futures_flow.py:65`
- **Description**: `if price_start and price_end > price_start:` uses truthiness of float. `price_start == 0.0` silently returns HOLD. Inconsistent with explicit NaN guard above it.
- **Fix**: Use `if price_start != 0 and price_end > price_start:`.

---

## P2 Findings

### A-SM-11: Cumulative VWAP Never Resets — Stale for 24/7 Markets [P2]
- **File**: `portfolio/signals/volume_flow.py:61-69`
- **Description**: VWAP computed from beginning of entire DataFrame (days of data). No session reset for crypto/metals 24/7 markets. Creates permanent directional bias.

### A-SM-12: compute_calendar_signal Silent HOLD When "time" Column Absent [P2]
- **File**: `portfolio/signals/calendar_seasonal.py:371-373`
- **Description**: Requires `"time"` column in DataFrame. If absent (common case for OHLCV frames), entire signal always returns HOLD silently.

### A-SM-13: Fixed GARCH Parameters — Not Fitted, Misleading Name [P2]
- **File**: `portfolio/signals/volatility.py:253-254`
- **Description**: Uses hardcoded alpha=0.10, beta=0.85. No MLE fitting. Essentially EWMA-on-squared-returns, not true GARCH.

### A-SM-14: January Double-Count Between sell_in_may + january_effect [P2]
- **File**: `portfolio/signals/calendar_seasonal.py:157,183`
- **Description**: Both sub-signals vote BUY in January, inflating confidence from same seasonal factor.

### A-SM-15: Supply/Demand Zone Expansion Always Equals Full Zone Width [P2]
- **File**: `portfolio/signals/smart_money.py:373-379`

### A-SM-16: FVG Detection Inner Loop O(n²) Performance [P2]
- **File**: `portfolio/signals/smart_money.py:207-237`
