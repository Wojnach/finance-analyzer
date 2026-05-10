OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-signals-modules
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e127e-9fa8-7da0-a9a9-b19c7228f8e9
--------
user
You are an adversarial code reviewer cross-critiquing another reviewer's findings.

CONTEXT
- Codebase: Q:\finance-analyzer (autonomous trading system, Python).
- Subsystem: signals-modules
- The other reviewer (Claude Opus 4.7) audited this subsystem and produced
  the findings below.
- Your job: critique each finding by inspecting the actual source files and
  deciding whether the finding is correct.

PROCEDURE
For each finding in the list below:
1. Open the cited file(s) at the cited line(s) in the working tree.
2. Decide one of:
   - CONFIRMED: bug/issue is real, severity is reasonable.
   - PARTIAL: there is something there, but the analysis is wrong in a
     specific way (e.g., wrong line, wrong cause, wrong severity).
   - FALSE-POSITIVE: the finding is wrong; the code does not have the
     described problem. State why concretely (cite the actual code).
3. If you find a NEW issue while inspecting, list it under "New findings".

OUTPUT FORMAT (Markdown, no preamble)
## Verdicts

- [<orig severity>] <one-line restatement> — file:line
  Verdict: CONFIRMED | PARTIAL | FALSE-POSITIVE
  Reason: <one sentence citing actual code>
  (Adjustment: <if PARTIAL, what's actually wrong>)

## New findings (you, not Claude)

- [P1|P2|P3] <one-line> — file:line
  <one paragraph>

## Summary
- Confirmed: N
- Partial: N
- False-positive: N
- New from you: N

CLAUDE'S FINDINGS TO CRITIQUE:
=== BEGIN ===
# Adversarial Review: Finance Analyzer Signals Subsystem

Date: 2026-05-08
Scope: portfolio/signals/*.py (36+ modules)
Reviewer: Claude Haiku 4.5

---

## Findings

### portfolio/signals/trend.py:325-326
Ichimoku cloud uses forward-shifted SMA, enabling look-ahead bias.

Problem: senkou_a.shift(26) and senkou_b.shift(26) shift the spans 26 bars forward in time. The iloc[-1] comparison includes 26 bars of future data.

Fix: Change to shift(-26) or remove shift entirely; standard Ichimoku on live signals should use past-only lookbacks.

---

### portfolio/signals/volatility.py:57
BB squeeze threshold allows divide-by-zero when avg_width is zero or NaN.

Problem: avg_width > 0 check misses NaN values (np.nan > 0 returns False silently).

Fix: Use if avg_width > 0 and not np.isnan(avg_width).

---

### portfolio/signals/momentum.py:131
StochRSI division by zero when RSI range collapses.

Problem: When all RSI values in a window are identical, denom=0 becomes NaN; next bar can flip from NaN to valid number without smoothing.

Fix: If denom.isna().all(), return HOLD. Or add minimum range guard: denom.clip(lower=1.0).

---

### portfolio/signals/momentum.py:201
ROC denominator uses shifted (past) close, not current; conflates level with acceleration.

Problem: close.shift(period) creates self-referential lag. Lines 212, 216 treat ROC value as acceleration indicator.

Fix: Verify ROC intent. If acceleration needed, compute roc_now - roc_prev, not ROC value alone.

---

### portfolio/signals/calendar_seasonal.py:256
FOMC dates cache max() at module import; all future dates eventually become stale.

Problem: _FOMC_MAX_DATE computed once at import. Warning fires on every call after last FOMC date passes, but signal still votes. No halt, no escalation to critical_errors.jsonl.

Fix: Check FOMC list at compute time, not import time. Auto-extend via Fed calendar API.

---

### portfolio/signals/econ_calendar.py:127
post_event_relief BUY fires when recent event + next event >24h away; overlapping relief/risk windows.

Problem: If two high-impact events are 20h apart, relief window for event#1 persists while event#2 enters pre_event_risk zone. Silent conflict in composite vote.

Fix: Suppress relief when next event <24h away.

---

### portfolio/signals/metals_cross_asset.py:150
FRED fetch returns newest-first (descending order), but z-score assumes oldest-to-newest.

Problem: sort_order=desc, but values list is appended in order received; z-score lookback window is reversed chronologically.

Fix: Reverse values list after fetch: values = list(reversed(values)).

---

### portfolio/signals/cot_positioning.py:33
Deep context load has no staleness check; reflects week-old COT data indefinitely.

Problem: _load_deep_context() loads JSON with no timestamp validation. No warning if file is >1 week old (CFTC release cadence).

Fix: Check file mtime; if >7 days old, return None and fall back to API.

---

### portfolio/signals/credit_spread.py:159
OAS z-score uses current as values[0] included in historical mean; self-referential bias.

Problem: history = values[:lookback] includes current (values[0]); mean/stddev includes the value being scored.

Fix: Use history = values[1:1+lookback] to exclude current from historical sample.

---

### portfolio/signals/claude_fundamental.py:149
Earnings calendar cache TTL is 12h; earnings dates don't change daily. Inefficient yfinance calls.

Problem: _EARNINGS_CACHE_TTL = 43200 calls yfinance every 12h for all stocks; scales as 5N requests/day.

Fix: Increase TTL to 86400 (24h) or 604800 (7d); implement change-detection.

---

### portfolio/signals/forecast.py:37
Forecast TTL is 5 minutes; Chronos predictions stale in 12h/24h horizon signals.

Problem: _FORECAST_TTL = 300 means 24h forecast computed at 10:00 is reused at 10:04 (still 23h50m remaining).

Fix: Implement per-horizon TTL: 1h=5min, 3h=15min, 24h=60min.

---

### portfolio/signals/news_event.py
TICKER_SECTORS mapping hardcoded at module load; stale on corporate restructures.

Problem: Sector map is never refreshed. On M{{CLAUDE_REVIEW}}A or spinoff, econ_calendar fires pre_event_risk on wrong tickers.

Fix: Load TICKER_SECTORS at compute time from config/API, not import time.

---

### portfolio/signals/forecast.py:789
Forecast candles cached but never validated for stale OHLCV data.

Problem: _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv) has no check that last bar is current. Stale bar causes wrong direction prediction.

Fix: Validate df.iloc[-1]["time"] >= datetime.now(UTC) - timedelta(minutes=60); return None if stale.

---

### portfolio/signals/claude_fundamental.py:94
Cooldown check uses time.time() (wall-clock), not monotonic; vulnerable to NTP adjustments.

Problem: If system time steps backward 30 seconds, cache appears expired when still warm.

Fix: Use time.monotonic() for all cooldown checks.

---

### portfolio/signals/volatility.py:36
_atr() uses EMA, not RMA (Wilder smoothing); ATR values ~30% lower than TA-Lib standard.

Problem: tr.ewm(span=period) uses α=2/(period+1). Wilder's RMA uses α=1/period. EMA-based ATR is more reactive, channels narrow faster.

Fix: Replace with RMA from signal_utils or implement Wilder's formula.

---

### portfolio/signals/momentum.py:110
Stochastic %K threshold <20 for BUY is tight; no volume confirmation.

Problem: Oversold <20 is 1-2 bars of extreme moves. On BTC/XAG, 15m bars swing 1-3% frequently. Crossover-only is noise.

Fix: Add volume confirmation (vol > 1.2x avg) or tighten threshold to <10.

---

### portfolio/signals/structure.py
No gap detection; high/low breakout ignores overnight discontinuity (gaps 5-10% on earnings).

Problem: Assume continuous bars. Gap breakouts are catalyzed by off-market news, not trader conviction.

Fix: Add gap detection; ignore structure signals in gap bars.

---

### portfolio/signals/smart_money.py, oscillators.py, orderbook_flow.py (disabled)
Dead code still imported by signal_registry; consumes CPU on import despite being disabled.

Problem: 19 disabled modules ~3000 LOC total. AST parsing/compilation overhead ~2s per startup, every 60s loop.

Fix: Move disabled modules to portfolio/signals/_disabled/ and exclude from import loop; or add lazy-load flag.

---

### portfolio/signals/treasury_risk_rotation.py
Signal added 2026-05-07 with no backward compatibility test or validation data.

Problem: Added mid-session with zero historical accuracy data. If misprogrammed, contributes noise for 7 days before accuracy catches it.

Fix: Add 30-day warm-up shadow mode; new signals vote HOLD until 30 samples logged and accuracy >47%.

---

### portfolio/signals/realized_skewness.py, statistical_jump_regime.py, hurst_regime.py (disabled)
Feature-engineered indicators with >3-month stale "pending validation" label.

Problem: Added Apr 2026, still marked pending. 3+ weeks with no status update or removal. Consume review overhead.

Fix: Decide: enable with 30-day shadow or archive to _disabled/. Update DISABLED_SIGNALS with resolution timestamp.

---

Summary: 8 bugs (look-ahead, staleness, NaN, cache semantics), 8 risks (edge cases, validation, dead code, thresholds), 4 pending (new signals, maturity, design sync).
=== END ===

Use git/grep/read tools as needed. Be ruthless and cite code. Do not soften.

codex
I’m verifying each cited finding against the working tree, starting by reading the referenced files and surrounding logic rather than assuming the other review is accurate.
  → Inspect each cited file/line and capture the relevant code paths
  • Classify each finding as confirmed, partial, or false-positive with code-based reasons
  • Scan for any materially new issues encountered during inspection and draft the final verdict summary
codex
I’ve moved from the findings list to the source itself; next I’m checking each cited line in context and grouping repeats by file so the verdicts are grounded in the actual implementation.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files portfolio/signals' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git status --short' in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-10T15:26:12.085359Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-signals-modules'
'Q:/finance-analyzer/.worktrees/adv-signals-modules/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-signals-modules

 exited 1 in 529ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-signals-modules'
'Q:/finance-analyzer/.worktrees/adv-signals-modules/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-signals-modules

 succeeded in 547ms:
portfolio/signals\__init__.py
portfolio/signals\xtrend_equity_spillover.py
portfolio/signals\williams_vix_fix.py
portfolio/signals\vol_ratio_regime.py
portfolio/signals\volume_flow.py
portfolio/signals\volatility.py
portfolio/signals\vix_term_structure.py
portfolio/signals\trend.py
portfolio/signals\treasury_risk_rotation.py
portfolio/signals\structure.py
portfolio/signals\statistical_jump_regime.py
portfolio/signals\smart_money.py
portfolio/signals\shannon_entropy.py
portfolio/signals\residual_pair_reversion.py
portfolio/signals\realized_skewness.py
portfolio/signals\ovx_metals_spillover.py
portfolio/signals\oscillators.py
portfolio/signals\orderbook_flow.py
portfolio/signals\news_event.py
portfolio/signals\network_momentum.py
portfolio/signals\momentum_factors.py
portfolio/signals\momentum.py
portfolio/signals\metals_cross_asset.py
portfolio/signals\mean_reversion.py
portfolio/signals\mahalanobis_turbulence.py
portfolio/signals\macro_regime.py
portfolio/signals\hurst_regime.py
portfolio/signals\heikin_ashi.py
portfolio/signals\hash_ribbons.py
portfolio/signals\gold_real_yield_paradox.py
portfolio/signals\futures_flow.py
portfolio/signals\futures_basis.py
portfolio/signals\forecast.py
portfolio/signals\fibonacci.py
portfolio/signals\econ_calendar.py
portfolio/signals\dxy_cross_asset.py
portfolio/signals\drift_regime_gate.py
portfolio/signals\crypto_macro.py
portfolio/signals\crypto_evrp.py
portfolio/signals\crypto_cross_asset.py
portfolio/signals\cross_asset_tsmom.py
portfolio/signals\credit_spread.py
portfolio/signals\cot_positioning.py
portfolio/signals\copper_gold_ratio.py
portfolio/signals\complexity_gap_regime.py
portfolio/signals\claude_fundamental.py
portfolio/signals\candlestick.py
portfolio/signals\calendar_seasonal.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/trend.py'; Start=300; End=340},
  @{Path='portfolio/signals/volatility.py'; Start=1; End=120},
  @{Path='portfolio/signals/momentum.py'; Start=90; End=240}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/calendar_seasonal.py'; Start=1; End=320},
  @{Path='portfolio/signals/econ_calendar.py'; Start=90; End=170}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/metals_cross_asset.py'; Start=120; End=220},
  @{Path='portfolio/signals/cot_positioning.py'; Start=1; End=140},
  @{Path='portfolio/signals/credit_spread.py'; Start=130; End=210}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/claude_fundamental.py'; Start=1; End=230},
  @{Path='portfolio/signals/forecast.py'; Start=1; End=120},
  @{Path='portfolio/signals/forecast.py'; Start=740; End=830}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 723ms:

>>> portfolio/signals/trend.py:300-340
 300: 
 301: def _ichimoku_signal(high: pd.Series, low: pd.Series,
 302:                      close: pd.Series) -> tuple[str, float, float]:
 303:     """Ichimoku Cloud: price vs Senkou Span A/B.
 304: 
 305:     Tenkan-sen (9), Kijun-sen (26), Senkou Span A, Senkou Span B (52).
 306:     Cloud is projected 26 periods ahead, but for a current-bar signal we
 307:     compare the current close against the cloud value at the current bar
 308:     (which was projected 26 bars ago).
 309: 
 310:     Returns (signal, tenkan, kijun).
 311:     """
 312:     def _midline(s: pd.Series, period: int) -> pd.Series:
 313:         rh = s.rolling(window=period, min_periods=period).max()
 314:         rl = s.rolling(window=period, min_periods=period).min()
 315:         return (rh + rl) / 2.0
 316: 
 317:     # Need at least 52 + 26 = 78 bars for a meaningful cloud
 318:     if len(close) < 78:
 319:         return "HOLD", float("nan"), float("nan")
 320: 
 321:     # Use high/low for Tenkan and Kijun (standard Ichimoku computation)
 322:     tenkan = (_rolling_high(high, 9) + _rolling_low(low, 9)) / 2.0
 323:     kijun = (_rolling_high(high, 26) + _rolling_low(low, 26)) / 2.0
 324: 
 325:     senkou_a = ((tenkan + kijun) / 2.0).shift(26)
 326:     senkou_b = ((_rolling_high(high, 52) + _rolling_low(low, 52)) / 2.0).shift(26)
 327: 
 328:     span_a = senkou_a.iloc[-1]
 329:     span_b = senkou_b.iloc[-1]
 330:     tenkan_val = tenkan.iloc[-1]
 331:     kijun_val = kijun.iloc[-1]
 332: 
 333:     if pd.isna(span_a) or pd.isna(span_b):
 334:         return "HOLD", safe_float(tenkan_val), safe_float(kijun_val)
 335: 
 336:     price = close.iloc[-1]
 337:     cloud_top = max(span_a, span_b)
 338:     cloud_bottom = min(span_a, span_b)
 339: 
 340:     if price > cloud_top:

>>> portfolio/signals/volatility.py:1-120
   1: """Composite volatility signal — 7 sub-indicators with majority voting.
   2: 
   3: Sub-indicators:
   4:     1. BB Squeeze: detects low-volatility compression and breakout release
   5:     2. BB Breakout: price closing outside Bollinger Bands
   6:     3. ATR Expansion: volatility expansion combined with price direction
   7:     4. Keltner Channel(20, 1.5): trend breakout via EMA + ATR envelope
   8:     5. Historical Volatility: 20-day realized vol trend vs price direction
   9:     6. Donchian Channel(20): high/low breakout over rolling window
  10:     7. GARCH(1,1): conditional volatility vs realized — regime detection
  11: """
  12: 
  13: from __future__ import annotations
  14: 
  15: import logging
  16: from typing import Any
  17: 
  18: import numpy as np
  19: import pandas as pd
  20: 
  21: from portfolio.signal_utils import ema, majority_vote, sma, true_range
  22: 
  23: logger = logging.getLogger(__name__)
  24: 
  25: # ---------------------------------------------------------------------------
  26: # Minimum rows required.  BB squeeze ideally uses 120-period avg width, but
  27: # degrades gracefully with fewer rows (uses whatever is available via iloc).
  28: # 50 rows provides enough warm-up for the 20-period BB + basic statistics.
  29: # ---------------------------------------------------------------------------
  30: MIN_ROWS = 50
  31: 
  32: 
  33: def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
  34:     """Average True Range (EMA-smoothed)."""
  35:     tr = true_range(high, low, close)
  36:     return tr.ewm(span=period, adjust=False).mean()
  37: 
  38: 
  39: # ---------------------------------------------------------------------------
  40: # Sub-indicator implementations
  41: # ---------------------------------------------------------------------------
  42: 
  43: def _bb_squeeze(close: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series,
  44:                 bb_middle: pd.Series) -> tuple[str, dict]:
  45:     """BB Squeeze: width < 0.5 * 120-period avg width => squeeze ON.
  46: 
  47:     On release (width expands back) + price above upper => BUY,
  48:     price below lower => SELL.  During squeeze => HOLD.
  49:     """
  50:     bb_width = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
  51:     current_width = bb_width.iloc[-1]
  52: 
  53:     # Use up to 120 periods for the average; fall back to whatever is available
  54:     lookback = min(120, len(bb_width))
  55:     avg_width = bb_width.iloc[-lookback:].mean()
  56: 
  57:     squeeze_on = current_width < 0.5 * avg_width if avg_width > 0 else False
  58: 
  59:     # Check the previous bar to detect squeeze *release*
  60:     if len(bb_width) >= 2:
  61:         prev_width = bb_width.iloc[-2]
  62:         prev_squeeze = prev_width < 0.5 * avg_width if avg_width > 0 else False
  63:     else:
  64:         prev_squeeze = False
  65: 
  66:     price = close.iloc[-1]
  67:     upper = bb_upper.iloc[-1]
  68:     lower = bb_lower.iloc[-1]
  69: 
  70:     action = "HOLD"
  71:     if squeeze_on:
  72:         # Still compressed -- wait
  73:         action = "HOLD"
  74:     elif prev_squeeze and not squeeze_on:
  75:         # Squeeze just released
  76:         if price > upper:
  77:             action = "BUY"
  78:         elif price < lower:
  79:             action = "SELL"
  80:     # If no squeeze context at all, remain HOLD
  81: 
  82:     indicators = {"bb_width": float(current_width), "bb_squeeze_on": bool(squeeze_on)}
  83:     return action, indicators
  84: 
  85: 
  86: def _bb_breakout(close: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series) -> str:
  87:     """Price close above upper BB => BUY, below lower => SELL, else HOLD."""
  88:     price = close.iloc[-1]
  89:     if price > bb_upper.iloc[-1]:
  90:         return "BUY"
  91:     elif price < bb_lower.iloc[-1]:
  92:         return "SELL"
  93:     return "HOLD"
  94: 
  95: 
  96: def _atr_expansion(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple[str, dict]:
  97:     """ATR(14) > 1.5x its 20-period SMA => expansion.
  98: 
  99:     Expansion + price up => BUY, expansion + price down => SELL.
 100:     """
 101:     atr_series = _atr(high, low, close, 14)
 102:     atr_avg = sma(atr_series, 20)
 103: 
 104:     current_atr = atr_series.iloc[-1]
 105:     current_avg = atr_avg.iloc[-1]
 106: 
 107:     if np.isnan(current_atr) or np.isnan(current_avg) or current_avg == 0:
 108:         return "HOLD", {"atr": float(current_atr) if not np.isnan(current_atr) else 0.0,
 109:                         "atr_avg": float(current_avg) if not np.isnan(current_avg) else 0.0}
 110: 
 111:     expansion = current_atr > 1.5 * current_avg
 112: 
 113:     action = "HOLD"
 114:     if expansion:
 115:         # Determine price direction from recent closes
 116:         price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0.0
 117:         if price_change > 0:
 118:             action = "BUY"
 119:         elif price_change < 0:
 120:             action = "SELL"

>>> portfolio/signals/momentum.py:90-240
  90:     highest_high = high.rolling(window=k_period).max()
  91:     denom = highest_high - lowest_low
  92:     raw_k = 100.0 * (close - lowest_low) / denom.replace(0, np.nan)
  93:     k = raw_k.rolling(window=smooth_k).mean()
  94:     d = k.rolling(window=d_period).mean()
  95: 
  96:     k_val = k.iloc[-1]
  97:     d_val = d.iloc[-1]
  98: 
  99:     if np.isnan(k_val) or np.isnan(d_val):
 100:         return float("nan"), float("nan"), "HOLD"
 101: 
 102:     # Need at least two values to detect a cross
 103:     k_prev = k.iloc[-2] if len(k) >= 2 else np.nan
 104:     d_prev = d.iloc[-2] if len(d) >= 2 else np.nan
 105: 
 106:     if np.isnan(k_prev) or np.isnan(d_prev):
 107:         return k_val, d_val, "HOLD"
 108: 
 109:     # Bullish cross: %K crosses above %D in oversold zone
 110:     if k_prev <= d_prev and k_val > d_val and d_val < 20:
 111:         return k_val, d_val, "BUY"
 112: 
 113:     # Bearish cross: %K crosses below %D in overbought zone
 114:     if k_prev >= d_prev and k_val < d_val and d_val > 80:
 115:         return k_val, d_val, "SELL"
 116: 
 117:     return k_val, d_val, "HOLD"
 118: 
 119: 
 120: def _stochasticrsi(close: pd.Series, period: int = 14) -> tuple[float, str]:
 121:     """Stochastic RSI.
 122: 
 123:     Returns (stoch_rsi_value, signal).
 124:     StochRSI > 0.8 = overbought (SELL).
 125:     StochRSI < 0.2 = oversold (BUY).
 126:     """
 127:     rsi_values = rsi(close, period)
 128:     rsi_min = rsi_values.rolling(window=period).min()
 129:     rsi_max = rsi_values.rolling(window=period).max()
 130:     denom = rsi_max - rsi_min
 131:     stoch_rsi = (rsi_values - rsi_min) / denom.replace(0, np.nan)
 132: 
 133:     val = stoch_rsi.iloc[-1]
 134:     if np.isnan(val):
 135:         return float("nan"), "HOLD"
 136: 
 137:     if val < 0.2:
 138:         return val, "BUY"
 139:     if val > 0.8:
 140:         return val, "SELL"
 141:     return val, "HOLD"
 142: 
 143: 
 144: def _cci(high: pd.Series, low: pd.Series, close: pd.Series,
 145:          period: int = 20) -> tuple[float, str]:
 146:     """Commodity Channel Index.
 147: 
 148:     Returns (cci_value, signal).
 149:     CCI > 100 = overbought (SELL).
 150:     CCI < -100 = oversold (BUY).
 151:     """
 152:     tp = (high + low + close) / 3.0
 153:     tp_sma = sma(tp, period)
 154:     mean_dev = tp.rolling(window=period).apply(
 155:         lambda x: np.mean(np.abs(x - x.mean())), raw=True
 156:     )
 157:     cci_series = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))
 158: 
 159:     val = cci_series.iloc[-1]
 160:     if np.isnan(val):
 161:         return float("nan"), "HOLD"
 162: 
 163:     if val < -100:
 164:         return val, "BUY"
 165:     if val > 100:
 166:         return val, "SELL"
 167:     return val, "HOLD"
 168: 
 169: 
 170: def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
 171:                 period: int = 14) -> tuple[float, str]:
 172:     """Williams %R.
 173: 
 174:     Returns (williams_r_value, signal).
 175:     %R > -20 = overbought (SELL).
 176:     %R < -80 = oversold (BUY).
 177:     """
 178:     highest_high = high.rolling(window=period).max()
 179:     lowest_low = low.rolling(window=period).min()
 180:     denom = highest_high - lowest_low
 181:     wr = -100.0 * (highest_high - close) / denom.replace(0, np.nan)
 182: 
 183:     val = wr.iloc[-1]
 184:     if np.isnan(val):
 185:         return float("nan"), "HOLD"
 186: 
 187:     if val < -80:
 188:         return val, "BUY"
 189:     if val > -20:
 190:         return val, "SELL"
 191:     return val, "HOLD"
 192: 
 193: 
 194: def _rate_of_change(close: pd.Series, period: int = 12) -> tuple[float, str]:
 195:     """Rate of Change with acceleration check.
 196: 
 197:     Returns (roc_value, signal).
 198:     ROC > 0 with acceleration (ROC increasing) = BUY.
 199:     ROC < 0 with deceleration (ROC decreasing) = SELL.
 200:     """
 201:     roc = 100.0 * (close - close.shift(period)) / close.shift(period).replace(0, np.nan)
 202: 
 203:     val = roc.iloc[-1]
 204:     if np.isnan(val) or len(roc.dropna()) < 2:
 205:         return float("nan"), "HOLD"
 206: 
 207:     prev = roc.iloc[-2]
 208:     if np.isnan(prev):
 209:         return val, "HOLD"
 210: 
 211:     # ROC positive and accelerating
 212:     if val > 0 and val > prev:
 213:         return val, "BUY"
 214: 
 215:     # ROC negative and decelerating (becoming more negative)
 216:     if val < 0 and val < prev:
 217:         return val, "SELL"
 218: 
 219:     return val, "HOLD"
 220: 
 221: 
 222: def _ppo(close: pd.Series, fast: int = 12, slow: int = 26,
 223:          signal_period: int = 9) -> tuple[float, float, str]:
 224:     """Percentage Price Oscillator with signal line.
 225: 
 226:     Returns (ppo_value, ppo_signal_value, signal).
 227:     PPO crosses above signal = BUY.
 228:     PPO crosses below signal = SELL.
 229:     """
 230:     ema_fast = ema(close, fast)
 231:     ema_slow = ema(close, slow)
 232:     ppo_line = 100.0 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
 233:     signal_line = ema(ppo_line, signal_period)
 234: 
 235:     ppo_val = ppo_line.iloc[-1]
 236:     sig_val = signal_line.iloc[-1]
 237: 
 238:     if np.isnan(ppo_val) or np.isnan(sig_val):
 239:         return float("nan"), float("nan"), "HOLD"
 240: 

 succeeded in 727ms:

>>> portfolio/signals/calendar_seasonal.py:1-320
   1: """Calendar-based and seasonal trading signals.
   2: 
   3: Combines eight sub-indicators into a majority-vote composite:
   4:   1. Day-of-Week Effect       (Monday=SELL, Friday=BUY)
   5:   2. Turnaround Tuesday       (Tuesday reversal after red Monday)
   6:   3. Month-End Effect          (Last 3 calendar days bullish)
   7:   4. Sell in May / Halloween   (May-Oct=SELL, Nov-Apr=BUY)
   8:   5. January Effect            (Jan=BUY, Dec=SELL)
   9:   6. Pre-Holiday Effect        (Day before multi-day gap=BUY)
  10:   7. FOMC Drift                (2 days before FOMC=BUY, day of/after=HOLD)
  11:   8. Santa Claus Rally         (Last 5 trading days Dec + first 2 Jan=BUY)
  12: 
  13: Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
  14: majority vote; confidence is the fraction of non-HOLD votes that agree with
  15: the majority direction.  Maximum confidence is capped at 0.6 because
  16: calendar signals are inherently weak.
  17: """
  18: 
  19: from __future__ import annotations
  20: 
  21: import calendar
  22: import logging
  23: from datetime import date, datetime, timedelta
  24: 
  25: import pandas as pd
  26: 
  27: logger = logging.getLogger("portfolio.signals.calendar_seasonal")
  28: 
  29: # ---------------------------------------------------------------------------
  30: # FOMC meeting dates — imported from shared constant
  31: # ---------------------------------------------------------------------------
  32: from portfolio.fomc_dates import (
  33:     FOMC_ANNOUNCEMENT_DATES as _FOMC_ANNOUNCEMENT_DATES,
  34: )
  35: 
  36: # H14: Cache max() at module level — was recomputed on every call (420 calls/cycle).
  37: _FOMC_MAX_DATE = max(_FOMC_ANNOUNCEMENT_DATES) if _FOMC_ANNOUNCEMENT_DATES else None
  38: 
  39: # Maximum confidence for any calendar signal
  40: _MAX_CONFIDENCE = 0.6
  41: 
  42: # Minimum rows needed for Turnaround Tuesday check
  43: _MIN_BARS = 2
  44: 
  45: 
  46: # ---------------------------------------------------------------------------
  47: # Sub-signal functions
  48: # ---------------------------------------------------------------------------
  49: 
  50: def _day_of_week_effect(last_date: date) -> tuple[str, dict]:
  51:     """Monday historically bearish for equities (SELL), Friday bullish (BUY).
  52: 
  53:     Parameters
  54:     ----------
  55:     last_date : date
  56:         Date of the last bar.
  57: 
  58:     Returns
  59:     -------
  60:     tuple[str, dict]
  61:         Vote and indicators.
  62:     """
  63:     dow = last_date.weekday()  # 0=Mon, 4=Fri
  64:     indicators = {"day_of_week": dow, "day_name": last_date.strftime("%A")}
  65: 
  66:     if dow == 0:  # Monday
  67:         return "SELL", indicators
  68:     if dow == 4:  # Friday
  69:         return "BUY", indicators
  70:     return "HOLD", indicators
  71: 
  72: 
  73: def _turnaround_tuesday(df: pd.DataFrame, last_date: date) -> tuple[str, dict]:
  74:     """Tuesday reversal after a red Monday.
  75: 
  76:     If the last bar is Tuesday AND the prior bar closed below its open
  77:     (red candle on Monday), signal BUY.
  78: 
  79:     Parameters
  80:     ----------
  81:     df : pd.DataFrame
  82:         OHLCV data with at least 2 rows.
  83:     last_date : date
  84:         Date of the last bar.
  85: 
  86:     Returns
  87:     -------
  88:     tuple[str, dict]
  89:         Vote and indicators.
  90:     """
  91:     indicators = {"is_tuesday": False, "prior_bar_red": False}
  92: 
  93:     if last_date.weekday() != 1:  # Not Tuesday
  94:         return "HOLD", indicators
  95: 
  96:     indicators["is_tuesday"] = True
  97: 
  98:     if len(df) < 2:
  99:         return "HOLD", indicators
 100: 
 101:     prior_close = float(df["close"].iloc[-2])
 102:     prior_open = float(df["open"].iloc[-2])
 103:     prior_red = prior_close < prior_open
 104:     indicators["prior_bar_red"] = prior_red
 105: 
 106:     if prior_red:
 107:         return "BUY", indicators
 108:     return "HOLD", indicators
 109: 
 110: 
 111: def _month_end_effect(last_date: date) -> tuple[str, dict]:
 112:     """Last 3 calendar days of the month tend to be bullish.
 113: 
 114:     Parameters
 115:     ----------
 116:     last_date : date
 117:         Date of the last bar.
 118: 
 119:     Returns
 120:     -------
 121:     tuple[str, dict]
 122:         Vote and indicators.
 123:     """
 124:     days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
 125:     days_remaining = days_in_month - last_date.day
 126:     is_month_end = days_remaining < 3  # last 3 calendar days (day 29, 30, 31 of a 31-day month)
 127:     indicators = {
 128:         "is_month_end": is_month_end,
 129:         "days_remaining_in_month": days_remaining,
 130:     }
 131: 
 132:     if is_month_end:
 133:         return "BUY", indicators
 134:     return "HOLD", indicators
 135: 
 136: 
 137: def _sell_in_may(last_date: date) -> tuple[str, dict]:
 138:     """Sell in May and go away / Halloween indicator.
 139: 
 140:     May through October = historically weaker (SELL bias).
 141:     Only historically *strong* months vote BUY: Nov, Dec, Jan, Apr.
 142:     Transitional months (Feb, Mar) = HOLD — they are not consistently
 143:     strong enough to justify a standing BUY vote.
 144: 
 145:     Parameters
 146:     ----------
 147:     last_date : date
 148:         Date of the last bar.
 149: 
 150:     Returns
 151:     -------
 152:     tuple[str, dict]
 153:         Vote and indicators.
 154:     """
 155:     month = last_date.month
 156:     is_weak_period = 5 <= month <= 10
 157:     is_strong_month = month in (1, 4, 11, 12)  # Jan, Apr, Nov, Dec
 158:     indicators = {"month": month, "is_weak_period": is_weak_period}
 159: 
 160:     if is_weak_period:
 161:         return "SELL", indicators
 162:     if is_strong_month:
 163:         return "BUY", indicators
 164:     return "HOLD", indicators
 165: 
 166: 
 167: def _january_effect(last_date: date) -> tuple[str, dict]:
 168:     """January historically bullish for small caps; December tax-loss selling.
 169: 
 170:     Parameters
 171:     ----------
 172:     last_date : date
 173:         Date of the last bar.
 174: 
 175:     Returns
 176:     -------
 177:     tuple[str, dict]
 178:         Vote and indicators.
 179:     """
 180:     month = last_date.month
 181:     indicators = {"month": month}
 182: 
 183:     if month == 1:
 184:         return "BUY", indicators
 185:     if month == 12:
 186:         return "SELL", indicators
 187:     return "HOLD", indicators
 188: 
 189: 
 190: def _pre_holiday_effect(last_date: date) -> tuple[str, dict]:
 191:     """Trading day before a market holiday tends to be bullish.
 192: 
 193:     Checks for days preceding US market holidays (approximate).
 194:     Regular Fridays are NOT counted here — that is already handled by
 195:     ``_day_of_week_effect``.  Only true pre-holiday sessions (the
 196:     trading day before a market closure beyond the normal weekend) vote.
 197: 
 198:     Parameters
 199:     ----------
 200:     last_date : date
 201:         Date of the last bar.
 202: 
 203:     Returns
 204:     -------
 205:     tuple[str, dict]
 206:         Vote and indicators.
 207:     """
 208:     # Major US market holidays (month, day) — approximate, does not handle
 209:     # observed-date shifts (e.g., July 4 on Saturday → Friday off).
 210:     _US_HOLIDAYS = [
 211:         (1, 1),    # New Year's Day
 212:         (1, 20),   # MLK Day (approx — 3rd Monday)
 213:         (2, 17),   # Presidents' Day (approx — 3rd Monday)
 214:         (5, 26),   # Memorial Day (approx — last Monday)
 215:         (6, 19),   # Juneteenth
 216:         (7, 4),    # Independence Day
 217:         (9, 1),    # Labor Day (approx — 1st Monday)
 218:         (11, 27),  # Thanksgiving (approx — 4th Thursday)
 219:         (12, 25),  # Christmas
 220:     ]
 221: 
 222:     next_day = last_date + timedelta(days=1)
 223:     is_pre_holiday = (next_day.month, next_day.day) in _US_HOLIDAYS
 224:     indicators = {"is_pre_holiday": is_pre_holiday, "day_of_week": last_date.weekday()}
 225: 
 226:     if is_pre_holiday:
 227:         return "BUY", indicators
 228:     return "HOLD", indicators
 229: 
 230: 
 231: def _fomc_drift(last_date: date) -> tuple[str, dict]:
 232:     """Pre-FOMC announcement drift.
 233: 
 234:     The 24 hours before a scheduled FOMC announcement tend to drift upward.
 235:     BUY if within 2 days before an announcement date.
 236:     HOLD on the day of the announcement or the day after (volatility).
 237: 
 238:     Parameters
 239:     ----------
 240:     last_date : date
 241:         Date of the last bar.
 242: 
 243:     Returns
 244:     -------
 245:     tuple[str, dict]
 246:         Vote and indicators.
 247:     """
 248:     indicators = {
 249:         "is_pre_fomc": False,
 250:         "is_fomc_day": False,
 251:         "is_post_fomc": False,
 252:         "days_to_fomc": None,
 253:     }
 254: 
 255:     # BUG-118: Warn when all FOMC dates are in the past (data staleness)
 256:     if _FOMC_MAX_DATE is not None and last_date > _FOMC_MAX_DATE:  # H14: cached
 257:         logger.warning("calendar_seasonal: all FOMC dates are in the past — fomc_dates.py needs updating")
 258: 
 259:     for fomc_date in _FOMC_ANNOUNCEMENT_DATES:
 260:         delta = (fomc_date - last_date).days
 261: 
 262:         if delta == 0:
 263:             # Day of FOMC announcement
 264:             indicators["is_fomc_day"] = True
 265:             indicators["days_to_fomc"] = 0
 266:             return "HOLD", indicators
 267: 
 268:         if delta == -1:
 269:             # Day after FOMC announcement
 270:             indicators["is_post_fomc"] = True
 271:             indicators["days_to_fomc"] = -1
 272:             return "HOLD", indicators
 273: 
 274:         if 1 <= delta <= 2:
 275:             # 1-2 days before FOMC announcement
 276:             indicators["is_pre_fomc"] = True
 277:             indicators["days_to_fomc"] = delta
 278:             return "BUY", indicators
 279: 
 280:     return "HOLD", indicators
 281: 
 282: 
 283: def _santa_claus_rally(last_date: date) -> tuple[str, dict]:
 284:     """Santa Claus Rally: last 5 trading days of Dec + first 2 of Jan.
 285: 
 286:     Approximate by checking calendar dates: Dec 25-31 and Jan 1-3.
 287:     The actual trading days vary by year, but this is a reasonable proxy.
 288: 
 289:     Parameters
 290:     ----------
 291:     last_date : date
 292:         Date of the last bar.
 293: 
 294:     Returns
 295:     -------
 296:     tuple[str, dict]
 297:         Vote and indicators.
 298:     """
 299:     month = last_date.month
 300:     day = last_date.day
 301: 
 302:     # Last 5 trading days of Dec: approx Dec 24-31 (some are holidays/weekends)
 303:     is_late_dec = month == 12 and day >= 24
 304:     # First 2 trading days of Jan: approx Jan 1-3
 305:     is_early_jan = month == 1 and day <= 3
 306: 
 307:     is_santa_rally = is_late_dec or is_early_jan
 308:     indicators = {"is_santa_rally": is_santa_rally, "month": month, "day": day}
 309: 
 310:     if is_santa_rally:
 311:         return "BUY", indicators
 312:     return "HOLD", indicators
 313: 
 314: 
 315: # ---------------------------------------------------------------------------
 316: # Composite signal
 317: # ---------------------------------------------------------------------------
 318: 
 319: def compute_calendar_signal(df: pd.DataFrame) -> dict:
 320:     """Compute the composite calendar/seasonal signal.

>>> portfolio/signals/econ_calendar.py:90-170
  90: 
  91:     if evt["type"] in ("FOMC", "CPI") and evt["hours_until"] <= 24:
  92:         return "SELL", indicators
  93:     return "HOLD", indicators
  94: 
  95: 
  96: def _pre_event_risk(ref_date) -> tuple[str, dict]:
  97:     """Binary risk-off within 4h of any high-impact event.
  98: 
  99:     This is the strongest sub-signal: if ANY high-impact event is within
 100:     4 hours, vote SELL unconditionally.
 101:     """
 102:     nearby = events_within_hours(4, ref_date.date() if isinstance(ref_date, datetime) else ref_date)
 103:     high_impact = [e for e in nearby if e["impact"] == "high"]
 104: 
 105:     indicators = {"events_within_4h": len(nearby),
 106:                   "high_impact_within_4h": len(high_impact)}
 107: 
 108:     if high_impact:
 109:         indicators["nearest_event"] = f"{high_impact[0]['type']} in {high_impact[0]['hours_until']}h"
 110:         return "SELL", indicators
 111:     return "HOLD", indicators
 112: 
 113: 
 114: def _post_event_relief(ref_date) -> tuple[str, dict]:
 115:     """Post-event relief BUY signal.
 116: 
 117:     After a high-impact event has just passed (4-24h ago), the uncertainty
 118:     premium dissipates and markets tend to rally. Also, when the next event
 119:     is >72h away, reduced macro uncertainty is mildly bullish.
 120: 
 121:     BUG-218: This sub-signal adds BUY capability that was previously missing,
 122:     making the econ_calendar signal structurally balanced (not SELL-only).
 123:     """
 124:     indicators = {"post_event_relief": False, "event_free_window": False}
 125: 
 126:     # Check for recently passed high-impact events (4-24h ago = relief window)
 127:     recent = recent_high_impact_events(24)
 128:     relief_events = [e for e in recent if e.get("hours_since", 0) >= 4]
 129: 
 130:     if relief_events:
 131:         indicators["post_event_relief"] = True
 132:         indicators["relief_event"] = (
 133:             f"{relief_events[0]['type']} passed {relief_events[0]['hours_since']:.0f}h ago"
 134:         )
 135:         # Check that no new event is imminent (would negate the relief)
 136:         evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
 137:         if evt is None or evt["hours_until"] > 24:
 138:             return "BUY", indicators
 139: 
 140:     # Event-free calm window: next event >72h away
 141:     evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
 142:     if evt is not None and evt["hours_until"] > 72:
 143:         indicators["event_free_window"] = True
 144:         indicators["next_event_hours"] = evt["hours_until"]
 145:         return "BUY", indicators
 146: 
 147:     return "HOLD", indicators
 148: 
 149: 
 150: def _sector_exposure(ref_date, ticker: str) -> tuple[str, dict]:
 151:     """Check if this ticker's sector is affected by upcoming events.
 152: 
 153:     Maps event types to affected sectors via EVENT_SECTOR_MAP,
 154:     then checks if the ticker belongs to any affected sector.
 155:     """
 156:     evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
 157:     indicators = {"ticker_sectors": list(TICKER_SECTORS.get(ticker, set())),
 158:                   "event_affects_sector": False}
 159: 
 160:     if evt is None or evt["hours_until"] > 48:
 161:         return "HOLD", indicators
 162: 
 163:     affected_sectors = EVENT_SECTOR_MAP.get(evt["type"], set())
 164:     ticker_secs = TICKER_SECTORS.get(ticker, set())
 165:     overlap = affected_sectors & ticker_secs
 166: 
 167:     indicators["affected_sectors"] = list(affected_sectors)
 168:     indicators["overlap_sectors"] = list(overlap)
 169:     indicators["event_affects_sector"] = bool(overlap)
 170: 

 succeeded in 725ms:

>>> portfolio/signals/metals_cross_asset.py:120-220
 120: 
 121:     try:
 122:         from portfolio.http_retry import fetch_with_retry
 123:     except ImportError:
 124:         logger.debug("http_retry not available for FRED fetch")
 125:         return cache.get("data")
 126: 
 127:     try:
 128:         resp = fetch_with_retry(
 129:             "https://api.stlouisfed.org/fred/series/observations",
 130:             params={
 131:                 "series_id": series_id,
 132:                 "api_key": fred_api_key,
 133:                 "file_type": "json",
 134:                 "sort_order": "desc",
 135:                 "limit": _FRED_HISTORY_LIMIT,
 136:             },
 137:             timeout=_FRED_TIMEOUT,
 138:         )
 139:         data = resp.json() if hasattr(resp, "json") else __import__("json").loads(resp)
 140:         observations = data.get("observations", [])
 141:         values = []
 142:         for obs in observations:
 143:             val = obs.get("value", ".")
 144:             if val != ".":
 145:                 try:
 146:                     values.append(float(val))
 147:                 except (ValueError, TypeError):
 148:                     continue
 149: 
 150:         if values:
 151:             with _fred_cache_lock:
 152:                 cache["key"] = fred_api_key
 153:                 cache["data"] = values
 154:                 cache["time"] = now
 155:             logger.debug(
 156:                 "FRED %s fetched: %d values, latest=%.2f",
 157:                 series_id, len(values), values[0],
 158:             )
 159:             return values
 160:     except Exception:
 161:         logger.warning("FRED %s fetch failed", series_id, exc_info=True)
 162: 
 163:     return cache.get("data")
 164: 
 165: 
 166: def _compute_zscore(values: list[float], lookback: int = 252) -> float:
 167:     """Z-score of most recent value vs lookback history."""
 168:     if len(values) < 20:
 169:         return 0.0
 170:     n = min(lookback, len(values))
 171:     history = values[:n]
 172:     mean = sum(history) / len(history)
 173:     variance = sum((v - mean) ** 2 for v in history) / len(history)
 174:     std = variance ** 0.5
 175:     if std < 1e-10:
 176:         return 0.0
 177:     return (values[0] - mean) / std
 178: 
 179: 
 180: def _get_cross_asset_context(ticker: str) -> dict | None:
 181:     """Fetch cross-asset data. Prefer intraday (60m); fall back to daily.
 182: 
 183:     Returns a dict with per-source payloads AND a ``_using_intraday`` flag
 184:     so the caller can pick the right thresholds. Returns None only if the
 185:     ``metals_cross_assets`` module is not importable (module-level failure).
 186: 
 187:     Daily G/S ratio is ALWAYS fetched — its z-score is a stable 20-day
 188:     measure that's informative at every horizon, and pre-fetching it here
 189:     ensures the intraday path never makes an extra conditional call mid-
 190:     routing.
 191:     """
 192:     try:
 193:         from portfolio.metals_cross_assets import (
 194:             get_all_cross_asset_data,
 195:             get_all_cross_asset_intraday,
 196:             get_gold_silver_ratio,
 197:             get_gvz,
 198:         )
 199:     except ImportError:
 200:         logger.debug("metals_cross_assets module not available")
 201:         return None
 202: 
 203:     # Daily anchors — always fetched, cached with their own TTL elsewhere:
 204:     #   GVZ: no intraday source exists (CBOE daily index).
 205:     #   G/S daily: z-score is a stable 20-day measure used on every path.
 206:     gvz = get_gvz()
 207:     gs_daily = get_gold_silver_ratio()
 208: 
 209:     intraday = get_all_cross_asset_intraday()
 210:     intraday_ok = sum(
 211:         1 for key in ("copper", "gold_silver_ratio", "spy", "oil")
 212:         if intraday.get(key) is not None
 213:     )
 214:     # Require at least 3 of 4 sources to consider intraday healthy;
 215:     # one-off API hiccups shouldn't downgrade the whole signal to stale
 216:     # daily data.
 217:     use_intraday = intraday_ok >= 3
 218: 
 219:     result: dict = {"_using_intraday": use_intraday}
 220: 

>>> portfolio/signals/cot_positioning.py:1-140
   1: """CFTC Commitment of Traders (COT) positioning signal for precious metals.
   2: 
   3: Signal #33.  Contrarian positioning indicator using CFTC COT reports.
   4: Combines 4 sub-indicators via majority vote:
   5:     1. COT Index: 156-week percentile of speculative net positioning (contrarian)
   6:     2. Commercial Hedger Change: week-over-week smart money direction
   7:     3. Managed Money Intensity: speculator crowding z-score
   8:     4. Real Yield Direction: falling yields = bullish gold (with regime gate)
   9: 
  10: Applicable to XAU-USD and XAG-USD only.  Data sourced from precomputed
  11: deep context files (metals_precompute.py) with CFTC API fallback.
  12: 
  13: COT reports are published weekly (Friday for Tuesday data), so this signal
  14: is inherently slower-moving than price-based signals.  Max confidence
  15: capped at 0.7 to reflect data staleness.
  16: """
  17: from __future__ import annotations
  18: 
  19: import logging
  20: from pathlib import Path
  21: from typing import Any
  22: 
  23: from portfolio.signal_utils import majority_vote
  24: 
  25: logger = logging.getLogger("portfolio.signals.cot_positioning")
  26: 
  27: # SM-P1-4 (2026-05-02 adversarial follow-ups): absolute path resolution.
  28: # The previous code used relative `Path("data")` / `data/...` which silently
  29: # broke when the scheduled task CWD differed from the repo root (e.g.
  30: # PF-DataLoop launched from C:\Windows). The deep context and COT history
  31: # loaders would return None, the signal would silently fall back to API
  32: # fetching every cycle. Mirrors the c5b78210 ic_computation fix.
  33: _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
  34: 
  35: _METALS_TICKERS = {"XAU-USD", "XAG-USD"}
  36: _COMMODITY_MAP = {"XAU-USD": "gold", "XAG-USD": "silver"}
  37: 
  38: # COT Index thresholds (0-100 percentile scale)
  39: _COT_EXTREME_HIGH = 80  # Overbought (contrarian SELL)
  40: _COT_EXTREME_LOW = 20   # Oversold (contrarian BUY)
  41: 
  42: # Commercial hedger change threshold (contracts)
  43: _COMM_CHANGE_THRESHOLD = 5000
  44: 
  45: # Managed money z-score threshold
  46: _MM_ZSCORE_HIGH = 1.5
  47: _MM_ZSCORE_LOW = -1.5
  48: 
  49: # CFTC API settings for historical lookback
  50: _COT_HISTORY_WEEKS = 156  # 3 years
  51: _CFTC_LEGACY_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
  52: _CFTC_TIMEOUT = 15
  53: 
  54: 
  55: def _load_deep_context(ticker: str) -> dict | None:
  56:     """Load precomputed deep context for the given metal.
  57: 
  58:     SM-P1-4 (2026-05-02): use absolute _DATA_DIR — was relative `f"data/..."`
  59:     which silently broke when the loop's CWD wasn't the repo root.
  60:     """
  61:     from portfolio.file_utils import load_json
  62: 
  63:     metal = _COMMODITY_MAP.get(ticker)
  64:     if not metal:
  65:         return None
  66: 
  67:     path = str(_DATA_DIR / f"{metal}_deep_context.json")
  68:     ctx = load_json(path, default=None)
  69:     if not ctx or not isinstance(ctx, dict):
  70:         logger.debug("Deep context not available: %s", path)
  71:         return None
  72:     return ctx
  73: 
  74: 
  75: def _load_cot_history(metal: str) -> list[dict]:
  76:     """Load COT history from the local JSONL file.
  77: 
  78:     SM-P1-4 (2026-05-02): use absolute _DATA_DIR — was relative
  79:     `"data/cot_history.jsonl"`.
  80:     """
  81:     from portfolio.file_utils import load_jsonl
  82: 
  83:     entries = load_jsonl(str(_DATA_DIR / "cot_history.jsonl"))
  84:     return [e for e in entries if e.get("metal") == metal]
  85: 
  86: 
  87: def _fetch_cot_historical(commodity_name: str) -> list[dict]:
  88:     """Fetch historical COT data from CFTC SOCRATA API.
  89: 
  90:     Returns up to 156 weeks of net positioning data for z-score computation.
  91:     commodity_name should be 'GOLD' or 'SILVER'.
  92:     """
  93:     try:
  94:         import requests
  95: 
  96:         url = (
  97:             f"{_CFTC_LEGACY_URL}"
  98:             f"?$where=commodity_name='{commodity_name}'"
  99:             f"&$order=report_date_as_yyyy_mm_dd DESC"
 100:             f"&$limit={_COT_HISTORY_WEEKS}"
 101:         )
 102:         resp = requests.get(url, timeout=_CFTC_TIMEOUT)
 103:         resp.raise_for_status()
 104:         rows = resp.json()
 105: 
 106:         result = []
 107:         for r in rows:
 108:             nc_long = _int_safe(r.get("noncomm_positions_long_all"))
 109:             nc_short = _int_safe(r.get("noncomm_positions_short_all"))
 110:             comm_long = _int_safe(r.get("comm_positions_long_all"))
 111:             comm_short = _int_safe(r.get("comm_positions_short_all"))
 112: 
 113:             if nc_long is not None and nc_short is not None:
 114:                 result.append({
 115:                     "date": r.get("report_date_as_yyyy_mm_dd", ""),
 116:                     "nc_net": nc_long - nc_short,
 117:                     "comm_net": (comm_long - comm_short)
 118:                     if comm_long is not None and comm_short is not None
 119:                     else None,
 120:                     "oi": _int_safe(r.get("open_interest_all")),
 121:                 })
 122:         return result
 123:     except Exception as e:
 124:         logger.warning("CFTC historical fetch failed: %s", e)
 125:         return []
 126: 
 127: 
 128: def _int_safe(val) -> int | None:
 129:     """Safely convert to int, returning None on failure."""
 130:     if val is None:
 131:         return None
 132:     try:
 133:         return int(val)
 134:     except (ValueError, TypeError):
 135:         return None
 136: 
 137: 
 138: def _compute_cot_index(nc_net_history: list[int]) -> float | None:
 139:     """Compute COT Index as percentile of current net positioning.
 140: 

>>> portfolio/signals/credit_spread.py:130-210
 130:     if not cfg:
 131:         return ""
 132:     if isinstance(cfg, dict):
 133:         return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
 134:     return getattr(cfg, "fred_api_key", "") or getattr(
 135:         getattr(cfg, "golddigger", None), "fred_api_key", ""
 136:     ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
 137: 
 138: 
 139: def _is_safe_haven(ticker: str) -> bool:
 140:     """Return True if ticker is a safe-haven asset (gold/silver)."""
 141:     return ticker in _SAFE_HAVEN
 142: 
 143: 
 144: # ---------------------------------------------------------------------------
 145: # Sub-indicator 1: OAS Level Z-Score
 146: # ---------------------------------------------------------------------------
 147: 
 148: def _oas_zscore_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
 149:     """Z-score of current OAS vs 252d history."""
 150:     if len(values) < 50:
 151:         return "HOLD", {"oas_zscore": 0.0}
 152: 
 153:     lookback = min(252, len(values))
 154:     history = values[:lookback]
 155:     current = values[0]
 156:     mean = sum(history) / len(history)
 157:     variance = sum((v - mean) ** 2 for v in history) / len(history)
 158:     std = variance ** 0.5
 159: 
 160:     if std < 0.01:
 161:         return "HOLD", {"oas_zscore": 0.0}
 162: 
 163:     zscore = (current - mean) / std
 164: 
 165:     action = "HOLD"
 166:     if zscore > _ZSCORE_RISK_OFF:
 167:         # Risk-off: spreads unusually wide
 168:         action = "BUY" if safe_haven else "SELL"
 169:     elif zscore < _ZSCORE_RISK_ON:
 170:         # Risk-on: spreads unusually tight (complacent)
 171:         action = "SELL" if safe_haven else "BUY"
 172: 
 173:     return action, {"oas_zscore": round(zscore, 3), "oas_current": round(current, 2),
 174:                      "oas_mean_252d": round(mean, 2)}
 175: 
 176: 
 177: # ---------------------------------------------------------------------------
 178: # Sub-indicator 2: OAS 5d Momentum
 179: # ---------------------------------------------------------------------------
 180: 
 181: def _oas_momentum_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
 182:     """5-day change in OAS (widening vs tightening)."""
 183:     if len(values) < 6:
 184:         return "HOLD", {"oas_mom_5d": 0.0}
 185: 
 186:     mom = values[0] - values[5]
 187: 
 188:     action = "HOLD"
 189:     if mom > _MOM_5D_THRESHOLD:
 190:         # Spreads widening rapidly → risk-off
 191:         action = "BUY" if safe_haven else "SELL"
 192:     elif mom < -_MOM_5D_THRESHOLD:
 193:         # Spreads tightening → risk-on
 194:         action = "SELL" if safe_haven else "BUY"
 195: 
 196:     return action, {"oas_mom_5d": round(mom, 3)}
 197: 
 198: 
 199: # ---------------------------------------------------------------------------
 200: # Sub-indicator 3: Momentum Acceleration
 201: # ---------------------------------------------------------------------------
 202: 
 203: def _oas_acceleration_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
 204:     """Acceleration of OAS momentum (early warning of regime shift)."""
 205:     if len(values) < 11:
 206:         return "HOLD", {"oas_accel": 0.0}
 207: 
 208:     mom_current = values[0] - values[5]
 209:     mom_prev = values[5] - values[10]
 210:     accel = mom_current - mom_prev

 succeeded in 785ms:

>>> portfolio/signals/claude_fundamental.py:1-230
   1: """Claude Fundamental Signal (#28) — three-tier LLM cascade for fundamental analysis.
   2: 
   3: Tier 1 (Haiku):  Quick directional pulse, every ~1 min
   4: Tier 2 (Sonnet): Full 5-dimension fundamental analysis, every ~10 min
   5: Tier 3 (Opus):   Deep conviction with cross-asset reasoning, every ~30 min
   6: 
   7: Highest-tier fresh analysis wins (Opus > Sonnet > Haiku).
   8: Sub-signals: fundamental_quality, sector_positioning, valuation,
   9:              catalyst_assessment, macro_sensitivity.
  10: """
  11: 
  12: from __future__ import annotations
  13: 
  14: import json
  15: import logging
  16: import re
  17: import threading
  18: import time
  19: from datetime import UTC
  20: from pathlib import Path
  21: 
  22: import pandas as pd
  23: 
  24: from portfolio.file_utils import load_json
  25: from portfolio.signal_utils import majority_vote
  26: 
  27: logger = logging.getLogger("portfolio.signals.claude_fundamental")
  28: DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
  29: 
  30: _MAX_CONFIDENCE = 0.7
  31: _CF_LOG = DATA_DIR / "claude_fundamental_log.jsonl"
  32: 
  33: SUB_SIGNAL_NAMES = [
  34:     "fundamental_quality",
  35:     "sector_positioning",
  36:     "valuation",
  37:     "catalyst_assessment",
  38:     "macro_sensitivity",
  39: ]
  40: 
  41: # --- Three-tier cache ---
  42: _cache = {
  43:     "haiku":  {"results": {}, "ts": 0},
  44:     "sonnet": {"results": {}, "ts": 0},
  45:     "opus":   {"results": {}, "ts": 0},
  46: }
  47: _lock = threading.Lock()
  48: 
  49: _DEFAULT_HOLD = {
  50:     "action": "HOLD",
  51:     "confidence": 0.0,
  52:     "sub_signals": {},
  53:     "indicators": {},
  54: }
  55: 
  56: 
  57: def _get_cooldowns(config):
  58:     """Get per-tier cooldowns from config with defaults.
  59: 
  60:     Defaults are tuned for accuracy tracking — need enough samples to
  61:     measure hit rate. Haiku is cheap/fast so runs often. Opus is expensive
  62:     so runs sparingly.
  63:     """
  64:     cf = config.get("claude_fundamental", {})
  65:     return {
  66:         "haiku":  cf.get("haiku_cooldown", 300),    # 5 min default
  67:         "sonnet": cf.get("sonnet_cooldown", 1800),   # 30 min default
  68:         "opus":   cf.get("opus_cooldown", 7200),      # 2h default
  69:     }
  70: 
  71: 
  72: def _get_models(config):
  73:     """Get per-tier model aliases from config with defaults."""
  74:     cf = config.get("claude_fundamental", {})
  75:     return {
  76:         "haiku":  cf.get("haiku_model", "haiku"),
  77:         "sonnet": cf.get("sonnet_model", "sonnet"),
  78:         "opus":   cf.get("opus_model", "opus"),
  79:     }
  80: 
  81: 
  82: def _get_timeouts(config):
  83:     """Get per-tier CLI timeouts from config with defaults."""
  84:     cf = config.get("claude_fundamental", {})
  85:     return {
  86:         "haiku":  cf.get("haiku_timeout", 30),
  87:         "sonnet": cf.get("sonnet_timeout", 60),
  88:         "opus":   cf.get("opus_timeout", 120),
  89:     }
  90: 
  91: 
  92: def _needs_refresh(tier, cooldowns):
  93:     """Check if a tier's cache has expired."""
  94:     return time.time() - _cache[tier]["ts"] > cooldowns[tier]
  95: 
  96: 
  97: def _build_ticker_grid(summary):
  98:     """Build compact ticker grid from agent_summary_compact data."""
  99:     lines = []
 100:     tickers = summary.get("signals", summary.get("tickers", {}))
 101:     if not tickers:
 102:         logger.warning("claude_fundamental: empty ticker grid — summary has keys %s "
 103:                         "but no 'signals' or 'tickers' data. Signal will return HOLD.",
 104:                         list(summary.keys())[:5])
 105:         return ""
 106:     for ticker, data in tickers.items():
 107:         price = data.get("price_usd", data.get("price", "?"))
 108:         rsi_val = data.get("rsi", "?")
 109:         macd_val = data.get("macd_hist", "?")
 110:         regime = data.get("regime", "?")
 111:         consensus = data.get("consensus", "HOLD")
 112:         buy_count = data.get("buy_count", 0)
 113:         sell_count = data.get("sell_count", 0)
 114:         hold_count = data.get("hold_count", data.get("abstain_count", 0))
 115:         vol_ratio = data.get("volume_ratio", "?")
 116:         lines.append(
 117:             f"{ticker}: ${price} RSI={rsi_val} MACD={macd_val} "
 118:             f"regime={regime} vol={vol_ratio} "
 119:             f"consensus={consensus} ({buy_count}B/{sell_count}S/{hold_count}H)"
 120:         )
 121:     return "\n".join(lines)
 122: 
 123: 
 124: def _build_macro_block(macro):
 125:     """Build macro context string from macro data."""
 126:     if not macro:
 127:         return "Macro data unavailable."
 128:     parts = []
 129:     dxy = macro.get("dxy", {})
 130:     if dxy:
 131:         parts.append(f"DXY: {dxy.get('value', '?')} ({dxy.get('change_5d', '?')} 5d)")
 132:     treasury = macro.get("treasury", {})
 133:     if treasury:
 134:         y10 = treasury.get("10y", "?")
 135:         y2 = treasury.get("2y", "?")
 136:         spread = treasury.get("2s10s", "?")
 137:         parts.append(f"10Y: {y10}% | 2Y: {y2}% | 2s10s: {spread}")
 138:     fed = macro.get("fed", {})
 139:     if fed:
 140:         parts.append(f"FOMC: {fed.get('next_date', '?')} ({fed.get('days_until', '?')}d)")
 141:     fg = macro.get("fear_greed", {})
 142:     if fg:
 143:         parts.append(f"F&G: {fg.get('crypto', '?')}/{fg.get('stock', '?')}")
 144:     return " | ".join(parts) if parts else "Macro data unavailable."
 145: 
 146: 
 147: # Earnings calendar cache (refreshed every 12h)
 148: _earnings_cache = {"data": {}, "ts": 0}
 149: _EARNINGS_CACHE_TTL = 43200  # 12 hours
 150: 
 151: 
 152: def _get_earnings_calendar():
 153:     """Fetch next earnings dates from yfinance for all stock tickers.
 154: 
 155:     Cached for 12h to avoid excessive Yahoo Finance requests.
 156:     Returns dict of {ticker: {date, eps_estimate, days_until}}.
 157:     """
 158:     now = time.time()
 159:     if now - _earnings_cache["ts"] < _EARNINGS_CACHE_TTL and _earnings_cache["data"]:
 160:         return _earnings_cache["data"]
 161: 
 162:     from datetime import datetime
 163: 
 164:     try:
 165:         from portfolio.tickers import STOCK_SYMBOLS
 166:     except ImportError:
 167:         return {}
 168: 
 169:     result = {}
 170:     for ticker in STOCK_SYMBOLS:
 171:         try:
 172:             import yfinance as yf
 173:             t = yf.Ticker(ticker)
 174:             cal = t.calendar
 175:             if cal is None or (hasattr(cal, "empty") and cal.empty):
 176:                 continue
 177:             if isinstance(cal, dict):
 178:                 dates = cal.get("Earnings Date", [])
 179:                 eps_avg = cal.get("Earnings Average")
 180:             elif hasattr(cal, "loc"):
 181:                 dates = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else []
 182:                 eps_avg = cal.loc["Earnings Average"] if "Earnings Average" in cal.index else None
 183:             else:
 184:                 continue
 185: 
 186:             if dates:
 187:                 next_date = dates[0] if isinstance(dates, list) else dates
 188:                 if hasattr(next_date, "date"):
 189:                     next_date = next_date.date()
 190:                 today = datetime.now(UTC).date()
 191:                 days_until = (next_date - today).days
 192:                 result[ticker] = {
 193:                     "date": str(next_date),
 194:                     "eps_estimate": float(eps_avg) if eps_avg is not None else None,
 195:                     "days_until": days_until,
 196:                 }
 197:         except Exception:
 198:             logger.debug("Earnings fetch failed for ticker", exc_info=True)
 199:             continue
 200: 
 201:     _earnings_cache["data"] = result
 202:     _earnings_cache["ts"] = now
 203:     return result
 204: 
 205: 
 206: def _build_fundamentals_block(ticker, fundamentals, tier="haiku", earnings=None):
 207:     """Build a fundamentals data string for a ticker based on tier detail level.
 208: 
 209:     Args:
 210:         ticker: Stock ticker symbol
 211:         fundamentals: Dict of all fundamentals from alpha_vantage cache
 212:         tier: "haiku" (one-liner), "sonnet"/"opus" (detailed block)
 213: 
 214:     Returns:
 215:         Formatted string with fundamentals data, or empty string if none available.
 216:     """
 217:     fund = fundamentals.get(ticker) if fundamentals else None
 218:     if not fund:
 219:         return ""
 220: 
 221:     earn = (earnings or {}).get(ticker, {})
 222: 
 223:     if tier == "haiku":
 224:         # One-liner: key metrics only
 225:         parts = [ticker + ":"]
 226:         pe = fund.get("pe_ratio")
 227:         if pe is not None:
 228:             parts.append(f"PE={pe:.1f}")
 229:         rev_growth = fund.get("revenue_growth_yoy")
 230:         if rev_growth is not None:

>>> portfolio/signals/forecast.py:1-120
   1: """Forecast signal — price direction prediction from time-series models.
   2: 
   3: Combines Kronos (K-line foundation model), Chronos (Amazon), and Prophet
   4: into a majority-vote composite signal with four sub-signals:
   5:   1. kronos_1h  — Kronos 1-hour prediction
   6:   2. kronos_24h — Kronos 24-hour prediction
   7:   3. chronos_1h — Chronos 1-hour prediction
   8:   4. chronos_24h — Chronos 24-hour prediction
   9: 
  10: Reuses candle loading from portfolio.forecast_signal. Confidence capped at 0.7.
  11: Registered as enhanced signal #28 with weight=0 (shadow mode) initially.
  12: """
  13: 
  14: from __future__ import annotations
  15: 
  16: import json
  17: import logging
  18: import platform
  19: import subprocess
  20: import threading
  21: import time
  22: from concurrent.futures import ThreadPoolExecutor
  23: from concurrent.futures import TimeoutError as FuturesTimeout
  24: from datetime import UTC, datetime
  25: from pathlib import Path
  26: 
  27: import pandas as pd
  28: 
  29: from portfolio.file_utils import atomic_append_jsonl
  30: from portfolio.gpu_gate import gpu_gate
  31: from portfolio.shared_state import _cached
  32: from portfolio.signal_utils import majority_vote
  33: 
  34: logger = logging.getLogger("portfolio.signals.forecast")
  35: 
  36: # Cache TTL — forecasts don't change fast
  37: _FORECAST_TTL = 300  # 5 minutes
  38: 
  39: # Confidence cap (same as news_event, econ_calendar)
  40: _MAX_CONFIDENCE = 0.7
  41: 
  42: # Default Chronos timeout (seconds) — reduced from 120 to avoid long hangs
  43: _CHRONOS_TIMEOUT = 60
  44: 
  45: # Default Kronos subprocess timeout (seconds) — reduced from 90; fails fast
  46: _KRONOS_TIMEOUT = 30
  47: 
  48: # Forecast models master switch. Set to True to disable all model calls (early-return HOLD).
  49: # Circuit breakers remain as secondary protection — auto-trip on failure, 5min TTL.
  50: _FORECAST_MODELS_DISABLED = False
  51: 
  52: # Kronos inference — UN-RETIRED 2026-04-21 afternoon. The morning's retire
  53: # was premature. The real problem was not Kronos per se but the shadow-mode
  54: # implementation: when _KRONOS_SHADOW=True, forecast.py forced Kronos's
  55: # composite-vote sub-signal to HOLD (see lines ~811/820) while still counting
  56: # those HOLD votes inside _health_weighted_vote's majority tally — so when
  57: # Chronos voted BUY/BUY/BUY and Kronos (silently HOLD) also "voted" H/H/H,
  58: # the result was a 3-3 tie that broke to HOLD and polluted Chronos's verdict.
  59: #
  60: # The correct structure (this session):
  61: #   - Kronos subprocess runs normally, real prediction captured in
  62: #     raw_sub_signals["kronos_*"] for accuracy backfill.
  63: #   - In shadow mode the Kronos sub-signal is EXCLUDED FROM THE COMPOSITE
  64: #     VOTE POOL entirely (see _health_weighted_vote below) — it does not
  65: #     contribute HOLD weights, does not dilute Chronos.
  66: #   - Subprocess reliability (59 % success → 90 % target) remains a separate
  67: #     work stream. The shadow-logging path tolerates failure gracefully;
  68: #     un-retirement does not require the reliability fix to land first.
  69: #
  70: # Config re-read here so operators can flip kronos_enabled=true|"shadow"|false
  71: # without a code edit, matching the pre-retire behavior.
  72: _KRONOS_ENABLED = False
  73: _KRONOS_SHADOW = False
  74: 
  75: 
  76: def _init_kronos_enabled():
  77:     """Read kronos_enabled from config.json at import time.
  78: 
  79:     Values:
  80:       - `true`  → active voter (raw prediction contributes to composite vote)
  81:       - `"shadow"` → inference runs, raw prediction is logged, but the
  82:         sub-signal is EXCLUDED from the composite vote pool (see
  83:         _health_weighted_vote). Zero composite pollution.
  84:       - `false` (default) → subprocess skipped entirely.
  85:     """
  86:     global _KRONOS_ENABLED, _KRONOS_SHADOW
  87:     try:
  88:         from portfolio.file_utils import load_json as _load_json
  89:         _cfg = _load_json(
  90:             str(Path(__file__).resolve().parent.parent.parent / "config.json"),
  91:             {},
  92:         )
  93:         val = _cfg.get("forecast", {}).get("kronos_enabled", False)
  94:         if val == "shadow":
  95:             _KRONOS_ENABLED = True
  96:             _KRONOS_SHADOW = True
  97:         else:
  98:             _KRONOS_ENABLED = bool(val)
  99:             _KRONOS_SHADOW = False
 100:     except Exception as e:
 101:         logger.debug("Kronos init from config: %s", e)
 102: 
 103: 
 104: _init_kronos_enabled()
 105: 
 106: if platform.system() == "Windows":
 107:     _KRONOS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
 108:     _KRONOS_SCRIPT = r"Q:\models\kronos_infer.py"
 109: else:
 110:     _KRONOS_PYTHON = "/home/deck/models/.venv/bin/python"
 111:     _KRONOS_SCRIPT = "/home/deck/models/kronos_infer.py"
 112: 
 113: # Prediction log
 114: _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
 115: _PREDICTIONS_FILE = _DATA_DIR / "forecast_predictions.jsonl"
 116: _HEALTH_FILE = _DATA_DIR / "forecast_health.jsonl"
 117: 
 118: # Circuit breaker — after first failure, skip remaining tickers in this loop cycle.
 119: # Prevents 27 x 6s GPU timeouts when CUDA is broken.
 120: _CIRCUIT_BREAKER_TTL = 30  # 30 seconds before retry

>>> portfolio/signals/forecast.py:740-830
 740:     context : dict, optional
 741:         Dict with keys: ticker, config, macro.
 742: 
 743:     Returns
 744:     -------
 745:     dict
 746:         action, confidence, sub_signals, indicators
 747:     """
 748:     result = {
 749:         "action": "HOLD",
 750:         "confidence": 0.0,
 751:         "sub_signals": {
 752:             "kronos_1h": "HOLD",
 753:             "kronos_24h": "HOLD",
 754:             "chronos_1h": "HOLD",
 755:             "chronos_24h": "HOLD",
 756:         },
 757:         "indicators": {},
 758:     }
 759: 
 760:     # Bulletproof early return — skip ALL work when models are disabled
 761:     if _FORECAST_MODELS_DISABLED:
 762:         result["indicators"]["models_disabled"] = True
 763:         return result
 764: 
 765:     ticker = (context or {}).get("ticker", "")
 766:     if not ticker:
 767:         return result
 768: 
 769:     # Apply Chronos model config if specified
 770:     chronos_model = (context or {}).get("config", {}).get("forecast", {}).get("chronos_model")
 771:     if chronos_model:
 772:         try:
 773:             from portfolio.forecast_signal import set_chronos_model
 774:             set_chronos_model(chronos_model)
 775:         except Exception as e:
 776:             logger.debug("Chronos model config override failed: %s", e)
 777: 
 778:     config_forecast = (context or {}).get("config", {}).get("forecast", {})
 779: 
 780:     # Load candles (1h for Chronos, optionally 5m for Kronos)
 781:     cache_key = f"forecast_candles_{ticker}"
 782:     candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)
 783: 
 784:     # Load 5m candles for Kronos if configured (more granular context)
 785:     kronos_interval = config_forecast.get("kronos_interval", "1h")
 786:     if kronos_interval != "1h" and _KRONOS_ENABLED:
 787:         kronos_periods = config_forecast.get("kronos_periods", 500)
 788:         kronos_cache_key = f"forecast_candles_{ticker}_{kronos_interval}"
 789:         kronos_candles = _cached(kronos_cache_key, _FORECAST_TTL,
 790:                                   _load_candles_ohlcv, ticker, kronos_periods,
 791:                                   kronos_interval)
 792:     else:
 793:         kronos_candles = None
 794: 
 795:     if not candles or len(candles) < 50:
 796:         # Fallback to df close prices if available
 797:         if df is not None and len(df) >= 50 and "close" in df.columns:
 798:             close_prices = df["close"].values.tolist()
 799:         else:
 800:             result["indicators"]["error"] = "insufficient_candle_data"
 801:             return result
 802:     else:
 803:         close_prices = [c["close"] for c in candles]
 804: 
 805:     # If Kronos-specific candle fetch failed but df has full OHLCV data, build candle dicts
 806:     # from the DataFrame so Kronos still gets richer data than just close prices.
 807:     if kronos_candles is None and df is not None and len(df) >= 50:
 808:         ohlcv_cols = {"open", "high", "low", "close", "volume"}
 809:         if ohlcv_cols.issubset(df.columns):
 810:             try:
 811:                 kronos_candles = [
 812:                     {
 813:                         "open": float(row["open"]),
 814:                         "high": float(row["high"]),
 815:                         "low": float(row["low"]),
 816:                         "close": float(row["close"]),
 817:                         "volume": float(row["volume"]),
 818:                     }
 819:                     for _, row in df.iterrows()
 820:                 ]
 821:                 logger.debug(
 822:                     "Kronos candle fallback from df for %s (%d candles)", ticker, len(kronos_candles)
 823:                 )
 824:                 result["indicators"]["kronos_candles_source"] = "df_fallback"
 825:             except Exception as e:
 826:                 logger.debug("Kronos df candle fallback failed for %s: %s", ticker, e)
 827:                 kronos_candles = None
 828: 
 829:     current_price = close_prices[-1]
 830:     result["indicators"]["current_price"] = current_price

codex
I’ve verified the first batch against the actual implementations. Several of Claude’s claims are already collapsing under direct inspection, and I’m now checking the “disabled/dead code”, mapping-staleness, and “new signal with no validation” claims against registry and module wiring.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/news_event.py'; Start=1; End=220},
  @{Path='portfolio/signals/structure.py'; Start=1; End=260}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/treasury_risk_rotation.py'; Start=1; End=260},
  @{Path='portfolio/signals/realized_skewness.py'; Start=1; End=220},
  @{Path='portfolio/signals/statistical_jump_regime.py'; Start=1; End=220},
  @{Path='portfolio/signals/hurst_regime.py'; Start=1; End=220}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signal_registry.py'; Start=1; End=340},
  @{Path='portfolio/signals/__init__.py'; Start=1; End=220}
)
foreach ("'$f in $files) {
  if (Test-Path $f.Path) {
    Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
    $lines = Get-Content $f.Path
    for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
      Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
    }
  } else {
    Write-Output ("MISSING: {0}" -f $f.Path)
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "DISABLED_SIGNALS|disabled|pending validation|shadow mode|accuracy|treasury_risk_rotation|smart_money|oscillators|orderbook_flow|realized_skewness|statistical_jump_regime|hurst_regime" portfolio -S' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 666ms:
portfolio\signals\forecast.py:11:Registered as enhanced signal #28 with weight=0 (shadow mode) initially.
portfolio\signals\forecast.py:62:#     raw_sub_signals["kronos_*"] for accuracy backfill.
portfolio\signals\forecast.py:63:#   - In shadow mode the Kronos sub-signal is EXCLUDED FROM THE COMPOSITE
portfolio\signals\forecast.py:450:    shadow mode is for logging + accuracy backfill only — adding HOLD weights
portfolio\signals\forecast.py:454:    downstream accuracy tracking.
portfolio\signals\forecast.py:480:# Per-ticker accuracy cache TTL
portfolio\signals\forecast.py:483:# Default thresholds for accuracy gating
portfolio\signals\forecast.py:533:def _load_forecast_accuracy(cache_ttl=None):
portfolio\signals\forecast.py:534:    """Load per-ticker forecast accuracy, cached via _cached().
portfolio\signals\forecast.py:536:    Returns dict: {ticker: {accuracy, samples}} or empty dict on error.
portfolio\signals\forecast.py:542:            from portfolio.forecast_accuracy import get_all_ticker_accuracies
portfolio\signals\forecast.py:545:            logger.debug("Failed to load forecast accuracy: %s", e)
portfolio\signals\forecast.py:548:    return _cached("forecast_ticker_accuracy", ttl, _fetch)
portfolio\signals\forecast.py:551:def _load_forecast_subsignal_accuracy(cache_ttl=None, days=30):
portfolio\signals\forecast.py:552:    """Load raw sub-signal accuracy for 1h and 24h forecast votes."""
portfolio\signals\forecast.py:554:    cache_key = f"forecast_subsignal_accuracy_{days}"
portfolio\signals\forecast.py:558:            from portfolio.forecast_accuracy import compute_forecast_accuracy
portfolio\signals\forecast.py:561:                "1h": compute_forecast_accuracy(
portfolio\signals\forecast.py:564:                "24h": compute_forecast_accuracy(
portfolio\signals\forecast.py:569:            logger.debug("Failed to load forecast sub-signal accuracy: %s", e)
portfolio\signals\forecast.py:575:def _gate_subsignal_votes_by_accuracy(sub_signals, ticker, config_forecast=None):
portfolio\signals\forecast.py:576:    """Gate individual forecast sub-signals using raw historical accuracy."""
portfolio\signals\forecast.py:580:    lookback_days = cfg.get("subsignal_accuracy_days", 30)
portfolio\signals\forecast.py:581:    cache_ttl = cfg.get("subsignal_accuracy_cache_ttl", _ACCURACY_CACHE_TTL)
portfolio\signals\forecast.py:588:    accuracy_matrix = _load_forecast_subsignal_accuracy(cache_ttl=cache_ttl, days=lookback_days)
portfolio\signals\forecast.py:594:        horizon_stats = ((accuracy_matrix or {}).get(horizon) or {}).get(sub_name) or {}
portfolio\signals\forecast.py:597:        accuracy = None
portfolio\signals\forecast.py:601:            accuracy = float(ticker_stats["accuracy"])
portfolio\signals\forecast.py:605:            accuracy = float(horizon_stats["accuracy"])
portfolio\signals\forecast.py:610:        if accuracy is not None:
portfolio\signals\forecast.py:611:            gating = "held" if accuracy < hold_threshold else "raw"
portfolio\signals\forecast.py:617:            "accuracy": round(accuracy, 3) if accuracy is not None else None,
portfolio\signals\forecast.py:644:def _accuracy_weighted_vote(sub_signals, kronos_ok, chronos_ok, ticker="",
portfolio\signals\forecast.py:646:    """Vote with per-ticker accuracy gating, volatility gate, and regime discount.
portfolio\signals\forecast.py:651:    - Accuracy gate: accuracy < hold_threshold → force HOLD
portfolio\signals\forecast.py:652:    - Good accuracy: use raw vote, scale confidence by accuracy
portfolio\signals\forecast.py:656:    accuracy metric without fixing the underlying prediction quality.
portfolio\signals\forecast.py:659:    accuracy metadata for logging.
portfolio\signals\forecast.py:671:        "forecast_accuracy": None,
portfolio\signals\forecast.py:693:    # Load per-ticker accuracy
portfolio\signals\forecast.py:694:    all_acc = _load_forecast_accuracy(cfg.get("accuracy_cache_ttl"))
portfolio\signals\forecast.py:700:            gating_info["forecast_accuracy"] = ticker_acc["accuracy"]
portfolio\signals\forecast.py:704:    acc = ticker_acc["accuracy"]
portfolio\signals\forecast.py:706:    gating_info["forecast_accuracy"] = acc
portfolio\signals\forecast.py:715:        # Good accuracy — use raw vote, scale confidence by accuracy
portfolio\signals\forecast.py:760:    # Bulletproof early return — skip ALL work when models are disabled
portfolio\signals\forecast.py:762:        result["indicators"]["models_disabled"] = True
portfolio\signals\forecast.py:885:            # (not a forced HOLD), which unblocks meaningful accuracy
portfolio\signals\forecast.py:907:    gated_sub_signals, subsignal_gating = _gate_subsignal_votes_by_accuracy(
portfolio\signals\forecast.py:913:    # Accuracy-weighted vote — per-ticker accuracy gating + health exclusion
portfolio\signals\forecast.py:924:    result["action"], result["confidence"], gating_info = _accuracy_weighted_vote(
portfolio\signals\forecast.py:931:    result["indicators"]["forecast_accuracy"] = gating_info.get("forecast_accuracy")
portfolio\signals\forecast.py:936:    # Cap confidence (already capped inside _accuracy_weighted_vote, but belt-and-suspenders)
portfolio\signals\forecast.py:939:    # Log prediction for accuracy tracking (with dedup)
portfolio\signals\forecast.py:957:                "per_ticker_accuracy": gating_info.get("forecast_accuracy"),
portfolio\signals\claude_fundamental.py:60:    Defaults are tuned for accuracy tracking — need enough samples to
portfolio\signals\claude_fundamental.py:636:    """Persist tier refresh results for accuracy tracking and debugging.
portfolio\signals\claude_fundamental.py:717:    # Journal the results for accuracy tracking and debugging
portfolio\signals\claude_fundamental.py:753:    collapsed claude_fundamental from 59.4% to 37.9% recent accuracy.
portfolio\signals\copper_gold_ratio.py:10:    - 94% recession prediction accuracy (3-month sustained move)
portfolio\signals\realized_skewness.py:183:def compute_realized_skewness_signal(
portfolio\signals\hurst_regime.py:112:def _hurst_regime(h: float) -> str:
portfolio\signals\hurst_regime.py:212:def compute_hurst_regime_signal(df: pd.DataFrame,
portfolio\signals\hurst_regime.py:233:            "hurst_regime": "HOLD",
portfolio\signals\hurst_regime.py:270:    regime = _hurst_regime(h)
portfolio\signals\hurst_regime.py:284:            sub_signals["hurst_regime"] = trend_vote
portfolio\signals\hurst_regime.py:288:            logger.debug("hurst_regime failed", exc_info=True)
portfolio\signals\hurst_regime.py:289:            sub_signals["hurst_regime"] = "HOLD"
portfolio\signals\hurst_regime.py:301:            sub_signals["hurst_regime"] = mr_vote
portfolio\signals\hurst_regime.py:306:            sub_signals["hurst_regime"] = "HOLD"
portfolio\signals\hurst_regime.py:316:        sub_signals["hurst_regime"] = "HOLD"
portfolio\signals\orderbook_flow.py:22:logger = logging.getLogger("portfolio.signals.orderbook_flow")
portfolio\signals\orderbook_flow.py:92:def compute_orderbook_flow_signal(
portfolio\signals\oscillators.py:1:"""Composite advanced oscillators signal module.
portfolio\signals\vix_term_structure.py:10:to all assets; per-ticker accuracy gating auto-disables for assets
portfolio\signals\statistical_jump_regime.py:138:def compute_statistical_jump_regime_signal(
portfolio\signals\smart_money.py:401:def compute_smart_money_signal(df: pd.DataFrame) -> dict[str, Any]:
portfolio\signals\smart_money.py:439:        logger.warning("smart_money: input is not a DataFrame")
portfolio\signals\smart_money.py:448:        logger.warning("smart_money: missing columns %s", missing)
portfolio\signals\smart_money.py:453:            "smart_money: insufficient data (%d rows, need %d)",
portfolio\signals\smart_money.py:464:        logger.info("smart_money: too many NaN rows, only %d remain", len(df))
portfolio\signals\smart_money.py:484:            logger.debug("smart_money: BOS detection failed", exc_info=True)
portfolio\signals\smart_money.py:492:            logger.debug("smart_money: CHoCH detection failed", exc_info=True)
portfolio\signals\smart_money.py:502:            logger.debug("smart_money: FVG detection failed", exc_info=True)
portfolio\signals\smart_money.py:514:            logger.debug("smart_money: liquidity sweep detection failed", exc_info=True)
portfolio\signals\smart_money.py:526:            logger.debug("smart_money: S/D zone detection failed", exc_info=True)
portfolio\signals\smart_money.py:561:        logger.exception("smart_money: unexpected error computing signal")
portfolio\signals\treasury_risk_rotation.py:31:logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")
portfolio\signals\treasury_risk_rotation.py:62:                logger.warning("treasury_risk_rotation: missing IEF or TLT column")
portfolio\signals\treasury_risk_rotation.py:69:                logger.warning("treasury_risk_rotation: insufficient data rows")
portfolio\signals\treasury_risk_rotation.py:74:            logger.warning("treasury_risk_rotation yfinance fetch failed: %s", e)
portfolio\signals\treasury_risk_rotation.py:77:    return _cached("treasury_risk_rotation_yf", _CACHE_TTL, _do_fetch)
portfolio\signals\treasury_risk_rotation.py:153:def compute_treasury_risk_rotation_signal(

 succeeded in 688ms:
MISSING: portfolio/signal_registry.py

>>> portfolio/signals/__init__.py:1-220
   1: """Enhanced signal modules — grouped technical indicators.
   2: 
   3: Each module computes multiple sub-indicators and returns a composite vote.
   4: """

 succeeded in 774ms:

>>> portfolio/signals/news_event.py:1-220
   1: """News/event detection signal — headline velocity, keyword severity, sentiment shift.
   2: 
   3: Combines five sub-indicators into a majority-vote composite:
   4:   1. headline_velocity  — article count spike vs baseline
   5:   2. keyword_severity   — highest severity keyword in recent headlines
   6:   3. sentiment_shift    — rapid positive/negative skew in headline keywords
   7:   4. source_weight      — credible sources amplify signal
   8:   5. sector_impact      — keyword + sector mapping for directional vote
   9: 
  10: The ``context`` parameter is a dict with keys: ticker, config, macro.
  11: Headlines are fetched using the existing sentiment.py functions with caching.
  12: """
  13: 
  14: from __future__ import annotations
  15: 
  16: import logging
  17: import os
  18: from datetime import UTC, datetime
  19: 
  20: import pandas as pd
  21: 
  22: from portfolio.file_utils import atomic_write_json
  23: from portfolio.news_keywords import (
  24:     dissemination_score,
  25:     get_sector_impact,
  26:     is_credible_source,
  27:     keyword_severity,
  28:     score_headline,
  29: )
  30: from portfolio.shared_state import _cached
  31: from portfolio.signal_utils import majority_vote
  32: 
  33: logger = logging.getLogger("portfolio.signals.news_event")
  34: 
  35: # Cache TTL for outer headline fetch (5 min).
  36: # Inner NewsAPI cache uses NEWSAPI_TTL (30 min) to protect daily quota.
  37: _HEADLINE_TTL = 300
  38: 
  39: # Baseline article count (normal news volume)
  40: _BASELINE_ARTICLES = 5
  41: 
  42: # Max confidence cap
  43: _MAX_CONFIDENCE = 0.7
  44: 
  45: # Persisted headlines path (fish monitor reads this)
  46: _HEADLINES_PATH = os.path.join(
  47:     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
  48:     "data", "headlines_latest.json",
  49: )
  50: 
  51: 
  52: def _persist_headlines(ticker: str, headlines: list[dict]) -> None:
  53:     """Write top 10 scored headlines to disk for fish monitor consumption."""
  54:     if not headlines:
  55:         return
  56:     try:
  57:         scored = []
  58:         for h in headlines:
  59:             title = h.get("title", "")
  60:             if not title:
  61:                 continue
  62:             sev = keyword_severity(title)
  63:             weight, _ = score_headline(title)
  64:             # Determine sentiment from keyword direction
  65:             lower = title.lower()
  66:             if sev in ("critical", "high"):
  67:                 sentiment = "negative"
  68:             elif sev == "moderate":
  69:                 if any(kw in lower for kw in (
  70:                     "beat", "upgrade", "approval", "approved",
  71:                     "raise", "buyback", "split", "rally", "surge",
  72:                 )):
  73:                     sentiment = "positive"
  74:                 else:
  75:                     sentiment = "negative"
  76:             else:
  77:                 sentiment = "neutral"
  78:             scored.append({
  79:                 "title": title,
  80:                 "source": h.get("source", "unknown"),
  81:                 "severity": sev,
  82:                 "sentiment": sentiment,
  83:                 "_weight": weight,
  84:             })
  85:         # Sort by weight descending, take top 10
  86:         scored.sort(key=lambda x: x["_weight"], reverse=True)
  87:         top = scored[:10]
  88:         # Remove internal sort key
  89:         for item in top:
  90:             item.pop("_weight", None)
  91:         payload = {
  92:             "timestamp": datetime.now(UTC).isoformat(),
  93:             "ticker": ticker,
  94:             "headlines": top,
  95:         }
  96:         atomic_write_json(_HEADLINES_PATH, payload)
  97:     except Exception:
  98:         logger.debug("Failed to persist headlines for %s", ticker, exc_info=True)
  99: 
 100: 
 101: def _fetch_headlines(ticker: str, config: dict) -> list[dict]:
 102:     """Fetch headlines for a ticker + sector peers.
 103: 
 104:     Fetches ticker-specific headlines, then merges sector-wide headlines
 105:     from a representative peer. This ensures that "tariff" news hitting NVDA
 106:     also propagates to MU, TSM, AMD, etc.
 107:     """
 108:     if not ticker:
 109:         return []
 110:     short = ticker.upper().replace("-USD", "")
 111:     from portfolio.sentiment import _fetch_crypto_headlines, _fetch_stock_headlines, _is_crypto
 112: 
 113:     articles = []
 114:     try:
 115:         if _is_crypto(short):
 116:             articles = _cached(
 117:                 f"news_headlines_crypto_{short}",
 118:                 _HEADLINE_TTL,
 119:                 _fetch_crypto_headlines,
 120:                 short,
 121:             ) or []
 122:         else:
 123:             newsapi_key = config.get("newsapi_key", "")
 124:             articles = _cached(
 125:                 f"news_headlines_stock_{short}",
 126:                 _HEADLINE_TTL,
 127:                 _fetch_stock_headlines,
 128:                 short,
 129:                 newsapi_key or None,
 130:             ) or []
 131:     except Exception:
 132:         logger.debug("Failed to fetch headlines for %s", ticker, exc_info=True)
 133: 
 134:     # Merge sector-peer headlines: fetch from a representative ticker per sector
 135:     # so "tariff" news appearing in NVDA headlines also reaches MU, TSM, AMD, etc.
 136:     # Each sector has one representative whose headlines are shared with all members.
 137:     _SECTOR_REP_TICKER = {
 138:         "semiconductor": "NVDA",
 139:         "big_tech": "AAPL",
 140:         "ai": "NVDA",
 141:         "defense": "LMT",
 142:         "software": "PLTR",
 143:         "gaming": "TTWO",
 144:         "infrastructure": "VRT",
 145:         "metals": None,   # metals use Binance FAPI, not Yahoo
 146:         "crypto": None,   # crypto uses CryptoCompare categories (already shared)
 147:     }
 148:     try:
 149:         from portfolio.news_keywords import TICKER_SECTORS
 150:         seen_titles = {a.get("title", "").lower() for a in articles}
 151:         ticker_secs = TICKER_SECTORS.get(ticker, set())
 152:         for sec in ticker_secs:
 153:             rep = _SECTOR_REP_TICKER.get(sec)
 154:             if not rep or rep == short:
 155:                 continue  # skip if no rep, or if this IS the rep
 156:             newsapi_key = config.get("newsapi_key", "")
 157:             peer_articles = _cached(
 158:                 f"news_headlines_stock_{rep}",
 159:                 _HEADLINE_TTL,
 160:                 _fetch_stock_headlines,
 161:                 rep,
 162:                 newsapi_key or None,
 163:             ) or []
 164:             for a in peer_articles:
 165:                 title_lower = a.get("title", "").lower()
 166:                 if title_lower and title_lower not in seen_titles:
 167:                     articles.append(a)
 168:                     seen_titles.add(title_lower)
 169:     except Exception:
 170:         logger.debug("Failed to fetch sector headlines for %s", ticker, exc_info=True)
 171: 
 172:     return articles
 173: 
 174: 
 175: def _headline_velocity(headlines: list[dict]) -> tuple[str, dict]:
 176:     """Article count spike detection.
 177: 
 178:     If article count is 2x+ baseline AND most have negative keywords, vote SELL.
 179:     If article count is 2x+ baseline AND most have positive keywords, vote BUY.
 180:     Otherwise HOLD.
 181:     """
 182:     count = len(headlines)
 183:     indicators = {"article_count": count, "baseline": _BASELINE_ARTICLES}
 184: 
 185:     if count < _BASELINE_ARTICLES * 2:
 186:         return "HOLD", indicators
 187: 
 188:     # Check keyword sentiment direction
 189:     neg_count = 0
 190:     pos_count = 0
 191:     for h in headlines:
 192:         title = h.get("title", "")
 193:         sev = keyword_severity(title)
 194:         if sev in ("critical", "high"):
 195:             neg_count += 1
 196:         elif sev == "moderate":
 197:             # moderate keywords can be positive (earnings beat, upgrade)
 198:             lower = title.lower()
 199:             if any(kw in lower for kw in ("beat", "upgrade", "approval", "approved", "raise", "buyback", "split")):
 200:                 pos_count += 1
 201:             else:
 202:                 neg_count += 1
 203: 
 204:     indicators["neg_keyword_count"] = neg_count
 205:     indicators["pos_keyword_count"] = pos_count
 206: 
 207:     if neg_count > pos_count and neg_count >= 2:
 208:         return "SELL", indicators
 209:     if pos_count > neg_count and pos_count >= 2:
 210:         return "BUY", indicators
 211:     return "HOLD", indicators
 212: 
 213: 
 214: def _keyword_severity_vote(headlines: list[dict]) -> tuple[str, dict]:
 215:     """Highest severity keyword detection.
 216: 
 217:     Critical or high keywords → SELL (negative news dominates markets).
 218:     Multiple moderate positive → BUY.
 219:     """
 220:     max_sev = "normal"

>>> portfolio/signals/structure.py:1-260
   1: """Composite price-structure / breakout signal.
   2: 
   3: Combines four sub-indicators into a majority-vote composite:
   4:   1. Period High/Low Breakout  (52-week or available range)
   5:   2. Donchian Channel(55) Breakout
   6:   3. RSI(14) Centerline Cross
   7:   4. MACD(12,26,9) Zero-Line Cross
   8: 
   9: Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
  10: majority vote; confidence is the fraction of non-HOLD votes that agree with
  11: the majority direction.
  12: """
  13: 
  14: from __future__ import annotations
  15: 
  16: import logging
  17: 
  18: import numpy as np
  19: import pandas as pd
  20: 
  21: from portfolio.signal_utils import majority_vote, rsi
  22: 
  23: logger = logging.getLogger("portfolio.signals.structure")
  24: 
  25: # ---------------------------------------------------------------------------
  26: # Minimum data lengths for each sub-indicator
  27: # ---------------------------------------------------------------------------
  28: _MIN_BARS_RSI = 15       # RSI(14) needs at least 15 bars
  29: _MIN_BARS_MACD = 35      # MACD(12,26,9): 26 + 9 warm-up
  30: _MIN_BARS_DONCHIAN = 56  # Donchian(55) needs 56 to have two values
  31: _MIN_BARS_HIGHLOW = 20   # Bare minimum for period high/low to be meaningful
  32: 
  33: 
  34: def _macd_histogram(close: pd.Series,
  35:                     fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
  36:     """MACD histogram (MACD line minus signal line)."""
  37:     ema_fast = close.ewm(span=fast, adjust=False).mean()
  38:     ema_slow = close.ewm(span=slow, adjust=False).mean()
  39:     macd_line = ema_fast - ema_slow
  40:     signal_line = macd_line.ewm(span=signal, adjust=False).mean()
  41:     return macd_line - signal_line
  42: 
  43: 
  44: # ---------------------------------------------------------------------------
  45: # Sub-indicator votes
  46: # ---------------------------------------------------------------------------
  47: 
  48: def _highlow_breakout(df: pd.DataFrame) -> tuple[str, dict]:
  49:     """Period high/low proximity check.
  50: 
  51:     Uses all available data (ideally 252 daily bars for a 52-week window).
  52:     BUY if close is within 2% of the period high; SELL if within 2% of the
  53:     period low; HOLD otherwise.
  54:     """
  55:     indicators: dict = {"period_high": np.nan, "period_low": np.nan}
  56: 
  57:     if len(df) < _MIN_BARS_HIGHLOW:
  58:         return "HOLD", indicators
  59: 
  60:     # H13: Cap to last 252 bars (1 year daily / ~3 months hourly).
  61:     lookback_df = df.iloc[-252:]
  62:     high = lookback_df["high"].astype(float)
  63:     low = lookback_df["low"].astype(float)
  64:     close = lookback_df["close"].astype(float)
  65: 
  66:     period_high = high.max()
  67:     period_low = low.min()
  68:     current_close = close.iloc[-1]
  69: 
  70:     indicators["period_high"] = float(period_high)
  71:     indicators["period_low"] = float(period_low)
  72: 
  73:     if period_high == 0:
  74:         return "HOLD", indicators
  75: 
  76:     pct_from_high = (period_high - current_close) / period_high
  77:     pct_from_low = (current_close - period_low) / period_low if period_low != 0 else np.inf
  78: 
  79:     if pct_from_high <= 0.02:
  80:         return "BUY", indicators
  81:     if pct_from_low <= 0.02:
  82:         return "SELL", indicators
  83:     return "HOLD", indicators
  84: 
  85: 
  86: def _donchian_breakout(df: pd.DataFrame, period: int = 55) -> tuple[str, dict]:
  87:     """Donchian Channel(55) breakout.
  88: 
  89:     BUY when close breaks above the upper channel (highest high of prior
  90:     *period* bars).  SELL when close breaks below the lower channel (lowest
  91:     low of prior *period* bars).  HOLD when inside the channel.
  92:     """
  93:     indicators: dict = {"donchian_upper": np.nan, "donchian_lower": np.nan}
  94: 
  95:     if len(df) < period + 1:
  96:         return "HOLD", indicators
  97: 
  98:     high = df["high"].astype(float)
  99:     low = df["low"].astype(float)
 100:     close = df["close"].astype(float)
 101: 
 102:     # Channel based on the *previous* period bars (exclude the current bar)
 103:     upper = high.iloc[-(period + 1):-1].max()
 104:     lower = low.iloc[-(period + 1):-1].min()
 105:     current_close = close.iloc[-1]
 106: 
 107:     indicators["donchian_upper"] = float(upper)
 108:     indicators["donchian_lower"] = float(lower)
 109: 
 110:     if current_close > upper:
 111:         return "BUY", indicators
 112:     if current_close < lower:
 113:         return "SELL", indicators
 114:     return "HOLD", indicators
 115: 
 116: 
 117: def _rsi_centerline(df: pd.DataFrame) -> tuple[str, dict]:
 118:     """RSI(14) centerline cross.
 119: 
 120:     BUY when RSI > 60, SELL when RSI < 40 (wide deadband to filter noise).
 121:     """
 122:     indicators: dict = {"rsi": np.nan}
 123: 
 124:     close = df["close"].astype(float)
 125:     if len(df) < _MIN_BARS_RSI:
 126:         return "HOLD", indicators
 127: 
 128:     rsi_series = rsi(close)
 129:     rsi_val = rsi_series.iloc[-1]
 130:     indicators["rsi"] = float(rsi_val)
 131: 
 132:     if np.isnan(rsi_val):
 133:         return "HOLD", indicators
 134: 
 135:     if rsi_val > 60.0:
 136:         return "BUY", indicators
 137:     if rsi_val < 40.0:
 138:         return "SELL", indicators
 139:     return "HOLD", indicators
 140: 
 141: 
 142: def _macd_zeroline(df: pd.DataFrame) -> tuple[str, dict]:
 143:     """MACD(12,26,9) histogram zero-line cross.
 144: 
 145:     BUY when histogram crosses from negative to positive (current > 0 and
 146:     previous <= 0).  SELL when it crosses from positive to negative.
 147:     If no cross occurred on the latest bar, HOLD.
 148:     """
 149:     indicators: dict = {"macd_hist": np.nan}
 150: 
 151:     close = df["close"].astype(float)
 152:     if len(df) < _MIN_BARS_MACD:
 153:         return "HOLD", indicators
 154: 
 155:     hist = _macd_histogram(close)
 156: 
 157:     current = hist.iloc[-1]
 158:     previous = hist.iloc[-2]
 159:     indicators["macd_hist"] = float(current)
 160: 
 161:     if np.isnan(current) or np.isnan(previous):
 162:         return "HOLD", indicators
 163: 
 164:     # Bullish cross: histogram flips from non-positive to positive
 165:     if current > 0 and previous <= 0:
 166:         return "BUY", indicators
 167:     # Bearish cross: histogram flips from non-negative to negative
 168:     if current < 0 and previous >= 0:
 169:         return "SELL", indicators
 170:     return "HOLD", indicators
 171: 
 172: 
 173: # ---------------------------------------------------------------------------
 174: # Composite signal
 175: # ---------------------------------------------------------------------------
 176: 
 177: def compute_structure_signal(df: pd.DataFrame) -> dict:
 178:     """Compute the composite price-structure signal.
 179: 
 180:     Parameters
 181:     ----------
 182:     df : pd.DataFrame
 183:         OHLCV candle data with columns ``open``, ``high``, ``low``,
 184:         ``close``, ``volume``.  At least 20 rows recommended; ideally 100+.
 185: 
 186:     Returns
 187:     -------
 188:     dict
 189:         ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
 190:         ``sub_signals`` dict, and ``indicators`` dict.
 191:     """
 192:     result: dict = {
 193:         "action": "HOLD",
 194:         "confidence": 0.0,
 195:         "sub_signals": {
 196:             "high_low_breakout": "HOLD",
 197:             "donchian_55": "HOLD",
 198:             "rsi_centerline": "HOLD",
 199:             "macd_zeroline": "HOLD",
 200:         },
 201:         "indicators": {
 202:             "period_high": np.nan,
 203:             "period_low": np.nan,
 204:             "donchian_upper": np.nan,
 205:             "donchian_lower": np.nan,
 206:             "rsi": np.nan,
 207:             "macd_hist": np.nan,
 208:         },
 209:     }
 210: 
 211:     # ---- Validate input ----
 212:     if df is None or not isinstance(df, pd.DataFrame):
 213:         return result
 214: 
 215:     required_cols = {"open", "high", "low", "close", "volume"}
 216:     if not required_cols.issubset(set(df.columns)):
 217:         return result
 218: 
 219:     if len(df) < 2:
 220:         return result
 221: 
 222:     # ---- Compute each sub-indicator ----
 223:     try:
 224:         hl_action, hl_ind = _highlow_breakout(df)
 225:     except Exception:
 226:         logger.exception("high_low_breakout sub-signal failed")
 227:         hl_action, hl_ind = "HOLD", {"period_high": np.nan, "period_low": np.nan}
 228: 
 229:     try:
 230:         dc_action, dc_ind = _donchian_breakout(df, period=55)
 231:     except Exception:
 232:         logger.exception("donchian_55 sub-signal failed")
 233:         dc_action, dc_ind = "HOLD", {"donchian_upper": np.nan, "donchian_lower": np.nan}
 234: 
 235:     try:
 236:         rsi_action, rsi_ind = _rsi_centerline(df)
 237:     except Exception:
 238:         logger.exception("rsi_centerline sub-signal failed")
 239:         rsi_action, rsi_ind = "HOLD", {"rsi": np.nan}
 240: 
 241:     try:
 242:         macd_action, macd_ind = _macd_zeroline(df)
 243:     except Exception:
 244:         logger.exception("macd_zeroline sub-signal failed")
 245:         macd_action, macd_ind = "HOLD", {"macd_hist": np.nan}
 246: 
 247:     # ---- Populate sub-signals and indicators ----
 248:     result["sub_signals"]["high_low_breakout"] = hl_action
 249:     result["sub_signals"]["donchian_55"] = dc_action
 250:     result["sub_signals"]["rsi_centerline"] = rsi_action
 251:     result["sub_signals"]["macd_zeroline"] = macd_action
 252: 
 253:     result["indicators"].update(hl_ind)
 254:     result["indicators"].update(dc_ind)
 255:     result["indicators"].update(rsi_ind)
 256:     result["indicators"].update(macd_ind)
 257: 
 258:     # ---- Majority vote ----
 259:     votes = [hl_action, dc_action, rsi_action, macd_action]
 260:     result["action"], result["confidence"] = majority_vote(votes)

 succeeded in 857ms:

>>> portfolio/signals/treasury_risk_rotation.py:1-260
   1: """Treasury yield curve risk rotation signal.
   2: 
   3: Uses relative performance of IEF (7-10Y Treasury) vs TLT (20Y+ Treasury)
   4: as a cross-asset risk regime detector.  Steepening curve (TLT outperforms)
   5: signals risk-on; flattening/inverting (IEF outperforms) signals risk-off.
   6: 
   7: 4 sub-indicators via majority vote:
   8:     1. Curve Slope Direction   — 65d relative return sign
   9:     2. Slope Momentum          — 21d rate of change of the spread
  10:     3. Slope Z-Score           — 252d z-score of current spread
  11:     4. Regime Persistence      — days since last regime flip
  12: 
  13: Direction depends on asset class:
  14: - Risk-on assets (BTC, ETH, MSTR): steepening = BUY, flattening = SELL
  15: - Safe havens (XAU, XAG):          steepening = SELL, flattening = BUY
  16: 
  17: Data: yfinance IEF + TLT (free, no API key). Cached 1 hour.
  18: Source: Gayed (2014). An Intermarket Approach to Tactical Risk Rotation.
  19:         SSRN 2431022.
  20: """
  21: from __future__ import annotations
  22: 
  23: import logging
  24: 
  25: import numpy as np
  26: import pandas as pd
  27: 
  28: from portfolio.shared_state import _cached
  29: from portfolio.signal_utils import majority_vote, safe_float
  30: 
  31: logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")
  32: 
  33: MIN_ROWS = 30
  34: _CACHE_TTL = 3600
  35: 
  36: _SLOPE_LOOKBACK = 65
  37: _MOM_LOOKBACK = 21
  38: _ZSCORE_LOOKBACK = 252
  39: _SPREAD_THRESHOLD = 0.01
  40: _ZSCORE_THRESHOLD = 1.0
  41: _MOM_THRESHOLD = 0.005
  42: _PERSISTENCE_DAYS = 5
  43: 
  44: _SAFE_HAVENS = frozenset({"XAU-USD", "XAG-USD"})
  45: 
  46: 
  47: def _fetch_treasury_data() -> dict | None:
  48:     """Fetch IEF and TLT daily closes via yfinance.  Cached 1 hour."""
  49: 
  50:     def _do_fetch():
  51:         try:
  52:             import yfinance as yf
  53: 
  54:             data = yf.download(
  55:                 ["IEF", "TLT"], period="14mo", progress=False, threads=True,
  56:             )
  57:             if data is None or data.empty:
  58:                 return None
  59: 
  60:             close = data["Close"]
  61:             if "IEF" not in close.columns or "TLT" not in close.columns:
  62:                 logger.warning("treasury_risk_rotation: missing IEF or TLT column")
  63:                 return None
  64: 
  65:             ief = close["IEF"].dropna()
  66:             tlt = close["TLT"].dropna()
  67: 
  68:             if len(ief) < _SLOPE_LOOKBACK + 1 or len(tlt) < _SLOPE_LOOKBACK + 1:
  69:                 logger.warning("treasury_risk_rotation: insufficient data rows")
  70:                 return None
  71: 
  72:             return {"ief": ief, "tlt": tlt}
  73:         except Exception as e:
  74:             logger.warning("treasury_risk_rotation yfinance fetch failed: %s", e)
  75:             return None
  76: 
  77:     return _cached("treasury_risk_rotation_yf", _CACHE_TTL, _do_fetch)
  78: 
  79: 
  80: def _compute_spread_series(ief: pd.Series, tlt: pd.Series) -> pd.Series:
  81:     """Compute rolling spread = TLT_return - IEF_return for each lookback window."""
  82:     ief_ret = ief.pct_change(_SLOPE_LOOKBACK)
  83:     tlt_ret = tlt.pct_change(_SLOPE_LOOKBACK)
  84:     return tlt_ret - ief_ret
  85: 
  86: 
  87: def _sub_slope_direction(spread_current: float) -> str:
  88:     if spread_current > _SPREAD_THRESHOLD:
  89:         return "BUY"
  90:     if spread_current < -_SPREAD_THRESHOLD:
  91:         return "SELL"
  92:     return "HOLD"
  93: 
  94: 
  95: def _sub_slope_momentum(spread_series: pd.Series) -> str:
  96:     if len(spread_series) < _MOM_LOOKBACK + 1:
  97:         return "HOLD"
  98:     mom = float(spread_series.iloc[-1]) - float(spread_series.iloc[-_MOM_LOOKBACK - 1])
  99:     if mom > _MOM_THRESHOLD:
 100:         return "BUY"
 101:     if mom < -_MOM_THRESHOLD:
 102:         return "SELL"
 103:     return "HOLD"
 104: 
 105: 
 106: def _sub_slope_zscore(spread_series: pd.Series) -> tuple[float, str]:
 107:     n = min(_ZSCORE_LOOKBACK, len(spread_series))
 108:     if n < 30:
 109:         return 0.0, "HOLD"
 110:     window = spread_series.iloc[-n:]
 111:     mean = float(window.mean())
 112:     std = float(window.std())
 113:     if std < 1e-9:
 114:         return 0.0, "HOLD"
 115:     z = (float(spread_series.iloc[-1]) - mean) / std
 116:     if z > _ZSCORE_THRESHOLD:
 117:         return z, "BUY"
 118:     if z < -_ZSCORE_THRESHOLD:
 119:         return z, "SELL"
 120:     return z, "HOLD"
 121: 
 122: 
 123: def _sub_regime_persistence(spread_series: pd.Series) -> str:
 124:     """Count consecutive days the spread has stayed on the same side."""
 125:     if len(spread_series) < 2:
 126:         return "HOLD"
 127:     last = float(spread_series.iloc[-1])
 128:     if last == 0.0:
 129:         return "HOLD"
 130:     current_sign = 1 if last > 0 else -1
 131:     days = 0
 132:     for i in range(len(spread_series) - 1, -1, -1):
 133:         val = float(spread_series.iloc[i])
 134:         if val == 0.0:
 135:             break
 136:         s = 1 if val > 0 else -1
 137:         if s != current_sign:
 138:             break
 139:         days += 1
 140:     if days >= _PERSISTENCE_DAYS:
 141:         return "BUY" if current_sign > 0 else "SELL"
 142:     return "HOLD"
 143: 
 144: 
 145: def _invert(action: str) -> str:
 146:     if action == "BUY":
 147:         return "SELL"
 148:     if action == "SELL":
 149:         return "BUY"
 150:     return "HOLD"
 151: 
 152: 
 153: def compute_treasury_risk_rotation_signal(
 154:     df: pd.DataFrame, context: dict | None = None,
 155: ) -> dict:
 156:     """Compute treasury risk rotation signal."""
 157:     if df is None or len(df) < MIN_ROWS:
 158:         return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
 159: 
 160:     treasury = _fetch_treasury_data()
 161:     if treasury is None:
 162:         return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
 163: 
 164:     ief = treasury["ief"]
 165:     tlt = treasury["tlt"]
 166:     spread_series = _compute_spread_series(ief, tlt)
 167:     spread_series = spread_series.replace([np.inf, -np.inf], np.nan).dropna()
 168: 
 169:     if len(spread_series) < 30:
 170:         return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
 171: 
 172:     spread_current = float(spread_series.iloc[-1])
 173: 
 174:     v_direction = _sub_slope_direction(spread_current)
 175:     v_momentum = _sub_slope_momentum(spread_series)
 176:     zscore_val, v_zscore = _sub_slope_zscore(spread_series)
 177:     v_persistence = _sub_regime_persistence(spread_series)
 178: 
 179:     votes = [v_direction, v_momentum, v_zscore, v_persistence]
 180:     action, confidence = majority_vote(votes, count_hold=False)
 181: 
 182:     ticker = (context or {}).get("ticker", "")
 183:     is_safe_haven = ticker in _SAFE_HAVENS
 184:     if is_safe_haven:
 185:         action = _invert(action)
 186: 
 187:     ief_ret_65 = float(ief.iloc[-1] / ief.iloc[-min(_SLOPE_LOOKBACK, len(ief) - 1) - 1] - 1)
 188:     tlt_ret_65 = float(tlt.iloc[-1] / tlt.iloc[-min(_SLOPE_LOOKBACK, len(tlt) - 1) - 1] - 1)
 189: 
 190:     return {
 191:         "action": action,
 192:         "confidence": min(confidence, 0.7),
 193:         "sub_signals": {
 194:             "slope_direction": v_direction,
 195:             "slope_momentum": v_momentum,
 196:             "slope_zscore": v_zscore,
 197:             "regime_persistence": v_persistence,
 198:         },
 199:         "indicators": {
 200:             "spread_65d": safe_float(spread_current),
 201:             "zscore": safe_float(zscore_val),
 202:             "ief_ret_65d": safe_float(ief_ret_65),
 203:             "tlt_ret_65d": safe_float(tlt_ret_65),
 204:             "is_safe_haven": is_safe_haven,
 205:         },
 206:     }

>>> portfolio/signals/realized_skewness.py:1-220
   1: """Realized skewness directional signal module.
   2: 
   3: Computes 4 sub-indicators based on the 3rd moment of daily returns and returns
   4: a majority-vote composite BUY/SELL/HOLD signal with confidence score.
   5: 
   6: Sub-indicators:
   7:     1. Skewness Z-Score      — z-scored realized skewness vs rolling baseline
   8:     2. Skewness Momentum     — 5-bar delta of skewness (acceleration)
   9:     3. Kurtosis Confirmation — high kurtosis + negative skew = stronger signal
  10:     4. Rolling Skew Regime   — short-window skew vs long-window skew divergence
  11: 
  12: Academic evidence:
  13:     - Fernandez-Perez et al (2018): Sharpe 0.79, 8.01% annual on 27 commodity
  14:       futures. Low-skew assets outperform high-skew assets.
  15:     - ScienceDirect (2024): negative cross-sectional relationship between
  16:       skewness risk and future crypto returns.
  17: 
  18: Requires a pandas DataFrame with columns: open, high, low, close, volume
  19: and at least 60 rows of data (for z-score normalization).
  20: """
  21: from __future__ import annotations
  22: 
  23: import numpy as np
  24: import pandas as pd
  25: from scipy import stats
  26: 
  27: from portfolio.signal_utils import majority_vote, safe_float
  28: 
  29: # Minimum rows required.  We need at least ``SKEW_LOOKBACK`` returns for a
  30: # reliable skewness estimate, but we fall back to ``MIN_ROWS`` when the full
  31: # lookback is not yet available.
  32: MIN_ROWS = 60
  33: SKEW_LOOKBACK = 252       # ~1 year of daily data
  34: NORM_WINDOW = 60          # z-score normalisation window
  35: Z_BUY = -1.5             # negative skew = mean-reversion opportunity
  36: Z_SELL = 1.5              # positive skew = momentum exhaustion
  37: SKEW_MOM_PERIOD = 5       # bars for skewness momentum (acceleration)
  38: SHORT_SKEW_WINDOW = 20    # short-window skew for regime divergence
  39: 
  40: 
  41: def _compute_rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
  42:     """Compute rolling skewness using scipy.stats.skew (Fisher definition)."""
  43:     return returns.rolling(window=window, min_periods=max(window // 2, 20)).apply(
  44:         lambda x: stats.skew(x, nan_policy="omit") if len(x.dropna()) >= 20 else np.nan,
  45:         raw=False,
  46:     )
  47: 
  48: 
  49: def _sub_skew_zscore(returns: pd.Series) -> tuple[float, str, dict]:
  50:     """Sub-signal 1: z-scored realized skewness.
  51: 
  52:     Negative skew (fat left tail) historically precedes mean-reversion rallies.
  53:     Positive skew (fat right tail) signals momentum exhaustion.
  54:     """
  55:     lookback = min(SKEW_LOOKBACK, len(returns))
  56:     skew_val = stats.skew(returns.iloc[-lookback:].dropna())
  57:     if np.isnan(skew_val):
  58:         return 0.0, "HOLD", {"raw_skewness": np.nan, "skew_z": np.nan}
  59: 
  60:     # Compute rolling skewness for z-scoring
  61:     rolling_skew = _compute_rolling_skewness(returns, lookback)
  62:     recent = rolling_skew.iloc[-NORM_WINDOW:]
  63:     mean_skew = recent.mean()
  64:     std_skew = recent.std()
  65: 
  66:     if std_skew < 1e-8 or np.isnan(std_skew):
  67:         return 0.0, "HOLD", {"raw_skewness": safe_float(skew_val), "skew_z": 0.0}
  68: 
  69:     z = (skew_val - mean_skew) / std_skew
  70: 
  71:     indicators = {"raw_skewness": safe_float(skew_val), "skew_z": safe_float(z)}
  72: 
  73:     if z < Z_BUY:
  74:         confidence = min(abs(z) / 3.0, 1.0)
  75:         return confidence, "BUY", indicators
  76:     elif z > Z_SELL:
  77:         confidence = min(abs(z) / 3.0, 1.0)
  78:         return confidence, "SELL", indicators
  79:     return 0.0, "HOLD", indicators
  80: 
  81: 
  82: def _sub_skew_momentum(returns: pd.Series) -> tuple[float, str, dict]:
  83:     """Sub-signal 2: skewness momentum (acceleration).
  84: 
  85:     If skewness is falling rapidly (becoming more negative), the asset is
  86:     developing a fat left tail — contrarian BUY.  Rising skewness toward
  87:     positive = SELL.
  88:     """
  89:     lookback = min(SKEW_LOOKBACK, len(returns))
  90:     rolling_skew = _compute_rolling_skewness(returns, lookback)
  91:     valid = rolling_skew.dropna()
  92: 
  93:     if len(valid) < SKEW_MOM_PERIOD + 1:
  94:         return 0.0, "HOLD", {"skew_momentum": np.nan}
  95: 
  96:     current = valid.iloc[-1]
  97:     past = valid.iloc[-(SKEW_MOM_PERIOD + 1)]
  98:     delta = current - past
  99: 
 100:     indicators = {"skew_momentum": safe_float(delta)}
 101: 
 102:     # Falling skew (becoming more negative) = BUY
 103:     if delta < -0.3:
 104:         return min(abs(delta) / 1.0, 0.8), "BUY", indicators
 105:     # Rising skew (becoming more positive) = SELL
 106:     elif delta > 0.3:
 107:         return min(abs(delta) / 1.0, 0.8), "SELL", indicators
 108:     return 0.0, "HOLD", indicators
 109: 
 110: 
 111: def _sub_kurtosis_confirm(returns: pd.Series) -> tuple[float, str, dict]:
 112:     """Sub-signal 3: kurtosis confirmation.
 113: 
 114:     High kurtosis (fat tails) + negative skew = strong mean-reversion BUY.
 115:     High kurtosis + positive skew = strong momentum exhaustion SELL.
 116:     Low kurtosis = no extreme tail structure = HOLD.
 117:     """
 118:     lookback = min(SKEW_LOOKBACK, len(returns))
 119:     window = returns.iloc[-lookback:].dropna()
 120:     if len(window) < 20:
 121:         return 0.0, "HOLD", {"kurtosis": np.nan}
 122: 
 123:     kurt = stats.kurtosis(window)  # excess kurtosis (0 for normal)
 124:     skew_val = stats.skew(window)
 125: 
 126:     indicators = {
 127:         "kurtosis": safe_float(kurt),
 128:         "kurtosis_skew_product": safe_float(kurt * skew_val),
 129:     }
 130: 
 131:     # High kurtosis threshold (excess kurtosis > 1.0 = leptokurtic)
 132:     if kurt <= 1.0:
 133:         return 0.0, "HOLD", indicators
 134: 
 135:     # Negative skew + high kurtosis = BUY (fat left tail, expected snap-back)
 136:     if skew_val < -0.5:
 137:         confidence = min(kurt / 5.0, 0.8)
 138:         return confidence, "BUY", indicators
 139:     # Positive skew + high kurtosis = SELL (fat right tail, expected pullback)
 140:     elif skew_val > 0.5:
 141:         confidence = min(kurt / 5.0, 0.8)
 142:         return confidence, "SELL", indicators
 143:     return 0.0, "HOLD", indicators
 144: 
 145: 
 146: def _sub_skew_regime_divergence(returns: pd.Series) -> tuple[float, str, dict]:
 147:     """Sub-signal 4: short-window vs long-window skewness divergence.
 148: 
 149:     When recent (20-bar) skewness diverges significantly from long-term
 150:     (252-bar) skewness, it signals a regime change.
 151:     """
 152:     if len(returns) < SKEW_LOOKBACK:
 153:         long_window = max(len(returns), MIN_ROWS)
 154:     else:
 155:         long_window = SKEW_LOOKBACK
 156: 
 157:     short_data = returns.iloc[-SHORT_SKEW_WINDOW:].dropna()
 158:     long_data = returns.iloc[-long_window:].dropna()
 159: 
 160:     if len(short_data) < 15 or len(long_data) < 30:
 161:         return 0.0, "HOLD", {"skew_divergence": np.nan}
 162: 
 163:     short_skew = stats.skew(short_data)
 164:     long_skew = stats.skew(long_data)
 165:     divergence = short_skew - long_skew
 166: 
 167:     indicators = {
 168:         "short_skew": safe_float(short_skew),
 169:         "long_skew": safe_float(long_skew),
 170:         "skew_divergence": safe_float(divergence),
 171:     }
 172: 
 173:     # Short-term skew much more negative than long-term = recent sell-off,
 174:     # expect mean reversion = BUY
 175:     if divergence < -0.8:
 176:         return min(abs(divergence) / 2.0, 0.7), "BUY", indicators
 177:     # Short-term skew much more positive = recent euphoria, expect pullback = SELL
 178:     elif divergence > 0.8:
 179:         return min(abs(divergence) / 2.0, 0.7), "SELL", indicators
 180:     return 0.0, "HOLD", indicators
 181: 
 182: 
 183: def compute_realized_skewness_signal(
 184:     df: pd.DataFrame, context: dict = None
 185: ) -> dict:
 186:     """Compute realized skewness directional signal.
 187: 
 188:     Args:
 189:         df: DataFrame with columns: open, high, low, close, volume
 190:         context: Optional dict with keys: ticker, config, asset_class, regime
 191: 
 192:     Returns:
 193:         dict with keys: action, confidence, sub_signals, indicators
 194:     """
 195:     if df is None or len(df) < MIN_ROWS:
 196:         return {
 197:             "action": "HOLD",
 198:             "confidence": 0.0,
 199:             "sub_signals": {},
 200:             "indicators": {},
 201:         }
 202: 
 203:     close = df["close"].copy()
 204:     returns = close.pct_change(fill_method=None).dropna()
 205: 
 206:     if len(returns) < MIN_ROWS - 1:
 207:         return {
 208:             "action": "HOLD",
 209:             "confidence": 0.0,
 210:             "sub_signals": {},
 211:             "indicators": {},
 212:         }
 213: 
 214:     # Compute all sub-signals
 215:     conf1, vote1, ind1 = _sub_skew_zscore(returns)
 216:     conf2, vote2, ind2 = _sub_skew_momentum(returns)
 217:     conf3, vote3, ind3 = _sub_kurtosis_confirm(returns)
 218:     conf4, vote4, ind4 = _sub_skew_regime_divergence(returns)
 219: 
 220:     sub_signals = {

>>> portfolio/signals/statistical_jump_regime.py:1-220
   1: """Statistical Jump Model regime signal module.
   2: 
   3: Detects market regime (bull/bear/neutral) using statistical jump detection
   4: with a persistence penalty to reduce whiplash from frequent regime flips.
   5: 
   6: Based on: Shu, Yu, Mulvey 2024 — "Statistical Jump Model for regime detection"
   7: (12 citations). Simplified implementation using threshold-based jump detection
   8: instead of full EM algorithm. Core insight preserved: persistence penalty
   9: penalizes frequent regime switches.
  10: 
  11: Sub-indicators:
  12:     1. Jump Detection     — returns exceeding k*vol threshold
  13:     2. Regime Persistence — consecutive bars in current regime (min N to act)
  14:     3. Volatility Regime  — low/normal/high vol classification
  15:     4. Trend Confirmation — SMA slope direction alignment
  16: 
  17: Requires a pandas DataFrame with columns: open, high, low, close, volume
  18: and at least 50 rows of data (for rolling volatility + SMA baseline).
  19: """
  20: 
  21: from __future__ import annotations
  22: 
  23: import numpy as np
  24: import pandas as pd
  25: 
  26: from portfolio.signal_utils import majority_vote, safe_float, sma
  27: 
  28: # ---------------------------------------------------------------------------
  29: # Parameters
  30: # ---------------------------------------------------------------------------
  31: MIN_ROWS = 50          # Need enough history for rolling stats
  32: RETURN_WINDOW = 20     # Window for computing rolling returns
  33: VOL_WINDOW = 20        # Window for rolling volatility
  34: JUMP_THRESHOLD = 2.0   # Returns > k * vol = jump event
  35: PERSISTENCE_MIN = 3    # Minimum consecutive bars in regime to act
  36: SMA_PERIOD = 20        # Trend confirmation SMA period
  37: VOL_LOW_PCTILE = 25    # Below this percentile = low-vol regime
  38: VOL_HIGH_PCTILE = 75   # Above this percentile = high-vol regime
  39: REGIME_DECAY = 10      # Bars before regime confidence starts decaying
  40: 
  41: 
  42: def _detect_jumps(returns: pd.Series, vol: pd.Series,
  43:                   threshold: float = JUMP_THRESHOLD) -> pd.Series:
  44:     """Detect jump events where abs(return) > threshold * rolling vol.
  45: 
  46:     Returns a Series of +1 (positive jump), -1 (negative jump), 0 (no jump).
  47:     """
  48:     jump_up = (returns > threshold * vol).astype(int)
  49:     jump_down = (returns < -threshold * vol).astype(int)
  50:     return jump_up - jump_down
  51: 
  52: 
  53: def _classify_vol_regime(vol: pd.Series, window: int = 252) -> pd.Series:
  54:     """Classify volatility regime using rolling percentile rank.
  55: 
  56:     Returns: 'low_vol', 'normal', or 'high_vol' for each bar.
  57:     """
  58:     # Use available history up to `window` bars for percentile rank
  59:     effective_window = min(window, len(vol))
  60:     if effective_window < 20:
  61:         return pd.Series("normal", index=vol.index)
  62: 
  63:     rank = vol.rolling(window=effective_window, min_periods=20).apply(
  64:         lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
  65:     )
  66: 
  67:     result = pd.Series("normal", index=vol.index)
  68:     result[rank < VOL_LOW_PCTILE / 100] = "low_vol"
  69:     result[rank > VOL_HIGH_PCTILE / 100] = "high_vol"
  70:     return result
  71: 
  72: 
  73: def _compute_regime_with_persistence(jumps: pd.Series,
  74:                                      persistence_min: int = PERSISTENCE_MIN
  75:                                      ) -> tuple[list[str], list[int]]:
  76:     """Apply persistence penalty: track regime and count consecutive bars.
  77: 
  78:     Regime only changes after persistence_min consecutive opposing signals.
  79: 
  80:     Returns:
  81:         regimes: list of 'bull', 'bear', or 'neutral'
  82:         persistence: list of consecutive bars in current regime
  83:     """
  84:     n = len(jumps)
  85:     regimes = ["neutral"] * n
  86:     persistence = [0] * n
  87:     current_regime = "neutral"
  88:     current_count = 0
  89:     opposing_count = 0
  90: 
  91:     for i in range(n):
  92:         j = jumps.iloc[i]
  93: 
  94:         if j == 0:
  95:             # No jump — maintain current regime, reset opposing count slowly
  96:             opposing_count = max(0, opposing_count - 1)
  97:         elif current_regime == "neutral":
  98:             # In neutral, any jump starts a directional regime attempt
  99:             if j > 0:
 100:                 opposing_count = 0
 101:                 current_count += 1
 102:                 if current_count >= persistence_min:
 103:                     current_regime = "bull"
 104:                     current_count = persistence_min
 105:             elif j < 0:
 106:                 opposing_count = 0
 107:                 current_count += 1
 108:                 if current_count >= persistence_min:
 109:                     current_regime = "bear"
 110:                     current_count = persistence_min
 111:         elif current_regime == "bull":
 112:             if j > 0:
 113:                 current_count += 1
 114:                 opposing_count = 0
 115:             elif j < 0:
 116:                 opposing_count += 1
 117:                 if opposing_count >= persistence_min:
 118:                     current_regime = "bear"
 119:                     current_count = opposing_count
 120:                     opposing_count = 0
 121:         elif current_regime == "bear":
 122:             if j < 0:
 123:                 current_count += 1
 124:                 opposing_count = 0
 125:             elif j > 0:
 126:                 opposing_count += 1
 127:                 if opposing_count >= persistence_min:
 128:                     current_regime = "bull"
 129:                     current_count = opposing_count
 130:                     opposing_count = 0
 131: 
 132:         regimes[i] = current_regime
 133:         persistence[i] = current_count
 134: 
 135:     return regimes, persistence
 136: 
 137: 
 138: def compute_statistical_jump_regime_signal(
 139:     df: pd.DataFrame, context: dict = None
 140: ) -> dict:
 141:     """Compute Statistical Jump Model regime signal.
 142: 
 143:     Args:
 144:         df: DataFrame with columns: open, high, low, close, volume
 145:             (minimum 50 rows)
 146:         context: Optional dict with keys: ticker, config, asset_class, regime
 147: 
 148:     Returns:
 149:         dict with keys:
 150:             action: "BUY" | "SELL" | "HOLD"
 151:             confidence: float 0.0-1.0
 152:             sub_signals: dict of sub-indicator votes
 153:             indicators: dict of raw indicator values
 154:     """
 155:     empty = {
 156:         "action": "HOLD", "confidence": 0.0,
 157:         "sub_signals": {}, "indicators": {},
 158:     }
 159: 
 160:     if df is None or len(df) < MIN_ROWS:
 161:         return empty
 162: 
 163:     close = df["close"].astype(float)
 164:     if close.isna().sum() > len(close) * 0.3:
 165:         return empty
 166: 
 167:     # Forward-fill small NaN gaps
 168:     close = close.ffill()
 169: 
 170:     # --- Sub-indicator 1: Jump Detection ---
 171:     log_returns = np.log(close / close.shift(1)).dropna()
 172:     if len(log_returns) < RETURN_WINDOW:
 173:         return empty
 174: 
 175:     rolling_vol = log_returns.rolling(window=VOL_WINDOW, min_periods=10).std()
 176:     jumps = _detect_jumps(log_returns, rolling_vol)
 177: 
 178:     # --- Sub-indicator 2: Regime with Persistence ---
 179:     regimes, persistence = _compute_regime_with_persistence(jumps)
 180:     current_regime = regimes[-1]
 181:     current_persistence = persistence[-1]
 182: 
 183:     # Jump detection vote
 184:     if current_regime == "bull" and current_persistence >= PERSISTENCE_MIN:
 185:         jump_vote = "BUY"
 186:     elif current_regime == "bear" and current_persistence >= PERSISTENCE_MIN:
 187:         jump_vote = "SELL"
 188:     else:
 189:         jump_vote = "HOLD"
 190: 
 191:     # --- Sub-indicator 3: Volatility Regime ---
 192:     vol_regime = _classify_vol_regime(rolling_vol)
 193:     current_vol_regime = vol_regime.iloc[-1] if len(vol_regime) > 0 else "normal"
 194: 
 195:     # In high-vol regimes, trend signals are less reliable → bias toward HOLD
 196:     # In low-vol regimes, trend signals are more reliable → amplify
 197:     if current_vol_regime == "high_vol":
 198:         vol_vote = "HOLD"  # High vol = uncertain, don't add directional signal
 199:     elif current_vol_regime == "low_vol":
 200:         # Low vol tends to precede breakouts — use trend direction
 201:         sma_val = sma(close, SMA_PERIOD)
 202:         if len(sma_val) > 0 and not np.isnan(sma_val.iloc[-1]):
 203:             slope = (sma_val.iloc[-1] - sma_val.iloc[-5]) if len(sma_val) >= 5 else 0
 204:             vol_vote = "BUY" if slope > 0 else "SELL" if slope < 0 else "HOLD"
 205:         else:
 206:             vol_vote = "HOLD"
 207:     else:
 208:         vol_vote = "HOLD"
 209: 
 210:     # --- Sub-indicator 4: Trend Confirmation ---
 211:     sma_series = sma(close, SMA_PERIOD)
 212:     if len(sma_series) >= 5:
 213:         sma_current = sma_series.iloc[-1]
 214:         sma_prev = sma_series.iloc[-5]
 215:         if not (np.isnan(sma_current) or np.isnan(sma_prev)):
 216:             sma_slope = (sma_current - sma_prev) / sma_prev if sma_prev != 0 else 0
 217:             if sma_slope > 0.005:
 218:                 trend_vote = "BUY"
 219:             elif sma_slope < -0.005:
 220:                 trend_vote = "SELL"

>>> portfolio/signals/hurst_regime.py:1-220
   1: """Hurst exponent regime detector signal module.
   2: 
   3: Computes the rolling Hurst exponent via Rescaled Range (R/S) analysis to
   4: classify the current market regime as trending, mean-reverting, or random walk.
   5: Produces directional votes based on the detected regime:
   6: 
   7:     - Trending (H > 0.55): vote in trend direction (EMA slope)
   8:     - Mean-reverting (H < 0.45): vote contrarian at RSI extremes
   9:     - Random walk (0.45 <= H <= 0.55): HOLD (no edge)
  10: 
  11: Sub-signals:
  12:     1. Hurst Regime     — regime classification from R/S analysis
  13:     2. Trend Direction   — EMA(9)/EMA(21) slope when trending
  14:     3. MR Extreme        — RSI(14) contrarian when mean-reverting
  15:     4. Hurst Momentum    — rate-of-change of H (regime shift detection)
  16: 
  17: Requires a pandas DataFrame with columns: open, high, low, close, volume
  18: and at least 160 rows of data (for 150-bar R/S window + buffer).
  19: """
  20: 
  21: from __future__ import annotations
  22: 
  23: import logging
  24: 
  25: import numpy as np
  26: import pandas as pd
  27: 
  28: logger = logging.getLogger(__name__)
  29: 
  30: from portfolio.signal_utils import ema, majority_vote, rsi, safe_float
  31: 
  32: # Minimum rows for the R/S analysis window plus lookback buffer.
  33: # The largest sub-window is 128 bars, and we need at least 2 chunks,
  34: # so 256 minimum for R/S.  We use 160 rows as the practical minimum
  35: # (smaller sub-windows still produce valid estimates).
  36: MIN_ROWS = 160
  37: 
  38: # R/S sub-window sizes (powers of 2 for clean chunking)
  39: SUB_WINDOWS = (8, 16, 32, 64, 128)
  40: 
  41: # Regime thresholds (from FractalCycles guide + literature consensus)
  42: TRENDING_THRESHOLD = 0.55
  43: MR_THRESHOLD = 0.45
  44: 
  45: # Hurst momentum lookback for rate-of-change
  46: HURST_ROC_LOOKBACK = 10
  47: 
  48: 
  49: # ── R/S Analysis ─────────────────────────────────────────────────────────
  50: 
  51: def _rescaled_range(returns: np.ndarray, n: int) -> float:
  52:     """Compute mean R/S statistic for sub-window size *n*.
  53: 
  54:     Splits *returns* into non-overlapping chunks of size *n*, computes the
  55:     rescaled range for each chunk, and returns the mean.
  56:     """
  57:     n_chunks = len(returns) // n
  58:     if n_chunks < 1:
  59:         return np.nan
  60: 
  61:     rs_values = []
  62:     for i in range(n_chunks):
  63:         chunk = returns[i * n : (i + 1) * n]
  64:         mean_c = chunk.mean()
  65:         cumdev = np.cumsum(chunk - mean_c)
  66:         r = cumdev.max() - cumdev.min()
  67:         s = chunk.std(ddof=1)
  68:         if s > 0 and np.isfinite(r):
  69:             rs_values.append(r / s)
  70: 
  71:     return float(np.mean(rs_values)) if rs_values else np.nan
  72: 
  73: 
  74: def _compute_hurst(returns: np.ndarray,
  75:                    sub_windows: tuple[int, ...] = SUB_WINDOWS) -> float:
  76:     """Compute Hurst exponent via R/S analysis.
  77: 
  78:     For each sub-window size, compute the mean R/S.  Then fit
  79:     log(R/S) = H * log(n) + c  via OLS to get the Hurst exponent H.
  80: 
  81:     Returns NaN if fewer than 3 valid sub-window sizes.
  82:     """
  83:     log_rs = []
  84:     log_n = []
  85: 
  86:     for n in sub_windows:
  87:         if len(returns) < n * 2:
  88:             continue
  89:         rs = _rescaled_range(returns, n)
  90:         if np.isfinite(rs) and rs > 0:
  91:             log_rs.append(np.log(rs))
  92:             log_n.append(np.log(n))
  93: 
  94:     if len(log_n) < 3:
  95:         return np.nan
  96: 
  97:     # OLS: H = slope of log(R/S) vs log(n)
  98:     x = np.array(log_n)
  99:     y = np.array(log_rs)
 100:     x_mean = x.mean()
 101:     y_mean = y.mean()
 102:     ss_xx = np.sum((x - x_mean) ** 2)
 103:     if ss_xx == 0:
 104:         return np.nan
 105: 
 106:     h = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
 107:     return np.clip(h, 0.0, 1.0)
 108: 
 109: 
 110: # ── Sub-signal 1: Hurst Regime Classification ────────────────────────────
 111: 
 112: def _hurst_regime(h: float) -> str:
 113:     """Classify regime from Hurst exponent value."""
 114:     if np.isnan(h):
 115:         return "unknown"
 116:     if h > TRENDING_THRESHOLD:
 117:         return "trending"
 118:     if h < MR_THRESHOLD:
 119:         return "mean_reverting"
 120:     return "random_walk"
 121: 
 122: 
 123: # ── Sub-signal 2: Trend Direction (when trending) ────────────────────────
 124: 
 125: def _trend_direction(close: pd.Series) -> tuple[float, str]:
 126:     """Determine trend direction via EMA(9) vs EMA(21) spread.
 127: 
 128:     Returns (spread_pct, vote).
 129:     """
 130:     if len(close) < 21:
 131:         return 0.0, "HOLD"
 132: 
 133:     ema9 = ema(close, span=9)
 134:     ema21 = ema(close, span=21)
 135:     e9 = float(ema9.iloc[-1])
 136:     e21 = float(ema21.iloc[-1])
 137: 
 138:     if np.isnan(e9) or np.isnan(e21) or e21 == 0:
 139:         return 0.0, "HOLD"
 140: 
 141:     spread = (e9 - e21) / e21 * 100  # percent spread
 142: 
 143:     if spread > 0.3:
 144:         return safe_float(spread), "BUY"
 145:     if spread < -0.3:
 146:         return safe_float(spread), "SELL"
 147:     return safe_float(spread), "HOLD"
 148: 
 149: 
 150: # ── Sub-signal 3: Mean-Reversion Extreme (when MR) ──────────────────────
 151: 
 152: def _mr_extreme(close: pd.Series) -> tuple[float, str]:
 153:     """Contrarian signal at RSI extremes (for mean-reverting regime).
 154: 
 155:     Returns (rsi_value, vote).
 156:     """
 157:     if len(close) < 15:
 158:         return float("nan"), "HOLD"
 159: 
 160:     rsi_vals = rsi(close, period=14)
 161:     val = rsi_vals.iloc[-1]
 162: 
 163:     if np.isnan(val):
 164:         return float("nan"), "HOLD"
 165: 
 166:     val = float(val)
 167:     if val < 30:
 168:         return val, "BUY"
 169:     if val > 70:
 170:         return val, "SELL"
 171:     return val, "HOLD"
 172: 
 173: 
 174: # ── Sub-signal 4: Hurst Momentum (regime shift detection) ───────────────
 175: 
 176: def _hurst_momentum(close: pd.Series, window: int = MIN_ROWS,
 177:                     roc_lookback: int = HURST_ROC_LOOKBACK) -> tuple[float, float, str]:
 178:     """Compute rate-of-change of Hurst exponent.
 179: 
 180:     Rising H (positive ROC) = trend strengthening -> favor trend direction.
 181:     Falling H (negative ROC) = trend weakening -> caution / favor MR.
 182: 
 183:     Returns (hurst_roc, current_h, vote).
 184:     """
 185:     needed = window + roc_lookback
 186:     if len(close) < needed:
 187:         return float("nan"), float("nan"), "HOLD"
 188: 
 189:     returns = close.pct_change(fill_method=None).dropna().values
 190: 
 191:     # Current Hurst
 192:     h_now = _compute_hurst(returns[-window:])
 193:     # Hurst roc_lookback bars ago
 194:     h_prev = _compute_hurst(returns[-(window + roc_lookback):-roc_lookback])
 195: 
 196:     if np.isnan(h_now) or np.isnan(h_prev):
 197:         return float("nan"), safe_float(h_now), "HOLD"
 198: 
 199:     roc = h_now - h_prev
 200: 
 201:     # Strong rising Hurst = trend strengthening -> vote with trend
 202:     # Strong falling Hurst = trend weakening -> favor caution
 203:     if roc > 0.05:
 204:         return safe_float(roc), safe_float(h_now), "BUY"   # trend strengthening
 205:     if roc < -0.05:
 206:         return safe_float(roc), safe_float(h_now), "SELL"   # trend weakening
 207:     return safe_float(roc), safe_float(h_now), "HOLD"
 208: 
 209: 
 210: # ── Public API ───────────────────────────────────────────────────────────
 211: 
 212: def compute_hurst_regime_signal(df: pd.DataFrame,
 213:                                 context: dict | None = None) -> dict:
 214:     """Compute Hurst regime detector signal from OHLCV data.
 215: 
 216:     Parameters
 217:     ----------
 218:     df : pd.DataFrame
 219:         Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``
 220:         with at least 160 rows.

codex
I’ve confirmed the code paths for the sector map and structure module. The remaining work is mostly registry/loader plumbing plus whether the “disabled” and “shadow mode” claims are actually reflected in the codebase, not just asserted in prose.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "signal_registry|SIGNAL_REGISTRY|DISABLED_SIGNALS|ENHANCED_SIGNALS|compute_.*_signal|treasury_risk_rotation" portfolio -S' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/smart_money.py'; Start=1; End=80},
  @{Path='portfolio/signals/oscillators.py'; Start=1; End=80},
  @{Path='portfolio/signals/orderbook_flow.py'; Start=1; End=80}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-ChildItem -Recurse portfolio | Select-String -Pattern 'disabled' -CaseSensitive:"'$false | Select-Object Path, LineNumber, Line | Format-Table -AutoSize | Out-String -Width 220' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse portfolio -Filter *registry*.py | ForEach-Object { $_.FullName }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 808ms:
portfolio\signals\credit_spread.py:250:def compute_credit_spread_signal(
portfolio\signals\cross_asset_tsmom.py:174:def compute_cross_asset_tsmom_signal(
portfolio\signals\cot_positioning.py:309:def compute_cot_positioning_signal(
portfolio\signals\complexity_gap_regime.py:200:def compute_complexity_gap_regime_signal(
portfolio\signals\drift_regime_gate.py:134:def compute_drift_regime_gate_signal(
portfolio\signals\econ_calendar.py:176:def compute_econ_calendar_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\copper_gold_ratio.py:169:def compute_copper_gold_ratio_signal(
portfolio\signals\claude_fundamental.py:894:def compute_claude_fundamental_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\dxy_cross_asset.py:37:def compute_dxy_cross_asset_signal(
portfolio\signals\crypto_cross_asset.py:168:def compute_crypto_cross_asset_signal(df: pd.DataFrame,
portfolio\signals\crypto_macro.py:197:def compute_crypto_macro_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\forecast.py:733:def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\crypto_evrp.py:264:def compute_crypto_evrp_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\candlestick.py:33:def compute_candlestick_signal(df: pd.DataFrame) -> dict:
portfolio\signals\fibonacci.py:373:def compute_fibonacci_signal(df: pd.DataFrame) -> dict:
portfolio\signals\futures_basis.py:184:def compute_futures_basis_signal(df=None, context: dict = None) -> dict:
portfolio\signals\calendar_seasonal.py:319:def compute_calendar_signal(df: pd.DataFrame) -> dict:
portfolio\signals\futures_flow.py:223:def compute_futures_flow_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\gold_real_yield_paradox.py:240:def compute_gold_real_yield_paradox_signal(
portfolio\signals\hash_ribbons.py:224:def compute_hash_ribbons_signal(
portfolio\signals\xtrend_equity_spillover.py:187:def compute_xtrend_equity_spillover_signal(
portfolio\signals\williams_vix_fix.py:173:def compute_williams_vix_fix_signal(
portfolio\signals\vol_ratio_regime.py:128:def compute_vol_ratio_regime_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\volume_flow.py:232:def compute_volume_flow_signal(df: pd.DataFrame) -> dict[str, Any]:
portfolio\signals\volatility.py:292:def compute_volatility_signal(df: pd.DataFrame) -> dict[str, Any]:
portfolio\signals\vix_term_structure.py:141:def compute_vix_term_structure_signal(df=None, context=None, **kwargs) -> dict:
portfolio\signals\trend.py:432:def compute_trend_signal(df: pd.DataFrame) -> dict:
portfolio\signals\treasury_risk_rotation.py:31:logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")
portfolio\signals\treasury_risk_rotation.py:62:                logger.warning("treasury_risk_rotation: missing IEF or TLT column")
portfolio\signals\treasury_risk_rotation.py:69:                logger.warning("treasury_risk_rotation: insufficient data rows")
portfolio\signals\treasury_risk_rotation.py:74:            logger.warning("treasury_risk_rotation yfinance fetch failed: %s", e)
portfolio\signals\treasury_risk_rotation.py:77:    return _cached("treasury_risk_rotation_yf", _CACHE_TTL, _do_fetch)
portfolio\signals\treasury_risk_rotation.py:153:def compute_treasury_risk_rotation_signal(
portfolio\signals\mahalanobis_turbulence.py:245:def compute_mahalanobis_turbulence_signal(
portfolio\signals\network_momentum.py:307:def compute_network_momentum_signal(
portfolio\signals\momentum.py:288:def compute_momentum_signal(df: pd.DataFrame) -> dict:
portfolio\signals\metals_cross_asset.py:266:def compute_metals_cross_asset_signal(
portfolio\signals\orderbook_flow.py:92:def compute_orderbook_flow_signal(
portfolio\signals\shannon_entropy.py:171:def compute_shannon_entropy_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\realized_skewness.py:183:def compute_realized_skewness_signal(
portfolio\signals\smart_money.py:401:def compute_smart_money_signal(df: pd.DataFrame) -> dict[str, Any]:
portfolio\signals\news_event.py:499:def compute_news_event_signal(df: pd.DataFrame, context: dict = None) -> dict:
portfolio\signals\macro_regime.py:301:def compute_macro_regime_signal(df: pd.DataFrame, macro: dict = None) -> dict:
portfolio\signals\hurst_regime.py:212:def compute_hurst_regime_signal(df: pd.DataFrame,
portfolio\signals\momentum_factors.py:365:def compute_momentum_factors_signal(df: pd.DataFrame, context: dict | None = None) -> dict:
portfolio\signals\residual_pair_reversion.py:237:def compute_residual_pair_reversion_signal(
portfolio\signals\mean_reversion.py:398:def compute_mean_reversion_signal(df: pd.DataFrame, context: dict | None = None) -> dict:
portfolio\signals\ovx_metals_spillover.py:166:def compute_ovx_metals_spillover_signal(
portfolio\signals\statistical_jump_regime.py:138:def compute_statistical_jump_regime_signal(
portfolio\signals\oscillators.py:434:def compute_oscillator_signal(df: pd.DataFrame) -> dict:
portfolio\signals\heikin_ashi.py:481:def compute_heikin_ashi_signal(df: pd.DataFrame) -> dict:
portfolio\signals\structure.py:177:def compute_structure_signal(df: pd.DataFrame) -> dict:

 succeeded in 820ms:
 succeeded in 936ms:

>>> portfolio/signals/smart_money.py:1-80
   1: """Composite Smart Money Concepts (SMC) and market structure signal.
   2: 
   3: Combines five sub-indicators into a single BUY/SELL/HOLD vote via majority
   4: voting:
   5:     1. Break of Structure (BOS) - swing high/low breakouts
   6:     2. Change of Character (CHoCH) - trend reversal detection
   7:     3. Fair Value Gap (FVG) - unfilled 3-candle gaps being revisited
   8:     4. Liquidity Sweep / Stop Hunt - wick-based fake-out reversals
   9:     5. Supply and Demand Zones - institutional order flow zones
  10: 
  11: Requires a pandas DataFrame with columns: open, high, low, close, volume.
  12: At least 50 rows recommended; returns HOLD on insufficient data.
  13: """
  14: 
  15: from __future__ import annotations
  16: 
  17: import logging
  18: from typing import Any
  19: 
  20: import numpy as np
  21: import pandas as pd
  22: 
  23: from portfolio.signal_utils import majority_vote
  24: 
  25: logger = logging.getLogger(__name__)
  26: 
  27: MIN_ROWS = 50
  28: _SWING_LOOKBACK = 3          # bars on each side for swing detection
  29: _FVG_SCAN_BARS = 20          # how far back to scan for unfilled FVGs
  30: _LIQUIDITY_SWEEP_PCT = 0.005 # wick must exceed extreme by >0.5%
  31: _STRONG_BODY_MULT = 2.0      # body > 2x avg body = strong candle
  32: _SUPPLY_DEMAND_LOOKBACK = 30 # bars to scan for S/D zones
  33: _ZONE_PROXIMITY_PCT = 0.005  # within 0.5% of zone boundary counts as "in zone"
  34: 
  35: 
  36: # ---------------------------------------------------------------------------
  37: # Helpers
  38: # ---------------------------------------------------------------------------
  39: 
  40: def _body(open_s: pd.Series, close_s: pd.Series) -> pd.Series:
  41:     """Absolute candle body size."""
  42:     return (close_s - open_s).abs()
  43: 
  44: 
  45: def _find_swing_highs(highs: np.ndarray, lookback: int = _SWING_LOOKBACK) -> list[tuple[int, float]]:
  46:     """Find swing highs: bar whose high > high of `lookback` bars on each side.
  47: 
  48:     Returns list of (index, high_value) tuples sorted by index ascending.
  49:     """
  50:     swings: list[tuple[int, float]] = []
  51:     n = len(highs)
  52:     for i in range(lookback, n - lookback):
  53:         is_swing = True
  54:         for j in range(1, lookback + 1):
  55:             if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
  56:                 is_swing = False
  57:                 break
  58:         if is_swing:
  59:             swings.append((i, float(highs[i])))
  60:     return swings
  61: 
  62: 
  63: def _find_swing_lows(lows: np.ndarray, lookback: int = _SWING_LOOKBACK) -> list[tuple[int, float]]:
  64:     """Find swing lows: bar whose low < low of `lookback` bars on each side.
  65: 
  66:     Returns list of (index, low_value) tuples sorted by index ascending.
  67:     """
  68:     swings: list[tuple[int, float]] = []
  69:     n = len(lows)
  70:     for i in range(lookback, n - lookback):
  71:         is_swing = True
  72:         for j in range(1, lookback + 1):
  73:             if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
  74:                 is_swing = False
  75:                 break
  76:         if is_swing:
  77:             swings.append((i, float(lows[i])))
  78:     return swings
  79: 
  80: 

>>> portfolio/signals/oscillators.py:1-80
   1: """Composite advanced oscillators signal module.
   2: 
   3: Computes 8 oscillator sub-indicators and returns a majority-vote composite
   4: BUY/SELL/HOLD signal with confidence score.
   5: 
   6: Sub-indicators:
   7:     1. Awesome Oscillator          (AO zero-cross + twin-peaks saucer)
   8:     2. Aroon Oscillator (25)       (trend strength via high/low recency)
   9:     3. Vortex Indicator (14)       (VI+ vs VI- directional movement)
  10:     4. Chande Momentum Osc (9)     (normalized momentum extremes)
  11:     5. Know Sure Thing (KST)       (multi-ROC composite + signal crossover)
  12:     6. Schaff Trend Cycle (23, 50) (MACD + double stochastic smoothing)
  13:     7. TRIX (15)                   (triple-smoothed EMA rate of change)
  14:     8. Coppock Curve (14, 11, 10)  (WMA of dual ROC, long-term buy signal)
  15: 
  16: Requires a pandas DataFrame with columns: open, high, low, close, volume
  17: and at least 50 rows of data.
  18: """
  19: 
  20: from __future__ import annotations
  21: 
  22: import numpy as np
  23: import pandas as pd
  24: 
  25: from portfolio.signal_utils import ema, majority_vote, roc, safe_float, sma, true_range, wma
  26: 
  27: # ---------------------------------------------------------------------------
  28: # Minimum rows required for reliable computation.  The longest lookback chain
  29: # is the Coppock Curve (ROC-14 needs 14 bars + WMA-10 needs 10 more = 24)
  30: # and Schaff Trend Cycle (50-period slow EMA warm-up), but we ask for 50
  31: # to give every indicator a reasonable warm-up.
  32: # ---------------------------------------------------------------------------
  33: MIN_ROWS = 50
  34: 
  35: 
  36: # ---- sub-indicator 1: Awesome Oscillator -----------------------------------
  37: 
  38: def _awesome_oscillator(high: pd.Series, low: pd.Series) -> tuple[float, str]:
  39:     """Awesome Oscillator: SMA(5, median) - SMA(34, median).
  40: 
  41:     - AO crosses above 0 = BUY
  42:     - AO crosses below 0 = SELL
  43:     - Twin peaks (saucer): AO below 0, two dips where second is higher = BUY
  44: 
  45:     Returns (ao_value, signal).
  46:     """
  47:     median_price = (high + low) / 2.0
  48:     ao = sma(median_price, 5) - sma(median_price, 34)
  49: 
  50:     val = ao.iloc[-1]
  51:     if np.isnan(val):
  52:         return float("nan"), "HOLD"
  53: 
  54:     # Need at least 2 values for crossover detection
  55:     if len(ao.dropna()) < 2:
  56:         return safe_float(val), "HOLD"
  57: 
  58:     prev = ao.dropna().iloc[-2]
  59:     if np.isnan(prev):
  60:         return safe_float(val), "HOLD"
  61: 
  62:     # Zero-line crossover
  63:     if prev <= 0 and val > 0:
  64:         return safe_float(val), "BUY"
  65:     if prev >= 0 and val < 0:
  66:         return safe_float(val), "SELL"
  67: 
  68:     # Twin peaks (saucer) detection: AO below zero, look for two local
  69:     # minima in recent history where second dip is higher (less negative).
  70:     # This is a bullish reversal signal.
  71:     ao_clean = ao.dropna()
  72:     if len(ao_clean) >= 10 and val < 0:
  73:         recent = ao_clean.iloc[-10:].values
  74:         # Find local minima (simple: value lower than both neighbours)
  75:         minima_indices = []
  76:         for i in range(1, len(recent) - 1):
  77:             if recent[i] < recent[i - 1] and recent[i] < recent[i + 1]:
  78:                 minima_indices.append(i)
  79: 
  80:         if len(minima_indices) >= 2:

>>> portfolio/signals/orderbook_flow.py:1-80
   1: """Orderbook flow signal — microstructure-based short-term prediction.
   2: 
   3: Signal #31.  Combines 6 microstructure sub-indicators via majority vote:
   4:     1. Depth Imbalance: ln(V_bid) - ln(V_ask) direction
   5:     2. Trade Flow Imbalance: signed volume ratio direction
   6:     3. VPIN Toxicity: high VPIN confirms directional flow
   7:     4. OFI Direction: order flow imbalance trend
   8:     5. Spread Health: abnormally wide spread → caution
   9:     6. Trade Pressure: trade-through imbalance (buy vs sell throughs)
  10: 
  11: Applicable to metals (XAU-USD, XAG-USD) and crypto (BTC-USD, ETH-USD).
  12: Requires context: calls metals_orderbook + microstructure modules.
  13: Returns HOLD for stock tickers or when data unavailable.
  14: """
  15: from __future__ import annotations
  16: 
  17: import logging
  18: from typing import Any
  19: 
  20: from portfolio.signal_utils import majority_vote
  21: 
  22: logger = logging.getLogger("portfolio.signals.orderbook_flow")
  23: 
  24: _APPLICABLE_TICKERS = {"XAU-USD", "XAG-USD", "BTC-USD", "ETH-USD"}
  25: 
  26: _DEPTH_IMBALANCE_THRESHOLD = 0.5
  27: _TRADE_IMBALANCE_THRESHOLD = 0.3
  28: _VPIN_HIGH = 0.6
  29: _VPIN_TOXICITY = 0.7  # threshold for flagging vol expansion risk
  30: _OFI_ZSCORE_THRESHOLD = 1.5  # z-score threshold (was absolute 5.0)
  31: _OFI_THRESHOLD = 5.0  # fallback when z-score unavailable
  32: _SPREAD_ZSCORE_DANGER = 2.0
  33: _TRADE_THROUGH_THRESHOLD = 2
  34: 
  35: 
  36: def _get_microstructure_context(ticker: str) -> dict | None:
  37:     """Fetch live microstructure data for the given ticker."""
  38:     try:
  39:         from portfolio.metals_orderbook import get_orderbook_depth, get_recent_trades
  40:         from portfolio.microstructure import (
  41:             compute_vpin,
  42:             depth_imbalance,
  43:             detect_trade_throughs,
  44:             trade_flow_imbalance,
  45:         )
  46:     except ImportError:
  47:         logger.debug("Microstructure modules not available")
  48:         return None
  49: 
  50:     depth = get_orderbook_depth(ticker, limit=20)
  51:     trades = get_recent_trades(ticker, limit=200)
  52:     if depth is None or trades is None:
  53:         return None
  54: 
  55:     di = depth_imbalance(depth)
  56:     tfi = trade_flow_imbalance(trades)
  57:     vpin = compute_vpin(trades, n_buckets=20)
  58:     tt = detect_trade_throughs(trades)
  59: 
  60:     if tfi is None:
  61:         return None
  62: 
  63:     # Read accumulated OFI, OFI z-score, multiscale, and spread z-score
  64:     ofi = 0.0
  65:     ofi_zscore = 0.0
  66:     sz = 0.0
  67:     flow_acceleration = 0.0
  68:     try:
  69:         from portfolio.microstructure_state import load_persisted_state
  70:         ms_state = load_persisted_state(ticker)
  71:         if ms_state:
  72:             ofi = ms_state.get("ofi", 0.0)
  73:             ofi_zscore = ms_state.get("ofi_zscore", 0.0)
  74:             sz = ms_state.get("spread_zscore", 0.0)
  75:             flow_acceleration = ms_state.get("flow_acceleration", 0.0)
  76:     except ImportError:
  77:         pass
  78: 
  79:     return {
  80:         "depth_imbalance": di,

 succeeded in 959ms:

Path                                                                             LineNumber Line
----                                                                             ---------- ----
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py         50 _FORECAST_MODELS_DISABLED = False
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        760     # Bulletproof early return — skip ALL work when models are disabled
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        761     if _FORECAST_MODELS_DISABLED:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py        762         result["indicators"]["models_disabled"] = True



exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"DISABLED_SIGNALS|ENHANCED_SIGNALS|SIGNAL_MODULES|compute_.*_signal\\(|importlib|portfolio\\.signals\\.|treasury_risk_rotation|smart_money|oscillators|orderbook_flow|realized_skewness|statistical_jump_regime|hurst_regime\" -S" in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Where-Object { $_.Extension -eq '"'.py' } | Select-String -Pattern 'portfolio\\.signals' | Select-Object Path, LineNumber, Line | Format-Table -AutoSize | Out-String -Width 240" in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-10T15:27:19.673401Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 624ms:
2026-05-10T15:27:19.861739Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:

Path                                                                                            LineNumber Line
----                                                                                            ---------- ----
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\calendar_seasonal.py               27 logger = logging.getLogger("portfolio.signals.calendar_seasonal")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py              27 logger = logging.getLogger("portfolio.signals.claude_fundamental")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\complexity_gap_regime.py           37 logger = logging.getLogger("portfolio.signals.complexity_gap_regime")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\cot_positioning.py                 25 logger = logging.getLogger("portfolio.signals.cot_positioning")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\credit_spread.py                   29 logger = logging.getLogger("portfolio.signals.credit_spread")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\cross_asset_tsmom.py               33 logger = logging.getLogger("portfolio.signals.cross_asset_tsmom")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\crypto_cross_asset.py              30 logger = logging.getLogger("portfolio.signals.crypto_cross_asset")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\crypto_evrp.py                     37 logger = logging.getLogger("portfolio.signals.crypto_evrp")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\crypto_macro.py                    25 logger = logging.getLogger("portfolio.signals.crypto_macro")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\dxy_cross_asset.py                 24 logger = logging.getLogger("portfolio.signals.dxy_cross_asset")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\econ_calendar.py                   33 logger = logging.getLogger("portfolio.signals.econ_calendar")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py                        34 logger = logging.getLogger("portfolio.signals.forecast")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\futures_basis.py                   33 logger = logging.getLogger("portfolio.signals.futures_basis")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\futures_flow.py                    27 logger = logging.getLogger("portfolio.signals.futures_flow")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\hash_ribbons.py                    33 logger = logging.getLogger("portfolio.signals.hash_ribbons")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\macro_regime.py                    29 logger = logging.getLogger("portfolio.signals.macro_regime")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\mahalanobis_turbulence.py          40 logger = logging.getLogger("portfolio.signals.mahalanobis_turbulence")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\metals_cross_asset.py              39 logger = logging.getLogger("portfolio.signals.metals_cross_asset")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\network_momentum.py                34 logger = logging.getLogger("portfolio.signals.network_momentum")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\news_event.py                      33 logger = logging.getLogger("portfolio.signals.news_event")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\orderbook_flow.py                  22 logger = logging.getLogger("portfolio.signals.orderbook_flow")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\ovx_metals_spillover.py            36 logger = logging.getLogger("portfolio.signals.ovx_metals_spillover")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\residual_pair_reversion.py         39 logger = logging.getLogger("portfolio.signals.residual_pair_reversion")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\structure.py                       23 logger = logging.getLogger("portfolio.signals.structure")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\treasury_risk_rotation.py          31 logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\vix_term_structure.py              30 logger = logging.getLogger("portfolio.signals.vix_term_structure")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\xtrend_equity_spillover.py         32 logger = logging.getLogger("portfolio.signals.xtrend_equity_spillover")


[31;1mGet-ChildItem: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mGet-ChildItem -Recurse -File[0m | Where-Object { $_.Extension -eq '.py'  …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mAccess to the path 'Q:\finance-analyzer\.worktrees\adv-signals-modules\.pytest_tmp' is denied.[0m

 exited 1 in 812ms:

Path                                                                                            LineNumber Line
----                                                                                            ---------- ----
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\calendar_seasonal.py               27 logger = logging.getLogger("portfolio.signals.calendar_seasonal")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py              27 logger = logging.getLogger("portfolio.signals.claude_fundamental")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\complexity_gap_regime.py           37 logger = logging.getLogger("portfolio.signals.complexity_gap_regime")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\cot_positioning.py                 25 logger = logging.getLogger("portfolio.signals.cot_positioning")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\credit_spread.py                   29 logger = logging.getLogger("portfolio.signals.credit_spread")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\cross_asset_tsmom.py               33 logger = logging.getLogger("portfolio.signals.cross_asset_tsmom")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\crypto_cross_asset.py              30 logger = logging.getLogger("portfolio.signals.crypto_cross_asset")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\crypto_evrp.py                     37 logger = logging.getLogger("portfolio.signals.crypto_evrp")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\crypto_macro.py                    25 logger = logging.getLogger("portfolio.signals.crypto_macro")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\dxy_cross_asset.py                 24 logger = logging.getLogger("portfolio.signals.dxy_cross_asset")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\econ_calendar.py                   33 logger = logging.getLogger("portfolio.signals.econ_calendar")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\forecast.py                        34 logger = logging.getLogger("portfolio.signals.forecast")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\futures_basis.py                   33 logger = logging.getLogger("portfolio.signals.futures_basis")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\futures_flow.py                    27 logger = logging.getLogger("portfolio.signals.futures_flow")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\hash_ribbons.py                    33 logger = logging.getLogger("portfolio.signals.hash_ribbons")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\macro_regime.py                    29 logger = logging.getLogger("portfolio.signals.macro_regime")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\mahalanobis_turbulence.py          40 logger = logging.getLogger("portfolio.signals.mahalanobis_turbulence")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\metals_cross_asset.py              39 logger = logging.getLogger("portfolio.signals.metals_cross_asset")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\network_momentum.py                34 logger = logging.getLogger("portfolio.signals.network_momentum")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\news_event.py                      33 logger = logging.getLogger("portfolio.signals.news_event")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\orderbook_flow.py                  22 logger = logging.getLogger("portfolio.signals.orderbook_flow")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\ovx_metals_spillover.py            36 logger = logging.getLogger("portfolio.signals.ovx_metals_spillover")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\residual_pair_reversion.py         39 logger = logging.getLogger("portfolio.signals.residual_pair_reversion")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\structure.py                       23 logger = logging.getLogger("portfolio.signals.structure")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\treasury_risk_rotation.py          31 logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\vix_term_structure.py              30 logger = logging.getLogger("portfolio.signals.vix_term_structure")
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\xtrend_equity_spillover.py         32 logger = logging.getLogger("portfolio.signals.xtrend_equity_spillover")


[31;1mGet-ChildItem: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mGet-ChildItem -Recurse -File[0m | Where-Object { $_.Extension -eq '.py'  …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mAccess to the path 'Q:\finance-analyzer\.worktrees\adv-signals-modules\.pytest_tmp' is denied.[0m

codex
The loader/registry part is murkier because the claimed `signal_registry` file does not exist at the cited path. I’m tracing actual references to these modules in the repo now; if they aren’t wired anywhere, the “dead code still imported” claim is dead on arrival.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "smart_money|orderbook_flow|oscillators|treasury_risk_rotation|realized_skewness|hurst_regime|statistical_jump_regime" -g "!portfolio/signals/*" -g "!.pytest_tmp/*" -S .' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"importlib|__import__|pkgutil|glob\\(|portfolio/signals|portfolio\\.signals\" -g \""'!.pytest_tmp/*" -S .' in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-10T15:27:28.648448Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:

 exited 1 in 542ms:
 succeeded in 542ms:
.\portfolio\signals\complexity_gap_regime.py:37:logger = logging.getLogger("portfolio.signals.complexity_gap_regime")
.\portfolio\signals\crypto_cross_asset.py:3:Mirrors `portfolio/signals/metals_cross_asset.py` for the crypto subsystem.
.\portfolio\signals\crypto_cross_asset.py:30:logger = logging.getLogger("portfolio.signals.crypto_cross_asset")
.\portfolio\signals\dxy_cross_asset.py:24:logger = logging.getLogger("portfolio.signals.dxy_cross_asset")
.\portfolio\signals\crypto_evrp.py:37:logger = logging.getLogger("portfolio.signals.crypto_evrp")
.\portfolio\signals\crypto_macro.py:25:logger = logging.getLogger("portfolio.signals.crypto_macro")
.\portfolio\signals\cross_asset_tsmom.py:33:logger = logging.getLogger("portfolio.signals.cross_asset_tsmom")
.\portfolio\signals\claude_fundamental.py:27:logger = logging.getLogger("portfolio.signals.claude_fundamental")
.\portfolio\signals\calendar_seasonal.py:27:logger = logging.getLogger("portfolio.signals.calendar_seasonal")
.\portfolio\signals\econ_calendar.py:33:logger = logging.getLogger("portfolio.signals.econ_calendar")
.\portfolio\signals\forecast.py:34:logger = logging.getLogger("portfolio.signals.forecast")
.\portfolio\signals\futures_basis.py:33:logger = logging.getLogger("portfolio.signals.futures_basis")
.\portfolio\signals\futures_flow.py:27:logger = logging.getLogger("portfolio.signals.futures_flow")
.\portfolio\signals\cot_positioning.py:25:logger = logging.getLogger("portfolio.signals.cot_positioning")
.\portfolio\signals\credit_spread.py:29:logger = logging.getLogger("portfolio.signals.credit_spread")
.\portfolio\signals\hash_ribbons.py:33:logger = logging.getLogger("portfolio.signals.hash_ribbons")
.\portfolio\signals\mahalanobis_turbulence.py:40:logger = logging.getLogger("portfolio.signals.mahalanobis_turbulence")
.\portfolio\signals\macro_regime.py:29:logger = logging.getLogger("portfolio.signals.macro_regime")
.\portfolio\signals\network_momentum.py:34:logger = logging.getLogger("portfolio.signals.network_momentum")
.\portfolio\signals\metals_cross_asset.py:39:logger = logging.getLogger("portfolio.signals.metals_cross_asset")
.\portfolio\signals\metals_cross_asset.py:139:        data = resp.json() if hasattr(resp, "json") else __import__("json").loads(resp)
.\portfolio\signals\news_event.py:33:logger = logging.getLogger("portfolio.signals.news_event")
.\portfolio\signals\orderbook_flow.py:22:logger = logging.getLogger("portfolio.signals.orderbook_flow")
.\portfolio\signals\residual_pair_reversion.py:39:logger = logging.getLogger("portfolio.signals.residual_pair_reversion")
.\portfolio\signals\structure.py:23:logger = logging.getLogger("portfolio.signals.structure")
.\portfolio\signals\ovx_metals_spillover.py:36:logger = logging.getLogger("portfolio.signals.ovx_metals_spillover")
.\portfolio\signals\treasury_risk_rotation.py:31:logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")
.\portfolio\signals\vix_term_structure.py:30:logger = logging.getLogger("portfolio.signals.vix_term_structure")
.\portfolio\signals\xtrend_equity_spillover.py:32:logger = logging.getLogger("portfolio.signals.xtrend_equity_spillover")

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Force | Select-Object Name, Mode | Format-Table -AutoSize | Out-String -Width 200' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 591ms:

Name        Mode
----        ----
.pytest_tmp d----
portfolio   d----
.git        -a-h-
BASELINE.md -a---



codex
I’ve isolated the remaining ambiguous cases to helper functions: event time helpers, candle loader semantics, OAS/FRED value ordering, and where the sector map actually lives. I’m reading those next so the verdicts aren’t based on guessed data conventions.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/forecast.py'; Start=120; End=260},
  @{Path='portfolio/signals/forecast.py'; Start=430; End=740}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/econ_calendar.py'; Start=1; End=90},
  @{Path='portfolio/signals/econ_calendar.py'; Start=170; End=260},
  @{Path='portfolio/news_keywords.py'; Start=1; End=260}
)
foreach ("'$f in $files) {
  if (Test-Path $f.Path) {
    Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
    $lines = Get-Content $f.Path
    for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
      Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
    }
  } else {
    Write-Output ("MISSING: {0}" -f $f.Path)
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/metals_cross_asset.py'; Start=1; End=120},
  @{Path='portfolio/signals/credit_spread.py'; Start=1; End=140}
)
foreach ("'$f in $files) {
  Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
  $lines = Get-Content $f.Path
  for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
    Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "def _cached|class .*cached|def events_within_hours|def recent_high_impact_events|def next_event|TICKER_SECTORS|_load_candles_ohlcv|credit spread|OAS|FRED" portfolio -S' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 581ms:
portfolio\signals\forecast.py:248:def _load_candles_ohlcv(ticker: str, periods: int = 168,
portfolio\signals\forecast.py:782:    candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)
portfolio\signals\forecast.py:790:                                  _load_candles_ohlcv, ticker, kronos_periods,
portfolio\signals\econ_calendar.py:30:from portfolio.news_keywords import TICKER_SECTORS
portfolio\signals\econ_calendar.py:157:    indicators = {"ticker_sectors": list(TICKER_SECTORS.get(ticker, set())),
portfolio\signals\econ_calendar.py:164:    ticker_secs = TICKER_SECTORS.get(ticker, set())
portfolio\signals\gold_real_yield_paradox.py:10:Data: FRED DFII10 (10Y TIPS real yield), cached 4 hours.
portfolio\signals\gold_real_yield_paradox.py:29:_FRED_SERIES = "DFII10"
portfolio\signals\gold_real_yield_paradox.py:30:_FRED_TIMEOUT = 15
portfolio\signals\gold_real_yield_paradox.py:57:    """Fetch 10Y TIPS real yield from FRED. Returns list newest-first."""
portfolio\signals\gold_real_yield_paradox.py:68:        logger.debug("No FRED API key for real yield fetch")
portfolio\signals\gold_real_yield_paradox.py:81:                "series_id": _FRED_SERIES,
portfolio\signals\gold_real_yield_paradox.py:87:            timeout=_FRED_TIMEOUT,
portfolio\signals\gold_real_yield_paradox.py:115:        logger.warning("FRED real yield fetch failed", exc_info=True)
portfolio\signals\gold_real_yield_paradox.py:272:        logger.debug("No FRED API key — cannot compute GYDI")
portfolio\signals\gold_real_yield_paradox.py:290:    ])[::-1]  # reverse: FRED is newest-first, gold is oldest-first
portfolio\signals\credit_spread.py:3:Signal #35.  Uses ICE BofA High Yield Option-Adjusted Spread (HY OAS)
portfolio\signals\credit_spread.py:4:from FRED API as a measure of institutional credit risk appetite.
portfolio\signals\credit_spread.py:6:    1. OAS Level Z-Score:    where current spread sits vs 252d history
portfolio\signals\credit_spread.py:7:    2. OAS 5d Momentum:      rate of spread widening/tightening
portfolio\signals\credit_spread.py:8:    3. OAS Momentum Accel:   acceleration of momentum (early warning)
portfolio\signals\credit_spread.py:15:Data: FRED API series BAMLH0A0HYM2 (daily, free with API key).
portfolio\signals\credit_spread.py:18:Requires context dict with keys: ticker, config (for FRED API key).
portfolio\signals\credit_spread.py:40:_ZSCORE_RISK_OFF = 1.5      # OAS z-score above this = risk-off
portfolio\signals\credit_spread.py:41:_ZSCORE_RISK_ON = -1.0      # OAS z-score below this = risk-on (complacent)
portfolio\signals\credit_spread.py:45:_HISTORY_LIMIT = 300        # FRED observations to fetch (covers ~252 trading days)
portfolio\signals\credit_spread.py:46:_FRED_SERIES = "BAMLH0A0HYM2"
portfolio\signals\credit_spread.py:47:_FRED_TIMEOUT = 15
portfolio\signals\credit_spread.py:57:    """Fetch HY OAS history from FRED.  Returns list of floats (newest first).
portfolio\signals\credit_spread.py:70:        logger.debug("No FRED API key — cannot fetch HY OAS")
portfolio\signals\credit_spread.py:88:                "series_id": _FRED_SERIES,
portfolio\signals\credit_spread.py:94:            timeout=_FRED_TIMEOUT,
portfolio\signals\credit_spread.py:116:            logger.debug("HY OAS fetched: %d observations, current=%.2f", len(values), values[0])
portfolio\signals\credit_spread.py:120:        logger.warning("FRED HY OAS fetch failed", exc_info=True)
portfolio\signals\credit_spread.py:126:    """Extract FRED API key from context -> config."""
portfolio\signals\credit_spread.py:145:# Sub-indicator 1: OAS Level Z-Score
portfolio\signals\credit_spread.py:149:    """Z-score of current OAS vs 252d history."""
portfolio\signals\credit_spread.py:178:# Sub-indicator 2: OAS 5d Momentum
portfolio\signals\credit_spread.py:182:    """5-day change in OAS (widening vs tightening)."""
portfolio\signals\credit_spread.py:204:    """Acceleration of OAS momentum (early warning of regime shift)."""
portfolio\signals\credit_spread.py:228:    """Hard threshold: OAS above 500bp = crisis mode."""
portfolio\signals\credit_spread.py:253:    """Compute credit spread risk appetite signal.
portfolio\signals\credit_spread.py:256:        df: OHLCV DataFrame (unused — credit spread data fetched from FRED).
portfolio\signals\credit_spread.py:279:    # Get FRED API key from config
portfolio\signals\credit_spread.py:291:        logger.debug("No FRED API key available for credit spread signal")
portfolio\signals\credit_spread.py:294:    # Fetch HY OAS data
portfolio\signals\cot_positioning.py:279:    # Try to get FRED data from deep context
portfolio\signals\metals_cross_asset.py:24:2026-04-26: Added EPU + TIPS real yield from FRED API as sub-signals #7-8.
portfolio\signals\metals_cross_asset.py:67:# --- FRED-sourced macro indicators (daily, 4h cache) ---
portfolio\signals\metals_cross_asset.py:72:_EPU_FRED_SERIES = "USEPUINDXD"
portfolio\signals\metals_cross_asset.py:74:# TIPS real yield (10Y TIPS, FRED series DFII10).
portfolio\signals\metals_cross_asset.py:78:_TIPS_FRED_SERIES = "DFII10"
portfolio\signals\metals_cross_asset.py:80:_FRED_TIMEOUT = 15
portfolio\signals\metals_cross_asset.py:81:_FRED_CACHE_TTL = 4 * 3600
portfolio\signals\metals_cross_asset.py:82:_FRED_HISTORY_LIMIT = 300
portfolio\signals\metals_cross_asset.py:84:# Module-level FRED caches (separate dict per series)
portfolio\signals\metals_cross_asset.py:91:    """Extract FRED API key from context → config."""
portfolio\signals\metals_cross_asset.py:107:    """Fetch a FRED series.  Returns list of floats (newest first), cached 4h."""
portfolio\signals\metals_cross_asset.py:113:            and now - cache.get("time", 0) < _FRED_CACHE_TTL
portfolio\signals\metals_cross_asset.py:118:        logger.debug("No FRED API key — cannot fetch %s", series_id)
portfolio\signals\metals_cross_asset.py:124:        logger.debug("http_retry not available for FRED fetch")
portfolio\signals\metals_cross_asset.py:135:                "limit": _FRED_HISTORY_LIMIT,
portfolio\signals\metals_cross_asset.py:137:            timeout=_FRED_TIMEOUT,
portfolio\signals\metals_cross_asset.py:156:                "FRED %s fetched: %d values, latest=%.2f",
portfolio\signals\metals_cross_asset.py:161:        logger.warning("FRED %s fetch failed", series_id, exc_info=True)
portfolio\signals\metals_cross_asset.py:404:    # Sub 7: EPU (Economic Policy Uncertainty) — daily FRED
portfolio\signals\metals_cross_asset.py:408:    epu_values = _fetch_fred_values(_EPU_FRED_SERIES, fred_key, _epu_cache)
portfolio\signals\metals_cross_asset.py:418:    # Sub 8: TIPS Real Yield direction — daily FRED (DFII10)
portfolio\signals\metals_cross_asset.py:421:    tips_values = _fetch_fred_values(_TIPS_FRED_SERIES, fred_key, _tips_cache)
portfolio\signals\news_event.py:149:        from portfolio.news_keywords import TICKER_SECTORS
portfolio\signals\news_event.py:151:        ticker_secs = TICKER_SECTORS.get(ticker, set())

 succeeded in 628ms:

>>> portfolio/signals/econ_calendar.py:1-90
   1: """Economic calendar signal — event proximity, risk-off, and post-event relief.
   2: 
   3: Combines five sub-indicators into a majority-vote composite:
   4:   1. event_proximity    — hours until next event; <4h risk-off, <24h cautious
   5:   2. event_type         — classify event for informational purposes
   6:   3. pre_event_risk     — binary risk-off within 4h of high-impact event
   7:   4. sector_exposure    — is this ticker's sector affected by the upcoming event?
   8:   5. post_event_relief  — BUY after high-impact event passes (uncertainty premium removed)
   9: 
  10: BUG-218 fix: added post_event_relief sub-signal so the composite can emit BUY,
  11: not just SELL/HOLD. Previously structurally SELL-only.
  12: 
  13: The ``context`` parameter is a dict with keys: ticker, config, macro.
  14: Uses economic calendar dates from econ_dates.py and FOMC dates from fomc_dates.py.
  15: """
  16: 
  17: from __future__ import annotations
  18: 
  19: import logging
  20: from datetime import UTC, datetime
  21: 
  22: import pandas as pd
  23: 
  24: from portfolio.econ_dates import (
  25:     EVENT_SECTOR_MAP,
  26:     events_within_hours,
  27:     next_event,
  28:     recent_high_impact_events,
  29: )
  30: from portfolio.news_keywords import TICKER_SECTORS
  31: from portfolio.signal_utils import majority_vote
  32: 
  33: logger = logging.getLogger("portfolio.signals.econ_calendar")
  34: 
  35: # Max confidence cap
  36: _MAX_CONFIDENCE = 0.7
  37: 
  38: 
  39: def _get_current_date(df: pd.DataFrame) -> datetime:
  40:     """Extract current timestamp from df or use now()."""
  41:     if df is not None and "time" in df.columns and len(df) > 0:
  42:         last_time = df["time"].iloc[-1]
  43:         if isinstance(last_time, pd.Timestamp):
  44:             return last_time.to_pydatetime().replace(tzinfo=UTC)
  45:     return datetime.now(UTC)
  46: 
  47: 
  48: def _event_proximity(ref_date) -> tuple[str, dict]:
  49:     """Hours until next event → risk-off signal.
  50: 
  51:     <4h = SELL (risk-off before high-vol event)
  52:     4-24h = cautious SELL (only for high-impact events)
  53:     >24h = HOLD
  54:     """
  55:     evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
  56:     indicators = {"next_event": None, "hours_until": None}
  57: 
  58:     if evt is None:
  59:         return "HOLD", indicators
  60: 
  61:     indicators["next_event"] = f"{evt['type']} {evt['date'].isoformat()}"
  62:     indicators["hours_until"] = evt["hours_until"]
  63: 
  64:     if evt["hours_until"] <= 4:
  65:         return "SELL", indicators
  66:     if evt["hours_until"] <= 24 and evt["impact"] == "high":
  67:         return "SELL", indicators
  68:     return "HOLD", indicators
  69: 
  70: 
  71: def _event_type_info(ref_date) -> tuple[str, dict]:
  72:     """Classify upcoming event type (informational, doesn't vote strongly).
  73: 
  74:     FOMC and CPI = high uncertainty → lean SELL.
  75:     NFP = moderate → HOLD.
  76:     GDP and others = low → HOLD.
  77:     """
  78:     evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
  79:     indicators = {"event_type": None, "event_impact": None}
  80: 
  81:     if evt is None:
  82:         return "HOLD", indicators
  83: 
  84:     indicators["event_type"] = evt["type"]
  85:     indicators["event_impact"] = evt["impact"]
  86: 
  87:     # Only vote if event is within 48h
  88:     if evt["hours_until"] > 48:
  89:         return "HOLD", indicators
  90: 

>>> portfolio/signals/econ_calendar.py:170-260
 170: 
 171:     if overlap and evt["hours_until"] <= 24:
 172:         return "SELL", indicators
 173:     return "HOLD", indicators
 174: 
 175: 
 176: def compute_econ_calendar_signal(df: pd.DataFrame, context: dict = None) -> dict:
 177:     """Compute the composite economic calendar signal.
 178: 
 179:     Parameters
 180:     ----------
 181:     df : pd.DataFrame
 182:         OHLCV data (used to extract current timestamp).
 183:     context : dict, optional
 184:         Dict with keys: ticker, config, macro.
 185: 
 186:     Returns
 187:     -------
 188:     dict
 189:         action, confidence, sub_signals, indicators
 190:     """
 191:     result = {
 192:         "action": "HOLD",
 193:         "confidence": 0.0,
 194:         "sub_signals": {
 195:             "event_proximity": "HOLD",
 196:             "event_type": "HOLD",
 197:             "pre_event_risk": "HOLD",
 198:             "sector_exposure": "HOLD",
 199:             "post_event_relief": "HOLD",
 200:         },
 201:         "indicators": {},
 202:     }
 203: 
 204:     ticker = ""
 205:     if context:
 206:         ticker = context.get("ticker", "")
 207: 
 208:     ref_date = _get_current_date(df)
 209: 
 210:     # Compute each sub-signal
 211:     try:
 212:         prox_action, prox_ind = _event_proximity(ref_date)
 213:         # BUG-118: Warn when all economic dates have passed (data staleness)
 214:         if prox_ind.get("next_event") is None:
 215:             logger.warning("econ_calendar: no future events found — dates may need updating")
 216:     except Exception:
 217:         logger.exception("event_proximity sub-signal failed")
 218:         prox_action, prox_ind = "HOLD", {}
 219: 
 220:     try:
 221:         type_action, type_ind = _event_type_info(ref_date)
 222:     except Exception:
 223:         logger.exception("event_type sub-signal failed")
 224:         type_action, type_ind = "HOLD", {}
 225: 
 226:     try:
 227:         risk_action, risk_ind = _pre_event_risk(ref_date)
 228:     except Exception:
 229:         logger.exception("pre_event_risk sub-signal failed")
 230:         risk_action, risk_ind = "HOLD", {}
 231: 
 232:     try:
 233:         sec_action, sec_ind = _sector_exposure(ref_date, ticker)
 234:     except Exception:
 235:         logger.exception("sector_exposure sub-signal failed")
 236:         sec_action, sec_ind = "HOLD", {}
 237: 
 238:     try:
 239:         relief_action, relief_ind = _post_event_relief(ref_date)
 240:     except Exception:
 241:         logger.exception("post_event_relief sub-signal failed")
 242:         relief_action, relief_ind = "HOLD", {}
 243: 
 244:     # Populate result
 245:     result["sub_signals"]["event_proximity"] = prox_action
 246:     result["sub_signals"]["event_type"] = type_action
 247:     result["sub_signals"]["pre_event_risk"] = risk_action
 248:     result["sub_signals"]["sector_exposure"] = sec_action
 249:     result["sub_signals"]["post_event_relief"] = relief_action
 250: 
 251:     result["indicators"].update({f"proximity_{k}": v for k, v in prox_ind.items()})
 252:     result["indicators"].update({f"type_{k}": v for k, v in type_ind.items()})
 253:     result["indicators"].update({f"risk_{k}": v for k, v in risk_ind.items()})
 254:     result["indicators"].update({f"exposure_{k}": v for k, v in sec_ind.items()})
 255:     result["indicators"].update({f"relief_{k}": v for k, v in relief_ind.items()})
 256: 
 257:     # Majority vote (5 sub-signals: 3 SELL-capable + 1 BUY-capable + 1 either)
 258:     votes = [prox_action, type_action, risk_action, sec_action, relief_action]
 259:     result["action"], result["confidence"] = majority_vote(votes)
 260: 
MISSING: portfolio/news_keywords.py

 succeeded in 648ms:

>>> portfolio/signals/metals_cross_asset.py:1-120
   1: """Cross-asset signal for metals -- correlated market indicators.
   2: 
   3: Signal #32.  Combines 8 cross-asset sub-indicators via majority vote:
   4:     1. Copper Momentum: copper up -> industrial demand -> silver bullish
   5:     2. GVZ (Gold VIX): high implied vol signals breakout/reversal
   6:     3. Gold/Silver Ratio: mean-reversion signal (high = silver cheap)
   7:     4. G/S Ratio Velocity: rate of change — falling = silver outperforming
   8:     5. SPY Momentum: risk-on/risk-off gauge
   9:     6. Oil Momentum: inflation expectations proxy
  10:     7. EPU (Economic Policy Uncertainty): high uncertainty -> safe haven BUY
  11:     8. TIPS Real Yield direction: falling real yields -> BUY metals
  12: 
  13: Applicable to XAU-USD and XAG-USD only.
  14: Gold and silver interpret some signals differently (e.g. G/S ratio).
  15: 
  16: 2026-04-13: Horizon realignment after live measurement showed 29.1% on
  17: XAG 3h (178 BUY / 1 SELL bias over 179 samples). Root cause was using
  18: 5-day / 1-day lookbacks against 3-hour outcomes — lagged features with
  19: no intraday resolution. Fix: switch primary data to intraday (60m bars
  20: via `get_all_cross_asset_intraday`) and tighten thresholds proportionally.
  21: Daily data retained as fallback when intraday fetch fails (weekend,
  22: yfinance hiccup, etc.). GVZ stays daily — it's a daily-published index.
  23: 
  24: 2026-04-26: Added EPU + TIPS real yield from FRED API as sub-signals #7-8.
  25: EPU improves gold RMSE by ~18% (Baker/Bloom/Davis 2016). TIPS real yield
  26: direction captures opportunity-cost channel (metals pay no yield — when
  27: real yields fall, holding metals becomes relatively more attractive).
  28: Both are daily-cadence indicators like GVZ.
  29: """
  30: from __future__ import annotations
  31: 
  32: import logging
  33: import threading
  34: import time
  35: from typing import Any
  36: 
  37: from portfolio.signal_utils import majority_vote
  38: 
  39: logger = logging.getLogger("portfolio.signals.metals_cross_asset")
  40: 
  41: _METALS_TICKERS = {"XAU-USD", "XAG-USD"}
  42: 
  43: # 3h-calibrated thresholds. Rationale:
  44: #   Copper: daily 5d threshold was 1.5% (~0.3%/day). 3h typical range ~0.2%
  45: #     — threshold 0.4% captures a clearly directional 3h move.
  46: #   SPY: daily 1d threshold was 0.8%. Intraday 3h range typically 0.2-0.5%
  47: #     — threshold 0.25% catches risk-on/off shifts at 1-3h scale.
  48: #   Oil: daily 5d threshold was 2.0% (~0.4%/day). 3h typical range 0.3-0.8%
  49: #     — threshold 0.5% captures meaningful oil moves.
  50: #   G/S ratio velocity: daily 5d was 2.0%. 3h ratio moves typically 0.3-0.7%
  51: #     — threshold 0.5% captures silver-vs-gold divergence in the last 3h.
  52: _COPPER_MOVE_INTRADAY_PCT = 0.4
  53: _SPY_MOVE_INTRADAY_PCT = 0.25
  54: _OIL_MOVE_INTRADAY_PCT = 0.5
  55: _GS_VELOCITY_INTRADAY_PCT = 0.5
  56: 
  57: # Daily thresholds retained for the fallback path (intraday fetch failure).
  58: _COPPER_MOVE_DAILY_PCT = 1.5
  59: _SPY_MOVE_DAILY_PCT = 0.8
  60: _OIL_MOVE_DAILY_PCT = 2.0
  61: _GS_VELOCITY_DAILY_PCT = 2.0
  62: 
  63: _GVZ_ZSCORE_HIGH = 1.5
  64: _GVZ_ZSCORE_LOW = -1.0
  65: _GS_RATIO_ZSCORE = 1.5
  66: 
  67: # --- FRED-sourced macro indicators (daily, 4h cache) ---
  68: # EPU: Economic Policy Uncertainty (Baker/Bloom/Davis daily news index).
  69: # High uncertainty → flight to safety → BUY metals.
  70: _EPU_ZSCORE_HIGH = 1.5
  71: _EPU_ZSCORE_LOW = -1.0
  72: _EPU_FRED_SERIES = "USEPUINDXD"
  73: 
  74: # TIPS real yield (10Y TIPS, FRED series DFII10).
  75: # Direction matters: falling real yields → lower opportunity cost → BUY metals.
  76: # Threshold: 10bp change in 5d-vs-5d moving avg is meaningful.
  77: _TIPS_CHANGE_THRESHOLD = 0.10
  78: _TIPS_FRED_SERIES = "DFII10"
  79: 
  80: _FRED_TIMEOUT = 15
  81: _FRED_CACHE_TTL = 4 * 3600
  82: _FRED_HISTORY_LIMIT = 300
  83: 
  84: # Module-level FRED caches (separate dict per series)
  85: _epu_cache: dict = {}
  86: _tips_cache: dict = {}
  87: _fred_cache_lock = threading.Lock()
  88: 
  89: 
  90: def _get_fred_key(context: dict | None) -> str:
  91:     """Extract FRED API key from context → config."""
  92:     if not context:
  93:         return ""
  94:     cfg = context.get("config")
  95:     if not cfg:
  96:         return ""
  97:     if isinstance(cfg, dict):
  98:         return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
  99:     return getattr(cfg, "fred_api_key", "") or getattr(
 100:         getattr(cfg, "golddigger", None), "fred_api_key", ""
 101:     ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
 102: 
 103: 
 104: def _fetch_fred_values(
 105:     series_id: str, fred_api_key: str, cache: dict,
 106: ) -> list[float] | None:
 107:     """Fetch a FRED series.  Returns list of floats (newest first), cached 4h."""
 108:     now = time.time()
 109:     with _fred_cache_lock:
 110:         if (
 111:             cache.get("key") == fred_api_key
 112:             and cache.get("data")
 113:             and now - cache.get("time", 0) < _FRED_CACHE_TTL
 114:         ):
 115:             return cache["data"]
 116: 
 117:     if not fred_api_key:
 118:         logger.debug("No FRED API key — cannot fetch %s", series_id)
 119:         return cache.get("data")
 120: 

>>> portfolio/signals/credit_spread.py:1-140
   1: """Credit spread risk appetite signal — cross-asset macro indicator.
   2: 
   3: Signal #35.  Uses ICE BofA High Yield Option-Adjusted Spread (HY OAS)
   4: from FRED API as a measure of institutional credit risk appetite.
   5: Combines 4 sub-indicators via majority vote:
   6:     1. OAS Level Z-Score:    where current spread sits vs 252d history
   7:     2. OAS 5d Momentum:      rate of spread widening/tightening
   8:     3. OAS Momentum Accel:   acceleration of momentum (early warning)
   9:     4. Crisis Level Gate:    hard threshold at 500bp (every bear market)
  10: 
  11: Direction depends on asset class:
  12: - Gold/Silver (safe haven):  risk-off → BUY, risk-on → SELL
  13: - Crypto/Stocks (risk-on):   risk-off → SELL, risk-on → BUY
  14: 
  15: Data: FRED API series BAMLH0A0HYM2 (daily, free with API key).
  16: Cached for 4 hours since data updates daily.
  17: 
  18: Requires context dict with keys: ticker, config (for FRED API key).
  19: """
  20: from __future__ import annotations
  21: 
  22: import logging
  23: import time
  24: from typing import Any
  25: 
  26: from portfolio.file_utils import load_json
  27: from portfolio.signal_utils import majority_vote
  28: 
  29: logger = logging.getLogger("portfolio.signals.credit_spread")
  30: 
  31: # ---------------------------------------------------------------------------
  32: # Asset-class classification for directional interpretation
  33: # ---------------------------------------------------------------------------
  34: _SAFE_HAVEN = {"XAU-USD", "XAG-USD"}
  35: _RISK_ASSETS = {"BTC-USD", "ETH-USD", "MSTR"}
  36: 
  37: # ---------------------------------------------------------------------------
  38: # Thresholds
  39: # ---------------------------------------------------------------------------
  40: _ZSCORE_RISK_OFF = 1.5      # OAS z-score above this = risk-off
  41: _ZSCORE_RISK_ON = -1.0      # OAS z-score below this = risk-on (complacent)
  42: _MOM_5D_THRESHOLD = 0.20    # 20bp 5d change = meaningful momentum
  43: _ACCEL_THRESHOLD = 0.15     # 15bp acceleration = early warning
  44: _CRISIS_LEVEL = 5.0         # 500bp = crisis threshold
  45: _HISTORY_LIMIT = 300        # FRED observations to fetch (covers ~252 trading days)
  46: _FRED_SERIES = "BAMLH0A0HYM2"
  47: _FRED_TIMEOUT = 15
  48: _CACHE_TTL = 4 * 3600       # 4 hours
  49: 
  50: # ---------------------------------------------------------------------------
  51: # Module-level cache
  52: # ---------------------------------------------------------------------------
  53: _oas_cache: dict = {}
  54: 
  55: 
  56: def _fetch_hy_oas(fred_api_key: str) -> list[float] | None:
  57:     """Fetch HY OAS history from FRED.  Returns list of floats (newest first).
  58: 
  59:     Cached for 4 hours since data is daily.
  60:     """
  61:     now = time.time()
  62:     if (
  63:         _oas_cache.get("key") == fred_api_key
  64:         and _oas_cache.get("data")
  65:         and now - _oas_cache.get("time", 0) < _CACHE_TTL
  66:     ):
  67:         return _oas_cache["data"]
  68: 
  69:     if not fred_api_key:
  70:         logger.debug("No FRED API key — cannot fetch HY OAS")
  71:         return _oas_cache.get("data")
  72: 
  73:     try:
  74:         from portfolio.http_retry import fetch_with_retry
  75:     except ImportError:
  76:         import requests
  77: 
  78:         class _Shim:
  79:             @staticmethod
  80:             def __call__(url, **kwargs):
  81:                 return requests.get(url, **kwargs)
  82:         fetch_with_retry = _Shim()
  83: 
  84:     try:
  85:         resp = fetch_with_retry(
  86:             "https://api.stlouisfed.org/fred/series/observations",
  87:             params={
  88:                 "series_id": _FRED_SERIES,
  89:                 "api_key": fred_api_key,
  90:                 "file_type": "json",
  91:                 "sort_order": "desc",
  92:                 "limit": _HISTORY_LIMIT,
  93:             },
  94:             timeout=_FRED_TIMEOUT,
  95:         )
  96:         if hasattr(resp, "json"):
  97:             data = resp.json()
  98:         else:
  99:             import json
 100:             data = json.loads(resp)
 101: 
 102:         observations = data.get("observations", [])
 103:         values = []
 104:         for obs in observations:
 105:             val = obs.get("value", ".")
 106:             if val != ".":
 107:                 try:
 108:                     values.append(float(val))
 109:                 except (ValueError, TypeError):
 110:                     continue
 111: 
 112:         if values:
 113:             _oas_cache["key"] = fred_api_key
 114:             _oas_cache["data"] = values
 115:             _oas_cache["time"] = now
 116:             logger.debug("HY OAS fetched: %d observations, current=%.2f", len(values), values[0])
 117:             return values
 118: 
 119:     except Exception:
 120:         logger.warning("FRED HY OAS fetch failed", exc_info=True)
 121: 
 122:     return _oas_cache.get("data")
 123: 
 124: 
 125: def _get_fred_key(context: dict | None) -> str:
 126:     """Extract FRED API key from context -> config."""
 127:     if not context:
 128:         return ""
 129:     cfg = context.get("config")
 130:     if not cfg:
 131:         return ""
 132:     if isinstance(cfg, dict):
 133:         return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
 134:     return getattr(cfg, "fred_api_key", "") or getattr(
 135:         getattr(cfg, "golddigger", None), "fred_api_key", ""
 136:     ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
 137: 
 138: 
 139: def _is_safe_haven(ticker: str) -> bool:
 140:     """Return True if ticker is a safe-haven asset (gold/silver)."""

 succeeded in 701ms:

>>> portfolio/signals/forecast.py:120-260
 120: _CIRCUIT_BREAKER_TTL = 30  # 30 seconds before retry
 121: _kronos_tripped_until = 0.0  # monotonic timestamp when breaker resets
 122: _chronos_tripped_until = 0.0
 123: 
 124: # BUG-102: Lock protects circuit breaker state and dedup cache from ThreadPoolExecutor races.
 125: # The read-check-write pattern in _log_health() is not atomic without a lock.
 126: _forecast_lock = threading.Lock()
 127: 
 128: # Prediction dedup — track last logged timestamp per ticker to avoid
 129: # logging cached replays. Key: ticker, value: ISO-8601 timestamp.
 130: _PREDICTION_DEDUP_TTL = 60  # seconds — don't re-log within this window
 131: _PREDICTION_DEDUP_EVICT_AGE = 600  # BUG-106: evict entries older than 10 minutes
 132: _last_prediction_ts: dict[str, float] = {}  # ticker -> monotonic timestamp
 133: 
 134: 
 135: def _extract_json_from_stdout(stdout: str | None) -> dict | None:
 136:     """Extract JSON from potentially contaminated subprocess stdout.
 137: 
 138:     HuggingFace's from_pretrained() prints to stdout during model loading,
 139:     which contaminates the subprocess output before the JSON result.
 140:     This function handles that by finding the first '{' and parsing from there.
 141: 
 142:     Returns parsed dict on success, None on failure.
 143:     """
 144:     if not stdout:
 145:         return None
 146: 
 147:     text = stdout.strip()
 148:     if not text:
 149:         return None
 150: 
 151:     # Fast path: stdout starts with '{' — clean JSON
 152:     if text.startswith("{"):
 153:         try:
 154:             return json.loads(text)
 155:         except json.JSONDecodeError:
 156:             pass
 157: 
 158:     # Slow path: find first '{' and try parsing from there
 159:     brace_idx = text.find("{")
 160:     if brace_idx > 0:
 161:         try:
 162:             parsed = json.loads(text[brace_idx:])
 163:             logger.debug("JSON extracted via brace-offset fallback (offset=%d, len=%d)", brace_idx, len(text))
 164:             return parsed
 165:         except json.JSONDecodeError:
 166:             pass
 167: 
 168:     # Last resort: scan lines in reverse for a JSON line
 169:     for line in reversed(text.splitlines()):
 170:         line = line.strip()
 171:         if line.startswith("{"):
 172:             try:
 173:                 parsed = json.loads(line)
 174:                 logger.debug("JSON extracted via reverse-line-scan fallback (len=%d)", len(text))
 175:                 return parsed
 176:             except json.JSONDecodeError:
 177:                 continue
 178: 
 179:     logger.debug("JSON extraction failed — all 3 strategies exhausted (len=%d)", len(text) if text else 0)
 180:     return None
 181: 
 182: 
 183: def _kronos_circuit_open() -> bool:
 184:     with _forecast_lock:
 185:         return time.monotonic() < _kronos_tripped_until
 186: 
 187: 
 188: def _trip_kronos():
 189:     global _kronos_tripped_until
 190:     with _forecast_lock:
 191:         _kronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
 192:     logger.warning("Kronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)
 193: 
 194: 
 195: def _chronos_circuit_open() -> bool:
 196:     with _forecast_lock:
 197:         return time.monotonic() < _chronos_tripped_until
 198: 
 199: 
 200: def _trip_chronos():
 201:     global _chronos_tripped_until
 202:     with _forecast_lock:
 203:         _chronos_tripped_until = time.monotonic() + _CIRCUIT_BREAKER_TTL
 204:     logger.warning("Chronos circuit breaker TRIPPED — skipping for %ds", _CIRCUIT_BREAKER_TTL)
 205: 
 206: 
 207: def reset_circuit_breakers():
 208:     """Reset both circuit breakers (for testing or manual recovery)."""
 209:     global _kronos_tripped_until, _chronos_tripped_until
 210:     with _forecast_lock:
 211:         _kronos_tripped_until = 0.0
 212:         _chronos_tripped_until = 0.0
 213: 
 214: 
 215: def _log_health(model: str, ticker: str, success: bool, duration_ms: int, error: str = ""):
 216:     """Append a line to forecast_health.jsonl for persistent success/failure tracking.
 217: 
 218:     On success, auto-resets the relevant circuit breaker so recovered models
 219:     resume immediately instead of waiting for the full TTL (BUG-56 fix).
 220:     """
 221:     global _kronos_tripped_until, _chronos_tripped_until
 222:     try:
 223:         entry = {
 224:             "ts": datetime.now(UTC).isoformat(),
 225:             "model": model,
 226:             "ticker": ticker,
 227:             "ok": success,
 228:             "ms": duration_ms,
 229:         }
 230:         if error:
 231:             entry["error"] = error[:200]
 232:         atomic_append_jsonl(_HEALTH_FILE, entry)
 233:     except Exception as e:
 234:         logger.debug("Forecast health logging failed: %s", e)
 235: 
 236:     # Auto-reset circuit breaker on success — faster recovery from transient failures
 237:     # BUG-102: Use lock to make read-check-write atomic
 238:     if success:
 239:         with _forecast_lock:
 240:             if model == "kronos" and _kronos_tripped_until > 0:
 241:                 _kronos_tripped_until = 0.0
 242:                 logger.info("Kronos circuit breaker RESET on successful %s", ticker)
 243:             elif model == "chronos" and _chronos_tripped_until > 0:
 244:                 _chronos_tripped_until = 0.0
 245:                 logger.info("Chronos circuit breaker RESET on successful %s", ticker)
 246: 
 247: 
 248: def _load_candles_ohlcv(ticker: str, periods: int = 168,
 249:                         interval: str = "1h") -> list[dict] | None:
 250:     """Load recent OHLCV candles as list of dicts.
 251: 
 252:     Args:
 253:         ticker: Instrument ticker (e.g., "BTC-USD")
 254:         periods: Number of candles to fetch
 255:         interval: Candle interval ("1h", "5m", "15m", etc.)
 256:     """
 257:     from portfolio.tickers import SYMBOLS
 258: 
 259:     source_info = SYMBOLS.get(ticker, {})
 260: 

>>> portfolio/signals/forecast.py:430-740
 430:         else:
 431:             _log_health("chronos", _ticker, True, ms)
 432:         return result
 433:     except Exception as e:
 434:         ms = round((time.time() - t0) * 1000)
 435:         logger.warning("Chronos failed: %s", e)
 436:         _log_health("chronos", _ticker, False, ms, str(e)[:200])
 437:         _trip_chronos()
 438:         return None
 439: 
 440: 
 441: def _health_weighted_vote(sub_signals, kronos_ok, chronos_ok):
 442:     """Vote only using sub-signals from healthy (working) models.
 443: 
 444:     When Kronos is dead (high failure rate), its 2 permanent HOLD votes
 445:     dilute the 4-vote majority and make the signal always return HOLD.
 446:     This function excludes dead models from the vote.
 447: 
 448:     **Shadow exclusion (2026-04-21)**: when `_KRONOS_SHADOW` is True, Kronos's
 449:     sub-signal is EXCLUDED from the vote pool regardless of `kronos_ok`. The
 450:     shadow mode is for logging + accuracy backfill only — adding HOLD weights
 451:     to the composite would tie against Chronos's verdicts (observed 3668
 452:     times in the previous shadow-mode run, all forced to HOLD). The raw
 453:     Kronos prediction is still captured in `raw_sub_signals["kronos_*"]` for
 454:     downstream accuracy tracking.
 455: 
 456:     1h horizon gets 2x weight (counted twice) because short-term predictions
 457:     are more actionable and Chronos 24h predictions are less reliable.
 458:     """
 459:     alive_votes = []
 460:     # Shadow mode: Kronos runs, logs its raw prediction, but does NOT
 461:     # contribute to the composite vote. Only vote when the model is both
 462:     # healthy AND live (not shadow).
 463:     if kronos_ok and not _KRONOS_SHADOW:
 464:         # 1h gets double weight
 465:         alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
 466:         alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
 467:         alive_votes.append(sub_signals.get("kronos_24h", "HOLD"))
 468:     if chronos_ok:
 469:         # 1h gets double weight
 470:         alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
 471:         alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
 472:         alive_votes.append(sub_signals.get("chronos_24h", "HOLD"))
 473: 
 474:     if not alive_votes:
 475:         return "HOLD", 0.0
 476: 
 477:     return majority_vote(alive_votes)
 478: 
 479: 
 480: # Per-ticker accuracy cache TTL
 481: _ACCURACY_CACHE_TTL = 1800  # 30 minutes
 482: 
 483: # Default thresholds for accuracy gating
 484: _HOLD_THRESHOLD = 0.55        # Below this: force HOLD (signal can't predict)
 485: _MIN_SAMPLES = 10             # Below this: use raw vote (insufficient data)
 486: 
 487: # Volatility gate — force HOLD when ATR% exceeds threshold
 488: # Chronos predicts negligible moves (~0.1% avg), so high-volatility environments
 489: # where actual moves are 3-5% make the signal useless.
 490: _VOL_GATE_CRYPTO = 0.03       # 3% ATR for crypto
 491: _VOL_GATE_DEFAULT = 0.02      # 2% ATR for metals/stocks
 492: 
 493: # Regime-aware confidence discount — Chronos has a mean-reversion bias
 494: # (predicts small moves back to mean). In trending markets this is wrong.
 495: _REGIME_DISCOUNT_TRENDING = 0.5   # Halve confidence in trending regimes
 496: _REGIME_DISCOUNT_HIGH_VOL = 0.6   # Reduce confidence in high-vol regimes
 497: _REGIME_NEUTRAL = 1.0             # No discount in ranging/neutral regimes
 498: 
 499: 
 500: def _compute_atr_pct(close_prices: list[float], period: int = 14) -> float | None:
 501:     """Compute ATR% from close prices (approximation using close-to-close).
 502: 
 503:     Returns ATR as fraction of current price (e.g. 0.03 = 3%), or None
 504:     if insufficient data.
 505:     """
 506:     if not close_prices or len(close_prices) < period + 1:
 507:         return None
 508:     # Approximate true range from close-to-close changes
 509:     trs = [abs(close_prices[i] - close_prices[i - 1]) for i in range(1, len(close_prices))]
 510:     if len(trs) < period:
 511:         return None
 512:     # EMA-smoothed ATR over last `period` values
 513:     recent_trs = trs[-period * 2:]  # use more data for EMA warmup
 514:     atr = recent_trs[0]
 515:     alpha = 2.0 / (period + 1)
 516:     for tr in recent_trs[1:]:
 517:         atr = alpha * tr + (1 - alpha) * atr
 518:     current = close_prices[-1]
 519:     if current <= 0:
 520:         return None
 521:     return atr / current
 522: 
 523: 
 524: def _is_crypto_ticker(ticker: str) -> bool:
 525:     """Check if ticker is crypto (BTC-USD, ETH-USD)."""
 526:     try:
 527:         from portfolio.tickers import CRYPTO_SYMBOLS
 528:         return ticker in CRYPTO_SYMBOLS
 529:     except ImportError:
 530:         return ticker in {"BTC-USD", "ETH-USD"}
 531: 
 532: 
 533: def _load_forecast_accuracy(cache_ttl=None):
 534:     """Load per-ticker forecast accuracy, cached via _cached().
 535: 
 536:     Returns dict: {ticker: {accuracy, samples}} or empty dict on error.
 537:     """
 538:     ttl = cache_ttl or _ACCURACY_CACHE_TTL
 539: 
 540:     def _fetch():
 541:         try:
 542:             from portfolio.forecast_accuracy import get_all_ticker_accuracies
 543:             return get_all_ticker_accuracies(horizon="24h", days=14)
 544:         except Exception as e:
 545:             logger.debug("Failed to load forecast accuracy: %s", e)
 546:             return {}
 547: 
 548:     return _cached("forecast_ticker_accuracy", ttl, _fetch)
 549: 
 550: 
 551: def _load_forecast_subsignal_accuracy(cache_ttl=None, days=30):
 552:     """Load raw sub-signal accuracy for 1h and 24h forecast votes."""
 553:     ttl = cache_ttl or _ACCURACY_CACHE_TTL
 554:     cache_key = f"forecast_subsignal_accuracy_{days}"
 555: 
 556:     def _fetch():
 557:         try:
 558:             from portfolio.forecast_accuracy import compute_forecast_accuracy
 559: 
 560:             return {
 561:                 "1h": compute_forecast_accuracy(
 562:                     horizon="1h", days=days, use_raw_sub_signals=True
 563:                 ),
 564:                 "24h": compute_forecast_accuracy(
 565:                     horizon="24h", days=days, use_raw_sub_signals=True
 566:                 ),
 567:             }
 568:         except Exception as e:
 569:             logger.debug("Failed to load forecast sub-signal accuracy: %s", e)
 570:             return {}
 571: 
 572:     return _cached(cache_key, ttl, _fetch)
 573: 
 574: 
 575: def _gate_subsignal_votes_by_accuracy(sub_signals, ticker, config_forecast=None):
 576:     """Gate individual forecast sub-signals using raw historical accuracy."""
 577:     cfg = config_forecast or {}
 578:     hold_threshold = cfg.get("subsignal_hold_threshold", cfg.get("hold_threshold", _HOLD_THRESHOLD))
 579:     min_samples = cfg.get("subsignal_min_samples", cfg.get("min_samples", _MIN_SAMPLES))
 580:     lookback_days = cfg.get("subsignal_accuracy_days", 30)
 581:     cache_ttl = cfg.get("subsignal_accuracy_cache_ttl", _ACCURACY_CACHE_TTL)
 582: 
 583:     gated = dict(sub_signals)
 584:     info = {}
 585:     if not ticker:
 586:         return gated, info
 587: 
 588:     accuracy_matrix = _load_forecast_subsignal_accuracy(cache_ttl=cache_ttl, days=lookback_days)
 589:     for sub_name, vote in sub_signals.items():
 590:         if vote == "HOLD":
 591:             continue
 592: 
 593:         horizon = "1h" if sub_name.endswith("_1h") else "24h"
 594:         horizon_stats = ((accuracy_matrix or {}).get(horizon) or {}).get(sub_name) or {}
 595:         ticker_stats = (horizon_stats.get("by_ticker") or {}).get(ticker)
 596: 
 597:         accuracy = None
 598:         samples = 0
 599:         source = None
 600:         if ticker_stats and ticker_stats.get("total", 0) >= min_samples:
 601:             accuracy = float(ticker_stats["accuracy"])
 602:             samples = int(ticker_stats["total"])
 603:             source = "ticker"
 604:         elif horizon_stats.get("total", 0) >= min_samples:
 605:             accuracy = float(horizon_stats["accuracy"])
 606:             samples = int(horizon_stats["total"])
 607:             source = "global"
 608: 
 609:         gating = "insufficient_data"
 610:         if accuracy is not None:
 611:             gating = "held" if accuracy < hold_threshold else "raw"
 612:             if gating == "held":
 613:                 gated[sub_name] = "HOLD"
 614: 
 615:         info[sub_name] = {
 616:             "gating": gating,
 617:             "accuracy": round(accuracy, 3) if accuracy is not None else None,
 618:             "samples": samples,
 619:             "source": source,
 620:         }
 621: 
 622:     return gated, info
 623: 
 624: 
 625: def _regime_discount(regime: str, config_forecast: dict | None = None) -> float:
 626:     """Return confidence multiplier based on market regime.
 627: 
 628:     Chronos has a mean-reversion bias — it predicts small moves back to mean.
 629:     In trending markets, this is wrong, so we discount confidence.
 630:     """
 631:     cfg = config_forecast or {}
 632:     if not regime:
 633:         return _REGIME_NEUTRAL
 634:     r = regime.lower()
 635:     if r in ("trending-up", "trending-down", "breakout"):
 636:         return cfg.get("regime_discount_trending", _REGIME_DISCOUNT_TRENDING)
 637:     elif r in ("high-vol", "capitulation"):
 638:         return cfg.get("regime_discount_high_vol", _REGIME_DISCOUNT_HIGH_VOL)
 639:     else:
 640:         # range-bound, neutral — mean-reversion is appropriate
 641:         return _REGIME_NEUTRAL
 642: 
 643: 
 644: def _accuracy_weighted_vote(sub_signals, kronos_ok, chronos_ok, ticker="",
 645:                             config_forecast=None, atr_pct=None, regime=None):
 646:     """Vote with per-ticker accuracy gating, volatility gate, and regime discount.
 647: 
 648:     Extends _health_weighted_vote with:
 649:     - Volatility gate: high ATR% → force HOLD (Chronos can't predict big moves)
 650:     - Regime discount: trending markets → reduce confidence (mean-reversion bias)
 651:     - Accuracy gate: accuracy < hold_threshold → force HOLD
 652:     - Good accuracy: use raw vote, scale confidence by accuracy
 653:     - Insufficient samples: use raw vote (not enough data to judge)
 654: 
 655:     Bad tickers abstain (HOLD) rather than invert — inversion games the
 656:     accuracy metric without fixing the underlying prediction quality.
 657: 
 658:     Returns (action, confidence, gating_info) where gating_info is a dict with
 659:     accuracy metadata for logging.
 660:     """
 661:     cfg = config_forecast or {}
 662:     hold_thresh = cfg.get("hold_threshold", _HOLD_THRESHOLD)
 663:     min_samples = cfg.get("min_samples", _MIN_SAMPLES)
 664: 
 665:     # Start with health-weighted vote as baseline
 666:     base_action, base_conf = _health_weighted_vote(
 667:         sub_signals, kronos_ok, chronos_ok
 668:     )
 669: 
 670:     gating_info = {
 671:         "forecast_accuracy": None,
 672:         "forecast_samples": 0,
 673:         "forecast_gating": "raw",
 674:         "forecast_inverted": False,
 675:         "base_action": base_action,
 676:         "base_confidence": base_conf,
 677:         "atr_pct": atr_pct,
 678:     }
 679: 
 680:     if not ticker:
 681:         return base_action, base_conf, gating_info
 682: 
 683:     # Volatility gate — Chronos predicts negligible moves (~0.1% avg),
 684:     # so high-vol environments make the signal useless
 685:     if atr_pct is not None:
 686:         vol_thresh = cfg.get("vol_gate_crypto", _VOL_GATE_CRYPTO) \
 687:             if _is_crypto_ticker(ticker) \
 688:             else cfg.get("vol_gate_default", _VOL_GATE_DEFAULT)
 689:         if atr_pct > vol_thresh:
 690:             gating_info["forecast_gating"] = "vol_gated"
 691:             return "HOLD", 0.0, gating_info
 692: 
 693:     # Load per-ticker accuracy
 694:     all_acc = _load_forecast_accuracy(cfg.get("accuracy_cache_ttl"))
 695:     ticker_acc = all_acc.get(ticker) if all_acc else None
 696: 
 697:     if ticker_acc is None or ticker_acc.get("samples", 0) < min_samples:
 698:         gating_info["forecast_gating"] = "insufficient_data"
 699:         if ticker_acc:
 700:             gating_info["forecast_accuracy"] = ticker_acc["accuracy"]
 701:             gating_info["forecast_samples"] = ticker_acc["samples"]
 702:         return base_action, base_conf, gating_info
 703: 
 704:     acc = ticker_acc["accuracy"]
 705:     samples = ticker_acc["samples"]
 706:     gating_info["forecast_accuracy"] = acc
 707:     gating_info["forecast_samples"] = samples
 708: 
 709:     if acc < hold_thresh:
 710:         # Below threshold — signal can't predict this ticker, abstain
 711:         gating_info["forecast_gating"] = "held"
 712:         return "HOLD", 0.0, gating_info
 713: 
 714:     else:
 715:         # Good accuracy — use raw vote, scale confidence by accuracy
 716:         gating_info["forecast_gating"] = "raw"
 717:         # Apply regime discount (trending → lower confidence for mean-reversion bias)
 718:         r_discount = _regime_discount(regime, cfg)
 719:         gating_info["regime_discount"] = r_discount
 720:         scaled_conf = base_conf * acc * r_discount
 721:         return base_action, min(scaled_conf, _MAX_CONFIDENCE), gating_info
 722: 
 723: 
 724: def _direction_to_action(direction: str) -> str:
 725:     """Convert direction string to action."""
 726:     if direction in ("up", "BUY"):
 727:         return "BUY"
 728:     if direction in ("down", "SELL"):
 729:         return "SELL"
 730:     return "HOLD"
 731: 
 732: 
 733: def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
 734:     """Compute the composite forecast signal.
 735: 
 736:     Parameters
 737:     ----------
 738:     df : pd.DataFrame
 739:         OHLCV data (used as fallback if candle fetch fails).
 740:     context : dict, optional

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/econ_dates.py'; Start=1; End=320},
  @{Path='portfolio/shared_state.py'; Start=1; End=220}
)
foreach ("'$f in $files) {
  if (Test-Path $f.Path) {
    Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
    $lines = Get-Content $f.Path
    for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
      Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
    }
  } else {
    Write-Output ("MISSING: {0}" -f $f.Path)
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$files = @(
  @{Path='"'portfolio/signals/gold_real_yield_paradox.py'; Start=50; End=320},
  @{Path='portfolio/fomc_dates.py'; Start=1; End=220}
)
foreach ("'$f in $files) {
  if (Test-Path $f.Path) {
    Write-Output ("`n>>> {0}:{1}-{2}" -f $f.Path, $f.Start, $f.End)
    $lines = Get-Content $f.Path
    for ($i = $f.Start; $i -le $f.End -and $i -le $lines.Length; $i++) {
      Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1]))
    }
  } else {
    Write-Output ("MISSING: {0}" -f $f.Path)
  }
}' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 538ms:
MISSING: portfolio/econ_dates.py
MISSING: portfolio/shared_state.py

 succeeded in 598ms:

>>> portfolio/signals/gold_real_yield_paradox.py:50-320
  50:         return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
  51:     return getattr(cfg, "fred_api_key", "") or getattr(
  52:         getattr(cfg, "golddigger", None), "fred_api_key", ""
  53:     ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
  54: 
  55: 
  56: def _fetch_real_yield(fred_api_key: str) -> list[float] | None:
  57:     """Fetch 10Y TIPS real yield from FRED. Returns list newest-first."""
  58:     now = time.time()
  59:     with _yield_cache_lock:
  60:         if (
  61:             _yield_cache.get("key") == fred_api_key
  62:             and _yield_cache.get("data")
  63:             and now - _yield_cache.get("time", 0) < _CACHE_TTL
  64:         ):
  65:             return _yield_cache["data"]
  66: 
  67:     if not fred_api_key:
  68:         logger.debug("No FRED API key for real yield fetch")
  69:         return _yield_cache.get("data")
  70: 
  71:     try:
  72:         from portfolio.http_retry import fetch_with_retry
  73:     except ImportError:
  74:         logger.warning("http_retry not available")
  75:         return _yield_cache.get("data")
  76: 
  77:     try:
  78:         resp = fetch_with_retry(
  79:             "https://api.stlouisfed.org/fred/series/observations",
  80:             params={
  81:                 "series_id": _FRED_SERIES,
  82:                 "api_key": fred_api_key,
  83:                 "file_type": "json",
  84:                 "sort_order": "desc",
  85:                 "limit": _HISTORY_LIMIT,
  86:             },
  87:             timeout=_FRED_TIMEOUT,
  88:         )
  89:         if hasattr(resp, "json"):
  90:             data = resp.json()
  91:         else:
  92:             import json
  93:             data = json.loads(resp.text if hasattr(resp, "text") else resp)
  94: 
  95:         observations = data.get("observations", [])
  96:         values = []
  97:         for obs in observations:
  98:             val = obs.get("value", ".")
  99:             if val == ".":
 100:                 continue
 101:             try:
 102:                 values.append(float(val))
 103:             except (ValueError, TypeError):
 104:                 continue
 105: 
 106:         if values:
 107:             with _yield_cache_lock:
 108:                 _yield_cache["key"] = fred_api_key
 109:                 _yield_cache["data"] = values
 110:                 _yield_cache["time"] = now
 111:             logger.debug("Real yield fetched: %d obs, current=%.3f", len(values), values[0])
 112:             return values
 113: 
 114:     except Exception:
 115:         logger.warning("FRED real yield fetch failed", exc_info=True)
 116: 
 117:     return _yield_cache.get("data")
 118: 
 119: 
 120: def _paradox_spread(gold_returns_30d: float, yield_change_30d: float) -> tuple[str, dict]:
 121:     """Sub-indicator 1: gold and yield both rising (paradox regime = BUY)."""
 122:     both_positive = gold_returns_30d > 0 and yield_change_30d > 0
 123: 
 124:     if both_positive:
 125:         magnitude = (
 126:             min(abs(gold_returns_30d) / 0.10, 1.0) * 50
 127:             + min(abs(yield_change_30d) / 0.75, 1.0) * 50
 128:         )
 129:         action = "BUY"
 130:     elif gold_returns_30d < 0 and yield_change_30d < 0:
 131:         magnitude = (
 132:             min(abs(gold_returns_30d) / 0.10, 1.0) * 50
 133:             + min(abs(yield_change_30d) / 0.75, 1.0) * 50
 134:         ) * 0.5
 135:         action = "HOLD"
 136:     else:
 137:         magnitude = 0.0
 138:         action = "HOLD"
 139: 
 140:     return action, {
 141:         "paradox_score": round(magnitude, 1),
 142:         "gold_30d_ret": round(gold_returns_30d, 4),
 143:         "yield_30d_chg": round(yield_change_30d, 4),
 144:     }
 145: 
 146: 
 147: def _correlation_break(
 148:     gold_daily_returns: np.ndarray, yield_daily_changes: np.ndarray,
 149:     baseline_corr: float,
 150: ) -> tuple[str, dict]:
 151:     """Sub-indicator 2: 30d correlation deviates from 3yr baseline."""
 152:     if len(gold_daily_returns) < 30 or len(yield_daily_changes) < 30:
 153:         return "HOLD", {"corr_break_score": 0.0}
 154: 
 155:     recent_gold = gold_daily_returns[-30:]
 156:     recent_yield = yield_daily_changes[-30:]
 157: 
 158:     if np.std(recent_gold) < 1e-10 or np.std(recent_yield) < 1e-10:
 159:         return "HOLD", {"corr_break_score": 0.0}
 160: 
 161:     corr_30d = float(np.corrcoef(recent_gold, recent_yield)[0, 1])
 162:     if np.isnan(corr_30d):
 163:         return "HOLD", {"corr_break_score": 0.0}
 164: 
 165:     deviation = abs(corr_30d - baseline_corr)
 166:     break_score = min(deviation / 0.5 * 100, 100)
 167: 
 168:     if corr_30d > baseline_corr + 0.3:
 169:         action = "BUY"
 170:     elif corr_30d < baseline_corr - 0.3:
 171:         action = "SELL"
 172:     else:
 173:         action = "HOLD"
 174: 
 175:     return action, {
 176:         "corr_break_score": round(break_score, 1),
 177:         "corr_30d": round(corr_30d, 3),
 178:         "baseline_corr": round(baseline_corr, 3),
 179:     }
 180: 
 181: 
 182: def _momentum_split(
 183:     gold_close: pd.Series, yield_values: list[float],
 184: ) -> tuple[str, dict]:
 185:     """Sub-indicator 3: gold trend vs yield trend divergence."""
 186:     if len(gold_close) < 200 or len(yield_values) < 50:
 187:         return "HOLD", {"momentum_split_score": 0.0}
 188: 
 189:     gold_sma50 = float(gold_close.iloc[-50:].mean())
 190:     gold_sma200 = float(gold_close.iloc[-200:].mean())
 191:     gold_mom = gold_sma50 / gold_sma200 - 1 if gold_sma200 > 0 else 0.0
 192: 
 193:     yield_current = yield_values[0]
 194:     prior_yields = yield_values[1:51]
 195:     yield_sma50 = sum(prior_yields) / len(prior_yields) if prior_yields else yield_current
 196:     yield_mom = yield_current - yield_sma50
 197: 
 198:     gold_up = gold_mom > 0.005
 199:     yield_up = yield_mom > 0.05
 200: 
 201:     if gold_up and yield_up:
 202:         score = 75.0
 203:         action = "BUY"
 204:     elif not gold_up and not yield_up:
 205:         score = 25.0
 206:         action = "SELL"
 207:     else:
 208:         score = 50.0
 209:         action = "HOLD"
 210: 
 211:     return action, {
 212:         "momentum_split_score": round(score, 1),
 213:         "gold_mom": round(gold_mom, 4),
 214:         "yield_mom": round(yield_mom, 4),
 215:     }
 216: 
 217: 
 218: def _compute_baseline_correlation(
 219:     gold_daily_returns: np.ndarray, yield_daily_changes: np.ndarray,
 220:     window: int = 756,
 221: ) -> float:
 222:     """3-year rolling baseline of 90d gold-yield correlation."""
 223:     n = min(len(gold_daily_returns), len(yield_daily_changes), window)
 224:     if n < 90:
 225:         return -0.45
 226: 
 227:     correlations = []
 228:     for start in range(0, n - 90, 30):
 229:         g = gold_daily_returns[start : start + 90]
 230:         y = yield_daily_changes[start : start + 90]
 231:         if np.std(g) < 1e-10 or np.std(y) < 1e-10:
 232:             continue
 233:         c = np.corrcoef(g, y)[0, 1]
 234:         if not np.isnan(c):
 235:             correlations.append(c)
 236: 
 237:     return float(np.mean(correlations)) if correlations else -0.45
 238: 
 239: 
 240: def compute_gold_real_yield_paradox_signal(
 241:     df: pd.DataFrame, context: dict | None = None, **kwargs: Any,
 242: ) -> dict:
 243:     """Compute GYDI signal for gold (and silver via gold proxy).
 244: 
 245:     Args:
 246:         df: OHLCV DataFrame with at least MIN_ROWS rows.
 247:         context: dict with keys {ticker, config, asset_class}.
 248: 
 249:     Returns:
 250:         dict with keys: action, confidence, sub_signals, indicators.
 251:     """
 252:     empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
 253: 
 254:     if df is None or len(df) < MIN_ROWS:
 255:         return empty
 256: 
 257:     ticker = (context or {}).get("ticker", "")
 258:     if ticker and ticker not in _APPLICABLE_TICKERS:
 259:         return empty
 260: 
 261:     fred_key = _get_fred_key(context)
 262:     if not fred_key:
 263:         try:
 264:             from portfolio.file_utils import load_json
 265:             cfg = load_json("config.json")
 266:             if cfg:
 267:                 fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
 268:         except Exception:
 269:             logger.debug("config.json fallback read failed", exc_info=True)
 270: 
 271:     if not fred_key:
 272:         logger.debug("No FRED API key — cannot compute GYDI")
 273:         return empty
 274: 
 275:     yield_values = _fetch_real_yield(fred_key)
 276:     if not yield_values or len(yield_values) < 30:
 277:         logger.debug("Insufficient real yield data (%d obs)", len(yield_values) if yield_values else 0)
 278:         return empty
 279: 
 280:     close = df["close"].astype(float)
 281: 
 282:     gold_30d_return = float(close.iloc[-1] / close.iloc[-min(30, len(close))] - 1)
 283:     yield_30d_change = yield_values[0] - yield_values[min(29, len(yield_values) - 1)]
 284: 
 285:     gold_daily_returns = close.pct_change(fill_method=None).dropna().values
 286:     min_len = min(len(gold_daily_returns), len(yield_values) - 1)
 287:     yield_daily_changes = np.array([
 288:         yield_values[i] - yield_values[i + 1]
 289:         for i in range(min_len)
 290:     ])[::-1]  # reverse: FRED is newest-first, gold is oldest-first
 291:     gold_daily_returns = gold_daily_returns[-min_len:]
 292: 
 293:     baseline_corr = _compute_baseline_correlation(gold_daily_returns, yield_daily_changes)
 294: 
 295:     act1, ind1 = _paradox_spread(gold_30d_return, yield_30d_change)
 296:     act2, ind2 = _correlation_break(gold_daily_returns, yield_daily_changes, baseline_corr)
 297:     act3, ind3 = _momentum_split(close, yield_values)
 298: 
 299:     votes = [act1, act2, act3]
 300:     action, confidence = majority_vote(votes, count_hold=False)
 301: 
 302:     gydi = (
 303:         ind1.get("paradox_score", 0) * 0.40
 304:         + ind2.get("corr_break_score", 0) * 0.35
 305:         + ind3.get("momentum_split_score", 0) * 0.25
 306:     )
 307: 
 308:     regime = "LOW"
 309:     if gydi >= 75:
 310:         regime = "CRITICAL"
 311:     elif gydi >= 50:
 312:         regime = "HIGH"
 313:     elif gydi >= 30:
 314:         regime = "ELEVATED"
 315: 
 316:     indicators = {**ind1, **ind2, **ind3, "gydi": round(gydi, 1), "gydi_regime": regime}
 317: 
 318:     return {
 319:         "action": action,
 320:         "confidence": min(confidence, 0.7),
MISSING: portfolio/fomc_dates.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/forecast.py'; for ("'$i = 248; $i -le 430; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 576ms:
 248: def _load_candles_ohlcv(ticker: str, periods: int = 168,
 249:                         interval: str = "1h") -> list[dict] | None:
 250:     """Load recent OHLCV candles as list of dicts.
 251: 
 252:     Args:
 253:         ticker: Instrument ticker (e.g., "BTC-USD")
 254:         periods: Number of candles to fetch
 255:         interval: Candle interval ("1h", "5m", "15m", etc.)
 256:     """
 257:     from portfolio.tickers import SYMBOLS
 258: 
 259:     source_info = SYMBOLS.get(ticker, {})
 260: 
 261:     # Determine the data source — needed to apply source-specific interval constraints
 262:     if "binance" in source_info:
 263:         source = "binance"
 264:     elif "binance_fapi" in source_info:
 265:         source = "binance_fapi"
 266:     elif "alpaca" in source_info:
 267:         source = "alpaca"
 268:     else:
 269:         source = None
 270: 
 271:     # Alpaca minimum supported interval is 15m — fall back if configured interval is smaller.
 272:     # alpaca_klines() does its own mapping; pass the raw internal interval directly.
 273:     if source == "alpaca" and interval in ("1m", "3m", "5m"):
 274:         logger.debug(
 275:             "Alpaca does not support %s interval for %s — falling back to 15m", interval, ticker
 276:         )
 277:         interval = "15m"
 278: 
 279:     try:
 280:         if source == "binance":
 281:             from portfolio.data_collector import binance_klines
 282:             symbol = source_info["binance"]
 283:             df = binance_klines(symbol, interval=interval, limit=periods)
 284:         elif source == "binance_fapi":
 285:             from portfolio.data_collector import binance_fapi_klines
 286:             symbol = source_info["binance_fapi"]
 287:             df = binance_fapi_klines(symbol, interval=interval, limit=periods)
 288:         elif source == "alpaca":
 289:             # Pass the raw internal interval — alpaca_klines() handles the mapping itself.
 290:             from portfolio.data_collector import alpaca_klines
 291:             symbol = source_info["alpaca"]
 292:             df = alpaca_klines(symbol, interval=interval, limit=periods)
 293:         else:
 294:             return None
 295: 
 296:         if df is not None and len(df) > 30:
 297:             candles = []
 298:             for _, row in df.iterrows():
 299:                 candles.append({
 300:                     "open": float(row.get("open", row.get("close", 0))),
 301:                     "high": float(row.get("high", row.get("close", 0))),
 302:                     "low": float(row.get("low", row.get("close", 0))),
 303:                     "close": float(row["close"]),
 304:                     "volume": float(row.get("volume", 0)),
 305:                 })
 306:             return candles
 307:     except Exception as e:
 308:         logger.debug("OHLCV fetch failed for %s (interval=%s): %s", ticker, interval, e)
 309: 
 310:     return None
 311: 
 312: 
 313: def _run_kronos(candles: list[dict], horizons: tuple = (1, 24), _ticker: str = "") -> dict | None:
 314:     """Run Kronos inference via subprocess with GPU gating."""
 315:     if not _KRONOS_ENABLED:
 316:         return None
 317:     if _kronos_circuit_open():
 318:         return None
 319:     t0 = time.time()
 320:     try:
 321:         with gpu_gate("kronos", timeout=90) as acquired:
 322:             if not acquired:
 323:                 logger.warning("GPU gate timeout for Kronos %s", _ticker)
 324:                 return None
 325:             return _run_kronos_inner(candles, horizons, _ticker, t0)
 326:     except Exception as e:
 327:         ms = round((time.time() - t0) * 1000)
 328:         logger.warning("Kronos GPU gate error: %s", e)
 329:         _log_health("kronos", _ticker, False, ms, str(e)[:200])
 330:         _trip_kronos()
 331:         return None
 332: 
 333: 
 334: def _run_kronos_inner(candles, horizons, _ticker, t0):
 335:     """Kronos inference (called inside GPU gate)."""
 336:     try:
 337:         # Read tunable params from config
 338:         try:
 339:             from portfolio.file_utils import load_json
 340:             cfg = load_json(str(Path(__file__).resolve().parent.parent.parent / "config.json"), {})
 341:             fc = cfg.get("forecast", {})
 342:         except Exception:
 343:             logger.debug("Forecast config load failed, using defaults", exc_info=True)
 344:             fc = {}
 345: 
 346:         input_data = json.dumps({
 347:             "candles": candles,
 348:             "prices_close": [c["close"] for c in candles],
 349:             "temperature": fc.get("kronos_temperature", 1.0),
 350:             "top_p": fc.get("kronos_top_p", 0.9),
 351:             "sample_count": fc.get("kronos_samples", 3),
 352:         })
 353:         proc = subprocess.run(
 354:             [_KRONOS_PYTHON, _KRONOS_SCRIPT,
 355:              "--horizons", ",".join(str(h) for h in horizons)],
 356:             input=input_data,
 357:             capture_output=True,
 358:             text=True,
 359:             timeout=_KRONOS_TIMEOUT,
 360:         )
 361:         ms = round((time.time() - t0) * 1000)
 362:         if proc.returncode != 0:
 363:             err = proc.stderr[:200]
 364:             logger.warning("Kronos subprocess failed: %s", err)
 365:             _log_health("kronos", _ticker, False, ms, err)
 366:             _trip_kronos()
 367:             return None
 368:         if not proc.stdout or not proc.stdout.strip():
 369:             _log_health("kronos", _ticker, False, ms, "empty_stdout")
 370:             _trip_kronos()
 371:             return None
 372:         result = _extract_json_from_stdout(proc.stdout)
 373:         if result is None:
 374:             # JSON extraction failed — log actual stdout for diagnostics
 375:             preview = repr(proc.stdout[:200])
 376:             logger.warning("Kronos stdout not valid JSON for %s: %s", _ticker, preview)
 377:             _log_health("kronos", _ticker, False, ms, f"json_extract_failed: {preview[:150]}")
 378:             _trip_kronos()
 379:             return None
 380:         if not result or not result.get("results"):
 381:             _log_health("kronos", _ticker, False, ms, "empty_results")
 382:             _trip_kronos()
 383:             return None
 384:         _log_health("kronos", _ticker, True, ms)
 385:         return result
 386:     except Exception as e:
 387:         ms = round((time.time() - t0) * 1000)
 388:         logger.warning("Kronos subprocess error (v2): %s", e)
 389:         _log_health("kronos", _ticker, False, ms, str(e)[:200])
 390:         _trip_kronos()
 391:         return None
 392: 
 393: 
 394: def _run_chronos(prices: list[float], horizons: tuple = (1, 24), _ticker: str = "",
 395:                  timeout: int | None = None) -> dict | None:
 396:     """Run Chronos forecast (in-process, lazy-loaded) with GPU gating and timeout."""
 397:     if _chronos_circuit_open():
 398:         return None
 399: 
 400:     with gpu_gate("chronos", timeout=120) as acquired:
 401:         if not acquired:
 402:             logger.warning("GPU gate timeout for Chronos %s", _ticker)
 403:             return None
 404:         return _run_chronos_inner(prices, horizons, _ticker, timeout)
 405: 
 406: 
 407: def _run_chronos_inner(prices, horizons, _ticker, timeout):
 408:     """Chronos inference (called inside GPU gate)."""
 409:     t0 = time.time()
 410:     _timeout = timeout or _CHRONOS_TIMEOUT
 411:     try:
 412:         from portfolio.forecast_signal import forecast_chronos
 413: 
 414:         # Run in thread with timeout to prevent hangs
 415:         with ThreadPoolExecutor(max_workers=1) as executor:
 416:             future = executor.submit(forecast_chronos, "", prices, horizons=horizons)
 417:             try:
 418:                 result = future.result(timeout=_timeout)
 419:             except FuturesTimeout:
 420:                 ms = round((time.time() - t0) * 1000)
 421:                 logger.warning("Chronos timed out after %ds for %s", _timeout, _ticker)
 422:                 _log_health("chronos", _ticker, False, ms, f"timeout_{_timeout}s")
 423:                 _trip_chronos()
 424:                 return None
 425: 
 426:         ms = round((time.time() - t0) * 1000)
 427:         if result is None:
 428:             _log_health("chronos", _ticker, False, ms, "returned_none")
 429:             _trip_chronos()
 430:         else:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/cot_positioning.py'; for ("'$i = 140; $i -le 360; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 584ms:
 140: 
 141:     Formula: (Current - Min_156w) / (Max_156w - Min_156w) * 100
 142:     Returns 0-100 scale, or None if insufficient data.
 143:     """
 144:     if len(nc_net_history) < 10:  # Need minimum history for meaningful percentile
 145:         return None
 146: 
 147:     current = nc_net_history[0]  # Most recent
 148:     hist_min = min(nc_net_history)
 149:     hist_max = max(nc_net_history)
 150:     hist_range = hist_max - hist_min
 151: 
 152:     if hist_range == 0:
 153:         return 50.0  # No variation
 154: 
 155:     return round((current - hist_min) / hist_range * 100, 1)
 156: 
 157: 
 158: def _sub_cot_index(cot_data: dict, historical: list[dict]) -> tuple[str, float, dict]:
 159:     """Sub-indicator 1: COT Index percentile (contrarian).
 160: 
 161:     >80 = speculators extremely long = contrarian SELL.
 162:     <20 = speculators extremely short = contrarian BUY.
 163:     """
 164:     indicators = {"cot_index": None, "nc_net": None}
 165: 
 166:     nc_net = cot_data.get("noncomm_net")
 167:     if nc_net is None:
 168:         return "HOLD", 0.0, indicators
 169: 
 170:     indicators["nc_net"] = nc_net
 171: 
 172:     # Build history of nc_net values
 173:     nc_net_history = [nc_net]
 174:     for h in historical:
 175:         val = h.get("nc_net")
 176:         if val is not None:
 177:             nc_net_history.append(val)
 178: 
 179:     cot_index = _compute_cot_index(nc_net_history)
 180:     if cot_index is None:
 181:         return "HOLD", 0.0, indicators
 182: 
 183:     indicators["cot_index"] = cot_index
 184: 
 185:     if cot_index > _COT_EXTREME_HIGH:
 186:         # Speculators extremely long — contrarian SELL
 187:         intensity = min((cot_index - _COT_EXTREME_HIGH) / 20.0, 1.0)
 188:         return "SELL", round(0.4 + 0.3 * intensity, 2), indicators
 189:     elif cot_index < _COT_EXTREME_LOW:
 190:         # Speculators extremely short — contrarian BUY
 191:         intensity = min((_COT_EXTREME_LOW - cot_index) / 20.0, 1.0)
 192:         return "BUY", round(0.4 + 0.3 * intensity, 2), indicators
 193: 
 194:     return "HOLD", 0.0, indicators
 195: 
 196: 
 197: def _sub_commercial_change(cot_data: dict) -> tuple[str, dict]:
 198:     """Sub-indicator 2: Commercial hedger net change (smart money).
 199: 
 200:     Commercial traders are hedgers with actual exposure — their positioning
 201:     reflects fundamental supply/demand knowledge.  Increasing net long
 202:     (less short) = bullish signal.
 203:     """
 204:     indicators = {"comm_net": None, "comm_net_change": None}
 205: 
 206:     comm_net = cot_data.get("comm_net")
 207:     if comm_net is None:
 208:         return "HOLD", indicators
 209: 
 210:     indicators["comm_net"] = comm_net
 211: 
 212:     # Use the WoW change from metals_precompute if available
 213:     change = cot_data.get("noncomm_net_change")
 214:     if change is not None:
 215:         # If non-commercial net is INCREASING, commercials are getting more short
 216:         # (they're the counterparty).  Speculators adding longs = bearish contrarian.
 217:         indicators["comm_net_change"] = -change  # Commercial change is inverse
 218:         if change > _COMM_CHANGE_THRESHOLD:
 219:             return "SELL", indicators  # Specs adding longs aggressively
 220:         elif change < -_COMM_CHANGE_THRESHOLD:
 221:             return "BUY", indicators   # Specs liquidating longs
 222: 
 223:     return "HOLD", indicators
 224: 
 225: 
 226: def _sub_managed_money(cot_data: dict, historical: list[dict]) -> tuple[str, dict]:
 227:     """Sub-indicator 3: Managed money (hedge fund) sentiment.
 228: 
 229:     Managed money net long/short intensity relative to recent history.
 230:     Extreme long = contrarian SELL.  Extreme short = contrarian BUY.
 231:     """
 232:     indicators = {"mm_net": None, "mm_zscore": None}
 233: 
 234:     mm_net = cot_data.get("managed_money_net")
 235:     if mm_net is None:
 236:         return "HOLD", indicators
 237: 
 238:     indicators["mm_net"] = mm_net
 239: 
 240:     # Compute z-score against local history
 241:     mm_history = [mm_net]
 242:     for e in historical:
 243:         val = e.get("mm_net")
 244:         if val is not None:
 245:             mm_history.append(val)
 246: 
 247:     if len(mm_history) < 5:
 248:         return "HOLD", indicators
 249: 
 250:     import numpy as np
 251: 
 252:     mean = np.mean(mm_history)
 253:     std = np.std(mm_history)
 254:     if std < 1:
 255:         return "HOLD", indicators
 256: 
 257:     zscore = (mm_net - mean) / std
 258:     indicators["mm_zscore"] = round(float(zscore), 2)
 259: 
 260:     if zscore > _MM_ZSCORE_HIGH:
 261:         return "SELL", indicators  # Managed money extremely long — contrarian SELL
 262:     elif zscore < _MM_ZSCORE_LOW:
 263:         return "BUY", indicators   # Managed money extremely short — contrarian BUY
 264: 
 265:     return "HOLD", indicators
 266: 
 267: 
 268: def _sub_real_yield(deep_ctx: dict, ticker: str) -> tuple[str, dict]:
 269:     """Sub-indicator 4: Real yield direction (gold-specific).
 270: 
 271:     Falling real yields = lower opportunity cost for gold = BUY.
 272:     Rising real yields = higher opportunity cost = SELL.
 273: 
 274:     CAVEAT: The gold-real yield inverse correlation has weakened since 2022.
 275:     This sub-indicator carries lower weight and is gated by direction clarity.
 276:     """
 277:     indicators = {"real_yield": None, "real_yield_direction": None}
 278: 
 279:     # Try to get FRED data from deep context
 280:     fred = None
 281:     if isinstance(deep_ctx, dict):
 282:         # Check in refresh_data section
 283:         refresh = deep_ctx.get("refresh_data", {})
 284:         if isinstance(refresh, dict):
 285:             fred = refresh.get("fred")
 286:         # Also check top-level
 287:         if fred is None:
 288:             fred = deep_ctx.get("fred")
 289: 
 290:     if not isinstance(fred, dict):
 291:         return "HOLD", indicators
 292: 
 293:     real_yield = fred.get("real_yield")
 294:     direction = fred.get("real_yield_direction")
 295: 
 296:     if real_yield is not None:
 297:         indicators["real_yield"] = real_yield
 298:     if direction is not None:
 299:         indicators["real_yield_direction"] = direction
 300: 
 301:     if direction == "falling":
 302:         return "BUY", indicators  # Falling yields = bullish gold
 303:     elif direction == "rising":
 304:         return "SELL", indicators  # Rising yields = bearish gold
 305: 
 306:     return "HOLD", indicators
 307: 
 308: 
 309: def compute_cot_positioning_signal(
 310:     df: Any,
 311:     context: dict | None = None,
 312:     **kwargs,
 313: ) -> dict:
 314:     """Compute COT positioning composite signal for precious metals.
 315: 
 316:     Args:
 317:         df: Price DataFrame (used minimally — this is a fundamental signal).
 318:         context: Must contain 'ticker' key for asset identification.
 319: 
 320:     Returns:
 321:         Standard signal dict with action, confidence, sub_signals, indicators.
 322:     """
 323:     empty = {
 324:         "action": "HOLD",
 325:         "confidence": 0.0,
 326:         "sub_signals": {},
 327:         "indicators": {},
 328:     }
 329: 
 330:     # Extract ticker from context or kwargs
 331:     ticker = ""
 332:     if context and isinstance(context, dict):
 333:         ticker = context.get("ticker", "")
 334:     ticker = kwargs.get("ticker", ticker)
 335: 
 336:     if ticker not in _METALS_TICKERS:
 337:         return empty
 338: 
 339:     metal = _COMMODITY_MAP[ticker]
 340: 
 341:     # Load precomputed deep context
 342:     deep_ctx = _load_deep_context(ticker)
 343: 
 344:     # Extract COT data from deep context
 345:     cot_data = None
 346:     if deep_ctx:
 347:         refresh = deep_ctx.get("refresh_data", {})
 348:         if isinstance(refresh, dict):
 349:             cot_key = f"cot_{metal}"
 350:             cot_data = refresh.get(cot_key)
 351: 
 352:     if not cot_data or not isinstance(cot_data, dict):
 353:         logger.debug("No COT data available for %s", ticker)
 354:         return empty
 355: 
 356:     # Load local COT history for z-score computation
 357:     local_history = _load_cot_history(metal)
 358: 
 359:     # If local history is thin (<20 entries), fetch from CFTC API
 360:     historical_data = []

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/cot_positioning.py'; for ("'$i = 360; $i -le 460; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 547ms:
 360:     historical_data = []
 361:     if len(local_history) < 20:
 362:         cftc_name = "GOLD" if metal == "gold" else "SILVER"
 363:         historical_data = _fetch_cot_historical(cftc_name)
 364:     else:
 365:         historical_data = local_history
 366: 
 367:     # Compute sub-indicators
 368:     votes = []
 369:     sub_signals = {}
 370:     all_indicators = {}
 371: 
 372:     # Sub 1: COT Index (contrarian percentile)
 373:     cot_vote, cot_conf, cot_ind = _sub_cot_index(cot_data, historical_data)
 374:     sub_signals["cot_index"] = cot_vote
 375:     all_indicators.update(cot_ind)
 376:     votes.append(cot_vote)
 377: 
 378:     # Sub 2: Commercial hedger change (smart money)
 379:     comm_vote, comm_ind = _sub_commercial_change(cot_data)
 380:     sub_signals["commercial_change"] = comm_vote
 381:     all_indicators.update(comm_ind)
 382:     votes.append(comm_vote)
 383: 
 384:     # Sub 3: Managed money intensity
 385:     mm_vote, mm_ind = _sub_managed_money(cot_data, historical_data)
 386:     sub_signals["managed_money"] = mm_vote
 387:     all_indicators.update(mm_ind)
 388:     votes.append(mm_vote)
 389: 
 390:     # Sub 4: Real yield direction (gold-specific, lower weight)
 391:     ry_vote, ry_ind = _sub_real_yield(deep_ctx, ticker)
 392:     sub_signals["real_yield"] = ry_vote
 393:     all_indicators.update(ry_ind)
 394:     votes.append(ry_vote)
 395: 
 396:     # Add report date to indicators
 397:     all_indicators["report_date"] = cot_data.get("report_date")
 398:     all_indicators["ticker"] = ticker
 399: 
 400:     # Majority vote
 401:     action, confidence = majority_vote(votes, count_hold=False)
 402: 
 403:     # Cap confidence at 0.7 (external data, weekly update cycle)
 404:     confidence = min(confidence, 0.7)
 405: 
 406:     return {
 407:         "action": action,
 408:         "confidence": round(confidence, 4),
 409:         "sub_signals": sub_signals,
 410:         "indicators": all_indicators,
 411:     }
 412: 
 413: 
 414: 
 415: 
 416: 
 417: 
 418: 
 419: 
 420: 
 421: 
 422: 
 423: 
 424: 
 425: 
 426: 
 427: 
 428: 
 429: 
 430: 
 431: 
 432: 
 433: 
 434: 
 435: 
 436: 
 437: 
 438: 
 439: 
 440: 
 441: 
 442: 
 443: 
 444: 
 445: 
 446: 
 447: 
 448: 
 449: 
 450: 
 451: 
 452: 
 453: 
 454: 
 455: 
 456: 
 457: 
 458: 
 459: 
 460: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_cache\\[.*\\]\\[\\\"ts\\\"\\]|time\\.time\\(|monotonic\\(|_needs_refresh\\(\" portfolio/signals/claude_fundamental.py -S" in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-10T15:29:27.727463Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
rg: regex parse error:
    (?:_cache\[.*\]\[\)
    ^
error: unclosed group

 exited 1 in 516ms:
rg: regex parse error:
    (?:_cache\[.*\]\[\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"time\\.time\\(|_needs_refresh\\(|\\[\\\"ts\\\"\\]\" portfolio/signals/claude_fundamental.py -S" in Q:\finance-analyzer\.worktrees\adv-signals-modules
2026-05-10T15:29:34.419581Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
rg: regex parse error:
    (?:time\.time\(|_needs_refresh\(|\[\)
    ^
error: unclosed group

 exited 1 in 580ms:
rg: regex parse error:
    (?:time\.time\(|_needs_refresh\(|\[\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'portfolio/signals/claude_fundamental.py' -Pattern 'time.time\\(','_needs_refresh\\(','\\[\"ts\"\\]' | ForEach-Object { \"{0}:{1}: {2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 541ms:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:92: def _needs_refresh(tier, cooldowns):
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:94: return time.time() - _cache[tier]["ts"] > cooldowns[tier]
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:158: now = time.time()
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:159: if now - _earnings_cache["ts"] < _EARNINGS_CACHE_TTL and _earnings_cache["data"]:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:202: _earnings_cache["ts"] = now
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:713: _cache[tier]["ts"] = time.time()
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:925: if _needs_refresh(tier, cooldowns):
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:927: if _needs_refresh(tier, cooldowns):
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:929: _cache[tier]["ts"] = time.time()

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/claude_fundamental.py'; foreach ("'$range in @(@{s=680;e=740}, @{s=900;e=940})) { for ($i=$range.s; $i -le $range.e; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }; Write-Output '"'' }" in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 583ms:
 680:     timeouts = _get_timeouts(config)
 681: 
 682:     # Read the compact summary
 683:     summary_path = DATA_DIR / "agent_summary_compact.json"
 684:     if not summary_path.exists():
 685:         logger.warning("agent_summary_compact.json not found, skipping %s refresh", tier)
 686:         return
 687:     summary = load_json(summary_path, default={})
 688:     macro = context.get("macro")
 689:     # Also try to get macro from summary if not in context
 690:     if not macro:
 691:         macro = summary.get("macro", {})
 692: 
 693:     if tier == "haiku":
 694:         prompt = _build_haiku_prompt(summary, macro)
 695:         raw = _call_claude_cli(models["haiku"], prompt, timeout=timeouts["haiku"])
 696:         results = _parse_haiku_response(raw)
 697:     elif tier == "sonnet":
 698:         prompt = _build_sonnet_prompt(summary, macro)
 699:         raw = _call_claude_cli(models["sonnet"], prompt, timeout=timeouts["sonnet"])
 700:         results = _parse_sonnet_response(raw)
 701:     else:  # opus
 702:         portfolios = {}
 703:         for pf in ("portfolio_state.json", "portfolio_state_bold.json"):
 704:             pf_path = DATA_DIR / pf
 705:             if pf_path.exists():
 706:                 portfolios[pf] = load_json(pf_path, default={})
 707:         prompt = _build_opus_prompt(summary, macro, portfolios)
 708:         raw = _call_claude_cli(models["opus"], prompt, timeout=timeouts["opus"])
 709:         results = _parse_opus_response(raw)
 710: 
 711:     with _lock:
 712:         _cache[tier]["results"] = results
 713:         _cache[tier]["ts"] = time.time()
 714: 
 715:     logger.info("Claude fundamental %s refreshed: %d tickers", tier, len(results))
 716: 
 717:     # Journal the results for accuracy tracking and debugging
 718:     _journal_refresh(tier, results)
 719: 
 720: 
 721: def _bias_rate_from_entries(entries: list, tier: str, ticker: str | None = None) -> tuple[float, int, str | None]:
 722:     """Compute (bias_rate, non_hold_n, biased_label) for a tier (optionally
 723:     scoped to one ticker) from a journal entry list.
 724: 
 725:     Returns (0.0, 0, None) when there's not enough data; the caller decides
 726:     the threshold + min-samples to apply.
 727:     """
 728:     votes = [
 729:         e.get("action", "HOLD")
 730:         for e in entries
 731:         if e.get("tier") == tier
 732:         and not e.get("is_abstention", False)
 733:         and (ticker is None or e.get("ticker") == ticker)
 734:     ]
 735:     # Only check most recent 30 votes for the (tier, ticker?) slice
 736:     votes = votes[-30:]
 737:     non_hold = [v for v in votes if v != "HOLD"]
 738:     if not non_hold:
 739:         return 0.0, 0, None
 740:     from collections import Counter

 900: 
 901:     Returns:
 902:         dict with action, confidence, sub_signals, indicators
 903:     """
 904:     if context is None:
 905:         return dict(_DEFAULT_HOLD)
 906: 
 907:     config = context.get("config", {})
 908:     cf_config = config.get("claude_fundamental", {})
 909:     if not cf_config.get("enabled", True):
 910:         return dict(_DEFAULT_HOLD)
 911: 
 912:     # Market hours gate — only refresh during EU+US hours (07:00-21:00 UTC weekdays).
 913:     # Fundamentals don't change overnight and we don't want to waste claude calls.
 914:     # Cached results from the last open-hours refresh are still served during off-hours.
 915:     from portfolio.market_timing import get_market_state
 916:     market_state, _, _ = get_market_state()
 917:     skip_refresh = market_state in ("closed", "weekend", "holiday")
 918: 
 919:     cooldowns = _get_cooldowns(config)
 920: 
 921:     # Refresh in background thread — never block the signal loop.
 922:     # Fundamentals change on a hours/days timescale, not minutes.
 923:     if not skip_refresh:
 924:         for tier in ("haiku", "sonnet", "opus"):
 925:             if _needs_refresh(tier, cooldowns):
 926:                 with _lock:
 927:                     if _needs_refresh(tier, cooldowns):
 928:                         # Mark as refreshing to prevent duplicate spawns
 929:                         _cache[tier]["ts"] = time.time()
 930:                         t = threading.Thread(
 931:                             target=_bg_refresh, args=(tier, context),
 932:                             daemon=True, name=f"cf-{tier}",
 933:                         )
 934:                         t.start()
 935: 
 936:     # Cascade lookup for this ticker
 937:     ticker = context.get("ticker", "")
 938:     result = _get_best_result(ticker)
 939:     if result:
 940:         return dict(result)  # copy to avoid mutation


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/metals_cross_asset.py'; for ("'$i = 360; $i -le 450; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 576ms:
 360: 
 361:     # Sub 4: G/S Ratio Velocity — now intraday when available
 362:     # Falling G/S ratio = silver outperforming gold = bullish silver
 363:     # Rising G/S ratio = gold outperforming silver = bearish silver
 364:     gs_vel = ctx["gs_velocity_pct"]
 365:     if is_silver:
 366:         if gs_vel < -gs_vel_thr:
 367:             sub_signals["gs_velocity"] = "BUY"   # Silver gaining vs gold
 368:         elif gs_vel > gs_vel_thr:
 369:             sub_signals["gs_velocity"] = "SELL"   # Silver losing vs gold
 370:         else:
 371:             sub_signals["gs_velocity"] = "HOLD"
 372:     else:
 373:         # For gold: rising G/S = gold outperforming -> BUY gold
 374:         if gs_vel > gs_vel_thr:
 375:             sub_signals["gs_velocity"] = "BUY"
 376:         elif gs_vel < -gs_vel_thr:
 377:             sub_signals["gs_velocity"] = "SELL"
 378:         else:
 379:             sub_signals["gs_velocity"] = "HOLD"
 380:     votes.append(sub_signals["gs_velocity"])
 381: 
 382:     # Sub 5: SPY Momentum (risk-on/risk-off) — now intraday when available
 383:     spy = ctx["spy_change_pct"]
 384:     if spy > spy_thr:
 385:         # Risk-on: silver benefits (industrial), gold neutral
 386:         sub_signals["spy_risk"] = "BUY" if is_silver else "HOLD"
 387:     elif spy < -spy_thr:
 388:         # Risk-off: gold benefits (safe haven), silver hurt
 389:         sub_signals["spy_risk"] = "BUY" if not is_silver else "SELL"
 390:     else:
 391:         sub_signals["spy_risk"] = "HOLD"
 392:     votes.append(sub_signals["spy_risk"])
 393: 
 394:     # Sub 6: Oil Momentum (inflation expectations) — now intraday when available
 395:     oil = ctx["oil_change_pct"]
 396:     if oil > oil_thr:
 397:         sub_signals["oil"] = "BUY"
 398:     elif oil < -oil_thr:
 399:         sub_signals["oil"] = "SELL"
 400:     else:
 401:         sub_signals["oil"] = "HOLD"
 402:     votes.append(sub_signals["oil"])
 403: 
 404:     # Sub 7: EPU (Economic Policy Uncertainty) — daily FRED
 405:     # High uncertainty → safe-haven demand → BUY both gold and silver
 406:     # Low uncertainty → risk-on → less safe-haven premium → SELL
 407:     fred_key = _get_fred_key(context)
 408:     epu_values = _fetch_fred_values(_EPU_FRED_SERIES, fred_key, _epu_cache)
 409:     epu_zscore = _compute_zscore(epu_values) if epu_values else 0.0
 410:     if epu_zscore > _EPU_ZSCORE_HIGH:
 411:         sub_signals["epu"] = "BUY"
 412:     elif epu_zscore < _EPU_ZSCORE_LOW:
 413:         sub_signals["epu"] = "SELL"
 414:     else:
 415:         sub_signals["epu"] = "HOLD"
 416:     votes.append(sub_signals["epu"])
 417: 
 418:     # Sub 8: TIPS Real Yield direction — daily FRED (DFII10)
 419:     # Falling real yields → lower opportunity cost of holding metals → BUY
 420:     # Rising real yields → higher opportunity cost → SELL
 421:     tips_values = _fetch_fred_values(_TIPS_FRED_SERIES, fred_key, _tips_cache)
 422:     if tips_values and len(tips_values) >= 10:
 423:         tips_recent = sum(tips_values[:5]) / 5
 424:         tips_older = sum(tips_values[5:10]) / 5
 425:         tips_change = tips_recent - tips_older
 426:     else:
 427:         tips_change = 0.0
 428: 
 429:     if tips_change < -_TIPS_CHANGE_THRESHOLD:
 430:         sub_signals["tips_yield"] = "BUY"
 431:     elif tips_change > _TIPS_CHANGE_THRESHOLD:
 432:         sub_signals["tips_yield"] = "SELL"
 433:     else:
 434:         sub_signals["tips_yield"] = "HOLD"
 435:     votes.append(sub_signals["tips_yield"])
 436: 
 437:     action, confidence = majority_vote(votes)
 438: 
 439:     return {
 440:         "action": action,
 441:         "confidence": round(confidence, 4),
 442:         "sub_signals": sub_signals,
 443:         "indicators": {
 444:             "copper_change": round(cu, 3),
 445:             "gvz_zscore": round(gvz, 2),
 446:             "gs_ratio_zscore": round(gsr, 2),
 447:             "gs_velocity": round(gs_vel, 3),
 448:             "spy_change": round(spy, 3),
 449:             "oil_change": round(oil, 3),
 450:             "epu_zscore": round(epu_zscore, 2),

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'portfolio/signals/realized_skewness.py','portfolio/signals/statistical_jump_regime.py','portfolio/signals/hurst_regime.py' -Pattern 'disabled|pending validation|shadow|warm-up' -CaseSensitive:"'$false | ForEach-Object { "{0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 542ms:
codex
I’ve finished the direct checks for most of Claude’s items. One additional code-path in `claude_fundamental` now looks suspect on its own merits, so I’m verifying that separately before I finalize the “new findings” section.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/claude_fundamental.py'; for ("'$i = 600; $i -le 680; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 554ms:
 600:     results = {}
 601:     for ticker, val in data.items():
 602:         if not isinstance(val, dict):
 603:             continue
 604:         sub_signals = val.get("sub_signals", {})
 605:         clean_subs = {}
 606:         for name in SUB_SIGNAL_NAMES:
 607:             vote = sub_signals.get(name, "HOLD").upper()
 608:             if vote not in ("BUY", "SELL", "HOLD"):
 609:                 vote = "HOLD"
 610:             clean_subs[name] = vote
 611: 
 612:         votes = list(clean_subs.values())
 613:         action, conf = majority_vote(votes)
 614:         conf = min(conf, _MAX_CONFIDENCE)
 615: 
 616:         # Opus provides its own conviction — use it to scale confidence
 617:         opus_conviction = min(float(val.get("conviction", 0.5)), 1.0)
 618:         # Blend: majority_vote confidence weighted by Opus conviction
 619:         conf = min(conf * opus_conviction, _MAX_CONFIDENCE)
 620: 
 621:         results[ticker] = {
 622:             "action": action,
 623:             "confidence": round(conf, 4),
 624:             "sub_signals": clean_subs,
 625:             "indicators": {
 626:                 "_tier": "opus",
 627:                 "reasoning": val.get("reasoning", ""),
 628:                 "conviction": opus_conviction,
 629:                 "contrarian_flag": bool(val.get("contrarian_flag", False)),
 630:             },
 631:         }
 632:     return results
 633: 
 634: 
 635: def _journal_refresh(tier: str, results: dict) -> None:
 636:     """Persist tier refresh results for accuracy tracking and debugging.
 637: 
 638:     2026-04-21 v2: LOG EVERYTHING. An earlier v1 of this function silently
 639:     skipped rows that looked like abstentions (Haiku HOLD-0.0-empty-reason).
 640:     That was the wrong call — those rows are load-bearing evidence of how
 641:     often each tier declines to make a call, and the abstention rate itself
 642:     is a health signal. The revised approach is to ALWAYS write the row but
 643:     tag intentional abstentions with `is_abstention: true` so downstream
 644:     analysis can filter at read-time without losing the data. See the
 645:     `feedback_log_everything.md` memory for the full rationale.
 646:     """
 647:     import datetime as _dt
 648: 
 649:     from portfolio.file_utils import atomic_append_jsonl
 650: 
 651:     ts = _dt.datetime.now(_dt.UTC).isoformat()
 652:     for ticker, result in results.items():
 653:         action = result.get("action", "HOLD")
 654:         confidence = result.get("confidence", 0.0)
 655:         reasoning = result.get("indicators", {}).get("reasoning", "")
 656:         is_abstention = (
 657:             action == "HOLD" and confidence == 0.0 and not reasoning
 658:         )
 659:         entry = {
 660:             "ts": ts,
 661:             "tier": tier,
 662:             "ticker": ticker,
 663:             "action": action,
 664:             "confidence": confidence,
 665:             "sub_signals": result.get("sub_signals", {}),
 666:             "reasoning": reasoning,
 667:             "contrarian_flag": result.get("indicators", {}).get("contrarian_flag", False),
 668:             "is_abstention": is_abstention,
 669:         }
 670:         try:
 671:             atomic_append_jsonl(_CF_LOG, entry)
 672:         except Exception as e:
 673:             logger.warning("Failed to journal cf result: %s", e)
 674: 
 675: 
 676: def _refresh_tier(tier, context):
 677:     """Refresh one tier's cache by calling the claude CLI."""
 678:     config = context.get("config", {})
 679:     models = _get_models(config)
 680:     timeouts = _get_timeouts(config)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'portfolio/signals/claude_fundamental.py' -Pattern 'def _bg_refresh|try:|except Exception' | ForEach-Object { \"{0}:{1}: {2}\" -f "'$_.Path, $_.LineNumber, $_.Line.Trim() }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 533ms:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:164: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:171: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:197: except Exception:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:296: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:299: except Exception as e:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:518: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:526: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:535: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:670: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:672: except Exception as e:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:763: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:766: except Exception:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:809: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:812: except Exception:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:886: def _bg_refresh(tier, context):
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:888: try:
Q:\finance-analyzer\.worktrees\adv-signals-modules\portfolio\signals\claude_fundamental.py:890: except Exception as e:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$lines = Get-Content '"'portfolio/signals/claude_fundamental.py'; for ("'$i = 880; $i -le 900; $i++) { Write-Output (("{0,4}: {1}" -f $i, $lines[$i-1])) }' in Q:\finance-analyzer\.worktrees\adv-signals-modules
 succeeded in 559ms:
 880:         result = effective.get(tier)
 881:         if result:
 882:             return result
 883:     return None
 884: 
 885: 
 886: def _bg_refresh(tier, context):
 887:     """Background refresh — runs in a daemon thread, never blocks the loop."""
 888:     try:
 889:         _refresh_tier(tier, context)
 890:     except Exception as e:
 891:         logger.warning("Claude fundamental %s bg-refresh failed: %s", tier, e)
 892: 
 893: 
 894: def compute_claude_fundamental_signal(df: pd.DataFrame, context: dict = None) -> dict:
 895:     """Signal entry point — called once per ticker by signal_engine.
 896: 
 897:     Args:
 898:         df: OHLCV DataFrame (not used directly — fundamentals come from LLM knowledge)
 899:         context: dict with keys: ticker, config, macro
 900: 

codex
The verdicts are settled. One separate issue is real in `claude_fundamental`: a failed background refresh suppresses retries for a full cooldown because the cache timestamp is advanced before the worker succeeds.
  ✓ Inspect each cited file/line and capture the relevant code paths
  ✓ Classify each finding as confirmed, partial, or false-positive with code-based reasons
  ✓ Scan for any materially new issues encountered during inspection and draft the final verdict summary
codex
## Verdicts

- [unrated] Ichimoku cloud uses future data via forward-shifted spans — portfolio/signals/trend.py:325
  Verdict: FALSE-POSITIVE. Reason: `shift(26)` at lines 325-326 moves each span to later indexes, so `iloc[-1]` reads the cloud projected from 26 bars ago, exactly as the docstring states at lines 306-308; it does not read future bars.

- [unrated] BB squeeze threshold allows divide-by-zero when `avg_width` is zero or NaN — portfolio/signals/volatility.py:57
  Verdict: FALSE-POSITIVE. Reason: the only division is already guarded by `bb_middle.replace(0, np.nan)` at line 50, and line 57 turns zero/NaN `avg_width` into `False` rather than an exception.

- [unrated] StochRSI divides by zero when the RSI range collapses — portfolio/signals/momentum.py:131
  Verdict: FALSE-POSITIVE. Reason: line 131 explicitly replaces zero denominators with `NaN`, and lines 133-135 return `HOLD` on `NaN`, so there is no unsafe divide-by-zero path.

- [unrated] ROC uses the wrong denominator and confuses level with acceleration — portfolio/signals/momentum.py:201
  Verdict: FALSE-POSITIVE. Reason: `(close - close.shift(period)) / close.shift(period)` at line 201 is the standard ROC formula, and the acceleration check is the separate `val > prev` / `val < prev` logic at lines 212 and 216.

- [unrated] FOMC date cache goes stale because `_FOMC_MAX_DATE` is computed at import — portfolio/signals/calendar_seasonal.py:256
  Verdict: PARTIAL. Reason: the real stale-data risk is the static `_FOMC_ANNOUNCEMENT_DATES` list itself, not the cached `max()` at line 37, and after `last_date > _FOMC_MAX_DATE` this sub-signal warns then returns `HOLD` rather than emitting a bad FOMC vote. Adjustment: the cause and impact are overstated; this is a static-calendar maintenance issue, not a `max()` caching bug.

- [unrated] `post_event_relief` can overlap with a new pre-event risk window — portfolio/signals/econ_calendar.py:127
  Verdict: FALSE-POSITIVE. Reason: after finding a relief event, lines 135-138 explicitly suppress the BUY unless the next event is absent or more than 24 hours away.

- [unrated] FRED data is newest-first but the metals z-score assumes oldest-to-newest — portfolio/signals/metals_cross_asset.py:150
  Verdict: FALSE-POSITIVE. Reason: `_fetch_fred_values()` documents and returns newest-first data at line 107, and `_compute_zscore()` intentionally treats `values[0]` as current at line 177; mean and standard deviation are order-invariant anyway.

- [unrated] Deep COT context has no staleness check and can stay week-old indefinitely — portfolio/signals/cot_positioning.py:33
  Verdict: CONFIRMED. Reason: `_load_deep_context()` at lines 67-72 loads and returns the JSON unconditionally, and `compute_cot_positioning_signal()` consumes it at lines 341-354 without any mtime or report-date freshness check.

- [unrated] OAS z-score includes the current value in its own historical sample — portfolio/signals/credit_spread.py:159
  Verdict: PARTIAL. Reason: line 154 uses `history = values[:lookback]` while line 155 also treats `values[0]` as current, so the current point does damp its own z-score slightly. Adjustment: the issue is real but minor statistical self-dilution, not a severe directional bias.

- [unrated] Earnings calendar TTL is too short and causes wasteful yfinance calls — portfolio/signals/claude_fundamental.py:149
  Verdict: FALSE-POSITIVE. Reason: lines 147-160 show a deliberate 12-hour cache for external calendar data, which is a tuning choice rather than a correctness defect.

- [unrated] Forecast TTL of 5 minutes makes 12h/24h horizon predictions stale — portfolio/signals/forecast.py:37
  Verdict: FALSE-POSITIVE. Reason: `_FORECAST_TTL` is only applied to candle loading at line 782, while the forecast models are run afterward on each call; this is not a cached-prediction TTL.

- [unrated] `TICKER_SECTORS` is hardcoded at module load in `news_event.py` — portfolio/signals/news_event.py:149
  Verdict: FALSE-POSITIVE. Reason: `TICKER_SECTORS` is imported inside `_fetch_headlines()` at lines 149-151, not at module import time.

- [unrated] Forecast candles are cached without any stale-ohlcv validation — portfolio/signals/forecast.py:789
  Verdict: PARTIAL. Reason: there is no freshness check before using cached candles, but the broader cause is `_load_candles_ohlcv()` itself discarding timestamps at lines 297-305, so this is not specific to line 789 or the Kronos-only path. Adjustment: the bug is real, but it is a loader-level design gap affecting both candle fetch paths.

- [unrated] Cooldown logic uses wall-clock time instead of monotonic time — portfolio/signals/claude_fundamental.py:94
  Verdict: PARTIAL. Reason: the code really does use `time.time()` for interval checks at lines 94, 713, and 929, so monotonic time would be more robust. Adjustment: Claude’s stated failure mode is wrong; a backward clock step delays refreshes, while a forward jump is what can expire cooldowns early.

- [unrated] ATR implementation is wrong because it uses EMA instead of Wilder RMA — portfolio/signals/volatility.py:36
  Verdict: FALSE-POSITIVE. Reason: the function is explicitly documented as “EMA-smoothed” at line 34 and intentionally implemented with `ewm(span=period)` at line 36.

- [unrated] Stochastic BUY threshold `<20` without volume confirmation is too noisy — portfolio/signals/momentum.py:110
  Verdict: FALSE-POSITIVE. Reason: this is strategy opinion, not a code defect; the implementation at lines 109-115 matches its own documented crossover rules.

- [unrated] Structure breakout logic is broken because it does not detect gaps — portfolio/signals/structure.py:48
  Verdict: FALSE-POSITIVE. Reason: the module never claims gap filtering, and the cited breakout functions at lines 48-114 are internally consistent with their stated period-high and Donchian logic.

- [unrated] Disabled modules are still imported by a signal registry and waste startup CPU — portfolio/signals/__init__.py:1
  Verdict: FALSE-POSITIVE. Reason: `portfolio/signals/__init__.py` contains no imports, `portfolio/signal_registry.py` is absent from this working tree, and repo-wide search found no references to `smart_money`, `oscillators`, or `orderbook_flow` outside their own modules.

- [unrated] `treasury_risk_rotation` was added live with no compatibility tests or validation data — portfolio/signals/treasury_risk_rotation.py:153
  Verdict: FALSE-POSITIVE. Reason: the file only defines `compute_treasury_risk_rotation_signal()` and its helpers; the claim about live activation, shadow-mode policy, or missing validation process is not substantiated anywhere in the inspected source.

- [unrated] `realized_skewness`, `statistical_jump_regime`, and `hurst_regime` are disabled stale “pending validation” modules — portfolio/signals/realized_skewness.py:1
  Verdict: FALSE-POSITIVE. Reason: none of the three files contains `disabled` or `pending validation` markers, and repo-wide search found no `DISABLED_SIGNALS` metadata in this working tree.

## New findings (you, not Claude)

- [P3] Failed Claude refresh suppresses retries for a full cooldown window — portfolio/signals/claude_fundamental.py:929
  `compute_claude_fundamental_signal()` sets `_cache[tier]["ts"] = time.time()` before starting the daemon thread at lines 925-934; if `_bg_refresh()` then fails or returns early (for example `_refresh_tier()` bails at lines 684-686 when `agent_summary_compact.json` is missing, or logs and swallows an exception at lines 888-891), `_needs_refresh()` treats the tier as fresh even though no new results were stored.

## Summary
- Confirmed: 1
- Partial: 4
- False-positive: 15
- New from you: 1
tokens used
124,583
## Verdicts

- [unrated] Ichimoku cloud uses future data via forward-shifted spans — portfolio/signals/trend.py:325
  Verdict: FALSE-POSITIVE. Reason: `shift(26)` at lines 325-326 moves each span to later indexes, so `iloc[-1]` reads the cloud projected from 26 bars ago, exactly as the docstring states at lines 306-308; it does not read future bars.

- [unrated] BB squeeze threshold allows divide-by-zero when `avg_width` is zero or NaN — portfolio/signals/volatility.py:57
  Verdict: FALSE-POSITIVE. Reason: the only division is already guarded by `bb_middle.replace(0, np.nan)` at line 50, and line 57 turns zero/NaN `avg_width` into `False` rather than an exception.

- [unrated] StochRSI divides by zero when the RSI range collapses — portfolio/signals/momentum.py:131
  Verdict: FALSE-POSITIVE. Reason: line 131 explicitly replaces zero denominators with `NaN`, and lines 133-135 return `HOLD` on `NaN`, so there is no unsafe divide-by-zero path.

- [unrated] ROC uses the wrong denominator and confuses level with acceleration — portfolio/signals/momentum.py:201
  Verdict: FALSE-POSITIVE. Reason: `(close - close.shift(period)) / close.shift(period)` at line 201 is the standard ROC formula, and the acceleration check is the separate `val > prev` / `val < prev` logic at lines 212 and 216.

- [unrated] FOMC date cache goes stale because `_FOMC_MAX_DATE` is computed at import — portfolio/signals/calendar_seasonal.py:256
  Verdict: PARTIAL. Reason: the real stale-data risk is the static `_FOMC_ANNOUNCEMENT_DATES` list itself, not the cached `max()` at line 37, and after `last_date > _FOMC_MAX_DATE` this sub-signal warns then returns `HOLD` rather than emitting a bad FOMC vote. Adjustment: the cause and impact are overstated; this is a static-calendar maintenance issue, not a `max()` caching bug.

- [unrated] `post_event_relief` can overlap with a new pre-event risk window — portfolio/signals/econ_calendar.py:127
  Verdict: FALSE-POSITIVE. Reason: after finding a relief event, lines 135-138 explicitly suppress the BUY unless the next event is absent or more than 24 hours away.

- [unrated] FRED data is newest-first but the metals z-score assumes oldest-to-newest — portfolio/signals/metals_cross_asset.py:150
  Verdict: FALSE-POSITIVE. Reason: `_fetch_fred_values()` documents and returns newest-first data at line 107, and `_compute_zscore()` intentionally treats `values[0]` as current at line 177; mean and standard deviation are order-invariant anyway.

- [unrated] Deep COT context has no staleness check and can stay week-old indefinitely — portfolio/signals/cot_positioning.py:33
  Verdict: CONFIRMED. Reason: `_load_deep_context()` at lines 67-72 loads and returns the JSON unconditionally, and `compute_cot_positioning_signal()` consumes it at lines 341-354 without any mtime or report-date freshness check.

- [unrated] OAS z-score includes the current value in its own historical sample — portfolio/signals/credit_spread.py:159
  Verdict: PARTIAL. Reason: line 154 uses `history = values[:lookback]` while line 155 also treats `values[0]` as current, so the current point does damp its own z-score slightly. Adjustment: the issue is real but minor statistical self-dilution, not a severe directional bias.

- [unrated] Earnings calendar TTL is too short and causes wasteful yfinance calls — portfolio/signals/claude_fundamental.py:149
  Verdict: FALSE-POSITIVE. Reason: lines 147-160 show a deliberate 12-hour cache for external calendar data, which is a tuning choice rather than a correctness defect.

- [unrated] Forecast TTL of 5 minutes makes 12h/24h horizon predictions stale — portfolio/signals/forecast.py:37
  Verdict: FALSE-POSITIVE. Reason: `_FORECAST_TTL` is only applied to candle loading at line 782, while the forecast models are run afterward on each call; this is not a cached-prediction TTL.

- [unrated] `TICKER_SECTORS` is hardcoded at module load in `news_event.py` — portfolio/signals/news_event.py:149
  Verdict: FALSE-POSITIVE. Reason: `TICKER_SECTORS` is imported inside `_fetch_headlines()` at lines 149-151, not at module import time.

- [unrated] Forecast candles are cached without any stale-ohlcv validation — portfolio/signals/forecast.py:789
  Verdict: PARTIAL. Reason: there is no freshness check before using cached candles, but the broader cause is `_load_candles_ohlcv()` itself discarding timestamps at lines 297-305, so this is not specific to line 789 or the Kronos-only path. Adjustment: the bug is real, but it is a loader-level design gap affecting both candle fetch paths.

- [unrated] Cooldown logic uses wall-clock time instead of monotonic time — portfolio/signals/claude_fundamental.py:94
  Verdict: PARTIAL. Reason: the code really does use `time.time()` for interval checks at lines 94, 713, and 929, so monotonic time would be more robust. Adjustment: Claude’s stated failure mode is wrong; a backward clock step delays refreshes, while a forward jump is what can expire cooldowns early.

- [unrated] ATR implementation is wrong because it uses EMA instead of Wilder RMA — portfolio/signals/volatility.py:36
  Verdict: FALSE-POSITIVE. Reason: the function is explicitly documented as “EMA-smoothed” at line 34 and intentionally implemented with `ewm(span=period)` at line 36.

- [unrated] Stochastic BUY threshold `<20` without volume confirmation is too noisy — portfolio/signals/momentum.py:110
  Verdict: FALSE-POSITIVE. Reason: this is strategy opinion, not a code defect; the implementation at lines 109-115 matches its own documented crossover rules.

- [unrated] Structure breakout logic is broken because it does not detect gaps — portfolio/signals/structure.py:48
  Verdict: FALSE-POSITIVE. Reason: the module never claims gap filtering, and the cited breakout functions at lines 48-114 are internally consistent with their stated period-high and Donchian logic.

- [unrated] Disabled modules are still imported by a signal registry and waste startup CPU — portfolio/signals/__init__.py:1
  Verdict: FALSE-POSITIVE. Reason: `portfolio/signals/__init__.py` contains no imports, `portfolio/signal_registry.py` is absent from this working tree, and repo-wide search found no references to `smart_money`, `oscillators`, or `orderbook_flow` outside their own modules.

- [unrated] `treasury_risk_rotation` was added live with no compatibility tests or validation data — portfolio/signals/treasury_risk_rotation.py:153
  Verdict: FALSE-POSITIVE. Reason: the file only defines `compute_treasury_risk_rotation_signal()` and its helpers; the claim about live activation, shadow-mode policy, or missing validation process is not substantiated anywhere in the inspected source.

- [unrated] `realized_skewness`, `statistical_jump_regime`, and `hurst_regime` are disabled stale “pending validation” modules — portfolio/signals/realized_skewness.py:1
  Verdict: FALSE-POSITIVE. Reason: none of the three files contains `disabled` or `pending validation` markers, and repo-wide search found no `DISABLED_SIGNALS` metadata in this working tree.

## New findings (you, not Claude)

- [P3] Failed Claude refresh suppresses retries for a full cooldown window — portfolio/signals/claude_fundamental.py:929
  `compute_claude_fundamental_signal()` sets `_cache[tier]["ts"] = time.time()` before starting the daemon thread at lines 925-934; if `_bg_refresh()` then fails or returns early (for example `_refresh_tier()` bails at lines 684-686 when `agent_summary_compact.json` is missing, or logs and swallows an exception at lines 888-891), `_needs_refresh()` treats the tier as fresh even though no new results were stored.

## Summary
- Confirmed: 1
- Partial: 4
- False-positive: 15
- New from you: 1
