# Claude Review — signals-modules

## P0 (money-losing or data-corrupting)

- `portfolio/signals/econ_calendar.py:137` — stale calendar returns spurious BUY when `next_event()` returns None
  ```python
  evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
  if evt is None or evt["hours_until"] > 24:
      return "BUY", indicators
  ```
  In `_post_event_relief`, the check intends "no upcoming event within 24h → calm → BUY". When `next_event()` returns None it actually means **calendar exhausted** (all hardcoded dates are in the past, e.g. running in 2028 or with future dates in tests). Silent BUY on data staleness. The other event-free path (line 141-145) correctly guards with `evt is not None`. Fix: `if evt is not None and evt["hours_until"] > 24:`. Confidence 92.

- `portfolio/signals/credit_spread.py:285` — relative path `config.json` load fails when CWD ≠ repo root
  ```python
  cfg = load_json("config.json", default={}) or {}
  fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
  ```
  PF-DataLoop scheduled task can launch from `C:\Windows`. `cot_positioning.py` already fixed the same bug (SM-P1-4 — uses `_DATA_DIR = Path(__file__).resolve()...`). When CWD wrong, returns `{}`, FRED key empty, signal returns HOLD silently. `except Exception: pass` swallows. Active for BTC-USD/ETH-USD at non-gated horizons. Fix: absolute path. Confidence 88.

## P1 (high-confidence bugs)

- `portfolio/signals/volume_flow.py:323-324` — NaN price change defaults `price_up=True`, biasing volume RSI toward BUY
  ```python
  price_change = df["close"].diff().iloc[-1]
  price_up = price_change > 0 if not np.isnan(price_change) else True  # default neutral bias
  ```
  `_vote_volume_rsi` fires when `vrsi_val > 70`. `price_up=True` → BUY, False → SELL. NaN default to True means any volume spike on the first bar → BUY rather than HOLD. Comment says "neutral bias" but BUY is directional. Fix: return HOLD when NaN. Confidence 84.

- `portfolio/signals/volatility.py:160, 264` — inconsistent annualization within same composite signal
  `_historical_volatility` uses `np.sqrt(365)`. `_garch_signal` uses `np.sqrt(252)`. Both feed `compute_volatility_signal`. Indicator values to logs/consumers on different scales — same daily HV reports as 287% (HV) vs 238% (GARCH). Vote direction unaffected; consumer values misleading. Fix: per-asset-class convention. Confidence 85.

- `portfolio/signals/futures_flow.py:118` — unguarded `ls_ratio[-1]["longShortRatio"]`
  ```python
  if not ls_ratio:
      return "HOLD"
  latest = ls_ratio[-1]["longShortRatio"]
  ```
  Guard checks emptiness but not key presence. Binance schema change → `KeyError`. Same in `_top_vs_crowd` lines 135-136. Outer dispatch catches as HOLD but trips error counter and circuit breaker unnecessarily. Use `.get("longShortRatio")` with None check. Confidence 80.

- `portfolio/signals/cot_positioning.py:212-221` — `_sub_commercial_change` HOLD guard checks wrong field
  ```python
  change = cot_data.get("noncomm_net_change")  # reads NON-COMMERCIAL change
  if change is not None:
      indicators["comm_net_change"] = -change   # stored as COMMERCIAL (inverted)
      if change > _COMM_CHANGE_THRESHOLD:
          return "SELL", indicators
  ```
  HOLD guard at line 207-208 checks `comm_net is None` but vote requires `noncomm_net_change`. If `comm_net` present but `noncomm_net_change` None → silent HOLD via implicit fall-through. If `noncomm_net_change` present but `comm_net` None → returns HOLD before looking at the data. Confidence 81.

- `portfolio/signals/calendar_seasonal.py` — applies stock-market calendar effects (Monday=SELL, Sell-in-May, Santa, FOMC Drift) to 24/7 crypto/metals
  No ticker-aware gate. Per-ticker disabled at 1d for BTC/ETH/XAU/XAG (100% BUY bias confirms structural mismatch), but **active at 3h** for crypto/metals. Fix: add asset-class guard or `applicable_assets=["stocks"]` in registry. Confidence 83.

## P2 (concerns / smells)

- `portfolio/signals/volatility.py:244-251` — GARCH `len(returns) < 20` guard is dead code given `len(close) >= lookback=100`
  ```python
  prices = close.iloc[-lookback:].values
  returns = np.diff(np.log(prices))  # len = lookback - 1 = 99
  if len(returns) < 20: return "HOLD", ...
  ```
  Misleading — gives false comfort if `lookback` lowered below 21.

- `portfolio/signals/structure.py:73-83` — `_highlow_breakout` divides by `period_low` with `np.inf` fallback
  ```python
  pct_from_low = (current_close - period_low) / period_low if period_low != 0 else np.inf
  ```
  `np.inf` used in conditional only (not stored). Behavior correct on data quality issues but value-inspection downstream could surprise.

- `portfolio/signals/forecast.py:847-848` — `_cached` wraps `_run_chronos`; circuit breaker bypass within 5min TTL
  ```python
  chronos_key = f"chronos_forecast_{ticker}"
  chronos = _cached(chronos_key, _FORECAST_TTL, _run_chronos, close_prices, (1, 24), ticker)
  ```
  After successful call, breaker trips → cached result returned for up to 5min. `reset_circuit_breakers()` doesn't re-run until cache TTL expires. Stale result not labeled.

## Did NOT find

1. Wrong applicable_assets on active signals — `futures_flow`, `crypto_macro`, `cot_positioning`, `metals_cross_asset`, `dxy_cross_asset`, `credit_spread` all guard correctly via explicit ticker checks.
2. Confidence > 1.0 leaking — `majority_vote` caps; `_accuracy_weighted_vote` clamps with `_MAX_CONFIDENCE`.
3. Sign errors in momentum/trend — RSI divergence, stochastic, CCI, Williams %R, PPO, ADX direction logic correct.
4. Lookahead bias — `_donchian_breakout` uses `high.iloc[-(period+1):-1]` (excludes current bar).
5. NaN→0 directional vote — active modules guard NaN before voting.
6. Bare `except:` — all `except Exception` (not bare); `news_event._persist_headlines` debug-logs only.
7. `crypto_cross_asset.py` `"signal"` vs `"action"` — confirmed not registered, dead code.
