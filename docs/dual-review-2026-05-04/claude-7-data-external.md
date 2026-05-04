# Claude Review — data-external

## P0 (money-losing or data-corrupting)

- `portfolio/microstructure_state.py:205-213` — `persist_state()` calls `get_microstructure_state()` which appends to OFI history every persist tick → double-counts OFI
  ```python
  def persist_state() -> None:
      state = {}
      for ticker in _snapshot_buffers:
          ms = get_microstructure_state(ticker)   # side effect: record_ofi() inside
          ms["ts"] = int(time.time() * 1000)
          state[ticker] = ms
  ```
  `get_microstructure_state(ticker)` calls `record_ofi(ticker, ofi)` at line 185, appending to `_ofi_history`. `persist_state()` is called from `metals_loop.py` every 5 fast-tick cycles (~2.5–5 min). Each persist call appends an OFI value without new snapshot data, inflating the buffer and corrupting the z-score distribution. Fix: separate read from record. Confidence 90.

- `data/crypto_data.py:184-185` — hardcoded MSTR BTC holdings (499,096) diverge from `mstr_precompute.py` (471,107) by 6%, and shares outstanding diverge by 25%
  ```python
  MSTR_BTC_HOLDINGS = 499_096  # as of early 2026
  MSTR_SHARES_OUTSTANDING = 229_000_000  # approximate
  ```
  vs. `portfolio/mstr_precompute.py:35,37`: `_DEFAULT_BTC_HOLDINGS = 471_107`, `_DEFAULT_SHARES_OUTSTANDING = 287_000_000`. Both claim "early 2026". `data/crypto_data.py` is imported by `crypto_precompute.py:141`, so it is live in the precompute path. NAV premium math will be off by 20%+, potentially flipping buy/sell. Confidence 85.

## P1 (high-confidence bugs)

- `portfolio/crypto_precompute.py:185` — Binance funding rate fallback to `0.0` masks missing field
  ```python
  out[key_funding] = float(r.json().get("lastFundingRate", 0))
  ```
  A `0` default conflates missing field, partial response, and actual zero funding. Real BTC funding is ~0.0001. Stored 0.0 is interpreted by downstream as "shorts overleveraged" or "neutral" depending on sign logic. `portfolio/funding_rate.py:31-33` correctly returns `None` on missing fields. Fix: same pattern. Confidence 88.

- `portfolio/earnings_calendar.py:48-53` — Alpha Vantage EARNINGS calls bypass `_daily_budget_used` counter
  ```python
  # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
  # because there is no public increment function exported from that module.
  ```
  Self-documented bug. `len(STOCK_SYMBOLS)` AV calls per day not visible to `refresh_fundamentals_batch()`, which can exhaust the 25/day quota silently and return early thinking budget remains. Fix: export `_increment_budget`. Confidence 85.

- `portfolio/econ_dates.py:155, 180, 224, 273` — all CPI/NFP/GDP/FOMC events pinned to 14:00 UTC; CPI/NFP release at 08:30 ET (12:30/13:30 UTC depending on DST)
  ```python
  evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC)
  ```
  `is_macro_window()`, `events_within_hours()`, `recent_high_impact_events()` all use this. Pre-CPI risk-off suppression doesn't fire until after the print; post-CPI suppression starts late. FOMC (14:00 ET = 18:00/19:00 UTC) misplaced too. Confidence 82.

- `portfolio/metals_precompute.py:407-409, 458-460` — COT fetch uses raw `requests.get()` with no retry
  ```python
  resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
  resp.raise_for_status()
  ```
  CFTC SOCRATA endpoint has frequent 5xx and rate limits. Rest of the codebase routes through `fetch_with_retry`. Single transient failure marks COT failed for 7 days (`_REFRESH_INTERVALS["cot"] = 7*24*3600`). Confidence 83.

- `portfolio/crypto_macro_data.py:208-218` — reads `agent_summary_compact.json` for live BTC/gold prices, violating "always pull live" rule
  ```python
  summary = load_json(DATA_DIR / "agent_summary_compact.json")
  btc_price = signals.get("BTC-USD", {}).get("price_usd")
  gold_price = signals.get("XAU-USD", {}).get("price_usd")
  ```
  File can be hours stale during outages or `_tool_cache` TTL window. Gold/BTC ratio feeds Telegram reports. Stale ratio in volatile market silently misclassifies rotation signal. Confidence 80.

- `portfolio/metals_precompute.py:149-256` — `_fetch_market_data` silently returns `None` for un-refreshed sources, never loads cached values
  When `_should_refresh()` returns False, `result[key]` stays None even if valid data exists on disk. Callers (`_build_silver_context`) check `if market.get("slv")` and skip ETF overlay if None. Within the 4h cache window, ETF context vanishes — looks like fetch failure. Should load previously-written context and update only refreshed fields. Confidence 80.

## P2 (concerns / smells)

- `portfolio/sentiment.py:859` — `ab_key = f"{short}:{datetime.now(UTC).isoformat()}"` non-unique under microsecond-level concurrency
  Two calls within the same isoformat output collide → second overwrites first in `_pending_ab_entries`. Probability low (different tickers usually) but should include a uuid or counter.

- `portfolio/macro_context.py:287-292` — `_fred_10y_fallback()` reads config.json with raw `open()`
  Bypasses `file_utils.load_json()` and the module's own cache. Project rule says atomic I/O only.

- `portfolio/session_calendar.py:82-89` — `_make_session_end()` can produce a past datetime when called after session close
  ```python
  end = now.replace(hour=utc_hour, minute=cet_minute, second=0, microsecond=0)
  ```
  After 22:30 CET the `end` is set to today's 19:55 UTC. Comparison `session_open <= now < session_end` correctly returns False, but `SessionInfo.session_end` is in the past — confuses callers that schedule timers. Fix: add `timedelta(days=1)` when `end < now`.

## Did NOT find

1. Binance `10m` interval usage — no occurrences across all reviewed files.
2. Non-atomic JSON cache writes — all use `atomic_write_json`/`atomic_append_jsonl`.
3. Missing 429 handling for fear/greed — `fetch_json` via `http_retry` handles it.
4. NaN propagation from yfinance — `_fetch_dxy_intraday` checks `math.isnan`; indicators handle insufficient data.
5. Sentiment leakage (future news for past sentiment) — `sentiment_shadow_backfill.py` correctly uses `entry_time + timedelta(hours=hours)`.
6. Pagination bugs — Binance/Alpaca/FRED fetches use `limit=` or correctly advance `start_time`.
7. HTTP retry without backoff — production fetches use `fetch_with_retry`/`fetch_json`; only `metals_precompute.py` raw `requests.get` (P1 above).
