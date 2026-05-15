# Adversarial Review — 7 data-external (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/microstructure_state.py:208` — `persist_state` iterates `_snapshot_buffers` without holding `_buffer_lock`
  ```python
  def persist_state() -> None:
      state = {}
      for ticker in _snapshot_buffers:                # <-- unsynchronized iteration
          ms = get_microstructure_state(ticker)
          ...
  ```
  `metals_loop.py` fast-tick (10 s) calls `accumulate_snapshot()` (which calls `_ensure_buffer()` and adds new dict keys) concurrently with the 60 s `persist_state()` on the main cycle. Adding a new ticker key during iteration raises `RuntimeError: dictionary changed size during iteration`, which propagates out of `persist_state()` and (depending on the caller's try/except) silently disables persistence — the orderbook_flow signal then loses cross-process state visibility, and the `>2 min stale` guard in `load_persisted_state` at line 227 forces the signal into HOLD. Take a snapshot of the keys under the lock first: `with _buffer_lock: tickers = list(_snapshot_buffers)`.

- `portfolio/data_collector.py:280-294` — Race window between dispatcher market-state check and the inner `_fetch_klines` call
  ```python
  if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
      with _yfinance_lock:
          df = _fetch_klines(source, interval, limit)
  else:
      df = _fetch_klines(source, interval, limit)        # may still hit yfinance via the dispatcher branch at line 262
  ```
  `_fetch_klines` (lines 253-268) re-checks `_ss._current_market_state` and routes to `yfinance_klines()` whenever the state is closed/weekend/holiday. If the outer check at line 290 sees "open" but the inner check at line 262 sees "closed" (the state can flip across the 60 s cycle boundary — see `market_timing.update_market_state` semantics), the yfinance call runs **without** the shared lock that exists specifically because `yfinance.download` is not thread-safe. The historical fix comment at the top of `data_collector.py` (lines 274-277) acknowledges yfinance is not thread-safe; this race re-introduces the original bug. Move the lock acquisition inside `_fetch_klines` itself, keyed on the actual source taken.

- `portfolio/crypto_precompute.py:159-167, 179-196, 210-216, 225-232` — `requests.get` bypasses every rate-limiter, circuit-breaker, and retry layer the rest of the codebase enforces
  ```python
  r = requests.get("https://api.binance.com/api/v3/ticker/24hr", params={"symbol": sym}, timeout=_REQUEST_TIMEOUT)
  ...
  r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", ...)
  ...
  hist = yf.Ticker(sym).history(period="5d", interval="1d")    # no yfinance_lock either
  ```
  Six Binance calls + three yfinance calls per 4 h interval is small, but they execute without `_binance_limiter`, without `binance_spot_cb` / `binance_fapi_cb`, and without `yfinance_lock`. Concurrent with `portfolio/main.py`'s 60 s loop and the dashboard background threads, these calls can: (a) trip Binance's IP-level 429 ban (-1003) for all consumers in the process, (b) corrupt yfinance's shared state when called concurrent with `data_collector.yfinance_klines`. Route everything through `portfolio.shared_state._binance_limiter` + `portfolio.http_retry.fetch_json` and through `yfinance_lock` (the lock at `portfolio.shared_state.yfinance_lock` is module-level and already imported by `data_collector`).

- `portfolio/mstr_precompute.py:140-200` — Same `requests.get` / unlocked `yf.Ticker.history` pattern AND hardcoded fragile MSTR balance-sheet constants drive a live NAV-premium calculation
  ```python
  _DEFAULT_BTC_HOLDINGS = 471_107        # 2026-04 estimate
  _DEFAULT_DEBT_USD = 8_500_000_000
  _DEFAULT_SHARES_OUTSTANDING = 287_000_000
  ```
  `_compute_nav_premium` (line 223) multiplies `btc_holdings * btc_price - debt_usd` then divides into `mstr_price * shares_outstanding`. If MSTR issues stock or buys more BTC between manual refreshes, the premium math drifts. With `data/crypto_data.py:184` independently hardcoding `MSTR_BTC_HOLDINGS = 499_096`, the two systems are already inconsistent (471,107 vs 499,096 — a ~6 % BTC-holdings divergence). Any L2 prompt that reads `mstr_deep_context.json` for NAV anchor gets stale data with no warning. At minimum, log a WARNING with the date and source whenever the defaults are used, and reject the precompute output when `_fetched_at` of holdings > N days.

- `data/crypto_data.py:73-85` — `get_fear_greed` IndexError on maintenance windows is swallowed into `None`, and on the silent-failure path the call gets cached as a stale prior value via the wider system
  ```python
  data = r.json().get("data", [{}])[0]
  ```
  When `alternative.me` is in maintenance it returns `{"data": []}`. `[].__getitem__(0)` raises `IndexError`, which is caught by the bare `except Exception as e` at line 83 and logged WARNING. `get_fear_greed` then returns `None`. This is the same maintenance-window pattern that `portfolio/fear_greed.py:106` defends against explicitly (with a "P1-13 / 04-29 DE-P1-2" historical comment) — but `data/crypto_data.py:73` was apparently never patched. The cached value from the previous successful fetch (FEAR_GREED_TTL = 300 s) is returned as long as it's fresh, so for ≤5 min it's masked; longer outages silently force F&G = None into the metals loop, which (per the system's contrarian-gate signal) flips behavior. Mirror the explicit empty-list guard from `portfolio/fear_greed.py`.

## P1 — high-confidence bugs (should fix)

- `portfolio/fx_rates.py:46-53` — Out-of-bounds FX rate silently returns `None`, callers get the hardcoded 10.50 fallback
  ```python
  if not (FX_RATE_MIN <= rate <= FX_RATE_MAX):
      logger.error("FX rate %.4f SEK/USD outside sane bounds (7-15) — ignoring", rate)
  else:
      with _fx_lock:
          _fx_cache["rate"] = rate
          _fx_cache["time"] = now
      return rate
  ```
  When the rate is rejected as out-of-bounds, the `else` branch is skipped, the function falls through to the cached/fallback path, and may end up returning `FX_RATE_FALLBACK = 10.50` (line 71). The Telegram alert fires only on the stale-cache path (line 64) or full fallback (line 70), not on the "API returned a number we don't trust" path. Track that distinct failure mode explicitly so portfolio_mgr / monte_carlo aren't using 10.50 SEK for valuations while the real rate has moved.

- `portfolio/metals_precompute.py:404-407, 458` and `portfolio/oil_precompute.py:503-507, 574-577` — Direct `requests.get` on CFTC + FRED bypasses rate limiting AND interpolates `commodity_name` without escaping
  ```python
  url = (
      "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
      f"?$where=commodity_name='{commodity_name}'"
      ...
  )
  resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
  ```
  `commodity_name` is hardcoded ("GOLD", "SILVER", "CRUDE OIL") today, so the SoQL injection surface is theoretical — but the `f"...='{commodity_name}'"` template makes this a one-config-change-away from injection if anyone wires `commodity_name` to a config field. Use `params={"$where": f"commodity_name='{commodity_name}'"}` so requests handles URL-encoding, and ideally pass the value via parametrization. Also no retry/circuit-breaker — a CFTC outage burns the cycle's `_safe_fetch` budget silently.

- `portfolio/fear_greed.py:134-154` — `yfinance.Ticker("^VIX").history(period="5d")` on a US holiday or extended weekend can return an empty DataFrame; the `if h.empty: return None` is fine, but the prior-day `prev = hist.iloc[-2]` calc in `fetch_vix` (`data_collector.py:181`) only checks `len(hist) > 1`, and **does not check if today's bar is real vs synthesized** — yfinance ^VIX prepost data can have intraday rows that haven't received an official close, producing artificially-small daily change %. Document the assumption or call `auto_adjust=True` and trim to fully-closed bars.

- `portfolio/data_collector.py:311-312` — Errors are stuffed back into the results dict with `(label, {"error": str(e)})` and treated as `result is not None`, so the caller treats failures and successes uniformly
  ```python
  except Exception as e:
      return (label, {"error": str(e)})
  ```
  The outer `collect_timeframes` only filters `if result is not None`. Downstream consumers in `signal_engine` then have to recognize the `{"error": ...}` schema vs the success schema; if they don't, they'll iterate over the error dict and silently treat absence of `"indicators"` key as "no signal" — exactly the silent-fail pattern this codebase has been bitten by before. Either propagate via the future's exception or return a structured failure tag the engine explicitly recognizes (e.g. `{"action": None, "error": ...}` consistent with success shape).

- `portfolio/earnings_calendar.py:48-52` — Comment explicitly admits AV daily-budget tracking is bypassed
  ```python
  # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
  # because there is no public increment function exported from that module.
  # Known limitation — earnings fetches consume 1 AV call each but are not
  # reflected in the budget tracker.  Each ticker only fetches once per 24h.
  ```
  Free tier is 25/day. Even with 24 h cache, restart loops or cache misses on day rollover can burn through several hidden calls; `should_batch_refresh` then thinks budget is fine while `_fetch_earnings_alpha_vantage` has already used it. Expose an increment helper in `alpha_vantage.py` or share `_daily_budget_used` via a module-level setter and call it from `_fetch_earnings_alpha_vantage`.

- `portfolio/sentiment.py:288-297` — Subprocess fallback path has `timeout=120` and `capture_output=True` but no explicit kill on timeout; on Windows, subprocess.run with timeout raises `TimeoutExpired` but the child may still be running with a leaked file handle if Python's cleanup races
  ```python
  proc = subprocess.run(
      [MODELS_PYTHON, script],
      input=json.dumps(texts),
      capture_output=True,
      text=True,
      timeout=120,
  )
  ```
  In itself, `subprocess.run` will kill the child on TimeoutExpired in modern Python, but the surrounding `_run_model` callers wrap this in their own try/except and silently return neutral. Combined with the 33 active signals × 5 tickers, a hung BERT subprocess can pin up to 120 s × N workers per cycle. The in-process path (now primary) avoids this, but the fallback remains a potential cycle-time stall. Add a hard `subprocess.Popen` + explicit `terminate()`/`kill()` chain and a per-tier subprocess budget.

- `portfolio/news_keywords.py:80-83` — Regex compilation uses `\b` word boundaries with `re.escape`-ed keywords, but the keyword list contains multi-word phrases like `"sec investigation"` and `"trade war"`
  ```python
  _KEYWORD_PATTERNS = [
      (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
      for kw, weight in sorted(ALL_KEYWORDS.items(), key=lambda x: -len(x[0]))
  ]
  ```
  `\b" + re.escape("sec investigation") + r"\b` becomes `\bsec\ investigation\b` (with escaped space). The escaped space matches a literal space, which is fine, but the word-boundary `\b` between `investigation` and end-of-pattern still works only because the next character in real input is whitespace or punctuation. The bigger concern: any keyword containing a non-word character (none today, but `"AT&T"` or `"S&P 500"` if anyone adds them) would silently never match because `\b` anchors require a `\w` on one side. Add a unit test asserting every entry in `ALL_KEYWORDS` actually matches a sample sentence containing it.

- `portfolio/microstructure_state.py:227` — Stale-state threshold of 2 minutes can be incorrectly triggered after a system clock skew or NTP step
  ```python
  age_ms = int(time.time() * 1000) - entry.get("ts", 0)
  if age_ms > 120_000:
      return None
  ```
  Cross-process clock skew (when the signal runs in the main loop process and the persistor runs in the metals_loop process) could mark fresh data stale. Use `time.monotonic` only within a single process and signed-difference handling. (Same pattern reappears in many `_load_*` calls under this subsystem.)

- `portfolio/sentiment_shadow_backfill.py:211-213` — `datetime.fromisoformat(ts)` then `if entry_time.tzinfo is None: entry_time = entry_time.replace(tzinfo=UTC)` assumes the legacy log was written in UTC
  Historical entries in `sentiment_ab_log.jsonl` predating the UTC migration would silently shift, producing incorrect target_time and pct_change values for the outcome backfill. Add a min-ts cutoff or, ideally, treat naive timestamps as the writer's local TZ (CET) and convert.

## P2 — concerns / smells (worth addressing)

- `portfolio/session_calendar.py:155-157` — `now.replace(hour=open_utc, minute=30, second=0)` doesn't reset `microsecond`
  Existing `now.microsecond` is preserved, meaning the comparison `session_open <= now < session_end` may miss the exact open-bell timestamp by sub-second amounts. Cosmetic but worth `microsecond=0` for parity with `session_end`.

- `portfolio/seasonality.py:62-66` — `mean_return` / `mean_volatility` cast to float even when `count = 0`
  ```python
  profile[str(hour)] = {
      "mean_return": float(row["mean_return"]),
      ...
      "count": int(row["count"]),
  }
  ```
  When count is small (1-2 samples for an hour due to data gaps), the mean_return is computed from too few samples and treated equally to a 20-sample mean. Caller `detrend_return` then subtracts a noisy mean. Skip hours with count < threshold or weight by count.

- `portfolio/social_sentiment.py:32, 65` — `requests.get` directly on Reddit's public JSON endpoint with no rate-limit handling, no retry, no shared User-Agent rotation
  Reddit applies progressive bans for unauthenticated requests > N/min. Subreddit fetches are wrapped in per-sub try/except but use a single static UA `"finance-analyzer/1.0"` that may already be 403'd. Doesn't use `http_retry.fetch_json` either. Cap or move behind the shared retry layer.

- `portfolio/macro_context.py:143` — `synth = 58.0 * (eurusd ** -0.576)` "value" is documented as meaningless but is returned in the same dict shape as the real path
  Downstream consumers that read `result["value"]` get a number that looks legitimate (~58) but isn't a real DXY level. A consumer that ever skips reading `result["source"]` (or that gets called from a new path not aware of the comment) will produce wrong DXY thresholding. Either set `value=None` for the synth path, or add a `value_is_synthetic: True` flag the consumers can assert on.

- `portfolio/funding_rate.py:23-39` — Defensive guards are good, but the rate thresholds (0.0003 SELL, -0.0001 BUY) are hardcoded and asymmetric
  Normal funding is ~0.0001 (0.01 %). The asymmetry (+30 bps vs -10 bps) is documented but undocumented why the negative threshold is 3× tighter. Move to config or annotate the bias.

- `portfolio/onchain_data.py:178, 187` — `latest = data[0] if isinstance(data[0], dict) else data[-1]`
  If `data[0]` isn't a dict but `data[-1]` also isn't, you get a non-dict at `latest` and the next line's `latest.get(...)` raises AttributeError. The outer `except Exception` swallows this. Tighten to `next((d for d in data if isinstance(d, dict)), None)`.

- `portfolio/alpha_vantage.py:266-269, 273-276, 291-293` — On any failure path the loop runs `_cb.record_failure()` and then `if not _cb.allow_request(): break`. After the break the function returns `success_count` (still 0) and silently exits; no Telegram alert ever fires from this module. Combined with the documented 3-week silent Layer 2 auth outage history, the absence of an upstream surfacing of "fundamentals haven't refreshed in N days" is concerning.

- `portfolio/bert_sentiment.py:286-296` — `BERT_SENTIMENT_USE_GPU` env-var opt-in to GPU is undocumented in CLAUDE.md
  Anyone enabling this will collide with llama-server's VRAM budget per the comment at line 269-281, but they'd need to read the source code to know that.

## Did NOT find

1. Silent failures: actively looked, plenty exist (data/crypto_data.py:83, sentiment.py shadow-paths, alpha_vantage.py budget exhaustion), surfaced above.
2. Race conditions: surfaced — microstructure_state.persist_state iteration race, data_collector market-state race.
3. Money-losing bugs: no direct PnL/sign errors in this subsystem (no order placement here); the closest is the FX-rate silent-fallback (P1) which affects portfolio valuations.
4. State corruption: JSONL appends go through atomic_append_jsonl; ratio_history dedup window is correct.
5. Logic errors that pass tests: news_keywords \b-anchor edge case (P1).
6. Resource leaks: subprocess timeout in sentiment.py fallback path is a potential pin-not-leak (P1).
7. Time/timezone bugs: session_calendar.py microsecond cosmetic (P2); sentiment_shadow_backfill TZ assumption (P1).
8. API misuse: no Binance `10m` usage found in this subsystem (good). The bigger misuse is the bypass of rate limiters / circuit breakers in crypto_precompute / mstr_precompute / metals_precompute / oil_precompute (P0/P1).
9. Trust boundary: CFTC SoQL `$where` interpolation (P1) is the only finding; news_keywords regex uses `re.escape` correctly.
10. Partial-state assumptions: data/crypto_data.py `data["data"][0]` is the canonical example (P0).
