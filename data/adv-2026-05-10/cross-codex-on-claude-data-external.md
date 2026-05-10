OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-data-external
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e127e-9fa8-7971-97b8-18c60e4784e7
--------
user
You are an adversarial code reviewer cross-critiquing another reviewer's findings.

CONTEXT
- Codebase: Q:\finance-analyzer (autonomous trading system, Python).
- Subsystem: data-external
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
# Adversarial Review: data-external subsystem (2026-05-08)

[P1] portfolio/fx_rates.py:42-48
**FX bounds-check rejects out-of-range rate but returns it anyway.**
Problem: When Frankfurter returns an out-of-range rate, the code logs ERROR but
returns the rate, bypassing cache. Next cycle then triggers stale fallback. Bounds
rejection should refuse the value, not pass it through.
Fix: On out-of-range, raise to the caller and let the cache fallback handle it
explicitly; do not return invalid rate.

[P1] portfolio/sentiment.py:233-237
**Headline dedupe only on title, ignores source/date.**
Problem: Same headline from Reuters + AP + Bloomberg counts as 1, suppressing
`dissemination_mult`. Underestimates actual news intensity.
Fix: Dedupe on `(title, published_date)` tuple, keep source list for dissemination.

[P1] portfolio/data_collector.py:96, 157
**Kline timestamps timezone-naive on both Binance and Alpaca paths.**
Problem: Cross-source merges risk misalignment (Binance comes UTC, Alpaca local). A
silent off-by-an-hour at DST boundary corrupts indicator calculations.
Fix: Add `utc=True` to both `pd.to_datetime()` calls; assert tz-aware downstream.

[P1] portfolio/shared_state.py:95-103
**None results not cached; stale fallback returned asymmetrically.**
Problem: API returning `None` skips cache update; next call returns stale data without
re-fetching. Flaky APIs cause oscillation between fresh-fail and stale-return that
distorts signal stability.
Fix: Cache `None` with short TTL (e.g., 30s) to break the oscillation; or expose retry
metric so callers can decide.

[P1] portfolio/earnings_calendar.py:49-52
**Alpha Vantage earnings calls bypass the daily budget counter.**
Problem: Each ticker fetch hits the Alpha Vantage 25/day quota but the budget tracker
isn't incremented. 100+ tickers of earnings refresh = silent 4x quota overrun.
Fix: Wrap every AV request through the same `_daily_budget_used` accounting.

[P1] portfolio/sentiment.py:804
**Headline filter fallback ignores per-source noise profiles.**
Problem: CryptoCompare press-wire (high noise) and NewsAPI metals (pristine) share the
same fallback threshold. Crypto fires noise-driven sentiment more often.
Fix: Per-source threshold; tune crypto floor higher than metals.

[P1] portfolio/alpha_vantage.py:281
**Budget reset at midnight not under lock.**
Problem: `_budget_reset_date` checked outside lock; race between reset and increment
at midnight boundary can lose 1–5 calls of accounting.
Fix: Move the reset check inside the same critical section as the increment.

[P1] portfolio/sentiment.py:876-888
**FinGPT enqueue failure silently drops A/B entry.**
Problem: Exception only logged at DEBUG; downstream `shadow[]` ends up missing FinGPT
votes. Accuracy tracking sees fewer A/B comparisons than expected.
Fix: Promote to WARNING; emit a metric on dropped FinGPT entries.

[P1] portfolio/data_collector.py:296-299
**Raw DataFrame discarded when indicator computation returns None.**
Problem: Whole timeframe entry dropped. Downstream modules needing raw OHLCV (orderbook
context, volume profile) get nothing for that timeframe.
Fix: Keep `_df` even when indicators fail; emit warning so consumers know indicators
are missing but raw data is available.

[P1] portfolio/futures_data.py:22-24
**Open Interest cache TTL 300s too long for momentum signals.**
Problem: Binance OI updates every second; a 5-min-old snapshot can flip the sign of a
3h trade signal at the boundary.
Fix: Drop TTL to 30–60s; or version the cache with last-update-tick and let signal
caller override.

[P1] portfolio/onchain_data.py:112, 260
**Malformed timestamp coerces to 0.0 → cache miss every restart.**
Problem: Persisted cache stores ISO string ts in some path, float in another; parse
mismatch returns 0.0 (ancient), forcing BGeometrics 15/day budget burn on every
restart.
Fix: Pick one format (float epoch), validate on read, recover gracefully.

[P2] portfolio/macro_context.py:141-144
**EURUSD synth fallback for DXY returns meaningless `value`.**
Problem: Comment admits value is arbitrary. Downstream code that ever reads it as a
real DXY level will silently produce wrong z-scores or thresholds.
Fix: Set `value=None` and add `value_is_synth=True` flag so consumers must opt in.

[P2] portfolio/sentiment.py:757-758
**Trading-Hero-LLM has known permabull bias on crypto, no de-rating.**
Problem: Despite documented bias, no correction factor is applied to its votes; per-
asset accuracy gate may already mask this, but the input is biased.
Fix: Apply per-source bias correction during sentiment aggregation, or rotate to a
calibrated model for crypto.

## Summary

11 P1 + 2 P2. Themes: cache-staleness lying about freshness, rate-limit accounting
gaps, timezone-naive datetimes, and per-source noise profiles being applied
indiscriminately. Several issues are silent failures that pass tests but degrade
signal quality slowly.
=== END ===

Use git/grep/read tools as needed. Be ruthless and cite code. Do not soften.

codex
I’m verifying each cited finding against the working tree, starting by pulling the exact source ranges Claude referenced and then tracing the surrounding control flow where the claim depends on behavior outside those lines.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files portfolio' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git status --short' in Q:\finance-analyzer\.worktrees\adv-data-external
2026-05-10T15:26:05.000301Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-data-external'
'Q:/finance-analyzer/.worktrees/adv-data-external/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-data-external

 exited 1 in 745ms:
fatal: detected dubious ownership in repository at 'Q:/finance-analyzer/.worktrees/adv-data-external'
'Q:/finance-analyzer/.worktrees/adv-data-external/.git' is owned by:
	HERC2/Herc2 (S-1-5-21-1167218111-2780401794-1724439369-1001)
but the current user is:
	HERC2/CodexSandboxOffline (S-1-5-21-1167218111-2780401794-1724439369-1004)
To add an exception for this directory, call:

	git config --global --add safe.directory Q:/finance-analyzer/.worktrees/adv-data-external

 succeeded in 761ms:
portfolio\social_sentiment.py
portfolio\sentiment.py
portfolio\onchain_data.py
portfolio\news_keywords.py
portfolio\market_health.py
portfolio\macro_context.py
portfolio\fx_rates.py
portfolio\futures_data.py
portfolio\fear_greed.py
portfolio\earnings_calendar.py
portfolio\data_collector.py
portfolio\crypto_macro_data.py
portfolio\bert_sentiment.py
portfolio\alpha_vantage.py

codex
One cited file, `portfolio/shared_state.py`, didn’t show up in the subsystem file list. I’m checking whether Claude cited a moved/deleted path versus a real file before evaluating that claim.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/fx_rates.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/data_collector.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/earnings_calendar.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/futures_data.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/macro_context.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/alpha_vantage.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/onchain_data.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "" portfolio/sentiment.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files | rg \"shared_state\\.py"'$|shared_state"' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 839ms:
1:"""Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""
2:
3:import logging
4:import time
5:from concurrent.futures import ThreadPoolExecutor, as_completed
6:from datetime import UTC, datetime
7:
8:import pandas as pd
9:
10:import portfolio.shared_state as _ss
11:from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
12:from portfolio.circuit_breaker import CircuitBreaker
13:from portfolio.http_retry import fetch_with_retry
14:from portfolio.indicators import compute_indicators, technical_signal
15:
16:logger = logging.getLogger("portfolio.data_collector")
17:
18:# --- Circuit breakers for each data source ---
19:
20:binance_spot_cb = CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)
21:binance_fapi_cb = CircuitBreaker("binance_fapi", failure_threshold=5, recovery_timeout=60)
22:alpaca_cb = CircuitBreaker("alpaca", failure_threshold=5, recovery_timeout=60)
23:
24:# BUG-179: Timeout for parallel timeframe fetches (seconds)
25:_TF_POOL_TIMEOUT = 60
26:ALPACA_INTERVAL_MAP = {
27:    "15m": ("15Min", 5),
28:    "1h": ("1Hour", 10),
29:    "1d": ("1Day", 365),
30:    "1w": ("1Week", 730),
31:    "1M": ("1Month", 1825),
32:}
33:
34:# yfinance interval mapping: our interval → (yf_interval, yf_period)
35:_YF_INTERVAL_MAP = {
36:    "15m": ("15m", "5d"),       # yfinance max for intraday <=60d
37:    "1h": ("1h", "30d"),
38:    "1d": ("1d", "365d"),
39:    "1w": ("1wk", "730d"),
40:    "1M": ("1mo", "1825d"),
41:}
42:
43:# Multi-timeframe analysis — (label, binance_interval, num_candles, cache_ttl_seconds)
44:TIMEFRAMES = [
45:    ("Now", "15m", 100, 0),  # ~25h data, refresh every cycle
46:    ("12h", "1h", 100, 300),  # ~4d data, cache 5min
47:    ("2d", "4h", 100, 900),  # ~17d data, cache 15min
48:    ("7d", "1d", 100, 3600),  # ~100d data, cache 1hr
49:    ("1mo", "3d", 100, 14400),  # ~300d data, cache 4hr
50:    ("3mo", "1w", 100, 43200),  # ~2yr data, cache 12hr
51:    ("6mo", "1M", 48, 86400),  # ~4yr data, cache 24hr
52:]
53:
54:STOCK_TIMEFRAMES = [
55:    ("Now", "15m", 100, 0),
56:    ("12h", "1h", 100, 300),
57:    ("2d", "1h", 48, 900),
58:    ("7d", "1d", 30, 3600),
59:    ("1mo", "1d", 100, 3600),
60:    ("3mo", "1w", 100, 43200),
61:    ("6mo", "1M", 48, 86400),
62:]
63:
64:
65:# --- Binance API ---
66:
67:_BINANCE_KLINE_COLS = [
68:    "open_time", "open", "high", "low", "close", "volume",
69:    "close_time", "quote_vol", "trades", "taker_buy_vol",
70:    "taker_buy_quote_vol", "ignore",
71:]
72:
73:
74:def _binance_fetch(base_url, cb, label, symbol, interval="5m", limit=100):
75:    """Shared Binance kline fetcher for spot and FAPI endpoints."""
76:    if not cb.allow_request():
77:        logger.warning("Binance %s circuit OPEN — skipping %s", label, symbol)
78:        raise ConnectionError(f"Binance {label} circuit open for {symbol}")
79:    try:
80:        r = fetch_with_retry(
81:            f"{base_url}/klines",
82:            params={"symbol": symbol, "interval": interval, "limit": limit},
83:            timeout=10,
84:        )
85:        if r is None:
86:            raise ConnectionError(f"Binance {label} klines request failed for {symbol}")
87:        r.raise_for_status()
88:        data = r.json()
89:        if not data:
90:            raise ConnectionError(
91:                f"Binance {label} returned empty data for {symbol} {interval}"
92:            )
93:        df = pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)
94:        for col in ["open", "high", "low", "close", "volume"]:
95:            df[col] = df[col].astype(float)
96:        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
97:        cb.record_success()
98:        return df
99:    except Exception:
100:        cb.record_failure()
101:        raise
102:
103:
104:def binance_klines(symbol, interval="5m", limit=100):
105:    return _binance_fetch(BINANCE_BASE, binance_spot_cb, "spot", symbol, interval, limit)
106:
107:
108:def binance_fapi_klines(symbol, interval="5m", limit=100):
109:    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
110:    return _binance_fetch(BINANCE_FAPI_BASE, binance_fapi_cb, "FAPI", symbol, interval, limit)
111:
112:
113:# --- Alpaca API ---
114:
115:
116:def alpaca_klines(ticker, interval="1d", limit=100):
117:    if interval not in ALPACA_INTERVAL_MAP:
118:        raise ValueError(f"Unsupported Alpaca interval: {interval}")
119:    if not alpaca_cb.allow_request():
120:        logger.warning("Alpaca circuit OPEN — skipping %s", ticker)
121:        raise ConnectionError(f"Alpaca circuit open for {ticker}")
122:    try:
123:        alpaca_tf, lookback_days = ALPACA_INTERVAL_MAP[interval]
124:        end = datetime.now(UTC)
125:        start = end - pd.Timedelta(days=lookback_days)
126:        r = fetch_with_retry(
127:            f"{ALPACA_BASE}/stocks/{ticker}/bars",
128:            headers=get_alpaca_headers(),
129:            params={
130:                "timeframe": alpaca_tf,
131:                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
132:                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
133:                "feed": "iex",
134:                "adjustment": "split",
135:            },
136:            timeout=10,
137:        )
138:        if r is None:
139:            raise ConnectionError(f"Alpaca request failed for {ticker}")
140:        r.raise_for_status()
141:        bars = r.json().get("bars") or []
142:        if not bars:
143:            raise ValueError(f"No Alpaca data for {ticker} interval={interval}")
144:        df = pd.DataFrame(bars)
145:        df = df.rename(
146:            columns={
147:                "o": "open",
148:                "h": "high",
149:                "l": "low",
150:                "c": "close",
151:                "v": "volume",
152:                "t": "time",
153:            }
154:        )
155:        for col in ["open", "high", "low", "close", "volume"]:
156:            df[col] = df[col].astype(float)
157:        df["time"] = pd.to_datetime(df["time"])
158:        alpaca_cb.record_success()
159:        return df.tail(limit)
160:    except Exception:
161:        alpaca_cb.record_failure()
162:        raise
163:
164:
165:# --- yfinance API ---
166:
167:
168:def fetch_vix():
169:    """Fetch current VIX level via yfinance. Returns dict or None."""
170:    try:
171:        import yfinance as yf
172:
173:        vix = yf.Ticker("^VIX")
174:        hist = vix.history(period="5d")
175:        if hist is None or hist.empty:
176:            return None
177:        # Flatten MultiIndex columns if present
178:        if isinstance(hist.columns, pd.MultiIndex):
179:            hist.columns = hist.columns.get_level_values(0)
180:        last = hist.iloc[-1]
181:        prev = hist.iloc[-2] if len(hist) > 1 else last
182:        current = float(last["Close"])
183:        prev_close = float(prev["Close"])
184:        change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0
185:
186:        # VIX regime classification
187:        if current >= 30:
188:            regime_hint = "high-vol"
189:        elif current >= 20:
190:            regime_hint = "elevated"
191:        elif current >= 15:
192:            regime_hint = "normal"
193:        else:
194:            regime_hint = "complacent"
195:
196:        return {
197:            "value": round(current, 2),
198:            "prev_close": round(prev_close, 2),
199:            "change_pct": round(change_pct, 2),
200:            "regime_hint": regime_hint,
201:        }
202:    except Exception as e:
203:        logger.warning("VIX fetch failed: %s", e)
204:        return None
205:
206:
207:def yfinance_klines(ticker, interval="1d", limit=100):
208:    """Fetch candles via yfinance with extended-hours data (prepost=True).
209:
210:    Returns a DataFrame matching alpaca_klines() format:
211:    columns: open, high, low, close, volume, time
212:    """
213:    import yfinance as yf
214:
215:    from portfolio.tickers import YF_MAP
216:
217:    yf_ticker = YF_MAP.get(ticker, ticker)
218:    if interval not in _YF_INTERVAL_MAP:
219:        raise ValueError(f"Unsupported yfinance interval: {interval}")
220:    yf_interval, yf_period = _YF_INTERVAL_MAP[interval]
221:
222:    df = yf.download(
223:        yf_ticker,
224:        period=yf_period,
225:        interval=yf_interval,
226:        prepost=True,
227:        progress=False,
228:        auto_adjust=True,
229:    )
230:    if df is None or df.empty:
231:        raise ValueError(f"No yfinance data for {yf_ticker} interval={interval}")
232:
233:    # yfinance returns MultiIndex columns when downloading single ticker too
234:    # (e.g. ('Close', 'NVDA')); flatten them
235:    if isinstance(df.columns, pd.MultiIndex):
236:        df.columns = df.columns.get_level_values(0)
237:
238:    df = df.rename(columns={
239:        "Open": "open", "High": "high", "Low": "low",
240:        "Close": "close", "Volume": "volume",
241:    })
242:    for col in ["open", "high", "low", "close", "volume"]:
243:        if col in df.columns:
244:            df[col] = df[col].astype(float)
245:    df["time"] = df.index
246:    df = df.reset_index(drop=True)
247:    return df.tail(limit)
248:
249:
250:# --- Kline dispatcher ---
251:
252:
253:def _fetch_klines(source, interval, limit):
254:    if "binance_fapi" in source:
255:        _ss._binance_limiter.wait()
256:        return binance_fapi_klines(source["binance_fapi"], interval=interval, limit=limit)
257:    elif "binance" in source:
258:        _ss._binance_limiter.wait()
259:        return binance_klines(source["binance"], interval=interval, limit=limit)
260:    elif "alpaca" in source:
261:        ticker = source["alpaca"]
262:        if _ss._current_market_state in ("closed", "weekend", "holiday"):
263:            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
264:            _ss._yfinance_limiter.wait()
265:            return yfinance_klines(ticker, interval=interval, limit=limit)
266:        _ss._alpaca_limiter.wait()
267:        return alpaca_klines(ticker, interval=interval, limit=limit)
268:    raise ValueError(f"Unknown source: {source}")
269:
270:
271:# --- Multi-timeframe collector ---
272:
273:
274:# yfinance is not thread-safe; serialize calls with a shared lock.
275:# H11/DC-R3-4: use the module-level lock from shared_state so all modules
276:# (fear_greed, golddigger/data_provider, data_collector) share one lock.
277:from portfolio.shared_state import yfinance_lock as _yfinance_lock
278:
279:
280:def _fetch_one_timeframe(source, source_key, label, interval, limit, ttl):
281:    """Fetch and process a single timeframe. Thread-safe."""
282:    cache_key = f"tf_{source_key}_{label}"
283:    if ttl > 0:
284:        with _ss._cache_lock:
285:            cached = _ss._tool_cache.get(cache_key)
286:            if cached and time.time() - cached["time"] < ttl:
287:                return (label, cached["data"])
288:    try:
289:        # yfinance is not thread-safe — serialize its calls
290:        if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
291:            with _yfinance_lock:
292:                df = _fetch_klines(source, interval, limit)
293:        else:
294:            df = _fetch_klines(source, interval, limit)
295:        ind = compute_indicators(df)
296:        if ind is None:
297:            logger.debug("%s/%s: insufficient data (%d rows), skipping",
298:                         source_key, label, len(df) if df is not None else 0)
299:            return None
300:        if label == "Now":
301:            action, conf = None, None
302:        else:
303:            action, conf = technical_signal(ind)
304:        entry = {"indicators": ind, "action": action, "confidence": conf}
305:        if label == "Now":
306:            entry["_df"] = df  # preserve raw DataFrame for enhanced signals
307:        if ttl > 0:
308:            with _ss._cache_lock:
309:                _ss._tool_cache[cache_key] = {"data": entry, "time": time.time()}
310:        return (label, entry)
311:    except Exception as e:
312:        return (label, {"error": str(e)})
313:
314:
315:def collect_timeframes(source):
316:    """Collect all timeframes for a source, fetching in parallel."""
317:    is_stock = "alpaca" in source
318:    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
319:    source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")
320:
321:    # BUG-179: Submit all timeframe fetches with timeout to prevent hangs
322:    with ThreadPoolExecutor(max_workers=len(tfs), thread_name_prefix=f"tf_{source_key}") as pool:
323:        futures = {
324:            pool.submit(_fetch_one_timeframe, source, source_key, label, interval, limit, ttl): label
325:            for label, interval, limit, ttl in tfs
326:        }
327:        raw_results = []
328:        try:
329:            for future in as_completed(futures, timeout=_TF_POOL_TIMEOUT):
330:                result = future.result()
331:                if result is not None:
332:                    raw_results.append(result)
333:        except TimeoutError:
334:            stuck = [lbl for f, lbl in futures.items() if not f.done()]
335:            logger.error("BUG-179: Timeframe pool timeout for %s. Stuck: %s",
336:                         source_key, stuck)
337:            for f in futures:
338:                f.cancel()
339:
340:    # Maintain original timeframe order
341:    tf_order = {label: i for i, (label, _, _, _) in enumerate(tfs)}
342:    raw_results.sort(key=lambda x: tf_order.get(x[0], 999))
343:    return raw_results

 succeeded in 839ms:
1:import logging
2:from pathlib import Path
3:
4:import pandas as pd
5:
6:from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
7:from portfolio.http_retry import fetch_with_retry
8:from portfolio.shared_state import VOLUME_TTL as _VOLUME_TTL
9:from portfolio.shared_state import _alpaca_limiter, _binance_limiter, _cached
10:
11:logger = logging.getLogger("portfolio.macro_context")
12:
13:CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
14:
15:from datetime import UTC
16:
17:from portfolio.tickers import TICKER_SOURCE_MAP as TICKER_MAP
18:
19:
20:def _alpaca_headers():
21:    return get_alpaca_headers()
22:
23:
24:DXY_TTL = 3600
25:# 2026-04-13: Intraday DXY cache is 3 min — 60m bars refresh at each hour
26:# boundary and we want to re-query shortly after the new bar closes.
27:DXY_INTRADAY_TTL = 180
28:TREASURY_TTL_VAL = 3600
29:
30:
31:def _fetch_dxy():
32:    """Fetch DXY data.
33:
34:    2026-04-14: routed via price_source — DXY (DX-Y.NYB) is in the
35:    yfinance allowed-fallback list (no free real-time alternative),
36:    but this preserves a single upgrade point for the day a real-time
37:    DXY feed becomes available.
38:    """
39:    from portfolio.price_source import fetch_klines
40:
41:    h = fetch_klines("DX-Y.NYB", interval="1d", limit=30, period="30d")
42:    if h is None or h.empty:
43:        return None
44:
45:    close = h["close"]
46:    current = float(close.iloc[-1])
47:    sma20 = float(close.rolling(20).mean().iloc[-1])
48:    pct_5d = (
49:        float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
50:    )
51:
52:    if current > sma20:
53:        trend = "strong"
54:    else:
55:        trend = "weak"
56:
57:    return {
58:        "value": round(current, 2),
59:        "sma20": round(sma20, 2),
60:        "trend": trend,
61:        "change_5d_pct": round(pct_5d, 2),
62:    }
63:
64:
65:def get_dxy():
66:    return _cached("dxy", DXY_TTL, _fetch_dxy)
67:
68:
69:# --- Intraday DXY (2026-04-13) ---------------------------------------------
70:# The daily _fetch_dxy above feeds a daily-resolution sub-signal inside
71:# macro_regime. For 1-3h metals prediction (silver had 46.4% consensus
72:# accuracy at 3h despite DXY's R² ~0.6 inverse correlation with silver),
73:# we need 60m-bar DXY data. Primary source is yfinance DX-Y.NYB intraday;
74:# fallback is EURUSD=X which makes up ~57.6% of DXY weight and gives us
75:# a usable synth when the primary pseudo-ticker's intraday feed hiccups.
76:
77:
78:def _dxy_features_from_close(close, *, source: str) -> dict | None:
79:    """Build the intraday DXY payload from a pandas Close series."""
80:    import math
81:
82:    if close is None or len(close) < 2:
83:        return None
84:    last = float(close.iloc[-1])
85:    if math.isnan(last):
86:        return None
87:
88:    def _pct(periods: int) -> float:
89:        if len(close) < periods + 1:
90:            return float("nan")
91:        prior = float(close.iloc[-1 - periods])
92:        if prior == 0 or math.isnan(prior):
93:            return float("nan")
94:        return (last / prior - 1) * 100
95:
96:    change_1h = _pct(1)
97:    change_3h = _pct(3)
98:
99:    return {
100:        "value": round(last, 4),
101:        "change_1h_pct": round(change_1h, 4) if not math.isnan(change_1h) else None,
102:        "change_3h_pct": round(change_3h, 4) if not math.isnan(change_3h) else None,
103:        "source": source,
104:    }
105:
106:
107:def _fetch_dxy_intraday():
108:    """Fetch intraday DXY (60m bars). Fallback chain: primary index → EURUSD synth.
109:
110:    2026-04-14: routed via price_source — DXY and EURUSD=X are both in
111:    the yfinance allowed-fallback list (no free real-time alternative).
112:    The router preserves the same fallback logic but centralizes it.
113:    """
114:    from portfolio.price_source import fetch_klines
115:
116:    def _download(ticker: str):
117:        try:
118:            df = fetch_klines(ticker, interval="60m", limit=120, period="5d")
119:            if df is None or df.empty or "close" not in df.columns:
120:                return None
121:            return df["close"].dropna()
122:        except Exception as exc:
123:            logger.debug("price_source intraday fetch failed for %s: %s", ticker, exc)
124:            return None
125:
126:    # Primary: DX-Y.NYB intraday 60m
127:    close = _download("DX-Y.NYB")
128:    result = _dxy_features_from_close(close, source="DX-Y.NYB")
129:    if result is not None and result.get("change_1h_pct") is not None:
130:        return result
131:
132:    # Fallback: synthesize from EURUSD=X spot.
133:    # DXY weights EUR at ~57.6%; the single-factor approximation
134:    # DXY ≈ c × EURUSD^(-0.576) captures the bulk of DXY's directional
135:    # variance. The constant 58.0 does NOT match real DXY levels (~99) —
136:    # it is arbitrary. Only ``change_1h_pct`` / ``change_3h_pct`` from
137:    # this synth path are usable — the ``value`` field is meaningless.
138:    # Downstream consumers (signals/dxy_cross_asset.py) only read the
139:    # change fields, so this is safe.
140:    eurusd = _download("EURUSD=X")
141:    if eurusd is None or len(eurusd) == 0:
142:        return None
143:    synth = 58.0 * (eurusd ** -0.576)
144:    return _dxy_features_from_close(synth, source="EURUSD=X-synth")
145:
146:
147:def get_dxy_intraday():
148:    """Cached accessor for intraday DXY features."""
149:    return _cached("dxy_intraday", DXY_INTRADAY_TTL, _fetch_dxy_intraday)
150:
151:
152:def _fetch_klines(ticker):
153:    source_type, symbol = TICKER_MAP.get(ticker, (None, None))
154:    if source_type in ("binance", "binance_fapi"):
155:        base_url = BINANCE_FAPI_BASE if source_type == "binance_fapi" else BINANCE_BASE
156:        _binance_limiter.wait()
157:        r = fetch_with_retry(
158:            f"{base_url}/klines",
159:            params={"symbol": symbol, "interval": "15m", "limit": 100},
160:            timeout=10,
161:        )
162:        if r is None:
163:            return None
164:        r.raise_for_status()
165:        raw = r.json()
166:        df = pd.DataFrame(
167:            raw,
168:            columns=[
169:                "open_time",
170:                "open",
171:                "high",
172:                "low",
173:                "close",
174:                "volume",
175:                "close_time",
176:                "quote_vol",
177:                "trades",
178:                "tb",
179:                "tq",
180:                "ignore",
181:            ],
182:        )
183:        for col in ["open", "high", "low", "close", "volume"]:
184:            df[col] = df[col].astype(float)
185:        return df
186:    elif source_type == "alpaca":
187:        from datetime import datetime
188:
189:        _alpaca_limiter.wait()
190:        end = datetime.now(UTC)
191:        start = end - pd.Timedelta(days=5)
192:        r = fetch_with_retry(
193:            f"{ALPACA_BASE}/stocks/{symbol}/bars",
194:            headers=_alpaca_headers(),
195:            params={
196:                "timeframe": "15Min",
197:                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
198:                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
199:                "feed": "iex",
200:            },
201:            timeout=10,
202:        )
203:        if r is None:
204:            return None
205:        r.raise_for_status()
206:        bars = r.json().get("bars") or []
207:        if not bars:
208:            return None
209:        df = pd.DataFrame(bars)
210:        df = df.rename(
211:            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
212:        )
213:        for col in ["open", "high", "low", "close", "volume"]:
214:            df[col] = df[col].astype(float)
215:        return df
216:    return None
217:
218:
219:def _fetch_volume_signal(ticker):
220:    """Compute volume signal from klines for a single ticker."""
221:    klines_df = _fetch_klines(ticker)
222:    if klines_df is None or klines_df.empty:
223:        return None
224:    vol = klines_df["volume"].astype(float)
225:    close = klines_df["close"].astype(float)
226:    if len(vol) < 2:
227:        return None
228:    last_vol = float(vol.iloc[-2])
229:    avg20 = (
230:        float(vol.iloc[:-1].rolling(20).mean().iloc[-1])
231:        if len(vol) >= 22
232:        else float(vol.iloc[:-1].mean())
233:    )
234:    ratio = last_vol / avg20 if avg20 > 0 else 1.0
235:
236:    # Price direction over last 3 completed candles
237:    if len(close) >= 5:
238:        price_change = float(close.iloc[-2] / close.iloc[-5] - 1)
239:    else:
240:        price_change = 0.0
241:
242:    # Volume spike (>1.5x avg) confirms direction
243:    # No spike = abstain (HOLD)
244:    if ratio > 1.5:
245:        if price_change > 0:
246:            action = "BUY"
247:        elif price_change < 0:
248:            action = "SELL"
249:        else:
250:            action = "HOLD"
251:    else:
252:        action = "HOLD"
253:
254:    return {
255:        "ratio": round(ratio, 2),
256:        "spike": ratio > 1.5,
257:        "price_change_3": round(price_change * 100, 2),
258:        "action": action,
259:    }
260:
261:
262:def get_volume_signal(ticker):
263:    return _cached(f"vol_{ticker}", _VOLUME_TTL, _fetch_volume_signal, ticker)
264:
265:
266:from portfolio.fomc_dates import FOMC_DATES_ISO as FOMC_DATES
267:
268:
269:def _fred_10y_fallback():
270:    """FRED DGS10 fallback when yfinance ^TNX fails.
271:
272:    Added 2026-04-09 after yfinance ^TNX fetch started returning None for
273:    extended periods (16h stale), triggering `TypeError('NoneType' object is
274:    not subscriptable')` from this function's callers.
275:
276:    Returns a dict in the same shape as the yfinance path would for the "10y"
277:    key — {yield_pct, change_5d} — or None if FRED is also unavailable.
278:    Reuses `portfolio.golddigger.data_provider.fetch_us10y`, which already
279:    has its own 1h cache + circuit breaker.
280:    """
281:    try:
282:        import json as _json
283:
284:        from portfolio.golddigger.data_provider import fetch_us10y
285:        fred_key = ""
286:        try:
287:            with open(CONFIG_FILE, encoding="utf-8") as f:
288:                fred_key = _json.load(f).get("golddigger", {}).get("fred_api_key", "") or ""
289:        except Exception:
290:            return None
291:        if not fred_key:
292:            return None
293:        yield_decimal = fetch_us10y(fred_key, series_id="DGS10")
294:        if yield_decimal is None:
295:            return None
296:        # fetch_us10y returns decimal (0.0425); yfinance path uses pct (4.25).
297:        return {"yield_pct": round(yield_decimal * 100, 3), "change_5d": 0.0}
298:    except Exception:
299:        logger.warning("FRED fallback failed for 10y", exc_info=True)
300:        return None
301:
302:
303:def _fetch_treasury():
304:    """Fetch treasury yield data, with FRED fallback for 10y.
305:
306:    2026-04-14: routed via price_source. Treasury tickers (^TNX, ^TYX)
307:    are CBOE-style indices with no free intraday alternative; the router
308:    sends them through yfinance. 2YY=F is a futures pseudo-ticker also
309:    with no free alternative.
310:    """
311:    from portfolio.price_source import fetch_klines
312:
313:    tickers = {"10y": "^TNX", "2y": "2YY=F", "30y": "^TYX"}
314:    result = {}
315:    for label, sym in tickers.items():
316:        try:
317:            h = fetch_klines(sym, interval="1d", limit=30, period="30d")
318:            if h is None or h.empty:
319:                continue
320:            close = h["close"]
321:            current = float(close.iloc[-1])
322:            pct_5d = (
323:                float((close.iloc[-1] / close.iloc[-5] - 1) * 100)
324:                if len(close) >= 5
325:                else 0
326:            )
327:            result[label] = {
328:                "yield_pct": round(current, 3),
329:                "change_5d": round(pct_5d, 2),
330:            }
331:        except Exception:
332:            logger.warning("Treasury fetch failed for %s", label, exc_info=True)
333:
334:    # FRED fallback for 10y when yfinance ^TNX is down (common symptom:
335:    # No data / NoneType errors). Other maturities don't have a clean FRED
336:    # fallback via this helper, so they stay yfinance-only.
337:    if "10y" not in result:
338:        fallback = _fred_10y_fallback()
339:        if fallback is not None:
340:            logger.info("Treasury 10y: using FRED fallback (%.3f%%)", fallback["yield_pct"])
341:            result["10y"] = fallback
342:
343:    if "10y" in result and "2y" in result:
344:        spread = result["10y"]["yield_pct"] - result["2y"]["yield_pct"]
345:        result["spread_2s10s"] = round(spread, 3)
346:        if spread < 0:
347:            result["curve"] = "inverted"
348:        elif spread < 0.2:
349:            result["curve"] = "flat"
350:        else:
351:            result["curve"] = "normal"
352:
353:    return result or None
354:
355:
356:def get_treasury():
357:    return _cached("treasury", TREASURY_TTL_VAL, _fetch_treasury)
358:
359:
360:def get_fed_calendar():
361:    from datetime import datetime, timedelta
362:
363:    today = datetime.now(UTC).strftime("%Y-%m-%d")
364:    upcoming = [d for d in FOMC_DATES if d >= today]
365:    if not upcoming:
366:        return None
367:
368:    next_date = upcoming[0]
369:    days_until = (
370:        datetime.strptime(next_date, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")
371:    ).days
372:
373:    is_meeting_day = today in FOMC_DATES
374:    is_day_before = any(
375:        (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
376:        == today
377:        for d in FOMC_DATES
378:    )
379:
380:    result = {
381:        "next_fomc": next_date,
382:        "days_until": days_until,
383:        "meetings_remaining": len(upcoming) // 2,
384:    }
385:    if is_meeting_day:
386:        result["warning"] = "FOMC meeting TODAY — expect volatility"
387:    elif is_day_before:
388:        result["warning"] = "FOMC meeting TOMORROW — positioning risk"
389:    elif days_until <= 7:
390:        result["warning"] = f"FOMC in {days_until} days — pre-meeting drift possible"
391:
392:    return result
393:
394:
395:if __name__ == "__main__":
396:    dxy = get_dxy()
397:    print(f"DXY: {dxy}")
398:    treasury = get_treasury()
399:    print(f"Treasury: {treasury}")
400:    fed = get_fed_calendar()
401:    print(f"Fed: {fed}")
402:    for t in list(TICKER_MAP.keys()):
403:        print(f"{t}: {get_volume_signal(t)}")

 succeeded in 872ms:
1:"""USD/SEK exchange rate fetching with caching and staleness alerts.
2:
3:NOTE: Not migrated to shared_state._cached() because this module has custom
4:staleness behaviour: Telegram alerting when stale, hardcoded 10.50 fallback,
5:and ConnectionError semantics. These would be lost with _cached().
6:"""
7:
8:import logging
9:import threading
10:import time
11:
12:from portfolio.api_utils import load_config as _load_config
13:from portfolio.http_retry import fetch_with_retry
14:
15:logger = logging.getLogger("portfolio.fx_rates")
16:
17:# Shared FX constants — imported by risk_management.py and monte_carlo_risk.py
18:FX_RATE_FALLBACK = 10.50
19:FX_RATE_MIN = 7.0
20:FX_RATE_MAX = 15.0
21:
22:# BUG-215: Thread-safe FX cache. Accessed from 8-worker ThreadPoolExecutor.
23:_fx_lock = threading.Lock()
24:_fx_cache = {"rate": None, "time": 0}
25:_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed
26:
27:
28:def fetch_usd_sek():
29:    now = time.time()
30:    with _fx_lock:
31:        cached_rate = _fx_cache["rate"]
32:        cached_time = _fx_cache["time"]
33:    if cached_rate and now - cached_time < 900:  # 15 min cache
34:        return cached_rate
35:    try:
36:        r = fetch_with_retry(
37:            "https://api.frankfurter.app/latest",
38:            params={"from": "USD", "to": "SEK"},
39:            timeout=10,
40:        )
41:        if r is None:
42:            raise ConnectionError("FX rate request failed after retries")
43:        r.raise_for_status()
44:        rate = float(r.json()["rates"]["SEK"])
45:        # BUG-117: Sanity check — SEK/USD should be in 7-15 range historically.
46:        # If outside this range, the API may be returning bad data.
47:        if not (FX_RATE_MIN <= rate <= FX_RATE_MAX):
48:            logger.error("FX rate %.4f SEK/USD outside sane bounds (7-15) — ignoring", rate)
49:        else:
50:            with _fx_lock:
51:                _fx_cache["rate"] = rate
52:                _fx_cache["time"] = now
53:            return rate
54:    except Exception as e:
55:        logger.warning("FX rate fetch failed: %s", e)
56:    # Fallback to stale cached value
57:    with _fx_lock:
58:        cached_rate = _fx_cache["rate"]
59:        cached_time = _fx_cache["time"]
60:    if cached_rate:
61:        age_secs = now - cached_time
62:        if age_secs > _FX_STALE_THRESHOLD:
63:            logger.warning("Using stale FX rate (%.1fh old)", age_secs / 3600)
64:            _fx_alert_telegram(age_secs)
65:        return cached_rate
66:    # Last resort: hardcoded fallback
67:    # BUG-117: Use ERROR level — hardcoded rate may be severely stale.
68:    # Portfolio valuations using this rate could be off by 10-15% if SEK has moved.
69:    logger.error("Using hardcoded FX fallback rate %.2f SEK — no cached or live rate available", FX_RATE_FALLBACK)
70:    _fx_alert_telegram(None)
71:    return FX_RATE_FALLBACK
72:
73:
74:def _fx_alert_telegram(age_secs):
75:    """Send a one-shot Telegram alert about FX rate issues. Fires at most once per 4h."""
76:    now = time.time()
77:    with _fx_lock:
78:        last_alert = _fx_cache.get("_last_fx_alert", 0)
79:        if now - last_alert < 14400:  # 4h cooldown between alerts
80:            return
81:        _fx_cache["_last_fx_alert"] = now
82:    try:
83:        config = _load_config()
84:        if age_secs is not None:
85:            msg = f"_FX WARNING: USD/SEK rate is {age_secs / 3600:.1f}h stale. API may be down._"
86:        else:
87:            msg = f"_FX WARNING: Using hardcoded fallback rate {FX_RATE_FALLBACK} SEK. No live or cached rate available._"
88:        from portfolio.message_store import send_or_store
89:        send_or_store(msg, config, category="error")
90:    except Exception as e:
91:        logger.debug("FX Telegram alert failed: %s", e)

 succeeded in 864ms:
1:"""Binance FAPI futures data — open interest, long/short ratios, funding history.
2:
3:Fetches public endpoints for crypto tickers (BTC-USD, ETH-USD only).
4:Uses existing infrastructure: fetch_json, _cached, _binance_limiter.
5:"""
6:
7:import logging
8:import time
9:
10:from portfolio.api_utils import BINANCE_FAPI_BASE, BINANCE_FUTURES_DATA
11:from portfolio.http_retry import fetch_json
12:from portfolio.shared_state import _binance_limiter, _cached
13:
14:logger = logging.getLogger("portfolio.futures_data")
15:
16:SYMBOL_MAP = {
17:    "BTC-USD": "BTCUSDT",
18:    "ETH-USD": "ETHUSDT",
19:}
20:
21:# Cache TTLs
22:_OI_TTL = 300        # 5 min
23:_LS_TTL = 300        # 5 min
24:_FUNDING_TTL = 900   # 15 min
25:
26:
27:def _fetch_json(url, params=None, timeout=10):
28:    """Fetch JSON from Binance FAPI with rate limiting and retry."""
29:    _binance_limiter.wait()
30:    return fetch_json(url, params=params, timeout=timeout, label="binance_fapi")
31:
32:
33:def get_open_interest(ticker):
34:    """Current open interest for a crypto ticker.
35:
36:    Returns: {oi, oi_usdt, symbol, time} or None.
37:    """
38:    if ticker not in SYMBOL_MAP:
39:        return None
40:    symbol = SYMBOL_MAP[ticker]
41:
42:    def _fetch():
43:        data = _fetch_json(
44:            f"{BINANCE_FAPI_BASE}/openInterest",
45:            params={"symbol": symbol},
46:        )
47:        if data is None:
48:            return None
49:        return {
50:            "oi": float(data["openInterest"]),
51:            "symbol": data["symbol"],
52:            "time": data.get("time", int(time.time() * 1000)),
53:        }
54:
55:    return _cached(f"futures_oi_{ticker}", _OI_TTL, _fetch)
56:
57:
58:def get_open_interest_history(ticker, period="5m", limit=30):
59:    """Historical open interest snapshots.
60:
61:    Returns: list of {oi, oi_usdt, timestamp} or None.
62:    """
63:    if ticker not in SYMBOL_MAP:
64:        return None
65:    symbol = SYMBOL_MAP[ticker]
66:
67:    def _fetch():
68:        data = _fetch_json(
69:            f"{BINANCE_FUTURES_DATA}/openInterestHist",
70:            params={"symbol": symbol, "period": period, "limit": limit},
71:        )
72:        if not data:
73:            return None
74:        return [
75:            {
76:                "oi": float(d["sumOpenInterest"]),
77:                "oi_usdt": float(d["sumOpenInterestValue"]),
78:                "timestamp": d["timestamp"],
79:            }
80:            for d in data
81:        ]
82:
83:    return _cached(f"futures_oi_hist_{ticker}_{period}", _OI_TTL, _fetch)
84:
85:
86:def get_long_short_ratio(ticker, period="5m", limit=30):
87:    """Global long/short account ratio.
88:
89:    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
90:    """
91:    if ticker not in SYMBOL_MAP:
92:        return None
93:    symbol = SYMBOL_MAP[ticker]
94:
95:    def _fetch():
96:        data = _fetch_json(
97:            f"{BINANCE_FUTURES_DATA}/globalLongShortAccountRatio",
98:            params={"symbol": symbol, "period": period, "limit": limit},
99:        )
100:        if not data:
101:            return None
102:        return [
103:            {
104:                "longShortRatio": float(d["longShortRatio"]),
105:                "longAccount": float(d["longAccount"]),
106:                "shortAccount": float(d["shortAccount"]),
107:                "timestamp": d["timestamp"],
108:            }
109:            for d in data
110:        ]
111:
112:    return _cached(f"futures_ls_{ticker}_{period}", _LS_TTL, _fetch)
113:
114:
115:def get_top_trader_position_ratio(ticker, period="5m", limit=30):
116:    """Top trader long/short position ratio.
117:
118:    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
119:    """
120:    if ticker not in SYMBOL_MAP:
121:        return None
122:    symbol = SYMBOL_MAP[ticker]
123:
124:    def _fetch():
125:        data = _fetch_json(
126:            f"{BINANCE_FUTURES_DATA}/topLongShortPositionRatio",
127:            params={"symbol": symbol, "period": period, "limit": limit},
128:        )
129:        if not data:
130:            return None
131:        return [
132:            {
133:                "longShortRatio": float(d["longShortRatio"]),
134:                "longAccount": float(d["longAccount"]),
135:                "shortAccount": float(d["shortAccount"]),
136:                "timestamp": d["timestamp"],
137:            }
138:            for d in data
139:        ]
140:
141:    return _cached(f"futures_top_pos_{ticker}_{period}", _LS_TTL, _fetch)
142:
143:
144:def get_top_trader_account_ratio(ticker, period="5m", limit=30):
145:    """Top trader long/short account ratio.
146:
147:    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
148:    """
149:    if ticker not in SYMBOL_MAP:
150:        return None
151:    symbol = SYMBOL_MAP[ticker]
152:
153:    def _fetch():
154:        data = _fetch_json(
155:            f"{BINANCE_FUTURES_DATA}/topLongShortAccountRatio",
156:            params={"symbol": symbol, "period": period, "limit": limit},
157:        )
158:        if not data:
159:            return None
160:        return [
161:            {
162:                "longShortRatio": float(d["longShortRatio"]),
163:                "longAccount": float(d["longAccount"]),
164:                "shortAccount": float(d["shortAccount"]),
165:                "timestamp": d["timestamp"],
166:            }
167:            for d in data
168:        ]
169:
170:    return _cached(f"futures_top_acct_{ticker}_{period}", _LS_TTL, _fetch)
171:
172:
173:def get_funding_rate_history(ticker, limit=100):
174:    """Historical funding rates.
175:
176:    Returns: list of {fundingRate, fundingTime, symbol} or None.
177:    """
178:    if ticker not in SYMBOL_MAP:
179:        return None
180:    symbol = SYMBOL_MAP[ticker]
181:
182:    def _fetch():
183:        data = _fetch_json(
184:            f"{BINANCE_FAPI_BASE}/fundingRate",
185:            params={"symbol": symbol, "limit": limit},
186:        )
187:        if not data:
188:            return None
189:        return [
190:            {
191:                "fundingRate": float(d["fundingRate"]),
192:                "fundingTime": d["fundingTime"],
193:                "symbol": d.get("symbol", symbol),
194:            }
195:            for d in data
196:        ]
197:
198:    return _cached(f"futures_funding_hist_{ticker}", _FUNDING_TTL, _fetch)
199:
200:
201:def get_all_futures_data(ticker):
202:    """Fetch all futures data for a ticker. Each sub-key can be None on failure.
203:
204:    Returns: dict with keys: open_interest, oi_history, ls_ratio,
205:             top_position_ratio, top_account_ratio, funding_history.
206:    """
207:    if ticker not in SYMBOL_MAP:
208:        return None
209:
210:    return {
211:        "open_interest": get_open_interest(ticker),
212:        "oi_history": get_open_interest_history(ticker),
213:        "ls_ratio": get_long_short_ratio(ticker),
214:        "top_position_ratio": get_top_trader_position_ratio(ticker),
215:        "top_account_ratio": get_top_trader_account_ratio(ticker),
216:        "funding_history": get_funding_rate_history(ticker),
217:    }
218:
219:
220:if __name__ == "__main__":
221:    import json
222:    for t in ["BTC-USD", "ETH-USD"]:
223:        print(f"\n=== {t} ===")
224:        result = get_all_futures_data(t)
225:        if result is None:
226:            print("  Not a crypto ticker — skipped")
227:        else:
228:            for k, v in result.items():
229:                if v is None:
230:                    print(f"  {k}: None (fetch failed)")
231:                elif isinstance(v, list):
232:                    print(f"  {k}: {len(v)} entries, latest={v[-1] if v else 'empty'}")
233:                else:
234:                    print(f"  {k}: {json.dumps(v, indent=2)}")

 succeeded in 880ms:
1:"""Earnings calendar — proximity gate to prevent trading near earnings.
2:
3:Fetches upcoming earnings dates from yfinance for US stock tickers.
4:When a ticker has earnings within GATE_DAYS, BUY signals are gated to HOLD
5:to avoid unpredictable binary events.
6:
7:Cache: per-ticker, 24h TTL (earnings dates don't change intraday).
8:"""
9:
10:import logging
11:import threading
12:import time
13:from contextlib import suppress
14:from datetime import UTC, datetime
15:
16:from portfolio.tickers import STOCK_SYMBOLS
17:
18:logger = logging.getLogger("portfolio.earnings_calendar")
19:
20:# Gate window: force HOLD if earnings within this many calendar days
21:GATE_DAYS = 2
22:
23:# Cache TTL: 24 hours (earnings dates are stable)
24:EARNINGS_CACHE_TTL = 86400
25:
26:# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
27:_earnings_cache: dict[str, dict] = {}
28:_earnings_lock = threading.Lock()
29:
30:
31:def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
32:    """Fetch next earnings date from Alpha Vantage EARNINGS endpoint.
33:
34:    Uses the already-configured AV API key and rate limiter.
35:    """
36:    try:
37:        from portfolio.api_utils import load_config
38:        from portfolio.http_retry import fetch_with_retry
39:        from portfolio.shared_state import _alpha_vantage_limiter
40:
41:        config = load_config()
42:        # C9/DC-R3-1: key lives under config["alpha_vantage"]["api_key"], not
43:        # the flat "alpha_vantage_key" key that doesn't exist.
44:        api_key = config.get("alpha_vantage", {}).get("api_key", "")
45:        if not api_key:
46:            return None
47:
48:        _alpha_vantage_limiter.wait()
49:        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
50:        # because there is no public increment function exported from that module.
51:        # Known limitation — earnings fetches consume 1 AV call each but are not
52:        # reflected in the budget tracker.  Each ticker only fetches once per 24h.
53:        r = fetch_with_retry(
54:            "https://www.alphavantage.co/query",
55:            params={
56:                "function": "EARNINGS",
57:                "symbol": ticker,
58:                "apikey": api_key,
59:            },
60:            timeout=10,
61:        )
62:        if r is None:
63:            return None
64:        data = r.json()
65:
66:        # AV EARNINGS returns quarterlyEarnings and annualEarnings
67:        quarterly = data.get("quarterlyEarnings", [])
68:        if not quarterly:
69:            return None
70:
71:        today = datetime.now(UTC).date()
72:        # Find the next upcoming earnings (reportedDate in the future or very recent)
73:        for q in quarterly:
74:            rd = q.get("reportedDate")
75:            if not rd or rd == "None":
76:                continue
77:            try:
78:                from datetime import date as _date
79:                ed = _date.fromisoformat(rd)
80:                days_until = (ed - today).days
81:                if days_until >= -1:
82:                    return {
83:                        "earnings_date": ed.isoformat(),
84:                        "days_until": days_until,
85:                        "gate_active": 0 <= days_until <= GATE_DAYS,
86:                        "timing": "unknown",
87:                    }
88:            except (ValueError, TypeError):
89:                continue
90:
91:        return None
92:    except Exception:
93:        logger.debug("Alpha Vantage earnings fetch failed for %s", ticker, exc_info=True)
94:        return None
95:
96:
97:def _fetch_earnings_yfinance(ticker: str) -> dict | None:
98:    """Fallback: fetch next earnings date from yfinance."""
99:    try:
100:        import yfinance as yf
101:
102:        t = yf.Ticker(ticker)
103:        with suppress(Exception):
104:            cal = t.calendar
105:            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
106:                if isinstance(cal, dict):
107:                    earnings_date = cal.get("Earnings Date")
108:                    if isinstance(earnings_date, list) and earnings_date:
109:                        earnings_date = earnings_date[0]
110:                else:
111:                    if "Earnings Date" in cal.index:
112:                        earnings_date = cal.loc["Earnings Date"].iloc[0]
113:                    else:
114:                        earnings_date = None
115:
116:                if earnings_date is not None:
117:                    if hasattr(earnings_date, "date"):
118:                        ed = earnings_date.date()
119:                    elif isinstance(earnings_date, str):
120:                        ed = datetime.fromisoformat(earnings_date).date()
121:                    else:
122:                        ed = None
123:
124:                    if ed:
125:                        today = datetime.now(UTC).date()
126:                        days_until = (ed - today).days
127:                        if days_until < -5:
128:                            return None
129:                        return {
130:                            "earnings_date": ed.isoformat(),
131:                            "days_until": days_until,
132:                            "gate_active": 0 <= days_until <= GATE_DAYS,
133:                            "timing": "unknown",
134:                        }
135:        return None
136:    except Exception:
137:        logger.debug("yfinance earnings fetch failed for %s", ticker, exc_info=True)
138:        return None
139:
140:
141:def _fetch_earnings_date(ticker: str) -> dict | None:
142:    """Fetch next earnings date — Alpha Vantage primary, yfinance fallback.
143:
144:    Returns dict with earnings_date, days_until, timing, or None.
145:    """
146:    # Primary: Alpha Vantage (already have API key + rate limiter)
147:    result = _fetch_earnings_alpha_vantage(ticker)
148:    if result:
149:        return result
150:
151:    # Fallback: yfinance
152:    return _fetch_earnings_yfinance(ticker)
153:
154:
155:def get_earnings_proximity(ticker: str) -> dict | None:
156:    """Get cached earnings proximity for a ticker.
157:
158:    Only works for STOCK_SYMBOLS — returns None for crypto/metals.
159:
160:    Returns:
161:        dict with earnings_date, days_until, gate_active, timing
162:        or None if no upcoming earnings found.
163:    """
164:    if ticker not in STOCK_SYMBOLS:
165:        return None
166:
167:    now = time.time()
168:    with _earnings_lock:
169:        cached = _earnings_cache.get(ticker)
170:        if cached and now - cached["time"] < EARNINGS_CACHE_TTL:
171:            return cached["data"]
172:
173:    # Fetch fresh
174:    data = _fetch_earnings_date(ticker)
175:
176:    with _earnings_lock:
177:        _earnings_cache[ticker] = {"data": data, "time": now}
178:
179:    return data
180:
181:
182:def should_gate_earnings(ticker: str) -> bool:
183:    """Check if ticker should be gated to HOLD due to nearby earnings.
184:
185:    Returns True if:
186:    - Ticker is a US stock (not crypto/metals)
187:    - Earnings are within GATE_DAYS calendar days
188:    """
189:    if ticker not in STOCK_SYMBOLS:
190:        return False
191:
192:    prox = get_earnings_proximity(ticker)
193:    if prox is None:
194:        return False
195:
196:    return prox.get("gate_active", False)
197:
198:
199:def get_all_earnings_proximity() -> dict:
200:    """Get earnings proximity for all stock tickers.
201:
202:    Returns dict keyed by ticker with proximity info.
203:    Used by reporting.py to enrich agent_summary.
204:    """
205:    result = {}
206:    for ticker in STOCK_SYMBOLS:
207:        prox = get_earnings_proximity(ticker)
208:        if prox:
209:            result[ticker] = prox
210:    return result
211:
212:
213:def clear_cache() -> None:
214:    """Clear the earnings cache (for testing)."""
215:    with _earnings_lock:
216:        _earnings_cache.clear()

 succeeded in 878ms:
1:"""Alpha Vantage fundamentals — daily-refresh stock fundamentals cache.
2:
3:Fetches OVERVIEW endpoint data for stock tickers and caches persistently.
4:Free tier: 25 requests/day, 5 requests/minute.
5:
6:Not used for crypto or metals (no OVERVIEW data available).
7:"""
8:
9:import logging
10:import threading
11:from datetime import UTC, datetime
12:from pathlib import Path
13:
14:from portfolio.circuit_breaker import CircuitBreaker
15:from portfolio.file_utils import load_json
16:from portfolio.http_retry import fetch_with_retry
17:from portfolio.shared_state import _alpha_vantage_limiter
18:from portfolio.tickers import STOCK_SYMBOLS
19:
20:logger = logging.getLogger("portfolio.alpha_vantage")
21:
22:BASE_DIR = Path(__file__).resolve().parent.parent
23:DATA_DIR = BASE_DIR / "data"
24:CACHE_FILE = DATA_DIR / "fundamentals_cache.json"
25:
26:AV_BASE_URL = "https://www.alphavantage.co/query"
27:
28:# Module-level state
29:_cache = {}  # ticker -> normalized fundamentals dict
30:_cache_lock = threading.Lock()
31:_daily_budget_used = 0
32:_budget_reset_date = ""  # ISO date string for budget tracking
33:_cb = CircuitBreaker("alpha_vantage", failure_threshold=3, recovery_timeout=300)
34:
35:
36:def load_persistent_cache():
37:    """Load fundamentals cache from disk on startup."""
38:    global _cache
39:    data = load_json(CACHE_FILE)
40:    if data is None:
41:        logger.info("No fundamentals cache found at %s", CACHE_FILE)
42:        return
43:    if isinstance(data, dict):
44:        with _cache_lock:
45:            _cache = data
46:        logger.info("Loaded fundamentals for %d tickers from cache", len(data))
47:
48:
49:def _save_persistent_cache():
50:    """Write current cache to disk atomically."""
51:    try:
52:        from portfolio.file_utils import atomic_write_json
53:        with _cache_lock:
54:            snapshot = dict(_cache)
55:        atomic_write_json(CACHE_FILE, snapshot)
56:    except Exception as e:
57:        logger.warning("Failed to save fundamentals cache: %s", e)
58:
59:
60:def get_fundamentals(ticker):
61:    """Return cached fundamentals for a ticker, or None if not available."""
62:    with _cache_lock:
63:        return _cache.get(ticker)
64:
65:
66:def get_all_fundamentals():
67:    """Return all cached fundamentals (for reporting)."""
68:    with _cache_lock:
69:        return dict(_cache)
70:
71:
72:def _normalize_overview(raw):
73:    """Convert Alpha Vantage OVERVIEW response to clean format.
74:
75:    AV returns "None" strings for missing values and all values as strings.
76:    """
77:    def _float(val, default=None):
78:        if val is None or val == "None" or val == "-" or val == "":
79:            return default
80:        try:
81:            return float(val)
82:        except (ValueError, TypeError):
83:            return default
84:
85:    def _int(val, default=None):
86:        if val is None or val == "None" or val == "-" or val == "":
87:            return default
88:        try:
89:            return int(float(val))
90:        except (ValueError, TypeError):
91:            return default
92:
93:    # Check for error responses
94:    if "Error Message" in raw or "Note" in raw:
95:        return None
96:
97:    # Require at least Symbol to be present
98:    if not raw.get("Symbol"):
99:        return None
100:
101:    result = {
102:        "pe_ratio": _float(raw.get("PERatio")),
103:        "forward_pe": _float(raw.get("ForwardPE")),
104:        "peg_ratio": _float(raw.get("PEGRatio")),
105:        "eps": _float(raw.get("EPS")),
106:        "revenue_growth_yoy": _float(raw.get("QuarterlyRevenueGrowthYOY")),
107:        "earnings_growth_yoy": _float(raw.get("QuarterlyEarningsGrowthYOY")),
108:        "profit_margin": _float(raw.get("ProfitMargin")),
109:        "market_cap": _int(raw.get("MarketCapitalization")),
110:        "sector": raw.get("Sector") if raw.get("Sector") != "None" else None,
111:        "industry": raw.get("Industry") if raw.get("Industry") != "None" else None,
112:        "dividend_yield": _float(raw.get("DividendYield")),
113:        "analyst_target": _float(raw.get("AnalystTargetPrice")),
114:        "analyst_ratings": {
115:            "strong_buy": _int(raw.get("AnalystRatingStrongBuy"), 0),
116:            "buy": _int(raw.get("AnalystRatingBuy"), 0),
117:            "hold": _int(raw.get("AnalystRatingHold"), 0),
118:            "sell": _int(raw.get("AnalystRatingSell"), 0),
119:            "strong_sell": _int(raw.get("AnalystRatingStrongSell"), 0),
120:        },
121:        "beta": _float(raw.get("Beta")),
122:        "w52_high": _float(raw.get("52WeekHigh")),
123:        "w52_low": _float(raw.get("52WeekLow")),
124:        "_fetched_at": datetime.now(UTC).isoformat(),
125:    }
126:    return result
127:
128:
129:def _fetch_overview(ticker, api_key):
130:    """Fetch OVERVIEW data for a single ticker from Alpha Vantage."""
131:    params = {
132:        "function": "OVERVIEW",
133:        "symbol": ticker,
134:        "apikey": api_key,
135:    }
136:    resp = fetch_with_retry(
137:        AV_BASE_URL,
138:        params=params,
139:        timeout=15,
140:        retries=2,
141:    )
142:    if resp is None:
143:        return None
144:    try:
145:        data = resp.json()
146:    except (ValueError, AttributeError):
147:        return None
148:
149:    # Rate limit info check
150:    if isinstance(data, dict) and "Note" in data:
151:        logger.warning("Alpha Vantage rate limit hit: %s", data["Note"][:100])
152:        return None
153:
154:    return data
155:
156:
157:def _check_budget():
158:    """Check and reset daily budget counter. Returns current usage count.
159:
160:    BUG-108: Protected by _cache_lock for thread safety.
161:    """
162:    global _daily_budget_used, _budget_reset_date
163:    today = datetime.now(UTC).strftime("%Y-%m-%d")
164:    with _cache_lock:
165:        if _budget_reset_date != today:
166:            _daily_budget_used = 0
167:            _budget_reset_date = today
168:        return _daily_budget_used
169:
170:
171:def _is_stale(ticker, max_stale_days=5):
172:    """Check if cached data for ticker is older than max_stale_days."""
173:    with _cache_lock:
174:        entry = _cache.get(ticker)
175:    if not entry:
176:        return True
177:    fetched_at = entry.get("_fetched_at")
178:    if not fetched_at:
179:        return True
180:    try:
181:        fetched_time = datetime.fromisoformat(fetched_at)
182:        age_seconds = (datetime.now(UTC) - fetched_time).total_seconds()
183:        return age_seconds > max_stale_days * 86400
184:    except (ValueError, TypeError):
185:        return True
186:
187:
188:def _cache_age_hours(ticker):
189:    """Return age of cached data in hours, or None if not cached."""
190:    with _cache_lock:
191:        entry = _cache.get(ticker)
192:    if not entry:
193:        return None
194:    fetched_at = entry.get("_fetched_at")
195:    if not fetched_at:
196:        return None
197:    try:
198:        fetched_time = datetime.fromisoformat(fetched_at)
199:        return (datetime.now(UTC) - fetched_time).total_seconds() / 3600
200:    except (ValueError, TypeError):
201:        return None
202:
203:
204:def refresh_fundamentals_batch(config):
205:    """Batch-refresh fundamentals for all stock tickers.
206:
207:    Respects rate limits (5/min) and daily budget (25/day).
208:    Returns number of successfully refreshed tickers.
209:    """
210:    global _daily_budget_used
211:
212:    av_config = config.get("alpha_vantage", {})
213:    if not av_config.get("enabled", True):
214:        logger.info("Alpha Vantage disabled in config")
215:        return 0
216:
217:    api_key = av_config.get("api_key", "")
218:    if not api_key:
219:        logger.warning("Alpha Vantage API key not configured")
220:        return 0
221:
222:    daily_budget = av_config.get("daily_budget", 25)
223:    skip_tickers = set(av_config.get("skip_tickers", []))
224:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
225:
226:    # Check circuit breaker
227:    if not _cb.allow_request():
228:        logger.info("Alpha Vantage circuit breaker active (%s)", _cb.state.value)
229:        return 0
230:
231:    budget_used = _check_budget()
232:    if budget_used >= daily_budget:
233:        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
234:        return 0
235:
236:    # Build refresh list: stock tickers not recently cached, not skipped
237:    tickers_to_refresh = []
238:    for ticker in sorted(STOCK_SYMBOLS):
239:        if ticker in skip_tickers:
240:            continue
241:        age = _cache_age_hours(ticker)
242:        if age is None or age > cache_ttl_hours:
243:            tickers_to_refresh.append(ticker)
244:
245:    if not tickers_to_refresh:
246:        logger.info("All fundamentals fresh, nothing to refresh")
247:        return 0
248:
249:    # Cap by remaining budget
250:    remaining_budget = daily_budget - budget_used
251:    tickers_to_refresh = tickers_to_refresh[:remaining_budget]
252:
253:    logger.info(
254:        "Refreshing fundamentals for %d tickers (budget: %d/%d used)",
255:        len(tickers_to_refresh), budget_used, daily_budget,
256:    )
257:
258:    success_count = 0
259:    for ticker in tickers_to_refresh:
260:        # Rate limit
261:        _alpha_vantage_limiter.wait()
262:
263:        try:
264:            raw = _fetch_overview(ticker, api_key)
265:            if raw is None:
266:                _cb.record_failure()
267:                if not _cb.allow_request():
268:                    break
269:                continue
270:
271:            normalized = _normalize_overview(raw)
272:            if normalized is None:
273:                logger.warning("Alpha Vantage: empty/error response for %s", ticker)
274:                _cb.record_failure()
275:                if not _cb.allow_request():
276:                    break
277:                continue
278:
279:            with _cache_lock:
280:                _cache[ticker] = normalized
281:                _daily_budget_used += 1  # BUG-108: increment under lock
282:            _cb.record_success()
283:            success_count += 1
284:            logger.info("Refreshed fundamentals for %s (PE=%.1f, sector=%s)",
285:                        ticker,
286:                        normalized.get("pe_ratio") or 0,
287:                        normalized.get("sector", "?"))
288:
289:        except Exception as e:
290:            logger.warning("Alpha Vantage fetch failed for %s: %s", ticker, e)
291:            _cb.record_failure()
292:            if not _cb.allow_request():
293:                break
294:
295:    if success_count > 0:
296:        _save_persistent_cache()
297:        logger.info("Fundamentals refresh complete: %d/%d succeeded", success_count, len(tickers_to_refresh))
298:
299:    return success_count
300:
301:
302:def should_batch_refresh(config):
303:    """Check if a batch refresh should run now.
304:
305:    Only runs when cache is >24h old. Designed to be called from the main loop.
306:    """
307:    av_config = config.get("alpha_vantage", {})
308:    if not av_config.get("enabled", True) or not av_config.get("api_key", ""):
309:        return False
310:
311:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
312:
313:    # Check if any stock ticker needs refresh
314:    for ticker in STOCK_SYMBOLS:
315:        if ticker in set(av_config.get("skip_tickers", [])):
316:            continue
317:        age = _cache_age_hours(ticker)
318:        if age is None or age > cache_ttl_hours:
319:            return True
320:
321:    return False

 succeeded in 892ms:
1:"""BGeometrics on-chain data integration for Bitcoin.
2:
3:Fetches MVRV Z-Score, SOPR, NUPL, realized price, exchange netflow,
4:and liquidation data from the free BGeometrics API (bitcoin-data.com).
5:
6:Free tier: 8 requests/hour, 15 requests/day.
7:Budget: 6 metrics x 2 refreshes/day = 12 requests.
8:Cache: 12 hours per metric (on-chain data doesn't change fast).
9:
10:Usage:
11:    from portfolio.onchain_data import get_onchain_data, interpret_onchain
12:    data = get_onchain_data()  # returns dict or None
13:    interp = interpret_onchain(data)  # returns interpretation dict
14:"""
15:
16:import logging
17:import time
18:from datetime import datetime
19:from pathlib import Path
20:
21:from portfolio.api_utils import load_config as _load_config
22:from portfolio.file_utils import atomic_write_json, load_json
23:from portfolio.http_retry import fetch_json
24:from portfolio.shared_state import _cached
25:
26:logger = logging.getLogger("portfolio.onchain_data")
27:
28:
29:def _coerce_epoch(value) -> float:
30:    """A-DE-5 (2026-04-11): Coerce a cache timestamp to a unix-epoch float.
31:
32:    Older versions of onchain_cache.json stored "ts" as an ISO 8601 string
33:    instead of an epoch number. The seeding code below does
34:    `time.time() - cache_ts < TTL` which crashes with TypeError when ts is
35:    a string. This helper accepts:
36:        - int / float           → returned as float
37:        - "1712345678"          → parsed as float
38:        - "2026-04-11T..."      → parsed via datetime.fromisoformat
39:        - anything else / fail  → 0.0 (treated as ancient → cache miss)
40:
41:    Returning 0 on failure is safe: it forces a cache miss, which costs
42:    one extra API call but never silently breaks the on-chain voter.
43:    """
44:    if isinstance(value, (int, float)):
45:        return float(value)
46:    if isinstance(value, str) and value:
47:        try:
48:            return float(value)
49:        except ValueError:
50:            pass
51:        try:
52:            # Handle "Z" suffix for UTC zulu time
53:            iso = value.replace("Z", "+00:00")
54:            return datetime.fromisoformat(iso).timestamp()
55:        except (ValueError, TypeError):
56:            pass
57:    # P1-14 (2026-05-02): defensive fall-through. Returning 0.0 silently
58:    # forces a cache miss, which costs one extra API call but never breaks
59:    # the on-chain voter. Log at DEBUG so operators investigating "why is
60:    # the BGeometrics 15-req/day budget burning every restart?" can see
61:    # this firing in the logs without breaking the safe-default behaviour.
62:    logger.debug(
63:        "_coerce_epoch: unparseable value type=%s repr=%.50r — returning 0.0",
64:        type(value).__name__,
65:        value,
66:    )
67:    return 0.0
68:
69:BASE_DIR = Path(__file__).resolve().parent.parent
70:DATA_DIR = BASE_DIR / "data"
71:CACHE_FILE = DATA_DIR / "onchain_cache.json"
72:
73:API_BASE = "https://bitcoin-data.com"
74:ONCHAIN_TTL = 43200  # 12 hours
75:
76:
77:# ---------------------------------------------------------------------------
78:# Config
79:# ---------------------------------------------------------------------------
80:
81:def _load_config_token():
82:    """Load BGeometrics API token from config.json."""
83:    try:
84:        config = _load_config()
85:        token = config.get("bgeometrics", {}).get("api_token", "")
86:        return token if token else None
87:    except Exception as e:
88:        logger.warning("BGeometrics token load failed: %s", e, exc_info=True)
89:        return None
90:
91:
92:# ---------------------------------------------------------------------------
93:# Persistent cache (survives restarts)
94:# ---------------------------------------------------------------------------
95:
96:def _save_onchain_cache(data):
97:    """Save on-chain data to persistent cache file."""
98:    try:
99:        # BUG-125: Use atomic_write_json to prevent corrupt cache on crash
100:        atomic_write_json(CACHE_FILE, data, ensure_ascii=False)
101:    except Exception:
102:        logger.warning("Failed to write onchain cache", exc_info=True)
103:
104:
105:def _load_onchain_cache(max_age_seconds=ONCHAIN_TTL):
106:    """Load on-chain data from persistent cache if fresh enough."""
107:    data = load_json(CACHE_FILE)
108:    if data is None:
109:        return None
110:    try:
111:        ts = data.get("ts", 0)
112:        if time.time() - ts > max_age_seconds:
113:            return None
114:        return data
115:    except Exception as e:
116:        logger.warning("On-chain cache load failed: %s", e, exc_info=True)
117:        return None
118:
119:
120:# ---------------------------------------------------------------------------
121:# Individual metric fetchers
122:# ---------------------------------------------------------------------------
123:
124:def _api_get(endpoint, token, params=None):
125:    """Make authenticated GET request to BGeometrics API.
126:
127:    Skips retries on failure since retrying just burns more of the
128:    8 req/hour free tier budget.
129:    """
130:    url = f"{API_BASE}{endpoint}"
131:    headers = {"Authorization": f"Bearer {token}"}
132:    return fetch_json(url, headers=headers, params=params, timeout=15,
133:                      retries=0, label=f"bgeometrics:{endpoint}")
134:
135:
136:def _fetch_mvrv(token):
137:    """Fetch latest MVRV and MVRV Z-Score."""
138:    data = _api_get("/v1/mvrv/last", token)
139:    if not data or not isinstance(data, dict):
140:        return None
141:    return {
142:        "mvrv": data.get("mvrv"),
143:        "mvrv_zscore": data.get("mvrvZScore"),
144:    }
145:
146:
147:def _fetch_sopr(token):
148:    """Fetch latest SOPR (Spent Output Profit Ratio)."""
149:    data = _api_get("/v1/sopr/last", token)
150:    if not data or not isinstance(data, dict):
151:        return None
152:    return {"sopr": data.get("sopr")}
153:
154:
155:def _fetch_nupl(token):
156:    """Fetch latest NUPL (Net Unrealized Profit/Loss)."""
157:    data = _api_get("/v1/nupl/last", token)
158:    if not data or not isinstance(data, dict):
159:        return None
160:    return {"nupl": data.get("nupl")}
161:
162:
163:def _fetch_realized_price(token):
164:    """Fetch latest realized price."""
165:    data = _api_get("/v1/realized-price/last", token)
166:    if not data or not isinstance(data, dict):
167:        return None
168:    return {"realized_price": data.get("realizedPrice")}
169:
170:
171:def _fetch_exchange_netflow(token):
172:    """Fetch latest exchange netflow (negative = accumulation)."""
173:    data = _api_get("/v1/exchange-netflow", token, params={"size": 1})
174:    if not data or not isinstance(data, list) or len(data) == 0:
175:        return None
176:    latest = data[0] if isinstance(data[0], dict) else data[-1]
177:    return {"netflow": latest.get("netflow")}
178:
179:
180:def _fetch_liquidations(token):
181:    """Fetch latest BTC liquidation data."""
182:    data = _api_get("/v1/btc-liquidations", token, params={"size": 1})
183:    if not data or not isinstance(data, list) or len(data) == 0:
184:        return None
185:    latest = data[0] if isinstance(data[0], dict) else data[-1]
186:    return {
187:        "long_liquidations": latest.get("longLiquidations"),
188:        "short_liquidations": latest.get("shortLiquidations"),
189:    }
190:
191:
192:# ---------------------------------------------------------------------------
193:# Main aggregator
194:# ---------------------------------------------------------------------------
195:
196:def _safe_float(val):
197:    """Convert API value to float, handling strings and None."""
198:    if val is None:
199:        return None
200:    try:
201:        return float(val)
202:    except (ValueError, TypeError):
203:        return None
204:
205:
206:def _fetch_all_onchain(token):
207:    """Fetch all 6 on-chain metrics and aggregate into a single dict.
208:
209:    Adds 1s delay between requests to respect free tier rate limits (8 req/hr).
210:    """
211:    result = {"ts": time.time()}
212:
213:    fetchers = [
214:        ("mvrv", _fetch_mvrv),
215:        ("sopr", _fetch_sopr),
216:        ("nupl", _fetch_nupl),
217:        ("realized_price", _fetch_realized_price),
218:        ("exchange_netflow", _fetch_exchange_netflow),
219:        ("liquidations", _fetch_liquidations),
220:    ]
221:
222:    any_success = False
223:    for i, (name, fetcher) in enumerate(fetchers):
224:        if i > 0:
225:            time.sleep(1)  # Rate limit: space out requests
226:        try:
227:            data = fetcher(token)
228:            if data:
229:                # Convert string values to float
230:                result.update({k: _safe_float(v) if k != "ts" else v
231:                              for k, v in data.items()})
232:                any_success = True
233:        except Exception:
234:            logger.warning("BGeometrics %s fetch failed", name, exc_info=True)
235:
236:    if not any_success:
237:        return None
238:
239:    # Save to persistent cache
240:    _save_onchain_cache(result)
241:    return result
242:
243:
244:def get_onchain_data():
245:    """Get on-chain data for BTC, using in-memory + persistent cache.
246:
247:    Returns dict with all available metrics, or None if unavailable.
248:    """
249:    # H12/DC-R3-5: On restart, seed the in-memory cache from the persistent cache
250:    # if it is still fresh. This avoids firing 6 BGeometrics API calls immediately
251:    # after every process restart (burns the 15 req/day budget).
252:    persistent = load_json(CACHE_FILE, default={})
253:    if persistent:
254:        # A-DE-5 (2026-04-11): Defensive timestamp parse. Older versions of
255:        # this cache stored "ts" as an ISO 8601 string instead of an epoch
256:        # float. The arithmetic on line below crashes with TypeError when
257:        # ts is a string, which then propagates up and silently disables
258:        # the on-chain BTC voter on the next restart. Detect and convert.
259:        raw_ts = persistent.get("ts", 0) or persistent.get("_fetched_at", 0)
260:        cache_ts = _coerce_epoch(raw_ts)
261:        if time.time() - cache_ts < ONCHAIN_TTL:
262:            # Still fresh — pre-populate the in-memory cache so _cached() returns
263:            # immediately without hitting the API.
264:            from portfolio.shared_state import _cache_lock, _tool_cache
265:            with _cache_lock:
266:                if "onchain_btc" not in _tool_cache:
267:                    _tool_cache["onchain_btc"] = {
268:                        "data": persistent,
269:                        "time": cache_ts,
270:                        "ttl": ONCHAIN_TTL,
271:                    }
272:                    logger.debug("On-chain: seeded in-memory cache from persistent (age %.0fs)",
273:                                 time.time() - cache_ts)
274:
275:    token = _load_config_token()
276:    if not token:
277:        # Try persistent cache even without token
278:        cached = _load_onchain_cache(max_age_seconds=ONCHAIN_TTL * 2)
279:        if cached:
280:            logger.debug("No BGeometrics token, using stale cache")
281:            return cached
282:        return None
283:
284:    return _cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)
285:
286:
287:# ---------------------------------------------------------------------------
288:# Interpretation helpers (for Layer 2 context)
289:# ---------------------------------------------------------------------------
290:
291:def interpret_onchain(data):
292:    """Interpret on-chain metrics into human-readable zones.
293:
294:    Returns dict with zone classifications for each available metric.
295:    """
296:    if not data:
297:        return {}
298:
299:    interp = {}
300:
301:    # MVRV Z-Score zones
302:    zscore = _safe_float(data.get("mvrv_zscore"))
303:    if zscore is not None:
304:        if zscore < 1:
305:            interp["mvrv_zone"] = "undervalued"
306:        elif zscore > 7:
307:            interp["mvrv_zone"] = "overheated"
308:        else:
309:            interp["mvrv_zone"] = "neutral"
310:
311:    # SOPR zones
312:    sopr = _safe_float(data.get("sopr"))
313:    if sopr is not None:
314:        if sopr < 0.97:
315:            interp["sopr_zone"] = "capitulation"
316:        elif sopr > 1.05:
317:            interp["sopr_zone"] = "profit_taking"
318:        else:
319:            interp["sopr_zone"] = "neutral"
320:
321:    # NUPL zones
322:    nupl = _safe_float(data.get("nupl"))
323:    if nupl is not None:
324:        if nupl < 0:
325:            interp["nupl_zone"] = "capitulation"
326:        elif nupl > 0.75:
327:            interp["nupl_zone"] = "euphoria"
328:        elif nupl > 0.5:
329:            interp["nupl_zone"] = "greed"
330:        elif nupl > 0.25:
331:            interp["nupl_zone"] = "optimism"
332:        else:
333:            interp["nupl_zone"] = "hope"
334:
335:    # Exchange netflow
336:    netflow = _safe_float(data.get("netflow"))
337:    if netflow is not None:
338:        if netflow < 0:
339:            interp["netflow_signal"] = "accumulation"
340:        else:
341:            interp["netflow_signal"] = "distribution"
342:
343:    return interp

2026-05-10T15:26:19.078573Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.9 seconds
Output:

 exited 1 in 899ms:
 succeeded in 921ms:
1:"""Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.
2:
3:Runs multiple sentiment models in shadow mode and logs results for comparison.
4:The primary (voting) model is the legacy model; shadow models are logged to
5:data/sentiment_ab_log.jsonl for accuracy tracking. Once a shadow model proves
6:superior (>60% on 200+ samples), it can be promoted.
7:
8:Models:
9:  - CryptoBERT: crypto headlines (legacy primary)
10:  - Trading-Hero-LLM: stock headlines (legacy primary)
11:  - FinGPT (Finance-Llama-8B / FinGPT-MT): GGUF via llama-cpp-python (shadow)
12:  - FinBERT: CPU fallback (shadow)
13:
14:Phase 3B: Cumulative headline clustering — groups related headlines and scores
15:them as a batch for richer "drumbeat effect" detection.
16:"""
17:
18:import json
19:import logging
20:import platform
21:import subprocess
22:import threading
23:from datetime import UTC, datetime
24:from pathlib import Path
25:
26:from portfolio.file_utils import atomic_append_jsonl
27:from portfolio.http_retry import fetch_json
28:
29:logger = logging.getLogger("portfolio.sentiment")
30:
31:CRYPTO_TICKERS = {"BTC", "ETH"}
32:
33:if platform.system() == "Windows":
34:    MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
35:    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
36:    TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
37:    FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
38:else:
39:    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
40:    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
41:    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
42:    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"
43:
44:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
45:AB_LOG_FILE = DATA_DIR / "sentiment_ab_log.jsonl"
46:
47:TICKER_CATEGORIES = {
48:    "BTC": "BTC",
49:    "ETH": "ETH",
50:    "XAU": "GOLD",
51:    "XAG": "SILVER",
52:    "PLTR": "TECHNOLOGY",
53:    "NVDA": "TECHNOLOGY",
54:    "AMD": "TECHNOLOGY",
55:    "GOOGL": "TECHNOLOGY",
56:    "AMZN": "TECHNOLOGY",
57:    "AAPL": "TECHNOLOGY",
58:    "AVGO": "TECHNOLOGY",
59:    "META": "TECHNOLOGY",
60:    "MU": "TECHNOLOGY",
61:    "SOUN": "TECHNOLOGY",
62:    "SMCI": "TECHNOLOGY",
63:    "TSM": "TECHNOLOGY",
64:    "TTWO": "TECHNOLOGY",
65:    "VRT": "TECHNOLOGY",
66:    "LMT": "TECHNOLOGY",
67:    "MSTR": "TECHNOLOGY",
68:}
69:
70:CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
71:
72:
73:def _is_crypto(ticker):
74:    return ticker.upper().replace("-USD", "") in CRYPTO_TICKERS
75:
76:
77:def _fetch_crypto_headlines(ticker="BTC", limit=20, *, cryptocompare_api_key=None):
78:    category = TICKER_CATEGORIES.get(ticker.upper(), ticker.upper())
79:    url = f"{CRYPTOCOMPARE_URL}&categories={category}"
80:    headers = {"User-Agent": "Mozilla/5.0"}
81:    if cryptocompare_api_key:
82:        headers["Authorization"] = f"Apikey {cryptocompare_api_key}"
83:    data = fetch_json(
84:        url,
85:        headers=headers,
86:        timeout=15,
87:        label="crypto_headlines",
88:    )
89:    if data is None:
90:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
91:    if isinstance(data, dict) and data.get("Response") == "Error":
92:        logger.warning("[CryptoCompare] API error: %s", data.get("Message", "unknown"))
93:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
94:    raw = data.get("Data", [])
95:    articles = list(raw)[:limit] if isinstance(raw, list) else []
96:    parsed = [
97:        {
98:            "title": a["title"],
99:            "source": a.get("source", "unknown"),
100:            "published": datetime.fromtimestamp(
101:                a["published_on"], tz=UTC
102:            ).isoformat(),
103:        }
104:        for a in articles
105:    ]
106:    if not parsed:
107:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
108:    return parsed
109:
110:
111:# Mapping from short crypto ticker to yfinance symbol for fallback
112:_CRYPTO_YFINANCE_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
113:
114:
115:def _fetch_crypto_headlines_yahoo_fallback(ticker, limit=20):
116:    """Fallback: fetch crypto headlines via yfinance when CryptoCompare fails."""
117:    yf_symbol = _CRYPTO_YFINANCE_MAP.get(ticker.upper())
118:    if not yf_symbol:
119:        return []
120:    try:
121:        articles = _fetch_yahoo_headlines(yf_symbol, limit=limit)
122:        if articles:
123:            logger.info("[CryptoCompare] fallback to Yahoo Finance for %s: %d articles",
124:                        ticker, len(articles))
125:        return articles
126:    except Exception as e:
127:        logger.debug("[Yahoo News] crypto fallback error for %s: %s", ticker, e)
128:        return []
129:
130:
131:def _fetch_yahoo_headlines(ticker, limit=10):
132:    import yfinance as yf
133:
134:    stock = yf.Ticker(ticker)
135:    news = stock.news or []
136:    articles = []
137:    for item in news[:limit]:
138:        content = item.get("content", item)
139:        title = content.get("title", "")
140:        if not title:
141:            continue
142:        pub = content.get("pubDate") or content.get("displayTime", "")
143:        provider = content.get("provider", {})
144:        source = (
145:            provider.get("displayName", "Yahoo Finance")
146:            if isinstance(provider, dict)
147:            else "Yahoo Finance"
148:        )
149:        articles.append(
150:            {
151:                "title": title,
152:                "source": source,
153:                "published": pub or datetime.now(UTC).isoformat(),
154:            }
155:        )
156:    return articles
157:
158:
159:def _fetch_newsapi_headlines(ticker, api_key, limit=10, query=None):
160:    """Fetch headlines from NewsAPI with optional custom search query."""
161:    search_q = query or ticker
162:    data = fetch_json(
163:        "https://newsapi.org/v2/everything",
164:        params={"q": search_q, "language": "en", "sortBy": "publishedAt",
165:                "pageSize": limit},
166:        headers={"User-Agent": "Mozilla/5.0", "X-Api-Key": api_key},
167:        timeout=15,
168:        label=f"newsapi:{ticker}",
169:    )
170:    if data is None:
171:        return []
172:    articles = data.get("articles", [])
173:    return [
174:        {
175:            "title": a.get("title", ""),
176:            "source": a.get("source", {}).get("name", "unknown"),
177:            "published": a.get("publishedAt", datetime.now(UTC).isoformat()),
178:        }
179:        for a in articles
180:        if a.get("title")
181:    ]
182:
183:
184:def _fetch_newsapi_with_tracking(ticker, api_key, limit=10, query=None):
185:    """Fetch from NewsAPI and track the call against daily quota.
186:
187:    H9/DC-R3-2: only count against budget when the fetch actually returned data
188:    (not on empty responses or errors), preventing spurious budget exhaustion.
189:    """
190:    from portfolio.shared_state import newsapi_track_call
191:    result = _fetch_newsapi_headlines(ticker, api_key, limit=limit, query=query)
192:    if result:  # only count against budget when we actually got data
193:        newsapi_track_call()
194:    return result
195:
196:
197:def _fetch_stock_headlines(ticker, newsapi_key=None, limit=20):
198:    """Fetch stock headlines. NewsAPI for priority tickers (metals), Yahoo for the rest."""
199:    from portfolio.shared_state import (
200:        _cached,
201:        newsapi_quota_ok,
202:        newsapi_search_query,
203:        newsapi_ttl_for_ticker,
204:    )
205:
206:    articles = []
207:
208:    # NewsAPI: only for priority tickers during active hours (metals get 20-min refresh)
209:    ttl = newsapi_ttl_for_ticker(ticker) if newsapi_key else None
210:    if ttl is not None and newsapi_key and newsapi_quota_ok():
211:        try:
212:            query = newsapi_search_query(ticker)
213:            cached_newsapi = _cached(
214:                f"newsapi_{ticker}",
215:                ttl,
216:                _fetch_newsapi_with_tracking,
217:                ticker,
218:                newsapi_key,
219:                limit,
220:                query,
221:            )
222:            if cached_newsapi:
223:                articles.extend(cached_newsapi)
224:        except Exception as e:
225:            logger.debug("[NewsAPI] error for %s: %s", ticker, e)
226:
227:    # Yahoo Finance: fallback for metals, primary for everything else
228:    if len(articles) < limit:
229:        try:
230:            remaining = max(0, limit - len(articles))
231:            if remaining > 0:
232:                yahoo_articles = _fetch_yahoo_headlines(ticker, limit=remaining)
233:                seen_titles = {a.get("title", "").lower() for a in articles}
234:                for ya in yahoo_articles:
235:                    if ya.get("title", "").lower() not in seen_titles:
236:                        articles.append(ya)
237:                        seen_titles.add(ya.get("title", "").lower())
238:        except Exception as e:
239:            logger.debug("[Yahoo News] error for %s: %s", ticker, e)
240:
241:    newsapi_count = len([a for a in articles if a.get("source", "") != "Yahoo Finance"])
242:    yahoo_count = len(articles) - newsapi_count
243:    if articles:
244:        logger.debug("[Headlines %s] %d NewsAPI + %d Yahoo = %d total",
245:                     ticker, newsapi_count, yahoo_count, len(articles))
246:
247:    return articles[:limit]
248:
249:
250:# 2026-04-09 (fix/bert-inproc-gpu): map subprocess script paths to in-process
251:# model names so _run_model can try the fast in-process path first and fall
252:# back to the old subprocess path on failure. See portfolio/bert_sentiment.py
253:# for the full rationale — short version: subprocess cold-load was ~3-10s per
254:# call, in-process on GPU is ~50-200ms per call, ~20-60x speedup with the
255:# same output shape.
256:_INPROC_BERT_MAP = {
257:    CRYPTOBERT_SCRIPT: "CryptoBERT",
258:    TRADING_HERO_SCRIPT: "Trading-Hero-LLM",
259:    FINBERT_SCRIPT: "FinBERT",
260:}
261:
262:
263:def _run_model(script, texts):
264:    """Run a sentiment model.
265:
266:    Tries the in-process BERT cache first (portfolio.bert_sentiment) because
267:    it avoids the ~3-10 s subprocess spawn + cold-load cost and runs on GPU
268:    if available. Falls back to the legacy subprocess pattern on any failure
269:    so the main loop stays up even if torch/transformers break or a model
270:    cache dir is missing.
271:    """
272:    model_name = _INPROC_BERT_MAP.get(script)
273:    if model_name is not None:
274:        try:
275:            from portfolio.bert_sentiment import predict as _bert_predict
276:            return _bert_predict(model_name, texts)
277:        except Exception as e:
278:            # Log once per (model, exception class) to keep the log clean if
279:            # we end up stuck on the subprocess fallback. sentiment.py already
280:            # has its own logger configured.
281:            logger.warning(
282:                "In-process BERT %s failed, falling back to subprocess: %s",
283:                model_name, e,
284:            )
285:
286:    # Legacy subprocess path (also used if script is not one of the three
287:    # known BERT models, though that doesn't happen today).
288:    proc = subprocess.run(
289:        [MODELS_PYTHON, script],
290:        input=json.dumps(texts),
291:        capture_output=True,
292:        text=True,
293:        timeout=120,
294:    )
295:    if proc.returncode != 0:
296:        raise RuntimeError(f"Model failed: {proc.stderr}")
297:    return json.loads(proc.stdout)
298:
299:
300:# ── Deferred fingpt A/B buffering ──────────────────────────────────────────
301:# Fingpt is a SHADOW sentiment signal — it never votes. Its output lands in
302:# data/sentiment_ab_log.jsonl alongside the primary model's vote (CryptoBERT
303:# for crypto, Trading-Hero-LLM for stocks) for accuracy comparison.
304:#
305:# Historical note (2026-04-09, feat/fingpt-in-llmbatch):
306:# - v1: inline subprocess per call, cold-loading the GGUF every time (70-90s
307:#   GPU lock holds, broke the cycle budget)
308:# - v2: warm NDJSON daemon (scripts/fingpt_daemon.py) on GPU full offload
309:#   (OOM'd with llama-server also resident)
310:# - v3: warm NDJSON daemon on CPU (60-150s/cycle inference, forced
311:#   _FINGPT_REQUEST_TIMEOUT_S 60→180 and _TICKER_POOL_TIMEOUT 120→500 hotfix
312:#   bumps; worked but ugly)
313:# - v4 (current): fingpt runs in portfolio.llm_batch as Phase 3 of the
314:#   shared llama_server rotation on port 8787, with full -ngl 99 GPU
315:#   offload like ministral3 and qwen3. Retires ~250 LOC of daemon +
316:#   client code.
317:#
318:# Because fingpt now runs post-cycle in a batched phase, its results arrive
319:# AFTER get_sentiment() has already returned to the signal engine. The
320:# primary model and FinBERT shadow are still computed inline, but their A/B
321:# log write is DEFERRED: get_sentiment() stashes the primary + finbert shadow
322:# + the raw headlines + the enqueued fingpt sub_keys into
323:# _pending_ab_entries[ab_key] and returns. Once flush_llm_batch() completes
324:# in main.py, sentiment.flush_ab_log() walks the pending entries, merges the
325:# batched fingpt results into each, and writes the final A/B log rows.
326:#
327:# This preserves the EXACT schema of sentiment_ab_log.jsonl that downstream
328:# accuracy tracking consumes: one row per get_sentiment() call, with a
329:# shadow[] array containing fingpt per-headline + cumulative + finbert.
330:#
331:# Known open issue: sentiment_ab_log.jsonl shows fingpt returning constant
332:# "neutral, 0.7 confidence" for every real headline — see the
333:# project_fingpt_parser_defaulting_neutral memory. That is a parser / prompt
334:# bug in /mnt/q/models/fingpt_infer.py, NOT a problem with this migration.
335:# Scheduled as the immediate follow-up after this PR merges.
336:
337:_pending_ab_entries: dict[str, dict] = {}
338:_pending_ab_lock = threading.Lock()
339:
340:
341:def _stash_ab_context(
342:    ab_key: str,
343:    ticker: str,
344:    primary_result: dict,
345:    all_articles: list[dict],
346:    diss_mult: float,
347:) -> None:
348:    """Store the inline portion of an A/B entry until the batched fingpt
349:    results arrive in flush_ab_log(). Called from get_sentiment().
350:
351:    Thread-safe — multiple ThreadPoolExecutor workers call this concurrently.
352:
353:    2026-04-28: cryptobert_shadow slot added; CryptoBERT was demoted from
354:    crypto primary to shadow. See get_sentiment docstring for rationale.
355:    """
356:    with _pending_ab_lock:
357:        _pending_ab_entries[ab_key] = {
358:            "ticker": ticker,
359:            "primary_result": primary_result,
360:            "finbert_shadow": None,  # filled in below by get_sentiment
361:            "cryptobert_shadow": None,  # filled in below for crypto tickers (2026-04-28)
362:            "all_articles": all_articles,
363:            "diss_mult": diss_mult,
364:            "fingpt_headlines_raw": None,  # filled in by Phase 3
365:            "fingpt_cumulatives_raw": {},  # sub_key → raw dict, filled in by Phase 3
366:        }
367:
368:
369:def _stash_finbert_shadow(ab_key: str, finbert_shadow: dict | None) -> None:
370:    """Attach the inline FinBERT shadow result to a pending A/B entry."""
371:    with _pending_ab_lock:
372:        entry = _pending_ab_entries.get(ab_key)
373:        if entry is not None:
374:            entry["finbert_shadow"] = finbert_shadow
375:
376:
377:def _stash_cryptobert_shadow(ab_key: str, cryptobert_shadow: dict | None) -> None:
378:    """Attach the inline CryptoBERT shadow result to a pending A/B entry.
379:
380:    Added 2026-04-28 when CryptoBERT was demoted from crypto primary to
381:    shadow. Symmetric with _stash_finbert_shadow.
382:    """
383:    with _pending_ab_lock:
384:        entry = _pending_ab_entries.get(ab_key)
385:        if entry is not None:
386:            entry["cryptobert_shadow"] = cryptobert_shadow
387:
388:
389:def _stash_fingpt_result(ab_key: str, sub_key: str, result) -> None:
390:    """Called from portfolio.llm_batch._flush_fingpt_phase with the parsed
391:    fingpt result for one (ab_key, sub_key) tuple.
392:
393:    sub_key is either "headlines" (result is a list of per-headline dicts)
394:    or "cumul:<N>" (result is a single cumulative dict).
395:    """
396:    with _pending_ab_lock:
397:        entry = _pending_ab_entries.get(ab_key)
398:        if entry is None:
399:            # get_sentiment was never called for this key this cycle — can
400:            # happen if enqueue_fingpt ran but the parent get_sentiment
401:            # raised before _stash_ab_context. Drop silently.
402:            return
403:        if sub_key == "headlines":
404:            entry["fingpt_headlines_raw"] = result
405:        elif sub_key.startswith("cumul:"):
406:            entry["fingpt_cumulatives_raw"][sub_key] = result
407:
408:
409:def flush_ab_log() -> None:
410:    """Walk _pending_ab_entries, merge batched fingpt results into shadow
411:    arrays, write one JSONL row per entry, and clear the buffer.
412:
413:    Called once per cycle by main.py immediately after flush_llm_batch()
414:    finishes Phase 3. Safe to call even if some fingpt results are missing
415:    (the server returned None for that prompt) — those slots just get
416:    dropped from the shadow array, same as the daemon-era behavior of
417:    logging a fingpt:error entry.
418:
419:    Thread-safe: acquires _pending_ab_lock for the entry snapshot, then
420:    clears the buffer under the same lock so no subsequent cycle can see
421:    leftover state.
422:    """
423:    with _pending_ab_lock:
424:        entries_snapshot = dict(_pending_ab_entries)
425:        _pending_ab_entries.clear()
426:
427:    if not entries_snapshot:
428:        return
429:
430:    for ab_key, entry in entries_snapshot.items():
431:        try:
432:            shadow: list[dict] = []
433:
434:            # Fingpt per-headline → aggregate via _aggregate_sentiments the
435:            # same way the old inline path did. If the raw list is missing
436:            # (server returned nothing), skip the entry silently.
437:            fingpt_raw = entry.get("fingpt_headlines_raw")
438:            if fingpt_raw:
439:                # Filter out None entries (per-prompt failures).
440:                usable = [r for r in fingpt_raw if r is not None]
441:                if usable:
442:                    try:
443:                        fg_overall, fg_avg = _aggregate_sentiments(
444:                            usable,
445:                            headlines=entry["all_articles"],
446:                            dissemination_mult=entry.get("diss_mult", 1.0),
447:                        )
448:                        shadow.append({
449:                            "model": usable[0].get("model", "fingpt:finance-llama-8b"),
450:                            "sentiment": fg_overall,
451:                            "confidence": round(fg_avg[fg_overall], 4),
452:                            "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
453:                        })
454:                    except Exception:
455:                        logger.debug(
456:                            "fingpt headlines aggregation failed for %s", ab_key,
457:                            exc_info=True,
458:                        )
459:
460:            # Fingpt cumulative clusters → one shadow entry per cluster.
461:            for _sub_key in sorted(entry.get("fingpt_cumulatives_raw", {})):
462:                cum = entry["fingpt_cumulatives_raw"][_sub_key]
463:                if cum is None:
464:                    continue
465:                shadow.append({
466:                    "model": cum.get("model", "fingpt:cumulative"),
467:                    "sentiment": cum.get("sentiment", "neutral"),
468:                    "confidence": cum.get("confidence", 0.0),
469:                    "headline_count": cum.get("headline_count", 0),
470:                })
471:
472:            # FinBERT shadow (already aggregated inline during get_sentiment).
473:            finbert = entry.get("finbert_shadow")
474:            if finbert is not None:
475:                shadow.append(finbert)
476:
477:            # CryptoBERT shadow (added 2026-04-28 — was the primary; demoted
478:            # to shadow due to 99.1% neutral output on press-wire input).
479:            # Crypto tickers only; entry stays None for stocks.
480:            cryptobert = entry.get("cryptobert_shadow")
481:            if cryptobert is not None:
482:                shadow.append(cryptobert)
483:
484:            if shadow:
485:                _log_ab_result(entry["ticker"], entry["primary_result"], shadow)
486:        except Exception:
487:            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)
488:
489:
490:def _run_finbert(texts):
491:    """Run FinBERT sentiment inference.
492:
493:    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
494:    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
495:    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
496:    first, falling back to the old subprocess path on any exception.
497:    """
498:    return _run_model(FINBERT_SCRIPT, texts)
499:
500:
501:# ---------------------------------------------------------------------------
502:# Headline clustering (Phase 3B)
503:# ---------------------------------------------------------------------------
504:
505:def _cluster_headlines(articles):
506:    """Group headlines by keyword overlap and time proximity.
507:
508:    Clusters enable cumulative sentiment analysis — many mildly negative
509:    headlines about the same topic together signal stronger negativity
510:    than scoring each independently.
511:
512:    Returns:
513:        List of clusters, each a list of article dicts
514:    """
515:    if not articles or len(articles) < 3:
516:        return [articles] if articles else []
517:
518:    from portfolio.news_keywords import score_headline
519:
520:    # Extract keywords per headline
521:    headline_keywords = []
522:    for a in articles:
523:        _, matched = score_headline(a.get("title", ""))
524:        # Also extract significant words (>4 chars, not stopwords)
525:        words = set()
526:        for word in a.get("title", "").lower().split():
527:            clean = word.strip(".,!?;:'\"()[]")
528:            if len(clean) > 4 and clean not in _STOPWORDS:
529:                words.add(clean)
530:        words.update(kw.lower() for kw in matched)
531:        headline_keywords.append(words)
532:
533:    # Simple greedy clustering by keyword overlap
534:    clusters = []
535:    assigned = set()
536:
537:    for i in range(len(articles)):
538:        if i in assigned:
539:            continue
540:        cluster = [articles[i]]
541:        assigned.add(i)
542:        kw_i = headline_keywords[i]
543:
544:        for j in range(i + 1, len(articles)):
545:            if j in assigned:
546:                continue
547:            kw_j = headline_keywords[j]
548:            overlap = len(kw_i & kw_j)
549:            # Cluster if they share 2+ keywords or 1 matched keyword
550:            if overlap >= 2 or (overlap >= 1 and kw_i & kw_j & _SIGNIFICANT_KEYWORDS):
551:                cluster.append(articles[j])
552:                assigned.add(j)
553:
554:        clusters.append(cluster)
555:
556:    return clusters
557:
558:
559:_STOPWORDS = {
560:    "about", "after", "again", "being", "between", "could", "during",
561:    "every", "first", "their", "there", "these", "those", "under",
562:    "which", "while", "would", "other", "still", "where", "before",
563:    "should", "since", "until", "years", "might", "price", "stock",
564:    "market", "shares", "today", "report", "quarter",
565:}
566:
567:_SIGNIFICANT_KEYWORDS = {
568:    "tariff", "tariffs", "war", "crash", "sanctions", "hack", "recession",
569:    "inflation", "rate", "cut", "hike", "layoffs", "earnings", "fomc",
570:    "bitcoin", "ethereum", "crypto", "nvidia", "semiconductor",
571:}
572:
573:
574:# ---------------------------------------------------------------------------
575:# Aggregation
576:# ---------------------------------------------------------------------------
577:
578:# Decisiveness margins added 2026-04-28. See docs/PLAN_sentiment_2026_04_28.md
579:# for the full rationale; tl;dr: the old aggregator returned the top label by
580:# 0.001 vs second, so a 0.34/0.33/0.33 split labeled "positive". Now we require
581:# a real margin before committing to a non-neutral verdict, and we default to
582:# label-majority over score-averaging so a few decisive headlines are not
583:# drowned by many tepid-neutral peers.
584:_DECISIVE_MARGIN_AVG = 0.05      # avg-mode: top-vs-second margin in prob units
585:_DECISIVE_MARGIN_PER_HEADLINE = 0.10  # majority-mode: per-headline label margin
586:_DECISIVE_MARGIN_MAJORITY = 1e-9  # majority-mode: top-vs-second weight margin
587:                                  # (zero-tolerance — exact ties go neutral)
588:
589:
590:def _compute_weights(sentiments, headlines, dissemination_mult):
591:    """Return per-sentiment weights from keyword scoring + dissemination."""
592:    if headlines and len(headlines) == len(sentiments):
593:        from portfolio.news_keywords import score_headline
594:        weights = []
595:        for h in headlines:
596:            title = h.get("title", "") if isinstance(h, dict) else str(h)
597:            w, _ = score_headline(title)
598:            weights.append(w * dissemination_mult)
599:    else:
600:        weights = [dissemination_mult] * len(sentiments)
601:    return weights
602:
603:
604:def _aggregate_sentiments(sentiments, headlines=None, dissemination_mult=1.0,
605:                           *, mode="majority"):
606:    """Aggregate sentiment scores into a single (label, avg_dict) verdict.
607:
608:    mode="majority" (default, 2026-04-28): label-majority vote.
609:        Each headline gets its own decisive label (top score must beat second
610:        by >=_DECISIVE_MARGIN_PER_HEADLINE, else "neutral"), then a weighted
611:        majority over those labels picks the verdict. Exact ties resolve to
612:        neutral. The returned avg_dict is still the score-weighted-average
613:        (kept identical for backward-compat with consumers that read
614:        avg_scores like sentiment_avg_scores in signal_engine.py:2452).
615:
616:    mode="average": legacy probability-averaging. Returns the top-scored
617:        label IF its margin over the second exceeds _DECISIVE_MARGIN_AVG;
618:        otherwise downgrades to "neutral". The pure-max-without-margin
619:        behavior was the source of the W16-W17 sentiment regression and is
620:        no longer reachable.
621:
622:    When headlines are provided, score weights from news_keywords.score_headline()
623:    amplify high-impact keywords (tariff/war/crash 3x). dissemination_mult
624:    multiplies all weights when news is widely cross-referenced.
625:    """
626:    if not sentiments:
627:        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
628:
629:    weights = _compute_weights(sentiments, headlines, dissemination_mult)
630:    total_w = sum(weights)
631:    if total_w == 0:
632:        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
633:
634:    pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
635:    neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
636:    neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
637:    avg = {
638:        "positive": pos_sum / total_w,
639:        "negative": neg_sum / total_w,
640:        "neutral":  neu_sum / total_w,
641:    }
642:
643:    if mode == "majority":
644:        verdict = _majority_label(sentiments, weights)
645:        return verdict, avg
646:
647:    # mode == "average" — legacy threshold-augmented score-averaging
648:    overall = max(avg, key=avg.get)
649:    sorted_scores = sorted(avg.values(), reverse=True)
650:    margin = sorted_scores[0] - sorted_scores[1]
651:    if margin < _DECISIVE_MARGIN_AVG and overall != "neutral":
652:        return "neutral", avg
653:    return overall, avg
654:
655:
656:def _majority_label(sentiments, weights):
657:    """Per-headline decisive label, then weighted majority vote.
658:
659:    Each headline classified as positive/negative/neutral with its own per-
660:    headline margin gate. Weights are summed per label; winner returned only
661:    if it beats the second by more than _DECISIVE_MARGIN_MAJORITY.
662:    """
663:    bucket = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
664:    for s, w in zip(sentiments, weights):
665:        scores = s["scores"]
666:        # decisive per-headline label: top must beat second by margin
667:        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
668:        top_label, top_score = ordered[0]
669:        second_score = ordered[1][1]
670:        if (top_score - second_score) < _DECISIVE_MARGIN_PER_HEADLINE:
671:            top_label = "neutral"
672:        bucket[top_label] += w
673:
674:    ordered_buckets = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
675:    winner_label, winner_w = ordered_buckets[0]
676:    runner_w = ordered_buckets[1][1]
677:    if (winner_w - runner_w) <= _DECISIVE_MARGIN_MAJORITY:
678:        return "neutral"
679:    return winner_label
680:
681:
682:def _filter_relevant_headlines(articles, ticker, *, fallback_n=3):
683:    """Drop wire-noise headlines before model inference.
684:
685:    Uses news_keywords.is_relevant_headline (keyword OR ticker-synonym match)
686:    plus a credible-source-with-long-title escape hatch (Reuters/Bloomberg/
687:    etc. + title >= 25 chars covers in-depth coverage that doesn't happen to
688:    mention the ticker by name).
689:
690:    Falls back to most-recent `fallback_n` if the filter would drop
691:    everything — better to have noisy signal than silent signal on slow
692:    news days.
693:    """
694:    if not articles:
695:        return []
696:
697:    from portfolio.news_keywords import is_credible_source, is_relevant_headline
698:
699:    kept = []
700:    for a in articles:
701:        title = a.get("title", "") if isinstance(a, dict) else str(a)
702:        if is_relevant_headline(title, ticker):
703:            kept.append(a)
704:            continue
705:        # Credible-source escape hatch: long titles from credible outlets are
706:        # almost always real coverage worth scoring.
707:        source = a.get("source", "") if isinstance(a, dict) else ""
708:        if is_credible_source(source) and len(title.strip()) >= 25:
709:            kept.append(a)
710:
711:    if kept:
712:        return kept
713:
714:    # All-irrelevant fallback: keep the most-recent N articles. Sort by the
715:    # `published` field (ISO timestamp string sorts chronologically), most
716:    # recent first. Articles without `published` sort last via empty-string
717:    # default.
718:    sorted_articles = sorted(
719:        articles,
720:        key=lambda a: a.get("published", "") if isinstance(a, dict) else "",
721:        reverse=True,
722:    )
723:    return sorted_articles[:fallback_n]
724:
725:
726:def _log_ab_result(ticker, primary_result, shadow_results):
727:    """Log A/B test results to sentiment_ab_log.jsonl for accuracy comparison."""
728:    try:
729:        entry = {
730:            "ts": datetime.now(UTC).isoformat(),
731:            "ticker": ticker,
732:            "primary": {
733:                "model": primary_result.get("model", "unknown"),
734:                "sentiment": primary_result.get("overall_sentiment", "unknown"),
735:                "confidence": primary_result.get("confidence", 0.0),
736:            },
737:            "shadow": shadow_results,
738:        }
739:        atomic_append_jsonl(AB_LOG_FILE, entry)
740:    except Exception:
741:        logger.debug("Failed to log A/B result", exc_info=True)
742:
743:
744:# ---------------------------------------------------------------------------
745:# Main entry point
746:# ---------------------------------------------------------------------------
747:
748:def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
749:                   *, cryptocompare_api_key=None) -> dict:
750:    """Get sentiment for a ticker using primary model + shadow A/B models.
751:
752:    2026-04-28 (fix/sentiment-relevance-and-aggregation): two changes here.
753:      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
754:         CryptoBERT was 99.1% neutral on 2,817 wire-feed samples (it was
755:         trained on crypto-twitter slang, not press-wire headlines).
756:         CryptoBERT now runs as a shadow for continuity of the 30d accuracy
757:         baseline. Trading-Hero is permabull on financial news but at least
758:         produces variance; the 47% directional accuracy gate still acts as
759:         a circuit breaker if it underperforms.
760:      2. Headlines pass through _filter_relevant_headlines() before model
761:         inference. Bare price-tickers ("Bitcoin: $67,123") and generic
762:         market-update boilerplate are dropped. Fallback keeps the most-
763:         recent N when the filter would drop everything.
764:
765:    Primary model (votes):  Trading-Hero-LLM (both crypto and stocks)
766:    Shadow models (logged): CryptoBERT (crypto only), FinGPT, FinBERT
767:
768:    Returns the primary model's result. Shadow results are logged to
769:    data/sentiment_ab_log.jsonl for accuracy tracking.
770:    """
771:    short = ticker.upper().replace("-USD", "")
772:    is_crypto = _is_crypto(short)
773:
774:    if is_crypto:
775:        articles = _fetch_crypto_headlines(
776:            short, cryptocompare_api_key=cryptocompare_api_key,
777:        )
778:    else:
779:        articles = _fetch_stock_headlines(short, newsapi_key=newsapi_key)
780:    # 2026-04-28: Trading-Hero-LLM is the primary across all asset classes.
781:    # See module/function docstring for the CryptoBERT demotion rationale.
782:    model_script = TRADING_HERO_SCRIPT
783:    model_name = "Trading-Hero-LLM"
784:
785:    social = social_posts or []
786:    raw_all = articles + social
787:    sources = {
788:        "news": len(articles),
789:        "reddit": sum(1 for p in social if "reddit" in p.get("source", "")),
790:    }
791:
792:    if not raw_all:
793:        return {
794:            "overall_sentiment": "unknown",
795:            "confidence": 0.0,
796:            "num_articles": 0,
797:            "model": model_name,
798:            "sources": sources,
799:            "details": [],
800:        }
801:
802:    # Drop wire-noise before inference. The filter has a most-recent-N
803:    # fallback so we never go silent on slow-news days.
804:    all_articles = _filter_relevant_headlines(raw_all, short)
805:    titles = [a["title"] for a in all_articles]
806:
807:    # Compute dissemination score for weight amplification
808:    diss_mult = 1.0
809:    try:
810:        from portfolio.news_keywords import dissemination_score
811:        diss_mult = dissemination_score(all_articles)
812:    except Exception:
813:        logger.debug("Dissemination score failed, using default 1.0", exc_info=True)
814:
815:    # --- Primary model (votes in consensus) ---
816:    sentiments = _run_model(model_script, titles)
817:    overall, avg = _aggregate_sentiments(sentiments, headlines=all_articles,
818:                                         dissemination_mult=diss_mult)
819:
820:    details = []
821:    for article, sent in zip(all_articles, sentiments):
822:        details.append(
823:            {
824:                "title": article["title"],
825:                "source": article["source"],
826:                "published": article["published"],
827:                "sentiment": sent.get("sentiment") or sent.get("label", "unknown"),
828:                "confidence": sent["confidence"],
829:                "scores": sent["scores"],
830:            }
831:        )
832:
833:    primary_result = {
834:        "overall_sentiment": overall,
835:        "confidence": round(avg[overall], 4),
836:        "num_articles": len(sentiments),
837:        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
838:        "model": model_name,
839:        "sources": sources,
840:        "details": details,
841:        "dissemination_score": diss_mult,
842:    }
843:
844:    # --- Shadow models (A/B testing — logged only, don't affect consensus) ---
845:    #
846:    # 2026-04-09: The A/B log write used to happen inline at the bottom of
847:    # this function. It is now DEFERRED to flush_ab_log() which is called
848:    # post-cycle from main.py after flush_llm_batch() completes Phase 3
849:    # (fingpt sentiment). Rationale: fingpt used to run in a bespoke NDJSON
850:    # daemon (scripts/fingpt_daemon.py, now retired) blocking inside this
851:    # function; moving fingpt into portfolio.llm_batch's shared llama_server
852:    # rotation means the fingpt result does not arrive until AFTER
853:    # get_sentiment() has returned. Rather than duplicate the A/B log entry
854:    # or block on the batch, we stash the primary + finbert + context here
855:    # and let flush_ab_log() assemble the final row.
856:    #
857:    # The primary model's voting result is still computed and returned
858:    # SYNCHRONOUSLY — batching only affects the shadow log, not the vote.
859:    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
860:    _stash_ab_context(ab_key, short, primary_result, all_articles, diss_mult)
861:
862:    # Shadow: FinGPT — enqueue for post-cycle Phase 3 execution. Zero-cost
863:    # here; the actual inference runs via llama_server finance-llama-8b
864:    # rotation after the ticker pool completes.
865:    #
866:    # 2026-04-10 (perf/llama-swap-reduction): gated by is_llm_on_cycle. Unlike
867:    # ministral/qwen3 (which go through _cached_or_enqueue's should_enqueue_fn),
868:    # fingpt enqueues directly because it doesn't use the signal cache — it
869:    # only writes to the A/B shadow log. When off-cycle, skip the enqueue
870:    # entirely so the llama_server phase 3 skips the fingpt model swap +
871:    # inference cost. Fingpt is a shadow signal, so skipping 2 of 3 cycles
872:    # just reduces A/B sample density from every cycle to every 3rd cycle,
873:    # which is fine for long-running statistical comparison.
874:    try:
875:        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
876:        if is_llm_on_cycle("fingpt"):
877:            enqueue_fingpt(
878:                ab_key, "headlines",
879:                {"mode": "headlines", "texts": titles, "ticker": short},
880:            )
881:            clusters = _cluster_headlines(all_articles)
882:            for idx, cluster in enumerate(clusters):
883:                if len(cluster) >= 3:
884:                    cluster_titles = [a["title"] for a in cluster]
885:                    enqueue_fingpt(
886:                        ab_key, f"cumul:{idx}",
887:                        {"mode": "cumulative", "texts": cluster_titles, "ticker": short},
888:                    )
889:    except Exception as e:
890:        logger.debug("FinGPT enqueue failed: %s", e)
891:
892:    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
893:    # and on CPU (no model swap cost) and we'd rather not add a fourth
894:    # phase to llm_batch for an already-shadow-of-shadow signal. Stash its
895:    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
896:    try:
897:        finbert_results = _run_finbert(titles)
898:        if finbert_results:
899:            fb_overall, fb_avg = _aggregate_sentiments(
900:                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
901:            )
902:            _stash_finbert_shadow(ab_key, {
903:                "model": "FinBERT",
904:                "sentiment": fb_overall,
905:                "confidence": round(fb_avg[fb_overall], 4),
906:                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
907:            })
908:    except Exception as e:
909:        logger.debug("FinBERT shadow failed: %s", e)
910:
911:    # Shadow: CryptoBERT — demoted from primary 2026-04-28. Kept as shadow
912:    # for crypto tickers only so we (a) preserve the 30d accuracy baseline
913:    # for comparison and (b) still notice if the model ever recovers from
914:    # its 99.1% neutral-output collapse. Stashed into the pending A/B
915:    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.
916:    if is_crypto:
917:        try:
918:            crypto_results = _run_model(CRYPTOBERT_SCRIPT, titles)
919:            if crypto_results:
920:                cb_overall, cb_avg = _aggregate_sentiments(
921:                    crypto_results, headlines=all_articles, dissemination_mult=diss_mult,
922:                )
923:                _stash_cryptobert_shadow(ab_key, {
924:                    "model": "CryptoBERT",
925:                    "sentiment": cb_overall,
926:                    "confidence": round(cb_avg[cb_overall], 4),
927:                    "avg_scores": {k: round(v, 4) for k, v in cb_avg.items()},
928:                })
929:        except Exception as e:
930:            logger.debug("CryptoBERT shadow failed: %s", e)
931:
932:    return primary_result
933:
934:
935:def get_crypto_sentiment(ticker="BTC") -> dict:
936:    return get_sentiment(ticker)
937:
938:
939:if __name__ == "__main__":
940:    for ticker in ["BTC", "ETH"]:
941:        print(f"\n{'='*60}")
942:        print(f"  Sentiment for {ticker}")
943:        print(f"{'='*60}")
944:        result = get_sentiment(ticker)
945:        print(
946:            f"Overall: {result['overall_sentiment']} "
947:            f"(confidence: {result['confidence']:.2%})"
948:        )
949:        print(f"Model: {result['model']}")
950:        print(f"Articles analyzed: {result['num_articles']}")
951:        print(f"Dissemination score: {result.get('dissemination_score', 1.0):.2f}")
952:        if result.get("avg_scores"):
953:            s = result["avg_scores"]
954:            print(
955:                f"Avg scores: pos={s['positive']:.3f} neg={s['negative']:.3f} neu={s['neutral']:.3f}"
956:            )
957:        print("\nTop headlines:")
958:        for d in result["details"][:5]:
959:            emoji = {"positive": "+", "negative": "-", "neutral": "~"}.get(
960:                d["sentiment"], "?"
961:            )
962:            print(
963:                f"  [{emoji}] {d['sentiment']:>8} ({d['confidence']:.2%}) {d['title']}"
964:            )

codex
I have the directly cited files. Next I’m drilling into the hidden dependencies behind the claims: `shared_state._cached`, `news_keywords.dissemination_score`, and whether `portfolio/shared_state.py` exists in this worktree at all.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Force portfolio | Select-Object Name,Length,Mode' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "def dissemination_score|def is_relevant_headline|def is_credible_source|def score_headline" portfolio/news_keywords.py' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "def _cached|newsapi_track_call|newsapi_quota_ok|newsapi_search_query|newsapi_ttl_for_ticker|yfinance_lock|_tool_cache|_cache_lock" -g "*.py"' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "shared_state" -g "*.py"' in Q:\finance-analyzer\.worktrees\adv-data-external
2026-05-10T15:26:33.720714Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 572ms:
 succeeded in 577ms:
139:def score_headline(title: str) -> tuple[float, list[str]]:
174:def is_credible_source(source: str) -> bool:
182:def dissemination_score(articles: list[dict]) -> float:
322:def is_relevant_headline(title: str, ticker: str) -> bool:

 succeeded in 622ms:

[32;1mName                [0m[32;1m Length[0m[32;1m Mode[0m
[32;1m----                [0m [32;1m------[0m [32;1m----[0m
alpha_vantage.py      10478 -a---
bert_sentiment.py     19504 -a---
crypto_macro_data.py  16564 -a---
data_collector.py     12435 -a---
earnings_calendar.py   7237 -a---
fear_greed.py          6928 -a---
futures_data.py        7095 -a---
fx_rates.py            3577 -a---
macro_context.py      13239 -a---
market_health.py      17149 -a---
news_keywords.py      12100 -a---
onchain_data.py       12057 -a---
sentiment.py          39408 -a---
social_sentiment.py    4718 -a---


2026-05-10T15:26:33.741800Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 594ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'portfolio/news_keywords.py'; "'$start=175; $end=240; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n shared_state portfolio' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'portfolio/sentiment.py'; "'$ranges=@(@(220,245),@(430,520),@(800,910)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Recurse -File | Where-Object { $_.Name -eq '"'shared_state.py' } | Select-Object -ExpandProperty FullName" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 568ms:
portfolio\macro_context.py:8:from portfolio.shared_state import VOLUME_TTL as _VOLUME_TTL
portfolio\macro_context.py:9:from portfolio.shared_state import _alpaca_limiter, _binance_limiter, _cached
portfolio\fx_rates.py:3:NOTE: Not migrated to shared_state._cached() because this module has custom
portfolio\onchain_data.py:24:from portfolio.shared_state import _cached
portfolio\onchain_data.py:264:            from portfolio.shared_state import _cache_lock, _tool_cache
portfolio\futures_data.py:12:from portfolio.shared_state import _binance_limiter, _cached
portfolio\sentiment.py:190:    from portfolio.shared_state import newsapi_track_call
portfolio\sentiment.py:199:    from portfolio.shared_state import (
portfolio\alpha_vantage.py:17:from portfolio.shared_state import _alpha_vantage_limiter
portfolio\data_collector.py:10:import portfolio.shared_state as _ss
portfolio\data_collector.py:275:# H11/DC-R3-4: use the module-level lock from shared_state so all modules
portfolio\data_collector.py:277:from portfolio.shared_state import yfinance_lock as _yfinance_lock
portfolio\earnings_calendar.py:39:        from portfolio.shared_state import _alpha_vantage_limiter
portfolio\market_health.py:17:from portfolio.shared_state import _cached
portfolio\fear_greed.py:130:    from portfolio.shared_state import yfinance_lock
portfolio\crypto_macro_data.py:23:from portfolio.shared_state import _cached

 succeeded in 612ms:
175:    """Check if a source name matches a credible financial news source."""
176:    if not source:
177:        return False
178:    lower = source.lower().strip()
179:    return any(cs in lower for cs in CREDIBLE_SOURCES)
180:
181:
182:def dissemination_score(articles: list[dict]) -> float:
183:    """Score how widely news has spread (FinGPT dissemination-aware pattern).
184:
185:    Factors:
186:    1. Unique source count — more sources = wider spread
187:    2. Source diversity — credible sources (Reuters, Bloomberg) weight more
188:    3. Time clustering — articles within 1h of each other = breaking news
189:
190:    Returns:
191:        Float multiplier (1.0 = normal, up to 3.0 for breaking news with wide coverage).
192:        Used to amplify headline weights in sentiment aggregation.
193:    """
194:    if not articles or len(articles) < 2:
195:        return 1.0
196:
197:    # Factor 1: Unique source count
198:    sources = set()
199:    for a in articles:
200:        src = a.get("source", "unknown").lower().strip()
201:        if src:
202:            sources.add(src)
203:    source_count = len(sources)
204:    # 1 source = 1.0, 3+ sources = 1.5, 5+ = 2.0
205:    source_factor = min(1.0 + (source_count - 1) * 0.25, 2.0)
206:
207:    # Factor 2: Source diversity — credible source presence
208:    credible_count = sum(1 for s in sources if any(cs in s for cs in CREDIBLE_SOURCES))
209:    diversity_factor = 1.0
210:    if credible_count >= 2:
211:        diversity_factor = 1.5
212:    elif credible_count >= 1:
213:        diversity_factor = 1.25
214:
215:    # Factor 3: Time clustering — articles within 1h of each other
216:    from datetime import datetime
217:    timestamps = []
218:    for a in articles:
219:        pub = a.get("published", "")
220:        if not pub:
221:            continue
222:        try:
223:            if isinstance(pub, (int, float)):
224:                ts = datetime.fromtimestamp(pub, tz=UTC)
225:            else:
226:                # Try ISO format
227:                pub_str = str(pub).replace("Z", "+00:00")
228:                ts = datetime.fromisoformat(pub_str)
229:            timestamps.append(ts.timestamp())
230:        except (ValueError, TypeError, OSError):
231:            continue
232:
233:    clustering_factor = 1.0
234:    if len(timestamps) >= 3:
235:        timestamps.sort()
236:        # Check if most articles appeared within a 1-hour window
237:        window = 3600  # 1 hour
238:        max_cluster = 1
239:        for i in range(len(timestamps)):
240:            cluster = sum(1 for t in timestamps if abs(t - timestamps[i]) <= window)

 succeeded in 633ms:
 succeeded in 663ms:
220:                query,
221:            )
222:            if cached_newsapi:
223:                articles.extend(cached_newsapi)
224:        except Exception as e:
225:            logger.debug("[NewsAPI] error for %s: %s", ticker, e)
226:
227:    # Yahoo Finance: fallback for metals, primary for everything else
228:    if len(articles) < limit:
229:        try:
230:            remaining = max(0, limit - len(articles))
231:            if remaining > 0:
232:                yahoo_articles = _fetch_yahoo_headlines(ticker, limit=remaining)
233:                seen_titles = {a.get("title", "").lower() for a in articles}
234:                for ya in yahoo_articles:
235:                    if ya.get("title", "").lower() not in seen_titles:
236:                        articles.append(ya)
237:                        seen_titles.add(ya.get("title", "").lower())
238:        except Exception as e:
239:            logger.debug("[Yahoo News] error for %s: %s", ticker, e)
240:
241:    newsapi_count = len([a for a in articles if a.get("source", "") != "Yahoo Finance"])
242:    yahoo_count = len(articles) - newsapi_count
243:    if articles:
244:        logger.debug("[Headlines %s] %d NewsAPI + %d Yahoo = %d total",
245:                     ticker, newsapi_count, yahoo_count, len(articles))
---
430:    for ab_key, entry in entries_snapshot.items():
431:        try:
432:            shadow: list[dict] = []
433:
434:            # Fingpt per-headline → aggregate via _aggregate_sentiments the
435:            # same way the old inline path did. If the raw list is missing
436:            # (server returned nothing), skip the entry silently.
437:            fingpt_raw = entry.get("fingpt_headlines_raw")
438:            if fingpt_raw:
439:                # Filter out None entries (per-prompt failures).
440:                usable = [r for r in fingpt_raw if r is not None]
441:                if usable:
442:                    try:
443:                        fg_overall, fg_avg = _aggregate_sentiments(
444:                            usable,
445:                            headlines=entry["all_articles"],
446:                            dissemination_mult=entry.get("diss_mult", 1.0),
447:                        )
448:                        shadow.append({
449:                            "model": usable[0].get("model", "fingpt:finance-llama-8b"),
450:                            "sentiment": fg_overall,
451:                            "confidence": round(fg_avg[fg_overall], 4),
452:                            "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
453:                        })
454:                    except Exception:
455:                        logger.debug(
456:                            "fingpt headlines aggregation failed for %s", ab_key,
457:                            exc_info=True,
458:                        )
459:
460:            # Fingpt cumulative clusters → one shadow entry per cluster.
461:            for _sub_key in sorted(entry.get("fingpt_cumulatives_raw", {})):
462:                cum = entry["fingpt_cumulatives_raw"][_sub_key]
463:                if cum is None:
464:                    continue
465:                shadow.append({
466:                    "model": cum.get("model", "fingpt:cumulative"),
467:                    "sentiment": cum.get("sentiment", "neutral"),
468:                    "confidence": cum.get("confidence", 0.0),
469:                    "headline_count": cum.get("headline_count", 0),
470:                })
471:
472:            # FinBERT shadow (already aggregated inline during get_sentiment).
473:            finbert = entry.get("finbert_shadow")
474:            if finbert is not None:
475:                shadow.append(finbert)
476:
477:            # CryptoBERT shadow (added 2026-04-28 — was the primary; demoted
478:            # to shadow due to 99.1% neutral output on press-wire input).
479:            # Crypto tickers only; entry stays None for stocks.
480:            cryptobert = entry.get("cryptobert_shadow")
481:            if cryptobert is not None:
482:                shadow.append(cryptobert)
483:
484:            if shadow:
485:                _log_ab_result(entry["ticker"], entry["primary_result"], shadow)
486:        except Exception:
487:            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)
488:
489:
490:def _run_finbert(texts):
491:    """Run FinBERT sentiment inference.
492:
493:    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
494:    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
495:    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
496:    first, falling back to the old subprocess path on any exception.
497:    """
498:    return _run_model(FINBERT_SCRIPT, texts)
499:
500:
501:# ---------------------------------------------------------------------------
502:# Headline clustering (Phase 3B)
503:# ---------------------------------------------------------------------------
504:
505:def _cluster_headlines(articles):
506:    """Group headlines by keyword overlap and time proximity.
507:
508:    Clusters enable cumulative sentiment analysis — many mildly negative
509:    headlines about the same topic together signal stronger negativity
510:    than scoring each independently.
511:
512:    Returns:
513:        List of clusters, each a list of article dicts
514:    """
515:    if not articles or len(articles) < 3:
516:        return [articles] if articles else []
517:
518:    from portfolio.news_keywords import score_headline
519:
520:    # Extract keywords per headline
---
800:        }
801:
802:    # Drop wire-noise before inference. The filter has a most-recent-N
803:    # fallback so we never go silent on slow-news days.
804:    all_articles = _filter_relevant_headlines(raw_all, short)
805:    titles = [a["title"] for a in all_articles]
806:
807:    # Compute dissemination score for weight amplification
808:    diss_mult = 1.0
809:    try:
810:        from portfolio.news_keywords import dissemination_score
811:        diss_mult = dissemination_score(all_articles)
812:    except Exception:
813:        logger.debug("Dissemination score failed, using default 1.0", exc_info=True)
814:
815:    # --- Primary model (votes in consensus) ---
816:    sentiments = _run_model(model_script, titles)
817:    overall, avg = _aggregate_sentiments(sentiments, headlines=all_articles,
818:                                         dissemination_mult=diss_mult)
819:
820:    details = []
821:    for article, sent in zip(all_articles, sentiments):
822:        details.append(
823:            {
824:                "title": article["title"],
825:                "source": article["source"],
826:                "published": article["published"],
827:                "sentiment": sent.get("sentiment") or sent.get("label", "unknown"),
828:                "confidence": sent["confidence"],
829:                "scores": sent["scores"],
830:            }
831:        )
832:
833:    primary_result = {
834:        "overall_sentiment": overall,
835:        "confidence": round(avg[overall], 4),
836:        "num_articles": len(sentiments),
837:        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
838:        "model": model_name,
839:        "sources": sources,
840:        "details": details,
841:        "dissemination_score": diss_mult,
842:    }
843:
844:    # --- Shadow models (A/B testing — logged only, don't affect consensus) ---
845:    #
846:    # 2026-04-09: The A/B log write used to happen inline at the bottom of
847:    # this function. It is now DEFERRED to flush_ab_log() which is called
848:    # post-cycle from main.py after flush_llm_batch() completes Phase 3
849:    # (fingpt sentiment). Rationale: fingpt used to run in a bespoke NDJSON
850:    # daemon (scripts/fingpt_daemon.py, now retired) blocking inside this
851:    # function; moving fingpt into portfolio.llm_batch's shared llama_server
852:    # rotation means the fingpt result does not arrive until AFTER
853:    # get_sentiment() has returned. Rather than duplicate the A/B log entry
854:    # or block on the batch, we stash the primary + finbert + context here
855:    # and let flush_ab_log() assemble the final row.
856:    #
857:    # The primary model's voting result is still computed and returned
858:    # SYNCHRONOUSLY — batching only affects the shadow log, not the vote.
859:    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
860:    _stash_ab_context(ab_key, short, primary_result, all_articles, diss_mult)
861:
862:    # Shadow: FinGPT — enqueue for post-cycle Phase 3 execution. Zero-cost
863:    # here; the actual inference runs via llama_server finance-llama-8b
864:    # rotation after the ticker pool completes.
865:    #
866:    # 2026-04-10 (perf/llama-swap-reduction): gated by is_llm_on_cycle. Unlike
867:    # ministral/qwen3 (which go through _cached_or_enqueue's should_enqueue_fn),
868:    # fingpt enqueues directly because it doesn't use the signal cache — it
869:    # only writes to the A/B shadow log. When off-cycle, skip the enqueue
870:    # entirely so the llama_server phase 3 skips the fingpt model swap +
871:    # inference cost. Fingpt is a shadow signal, so skipping 2 of 3 cycles
872:    # just reduces A/B sample density from every cycle to every 3rd cycle,
873:    # which is fine for long-running statistical comparison.
874:    try:
875:        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
876:        if is_llm_on_cycle("fingpt"):
877:            enqueue_fingpt(
878:                ab_key, "headlines",
879:                {"mode": "headlines", "texts": titles, "ticker": short},
880:            )
881:            clusters = _cluster_headlines(all_articles)
882:            for idx, cluster in enumerate(clusters):
883:                if len(cluster) >= 3:
884:                    cluster_titles = [a["title"] for a in cluster]
885:                    enqueue_fingpt(
886:                        ab_key, f"cumul:{idx}",
887:                        {"mode": "cumulative", "texts": cluster_titles, "ticker": short},
888:                    )
889:    except Exception as e:
890:        logger.debug("FinGPT enqueue failed: %s", e)
891:
892:    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
893:    # and on CPU (no model swap cost) and we'd rather not add a fourth
894:    # phase to llm_batch for an already-shadow-of-shadow signal. Stash its
895:    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
896:    try:
897:        finbert_results = _run_finbert(titles)
898:        if finbert_results:
899:            fb_overall, fb_avg = _aggregate_sentiments(
900:                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
901:            )
902:            _stash_finbert_shadow(ab_key, {
903:                "model": "FinBERT",
904:                "sentiment": fb_overall,
905:                "confidence": round(fb_avg[fb_overall], 4),
906:                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
907:            })
908:    except Exception as e:
909:        logger.debug("FinBERT shadow failed: %s", e)
910:
---

codex
The cited `shared_state.py` does not exist in this sparse worktree, but the subsystem imports it. I’m checking the parent repo copy so I can evaluate the claim against the actual module rather than treating the citation as a dead path.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "if (Test-Path 'Q:\\finance-analyzer\\portfolio\\shared_state.py') { Get-Content 'Q:\\finance-analyzer\\portfolio\\shared_state.py' | Select-Object -First 220 | % {"'$global:i=($global:i+1); '"'{0}:{1}' -f "'$global:i,$_} } else { '"'MISSING' }" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "if (Test-Path 'Q:\\finance-analyzer\\portfolio\\shared_state.py') { rg -n \"def _cached|newsapi_track_call|newsapi_quota_ok|newsapi_search_query|newsapi_ttl_for_ticker|_tool_cache|_cache_lock|yfinance_lock\" 'Q:\\finance-analyzer\\portfolio\\shared_state.py' } else { 'MISSING' }" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import pandas as pd
print(pd.to_datetime([1710000000000], unit='ms'))
print(pd.to_datetime(['2026-01-01T12:00:00Z']))
print(pd.to_datetime(['2026-01-01T12:00:00+00:00']))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 619ms:
15:_tool_cache = {}
22:_cache_lock = threading.Lock()
37:def _cached(key, ttl, func, *args):
49:    with _cache_lock:
50:        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
51:            return _tool_cache[key]["data"]
54:        if len(_tool_cache) > _CACHE_MAX_SIZE:
55:            expired = [k for k, v in _tool_cache.items()
58:                del _tool_cache[k]
60:            if len(_tool_cache) > _CACHE_MAX_SIZE:
62:                    _tool_cache, key=lambda k: _tool_cache[k]["time"]
66:                    del _tool_cache[k]
80:            if key in _tool_cache:
81:                age = now - _tool_cache[key]["time"]
85:                    return _tool_cache[key]["data"]
94:        with _cache_lock:
100:                _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
105:        with _cache_lock:
112:        with _cache_lock:
115:            if key in _tool_cache:
116:                age = now - _tool_cache[key]["time"]
124:                _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
125:                return _tool_cache[key]["data"]
129:def _cached_or_enqueue(key, ttl, enqueue_fn, context,
158:    with _cache_lock:
159:        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
160:            return _tool_cache[key]["data"]
175:        if key in _tool_cache:
176:            age = now - _tool_cache[key]["time"]
177:            cached_data = _tool_cache[key]["data"]
229:    with _cache_lock:
233:        _tool_cache[key] = {
287:yfinance_lock = threading.Lock()
325:def newsapi_quota_ok() -> bool:
341:def newsapi_track_call():
351:def newsapi_ttl_for_ticker(ticker: str):
374:def newsapi_search_query(ticker: str) -> str:
389:NEWSAPI_TTL = 1800       # 30 min fallback — overridden by newsapi_ttl_for_ticker()

 succeeded in 685ms:
1:"""Shared mutable state for the portfolio system.
2:
3:All modules that need caching or cross-cycle state import from here.
4:This avoids circular imports and provides a single source of truth.
5:"""
6:
7:import logging
8:import threading
9:import time
10:from datetime import UTC
11:
12:logger = logging.getLogger("portfolio.shared_state")
13:
14:# Tool cache — avoid re-running expensive tools every cycle
15:_tool_cache = {}
16:
17:# Retry cooldown for _cached() errors
18:_RETRY_COOLDOWN = 60
19:
20:
21:_CACHE_MAX_SIZE = 512  # evict expired entries when cache exceeds this size
22:_cache_lock = threading.Lock()
23:
24:# BUG-166: Dogpile/thundering-herd prevention.
25:# Tracks which keys are currently being refreshed. When a thread sees a cache
26:# miss and the key is already loading, it returns stale data (if available)
27:# instead of calling the function redundantly.
28:_loading_keys: set[str] = set()
29:_LOADING_TIMEOUT = 120  # seconds to wait for a loading thread before giving up
30:# C11/SS1: Track when each key was added to _loading_keys for eviction of
31:# permanently stuck keys (batch flush crash before updating cache).
32:_loading_timestamps: dict[str, float] = {}
33:
34:_MAX_STALE_FACTOR = 3  # return None if cached data is older than TTL * this factor
35:
36:
37:def _cached(key, ttl, func, *args):
38:    """Cache-through helper: returns cached data if fresh, else calls func.
39:
40:    Dogpile prevention (BUG-166): when multiple threads detect a cache miss
41:    simultaneously, only one thread fetches the data. Others return stale
42:    data if available, preventing redundant expensive calls (LLM inference,
43:    API requests) and model swap contention.
44:
45:    On error, returns stale data if it's less than TTL * _MAX_STALE_FACTOR old.
46:    Beyond that, returns None to prevent trading on dangerously old data.
47:    """
48:    now = time.time()
49:    with _cache_lock:
50:        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
51:            return _tool_cache[key]["data"]
52:        # Evict expired entries when cache grows too large
53:        # Use TTL-aware eviction: entries expire after ttl * _MAX_STALE_FACTOR
54:        if len(_tool_cache) > _CACHE_MAX_SIZE:
55:            expired = [k for k, v in _tool_cache.items()
56:                       if now - v["time"] > v.get("ttl", 3600) * _MAX_STALE_FACTOR]
57:            for k in expired:
58:                del _tool_cache[k]
59:            # LRU fallback: if still over limit (all entries fresh), evict oldest 25%
60:            if len(_tool_cache) > _CACHE_MAX_SIZE:
61:                sorted_keys = sorted(
62:                    _tool_cache, key=lambda k: _tool_cache[k]["time"]
63:                )
64:                evict_count = len(sorted_keys) // 4 or 1
65:                for k in sorted_keys[:evict_count]:
66:                    del _tool_cache[k]
67:
68:        # C11/SS1: Evict stuck loading keys older than _LOADING_TIMEOUT seconds.
69:        _now_evict = time.time()
70:        stuck = [k for k, ts in _loading_timestamps.items()
71:                 if _now_evict - ts > _LOADING_TIMEOUT]
72:        for k in stuck:
73:            _loading_keys.discard(k)
74:            stuck_duration = _now_evict - _loading_timestamps.pop(k, _now_evict)
75:            logger.warning("[%s] evicted stuck loading key after %.0fs (timeout %ds)", k, stuck_duration, _LOADING_TIMEOUT)
76:
77:        # BUG-166: Dogpile prevention — if another thread is already loading
78:        # this key, return stale data instead of calling func redundantly.
79:        if key in _loading_keys:
80:            if key in _tool_cache:
81:                age = now - _tool_cache[key]["time"]
82:                max_stale = ttl * _MAX_STALE_FACTOR
83:                if age <= max_stale:
84:                    logger.debug("[%s] stale-while-revalidate (another thread loading)", key)
85:                    return _tool_cache[key]["data"]
86:            # No stale data available — return None rather than pile on
87:            logger.debug("[%s] no stale data, another thread loading — returning None", key)
88:            return None
89:        _loading_keys.add(key)
90:        _loading_timestamps[key] = time.time()
91:
92:    try:
93:        data = func(*args)
94:        with _cache_lock:
95:            # Don't cache None results — they typically indicate transient
96:            # API failures that returned None instead of raising. Caching
97:            # None hides the failure for the entire TTL duration, preventing
98:            # retry. Stale data (if any) is preserved for fallback.
99:            if data is not None:
100:                _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
101:            _loading_keys.discard(key)
102:            _loading_timestamps.pop(key, None)  # BUG-213: clean up on success path
103:        return data
104:    except KeyboardInterrupt:
105:        with _cache_lock:
106:            _loading_keys.discard(key)
107:            _loading_timestamps.pop(key, None)
108:        logger.warning("[%s] interrupted (KeyboardInterrupt), returning None", key)
109:        return None
110:    except Exception as e:
111:        logger.warning("[%s] error: %s", key, e)
112:        with _cache_lock:
113:            _loading_keys.discard(key)
114:            _loading_timestamps.pop(key, None)
115:            if key in _tool_cache:
116:                age = now - _tool_cache[key]["time"]
117:                max_stale = ttl * _MAX_STALE_FACTOR
118:                if age > max_stale:
119:                    logger.warning(
120:                        "[%s] stale data too old (%.0fs > %.0fs max), returning None",
121:                        key, age, max_stale,
122:                    )
123:                    return None
124:                _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
125:                return _tool_cache[key]["data"]
126:        return None
127:
128:
129:def _cached_or_enqueue(key, ttl, enqueue_fn, context,
130:                        should_enqueue_fn=None, max_stale_factor=None):
131:    """Check cache — if fresh return it, if expired enqueue for batch and return stale.
132:
133:    Unlike _cached(), this never calls the model directly. On miss, it adds
134:    the request to the batch queue and returns stale data (or None).
135:
136:    Dogpile prevention (Codex finding #5): uses _loading_keys to avoid
137:    re-enqueuing the same key every cycle if the batch flush hasn't run yet.
138:
139:    2026-04-10 (perf/llama-swap-reduction) — two new optional parameters to
140:    support rotation scheduling of LLM signals:
141:
142:    - should_enqueue_fn: callable returning bool. If provided and the cache
143:      is stale-but-present, skip the enqueue when the callback says "no"
144:      (rotation off-cycle). If stale data is NOT available, force-enqueue
145:      regardless of the callback — we cannot leave the caller empty-handed
146:      when no stale fallback exists. Default None means "always enqueue",
147:      which preserves the pre-rotation behavior for every existing caller.
148:
149:    - max_stale_factor: integer override for how stale data can be returned,
150:      in multiples of ttl. Default None means use the module-level
151:      _MAX_STALE_FACTOR. LLM rotation passes 5 here so each rotated vote
152:      can stay valid across the full rotation cycle (3 * TTL) plus slippage.
153:    """
154:    now = time.time()
155:    effective_stale_factor = (
156:        max_stale_factor if max_stale_factor is not None else _MAX_STALE_FACTOR
157:    )
158:    with _cache_lock:
159:        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
160:            return _tool_cache[key]["data"]
161:
162:        # Check stale availability BEFORE deciding whether to enqueue, because
163:        # the rotation gate can only safely skip enqueue when we have stale
164:        # fallback to return. If stale is exhausted we must force-enqueue.
165:        #
166:        # 2026-04-10 code-review finding N1: also treat cached `data is None`
167:        # as "stale NOT available". main.py writes _update_cache(key, None)
168:        # when a flush fails as a short-lived retry cooldown; without this
169:        # check, the rotation gate would skip enqueue on those None entries
170:        # and the caller would see None for up to 3 rotation cycles
171:        # (~3 minutes) before retry. The extra `is not None` guard force-
172:        # enqueues on failed-cache entries matching legacy recovery behavior.
173:        stale_data = None
174:        stale_available = False
175:        if key in _tool_cache:
176:            age = now - _tool_cache[key]["time"]
177:            cached_data = _tool_cache[key]["data"]
178:            if age <= ttl * effective_stale_factor and cached_data is not None:
179:                stale_available = True
180:                stale_data = cached_data
181:
182:        # Decide whether to enqueue:
183:        # - Default (no should_enqueue_fn): always enqueue (legacy behavior)
184:        # - Callback returns True: enqueue (rotation on-cycle, or force path)
185:        # - Callback returns False AND stale available: skip (rotation off-cycle,
186:        #   stale fallback carries us until next on-cycle)
187:        # - Callback returns False AND stale NOT available: enqueue anyway
188:        #   (fresh cold path; caller has no fallback, we must refresh)
189:        if should_enqueue_fn is None:
190:            should_enq = True
191:        else:
192:            try:
193:                should_enq = bool(should_enqueue_fn()) or not stale_available
194:            except Exception as e:
195:                logger.warning(
196:                    "[%s] should_enqueue_fn raised, defaulting to enqueue: %s",
197:                    key, e,
198:                )
199:                should_enq = True
200:
201:        if should_enq and enqueue_fn and context is not None and key not in _loading_keys:
202:            _loading_keys.add(key)
203:            # C11/SS1: Track enqueue time for stuck-key eviction.
204:            _loading_timestamps[key] = time.time()
205:            try:
206:                enqueue_fn(key, context)
207:            except Exception as e:
208:                _loading_keys.discard(key)
209:                _loading_timestamps.pop(key, None)
210:                logger.warning("[%s] enqueue_fn raised, key released: %s", key, e)
211:
212:        # Return stale if available
213:        if stale_available:
214:            return stale_data
215:    return None
216:
217:
218:# 2026-04-10 (perf/llama-swap-reduction): monotonic counter of full-LLM
219:# batch flushes that actually processed work. Drives rotation scheduling in
220:# portfolio.llm_batch.is_llm_on_cycle — incremented at the end of

 succeeded in 1595ms:
DatetimeIndex(['2024-03-09 16:00:00'], dtype='datetime64[ms]', freq=None)
DatetimeIndex(['2026-01-01 12:00:00+00:00'], dtype='datetime64[us, UTC]', freq=None)
DatetimeIndex(['2026-01-01 12:00:00+00:00'], dtype='datetime64[us, UTC]', freq=None)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\shared_state.py'; "'$start=220; $end=410; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"dxy_intraday|get_dxy_intraday|EURUSD=X-synth|value_is_synth|change_1h_pct|change_3h_pct\" -g \"*.py\" Q:\\finance-analyzer" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_open_interest\\(|open_interest|oi_history|get_all_futures_data\" -g \"*.py\" Q:\\finance-analyzer" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_onchain_data\\(|onchain_btc|_load_onchain_cache|onchain_cache|_fetched_at\" -g \"*.py\" Q:\\finance-analyzer" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 657ms:
Q:\finance-analyzer\portfolio\macro_context.py:101:        "change_1h_pct": round(change_1h, 4) if not math.isnan(change_1h) else None,
Q:\finance-analyzer\portfolio\macro_context.py:102:        "change_3h_pct": round(change_3h, 4) if not math.isnan(change_3h) else None,
Q:\finance-analyzer\portfolio\macro_context.py:107:def _fetch_dxy_intraday():
Q:\finance-analyzer\portfolio\macro_context.py:129:    if result is not None and result.get("change_1h_pct") is not None:
Q:\finance-analyzer\portfolio\macro_context.py:136:    # it is arbitrary. Only ``change_1h_pct`` / ``change_3h_pct`` from
Q:\finance-analyzer\portfolio\macro_context.py:144:    return _dxy_features_from_close(synth, source="EURUSD=X-synth")
Q:\finance-analyzer\portfolio\macro_context.py:147:def get_dxy_intraday():
Q:\finance-analyzer\portfolio\macro_context.py:149:    return _cached("dxy_intraday", DXY_INTRADAY_TTL, _fetch_dxy_intraday)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:223:    """Copper 60m bars. Exposes change_1h_pct + change_3h_pct."""
Q:\finance-analyzer\portfolio\metals_cross_assets.py:233:            "change_1h_pct": _pct_change(close, 1),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:234:            "change_3h_pct": _pct_change(close, 3),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:241:    """Gold/Silver ratio 60m bars. Exposes ratio_change_3h_pct."""
Q:\finance-analyzer\portfolio\metals_cross_assets.py:257:            "change_1h_pct": _pct_change(ratio, 1),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:258:            "change_3h_pct": _pct_change(ratio, 3),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:275:            "change_1h_pct": _pct_change(close, 1),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:276:            "change_3h_pct": _pct_change(close, 3),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:293:            "change_1h_pct": _pct_change(close, 1),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:294:            "change_3h_pct": _pct_change(close, 3),
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:20:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:24:            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.4,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:31:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:35:            "value": 104.0, "change_1h_pct": 0.25, "change_3h_pct": 0.4,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:41:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:45:            "value": 104.0, "change_1h_pct": -0.3, "change_3h_pct": -0.5,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:51:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:56:            "value": 104.0, "change_1h_pct": -0.1, "change_3h_pct": -0.15,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:63:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:70:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:77:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:85:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:90:            "value": 104.0, "change_1h_pct": None, "change_3h_pct": -0.3,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:96:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:102:            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.5,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:108:            "value": 104.0, "change_1h_pct": -0.5, "change_3h_pct": -0.8,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:117:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:122:            "value": 104.0, "change_1h_pct": -2.0, "change_3h_pct": -3.0,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:128:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:132:            "value": 104.0, "change_1h_pct": -0.2, "change_3h_pct": -0.3,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:133:            "source": "EURUSD=X-synth",
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:136:        assert result["indicators"]["source"] == "EURUSD=X-synth"
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:138:    @patch("portfolio.macro_context.get_dxy_intraday")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:142:            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.3,
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:150:    """Tests for macro_context._fetch_dxy_intraday fallback logic."""
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:156:        from portfolio.macro_context import _fetch_dxy_intraday
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:163:        result = _fetch_dxy_intraday()
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:167:        assert abs(result["change_1h_pct"] - 0.0959) < 0.01
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:169:        assert abs(result["change_3h_pct"] - 0.2882) < 0.01
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:175:        from portfolio.macro_context import _fetch_dxy_intraday
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:184:        result = _fetch_dxy_intraday()
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:186:        assert result["source"] == "EURUSD=X-synth"
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:188:        assert result["change_1h_pct"] < 0
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:194:        from portfolio.macro_context import _fetch_dxy_intraday
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:196:        result = _fetch_dxy_intraday()
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:62:        from portfolio.macro_context import get_dxy_intraday
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:64:        logger.debug("macro_context.get_dxy_intraday not available")
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:67:    data = get_dxy_intraday()
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:71:    change_1h = data.get("change_1h_pct")
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:93:            "dxy_change_1h_pct": round(change_1h, 3),
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:94:            "dxy_change_3h_pct": (
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:95:                round(data["change_3h_pct"], 3)
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:96:                if data.get("change_3h_pct") is not None else None
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:240:        result["copper_change_pct"] = copper["change_3h_pct"] if copper else 0.0
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:241:        result["gs_velocity_pct"] = gs["change_3h_pct"] if gs else 0.0
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:242:        result["spy_change_pct"] = spy["change_3h_pct"] if spy else 0.0
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:243:        result["oil_change_pct"] = oil["change_3h_pct"] if oil else 0.0
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:291:            "copper": {"change_3h_pct": 0.5},
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:292:            "gold_silver_ratio": {"change_3h_pct": 0.2},
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:293:            "spy": {"change_3h_pct": 0.3},
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:317:            "copper": {"change_3h_pct": 0.5},
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:319:            "spy": {"change_3h_pct": 0.3},
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:320:            "oil": {"change_3h_pct": 0.2},
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:345:            "copper": {"change_3h_pct": 0.5},

 succeeded in 669ms:
Q:\finance-analyzer\tests\test_crypto_precompute.py:15:            "btc_funding_rate": 0.0001, "btc_open_interest": 100000,
Q:\finance-analyzer\tests\test_crypto_precompute.py:18:            "eth_funding_rate": 0.00008, "eth_open_interest": 50000,
Q:\finance-analyzer\tests\test_crypto_precompute.py:43:             "open_interest", "onchain"])
Q:\finance-analyzer\tests\test_crypto_macro.py:24:            {"instrument_name": "BTC-28MAR26-65000-P", "open_interest": 100},
Q:\finance-analyzer\tests\test_crypto_macro.py:25:            {"instrument_name": "BTC-28MAR26-70000-C", "open_interest": 200},
Q:\finance-analyzer\tests\test_crypto_macro.py:26:            {"instrument_name": "BTC-28MAR26-70000-P", "open_interest": 150},
Q:\finance-analyzer\tests\test_crypto_macro.py:27:            {"instrument_name": "BTC-28MAR26-75000-C", "open_interest": 100},
Q:\finance-analyzer\tests\test_crypto_macro.py:45:            {"instrument_name": "BTC-28MAR26-70000-C", "open_interest": 100},
Q:\finance-analyzer\tests\test_crypto_macro.py:46:            {"instrument_name": "BTC-28MAR26-70000-P", "open_interest": 120},
Q:\finance-analyzer\tests\test_crypto_macro.py:66:    def test_no_open_interest(self):
Q:\finance-analyzer\tests\test_crypto_macro.py:71:            {"instrument_name": "BTC-28MAR26-70000-C", "open_interest": 0},
Q:\finance-analyzer\tests\test_crypto_macro.py:72:            {"instrument_name": "BTC-28MAR26-70000-P", "open_interest": 0},
Q:\finance-analyzer\portfolio\crypto_precompute.py:122:        "btc_open_interest": None,
Q:\finance-analyzer\portfolio\crypto_precompute.py:123:        "eth_open_interest": None,
Q:\finance-analyzer\portfolio\crypto_precompute.py:175:            ("BTCUSDT", "btc_funding_rate", "btc_open_interest"),
Q:\finance-analyzer\portfolio\crypto_precompute.py:176:            ("ETHUSDT", "eth_funding_rate", "eth_open_interest"),
Q:\finance-analyzer\portfolio\crypto_precompute.py:266:            "open_interest": market.get("btc_open_interest"),
Q:\finance-analyzer\portfolio\crypto_precompute.py:273:            "open_interest": market.get("eth_open_interest"),
Q:\finance-analyzer\portfolio\crypto_macro_data.py:75:        oi = item.get("open_interest", 0) or 0
Q:\finance-analyzer\portfolio\oil_precompute.py:520:    oi = _int(latest.get("open_interest_all"))
Q:\finance-analyzer\portfolio\oil_precompute.py:542:        "open_interest": oi,
Q:\finance-analyzer\portfolio\oil_precompute.py:696:        "oi": cot_data.get("open_interest"),
Q:\finance-analyzer\portfolio\metals_precompute.py:417:    oi = _int(latest.get("open_interest_all"))
Q:\finance-analyzer\portfolio\metals_precompute.py:436:        "open_interest": oi,
Q:\finance-analyzer\portfolio\metals_precompute.py:606:        "oi": cot_data.get("open_interest"),
Q:\finance-analyzer\portfolio\reporting.py:406:        from portfolio.futures_data import get_all_futures_data
Q:\finance-analyzer\portfolio\reporting.py:410:            fdata = get_all_futures_data(ticker)
Q:\finance-analyzer\portfolio\reporting.py:414:            oi = fdata.get("open_interest")
Q:\finance-analyzer\portfolio\reporting.py:416:                entry["open_interest"] = oi.get("oi")
Q:\finance-analyzer\portfolio\reporting.py:417:            oi_hist = fdata.get("oi_history")
Q:\finance-analyzer\tests\test_futures_flow.py:42:def oi_history_rising():
Q:\finance-analyzer\tests\test_futures_flow.py:51:def oi_history_falling():
Q:\finance-analyzer\tests\test_futures_flow.py:118:        from portfolio.futures_data import get_open_interest
Q:\finance-analyzer\tests\test_futures_flow.py:119:        assert get_open_interest("NVDA") is None
Q:\finance-analyzer\tests\test_futures_flow.py:122:        from portfolio.futures_data import get_all_futures_data
Q:\finance-analyzer\tests\test_futures_flow.py:123:        assert get_all_futures_data("AAPL") is None
Q:\finance-analyzer\tests\test_futures_flow.py:125:    def test_non_crypto_oi_history_returns_none(self):
Q:\finance-analyzer\tests\test_futures_flow.py:126:        from portfolio.futures_data import get_open_interest_history
Q:\finance-analyzer\tests\test_futures_flow.py:127:        assert get_open_interest_history("GOOGL") is None
Q:\finance-analyzer\tests\test_futures_flow.py:146:    def test_get_open_interest_success(self, mock_fetch):
Q:\finance-analyzer\tests\test_futures_flow.py:152:        from portfolio.futures_data import get_open_interest
Q:\finance-analyzer\tests\test_futures_flow.py:158:        result = get_open_interest("BTC-USD")
Q:\finance-analyzer\tests\test_futures_flow.py:164:    def test_get_open_interest_failure(self, mock_fetch):
Q:\finance-analyzer\tests\test_futures_flow.py:166:        from portfolio.futures_data import get_open_interest
Q:\finance-analyzer\tests\test_futures_flow.py:170:        result = get_open_interest("ETH-USD")
Q:\finance-analyzer\tests\test_futures_flow.py:174:    def test_get_oi_history_success(self, mock_fetch):
Q:\finance-analyzer\tests\test_futures_flow.py:179:        from portfolio.futures_data import get_open_interest_history
Q:\finance-analyzer\tests\test_futures_flow.py:183:        result = get_open_interest_history("BTC-USD")
Q:\finance-analyzer\tests\test_futures_flow.py:219:    def test_get_all_futures_data_partial_failure(self, mock_fetch):
Q:\finance-analyzer\tests\test_futures_flow.py:229:        from portfolio.futures_data import get_all_futures_data
Q:\finance-analyzer\tests\test_futures_flow.py:236:        result = get_all_futures_data("BTC-USD")
Q:\finance-analyzer\tests\test_futures_flow.py:238:        assert result["open_interest"] is not None
Q:\finance-analyzer\tests\test_futures_flow.py:240:        assert result["oi_history"] is None
Q:\finance-analyzer\tests\test_futures_flow.py:255:    def test_rising_oi_rising_price_buy(self, oi_history_rising, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:257:        assert _oi_trend(oi_history_rising, sample_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:259:    def test_rising_oi_falling_price_sell(self, oi_history_rising, falling_df):
Q:\finance-analyzer\tests\test_futures_flow.py:261:        assert _oi_trend(oi_history_rising, falling_df) == "SELL"
Q:\finance-analyzer\tests\test_futures_flow.py:263:    def test_falling_oi_hold(self, oi_history_falling, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:265:        assert _oi_trend(oi_history_falling, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:280:    def test_zero_price_start_returns_hold(self, oi_history_rising):
Q:\finance-analyzer\tests\test_futures_flow.py:302:        result = _oi_trend(oi_history_rising, df2)
Q:\finance-analyzer\tests\test_futures_flow.py:307:    def test_price_up_oi_down_sell(self, oi_history_falling, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:309:        assert _oi_divergence(oi_history_falling, sample_df) == "SELL"
Q:\finance-analyzer\tests\test_futures_flow.py:311:    def test_price_down_oi_down_buy(self, oi_history_falling, falling_df):
Q:\finance-analyzer\tests\test_futures_flow.py:313:        assert _oi_divergence(oi_history_falling, falling_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:421:            "open_interest": {"oi": 50000, "symbol": "BTCUSDT", "time": 1000},
Q:\finance-analyzer\tests\test_futures_flow.py:422:            "oi_history": [
Q:\finance-analyzer\tests\test_futures_flow.py:442:            "open_interest": {"oi": 50000, "symbol": "BTCUSDT", "time": 1000},
Q:\finance-analyzer\tests\test_futures_flow.py:443:            "oi_history": [
Q:\finance-analyzer\tests\test_futures_flow.py:468:        assert "open_interest" in result["indicators"]
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:120:                    "oi": _int_safe(r.get("open_interest_all")),
Q:\finance-analyzer\portfolio\futures_data.py:33:def get_open_interest(ticker):
Q:\finance-analyzer\portfolio\futures_data.py:58:def get_open_interest_history(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:201:def get_all_futures_data(ticker):
Q:\finance-analyzer\portfolio\futures_data.py:204:    Returns: dict with keys: open_interest, oi_history, ls_ratio,
Q:\finance-analyzer\portfolio\futures_data.py:211:        "open_interest": get_open_interest(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:212:        "oi_history": get_open_interest_history(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:224:        result = get_all_futures_data(t)
Q:\finance-analyzer\portfolio\signals\futures_flow.py:43:def _oi_trend(oi_history, df):
Q:\finance-analyzer\portfolio\signals\futures_flow.py:50:    if not oi_history or len(oi_history) < _MIN_HISTORY:
Q:\finance-analyzer\portfolio\signals\futures_flow.py:53:    recent_oi = [d.get("oi", 0) or 0 for d in oi_history[-_MIN_HISTORY:]]  # H16
Q:\finance-analyzer\portfolio\signals\futures_flow.py:79:def _oi_divergence(oi_history, df):
Q:\finance-analyzer\portfolio\signals\futures_flow.py:85:    if not oi_history or len(oi_history) < _MIN_HISTORY:
Q:\finance-analyzer\portfolio\signals\futures_flow.py:88:    recent_oi = [d.get("oi", 0) or 0 for d in oi_history[-_MIN_HISTORY:]]  # H16
Q:\finance-analyzer\portfolio\signals\futures_flow.py:183:def _oi_acceleration(oi_history, df):
Q:\finance-analyzer\portfolio\signals\futures_flow.py:189:    if not oi_history or len(oi_history) < _MIN_HISTORY * 2:
Q:\finance-analyzer\portfolio\signals\futures_flow.py:192:    oi_values = [d.get("oi", 0) or 0 for d in oi_history]  # H16
Q:\finance-analyzer\portfolio\signals\futures_flow.py:246:    from portfolio.futures_data import get_all_futures_data
Q:\finance-analyzer\portfolio\signals\futures_flow.py:250:        get_all_futures_data,
Q:\finance-analyzer\portfolio\signals\futures_flow.py:262:    oi_history = futures.get("oi_history")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:266:    current_oi = futures.get("open_interest")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:270:        "oi_trend": _oi_trend(oi_history, df),
Q:\finance-analyzer\portfolio\signals\futures_flow.py:271:        "oi_divergence": _oi_divergence(oi_history, df),
Q:\finance-analyzer\portfolio\signals\futures_flow.py:275:        "oi_acceleration": _oi_acceleration(oi_history, df),
Q:\finance-analyzer\portfolio\signals\futures_flow.py:285:        indicators["open_interest"] = current_oi.get("oi")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:286:    if oi_history and len(oi_history) >= 2:
Q:\finance-analyzer\portfolio\signals\futures_flow.py:287:        oi_vals = [d["oi"] for d in oi_history]
Q:\finance-analyzer\tests\test_reporting_core.py:471:        summary["futures_data"] = {"BTC-USD": {"open_interest": 500000}}
Q:\finance-analyzer\tests\test_signal_bug_fixes.py:79:        oi_history = [
Q:\finance-analyzer\tests\test_signal_bug_fixes.py:84:        assert _oi_trend(oi_history, df) == "BUY"  # rising OI + rising price
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:32:        "open_interest": 400000,

 succeeded in 679ms:
Q:\finance-analyzer\data\crypto_data.py:221:        raw = get_onchain_data()
Q:\finance-analyzer\portfolio\alpha_vantage.py:124:        "_fetched_at": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\alpha_vantage.py:177:    fetched_at = entry.get("_fetched_at")
Q:\finance-analyzer\portfolio\alpha_vantage.py:194:    fetched_at = entry.get("_fetched_at")
Q:\finance-analyzer\tests\test_alpha_vantage.py:95:        assert "_fetched_at" in result
Q:\finance-analyzer\tests\test_alpha_vantage.py:145:    def test_fetched_at_is_recent(self):
Q:\finance-analyzer\tests\test_alpha_vantage.py:147:        fetched = datetime.fromisoformat(result["_fetched_at"])
Q:\finance-analyzer\tests\test_alpha_vantage.py:189:        cache_data = {"NVDA": {"pe_ratio": 55.3, "_fetched_at": "2026-02-25T12:00:00+00:00"}}
Q:\finance-analyzer\tests\test_alpha_vantage.py:207:            alpha_vantage._cache["NVDA"] = {"pe_ratio": 55.3, "_fetched_at": old_time}
Q:\finance-analyzer\tests\test_alpha_vantage.py:213:            alpha_vantage._cache["NVDA"] = {"pe_ratio": 55.3, "_fetched_at": now}
Q:\finance-analyzer\tests\test_alpha_vantage.py:222:            alpha_vantage._cache["NVDA"] = {"_fetched_at": now}
Q:\finance-analyzer\tests\test_alpha_vantage.py:345:                alpha_vantage._cache[ticker] = {"_fetched_at": now}
Q:\finance-analyzer\tests\test_alpha_vantage.py:378:                alpha_vantage._cache[ticker] = {"_fetched_at": now}
Q:\finance-analyzer\portfolio\crypto_macro_data.py:330:        onchain = get_onchain_data()
Q:\finance-analyzer\portfolio\crypto_precompute.py:124:        "onchain_btc": None,
Q:\finance-analyzer\portfolio\crypto_precompute.py:148:        out["onchain_btc"] = get_onchain_summary()
Q:\finance-analyzer\portfolio\crypto_precompute.py:150:        logger.warning("onchain_btc fetch failed: %s", exc)
Q:\finance-analyzer\portfolio\crypto_precompute.py:267:            "onchain": market.get("onchain_btc"),
Q:\finance-analyzer\tests\test_crypto_precompute.py:16:            "onchain_btc": {"mvrv": 1.8},
Q:\finance-analyzer\portfolio\onchain_data.py:12:    data = get_onchain_data()  # returns dict or None
Q:\finance-analyzer\portfolio\onchain_data.py:32:    Older versions of onchain_cache.json stored "ts" as an ISO 8601 string
Q:\finance-analyzer\portfolio\onchain_data.py:71:CACHE_FILE = DATA_DIR / "onchain_cache.json"
Q:\finance-analyzer\portfolio\onchain_data.py:96:def _save_onchain_cache(data):
Q:\finance-analyzer\portfolio\onchain_data.py:105:def _load_onchain_cache(max_age_seconds=ONCHAIN_TTL):
Q:\finance-analyzer\portfolio\onchain_data.py:240:    _save_onchain_cache(result)
Q:\finance-analyzer\portfolio\onchain_data.py:244:def get_onchain_data():
Q:\finance-analyzer\portfolio\onchain_data.py:259:        raw_ts = persistent.get("ts", 0) or persistent.get("_fetched_at", 0)
Q:\finance-analyzer\portfolio\onchain_data.py:266:                if "onchain_btc" not in _tool_cache:
Q:\finance-analyzer\portfolio\onchain_data.py:267:                    _tool_cache["onchain_btc"] = {
Q:\finance-analyzer\portfolio\onchain_data.py:278:        cached = _load_onchain_cache(max_age_seconds=ONCHAIN_TTL * 2)
Q:\finance-analyzer\portfolio\onchain_data.py:284:    return _cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)
Q:\finance-analyzer\portfolio\reporting.py:391:        onchain = get_onchain_data()
Q:\finance-analyzer\portfolio\signal_engine.py:2873:            oc = _cached("onchain_btc_signal", ONCHAIN_TTL, get_onchain_data)
Q:\finance-analyzer\tests\test_onchain_data.py:117:# Tests for the main get_onchain_data() aggregator
Q:\finance-analyzer\tests\test_onchain_data.py:123:    @patch("portfolio.onchain_data._load_onchain_cache")
Q:\finance-analyzer\tests\test_onchain_data.py:129:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:153:        cache_key = "onchain_btc"
Q:\finance-analyzer\tests\test_onchain_data.py:156:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:185:        ss._tool_cache.pop("onchain_btc", None)
Q:\finance-analyzer\tests\test_onchain_data.py:187:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:279:    @patch("portfolio.onchain_data._load_onchain_cache")
Q:\finance-analyzer\tests\test_onchain_data.py:285:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:289:        from portfolio.onchain_data import _load_onchain_cache, _save_onchain_cache
Q:\finance-analyzer\tests\test_onchain_data.py:291:        _save_onchain_cache(test_data)
Q:\finance-analyzer\tests\test_onchain_data.py:292:        loaded = _load_onchain_cache(max_age_seconds=3600)
Q:\finance-analyzer\tests\test_onchain_data.py:297:        from portfolio.onchain_data import _load_onchain_cache, _save_onchain_cache
Q:\finance-analyzer\tests\test_onchain_data.py:299:        _save_onchain_cache(test_data)
Q:\finance-analyzer\tests\test_onchain_data.py:300:        loaded = _load_onchain_cache(max_age_seconds=3600)
Q:\finance-analyzer\tests\test_onchain_data.py:309:    """A-DE-5 (2026-04-11): Old onchain_cache.json files stored "ts" as
Q:\finance-analyzer\tests\test_onchain_data.py:356:        seeding code in get_onchain_data() must NOT crash when reading
Q:\finance-analyzer\tests\test_onchain_data.py:360:        cache_file = tmp_path / "onchain_cache.json"
Q:\finance-analyzer\tests\test_onchain_data.py:369:        od.get_onchain_data()

 succeeded in 723ms:
220:# portfolio.llm_batch.is_llm_on_cycle — incremented at the end of
221:# flush_llm_batch() iff at least one phase had queued items. In-memory only,
222:# resets to 0 on process start; on restart the rotation deterministically
223:# restarts at ministral with a cold-start warmup cycle that runs all LLMs.
224:_full_llm_cycle_count = 0
225:
226:
227:def _update_cache(key, data, ttl=None):
228:    """Update a cache entry directly (for batch flush results)."""
229:    with _cache_lock:
230:        _loading_keys.discard(key)
231:        # C11/SS1: Clean up timestamp when key is resolved.
232:        _loading_timestamps.pop(key, None)
233:        _tool_cache[key] = {
234:            "data": data,
235:            "time": time.time(),
236:            "ttl": ttl or 900,
237:        }
238:
239:
240:# Cycle counter — incremented at the start of each run() to invalidate per-cycle caches
241:_run_cycle_id = 0
242:
243:# Current market state — updated each run() cycle, used by data_collector for yfinance fallback
244:_current_market_state = "open"
245:
246:# Regime detection cache (invalidated each cycle)
247:# BUG-169: Protected by _regime_lock — accessed from 8 concurrent ThreadPoolExecutor threads
248:_regime_cache = {}
249:_regime_cache_cycle = 0
250:_regime_lock = threading.Lock()
251:
252:
253:# --- Rate limiters ---
254:
255:class _RateLimiter:
256:    """Token-bucket rate limiter. Sleeps when calls exceed rate."""
257:    def __init__(self, max_per_minute, name=""):
258:        self.interval = 60.0 / max_per_minute
259:        self.last_call = 0.0
260:        self.name = name
261:        self._lock = threading.Lock()
262:
263:    def wait(self):
264:        # BUG-212: Sleep OUTSIDE the lock to avoid blocking all 8 worker
265:        # threads. Calculate sleep duration under the lock, release it,
266:        # then sleep.
267:        # Fix: Reserve the next slot (last_call = last_call + interval)
268:        # BEFORE releasing the lock, so parallel threads see the reserved
269:        # time and calculate a longer wait instead of stampeding.
270:        wait_time = 0.0
271:        with self._lock:
272:            now = time.time()
273:            elapsed = now - self.last_call
274:            if elapsed < self.interval:
275:                wait_time = self.interval - elapsed
276:            # Reserve the next slot atomically — even if we haven't slept yet,
277:            # the next thread to enter will see this and wait longer.
278:            self.last_call = self.last_call + self.interval if wait_time > 0 else now
279:        if wait_time > 0:
280:            time.sleep(wait_time)
281:
282:
283:# H11/DC-R3-4: yfinance is not thread-safe. This lock is shared across all
284:# modules (fear_greed, golddigger/data_provider, data_collector) so that
285:# concurrent calls from the 8-worker ThreadPoolExecutor are serialized.
286:# data_collector.py imports this lock instead of defining its own.
287:yfinance_lock = threading.Lock()
288:
289:# Alpaca IEX: 200 req/min → target 150/min to leave headroom
290:_alpaca_limiter = _RateLimiter(150, "alpaca")
291:# Binance: 1200 weight/min → very generous, but space out slightly
292:_binance_limiter = _RateLimiter(600, "binance")
293:# Yahoo Finance (yfinance): no official limit, but be polite — 30/min
294:_yfinance_limiter = _RateLimiter(30, "yfinance")
295:
296:
297:# Alpha Vantage: 5 req/min free tier
298:_alpha_vantage_limiter = _RateLimiter(5, "alpha_vantage")
299:
300:
301:# NewsAPI: 100 req/day free tier — tiered priority system
302:# Budget: metals (XAU, XAG) get 20-min refresh during active hours (~84/day)
303:# All other tickers: Yahoo-only (0 NewsAPI calls)
304:# BTC/ETH: already served by CryptoCompare, not NewsAPI
305:_newsapi_daily_count = 0
306:_newsapi_daily_reset = 0.0  # timestamp of last reset
307:_NEWSAPI_DAILY_BUDGET = 90  # leave 10-call margin
308:_newsapi_lock = threading.Lock()
309:
310:# Tier 1 = 20-min TTL during active hours; Tier 2 = 3h; rest = Yahoo-only
311:_NEWSAPI_PRIORITY = {"XAU": 1, "XAG": 1, "MSTR": 2}
312:
313:# Better search queries — raw ticker symbols return sparse results on NewsAPI
314:_NEWSAPI_SEARCH_QUERIES = {
315:    "XAU": "gold AND (price OR market OR ounce OR bullion OR futures OR commodity)",
316:    "XAG": "silver AND (price OR market OR ounce OR bullion OR futures OR commodity)",
317:    "MSTR": "MicroStrategy OR MSTR",
318:}
319:
320:# Active monitoring: 08:00-22:00 CET = 07:00-21:00 UTC
321:_NEWSAPI_ACTIVE_START_UTC = 7
322:_NEWSAPI_ACTIVE_END_UTC = 21
323:
324:
325:def newsapi_quota_ok() -> bool:
326:    """Check if we still have NewsAPI quota today. Thread-safe."""
327:    global _newsapi_daily_count, _newsapi_daily_reset
328:    now = time.time()
329:    with _newsapi_lock:
330:        # Reset counter at midnight UTC
331:        from datetime import datetime
332:        today_start = datetime.now(UTC).replace(
333:            hour=0, minute=0, second=0, microsecond=0
334:        ).timestamp()
335:        if _newsapi_daily_reset < today_start:
336:            _newsapi_daily_count = 0
337:            _newsapi_daily_reset = now
338:        return _newsapi_daily_count < _NEWSAPI_DAILY_BUDGET
339:
340:
341:def newsapi_track_call():
342:    """Increment NewsAPI daily counter. Call after each successful API request."""
343:    global _newsapi_daily_count
344:    with _newsapi_lock:
345:        _newsapi_daily_count += 1
346:        if _newsapi_daily_count == _NEWSAPI_DAILY_BUDGET:
347:            logger.warning("NewsAPI daily budget exhausted (%d/%d), falling back to Yahoo",
348:                          _newsapi_daily_count, _NEWSAPI_DAILY_BUDGET)
349:
350:
351:def newsapi_ttl_for_ticker(ticker: str):
352:    """Dynamic TTL based on ticker priority and time of day.
353:
354:    Returns TTL in seconds, or None to skip NewsAPI for this ticker.
355:    Tier 1 (metals): 20-min during active hours (08:00-22:00 CET).
356:    Other tickers: None (Yahoo-only, saves budget for metals).
357:    """
358:    short = ticker.upper().replace("-USD", "")
359:    priority = _NEWSAPI_PRIORITY.get(short)
360:    if priority is None:
361:        return None
362:
363:    from datetime import datetime
364:    hour_utc = datetime.now(UTC).hour
365:    is_active = _NEWSAPI_ACTIVE_START_UTC <= hour_utc < _NEWSAPI_ACTIVE_END_UTC
366:
367:    if is_active:
368:        if priority == 1:
369:            return 1200   # 20 min — metals
370:        return 10800      # 3h — secondary (MSTR etc.)
371:    return None  # off-hours: Yahoo-only
372:
373:
374:def newsapi_search_query(ticker: str) -> str:
375:    """Optimized search query for NewsAPI. Falls back to ticker symbol."""
376:    short = ticker.upper().replace("-USD", "")
377:    return _NEWSAPI_SEARCH_QUERIES.get(short, short)
378:
379:
380:# TTL constants for tool caching
381:FUNDAMENTALS_TTL = 86400  # 24 hours
382:ONCHAIN_TTL = 43200      # 12 hours (on-chain data updates slowly)
383:FEAR_GREED_TTL = 300     # 5 min
384:SENTIMENT_TTL = 900      # 15 min
385:MINISTRAL_TTL = 900      # 15 min
386:ML_SIGNAL_TTL = 900      # 15 min
387:FUNDING_RATE_TTL = 900   # 15 min
388:VOLUME_TTL = 300         # 5 min
389:NEWSAPI_TTL = 1800       # 30 min fallback — overridden by newsapi_ttl_for_ticker()

codex
I’ve got enough to start separating real defects from story inflation. Two patterns are already clear: some findings confuse comments with behavior, and some miss the actual fallback path entirely.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\signals\\dxy_cross_asset.py'; "'$start=60; $end=110; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\tests\\test_dxy_cross_asset_signal.py'; "'$ranges=@(@(128,140),@(150,190)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\reporting.py'; "'$start=400; $end=430; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\signals\\futures_flow.py'; "'$start=246; $end=290; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 613ms:
60:
61:    try:
62:        from portfolio.macro_context import get_dxy_intraday
63:    except ImportError:
64:        logger.debug("macro_context.get_dxy_intraday not available")
65:        return empty
66:
67:    data = get_dxy_intraday()
68:    if data is None:
69:        return empty
70:
71:    change_1h = data.get("change_1h_pct")
72:    if change_1h is None:
73:        return empty
74:
75:    if change_1h < -_DXY_1H_THRESHOLD_PCT:
76:        # Weak USD → strong metals → BUY
77:        action = "BUY"
78:        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
79:    elif change_1h > _DXY_1H_THRESHOLD_PCT:
80:        # Strong USD → weak metals → SELL
81:        action = "SELL"
82:        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
83:    else:
84:        action = "HOLD"
85:        confidence = 0.0
86:
87:    return {
88:        "action": action,
89:        "confidence": round(confidence, 4),
90:        "sub_signals": {"dxy_1h": action},
91:        "indicators": {
92:            "dxy_value": data.get("value"),
93:            "dxy_change_1h_pct": round(change_1h, 3),
94:            "dxy_change_3h_pct": (
95:                round(data["change_3h_pct"], 3)
96:                if data.get("change_3h_pct") is not None else None
97:            ),
98:            "source": data.get("source", "unknown"),
99:        },
100:    }

 succeeded in 594ms:
128:    @patch("portfolio.macro_context.get_dxy_intraday")
129:    def test_indicators_include_source(self, mock_dxy):
130:        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
131:        mock_dxy.return_value = {
132:            "value": 104.0, "change_1h_pct": -0.2, "change_3h_pct": -0.3,
133:            "source": "EURUSD=X-synth",
134:        }
135:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
136:        assert result["indicators"]["source"] == "EURUSD=X-synth"
137:
138:    @patch("portfolio.macro_context.get_dxy_intraday")
139:    def test_sub_signals_present(self, mock_dxy):
140:        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
---
150:    """Tests for macro_context._fetch_dxy_intraday fallback logic."""
151:
152:    @patch("yfinance.download")
153:    def test_primary_source_used_when_available(self, mock_dl):
154:        import pandas as pd
155:
156:        from portfolio.macro_context import _fetch_dxy_intraday
157:        # Generate 5 hours of DX-Y.NYB bars
158:        idx = pd.date_range("2026-04-13 09:00", periods=5, freq="h")
159:        df = pd.DataFrame(
160:            {"Close": [104.0, 104.1, 104.2, 104.3, 104.4]}, index=idx,
161:        )
162:        mock_dl.return_value = df
163:        result = _fetch_dxy_intraday()
164:        assert result is not None
165:        assert result["source"] == "DX-Y.NYB"
166:        # change_1h: (104.4 / 104.3 - 1) * 100 ≈ +0.096%
167:        assert abs(result["change_1h_pct"] - 0.0959) < 0.01
168:        # change_3h: (104.4 / 104.1 - 1) * 100 ≈ +0.288%
169:        assert abs(result["change_3h_pct"] - 0.2882) < 0.01
170:
171:    @patch("yfinance.download")
172:    def test_falls_back_to_eurusd_synth_on_empty(self, mock_dl):
173:        import pandas as pd
174:
175:        from portfolio.macro_context import _fetch_dxy_intraday
176:
177:        idx = pd.date_range("2026-04-13 09:00", periods=5, freq="h")
178:        # First call (DX-Y.NYB) returns empty; second (EURUSD=X) returns data
179:        empty_df = pd.DataFrame()
180:        eurusd_df = pd.DataFrame(
181:            {"Close": [1.170, 1.171, 1.172, 1.173, 1.174]}, index=idx,
182:        )
183:        mock_dl.side_effect = [empty_df, eurusd_df]
184:        result = _fetch_dxy_intraday()
185:        assert result is not None
186:        assert result["source"] == "EURUSD=X-synth"
187:        # Synth: 58 * eurusd^-0.576 — verify direction: eurusd went UP → synth DXY should go DOWN
188:        assert result["change_1h_pct"] < 0
189:
190:    @patch("yfinance.download")
---

 succeeded in 606ms:
246:    from portfolio.futures_data import get_all_futures_data
247:    futures = _cached(
248:        f"futures_flow_data_{ticker}",
249:        300,
250:        get_all_futures_data,
251:        ticker,
252:    )
253:
254:    if futures is None:
255:        return {
256:            "action": "HOLD",
257:            "confidence": 0.0,
258:            "sub_signals": {},
259:            "indicators": {"error": "fetch_failed"},
260:        }
261:
262:    oi_history = futures.get("oi_history")
263:    ls_ratio = futures.get("ls_ratio")
264:    top_position = futures.get("top_position_ratio")
265:    funding_hist = futures.get("funding_history")
266:    current_oi = futures.get("open_interest")
267:
268:    # Compute sub-signals
269:    sub = {
270:        "oi_trend": _oi_trend(oi_history, df),
271:        "oi_divergence": _oi_divergence(oi_history, df),
272:        "ls_extreme": _ls_extreme(ls_ratio),
273:        "top_vs_crowd": _top_vs_crowd(top_position, ls_ratio),
274:        "funding_trend": _funding_trend(funding_hist),
275:        "oi_acceleration": _oi_acceleration(oi_history, df),
276:    }
277:
278:    # Majority vote — pass list of vote strings, not the dict
279:    action, confidence = majority_vote(list(sub.values()))
280:    confidence = min(confidence, _MAX_CONFIDENCE)
281:
282:    # Build indicators dict for reporting
283:    indicators = {}
284:    if current_oi:
285:        indicators["open_interest"] = current_oi.get("oi")
286:    if oi_history and len(oi_history) >= 2:
287:        oi_vals = [d["oi"] for d in oi_history]
288:        indicators["oi_change_pct"] = round(
289:            (oi_vals[-1] - oi_vals[0]) / oi_vals[0] * 100 if oi_vals[0] else 0, 2
290:        )

 succeeded in 626ms:
400:    except Exception:
401:        logger.warning("[reporting] onchain_data failed", exc_info=True)
402:        _module_warnings.append("onchain_data")
403:
404:    # Binance FAPI futures data (crypto only — OI, LS ratios, funding)
405:    try:
406:        from portfolio.futures_data import get_all_futures_data
407:        from portfolio.tickers import CRYPTO_SYMBOLS as _CRYPTO
408:        futures_section = {}
409:        for ticker in _CRYPTO:
410:            fdata = get_all_futures_data(ticker)
411:            if fdata is None:
412:                continue
413:            entry = {}
414:            oi = fdata.get("open_interest")
415:            if oi:
416:                entry["open_interest"] = oi.get("oi")
417:            oi_hist = fdata.get("oi_history")
418:            if oi_hist:
419:                entry["oi_value_usdt"] = oi_hist[-1].get("oi_usdt")
420:            ls = fdata.get("ls_ratio")
421:            if ls:
422:                latest = ls[-1]
423:                entry["ls_ratio"] = round(latest["longShortRatio"], 3)
424:                entry["long_pct"] = round(latest["longAccount"] * 100, 1)
425:                entry["short_pct"] = round(latest["shortAccount"] * 100, 1)
426:            top = fdata.get("top_position_ratio")
427:            if top:
428:                entry["top_trader_ls_ratio"] = round(top[-1]["longShortRatio"], 3)
429:            funding = fdata.get("funding_history")
430:            if funding:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"dxy_value\" -g \"*.py\" Q:\\finance-analyzer" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\tests\\test_onchain_data.py'; "'$ranges=@(@(300,380),@(120,200)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import inspect, importlib.util, sys
spec = importlib.util.spec_from_file_location('ec', r'Q:\\finance-analyzer\\.worktrees\\adv-data-external\\portfolio\\earnings_calendar.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod._fetch_earnings_alpha_vantage.__doc__)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 546ms:
Q:\finance-analyzer\tests\test_macro_regime_integration.py:355:        assert indicators["dxy_value"] == 105.0
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:92:            "dxy_value": data.get("value"),
Q:\finance-analyzer\portfolio\signals\macro_regime.py:121:    indicators: dict = {"dxy_value": np.nan, "dxy_change_5d_pct": np.nan}
Q:\finance-analyzer\portfolio\signals\macro_regime.py:124:    dxy_value = _safe_get(macro, "dxy", "value")
Q:\finance-analyzer\portfolio\signals\macro_regime.py:134:    if dxy_value is not None:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:136:            indicators["dxy_value"] = float(dxy_value)
Q:\finance-analyzer\portfolio\signals\macro_regime.py:343:            "dxy_value": np.nan,

 succeeded in 578ms:
300:        loaded = _load_onchain_cache(max_age_seconds=3600)
301:        assert loaded is None
302:
303:
304:# ============================================================================
305:# A-DE-5: _coerce_epoch — defensive timestamp parsing
306:# ============================================================================
307:
308:class TestCoerceEpoch:
309:    """A-DE-5 (2026-04-11): Old onchain_cache.json files stored "ts" as
310:    an ISO 8601 string instead of an epoch number. The previous code did
311:    `time.time() - cache_ts` which crashes with TypeError on a string,
312:    silently disabling the on-chain BTC voter on every restart that hit
313:    an old cache."""
314:
315:    def test_int_passthrough(self):
316:        from portfolio.onchain_data import _coerce_epoch
317:        assert _coerce_epoch(1712345678) == 1712345678.0
318:
319:    def test_float_passthrough(self):
320:        from portfolio.onchain_data import _coerce_epoch
321:        assert _coerce_epoch(1712345678.5) == 1712345678.5
322:
323:    def test_numeric_string_parsed(self):
324:        from portfolio.onchain_data import _coerce_epoch
325:        assert _coerce_epoch("1712345678") == 1712345678.0
326:
327:    def test_iso_string_parsed_to_epoch(self):
328:        from datetime import datetime
329:
330:        from portfolio.onchain_data import _coerce_epoch
331:        ts = "2026-04-09T12:00:00+00:00"
332:        expected = datetime(2026, 4, 9, 12, 0, 0, tzinfo=UTC).timestamp()
333:        assert _coerce_epoch(ts) == pytest.approx(expected)
334:
335:    def test_iso_z_suffix_parsed(self):
336:        from datetime import datetime
337:
338:        from portfolio.onchain_data import _coerce_epoch
339:        ts = "2026-04-09T12:00:00Z"
340:        expected = datetime(2026, 4, 9, 12, 0, 0, tzinfo=UTC).timestamp()
341:        assert _coerce_epoch(ts) == pytest.approx(expected)
342:
343:    def test_garbage_returns_zero(self):
344:        from portfolio.onchain_data import _coerce_epoch
345:        assert _coerce_epoch("not a date") == 0.0
346:        assert _coerce_epoch(None) == 0.0
347:        assert _coerce_epoch({"weird": "shape"}) == 0.0
348:        assert _coerce_epoch([]) == 0.0
349:
350:    def test_empty_string_returns_zero(self):
351:        from portfolio.onchain_data import _coerce_epoch
352:        assert _coerce_epoch("") == 0.0
353:
354:    def test_get_onchain_data_does_not_crash_on_iso_ts_cache(self, tmp_path, monkeypatch):
355:        """A-DE-5 end-to-end: write a cache with an ISO-string ts. The
356:        seeding code in get_onchain_data() must NOT crash when reading
357:        it back. Without _coerce_epoch, time.time() - "2026-04-09T..."
358:        raises TypeError."""
359:        import portfolio.onchain_data as od
360:        cache_file = tmp_path / "onchain_cache.json"
361:        cache_file.write_text(
362:            '{"mvrv": 1.5, "ts": "2026-04-09T12:00:00Z"}',
363:            encoding="utf-8",
364:        )
365:        monkeypatch.setattr(od, "CACHE_FILE", cache_file)
366:        # Force the no-token fast path so we don't hit the API in tests.
367:        monkeypatch.setattr(od, "_load_config_token", lambda: None)
368:        # This must not raise TypeError anymore:
369:        od.get_onchain_data()
370:
371:    def test_unparseable_value_emits_debug_log(self, caplog):
372:        """P1-14 (2026-05-02): When _coerce_epoch can't parse a value, it
373:        falls through to return 0.0. Operators previously had no signal
374:        that this was happening — a corrupt cache.ts silently forces a
375:        cache miss on every restart, burning the BGeometrics 15 req/day
376:        budget. Verify the defensive failure now emits a debug log."""
377:        import logging
378:
379:        from portfolio.onchain_data import _coerce_epoch
380:        caplog.set_level(logging.DEBUG, logger="portfolio.onchain_data")
---
120:class TestGetOnchainData:
121:    """Tests for the main data aggregation function."""
122:
123:    @patch("portfolio.onchain_data._load_onchain_cache")
124:    @patch("portfolio.onchain_data._load_config_token")
125:    def test_no_token_returns_none(self, mock_token, mock_cache):
126:        mock_token.return_value = None
127:        mock_cache.return_value = None
128:        from portfolio.onchain_data import get_onchain_data
129:        result = get_onchain_data()
130:        assert result is None
131:
132:    @patch("portfolio.onchain_data.load_json", return_value={})  # H12: bypass persistent cache
133:    @patch("portfolio.onchain_data._load_config_token")
134:    @patch("portfolio.onchain_data._fetch_mvrv")
135:    @patch("portfolio.onchain_data._fetch_sopr")
136:    @patch("portfolio.onchain_data._fetch_nupl")
137:    @patch("portfolio.onchain_data._fetch_realized_price")
138:    @patch("portfolio.onchain_data._fetch_exchange_netflow")
139:    @patch("portfolio.onchain_data._fetch_liquidations")
140:    def test_aggregates_all_metrics(self, mock_liq, mock_net, mock_rp,
141:                                    mock_nupl, mock_sopr, mock_mvrv, mock_token, mock_load_json):
142:        mock_token.return_value = "test_token"
143:        mock_mvrv.return_value = {"mvrv": 1.85, "mvrv_zscore": 0.42}
144:        mock_sopr.return_value = {"sopr": 1.02}
145:        mock_nupl.return_value = {"nupl": 0.45}
146:        mock_rp.return_value = {"realized_price": 30500.0}
147:        mock_net.return_value = {"netflow": -1250.5}
148:        mock_liq.return_value = {"long_liquidations": 15e6, "short_liquidations": 8e6}
149:
150:        # Bypass in-memory cache
151:        import portfolio.shared_state as ss
152:        from portfolio.onchain_data import get_onchain_data
153:        cache_key = "onchain_btc"
154:        ss._tool_cache.pop(cache_key, None)
155:
156:        result = get_onchain_data()
157:        assert result is not None
158:        assert result["mvrv"] == 1.85
159:        assert result["mvrv_zscore"] == 0.42
160:        assert result["sopr"] == 1.02
161:        assert result["nupl"] == 0.45
162:        assert result["realized_price"] == 30500.0
163:        assert result["netflow"] == -1250.5
164:
165:    @patch("portfolio.onchain_data.load_json", return_value={})  # H12: bypass persistent cache
166:    @patch("portfolio.onchain_data._load_config_token")
167:    @patch("portfolio.onchain_data._fetch_mvrv")
168:    @patch("portfolio.onchain_data._fetch_sopr")
169:    @patch("portfolio.onchain_data._fetch_nupl")
170:    @patch("portfolio.onchain_data._fetch_realized_price")
171:    @patch("portfolio.onchain_data._fetch_exchange_netflow")
172:    @patch("portfolio.onchain_data._fetch_liquidations")
173:    def test_partial_failure_still_returns(self, mock_liq, mock_net, mock_rp,
174:                                           mock_nupl, mock_sopr, mock_mvrv, mock_token, mock_load_json):
175:        mock_token.return_value = "test_token"
176:        mock_mvrv.return_value = {"mvrv": 1.85, "mvrv_zscore": 0.42}
177:        mock_sopr.return_value = None  # failed
178:        mock_nupl.return_value = None  # failed
179:        mock_rp.return_value = {"realized_price": 30500.0}
180:        mock_net.return_value = None  # failed
181:        mock_liq.return_value = None  # failed
182:
183:        import portfolio.shared_state as ss
184:        from portfolio.onchain_data import get_onchain_data
185:        ss._tool_cache.pop("onchain_btc", None)
186:
187:        result = get_onchain_data()
188:        assert result is not None
189:        assert result["mvrv"] == 1.85
190:        assert result.get("sopr") is None
191:        assert result["realized_price"] == 30500.0
192:
193:
194:# ---------------------------------------------------------------------------
195:# Tests for interpretation helpers
196:# ---------------------------------------------------------------------------
197:
198:class TestInterpretOnchain:
199:    """Tests for on-chain interpretation helpers."""
200:
---

2026-05-10T15:28:11.500008Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\earnings_calendar.py", line 16, in <module>
    from portfolio.tickers import STOCK_SYMBOLS
ModuleNotFoundError: No module named 'portfolio.tickers'

 exited 1 in 590ms:
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "Q:\finance-analyzer\.worktrees\adv-data-external\portfolio\earnings_calendar.py", line 16, in <module>
    from portfolio.tickers import STOCK_SYMBOLS
ModuleNotFoundError: No module named 'portfolio.tickers'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"47%|directional accuracy|accuracy gate|permabull|Trading-Hero-LLM\" Q:\\finance-analyzer\\portfolio Q:\\finance-analyzer\\tests" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
p = Path(r'Q:\\finance-analyzer\\.worktrees\\adv-data-external\\portfolio\\sentiment.py')
text = p.read_text(encoding='utf-8').splitlines()
for start,end in [(740,770),(870,895)]:
    for i in range(start,end+1):
        if i<=len(text):
            print(f'{i}:{text[i-1]}')
    print('---')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_fetch_one_timeframe\\(|collect_timeframes\\(|_df\\]|\\['_df'\\]|get\\('_df'\\)|_df\" Q:\\finance-analyzer\\portfolio Q:\\finance-analyzer\\tests" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"compute_indicators\\(|return None|insufficient data\" Q:\\finance-analyzer\\portfolio\\indicators.py" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 632ms:
Q:\finance-analyzer\tests\test_accuracy_degradation.py:717:        assert "LLM:       ministral 53% · qwen3 47%" in body
Q:\finance-analyzer\portfolio\accuracy_stats.py:943:        # recent directional accuracy, even though the overall `accuracy`
Q:\finance-analyzer\portfolio\accuracy_stats.py:1444:    accuracy gate falls through to global per-signal accuracy. A cache miss
Q:\finance-analyzer\portfolio\bert_sentiment.py:1:"""In-process BERT sentiment inference (CryptoBERT, Trading-Hero-LLM, FinBERT).
Q:\finance-analyzer\portfolio\bert_sentiment.py:90:    "Trading-Hero-LLM": {
Q:\finance-analyzer\portfolio\bert_sentiment.py:91:        "hf_name": "fuchenru/Trading-Hero-LLM",
Q:\finance-analyzer\portfolio\bert_sentiment.py:95:        # Trading-Hero-LLM's labels: {0: neutral, 1: positive, 2: negative}.
Q:\finance-analyzer\portfolio\backtester.py:8:  NEW: _weighted_consensus with accuracy gate + EWMA + regime + utility + MWU
Q:\finance-analyzer\portfolio\ic_computation.py:4:providing a more nuanced quality metric than simple directional accuracy.
Q:\finance-analyzer\portfolio\main.py:586:    #   fundamental correlation cluster, per-ticker directional accuracy,
Q:\finance-analyzer\portfolio\llm_batch.py:257:    # / Trading-Hero-LLM) is unaffected if this phase breaks.
Q:\finance-analyzer\tests\test_bert_sentiment.py:230:    assert "Trading-Hero-LLM" in names
Q:\finance-analyzer\tests\test_bert_sentiment.py:279:    assert not bert_sentiment.is_loaded("Trading-Hero-LLM")
Q:\finance-analyzer\tests\test_bert_sentiment.py:343:    """Trading-Hero-LLM: {0: neutral, 1: positive, 2: negative}."""
Q:\finance-analyzer\tests\test_bert_sentiment.py:347:    result = bert_sentiment.predict("Trading-Hero-LLM", ["Apple beats earnings"])
Q:\finance-analyzer\tests\test_bert_sentiment.py:352:    result = bert_sentiment.predict("Trading-Hero-LLM", ["Stock plunges"])
Q:\finance-analyzer\portfolio\meta_learner.py:54:# 2026-04-13 audit: old config showed severe overfitting (train 76% vs test 47%
Q:\finance-analyzer\tests\test_circuit_breaker_invariants.py:34:        """The accuracy gate after max relaxation must stay above the
Q:\finance-analyzer\portfolio\metals_precompute.py:1137:            "CME margin hikes (Jan +47%, Feb +15-18%) -- caused 31% crash from $121",
Q:\finance-analyzer\portfolio\sentiment.py:10:  - Trading-Hero-LLM: stock headlines (legacy primary)
Q:\finance-analyzer\portfolio\sentiment.py:258:    TRADING_HERO_SCRIPT: "Trading-Hero-LLM",
Q:\finance-analyzer\portfolio\sentiment.py:303:# for crypto, Trading-Hero-LLM for stocks) for accuracy comparison.
Q:\finance-analyzer\portfolio\sentiment.py:753:      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
Q:\finance-analyzer\portfolio\sentiment.py:757:         baseline. Trading-Hero is permabull on financial news but at least
Q:\finance-analyzer\portfolio\sentiment.py:758:         produces variance; the 47% directional accuracy gate still acts as
Q:\finance-analyzer\portfolio\sentiment.py:765:    Primary model (votes):  Trading-Hero-LLM (both crypto and stocks)
Q:\finance-analyzer\portfolio\sentiment.py:780:    # 2026-04-28: Trading-Hero-LLM is the primary across all asset classes.
Q:\finance-analyzer\portfolio\sentiment.py:783:    model_name = "Trading-Hero-LLM"
Q:\finance-analyzer\portfolio\shadow_registry.py:196:            "notes": "CPU-cheap shadow alongside CryptoBERT/Trading-Hero-LLM. "
Q:\finance-analyzer\portfolio\tickers.py:75:    # existing accuracy gate in signal_engine.py auto-disables any signal
Q:\finance-analyzer\portfolio\tickers.py:86:    # at 1d — above 47% gate, marginal but worth live validation. Shadow-safe
Q:\finance-analyzer\portfolio\tickers.py:87:    # since 2026-04-18. If it degrades below 47% the accuracy gate auto-disables.
Q:\finance-analyzer\portfolio\tickers.py:133:    # Chronos effective accuracy: 1h=45.4%, 24h=52.4% (4d ago). The 47% tiered
Q:\finance-analyzer\portfolio\tickers.py:134:    # accuracy gate will force-HOLD 1h while letting 24h contribute. Forecast stayed
Q:\finance-analyzer\portfolio\signal_engine.py:152:# 2026-04-10 found four signals sitting in the 45-47% band that the
Q:\finance-analyzer\portfolio\signal_engine.py:162:# signals with <5000 samples prove themselves at the standard 47% threshold.
Q:\finance-analyzer\portfolio\signal_engine.py:172:# Directional accuracy gate: signals whose BUY or SELL accuracy is below this
Q:\finance-analyzer\portfolio\signal_engine.py:319:# compute+consensus on the listed ticker only. The standard accuracy gate
Q:\finance-analyzer\portfolio\signal_engine.py:320:# (47%) still protects against degradation.
Q:\finance-analyzer\portfolio\signal_engine.py:363:# for a ticker, progressively relax the accuracy gate by _GATE_RELAXATION_STEP
Q:\finance-analyzer\portfolio\signal_engine.py:369:# Expected impact: kicks in during regime transitions where the 47% gate is
Q:\finance-analyzer\portfolio\signal_engine.py:370:# silencing several voters whose recent accuracy dipped to 45-47%. Keeps at
Q:\finance-analyzer\portfolio\signal_engine.py:373:# only the overall accuracy gate is relaxed.
Q:\finance-analyzer\portfolio\signal_engine.py:475:        # These signals are already auto-gated by the blended accuracy gate,
Q:\finance-analyzer\portfolio\signal_engine.py:639:    f"Relaxed overall accuracy gate "
Q:\finance-analyzer\portfolio\signal_engine.py:643:    f"accuracy gate and the claim that the directional gate is NEVER "
Q:\finance-analyzer\portfolio\signal_engine.py:786:        # The dynamic 45% accuracy gate also catches these, but explicit
Q:\finance-analyzer\portfolio\signal_engine.py:817:        # volatility_sig 47.2%, forecast 47.2% — marginal, let accuracy gate
Q:\finance-analyzer\portfolio\signal_engine.py:823:        # the accuracy gate defaults to 0.5 and passes the 47% gate. At 1d the
Q:\finance-analyzer\portfolio\signal_engine.py:1442:      3) accuracy gate at (base - relaxation), tiered for high-sample signals
Q:\finance-analyzer\portfolio\signal_engine.py:1658:    - IC < 0 with stable ICIR → penalty (contrarian, accuracy gate handles)
Q:\finance-analyzer\portfolio\signal_engine.py:1761:    Signals below the accuracy gate (with sufficient samples) are force-skipped —
Q:\finance-analyzer\portfolio\signal_engine.py:2057:    # Only the overall accuracy gate is relaxable — directional and correlation
Q:\finance-analyzer\portfolio\signal_engine.py:2070:            "Circuit breaker: relaxing accuracy gate by %.0fpp "
Q:\finance-analyzer\portfolio\signal_engine.py:2104:        # newer signals use the standard 47% gate.
Q:\finance-analyzer\portfolio\signal_engine.py:2142:        # Directional accuracy gate: gate individual BUY/SELL direction when
Q:\finance-analyzer\portfolio\signal_engine.py:2784:    # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM) — includes social posts
Q:\finance-analyzer\portfolio\signal_engine.py:3452:        # leaving accuracy_data = {} which bypasses the accuracy gate entirely.
Q:\finance-analyzer\portfolio\signal_engine.py:3721:    # --- Per-ticker consensus accuracy gate ---
Q:\finance-analyzer\tests\test_forecast_circuit_breaker.py:34:    """Force live-Kronos mode + bypass per-ticker accuracy gate.
Q:\finance-analyzer\tests\test_forecast_circuit_breaker.py:265:        votes through — the accuracy gate was added after this test was
Q:\finance-analyzer\tests\test_loop_contract_grace.py:265:        "last_trigger_reason": "BTC-USD consensus SELL (47%)",
Q:\finance-analyzer\tests\test_loop_contract_grace.py:273:         "reasons": ["BTC-USD consensus SELL (47%)"],
Q:\finance-analyzer\tests\test_signal_engine.py:397:        """macro_regime: overall 46.6% passes accuracy gate, but BUY=38.9% gated."""
Q:\finance-analyzer\tests\test_signal_engine_core.py:991:        to gate the 4 signals sitting in the 45-47% coin-flip-adjacent
Q:\finance-analyzer\tests\test_signal_engine_core.py:1780:    """BUG-164: Per-ticker consensus accuracy gate.
Q:\finance-analyzer\tests\test_signal_engine_core.py:1906:        """_RECENCY_MIN_SAMPLES should be 30 to match accuracy gate."""
Q:\finance-analyzer\tests\test_signal_engine_circuit_breaker.py:4:The circuit breaker relaxes the accuracy gate when cascaded gates would leave
Q:\finance-analyzer\tests\test_signal_engine_circuit_breaker.py:185:        beyond baseline (all sub-47% signals are also sub-41%). The strict
Q:\finance-analyzer\tests\test_signal_engine_circuit_breaker.py:391:            "escape the accuracy gate is designed to prevent."
Q:\finance-analyzer\tests\test_topn_gate.py:239:# Category 6: Interaction with accuracy gate
Q:\finance-analyzer\tests\test_topn_gate.py:248:        # top-1 by "accuracy" but the accuracy gate still removes it.
Q:\finance-analyzer\tests\test_topn_gate.py:255:        # gated is excluded by accuracy gate → only normal (SELL) participates
Q:\finance-analyzer\tests\test_topn_gate.py:260:        """Top-N excludes some signals; accuracy gate may exclude more from those remaining."""
Q:\finance-analyzer\tests\test_topn_gate.py:268:            # Wait — accuracy gate checks acc < threshold (0.45). 0.30 < 0.45, so gated.
Q:\finance-analyzer\tests\test_weighted_consensus.py:40:        # the 45-47% coin-flip-adjacent band per the 2026-04-10 audit.
Q:\finance-analyzer\tests\test_weighted_consensus.py:340:        """At 47% accuracy, weight = 0.47."""
Q:\finance-analyzer\tests\test_weighted_consensus.py:891:        """A signal with 0% directional accuracy contributes zero weight."""
Q:\finance-analyzer\tests\test_weighted_consensus.py:913:    """Signals that fail the overall accuracy gate but have strong directional
Q:\finance-analyzer\tests\test_weighted_consensus.py:917:        """heikin_ashi scenario: overall=42.6% (gated at 47%), SELL=55.7% (>=55%).

 succeeded in 657ms:
Q:\finance-analyzer\tests\test_3h_integration.py:9:def _make_df(n=100, close_start=100.0):
Q:\finance-analyzer\tests\test_3h_integration.py:31:        df = _make_df(100)
Q:\finance-analyzer\tests\test_3h_integration.py:47:        df = _make_df(100)
Q:\finance-analyzer\tests\test_3h_integration.py:58:        df = _make_df(100)
Q:\finance-analyzer\tests\conftest.py:148:def make_ohlcv_df(n=250, close_base=100.0, trend=0.0, volatility=1.0, seed=42):
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:178:    def _make_df(self, n=150, close_start=100.0):
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:203:        df = self._make_df(150)
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:233:        df = self._make_df(150)
Q:\finance-analyzer\tests\test_batch3_fixes.py:24:    def _make_df(self, n=10, up=True):
Q:\finance-analyzer\tests\test_batch3_fixes.py:46:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:62:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:78:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:97:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:113:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:202:    def test_empty_response_returns_empty_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:16:def _make_df(n=50, close_start=100.0, volume_start=1000.0, trend="flat"):
Q:\finance-analyzer\tests\test_confidence_penalties.py:43:def _make_df_volume_pattern(n=50, close_start=100.0, vol_pattern="flat", trend="up"):
Q:\finance-analyzer\tests\test_confidence_penalties.py:45:    df = _make_df(n, close_start, trend=trend)
Q:\finance-analyzer\tests\test_confidence_penalties.py:85:        df = _make_df(27)
Q:\finance-analyzer\tests\test_confidence_penalties.py:89:        df = _make_df(27, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:93:        df = _make_df(28, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:100:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:107:        df = _make_df(100, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:113:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:131:        df_up = _make_df(100, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:137:        df = _make_df(60, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:344:        df = _make_df(50, trend="flat")
Q:\finance-analyzer\tests\test_confidence_penalties.py:359:        df = _make_df(100, trend="up")  # strong trend -> higher ADX
Q:\finance-analyzer\tests\test_confidence_penalties.py:405:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:410:    def test_adx_none_when_no_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:422:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:434:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="down")
Q:\finance-analyzer\tests\test_confidence_penalties.py:443:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:450:        df = _make_df_volume_pattern(50, vol_pattern="expanding", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:457:    def test_no_trap_with_short_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:459:        df = _make_df(3)
Q:\finance-analyzer\tests\test_confidence_penalties.py:466:    def test_no_trap_with_none_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:475:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="down")
Q:\finance-analyzer\tests\test_confidence_penalties.py:484:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:493:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:648:        df = _make_df_volume_pattern(50, vol_pattern="declining")
Q:\finance-analyzer\tests\test_confidence_penalties.py:682:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:708:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:723:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:734:        df = _make_df_volume_pattern(50, vol_pattern="expanding", trend="up")
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:10:def _df(closes):
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:17:        out = compute_crypto_cross_asset_signal(_df([100, 101]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:25:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:33:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:42:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:50:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:61:        eth = _df([3500.0, 3675.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:62:        btc = _df([100000.0, 100000.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:63:        out = compute_crypto_cross_asset_signal(_df([3500, 3675]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:70:        eth = _df([3500.0, 3325.0])  # -5%
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:71:        btc = _df([100000.0, 100000.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:72:        out = compute_crypto_cross_asset_signal(_df([3500, 3325]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:81:        dxy = _df([100.0, 101.0])  # +1% — crypto headwind
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:82:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:89:        dxy = _df([100.0, 99.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:90:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:99:        spy = _df([580.0, 585.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:100:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:110:        gold = _df([4500.0, 4600.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:111:        btc = _df([100000.0, 100000.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:112:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:122:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:126:                "dxy_history": _df([100.0, 99.0]),  # DXY down -> BUY
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:127:                "spy_history": _df([580.0, 585.0]),  # SPY up -> BUY
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:128:                "eth_history": _df([3500.0, 3500.0]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:129:                "btc_history": _df([100000.0, 100000.0]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:130:                "gold_history": _df([4500.0, 4400.0]),  # gold down -> BUY (gold/btc ratio falls)
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:137:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\portfolio\data_collector.py:280:def _fetch_one_timeframe(source, source_key, label, interval, limit, ttl):
Q:\finance-analyzer\portfolio\data_collector.py:306:            entry["_df"] = df  # preserve raw DataFrame for enhanced signals
Q:\finance-analyzer\portfolio\data_collector.py:315:def collect_timeframes(source):
Q:\finance-analyzer\tests\test_crypto_macro.py:124:    def _make_df(self, close=70000):
Q:\finance-analyzer\tests\test_crypto_macro.py:139:            self._make_df(135),
Q:\finance-analyzer\tests\test_crypto_macro.py:151:                self._make_df(70000),
Q:\finance-analyzer\tests\test_data_collector.py:367:    def _make_yf_df(self, n=10, multi_index=False):
Q:\finance-analyzer\tests\test_data_collector.py:390:        mock_yf.download.return_value = self._make_yf_df(10)
Q:\finance-analyzer\tests\test_data_collector.py:424:        mock_yf.download.return_value = self._make_yf_df(5, multi_index=True)
Q:\finance-analyzer\tests\test_data_collector.py:445:        mock_yf.download.return_value = self._make_yf_df(20)
Q:\finance-analyzer\tests\test_data_collector.py:573:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:589:        results = collect_timeframes({"alpaca": "NVDA"})
Q:\finance-analyzer\tests\test_data_collector.py:604:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:615:        assert "_df" in now_entry  # raw DataFrame preserved for enhanced signals
Q:\finance-analyzer\tests\test_data_collector.py:626:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:642:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:661:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:684:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:698:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:716:        collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:717:        collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:733:        collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:748:        collect_timeframes({"binance_fapi": "XAUUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:819:                                "confidence": None, "_df": pd.DataFrame()})
Q:\finance-analyzer\tests\test_data_collector.py:826:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\tests\test_data_collector.py:847:        results = collect_timeframes({"binance": "BTCUSDT"})
Q:\finance-analyzer\portfolio\forecast_signal.py:197:    context_df = pd.DataFrame({
Q:\finance-analyzer\portfolio\forecast_signal.py:204:    pred_df = pipeline.predict_df(
Q:\finance-analyzer\portfolio\forecast_signal.py:205:        context_df,
Q:\finance-analyzer\portfolio\forecast_signal.py:217:        # pred_df has columns: id, timestamp, 0.1, 0.5, 0.9
Q:\finance-analyzer\portfolio\forecast_signal.py:218:        row = pred_df.iloc[h - 1]
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:9:def _make_df(n=30):
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:27:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:38:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:48:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:59:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:66:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="NVDA")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:73:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="BTC-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:81:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:93:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:105:        r1 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:111:        r2 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:125:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:135:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:145:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:179:        empty_df = pd.DataFrame()
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:180:        eurusd_df = pd.DataFrame(
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:183:        mock_dl.side_effect = [empty_df, eurusd_df]
Q:\finance-analyzer\tests\test_enhanced_signals.py:47:def _make_empty_df():
Q:\finance-analyzer\tests\test_enhanced_signals.py:51:def _make_nan_df(n=100):
Q:\finance-analyzer\tests\test_enhanced_signals.py:95:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:97:        result = compute_trend_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:141:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:143:        result = compute_volatility_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:147:    def test_empty_df_schema_matches_success_path(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:154:        empty = compute_volatility_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:259:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:261:        result = compute_structure_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:404:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:406:        result = compute_heikin_ashi_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:482:        result = compute_trend_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:487:        result = compute_volatility_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:492:        result = compute_candlestick_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:497:        result = compute_structure_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:502:        result = compute_fibonacci_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:507:        result = compute_smart_money_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:512:        result = compute_oscillator_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:517:        result = compute_heikin_ashi_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:522:        result = compute_volume_flow_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:533:    def _df_missing_volume(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:539:        result = compute_trend_signal(self._df_missing_volume())
Q:\finance-analyzer\tests\test_enhanced_signals.py:544:        result = compute_structure_signal(self._df_missing_volume())
Q:\finance-analyzer\tests\test_enhanced_signals.py:549:        result = compute_volume_flow_signal(self._df_missing_volume())
Q:\finance-analyzer\portfolio\linear_factor.py:12:    model.fit(signal_history_df, returns_series)
Q:\finance-analyzer\portfolio\linear_factor.py:53:    def fit(self, signals_df: pd.DataFrame, returns: pd.Series,
Q:\finance-analyzer\portfolio\linear_factor.py:58:            signals_df: DataFrame where columns are signal names and rows are
Q:\finance-analyzer\portfolio\linear_factor.py:61:            returns: Series of forward returns aligned with signals_df index.
Q:\finance-analyzer\portfolio\linear_factor.py:68:        common = signals_df.index.intersection(returns.index)
Q:\finance-analyzer\portfolio\linear_factor.py:74:        X = signals_df.loc[common].copy()
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:18:def _make_df(n=60, with_ohlcv=True):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:29:def _make_candles_df(n=60):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:31:    df = _make_df(n)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:67:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:70:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:87:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:90:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:102:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:105:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:117:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:122:                 patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:143:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:146:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:160:        binance_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:163:             patch("portfolio.data_collector.binance_klines", return_value=binance_df) as mock_bk:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:175:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:178:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:199:    def test_kronos_df_fallback_triggers_when_candle_load_fails(self):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:203:        df = _make_df(60)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:241:    def test_kronos_df_fallback_sets_source_indicator(self):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:245:        df = _make_df(60)  # has full OHLCV
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:272:    def test_kronos_df_fallback_skipped_when_df_close_only(self):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:276:        df = _make_df(60, with_ohlcv=False)  # close only
Q:\finance-analyzer\portfolio\macro_context.py:221:    klines_df = _fetch_klines(ticker)
Q:\finance-analyzer\portfolio\macro_context.py:222:    if klines_df is None or klines_df.empty:
Q:\finance-analyzer\portfolio\macro_context.py:224:    vol = klines_df["volume"].astype(float)
Q:\finance-analyzer\portfolio\macro_context.py:225:    close = klines_df["close"].astype(float)
Q:\finance-analyzer\tests\test_futures_flow.py:14:def sample_df():
Q:\finance-analyzer\tests\test_futures_flow.py:28:def falling_df():
Q:\finance-analyzer\tests\test_futures_flow.py:255:    def test_rising_oi_rising_price_buy(self, oi_history_rising, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:257:        assert _oi_trend(oi_history_rising, sample_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:259:    def test_rising_oi_falling_price_sell(self, oi_history_rising, falling_df):
Q:\finance-analyzer\tests\test_futures_flow.py:261:        assert _oi_trend(oi_history_rising, falling_df) == "SELL"
Q:\finance-analyzer\tests\test_futures_flow.py:263:    def test_falling_oi_hold(self, oi_history_falling, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:265:        assert _oi_trend(oi_history_falling, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:267:    def test_no_data_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:269:        assert _oi_trend(None, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:270:        assert _oi_trend([], sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:307:    def test_price_up_oi_down_sell(self, oi_history_falling, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:309:        assert _oi_divergence(oi_history_falling, sample_df) == "SELL"
Q:\finance-analyzer\tests\test_futures_flow.py:311:    def test_price_down_oi_down_buy(self, oi_history_falling, falling_df):
Q:\finance-analyzer\tests\test_futures_flow.py:313:        assert _oi_divergence(oi_history_falling, falling_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:315:    def test_no_data_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:317:        assert _oi_divergence(None, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:376:    def test_accelerating_oi_price_up_buy(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:382:        assert _oi_acceleration(oi, sample_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:384:    def test_no_data_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:386:        assert _oi_acceleration(None, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:387:        assert _oi_acceleration([], sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:395:    def test_non_crypto_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:397:        result = compute_futures_flow_signal(sample_df, context={"ticker": "NVDA"})
Q:\finance-analyzer\tests\test_futures_flow.py:401:    def test_no_context_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:403:        result = compute_futures_flow_signal(sample_df, context=None)
Q:\finance-analyzer\tests\test_futures_flow.py:408:    def test_fetch_failure_hold(self, mock_cached, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:411:        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
Q:\finance-analyzer\tests\test_futures_flow.py:417:    def test_confidence_cap(self, mock_cached, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:435:        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
Q:\finance-analyzer\tests\test_futures_flow.py:439:    def test_result_structure(self, mock_cached, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:456:        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
Q:\finance-analyzer\tests\test_futures_flow.py:472:    def test_metals_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:475:        result = compute_futures_flow_signal(sample_df, context={"ticker": "XAU-USD"})
Q:\finance-analyzer\portfolio\main.py:483:            tfs = collect_timeframes(source)
Q:\finance-analyzer\portfolio\main.py:487:            now_df = None
Q:\finance-analyzer\portfolio\main.py:490:                now_df = now_entry.get("_df")
Q:\finance-analyzer\portfolio\main.py:492:                now_df = _fetch_klines(source, interval="15m", limit=100)
Q:\finance-analyzer\portfolio\main.py:493:                ind = compute_indicators(now_df)
Q:\finance-analyzer\portfolio\main.py:503:                ind, ticker=name, config=config, timeframes=tfs, df=now_df
Q:\finance-analyzer\portfolio\main.py:555:                "tfs": tfs, "ind": ind, "now_df": now_df, "price": price,
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py:241:    def _make_df(self, n=100, close_start=100.0):
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py:270:        df = self._make_df(150)
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py:336:        df = self._make_df(150)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:139:        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:140:        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:141:        if gold_df.empty or silver_df.empty:
Q:\finance-analyzer\portfolio\metals_cross_assets.py:143:        gold_close = gold_df["Close"].dropna()
Q:\finance-analyzer\portfolio\metals_cross_assets.py:144:        silver_close = silver_df["Close"].dropna()
Q:\finance-analyzer\portfolio\metals_cross_assets.py:243:        gold_df = _yf_download("GC=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:244:        silver_df = _yf_download("SI=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:245:        if gold_df.empty or silver_df.empty:
Q:\finance-analyzer\portfolio\metals_cross_assets.py:247:        gold_close = gold_df["Close"].dropna()
Q:\finance-analyzer\portfolio\metals_cross_assets.py:248:        silver_close = silver_df["Close"].dropna()
Q:\finance-analyzer\tests\test_indicators_short.py:9:def _make_df(n=100, close_start=100.0):
Q:\finance-analyzer\tests\test_indicators_short.py:27:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:33:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:40:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:51:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:58:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:64:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:70:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:77:        df = _make_df(20)
Q:\finance-analyzer\tests\test_indicators_short.py:83:        df = _make_df(20)
Q:\finance-analyzer\tests\test_indicators_core.py:23:def _make_df(n, base=100.0, trend=0.0, noise=0.5, volume=1000.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:61:def _make_uptrend_df(n=60, base=100.0, step=2.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:63:    return _make_df(n, base=base, trend=step, noise=0.3, seed=seed)
Q:\finance-analyzer\tests\test_indicators_core.py:66:def _make_downtrend_df(n=60, base=200.0, step=-2.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:68:    return _make_df(n, base=base, trend=step, noise=0.3, seed=seed)
Q:\finance-analyzer\tests\test_indicators_core.py:71:def _make_flat_df(n=60, base=100.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:73:    return _make_df(n, base=base, trend=0.0, noise=0.1, seed=seed)
Q:\finance-analyzer\tests\test_indicators_core.py:89:    def test_returns_none_for_empty_df(self):
Q:\finance-analyzer\tests\test_indicators_core.py:90:        df = _make_df(0)
Q:\finance-analyzer\tests\test_indicators_core.py:94:        df = _make_df(1)
Q:\finance-analyzer\tests\test_indicators_core.py:98:        df = _make_df(25)
Q:\finance-analyzer\tests\test_indicators_core.py:102:        df = _make_df(26)
Q:\finance-analyzer\tests\test_indicators_core.py:108:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_core.py:118:        self.ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:143:        ind = compute_indicators(_make_flat_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:147:        ind = compute_indicators(_make_uptrend_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:151:        ind = compute_indicators(_make_downtrend_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:156:        ind = compute_indicators(_make_uptrend_df(80, step=3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:161:        ind = compute_indicators(_make_downtrend_df(80, base=500.0, step=-3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:166:        ind = compute_indicators(_make_df(26))
Q:\finance-analyzer\tests\test_indicators_core.py:181:        ind = compute_indicators(_make_uptrend_df(80, step=3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:188:        ind = compute_indicators(_make_downtrend_df(80, base=500.0, step=-3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:192:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:197:        ind = compute_indicators(_make_df(26))
Q:\finance-analyzer\tests\test_indicators_core.py:205:        ind = compute_indicators(_make_uptrend_df(80, step=3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:209:        ind = compute_indicators(_make_downtrend_df(80, base=500.0, step=-3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:217:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:221:        ind = compute_indicators(_make_uptrend_df(80))
Q:\finance-analyzer\tests\test_indicators_core.py:226:        ind = compute_indicators(_make_flat_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:232:        df = _make_flat_df(58, base=100.0, seed=10)
Q:\finance-analyzer\tests\test_indicators_core.py:248:        df = _make_flat_df(58, base=100.0, seed=10)
Q:\finance-analyzer\tests\test_indicators_core.py:266:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:270:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:274:        calm = compute_indicators(_make_df(60, noise=0.1))
Q:\finance-analyzer\tests\test_indicators_core.py:275:        volatile = compute_indicators(_make_df(60, noise=5.0))
Q:\finance-analyzer\tests\test_indicators_core.py:288:        df = _make_df(60, base=50.0)
Q:\finance-analyzer\tests\test_indicators_core.py:293:        df = _make_df(60, base=50.0)
Q:\finance-analyzer\tests\test_indicators_core.py:298:        df = _make_df(60, base=50.0)
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:99:def run_kronos_prediction(predictor, candles_df: pd.DataFrame,
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:106:        candles_df: DataFrame with open,high,low,close,volume,amount,timestamp
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:115:    ohlcv = candles_df[["open", "high", "low", "close", "volume", "amount"]].copy()
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:117:    x_timestamp = pd.DatetimeIndex(candles_df["timestamp"].values)
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:130:    pred_df = predictor.predict(
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:140:    return pred_df
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:282:                pred_df = run_kronos_prediction(
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:289:                predicted_close = float(pred_df.iloc[-1]["close"])
Q:\finance-analyzer\tests\test_linear_factor.py:24:    signals_df = pd.DataFrame({
Q:\finance-analyzer\tests\test_linear_factor.py:31:    return signals_df, returns_s
Q:\finance-analyzer\tests\test_lora_pipeline.py:20:def _make_df(n=200, base_price=69000.0, trend=0.0):
Q:\finance-analyzer\tests\test_lora_pipeline.py:72:        df = _make_df(200)
Q:\finance-analyzer\tests\test_lora_pipeline.py:139:        df = _make_df(300)
Q:\finance-analyzer\tests\test_lora_pipeline.py:152:        df = _make_df(50)
Q:\finance-analyzer\tests\test_lora_pipeline.py:159:        df = _make_df(300)
Q:\finance-analyzer\tests\test_lora_pipeline.py:316:        df = _make_df(100)
Q:\finance-analyzer\tests\test_lora_pipeline.py:330:        df = _make_df(100)
Q:\finance-analyzer\tests\test_lora_pipeline.py:337:        df = _make_df(100)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:232:    gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:233:    if gap_df is None or len(gap_df) < 20:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:237:    current_gap = gap_df["gap"].iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:238:    current_avg_corr = gap_df["avg_corr"].iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:239:    current_max_eig = gap_df["max_eig_norm"].iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:242:    gap_mean = gap_df["gap"].rolling(60, min_periods=20).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:243:    gap_std = gap_df["gap"].rolling(60, min_periods=20).std().iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:250:    slope_vote = _gap_slope_vote(gap_df["gap"], is_safe_haven)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:71:    cached = _CACHE.get("ratio_df")
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:78:        copper_df = _download("HG=F", period="1y", interval="1d")
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:79:        gold_df = _download("GC=F", period="1y", interval="1d")
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:83:            copper_df = yf.download("HG=F", period="1y", interval="1d",
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:85:            gold_df = yf.download("GC=F", period="1y", interval="1d",
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:91:    if copper_df is None or gold_df is None:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:93:    if copper_df.empty or gold_df.empty:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:97:    copper_close = _extract_close(copper_df)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:98:    gold_close = _extract_close(gold_df)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:113:    _CACHE["ratio_df"] = (now, combined)
Q:\finance-analyzer\tests\test_macro_regime_integration.py:20:def _make_df(n=250, close_base=100.0, trend=0.0):
Q:\finance-analyzer\tests\test_macro_regime_integration.py:148:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_macro_regime_integration.py:162:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_macro_regime_integration.py:369:        df = _make_df(n=250, trend=0.5)  # uptrend
Q:\finance-analyzer\tests\test_macro_regime_integration.py:390:        df = _make_df(n=5)  # short, SMA signals = HOLD
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:53:    signals_df: pd.DataFrame,
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:64:        signals_df: DataFrame of signal values (columns=signals, rows=time).
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:65:        returns: Series of forward returns aligned with signals_df.
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:75:    common = signals_df.index.intersection(returns.index)
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:76:    signals_df = signals_df.loc[common]
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:94:        train_X = signals_df.iloc[start:train_end]
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:96:        test_X = signals_df.iloc[train_end:test_end]
Q:\finance-analyzer\portfolio\train_signal_weights.py:51:        (signals_df, returns_series) or None if insufficient data.
Q:\finance-analyzer\portfolio\train_signal_weights.py:96:    signals_df = df[signal_cols].fillna(0.0)
Q:\finance-analyzer\portfolio\train_signal_weights.py:98:    return signals_df, returns
Q:\finance-analyzer\portfolio\train_signal_weights.py:121:    signals_df, returns = data
Q:\finance-analyzer\portfolio\train_signal_weights.py:123:               len(signals_df), len(signals_df.columns), horizon)
Q:\finance-analyzer\portfolio\train_signal_weights.py:127:    if not model.fit(signals_df, returns):
Q:\finance-analyzer\portfolio\train_signal_weights.py:136:        signals_df, returns,
Q:\finance-analyzer\portfolio\train_signal_weights.py:137:        train_window=min(720, len(signals_df) // 3),
Q:\finance-analyzer\portfolio\train_signal_weights.py:138:        test_window=min(168, len(signals_df) // 6),
Q:\finance-analyzer\portfolio\train_signal_weights.py:139:        step_size=min(168, len(signals_df) // 6),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:21:def _make_df(n=50):
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:72:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:85:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:93:            _make_df(), ticker="NVDA", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:103:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:112:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:121:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:130:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:139:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:148:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:157:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:166:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:175:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:191:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:201:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:211:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:220:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:230:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:240:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:249:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:269:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:382:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:398:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:415:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:431:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:444:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:459:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:483:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:501:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:518:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:534:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:551:            _make_df(),
Q:\finance-analyzer\tests\test_monte_carlo_risk.py:561:    def test_very_low_df_extreme_tails(self):
Q:\finance-analyzer\tests\test_momentum_fix.py:20:def _make_df(n: int = 60, trend: str = "up") -> pd.DataFrame:
Q:\finance-analyzer\tests\test_momentum_fix.py:127:        df = _make_df(60)
Q:\finance-analyzer\tests\test_momentum_fix.py:138:        df = _make_df(60)
Q:\finance-analyzer\tests\test_momentum_fix.py:145:        df = _make_df(60)
Q:\finance-analyzer\tests\test_momentum_fix.py:155:        df = _make_df(20)
Q:\finance-analyzer\tests\test_oil_loop.py:83:    fake_df = pd.DataFrame({
Q:\finance-analyzer\tests\test_oil_loop.py:92:               return_value=fake_df) as mock_fetch:
Q:\finance-analyzer\tests\test_oil_loop.py:111:def test_fetch_live_prices_returns_empty_when_router_returns_empty_df():
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:277:    turb_df = _compute_turbulence_series(closes)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:278:    if turb_df is None or len(turb_df) < _Z_WINDOW:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:282:    turb_series = turb_df["turbulence"]
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:283:    ar_series = turb_df["absorption_ratio"]
Q:\finance-analyzer\portfolio\signals\structure.py:61:    lookback_df = df.iloc[-252:]
Q:\finance-analyzer\portfolio\signals\structure.py:62:    high = lookback_df["high"].astype(float)
Q:\finance-analyzer\portfolio\signals\structure.py:63:    low = lookback_df["low"].astype(float)
Q:\finance-analyzer\portfolio\signals\structure.py:64:    close = lookback_df["close"].astype(float)
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:9:def _make_df(n=50):
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:32:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:48:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:58:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:67:            _make_df(), ticker="NVDA", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:83:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:106:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:123:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:140:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:157:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_price_source.py:17:def _empty_df() -> pd.DataFrame:
Q:\finance-analyzer\tests\test_price_source.py:75:        mock_fapi.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:81:        mock_fapi.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:87:        mock_spot.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:93:        mock_alpaca.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:99:        mock_yf.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:111:        mock_yf.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:120:        mock_yf.return_value = _empty_df()
Q:\finance-analyzer\tests\test_safety_guards.py:32:    def _make_df(n=60, base=100.0):
Q:\finance-analyzer\tests\test_safety_guards.py:46:        df = self._make_df()
Q:\finance-analyzer\tests\test_safety_guards.py:52:        df = self._make_df()
Q:\finance-analyzer\tests\test_safety_guards.py:58:        df = self._make_df()
Q:\finance-analyzer\tests\test_signals_candlestick.py:38:def _make_df(rows, columns=("open", "high", "low", "close")):
Q:\finance-analyzer\tests\test_signals_candlestick.py:82:        df = _make_df([(100, 105, 95, 102), (102, 108, 99, 106)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:103:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:117:        df = _make_df([(i, i + 1, i - 1, i + 0.5) for i in range(10, 25)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:122:        df = _make_df([(i, i + 1, i - 1, i - 0.5) for i in range(25, 10, -1)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:127:        df = _make_df([(100, 101, 99, 100) for _ in range(10)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:131:        df = _make_df([(100, 105, 95, 102)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:363:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:372:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:378:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:384:        df = _make_df([(100, 110, 90, 108)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:408:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:421:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:435:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:444:        df = _make_df([(100, 105, 95, 102)] * 10)
Q:\finance-analyzer\tests\test_signals_candlestick.py:465:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:477:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:487:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:493:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:506:        df = _make_df([
Q:\finance-analyzer\tests\test_signals_candlestick.py:528:        df = _make_df([(100, 100, 100, 100)] * 10)
Q:\finance-analyzer\tests\test_signals_candlestick.py:536:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_calendar.py:40:def _make_df(n: int = 10, start_date: str = "2026-02-16",
Q:\finance-analyzer\tests\test_signals_calendar.py:80:def _make_df_on_date(target_date: str, n: int = 10,
Q:\finance-analyzer\tests\test_signals_calendar.py:114:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:123:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:128:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:134:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:159:        df = _make_df(n=1)
Q:\finance-analyzer\tests\test_signals_calendar.py:165:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:198:        df = _make_df_on_date("2026-02-16", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:205:        df = _make_df_on_date("2026-02-20", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:215:        df = _make_df_on_date("2026-02-17", n=10, red_prior=True)
Q:\finance-analyzer\tests\test_signals_calendar.py:223:        df = _make_df_on_date("2026-02-17", n=10, red_prior=False)
Q:\finance-analyzer\tests\test_signals_calendar.py:230:        df = _make_df_on_date("2026-02-18", n=10)  # Wednesday
Q:\finance-analyzer\tests\test_signals_calendar.py:258:        df = _make_df_on_date("2026-01-30", n=10)  # Jan 30 = 1 day remaining
Q:\finance-analyzer\tests\test_signals_calendar.py:299:        df = _make_df_on_date("2026-06-15", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:305:        df = _make_df_on_date("2026-11-15", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:386:        df = _make_df_on_date("2026-03-16", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:430:        df = _make_df_on_date("2026-01-02", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:447:        df = _make_df_on_date("2026-03-11", n=10)  # Wednesday, mid-month
Q:\finance-analyzer\tests\test_signals_calendar.py:459:                df = _make_df_on_date(target.isoformat(), n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:472:        df = _make_df_on_date("2026-01-02", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:484:        df = _make_df_on_date("2026-07-06", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:495:        df = _make_df_on_date("2026-10-06", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:504:        df = _make_df_on_date("2026-12-14", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:510:        df = _make_df(n=10, start_date="2026-01-02")
Q:\finance-analyzer\tests\test_signals_fibonacci.py:54:def _make_uptrend_df(n: int = 100, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_fibonacci.py:60:def _make_downtrend_df(n: int = 100, start: float = 200.0, step: float = 0.5) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_fibonacci.py:66:def _make_flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_fibonacci.py:365:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:373:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:385:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:390:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:395:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:422:        df = _make_flat_df(100, price=50.0)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:428:        df = _make_downtrend_df(100, start=200.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:434:        df = _make_uptrend_df(100, start=100.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:490:        df = _make_uptrend_df(MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:20:def _make_df(n=100, last_time=None):
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:37:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:43:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:56:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:310:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:26:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:489:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:495:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:521:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:544:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:558:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:569:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:588:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:618:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:37:def _make_df(n: int = 100, close_base: float = 100.0,
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:59:def _make_declining_df(n: int = 30, start: float = 100.0,
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:80:def _make_rising_df(n: int = 30, start: float = 100.0,
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:106:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:114:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:119:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:124:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:133:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:144:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:169:        df = _make_df(2)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:182:        df = _make_df(3)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:196:        df = _make_df(50, close_base=100.0, volatility=0.1)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:224:        df = _make_df(50, close_base=100.0, volatility=0.1)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:568:        df = _make_declining_df(n=50, start=100.0, step=-1.0)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:582:        df = _make_rising_df(n=50, start=100.0, step=1.0)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:616:        df = _make_df(50)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:650:    def _make_hourly_df(self, n: int = 50, start_price: float = 100.0):
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:668:        df = self._make_hourly_df(n=50)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:709:        df_50 = self._make_hourly_df(n=50)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:710:        df_100 = self._make_hourly_df(n=100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:760:            df = self._make_hourly_df(n=20)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:36:def _make_df(n: int = 250, close_base: float = 100.0,
Q:\finance-analyzer\tests\test_signals_macro_regime.py:55:def _make_df_above_sma(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:72:def _make_df_below_sma(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:89:def _make_golden_cross_df(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:109:def _make_death_cross_df(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:135:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:145:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:154:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:160:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:166:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:175:    def test_very_short_df(self):
Q:\finance-analyzer\tests\test_signals_macro_regime.py:176:        df = _make_df(n=5)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:181:    def test_none_df_returns_hold(self):
Q:\finance-analyzer\tests\test_signals_macro_regime.py:186:    def test_empty_df_returns_hold(self):
Q:\finance-analyzer\tests\test_signals_macro_regime.py:199:        df = _make_df(n=1)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:208:        df = _make_df_above_sma()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:214:        df = _make_df_below_sma()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:220:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:226:        df = _make_df_above_sma(n=100)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:232:        df = _make_df_above_sma(n=50)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:238:        df = _make_df(n=30)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:243:        df = _make_df_above_sma()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:392:        df = _make_golden_cross_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:401:        df = _make_death_cross_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:409:        df = _make_df_above_sma(n=100)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:416:        df = _make_df(n=30)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:423:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:432:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:440:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:450:        df = _make_df(n=5)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:456:        df = _make_golden_cross_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:472:        df = _make_df(n=5)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:503:        df = _make_death_cross_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:15:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:244:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:268:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:292:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:297:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:303:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:331:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:350:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAG-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:359:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:377:        result = compute_cot_positioning_signal(_make_df(), ticker="BTC-USD")
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:14:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:53:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:67:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:78:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:90:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:111:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:139:        gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:140:        assert gap_df is not None
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:141:        assert "gap" in gap_df.columns
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:142:        assert "avg_corr" in gap_df.columns
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:143:        assert "max_eig_norm" in gap_df.columns
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:147:        gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:148:        assert gap_df is not None
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:149:        assert gap_df["gap"].notna().all()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:150:        assert np.isfinite(gap_df["gap"].values).all()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:170:        gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:171:        if gap_df is not None and len(gap_df) > 0:
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:173:            avg_gap = gap_df["gap"].mean()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:174:            avg_corr = gap_df["avg_corr"].mean()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:216:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_smart_money.py:40:def _make_df(
Q:\finance-analyzer\tests\test_signals_smart_money.py:63:def _make_uptrend_with_bos_df(n: int = 80) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:102:def _make_downtrend_then_reversal_df(n: int = 80) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:122:def _make_fvg_df(bullish: bool = True, n: int = 60) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:193:def _make_flat_df(n: int = 80, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:216:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:221:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:229:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:234:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:240:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:246:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:254:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:268:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:275:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:309:        df = _make_df(MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_smart_money.py:316:        df = _make_df(MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_smart_money.py:354:        df = _make_uptrend_with_bos_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:360:        df = _make_uptrend_with_bos_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:406:        df = _make_downtrend_then_reversal_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:462:        df = _make_fvg_df(bullish=True, n=60)
Q:\finance-analyzer\tests\test_signals_smart_money.py:512:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:644:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:702:        df = _make_df(80, seed=seed)
Q:\finance-analyzer\tests\test_signals_smart_money.py:711:        df = _make_df(80, seed=seed)
Q:\finance-analyzer\tests\test_signals_smart_money.py:720:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:727:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:740:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:745:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:751:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:757:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:806:        df = _make_uptrend_with_bos_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:816:        df = _make_downtrend_then_reversal_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:824:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:843:        df = _make_df(80, seed=42)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:55:def _make_df(
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:79:def _make_trending_df(
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:113:def _make_flat_df(n: int = 80, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:128:def _make_ha_candles_df(n: int = 20) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:130:    return _make_df(n=n)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:141:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:149:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:154:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:178:        df = _make_df(1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:185:        df = _make_df(9)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:192:        df = _make_df(10)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:260:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:265:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:270:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:275:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:300:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:310:        df = _make_trending_df(n=100, step=1.5)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:315:        df = _make_trending_df(n=100, step=-1.5)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:320:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:335:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:340:        df = _make_trending_df(n=100, step=2.0)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:345:        df = _make_trending_df(n=100, step=-2.0)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:350:        df = _make_df(10)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:367:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:373:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:380:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:386:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:392:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:399:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:404:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:409:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:415:        df = _make_df(80, close_base=100.0, volatility=0.5)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:807:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:813:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:819:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:847:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:858:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:864:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:878:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:884:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:890:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:917:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:928:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:941:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:947:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:953:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:959:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:965:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:980:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:987:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:997:        df = _make_df(1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1004:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1012:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1021:        df = _make_df(10)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1032:        df = _make_df(30)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1044:        df = _make_df(30)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:47:def _make_flat_df(n: int = 60, price: float = 100.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:61:def _make_trending_df(n: int = 80, start: float = 100.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:114:def _make_random_df(n: int = 100, seed: int = 42,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:143:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:152:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:158:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:166:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:173:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:182:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:219:        df = _make_random_df(MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:226:        df = _make_random_df(100).drop(columns=["volume"])
Q:\finance-analyzer\tests\test_signals_volume_flow.py:233:        df = _make_random_df(100).drop(columns=["close"])
Q:\finance-analyzer\tests\test_signals_volume_flow.py:258:    def _uptrend_df(self) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_volume_flow.py:259:        return _make_trending_df(n=80, start=50.0, step=1.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:264:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:274:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:282:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:290:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:296:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:308:    def _downtrend_df(self) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_volume_flow.py:309:        return _make_trending_df(n=80, start=150.0, step=-1.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:314:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:324:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:332:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:338:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:344:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:358:        df = _make_flat_df(n=80, price=100.0, volume=500.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:364:        df = _make_flat_df(n=80, price=100.0, volume=500.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:370:        df = _make_flat_df(n=80, price=100.0, volume=500.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:388:        df = _make_trending_df(n=80, step=0.5, volume=0.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:395:        df = _make_trending_df(n=80, step=0.5, volume=0.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:401:        df = _make_trending_df(n=80, step=0.5, volume=0.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:417:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:423:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:441:        df = _make_random_df(MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:451:        df = _make_random_df(MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:523:        df = _make_random_df(60)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:563:        df = _make_random_df(80)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:572:        df = _make_random_df(80)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:582:        df = _make_random_df(80)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:754:        df = _make_trending_df(n=80, step=1.0, volume=2_000.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:779:            df = _make_random_df(100, seed=seed)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:788:            df = _make_random_df(100, seed=seed)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:805:        df = _make_random_df(n)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:815:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:832:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:841:        df = _make_trending_df(n=80, start=50.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:849:        df = _make_trending_df(n=80, start=50.0, step=1.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:856:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:866:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:882:    def test_same_df_same_result(self):
Q:\finance-analyzer\tests\test_signals_volume_flow.py:883:        df = _make_random_df(100, seed=7)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:890:    def test_df_not_mutated(self):
Q:\finance-analyzer\tests\test_signals_volume_flow.py:892:        df = _make_random_df(100, seed=11)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:915:    def _make_three_day_df(self) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_volume_flow.py:940:        df = self._make_three_day_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:952:        df = self._make_three_day_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:14:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_credit_spread.py:43:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:68:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:75:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:85:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:110:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:225:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:236:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:247:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:258:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:269:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:292:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:311:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:324:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:336:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:15:def _make_df(n=100, base_price=50000.0, vol=0.02):
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:33:        df = _make_df(50)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:41:        df = _make_df(5)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:108:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:120:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:129:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:145:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:157:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:166:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:176:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:183:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:192:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:200:        df = _make_df(base_price=3000.0)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:211:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:220:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:234:        df = _make_df(n=50, vol=0.005)  # Low vol data
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:247:        df = _make_df(n=50, vol=0.05)  # High vol data
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:15:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:28:def _make_trending_up_df(n=100):
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:48:def _make_trending_down_df(n=100):
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:71:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:80:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:86:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:98:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:103:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:109:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:120:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:129:        df = _make_trending_up_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:135:        df = _make_trending_down_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:141:        df = _make_df(n=200, seed=123)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:157:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:173:        df = _make_trending_up_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:178:        df = _make_trending_down_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:187:        df = _make_trending_up_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:193:        df = _make_trending_down_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:199:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:206:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_futures_basis.py:19:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_futures_basis.py:280:    def test_df_parameter_ignored(self, mock_fetch):
Q:\finance-analyzer\tests\test_signal_futures_basis.py:283:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:19:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signals_news_event.py:37:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:43:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:49:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:67:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:78:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:89:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:94:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_oscillators.py:51:def _make_df(
Q:\finance-analyzer\tests\test_signals_oscillators.py:96:def _make_trending_df(
Q:\finance-analyzer\tests\test_signals_oscillators.py:125:def _make_flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_oscillators.py:180:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:190:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:196:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:204:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:210:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:217:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:223:        df = _make_df(n=100, seed=7)
Q:\finance-analyzer\tests\test_signals_oscillators.py:232:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:257:        df = _make_df(n=1)
Q:\finance-analyzer\tests\test_signals_oscillators.py:264:        df = _make_df(n=MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_oscillators.py:271:        df = _make_df(n=MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_oscillators.py:276:        df = _make_df(n=100).drop(columns=["close"])
Q:\finance-analyzer\tests\test_signals_oscillators.py:281:        df = _make_df(n=100).drop(columns=["high"])
Q:\finance-analyzer\tests\test_signals_oscillators.py:286:        df = _make_df(n=100).drop(columns=["low"])
Q:\finance-analyzer\tests\test_signals_oscillators.py:319:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:325:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:346:        df = _make_trending_df(n=200, start=10.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_oscillators.py:356:        df = _make_trending_df(n=200, start=300.0, step=-0.5)
Q:\finance-analyzer\tests\test_signals_oscillators.py:365:        df = _make_trending_df(n=150, start=10.0, step=0.3)
Q:\finance-analyzer\tests\test_signals_oscillators.py:373:        df = _make_trending_df(n=150, start=300.0, step=-0.3)
Q:\finance-analyzer\tests\test_signals_oscillators.py:388:        df = _make_flat_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:393:        df = _make_flat_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:399:        df = _make_flat_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:414:        df = _make_df(n=100, volatility=50.0, seed=99)
Q:\finance-analyzer\tests\test_signals_oscillators.py:419:        df = _make_df(n=100, volatility=50.0, seed=99)
Q:\finance-analyzer\tests\test_signals_oscillators.py:425:        df = _make_df(n=80)
Q:\finance-analyzer\tests\test_signals_oscillators.py:802:        df = _make_df(n=150, seed=77)
Q:\finance-analyzer\tests\test_signals_oscillators.py:817:    def test_large_df_completes_without_error(self):
Q:\finance-analyzer\tests\test_signals_oscillators.py:819:        df = _make_df(n=500, seed=101)
Q:\finance-analyzer\tests\test_signals_oscillators.py:824:    def test_1000_bar_df_completes(self):
Q:\finance-analyzer\tests\test_signals_oscillators.py:826:        df = _make_df(n=1000, seed=202)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:15:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:59:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:71:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:80:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:93:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:109:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:121:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:128:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:138:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:222:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:234:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:242:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:250:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:264:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:279:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:302:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:317:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:18:def _make_df(n=200, trend=0.001):
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:44:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:56:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:69:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:85:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:92:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:101:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:107:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:116:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:132:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:172:        df = _make_df(n=250, trend=0.01)
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:179:        df = _make_df(n=50)
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:214:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:223:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_improvements.py:43:def make_df(n=100, close_base=100.0, volatility=2.0):
Q:\finance-analyzer\tests\test_signal_improvements.py:83:        df = make_df()
Q:\finance-analyzer\tests\test_signal_improvements.py:92:        df = make_df(close_base=100.0)
Q:\finance-analyzer\tests\test_signal_improvements.py:97:        df = make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:16:def _make_df(n=200, seed=42):
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:29:def _make_trending_df(n=250, direction="up"):
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:47:def _make_mean_reverting_df(n=250):
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:160:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:166:        df = _make_trending_df(100, "down")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:208:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:217:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:225:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:239:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:250:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:256:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:263:        df = _make_df(n=300)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:270:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:286:        df = _make_trending_df(300, "up")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:293:        df = _make_trending_df(300, "down")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:299:        df = _make_mean_reverting_df(300)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:308:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:319:        df = _make_df(n=300)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:20:def _make_df(n=100, trend="up", with_datetime_index=False, hour=22):
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:48:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:82:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:87:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:93:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:99:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:105:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:111:        df = _make_df(trend="up")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:162:        df = _make_df(n=50, with_datetime_index=True, hour=22)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:167:        df = _make_df(n=50, trend="up", with_datetime_index=True, hour=22)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:173:        df = _make_df(n=50, with_datetime_index=True, hour=3)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:197:        df = _make_df(n=50, trend="up")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:203:        df = _make_df(n=50, trend="down")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:217:        df = _make_df(n=30)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:224:        df = _make_df(n=50)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:232:        df = _make_df(n=30, with_datetime_index=True, hour=14)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:238:        df = _make_df(n=30, with_datetime_index=True, hour=14)
Q:\finance-analyzer\tests\test_signal_ovx_metals_spillover.py:17:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_ovx_metals_spillover.py:142:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:18:def _make_df(n=100, trend=0.0, seed=42):
Q:\finance-analyzer\tests\test_signal_network_momentum.py:49:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:59:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:80:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:86:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:93:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:100:        df = _make_df(n=200, trend=0.1)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:207:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:223:        df = _make_df(n=100, seed=99)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:235:        df = _make_df(n=100, seed=123)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:250:            df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:266:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_pipeline.py:15:from conftest import make_ohlcv_df
Q:\finance-analyzer\tests\test_signal_pipeline.py:49:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:67:        df = make_ohlcv_df(n=250, close_base=130.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:78:        df = make_ohlcv_df(n=250, close_base=2000.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:96:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:128:        df = make_ohlcv_df(n=250, close_base=130.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:140:        df = make_ohlcv_df(n=250, close_base=2000.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:153:        df = make_ohlcv_df(n=250, close_base=100.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:173:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:186:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:196:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:256:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:266:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:278:    def test_enhanced_signals_present_with_df(self, _mock):
Q:\finance-analyzer\tests\test_signal_pipeline.py:281:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:295:    def test_enhanced_signals_hold_without_df(self, _mock):
Q:\finance-analyzer\tests\test_signal_pipeline.py:311:    def test_enhanced_signals_hold_with_short_df(self, _mock):
Q:\finance-analyzer\tests\test_signal_pipeline.py:314:        df = make_ohlcv_df(n=10)  # too short
Q:\finance-analyzer\tests\test_signal_pipeline.py:348:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:8:def _make_df(n=200):
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:21:def _make_skewed_df(n=300, skew_direction="negative"):
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:53:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:62:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:68:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:80:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:85:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:91:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:107:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:114:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:134:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:139:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:144:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:154:        df = _make_skewed_df(n=300, skew_direction="negative")
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:161:        df = _make_skewed_df(n=300, skew_direction="positive")
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:185:        df = _make_df(n=60)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:192:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:215:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:16:def _make_df(n=300, trend=0.0):
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:46:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:70:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:91:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:98:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:106:        df = _make_df(n=300, trend=0.1)
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:117:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:125:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:133:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:141:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:157:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:165:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:235:        df = _make_df(n=300, trend=0.1)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:13:def _make_df(n=250):
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:31:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:39:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:58:        df = _make_df(n=50)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:66:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:197:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:205:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:18:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:31:def _make_ratio_df(n=200, trend="neutral"):
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:65:        df = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:71:        df = _make_ratio_df(10)
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:76:        df = _make_ratio_df(250, "rising")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:86:        df = _make_ratio_df(200, "rising")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:96:        df = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:106:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:107:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:117:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:118:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:128:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:129:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:145:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:152:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:159:        mock_fetch.return_value = _make_ratio_df(200, "falling")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:160:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:172:        mock_fetch.return_value = _make_ratio_df(200, "falling")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:173:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:189:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:190:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:199:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:200:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:206:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:207:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:223:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:224:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:233:        data = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:237:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:244:        data = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:248:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:14:def _make_df(n=200, seed=42):
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:27:def _make_trending_df(n=200, direction=1):
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:45:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:54:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:63:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:78:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:83:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:89:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:95:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:147:        df = _make_trending_df(n=200, direction=1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:152:        df = _make_trending_df(n=200, direction=1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:159:        df = _make_trending_df(n=200, direction=-1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:164:        df = _make_trending_df(n=200, direction=-1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:171:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:16:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:54:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:65:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:73:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:85:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:92:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:111:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:121:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:240:    def test_none_df(self, mock_fetch):
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:253:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:266:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:12:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:25:def _make_trending_df(n=100, direction="up"):
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:41:def _make_volatile_df(n=100):
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:75:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:81:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:93:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:98:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:104:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:119:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:126:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:133:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:139:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:144:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:153:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:161:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:172:        df = _make_trending_df(n=200, direction="up")
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:179:        df = _make_trending_df(n=200, direction="down")
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:184:        df = _make_volatile_df(n=200)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:189:        df = _make_trending_df(n=300, direction="up")
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:265:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:270:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:275:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_engine_core.py:29:def _make_ohlcv_df(n=50, close_start=100.0, trend="flat", volume_start=1000.0):
Q:\finance-analyzer\tests\test_signal_engine_core.py:62:def _make_trap_df(price_trend="up", volume_pattern="declining", n=50):
Q:\finance-analyzer\tests\test_signal_engine_core.py:68:    df = _make_ohlcv_df(n, close_start=100.0, trend=price_trend, volume_start=1000.0)
Q:\finance-analyzer\tests\test_signal_engine_core.py:428:        df = _make_ohlcv_df(50, trend="flat")
Q:\finance-analyzer\tests\test_signal_engine_core.py:462:        df = _make_trap_df(price_trend="up", volume_pattern="declining")
Q:\finance-analyzer\tests\test_signal_engine_core.py:472:        df = _make_trap_df(price_trend="down", volume_pattern="declining")
Q:\finance-analyzer\tests\test_signal_engine_core.py:480:        df = _make_trap_df(price_trend="up", volume_pattern="expanding")
Q:\finance-analyzer\tests\test_signal_engine_core.py:488:        df = _make_trap_df(price_trend="up", volume_pattern="declining")
Q:\finance-analyzer\tests\test_signal_engine_core.py:495:    def test_no_trap_with_short_df(self):
Q:\finance-analyzer\tests\test_signal_engine_core.py:496:        df = _make_ohlcv_df(3)
Q:\finance-analyzer\tests\test_signal_engine_core.py:740:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_engine_core.py:752:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_engine_core.py:763:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_engine_core.py:773:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:17:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:30:def _make_trending_df(n=100, direction="up"):
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:46:def _make_noisy_df(n=100):
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:106:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:114:        df = _make_noisy_df(100)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:130:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:136:        df = _make_trending_df(100, "down")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:151:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:166:        df = _make_df(150)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:181:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:190:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:196:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:210:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:215:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:221:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:234:        df = _make_trending_df(200, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:240:        df = _make_trending_df(200, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:248:        df = _make_trending_df(200, "down")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:255:        df = _make_noisy_df(200)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:278:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:284:        df = _make_df(n=MIN_ROWS)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:290:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:16:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:37:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:52:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:68:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:82:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:96:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:108:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:122:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:138:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:235:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:250:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:263:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vix_term_structure.py:16:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_vix_term_structure.py:205:    def test_with_df_parameter(self, mock_cached):
Q:\finance-analyzer\tests\test_signal_vix_term_structure.py:207:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:15:def _make_df(n=100, trend=0.0, noise=1.0, seed=42):
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:43:def _make_trending_df(n=100):
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:45:    return _make_df(n=n, trend=0.5, noise=0.3)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:48:def _make_ranging_df(n=100):
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:50:    return _make_df(n=n, trend=0.0, noise=2.0)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:72:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:94:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:100:        df = _make_df(n=MIN_ROWS)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:105:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:111:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:120:            df = _make_df(seed=seed)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:129:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:140:        trending_df = _make_trending_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:141:        ranging_df = _make_ranging_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:143:        trending_ratio = _garman_klass_cc_ratio(trending_df).iloc[-1]
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:144:        ranging_ratio = _garman_klass_cc_ratio(ranging_df).iloc[-1]
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:152:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:160:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:210:        df = _make_trending_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:218:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:225:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:231:        df = _make_df(n=100, trend=0.01, noise=0.5)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:255:        df = _make_df(n=100, noise=10.0)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:261:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:267:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:274:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:282:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:17:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:30:def _make_crash_df(n=100):
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:53:def _make_complacent_df(n=100):
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:70:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:80:        df = _make_crash_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:89:        df = _make_complacent_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:107:        df = _make_crash_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:116:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:130:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:158:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:167:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:184:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:194:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:200:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:208:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:214:        df = _make_crash_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:225:        df = _make_complacent_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:232:            df = _make_df(n=100, seed=seed)
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:239:        df = _make_df(n=500)
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:19:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:59:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:70:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:82:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:97:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:104:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:120:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:135:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:163:        df = _make_df()
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:23:    def _make_df(self, n=30, close_override=None):
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:36:        df = self._make_df(30)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:48:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:58:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:69:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:79:        df = self._make_df(10)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:86:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:93:        df = self._make_df(50)
Q:\finance-analyzer\tests\test_vix_fetch.py:28:def _patch_yf_ticker(hist_df):
Q:\finance-analyzer\tests\test_vix_fetch.py:31:    mock_ticker.history.return_value = hist_df
Q:\finance-analyzer\tests\test_train_signal_weights.py:60:        signals_df, returns = result
Q:\finance-analyzer\tests\test_train_signal_weights.py:61:        assert len(signals_df) == 100
Q:\finance-analyzer\tests\test_train_signal_weights.py:62:        assert "rsi" in signals_df.columns
Q:\finance-analyzer\tests\test_train_signal_weights.py:75:        signals_df, _ = result
Q:\finance-analyzer\tests\test_train_signal_weights.py:76:        assert "ml" not in signals_df.columns
Q:\finance-analyzer\tests\test_train_signal_weights.py:77:        assert "funding" not in signals_df.columns

 succeeded in 659ms:
13:def compute_indicators(df, horizon=None):
24:        logger.debug("compute_indicators: insufficient data (%d rows, need %d)", len(df), min_rows)
25:        return None
31:        return None
34:        return None
47:        return None

 succeeded in 703ms:
740:    except Exception:
741:        logger.debug("Failed to log A/B result", exc_info=True)
742:
743:
744:# ---------------------------------------------------------------------------
745:# Main entry point
746:# ---------------------------------------------------------------------------
747:
748:def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
749:                   *, cryptocompare_api_key=None) -> dict:
750:    """Get sentiment for a ticker using primary model + shadow A/B models.
751:
752:    2026-04-28 (fix/sentiment-relevance-and-aggregation): two changes here.
753:      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
754:         CryptoBERT was 99.1% neutral on 2,817 wire-feed samples (it was
755:         trained on crypto-twitter slang, not press-wire headlines).
756:         CryptoBERT now runs as a shadow for continuity of the 30d accuracy
757:         baseline. Trading-Hero is permabull on financial news but at least
758:         produces variance; the 47% directional accuracy gate still acts as
759:         a circuit breaker if it underperforms.
760:      2. Headlines pass through _filter_relevant_headlines() before model
761:         inference. Bare price-tickers ("Bitcoin: $67,123") and generic
762:         market-update boilerplate are dropped. Fallback keeps the most-
763:         recent N when the filter would drop everything.
764:
765:    Primary model (votes):  Trading-Hero-LLM (both crypto and stocks)
766:    Shadow models (logged): CryptoBERT (crypto only), FinGPT, FinBERT
767:
768:    Returns the primary model's result. Shadow results are logged to
769:    data/sentiment_ab_log.jsonl for accuracy tracking.
770:    """
---
870:    # entirely so the llama_server phase 3 skips the fingpt model swap +
871:    # inference cost. Fingpt is a shadow signal, so skipping 2 of 3 cycles
872:    # just reduces A/B sample density from every cycle to every 3rd cycle,
873:    # which is fine for long-running statistical comparison.
874:    try:
875:        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
876:        if is_llm_on_cycle("fingpt"):
877:            enqueue_fingpt(
878:                ab_key, "headlines",
879:                {"mode": "headlines", "texts": titles, "ticker": short},
880:            )
881:            clusters = _cluster_headlines(all_articles)
882:            for idx, cluster in enumerate(clusters):
883:                if len(cluster) >= 3:
884:                    cluster_titles = [a["title"] for a in cluster]
885:                    enqueue_fingpt(
886:                        ab_key, f"cumul:{idx}",
887:                        {"mode": "cumulative", "texts": cluster_titles, "ticker": short},
888:                    )
889:    except Exception as e:
890:        logger.debug("FinGPT enqueue failed: %s", e)
891:
892:    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
893:    # and on CPU (no model swap cost) and we'd rather not add a fourth
894:    # phase to llm_batch for an already-shadow-of-shadow signal. Stash its
895:    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
---

codex
I’m checking whether the “keep raw OHLCV even when indicators fail” claim conflicts with the subsystem’s tested contract. If the tests only preserve `_df` on the `"Now"` frame, Claude’s “whole timeframe entry dropped” claim is overselling the design intent.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\tests\\test_data_collector.py'; "'$ranges=@(@(560,640),@(800,835)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\indicators.py'; "'$start=1; $end=60; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_fetch_earnings_alpha_vantage|_daily_budget_used|_check_budget|budget exhausted|daily budget\" Q:\\finance-analyzer\\tests Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"headlines\\]\" Q:\\finance-analyzer\\portfolio\\llm_batch.py Q:\\finance-analyzer\\portfolio\\sentiment.py" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 584ms:
Q:\finance-analyzer\portfolio\alpha_vantage.py:31:_daily_budget_used = 0
Q:\finance-analyzer\portfolio\alpha_vantage.py:157:def _check_budget():
Q:\finance-analyzer\portfolio\alpha_vantage.py:158:    """Check and reset daily budget counter. Returns current usage count.
Q:\finance-analyzer\portfolio\alpha_vantage.py:162:    global _daily_budget_used, _budget_reset_date
Q:\finance-analyzer\portfolio\alpha_vantage.py:166:            _daily_budget_used = 0
Q:\finance-analyzer\portfolio\alpha_vantage.py:168:        return _daily_budget_used
Q:\finance-analyzer\portfolio\alpha_vantage.py:207:    Respects rate limits (5/min) and daily budget (25/day).
Q:\finance-analyzer\portfolio\alpha_vantage.py:210:    global _daily_budget_used
Q:\finance-analyzer\portfolio\alpha_vantage.py:231:    budget_used = _check_budget()
Q:\finance-analyzer\portfolio\alpha_vantage.py:233:        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
Q:\finance-analyzer\portfolio\alpha_vantage.py:281:                _daily_budget_used += 1  # BUG-108: increment under lock
Q:\finance-analyzer\portfolio\earnings_calendar.py:31:def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:49:        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
Q:\finance-analyzer\portfolio\earnings_calendar.py:147:    result = _fetch_earnings_alpha_vantage(ticker)
Q:\finance-analyzer\tests\test_alpha_vantage.py:234:        alpha_vantage._daily_budget_used = 0
Q:\finance-analyzer\tests\test_alpha_vantage.py:289:        alpha_vantage._daily_budget_used = 25
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:102:    """_daily_budget_used must be protected by lock."""
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:104:    def test_check_budget_uses_lock(self):
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:105:        """_check_budget should use _cache_lock to protect budget counter."""
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:111:        original_used = av._daily_budget_used
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:118:                av._daily_budget_used = 10
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:121:            # _check_budget should return current count under lock
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:122:            count = av._check_budget()
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:126:                av._daily_budget_used = original_used
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:133:        original_used = av._daily_budget_used
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:137:            av._daily_budget_used = 0
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:146:                            av._daily_budget_used += 1
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:157:            assert av._daily_budget_used == 400
Q:\finance-analyzer\tests\test_bug_fixes_session_mar22.py:159:            av._daily_budget_used = original_used
Q:\finance-analyzer\portfolio\shared_state.py:347:            logger.warning("NewsAPI daily budget exhausted (%d/%d), falling back to Yahoo",

 succeeded in 590ms:
1:"""Technical indicator computation and regime detection."""
2:
3:import logging
4:
5:import numpy as np
6:import pandas as pd
7:
8:import portfolio.shared_state as _ss
9:
10:logger = logging.getLogger("portfolio.indicators")
11:
12:
13:def compute_indicators(df, horizon=None):
14:    if horizon == "3h":
15:        rsi_period = 7
16:        macd_fast, macd_slow, macd_signal_period = 8, 17, 9
17:        min_rows = macd_slow  # 17
18:    else:
19:        rsi_period = 14
20:        macd_fast, macd_slow, macd_signal_period = 12, 26, 9
21:        min_rows = macd_slow  # 26
22:
23:    if len(df) < min_rows:
24:        logger.debug("compute_indicators: insufficient data (%d rows, need %d)", len(df), min_rows)
25:        return None
26:    close = df["close"].copy()
27:
28:    # BUG-87: Guard against NaN in close series
29:    if close.iloc[-1] != close.iloc[-1]:  # NaN check (NaN != NaN is True)
30:        logger.warning("compute_indicators: last close is NaN, returning None")
31:        return None
32:    if close.isna().all():
33:        logger.warning("compute_indicators: all close values are NaN, returning None")
34:        return None
35:    # Forward-fill interior NaN gaps to prevent downstream NaN propagation
36:    if close.isna().any():
37:        logger.debug("compute_indicators: forward-filling %d NaN close values", close.isna().sum())
38:        close = close.ffill().bfill()
39:        df = df.copy()
40:        df["close"] = close
41:
42:    # BUG-209: Guard against zero/negative prices (Binance maintenance, API glitches).
43:    # A single zero-price candle produces RSI=50, MACD=0, ATR=0 — poisoning consensus.
44:    bad_prices = (close <= 0).sum()
45:    if bad_prices > 0:
46:        logger.warning("compute_indicators: %d zero/negative close values detected, returning None", bad_prices)
47:        return None
48:
49:    # RSI(rsi_period)
50:    delta = close.diff()
51:    gain = delta.where(delta > 0, 0.0)
52:    loss = (-delta).where(delta < 0, 0.0)
53:    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
54:    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
55:    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
56:    rs = avg_gain / avg_loss_safe
57:    rsi = 100 - (100 / (1 + rs))
58:
59:    # MACD(macd_fast, macd_slow, macd_signal_period)
60:    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()

 succeeded in 617ms:
560:# ===========================================================================
561:
562:class TestCollectTimeframes:
563:
564:    @patch("portfolio.data_collector._fetch_klines")
565:    @patch("portfolio.data_collector.compute_indicators")
566:    @patch("portfolio.data_collector.technical_signal")
567:    def test_crypto_uses_crypto_timeframes(self, mock_ts, mock_ci, mock_fk):
568:        """Crypto source uses TIMEFRAMES (not STOCK_TIMEFRAMES)."""
569:        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
570:        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
571:        mock_ts.return_value = ("HOLD", 0.5)
572:
573:        results = collect_timeframes({"binance": "BTCUSDT"})
574:
575:        # Should have entries for all crypto timeframes
576:        labels = [label for label, _ in results]
577:        expected_labels = [label for label, _, _, _ in TIMEFRAMES]
578:        assert labels == expected_labels
579:
580:    @patch("portfolio.data_collector._fetch_klines")
581:    @patch("portfolio.data_collector.compute_indicators")
582:    @patch("portfolio.data_collector.technical_signal")
583:    def test_stock_uses_stock_timeframes(self, mock_ts, mock_ci, mock_fk):
584:        """Alpaca source uses STOCK_TIMEFRAMES."""
585:        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
586:        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
587:        mock_ts.return_value = ("HOLD", 0.5)
588:
589:        results = collect_timeframes({"alpaca": "NVDA"})
590:
591:        labels = [label for label, _ in results]
592:        expected_labels = [label for label, _, _, _ in STOCK_TIMEFRAMES]
593:        assert labels == expected_labels
594:
595:    @patch("portfolio.data_collector._fetch_klines")
596:    @patch("portfolio.data_collector.compute_indicators")
597:    @patch("portfolio.data_collector.technical_signal")
598:    def test_now_timeframe_skips_technical_signal(self, mock_ts, mock_ci, mock_fk):
599:        """The 'Now' timeframe does not call technical_signal (action/conf are None)."""
600:        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
601:        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
602:        mock_ts.return_value = ("BUY", 0.8)
603:
604:        results = collect_timeframes({"binance": "BTCUSDT"})
605:
606:        now_entry = None
607:        for label, data in results:
608:            if label == "Now":
609:                now_entry = data
610:                break
611:
612:        assert now_entry is not None
613:        assert now_entry["action"] is None
614:        assert now_entry["confidence"] is None
615:        assert "_df" in now_entry  # raw DataFrame preserved for enhanced signals
616:
617:    @patch("portfolio.data_collector._fetch_klines")
618:    @patch("portfolio.data_collector.compute_indicators")
619:    @patch("portfolio.data_collector.technical_signal")
620:    def test_non_now_timeframes_call_technical_signal(self, mock_ts, mock_ci, mock_fk):
621:        """Non-Now timeframes call technical_signal and include action/confidence."""
622:        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
623:        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
624:        mock_ts.return_value = ("BUY", 0.75)
625:
626:        results = collect_timeframes({"binance": "BTCUSDT"})
627:
628:        non_now_entries = [(label, data) for label, data in results if label != "Now"]
629:        assert len(non_now_entries) > 0
630:        for _label, data in non_now_entries:
631:            if "error" not in data:
632:                assert data["action"] == "BUY"
633:                assert data["confidence"] == 0.75
634:
635:    @patch("portfolio.data_collector._fetch_klines")
636:    @patch("portfolio.data_collector.compute_indicators")
637:    def test_insufficient_data_skips_timeframe(self, mock_ci, mock_fk):
638:        """When compute_indicators returns None (insufficient data), the timeframe is skipped."""
639:        mock_fk.return_value = pd.DataFrame({"close": [1.0, 2.0]})  # too few rows
640:        mock_ci.return_value = None  # insufficient data
---
800:
801:# ---------------------------------------------------------------------------
802:# BUG-179: collect_timeframes timeout protection
803:# ---------------------------------------------------------------------------
804:
805:class TestCollectTimeframesTimeout:
806:    """BUG-179: Verify collect_timeframes handles stuck fetches gracefully."""
807:
808:    @patch("portfolio.data_collector._TF_POOL_TIMEOUT", 0.1)
809:    @patch("portfolio.data_collector._fetch_one_timeframe")
810:    def test_timeout_returns_partial_results(self, mock_fetch):
811:        """When some fetches hang, completed results are still returned."""
812:        import threading
813:
814:        hang_event = threading.Event()
815:
816:        def _side_effect(source, key, label, interval, limit, ttl):
817:            if label == "Now":
818:                return (label, {"indicators": {"rsi": 50}, "action": None,
819:                                "confidence": None, "_df": pd.DataFrame()})
820:            # Simulate hang on other timeframes
821:            hang_event.wait(timeout=5)
822:            return None
823:
824:        mock_fetch.side_effect = _side_effect
825:
826:        results = collect_timeframes({"binance": "BTCUSDT"})
827:        hang_event.set()  # unblock threads for cleanup
828:
829:        # Should have at least the "Now" result that completed
830:        labels = [label for label, _ in results]
831:        assert "Now" in labels
832:
833:    @patch("portfolio.data_collector._TF_POOL_TIMEOUT", 0.1)
834:    @patch("portfolio.data_collector._fetch_one_timeframe")
835:    def test_timeout_does_not_raise(self, mock_fetch):
---

2026-05-10T15:28:38.733114Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 598ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\tests\\test_data_collector.py'; "'$start=637; $end=660; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 488ms:
637:    def test_insufficient_data_skips_timeframe(self, mock_ci, mock_fk):
638:        """When compute_indicators returns None (insufficient data), the timeframe is skipped."""
639:        mock_fk.return_value = pd.DataFrame({"close": [1.0, 2.0]})  # too few rows
640:        mock_ci.return_value = None  # insufficient data
641:
642:        results = collect_timeframes({"binance": "BTCUSDT"})
643:
644:        # All timeframes should be skipped (none in results)
645:        assert len(results) == 0
646:
647:    @patch("portfolio.data_collector._fetch_klines")
648:    @patch("portfolio.data_collector.compute_indicators")
649:    @patch("portfolio.data_collector.technical_signal")
650:    def test_cache_hit_returns_cached_data(self, mock_ts, mock_ci, mock_fk):
651:        """Cached timeframes are returned without calling _fetch_klines again."""
652:        # Pre-populate cache for "12h" timeframe (TTL=300s)
653:        cache_key = "tf_BTCUSDT_12h"
654:        cached_entry = {"indicators": {"rsi": 42.0}, "action": "SELL", "confidence": 0.6}
655:        _ss._tool_cache[cache_key] = {"data": cached_entry, "time": time.time()}
656:
657:        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
658:        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
659:        mock_ts.return_value = ("HOLD", 0.5)
660:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"def enqueue_fingpt|_stash_fingpt_result|flush_ab_log|fingpt\" Q:\\finance-analyzer\\portfolio\\llm_batch.py Q:\\finance-analyzer\\portfolio\\sentiment.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"sentiment_ab_log|shadow\\[|FinGPT|fingpt\" Q:\\finance-analyzer\\portfolio\\accuracy_stats.py Q:\\finance-analyzer\\tests" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\llm_batch.py'; "'$ranges=@(@(30,120),@(650,760)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 543ms:
Q:\finance-analyzer\portfolio\llm_batch.py:6:query all tickers → swap to Qwen3 once → query all tickers → swap to fingpt
Q:\finance-analyzer\portfolio\llm_batch.py:14:2026-04-09 update (feat/fingpt-in-llmbatch): added Phase 3 for fingpt
Q:\finance-analyzer\portfolio\llm_batch.py:15:sentiment. Previously fingpt ran in its own bespoke NDJSON daemon
Q:\finance-analyzer\portfolio\llm_batch.py:16:(scripts/fingpt_daemon.py) on CPU (~60-150s/cycle). Moving it into this
Q:\finance-analyzer\portfolio\llm_batch.py:19:reduction in fingpt inference time. See project_fingpt_llmbatch_session
Q:\finance-analyzer\portfolio\llm_batch.py:32:# _fingpt_queue entries are (ab_key, sub_key, context) — sub_key is
Q:\finance-analyzer\portfolio\llm_batch.py:34:# cluster. The ab_key is shared by all fingpt calls for a single ticker's
Q:\finance-analyzer\portfolio\llm_batch.py:36:# sentiment_ab_log.jsonl entry by sentiment.flush_ab_log() post-cycle.
Q:\finance-analyzer\portfolio\llm_batch.py:37:_fingpt_queue: list[tuple[str, str, dict]] = []
Q:\finance-analyzer\portfolio\llm_batch.py:44:# + 9 s fingpt + 15-18 s of swaps) to running ONE model per cycle (~25-40 s
Q:\finance-analyzer\portfolio\llm_batch.py:59:#    also need to be callable directly (from sentiment.py for fingpt) without
Q:\finance-analyzer\portfolio\llm_batch.py:70:_LLM_ROTATION: tuple[str, ...] = ("ministral", "qwen3", "fingpt")
Q:\finance-analyzer\portfolio\llm_batch.py:86:        counter == 3  → slot 2 → fingpt only
Q:\finance-analyzer\portfolio\llm_batch.py:117:def enqueue_fingpt(ab_key: str, sub_key: str, context: dict) -> None:
Q:\finance-analyzer\portfolio\llm_batch.py:118:    """Add a fingpt sentiment request to the batch queue.
Q:\finance-analyzer\portfolio\llm_batch.py:122:            (e.g. "BTC:2026-04-09T18:04:00+00:00"). All fingpt calls for
Q:\finance-analyzer\portfolio\llm_batch.py:134:        if not any(k == ab_key and s == sub_key for k, s, _ in _fingpt_queue):
Q:\finance-analyzer\portfolio\llm_batch.py:135:            _fingpt_queue.append((ab_key, sub_key, context))
Q:\finance-analyzer\portfolio\llm_batch.py:170:    via _stash_fingpt_result() and emitted later by sentiment.flush_ab_log().
Q:\finance-analyzer\portfolio\llm_batch.py:175:    ministral → qwen3 → fingpt across successive flushes.
Q:\finance-analyzer\portfolio\llm_batch.py:180:        f_batch = list(_fingpt_queue)
Q:\finance-analyzer\portfolio\llm_batch.py:183:        _fingpt_queue.clear()
Q:\finance-analyzer\portfolio\llm_batch.py:202:    m_parsed = 0  # 2026-05-03 (fix/fingpt-batch-observability): track per-phase
Q:\finance-analyzer\portfolio\llm_batch.py:251:    # Phase 3: All fingpt sentiment queries (lock held for entire phase)
Q:\finance-analyzer\portfolio\llm_batch.py:252:    # Added 2026-04-09 (feat/fingpt-in-llmbatch). Replaces the bespoke
Q:\finance-analyzer\portfolio\llm_batch.py:253:    # scripts/fingpt_daemon.py NDJSON daemon with the shared llama_server
Q:\finance-analyzer\portfolio\llm_batch.py:254:    # rotation, trading ~1 extra swap for a ~70-120s reduction in fingpt
Q:\finance-analyzer\portfolio\llm_batch.py:261:                   # the fingpt parser stage that produces one parsed dict
Q:\finance-analyzer\portfolio\llm_batch.py:264:        # 2026-05-03 (fix/fingpt-batch-observability codex P3): renamed from
Q:\finance-analyzer\portfolio\llm_batch.py:265:        # "%d fingpt queries" because each f_batch entry can fan out to many
Q:\finance-analyzer\portfolio\llm_batch.py:266:        # per-headline prompts in _flush_fingpt_phase. The old wording showed
Q:\finance-analyzer\portfolio\llm_batch.py:267:        # "1 fingpt queries" right next to a summary line claiming "F=10/10",
Q:\finance-analyzer\portfolio\llm_batch.py:269:        logger.info("LLM batch: %d fingpt groups", len(f_batch))
Q:\finance-analyzer\portfolio\llm_batch.py:270:        # 2026-05-03 (fix/fingpt-batch-observability): _flush_fingpt_phase now
Q:\finance-analyzer\portfolio\llm_batch.py:272:        # below so a fingpt-only cycle no longer reports "0 results" when
Q:\finance-analyzer\portfolio\llm_batch.py:273:        # fingpt actually stashed its outputs to sentiment._pending_ab_entries.
Q:\finance-analyzer\portfolio\llm_batch.py:274:        f_metrics = _flush_fingpt_phase(f_batch)
Q:\finance-analyzer\portfolio\llm_batch.py:279:    # 2026-05-03 (fix/fingpt-batch-observability): replaced the old line
Q:\finance-analyzer\portfolio\llm_batch.py:281:    # only Phase 1+2 entries in the local `results` dict — fingpt-only
Q:\finance-analyzer\portfolio\llm_batch.py:283:    # New format shows parsed/queued for each phase so silent fingpt
Q:\finance-analyzer\portfolio\llm_batch.py:286:    # fingpt uses the metrics-tracked query count because one queue entry
Q:\finance-analyzer\portfolio\llm_batch.py:305:def _flush_fingpt_phase(f_batch: list[tuple[str, str, dict]]) -> dict:
Q:\finance-analyzer\portfolio\llm_batch.py:314:          "parsed": int,          # parsed dicts (non-None) handed to _stash_fingpt_result
Q:\finance-analyzer\portfolio\llm_batch.py:320:    logger which writes a tagged fingpt:error entry instead of silently
Q:\finance-analyzer\portfolio\llm_batch.py:323:    The whole phase is wrapped in try/except so fingpt errors never leak
Q:\finance-analyzer\portfolio\llm_batch.py:328:    2026-05-03 (fix/fingpt-batch-observability): added metrics return +
Q:\finance-analyzer\portfolio\llm_batch.py:343:        # fingpt_infer provides the prompt templates, stop tokens, and
Q:\finance-analyzer\portfolio\llm_batch.py:346:        # — if the import fails, fingpt just doesn't run this cycle.
Q:\finance-analyzer\portfolio\llm_batch.py:355:        import fingpt_infer  # noqa: E402  (path injection above)
Q:\finance-analyzer\portfolio\llm_batch.py:366:                prompt = fingpt_infer.CUMULATIVE_PROMPT.format(
Q:\finance-analyzer\portfolio\llm_batch.py:374:                    # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
Q:\finance-analyzer\portfolio\llm_batch.py:389:                template = fingpt_infer.PROMPT_TEMPLATES.get(
Q:\finance-analyzer\portfolio\llm_batch.py:391:                    next(iter(fingpt_infer.PROMPT_TEMPLATES.values())),
Q:\finance-analyzer\portfolio\llm_batch.py:398:                        # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
Q:\finance-analyzer\portfolio\llm_batch.py:425:                "fingpt: server returned None for all %d prompts "
Q:\finance-analyzer\portfolio\llm_batch.py:433:            parsed = _parse_fingpt_completion(text, fingpt_infer)
Q:\finance-analyzer\portfolio\llm_batch.py:440:        # broke upstream in fingpt_infer (template change, model swap to a
Q:\finance-analyzer\portfolio\llm_batch.py:450:                "fingpt: parser returned None for %d/%d completions "
Q:\finance-analyzer\portfolio\llm_batch.py:457:        # buffer is consumed by sentiment.flush_ab_log() which runs right
Q:\finance-analyzer\portfolio\llm_batch.py:460:        from portfolio.sentiment import _stash_fingpt_result
Q:\finance-analyzer\portfolio\llm_batch.py:477:                    parsed["model"] = "fingpt:cumulative"
Q:\finance-analyzer\portfolio\llm_batch.py:478:                _stash_fingpt_result(ab_key, sub_key, parsed)
Q:\finance-analyzer\portfolio\llm_batch.py:481:                _stash_fingpt_result(ab_key, sub_key, per_headline)
Q:\finance-analyzer\portfolio\llm_batch.py:489:        logger.warning("LLM batch fingpt phase failed: %s", repr(e), exc_info=True)
Q:\finance-analyzer\portfolio\llm_batch.py:493:def _parse_fingpt_completion(text: str | None, fingpt_infer) -> dict | None:
Q:\finance-analyzer\portfolio\llm_batch.py:496:    logger which writes a tagged fingpt:error entry).
Q:\finance-analyzer\portfolio\llm_batch.py:498:    2026-04-09 (fix/fingpt-parser-prompt): the original fingpt migration
Q:\finance-analyzer\portfolio\llm_batch.py:500:    fingpt_infer._parse_sentiment / _estimate_confidence + the Llama-3
Q:\finance-analyzer\portfolio\llm_batch.py:504:    it emit clean sentiment words. See /mnt/q/models/fingpt_infer.py for
Q:\finance-analyzer\portfolio\llm_batch.py:507:    2026-05-03 (fix/fingpt-batch-observability codex P2): also reject
Q:\finance-analyzer\portfolio\llm_batch.py:508:    empty / whitespace-only text. The production fingpt_infer._parse_sentiment
Q:\finance-analyzer\portfolio\llm_batch.py:513:    _flush_fingpt_phase looks healthy (parsed == queries) when in fact
Q:\finance-analyzer\portfolio\llm_batch.py:521:        sentiment = fingpt_infer._parse_sentiment(text)
Q:\finance-analyzer\portfolio\llm_batch.py:522:        confidence = fingpt_infer._estimate_confidence(text, sentiment)
Q:\finance-analyzer\portfolio\llm_batch.py:526:        other_labels = [lb for lb in fingpt_infer.SENTIMENT_LABELS if lb != sentiment]
Q:\finance-analyzer\portfolio\llm_batch.py:535:            "model": "fingpt:finance-llama-8b",
Q:\finance-analyzer\portfolio\llm_batch.py:538:        logger.debug("fingpt completion parse failed for text=%r", text, exc_info=True)
Q:\finance-analyzer\portfolio\sentiment.py:300:# ── Deferred fingpt A/B buffering ──────────────────────────────────────────
Q:\finance-analyzer\portfolio\sentiment.py:305:# Historical note (2026-04-09, feat/fingpt-in-llmbatch):
Q:\finance-analyzer\portfolio\sentiment.py:308:# - v2: warm NDJSON daemon (scripts/fingpt_daemon.py) on GPU full offload
Q:\finance-analyzer\portfolio\sentiment.py:313:# - v4 (current): fingpt runs in portfolio.llm_batch as Phase 3 of the
Q:\finance-analyzer\portfolio\sentiment.py:318:# Because fingpt now runs post-cycle in a batched phase, its results arrive
Q:\finance-analyzer\portfolio\sentiment.py:322:# + the raw headlines + the enqueued fingpt sub_keys into
Q:\finance-analyzer\portfolio\sentiment.py:324:# in main.py, sentiment.flush_ab_log() walks the pending entries, merges the
Q:\finance-analyzer\portfolio\sentiment.py:325:# batched fingpt results into each, and writes the final A/B log rows.
Q:\finance-analyzer\portfolio\sentiment.py:329:# shadow[] array containing fingpt per-headline + cumulative + finbert.
Q:\finance-analyzer\portfolio\sentiment.py:331:# Known open issue: sentiment_ab_log.jsonl shows fingpt returning constant
Q:\finance-analyzer\portfolio\sentiment.py:333:# project_fingpt_parser_defaulting_neutral memory. That is a parser / prompt
Q:\finance-analyzer\portfolio\sentiment.py:334:# bug in /mnt/q/models/fingpt_infer.py, NOT a problem with this migration.
Q:\finance-analyzer\portfolio\sentiment.py:348:    """Store the inline portion of an A/B entry until the batched fingpt
Q:\finance-analyzer\portfolio\sentiment.py:349:    results arrive in flush_ab_log(). Called from get_sentiment().
Q:\finance-analyzer\portfolio\sentiment.py:364:            "fingpt_headlines_raw": None,  # filled in by Phase 3
Q:\finance-analyzer\portfolio\sentiment.py:365:            "fingpt_cumulatives_raw": {},  # sub_key → raw dict, filled in by Phase 3
Q:\finance-analyzer\portfolio\sentiment.py:389:def _stash_fingpt_result(ab_key: str, sub_key: str, result) -> None:
Q:\finance-analyzer\portfolio\sentiment.py:390:    """Called from portfolio.llm_batch._flush_fingpt_phase with the parsed
Q:\finance-analyzer\portfolio\sentiment.py:391:    fingpt result for one (ab_key, sub_key) tuple.
Q:\finance-analyzer\portfolio\sentiment.py:400:            # happen if enqueue_fingpt ran but the parent get_sentiment
Q:\finance-analyzer\portfolio\sentiment.py:404:            entry["fingpt_headlines_raw"] = result
Q:\finance-analyzer\portfolio\sentiment.py:406:            entry["fingpt_cumulatives_raw"][sub_key] = result
Q:\finance-analyzer\portfolio\sentiment.py:409:def flush_ab_log() -> None:
Q:\finance-analyzer\portfolio\sentiment.py:410:    """Walk _pending_ab_entries, merge batched fingpt results into shadow
Q:\finance-analyzer\portfolio\sentiment.py:414:    finishes Phase 3. Safe to call even if some fingpt results are missing
Q:\finance-analyzer\portfolio\sentiment.py:417:    logging a fingpt:error entry.
Q:\finance-analyzer\portfolio\sentiment.py:437:            fingpt_raw = entry.get("fingpt_headlines_raw")
Q:\finance-analyzer\portfolio\sentiment.py:438:            if fingpt_raw:
Q:\finance-analyzer\portfolio\sentiment.py:440:                usable = [r for r in fingpt_raw if r is not None]
Q:\finance-analyzer\portfolio\sentiment.py:449:                            "model": usable[0].get("model", "fingpt:finance-llama-8b"),
Q:\finance-analyzer\portfolio\sentiment.py:456:                            "fingpt headlines aggregation failed for %s", ab_key,
Q:\finance-analyzer\portfolio\sentiment.py:461:            for _sub_key in sorted(entry.get("fingpt_cumulatives_raw", {})):
Q:\finance-analyzer\portfolio\sentiment.py:462:                cum = entry["fingpt_cumulatives_raw"][_sub_key]
Q:\finance-analyzer\portfolio\sentiment.py:466:                    "model": cum.get("model", "fingpt:cumulative"),
Q:\finance-analyzer\portfolio\sentiment.py:487:            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)
Q:\finance-analyzer\portfolio\sentiment.py:847:    # this function. It is now DEFERRED to flush_ab_log() which is called
Q:\finance-analyzer\portfolio\sentiment.py:849:    # (fingpt sentiment). Rationale: fingpt used to run in a bespoke NDJSON
Q:\finance-analyzer\portfolio\sentiment.py:850:    # daemon (scripts/fingpt_daemon.py, now retired) blocking inside this
Q:\finance-analyzer\portfolio\sentiment.py:851:    # function; moving fingpt into portfolio.llm_batch's shared llama_server
Q:\finance-analyzer\portfolio\sentiment.py:852:    # rotation means the fingpt result does not arrive until AFTER
Q:\finance-analyzer\portfolio\sentiment.py:855:    # and let flush_ab_log() assemble the final row.
Q:\finance-analyzer\portfolio\sentiment.py:868:    # fingpt enqueues directly because it doesn't use the signal cache — it
Q:\finance-analyzer\portfolio\sentiment.py:870:    # entirely so the llama_server phase 3 skips the fingpt model swap +
Q:\finance-analyzer\portfolio\sentiment.py:875:        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
Q:\finance-analyzer\portfolio\sentiment.py:876:        if is_llm_on_cycle("fingpt"):
Q:\finance-analyzer\portfolio\sentiment.py:877:            enqueue_fingpt(
Q:\finance-analyzer\portfolio\sentiment.py:885:                    enqueue_fingpt(
Q:\finance-analyzer\portfolio\sentiment.py:895:    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
Q:\finance-analyzer\portfolio\sentiment.py:915:    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.

 succeeded in 583ms:
30:_ministral_queue: list[tuple[str, dict]] = []   # (cache_key, context)
31:_qwen3_queue: list[tuple[str, dict]] = []       # (cache_key, context)
32:# _fingpt_queue entries are (ab_key, sub_key, context) — sub_key is
33:# "headlines" for per-headline inference or "cumul:<N>" for a cumulative
34:# cluster. The ab_key is shared by all fingpt calls for a single ticker's
35:# get_sentiment() invocation so the results can be stitched back into one
36:# sentiment_ab_log.jsonl entry by sentiment.flush_ab_log() post-cycle.
37:_fingpt_queue: list[tuple[str, str, dict]] = []
38:
39:
40:# 2026-04-10 (perf/llama-swap-reduction) — ROTATION SCHEDULING
41:#
42:# Rotation across the three llama-server LLMs reduces the LLM batch phase
43:# from running all 3 models every cycle (~85 s: 40 s Ministral + 19 s Qwen3
44:# + 9 s fingpt + 15-18 s of swaps) to running ONE model per cycle (~25-40 s
45:# depending on which one). Each LLM still gets a fresh vote every 3rd full-
46:# LLM batch, and _cached_or_enqueue returns stale data on the off-cycle 2
47:# of 3 cycles (max staleness is bounded by max_stale_factor=5 passed at the
48:# call site in signal_engine.py / sentiment.py).
49:#
50:# Design decisions (see docs/PLAN.md / plan file for full rationale):
51:#
52:# 1. Counter lives in shared_state._full_llm_cycle_count, increments AFTER
53:#    flush when the batch actually had work. In-memory only — restart resets
54:#    to 0 and triggers a warmup cycle that runs all three models to establish
55:#    a baseline before the rotation begins.
56:#
57:# 2. Rotation gate sits at the _cached_or_enqueue caller via should_enqueue_fn,
58:#    NOT inside enqueue_ministral/qwen3 themselves, because the enqueue helpers
59:#    also need to be callable directly (from sentiment.py for fingpt) without
60:#    going through _cached_or_enqueue. Gating inside the enqueue functions
61:#    would also poison _loading_keys when the rotation skips.
62:#
63:# 3. Counter advances once per flush-with-work, not once per loop iteration.
64:#    That way rotation is driven by actual LLM invocations, not by idle cache-
65:#    hit cycles where nothing needs to run.
66:#
67:# 4. Warmup: on the very first flush after process start (counter == 0), ALL
68:#    LLMs run so we have a full baseline before rotation kicks in. Subsequent
69:#    flushes rotate.
70:_LLM_ROTATION: tuple[str, ...] = ("ministral", "qwen3", "fingpt")
71:
72:
73:def is_llm_on_cycle(llm_name: str) -> bool:
74:    """Return True if `llm_name` is scheduled to run during the current cycle.
75:
76:    Called at enqueue time to decide whether to skip the enqueue. The current
77:    cycle's slot is `(shared_state._full_llm_cycle_count - 1) % 3` because
78:    the counter advances AFTER the flush — at enqueue time, the counter
79:    represents "how many flushes have already completed" and the next slot
80:    is `counter % 3`, but we want to treat "counter == 0" as a warmup in
81:    which everything runs. So:
82:
83:        counter == 0  → warmup → every LLM returns True
84:        counter == 1  → slot 0 → ministral only
85:        counter == 2  → slot 1 → qwen3 only
86:        counter == 3  → slot 2 → fingpt only
87:        counter == 4  → slot 0 → ministral again
88:        ...
89:
90:    Unknown llm_name raises ValueError (from tuple.index) — that's a
91:    programming error we want to catch in tests rather than silently
92:    return False.
93:    """
94:    from portfolio import shared_state as _ss
95:    count = _ss._full_llm_cycle_count
96:    if count == 0:
97:        return True  # warmup — run everything the first time through
98:    idx = _LLM_ROTATION.index(llm_name)  # raises ValueError for bad names
99:    slot = (count - 1) % len(_LLM_ROTATION)
100:    return slot == idx
101:
102:
103:def enqueue_ministral(cache_key, context):
104:    """Add a Ministral cache miss to the batch queue."""
105:    with _lock:
106:        if not any(k == cache_key for k, _ in _ministral_queue):
107:            _ministral_queue.append((cache_key, context))
108:
109:
110:def enqueue_qwen3(cache_key, context):
111:    """Add a Qwen3 cache miss to the batch queue."""
112:    with _lock:
113:        if not any(k == cache_key for k, _ in _qwen3_queue):
114:            _qwen3_queue.append((cache_key, context))
115:
116:
117:def enqueue_fingpt(ab_key: str, sub_key: str, context: dict) -> None:
118:    """Add a fingpt sentiment request to the batch queue.
119:
120:    Args:
---
---

 succeeded in 592ms:
Q:\finance-analyzer\tests\test_llm_batch.py:1:"""Tests for portfolio/llm_batch.py — focused on the fingpt Phase 3 added
Q:\finance-analyzer\tests\test_llm_batch.py:2:2026-04-09 as part of feat/fingpt-in-llmbatch. Phases 1 (ministral) and 2
Q:\finance-analyzer\tests\test_llm_batch.py:4:covers the new fingpt path end-to-end with mocked dependencies so the suite
Q:\finance-analyzer\tests\test_llm_batch.py:7:The tests drive `portfolio.llm_batch._flush_fingpt_phase` directly (rather
Q:\finance-analyzer\tests\test_llm_batch.py:10:- `sys.modules["fingpt_infer"]` with a stub providing `PROMPT_TEMPLATES`,
Q:\finance-analyzer\tests\test_llm_batch.py:15:- `portfolio.sentiment._stash_fingpt_result` with a recorder so the test
Q:\finance-analyzer\tests\test_llm_batch.py:29:def fake_fingpt_infer(monkeypatch):
Q:\finance-analyzer\tests\test_llm_batch.py:30:    """Install a stub `fingpt_infer` module in sys.modules so Phase 3 can
Q:\finance-analyzer\tests\test_llm_batch.py:33:    fake = types.ModuleType("fingpt_infer")
Q:\finance-analyzer\tests\test_llm_batch.py:55:    monkeypatch.setitem(sys.modules, "fingpt_infer", fake)
Q:\finance-analyzer\tests\test_llm_batch.py:88:    """Replace sentiment._stash_fingpt_result with a recorder. Returns a
Q:\finance-analyzer\tests\test_llm_batch.py:97:    monkeypatch.setattr(sentiment_mod, "_stash_fingpt_result", recorder)
Q:\finance-analyzer\tests\test_llm_batch.py:101:def test_fingpt_phase_empty_queue(fake_fingpt_infer, fake_llama_server, stash_recorder):
Q:\finance-analyzer\tests\test_llm_batch.py:103:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:104:    _flush_fingpt_phase([])
Q:\finance-analyzer\tests\test_llm_batch.py:108:def test_fingpt_phase_headlines(fake_fingpt_infer, fake_llama_server, stash_recorder):
Q:\finance-analyzer\tests\test_llm_batch.py:113:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:114:    _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:134:    assert result[0]["model"] == "fingpt:finance-llama-8b"
Q:\finance-analyzer\tests\test_llm_batch.py:141:def test_fingpt_phase_cumulative(fake_fingpt_infer, fake_llama_server, stash_recorder):
Q:\finance-analyzer\tests\test_llm_batch.py:146:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:147:    _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:165:    assert result["model"] == "fingpt:cumulative"
Q:\finance-analyzer\tests\test_llm_batch.py:169:def test_fingpt_phase_cumulative_no_boost_under_five(
Q:\finance-analyzer\tests\test_llm_batch.py:170:    fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:176:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:177:    _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:194:def test_fingpt_phase_error_returns_none(
Q:\finance-analyzer\tests\test_llm_batch.py:195:    fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:202:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:203:    _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:220:def test_fingpt_phase_preserves_order_across_tickers(
Q:\finance-analyzer\tests\test_llm_batch.py:221:    fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:236:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:237:    _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:252:def test_fingpt_phase_swallows_top_level_failure(
Q:\finance-analyzer\tests\test_llm_batch.py:253:    fake_fingpt_infer, stash_recorder, monkeypatch,
Q:\finance-analyzer\tests\test_llm_batch.py:264:    from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:266:    _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:274:def test_enqueue_fingpt_dedup():
Q:\finance-analyzer\tests\test_llm_batch.py:276:    ticker from piling up multiple fingpt requests if get_sentiment is
Q:\finance-analyzer\tests\test_llm_batch.py:278:    from portfolio.llm_batch import _fingpt_queue, _lock, enqueue_fingpt
Q:\finance-analyzer\tests\test_llm_batch.py:281:        _fingpt_queue.clear()
Q:\finance-analyzer\tests\test_llm_batch.py:284:    enqueue_fingpt("BTC:t0", "headlines", ctx)
Q:\finance-analyzer\tests\test_llm_batch.py:285:    enqueue_fingpt("BTC:t0", "headlines", ctx)  # dup
Q:\finance-analyzer\tests\test_llm_batch.py:286:    enqueue_fingpt("BTC:t0", "cumul:0", ctx)    # different sub_key — allowed
Q:\finance-analyzer\tests\test_llm_batch.py:287:    enqueue_fingpt("ETH:t0", "headlines", ctx)  # different ab_key — allowed
Q:\finance-analyzer\tests\test_llm_batch.py:290:        assert len(_fingpt_queue) == 3
Q:\finance-analyzer\tests\test_llm_batch.py:292:        keys = {(ab, sub) for ab, sub, _ in _fingpt_queue}
Q:\finance-analyzer\tests\test_llm_batch.py:298:        _fingpt_queue.clear()
Q:\finance-analyzer\tests\test_llm_batch.py:301:def test_flush_ab_log_merges_fingpt_and_finbert(monkeypatch, tmp_path):
Q:\finance-analyzer\tests\test_llm_batch.py:302:    """End-to-end buffer test: stash an A/B context, stash finbert + fingpt
Q:\finance-analyzer\tests\test_llm_batch.py:307:    tmp_log = tmp_path / "sentiment_ab_log.jsonl"
Q:\finance-analyzer\tests\test_llm_batch.py:328:    # Stash a 3-headline fingpt result (as Phase 3 would after inference)
Q:\finance-analyzer\tests\test_llm_batch.py:329:    sentiment_mod._stash_fingpt_result(ab_key, "headlines", [
Q:\finance-analyzer\tests\test_llm_batch.py:332:         "model": "fingpt:finance-llama-8b"},
Q:\finance-analyzer\tests\test_llm_batch.py:335:         "model": "fingpt:finance-llama-8b"},
Q:\finance-analyzer\tests\test_llm_batch.py:338:         "model": "fingpt:finance-llama-8b"},
Q:\finance-analyzer\tests\test_llm_batch.py:353:    assert "fingpt:finance-llama-8b" in shadow_models
Q:\finance-analyzer\tests\test_llm_batch.py:365:    tmp_log = tmp_path / "sentiment_ab_log.jsonl"
Q:\finance-analyzer\tests\test_llm_batch.py:379:# Verifies that is_llm_on_cycle() rotates across ministral → qwen3 → fingpt
Q:\finance-analyzer\tests\test_llm_batch.py:396:    runs all; from counter==1 onward, rotation: ministral → qwen3 → fingpt → …
Q:\finance-analyzer\tests\test_llm_batch.py:404:        assert is_llm_on_cycle("fingpt") is True
Q:\finance-analyzer\tests\test_llm_batch.py:411:        assert is_llm_on_cycle("fingpt") is False
Q:\finance-analyzer\tests\test_llm_batch.py:418:        assert is_llm_on_cycle("fingpt") is False
Q:\finance-analyzer\tests\test_llm_batch.py:420:    def test_counter_3_runs_fingpt_only(self, reset_rotation_counter):
Q:\finance-analyzer\tests\test_llm_batch.py:425:        assert is_llm_on_cycle("fingpt") is True
Q:\finance-analyzer\tests\test_llm_batch.py:433:        assert is_llm_on_cycle("fingpt") is False
Q:\finance-analyzer\tests\test_llm_batch.py:438:        counts = {"ministral": 0, "qwen3": 0, "fingpt": 0}
Q:\finance-analyzer\tests\test_llm_batch.py:444:        assert counts == {"ministral": 2, "qwen3": 2, "fingpt": 2}
Q:\finance-analyzer\tests\test_llm_batch.py:460:        from portfolio.llm_batch import _fingpt_queue, _lock, _ministral_queue, _qwen3_queue, flush_llm_batch
Q:\finance-analyzer\tests\test_llm_batch.py:464:            _fingpt_queue.clear()
Q:\finance-analyzer\tests\test_llm_batch.py:470:        self, reset_rotation_counter, fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:474:            _fingpt_queue,
Q:\finance-analyzer\tests\test_llm_batch.py:478:            enqueue_fingpt,
Q:\finance-analyzer\tests\test_llm_batch.py:487:            _fingpt_queue.clear()
Q:\finance-analyzer\tests\test_llm_batch.py:490:        enqueue_fingpt(
Q:\finance-analyzer\tests\test_llm_batch.py:676:# fingpt batch observability (fix/fingpt-batch-observability, 2026-05-03)
Q:\finance-analyzer\tests\test_llm_batch.py:679:# Verifies the metrics dict returned by _flush_fingpt_phase on every code
Q:\finance-analyzer\tests\test_llm_batch.py:682:# Phase 1+2 results, so a fingpt-only cycle always logged "0 results"
Q:\finance-analyzer\tests\test_llm_batch.py:683:# whether fingpt succeeded or silently failed. New format includes the
Q:\finance-analyzer\tests\test_llm_batch.py:688:    """_flush_fingpt_phase returns a metrics dict on every code path so the
Q:\finance-analyzer\tests\test_llm_batch.py:693:    def test_metrics_on_success(self, fake_fingpt_infer, fake_llama_server, stash_recorder):
Q:\finance-analyzer\tests\test_llm_batch.py:697:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:698:        metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:714:    def test_metrics_on_empty_queue(self, fake_fingpt_infer, fake_llama_server, stash_recorder):
Q:\finance-analyzer\tests\test_llm_batch.py:716:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:717:        metrics = _flush_fingpt_phase([])
Q:\finance-analyzer\tests\test_llm_batch.py:727:        self, fake_fingpt_infer, fake_llama_server, stash_recorder, caplog,
Q:\finance-analyzer\tests\test_llm_batch.py:735:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:737:            metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:751:        self, fake_fingpt_infer, fake_llama_server, stash_recorder, caplog, monkeypatch,
Q:\finance-analyzer\tests\test_llm_batch.py:760:        # exception and returns None, which is what _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:764:        monkeypatch.setattr(fake_fingpt_infer, "_parse_sentiment", boom)
Q:\finance-analyzer\tests\test_llm_batch.py:768:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:770:            metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:786:        self, fake_fingpt_infer, fake_llama_server, stash_recorder, caplog,
Q:\finance-analyzer\tests\test_llm_batch.py:791:        _parse_fingpt_completion, an all-empty cycle would log F=N/N (looks
Q:\finance-analyzer\tests\test_llm_batch.py:797:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:799:            metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:814:        self, fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:834:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:835:        metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:846:    def test_metrics_on_fingpt_infer_import_failure(
Q:\finance-analyzer\tests\test_llm_batch.py:849:        """Codex P3: cover the fingpt_infer import failure path — when
Q:\finance-analyzer\tests\test_llm_batch.py:850:        Q:\\models is missing or fingpt_infer.py raises at import time, the
Q:\finance-analyzer\tests\test_llm_batch.py:852:        Other LLM phases are unaffected since fingpt is shadow-only.
Q:\finance-analyzer\tests\test_llm_batch.py:856:        # Wipe any previously-imported fingpt_infer so the next import re-runs.
Q:\finance-analyzer\tests\test_llm_batch.py:857:        sys.modules.pop("fingpt_infer", None)
Q:\finance-analyzer\tests\test_llm_batch.py:860:        def explode_on_fingpt_infer(name, *args, **kwargs):
Q:\finance-analyzer\tests\test_llm_batch.py:861:            if name == "fingpt_infer":
Q:\finance-analyzer\tests\test_llm_batch.py:864:        monkeypatch.setattr(builtins, "__import__", explode_on_fingpt_infer)
Q:\finance-analyzer\tests\test_llm_batch.py:866:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:868:            metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:881:        self, fake_fingpt_infer, stash_recorder, caplog, monkeypatch,
Q:\finance-analyzer\tests\test_llm_batch.py:893:        from portfolio.llm_batch import _flush_fingpt_phase
Q:\finance-analyzer\tests\test_llm_batch.py:895:            metrics = _flush_fingpt_phase([
Q:\finance-analyzer\tests\test_llm_batch.py:911:    so a fingpt-only cycle no longer falsely reports "0 results".
Q:\finance-analyzer\tests\test_llm_batch.py:915:        self, fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:918:        """Drive flush_llm_batch with a fingpt-only queue and assert the
Q:\finance-analyzer\tests\test_llm_batch.py:922:            _fingpt_queue,
Q:\finance-analyzer\tests\test_llm_batch.py:926:            enqueue_fingpt,
Q:\finance-analyzer\tests\test_llm_batch.py:932:            _fingpt_queue.clear()
Q:\finance-analyzer\tests\test_llm_batch.py:935:        enqueue_fingpt(
Q:\finance-analyzer\tests\test_llm_batch.py:939:        # counter=2 → next slot is fingpt (after warmup logic), but we pass
Q:\finance-analyzer\tests\test_llm_batch.py:958:    def test_summary_log_shows_fingpt_silent_failure(
Q:\finance-analyzer\tests\test_llm_batch.py:959:        self, fake_fingpt_infer, fake_llama_server, stash_recorder,
Q:\finance-analyzer\tests\test_llm_batch.py:962:        """When fingpt actually fails (server returns all None), the
Q:\finance-analyzer\tests\test_llm_batch.py:966:            _fingpt_queue,
Q:\finance-analyzer\tests\test_llm_batch.py:970:            enqueue_fingpt,
Q:\finance-analyzer\tests\test_llm_batch.py:976:            _fingpt_queue.clear()
Q:\finance-analyzer\tests\test_llm_batch.py:979:        enqueue_fingpt(
Q:\finance-analyzer\tests\test_local_llm_report.py:217:        "shadows": {"fingpt": {"status": "shadow", "entered_shadow_ts": "2026-04-09T00:00:00Z"}},
Q:\finance-analyzer\tests\test_local_llm_report.py:218:        "stale": [{"signal": "fingpt", "days_in_shadow": 22}],
Q:\finance-analyzer\tests\test_sentiment_shadow_backfill.py:47:            {"model": "fingpt:finance-llama-8b", "sentiment": "positive", "confidence": 0.6,
Q:\finance-analyzer\tests\test_sentiment_shadow_backfill.py:76:    assert by_model["fingpt:finance-llama-8b"]["predicted_class"] == "BUY"
Q:\finance-analyzer\tests\test_sentiment_shadow_backfill.py:77:    assert by_model["fingpt:finance-llama-8b"]["correct"] is True
Q:\finance-analyzer\tests\test_sentiment_shadow_backfill.py:78:    assert by_model["fingpt:finance-llama-8b"]["agreement_with_primary"] is True
Q:\finance-analyzer\tests\test_sentiment_shadow_backfill.py:124:            {"model": "fingpt:x", "sentiment": "positive", "confidence": 0.6},
Q:\finance-analyzer\tests\test_shadow_registry.py:22:        "fingpt",
Q:\finance-analyzer\tests\test_shadow_registry.py:28:    assert "fingpt" in reg["shadows"]
Q:\finance-analyzer\tests\test_shadow_registry.py:29:    entry = reg["shadows"]["fingpt"]
Q:\finance-analyzer\tests\test_shadow_registry.py:37:        "fingpt",
Q:\finance-analyzer\tests\test_shadow_registry.py:43:    mod.add_shadow("fingpt", {"min_samples": 300}, notes="updated", path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:44:    entry = mod.load_registry(path=reg_path)["shadows"]["fingpt"]
Q:\finance-analyzer\tests\test_shadow_registry.py:51:    mod.add_shadow("fingpt", {"min_samples": 200}, path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:52:    result = mod.resolve_shadow("fingpt", "promoted", notes="shipped", path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:54:    entry = mod.load_registry(path=reg_path)["shadows"]["fingpt"]
Q:\finance-analyzer\tests\test_shadow_registry.py:64:    mod.add_shadow("fingpt", {"min_samples": 200}, path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:66:        mod.resolve_shadow("fingpt", "maybe", path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:71:    mod.add_shadow("fingpt", {"min_samples": 200}, entered_ts=entered, path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:73:    days = mod.days_in_shadow("fingpt", path=reg_path, now=now)
Q:\finance-analyzer\tests\test_shadow_registry.py:109:    assert "fingpt" in first["shadows"]
Q:\finance-analyzer\tests\test_shadow_registry.py:112:    mod.resolve_shadow("fingpt", "promoted", path=reg_path)
Q:\finance-analyzer\tests\test_shadow_registry.py:115:    assert second["shadows"]["fingpt"]["status"] == "promoted"

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import re, ast
from pathlib import Path
text = Path(r'Q:\\finance-analyzer\\portfolio\\tickers.py').read_text(encoding='utf-8')
m = re.search(r'STOCK_SYMBOLS\\s*=\\s*(\\{.*?\\})', text, re.S)
if m:
    syms = ast.literal_eval(m.group(1))
    print(len(syms))
    print(sorted(syms))
else:
    print('not found')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 560ms:
1
['MSTR']

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"STOCK_SYMBOLS|TICKER_SOURCE_MAP|MSTR|NVDA|AAPL\" Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 551ms:
20:    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
21:    "MSTR": {"alpaca": "MSTR"},
22:    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
23:    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
25:    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
33:STOCK_SYMBOLS = {"MSTR"}
36:ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS
51:TICKER_SOURCE_MAP = {
56:YF_MAP = {t: t for t in STOCK_SYMBOLS}
142:                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
145:                        # ETH 34.9% (146), MSTR 33.3% (264), XAU N/A. Not salvageable.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n '\"_df\"|\\['_\"\"\"??' Q:\\finance-analyzer\\portfolio Q:\\finance-analyzer\\tests" in Q:\finance-analyzer\.worktrees\adv-data-external
2026-05-10T15:29:56.389417Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:

 exited 1 in 539ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_df\" Q:\\finance-analyzer\\portfolio Q:\\finance-analyzer\\tests" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 522ms:
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:178:    def _make_df(self, n=150, close_start=100.0):
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:203:        df = self._make_df(150)
Q:\finance-analyzer\tests\test_adversarial_review_fixes.py:233:        df = self._make_df(150)
Q:\finance-analyzer\tests\test_3h_integration.py:9:def _make_df(n=100, close_start=100.0):
Q:\finance-analyzer\tests\test_3h_integration.py:31:        df = _make_df(100)
Q:\finance-analyzer\tests\test_3h_integration.py:47:        df = _make_df(100)
Q:\finance-analyzer\tests\test_3h_integration.py:58:        df = _make_df(100)
Q:\finance-analyzer\tests\test_batch3_fixes.py:24:    def _make_df(self, n=10, up=True):
Q:\finance-analyzer\tests\test_batch3_fixes.py:46:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:62:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:78:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:97:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:113:            df=self._make_df(),
Q:\finance-analyzer\tests\test_batch3_fixes.py:202:    def test_empty_response_returns_empty_df(self):
Q:\finance-analyzer\tests\conftest.py:148:def make_ohlcv_df(n=250, close_base=100.0, trend=0.0, volatility=1.0, seed=42):
Q:\finance-analyzer\portfolio\data_collector.py:306:            entry["_df"] = df  # preserve raw DataFrame for enhanced signals
Q:\finance-analyzer\portfolio\forecast_signal.py:197:    context_df = pd.DataFrame({
Q:\finance-analyzer\portfolio\forecast_signal.py:204:    pred_df = pipeline.predict_df(
Q:\finance-analyzer\portfolio\forecast_signal.py:205:        context_df,
Q:\finance-analyzer\portfolio\forecast_signal.py:217:        # pred_df has columns: id, timestamp, 0.1, 0.5, 0.9
Q:\finance-analyzer\portfolio\forecast_signal.py:218:        row = pred_df.iloc[h - 1]
Q:\finance-analyzer\tests\test_confidence_penalties.py:16:def _make_df(n=50, close_start=100.0, volume_start=1000.0, trend="flat"):
Q:\finance-analyzer\tests\test_confidence_penalties.py:43:def _make_df_volume_pattern(n=50, close_start=100.0, vol_pattern="flat", trend="up"):
Q:\finance-analyzer\tests\test_confidence_penalties.py:45:    df = _make_df(n, close_start, trend=trend)
Q:\finance-analyzer\tests\test_confidence_penalties.py:85:        df = _make_df(27)
Q:\finance-analyzer\tests\test_confidence_penalties.py:89:        df = _make_df(27, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:93:        df = _make_df(28, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:100:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:107:        df = _make_df(100, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:113:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:131:        df_up = _make_df(100, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:137:        df = _make_df(60, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:344:        df = _make_df(50, trend="flat")
Q:\finance-analyzer\tests\test_confidence_penalties.py:359:        df = _make_df(100, trend="up")  # strong trend -> higher ADX
Q:\finance-analyzer\tests\test_confidence_penalties.py:405:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:410:    def test_adx_none_when_no_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:422:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:434:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="down")
Q:\finance-analyzer\tests\test_confidence_penalties.py:443:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:450:        df = _make_df_volume_pattern(50, vol_pattern="expanding", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:457:    def test_no_trap_with_short_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:459:        df = _make_df(3)
Q:\finance-analyzer\tests\test_confidence_penalties.py:466:    def test_no_trap_with_none_df(self):
Q:\finance-analyzer\tests\test_confidence_penalties.py:475:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="down")
Q:\finance-analyzer\tests\test_confidence_penalties.py:484:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:493:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:648:        df = _make_df_volume_pattern(50, vol_pattern="declining")
Q:\finance-analyzer\tests\test_confidence_penalties.py:682:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:708:        df = _make_df(50, trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:723:        df = _make_df_volume_pattern(50, vol_pattern="declining", trend="up")
Q:\finance-analyzer\tests\test_confidence_penalties.py:734:        df = _make_df_volume_pattern(50, vol_pattern="expanding", trend="up")
Q:\finance-analyzer\portfolio\linear_factor.py:12:    model.fit(signal_history_df, returns_series)
Q:\finance-analyzer\portfolio\linear_factor.py:53:    def fit(self, signals_df: pd.DataFrame, returns: pd.Series,
Q:\finance-analyzer\portfolio\linear_factor.py:58:            signals_df: DataFrame where columns are signal names and rows are
Q:\finance-analyzer\portfolio\linear_factor.py:61:            returns: Series of forward returns aligned with signals_df index.
Q:\finance-analyzer\portfolio\linear_factor.py:68:        common = signals_df.index.intersection(returns.index)
Q:\finance-analyzer\portfolio\linear_factor.py:74:        X = signals_df.loc[common].copy()
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:10:def _df(closes):
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:17:        out = compute_crypto_cross_asset_signal(_df([100, 101]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:25:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:33:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:42:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:50:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:61:        eth = _df([3500.0, 3675.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:62:        btc = _df([100000.0, 100000.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:63:        out = compute_crypto_cross_asset_signal(_df([3500, 3675]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:70:        eth = _df([3500.0, 3325.0])  # -5%
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:71:        btc = _df([100000.0, 100000.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:72:        out = compute_crypto_cross_asset_signal(_df([3500, 3325]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:81:        dxy = _df([100.0, 101.0])  # +1% — crypto headwind
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:82:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:89:        dxy = _df([100.0, 99.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:90:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:99:        spy = _df([580.0, 585.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:100:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:110:        gold = _df([4500.0, 4600.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:111:        btc = _df([100000.0, 100000.0])
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:112:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:122:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:126:                "dxy_history": _df([100.0, 99.0]),  # DXY down -> BUY
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:127:                "spy_history": _df([580.0, 585.0]),  # SPY up -> BUY
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:128:                "eth_history": _df([3500.0, 3500.0]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:129:                "btc_history": _df([100000.0, 100000.0]),
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:130:                "gold_history": _df([4500.0, 4400.0]),  # gold down -> BUY (gold/btc ratio falls)
Q:\finance-analyzer\tests\test_crypto_cross_asset_signal.py:137:        out = compute_crypto_cross_asset_signal(_df([105000, 105100]), {
Q:\finance-analyzer\tests\test_crypto_macro.py:124:    def _make_df(self, close=70000):
Q:\finance-analyzer\tests\test_crypto_macro.py:139:            self._make_df(135),
Q:\finance-analyzer\tests\test_crypto_macro.py:151:                self._make_df(70000),
Q:\finance-analyzer\tests\test_data_collector.py:367:    def _make_yf_df(self, n=10, multi_index=False):
Q:\finance-analyzer\tests\test_data_collector.py:390:        mock_yf.download.return_value = self._make_yf_df(10)
Q:\finance-analyzer\tests\test_data_collector.py:424:        mock_yf.download.return_value = self._make_yf_df(5, multi_index=True)
Q:\finance-analyzer\tests\test_data_collector.py:445:        mock_yf.download.return_value = self._make_yf_df(20)
Q:\finance-analyzer\tests\test_data_collector.py:615:        assert "_df" in now_entry  # raw DataFrame preserved for enhanced signals
Q:\finance-analyzer\tests\test_data_collector.py:819:                                "confidence": None, "_df": pd.DataFrame()})
Q:\finance-analyzer\portfolio\main.py:487:            now_df = None
Q:\finance-analyzer\portfolio\main.py:490:                now_df = now_entry.get("_df")
Q:\finance-analyzer\portfolio\main.py:492:                now_df = _fetch_klines(source, interval="15m", limit=100)
Q:\finance-analyzer\portfolio\main.py:493:                ind = compute_indicators(now_df)
Q:\finance-analyzer\portfolio\main.py:503:                ind, ticker=name, config=config, timeframes=tfs, df=now_df
Q:\finance-analyzer\portfolio\main.py:555:                "tfs": tfs, "ind": ind, "now_df": now_df, "price": price,
Q:\finance-analyzer\portfolio\macro_context.py:221:    klines_df = _fetch_klines(ticker)
Q:\finance-analyzer\portfolio\macro_context.py:222:    if klines_df is None or klines_df.empty:
Q:\finance-analyzer\portfolio\macro_context.py:224:    vol = klines_df["volume"].astype(float)
Q:\finance-analyzer\portfolio\macro_context.py:225:    close = klines_df["close"].astype(float)
Q:\finance-analyzer\tests\test_enhanced_signals.py:47:def _make_empty_df():
Q:\finance-analyzer\tests\test_enhanced_signals.py:51:def _make_nan_df(n=100):
Q:\finance-analyzer\tests\test_enhanced_signals.py:95:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:97:        result = compute_trend_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:141:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:143:        result = compute_volatility_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:147:    def test_empty_df_schema_matches_success_path(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:154:        empty = compute_volatility_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:259:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:261:        result = compute_structure_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:404:    def test_empty_df(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:406:        result = compute_heikin_ashi_signal(_make_empty_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:482:        result = compute_trend_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:487:        result = compute_volatility_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:492:        result = compute_candlestick_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:497:        result = compute_structure_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:502:        result = compute_fibonacci_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:507:        result = compute_smart_money_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:512:        result = compute_oscillator_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:517:        result = compute_heikin_ashi_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:522:        result = compute_volume_flow_signal(_make_nan_df())
Q:\finance-analyzer\tests\test_enhanced_signals.py:533:    def _df_missing_volume(self):
Q:\finance-analyzer\tests\test_enhanced_signals.py:539:        result = compute_trend_signal(self._df_missing_volume())
Q:\finance-analyzer\tests\test_enhanced_signals.py:544:        result = compute_structure_signal(self._df_missing_volume())
Q:\finance-analyzer\tests\test_enhanced_signals.py:549:        result = compute_volume_flow_signal(self._df_missing_volume())
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:9:def _make_df(n=30):
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:27:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:38:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:48:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:59:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:66:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="NVDA")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:73:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="BTC-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:81:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:93:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:105:        r1 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:111:        r2 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:125:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:135:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:145:        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:179:        empty_df = pd.DataFrame()
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:180:        eurusd_df = pd.DataFrame(
Q:\finance-analyzer\tests\test_dxy_cross_asset_signal.py:183:        mock_dl.side_effect = [empty_df, eurusd_df]
Q:\finance-analyzer\portfolio\metals_cross_assets.py:139:        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:140:        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:141:        if gold_df.empty or silver_df.empty:
Q:\finance-analyzer\portfolio\metals_cross_assets.py:143:        gold_close = gold_df["Close"].dropna()
Q:\finance-analyzer\portfolio\metals_cross_assets.py:144:        silver_close = silver_df["Close"].dropna()
Q:\finance-analyzer\portfolio\metals_cross_assets.py:243:        gold_df = _yf_download("GC=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:244:        silver_df = _yf_download("SI=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:245:        if gold_df.empty or silver_df.empty:
Q:\finance-analyzer\portfolio\metals_cross_assets.py:247:        gold_close = gold_df["Close"].dropna()
Q:\finance-analyzer\portfolio\metals_cross_assets.py:248:        silver_close = silver_df["Close"].dropna()
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:18:def _make_df(n=60, with_ohlcv=True):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:29:def _make_candles_df(n=60):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:31:    df = _make_df(n)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:67:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:70:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:87:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:90:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:102:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:105:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:117:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:122:                 patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:143:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:146:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:160:        binance_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:163:             patch("portfolio.data_collector.binance_klines", return_value=binance_df) as mock_bk:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:175:        alpaca_df = _make_candles_df(80)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:178:             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:199:    def test_kronos_df_fallback_triggers_when_candle_load_fails(self):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:203:        df = _make_df(60)
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:241:    def test_kronos_df_fallback_sets_source_indicator(self):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:245:        df = _make_df(60)  # has full OHLCV
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:272:    def test_kronos_df_fallback_skipped_when_df_close_only(self):
Q:\finance-analyzer\tests\test_forecast_candle_loading.py:276:        df = _make_df(60, with_ohlcv=False)  # close only
Q:\finance-analyzer\tests\test_futures_flow.py:14:def sample_df():
Q:\finance-analyzer\tests\test_futures_flow.py:28:def falling_df():
Q:\finance-analyzer\tests\test_futures_flow.py:255:    def test_rising_oi_rising_price_buy(self, oi_history_rising, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:257:        assert _oi_trend(oi_history_rising, sample_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:259:    def test_rising_oi_falling_price_sell(self, oi_history_rising, falling_df):
Q:\finance-analyzer\tests\test_futures_flow.py:261:        assert _oi_trend(oi_history_rising, falling_df) == "SELL"
Q:\finance-analyzer\tests\test_futures_flow.py:263:    def test_falling_oi_hold(self, oi_history_falling, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:265:        assert _oi_trend(oi_history_falling, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:267:    def test_no_data_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:269:        assert _oi_trend(None, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:270:        assert _oi_trend([], sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:307:    def test_price_up_oi_down_sell(self, oi_history_falling, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:309:        assert _oi_divergence(oi_history_falling, sample_df) == "SELL"
Q:\finance-analyzer\tests\test_futures_flow.py:311:    def test_price_down_oi_down_buy(self, oi_history_falling, falling_df):
Q:\finance-analyzer\tests\test_futures_flow.py:313:        assert _oi_divergence(oi_history_falling, falling_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:315:    def test_no_data_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:317:        assert _oi_divergence(None, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:376:    def test_accelerating_oi_price_up_buy(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:382:        assert _oi_acceleration(oi, sample_df) == "BUY"
Q:\finance-analyzer\tests\test_futures_flow.py:384:    def test_no_data_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:386:        assert _oi_acceleration(None, sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:387:        assert _oi_acceleration([], sample_df) == "HOLD"
Q:\finance-analyzer\tests\test_futures_flow.py:395:    def test_non_crypto_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:397:        result = compute_futures_flow_signal(sample_df, context={"ticker": "NVDA"})
Q:\finance-analyzer\tests\test_futures_flow.py:401:    def test_no_context_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:403:        result = compute_futures_flow_signal(sample_df, context=None)
Q:\finance-analyzer\tests\test_futures_flow.py:408:    def test_fetch_failure_hold(self, mock_cached, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:411:        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
Q:\finance-analyzer\tests\test_futures_flow.py:417:    def test_confidence_cap(self, mock_cached, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:435:        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
Q:\finance-analyzer\tests\test_futures_flow.py:439:    def test_result_structure(self, mock_cached, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:456:        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
Q:\finance-analyzer\tests\test_futures_flow.py:472:    def test_metals_hold(self, sample_df):
Q:\finance-analyzer\tests\test_futures_flow.py:475:        result = compute_futures_flow_signal(sample_df, context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py:241:    def _make_df(self, n=100, close_start=100.0):
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py:270:        df = self._make_df(150)
Q:\finance-analyzer\tests\test_horizon_specific_blacklist.py:336:        df = self._make_df(150)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:232:    gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:233:    if gap_df is None or len(gap_df) < 20:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:237:    current_gap = gap_df["gap"].iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:238:    current_avg_corr = gap_df["avg_corr"].iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:239:    current_max_eig = gap_df["max_eig_norm"].iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:242:    gap_mean = gap_df["gap"].rolling(60, min_periods=20).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:243:    gap_std = gap_df["gap"].rolling(60, min_periods=20).std().iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:250:    slope_vote = _gap_slope_vote(gap_df["gap"], is_safe_haven)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:71:    cached = _CACHE.get("ratio_df")
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:78:        copper_df = _download("HG=F", period="1y", interval="1d")
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:79:        gold_df = _download("GC=F", period="1y", interval="1d")
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:83:            copper_df = yf.download("HG=F", period="1y", interval="1d",
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:85:            gold_df = yf.download("GC=F", period="1y", interval="1d",
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:91:    if copper_df is None or gold_df is None:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:93:    if copper_df.empty or gold_df.empty:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:97:    copper_close = _extract_close(copper_df)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:98:    gold_close = _extract_close(gold_df)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:113:    _CACHE["ratio_df"] = (now, combined)
Q:\finance-analyzer\portfolio\train_signal_weights.py:51:        (signals_df, returns_series) or None if insufficient data.
Q:\finance-analyzer\portfolio\train_signal_weights.py:96:    signals_df = df[signal_cols].fillna(0.0)
Q:\finance-analyzer\portfolio\train_signal_weights.py:98:    return signals_df, returns
Q:\finance-analyzer\portfolio\train_signal_weights.py:121:    signals_df, returns = data
Q:\finance-analyzer\portfolio\train_signal_weights.py:123:               len(signals_df), len(signals_df.columns), horizon)
Q:\finance-analyzer\portfolio\train_signal_weights.py:127:    if not model.fit(signals_df, returns):
Q:\finance-analyzer\portfolio\train_signal_weights.py:136:        signals_df, returns,
Q:\finance-analyzer\portfolio\train_signal_weights.py:137:        train_window=min(720, len(signals_df) // 3),
Q:\finance-analyzer\portfolio\train_signal_weights.py:138:        test_window=min(168, len(signals_df) // 6),
Q:\finance-analyzer\portfolio\train_signal_weights.py:139:        step_size=min(168, len(signals_df) // 6),
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:277:    turb_df = _compute_turbulence_series(closes)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:278:    if turb_df is None or len(turb_df) < _Z_WINDOW:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:282:    turb_series = turb_df["turbulence"]
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:283:    ar_series = turb_df["absorption_ratio"]
Q:\finance-analyzer\tests\test_lora_pipeline.py:20:def _make_df(n=200, base_price=69000.0, trend=0.0):
Q:\finance-analyzer\tests\test_lora_pipeline.py:72:        df = _make_df(200)
Q:\finance-analyzer\tests\test_lora_pipeline.py:139:        df = _make_df(300)
Q:\finance-analyzer\tests\test_lora_pipeline.py:152:        df = _make_df(50)
Q:\finance-analyzer\tests\test_lora_pipeline.py:159:        df = _make_df(300)
Q:\finance-analyzer\tests\test_lora_pipeline.py:316:        df = _make_df(100)
Q:\finance-analyzer\tests\test_lora_pipeline.py:330:        df = _make_df(100)
Q:\finance-analyzer\tests\test_lora_pipeline.py:337:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_core.py:23:def _make_df(n, base=100.0, trend=0.0, noise=0.5, volume=1000.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:61:def _make_uptrend_df(n=60, base=100.0, step=2.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:63:    return _make_df(n, base=base, trend=step, noise=0.3, seed=seed)
Q:\finance-analyzer\tests\test_indicators_core.py:66:def _make_downtrend_df(n=60, base=200.0, step=-2.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:68:    return _make_df(n, base=base, trend=step, noise=0.3, seed=seed)
Q:\finance-analyzer\tests\test_indicators_core.py:71:def _make_flat_df(n=60, base=100.0, seed=42):
Q:\finance-analyzer\tests\test_indicators_core.py:73:    return _make_df(n, base=base, trend=0.0, noise=0.1, seed=seed)
Q:\finance-analyzer\tests\test_indicators_core.py:89:    def test_returns_none_for_empty_df(self):
Q:\finance-analyzer\tests\test_indicators_core.py:90:        df = _make_df(0)
Q:\finance-analyzer\tests\test_indicators_core.py:94:        df = _make_df(1)
Q:\finance-analyzer\tests\test_indicators_core.py:98:        df = _make_df(25)
Q:\finance-analyzer\tests\test_indicators_core.py:102:        df = _make_df(26)
Q:\finance-analyzer\tests\test_indicators_core.py:108:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_core.py:118:        self.ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:143:        ind = compute_indicators(_make_flat_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:147:        ind = compute_indicators(_make_uptrend_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:151:        ind = compute_indicators(_make_downtrend_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:156:        ind = compute_indicators(_make_uptrend_df(80, step=3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:161:        ind = compute_indicators(_make_downtrend_df(80, base=500.0, step=-3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:166:        ind = compute_indicators(_make_df(26))
Q:\finance-analyzer\tests\test_indicators_core.py:181:        ind = compute_indicators(_make_uptrend_df(80, step=3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:188:        ind = compute_indicators(_make_downtrend_df(80, base=500.0, step=-3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:192:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:197:        ind = compute_indicators(_make_df(26))
Q:\finance-analyzer\tests\test_indicators_core.py:205:        ind = compute_indicators(_make_uptrend_df(80, step=3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:209:        ind = compute_indicators(_make_downtrend_df(80, base=500.0, step=-3.0))
Q:\finance-analyzer\tests\test_indicators_core.py:217:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:221:        ind = compute_indicators(_make_uptrend_df(80))
Q:\finance-analyzer\tests\test_indicators_core.py:226:        ind = compute_indicators(_make_flat_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:232:        df = _make_flat_df(58, base=100.0, seed=10)
Q:\finance-analyzer\tests\test_indicators_core.py:248:        df = _make_flat_df(58, base=100.0, seed=10)
Q:\finance-analyzer\tests\test_indicators_core.py:266:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:270:        ind = compute_indicators(_make_df(60))
Q:\finance-analyzer\tests\test_indicators_core.py:274:        calm = compute_indicators(_make_df(60, noise=0.1))
Q:\finance-analyzer\tests\test_indicators_core.py:275:        volatile = compute_indicators(_make_df(60, noise=5.0))
Q:\finance-analyzer\tests\test_indicators_core.py:288:        df = _make_df(60, base=50.0)
Q:\finance-analyzer\tests\test_indicators_core.py:293:        df = _make_df(60, base=50.0)
Q:\finance-analyzer\tests\test_indicators_core.py:298:        df = _make_df(60, base=50.0)
Q:\finance-analyzer\tests\test_linear_factor.py:24:    signals_df = pd.DataFrame({
Q:\finance-analyzer\tests\test_linear_factor.py:31:    return signals_df, returns_s
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:99:def run_kronos_prediction(predictor, candles_df: pd.DataFrame,
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:106:        candles_df: DataFrame with open,high,low,close,volume,amount,timestamp
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:115:    ohlcv = candles_df[["open", "high", "low", "close", "volume", "amount"]].copy()
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:117:    x_timestamp = pd.DatetimeIndex(candles_df["timestamp"].values)
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:130:    pred_df = predictor.predict(
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:140:    return pred_df
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:282:                pred_df = run_kronos_prediction(
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:289:                predicted_close = float(pred_df.iloc[-1]["close"])
Q:\finance-analyzer\tests\test_indicators_short.py:9:def _make_df(n=100, close_start=100.0):
Q:\finance-analyzer\tests\test_indicators_short.py:27:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:33:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:40:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:51:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:58:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:64:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:70:        df = _make_df(100)
Q:\finance-analyzer\tests\test_indicators_short.py:77:        df = _make_df(20)
Q:\finance-analyzer\tests\test_indicators_short.py:83:        df = _make_df(20)
Q:\finance-analyzer\tests\test_macro_regime_integration.py:20:def _make_df(n=250, close_base=100.0, trend=0.0):
Q:\finance-analyzer\tests\test_macro_regime_integration.py:148:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_macro_regime_integration.py:162:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_macro_regime_integration.py:369:        df = _make_df(n=250, trend=0.5)  # uptrend
Q:\finance-analyzer\tests\test_macro_regime_integration.py:390:        df = _make_df(n=5)  # short, SMA signals = HOLD
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:53:    signals_df: pd.DataFrame,
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:64:        signals_df: DataFrame of signal values (columns=signals, rows=time).
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:65:        returns: Series of forward returns aligned with signals_df.
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:75:    common = signals_df.index.intersection(returns.index)
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:76:    signals_df = signals_df.loc[common]
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:94:        train_X = signals_df.iloc[start:train_end]
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py:96:        test_X = signals_df.iloc[train_end:test_end]
Q:\finance-analyzer\portfolio\signals\structure.py:61:    lookback_df = df.iloc[-252:]
Q:\finance-analyzer\portfolio\signals\structure.py:62:    high = lookback_df["high"].astype(float)
Q:\finance-analyzer\portfolio\signals\structure.py:63:    low = lookback_df["low"].astype(float)
Q:\finance-analyzer\portfolio\signals\structure.py:64:    close = lookback_df["close"].astype(float)
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:21:def _make_df(n=50):
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:72:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:85:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:93:            _make_df(), ticker="NVDA", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:103:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:112:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:121:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:130:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:139:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:148:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:157:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:166:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:175:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:191:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:201:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:211:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:220:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:230:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:240:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:249:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:269:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:382:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:398:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:415:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:431:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:444:            _make_df(), ticker="XAU-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:459:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:483:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:501:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:518:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:534:            _make_df(),
Q:\finance-analyzer\tests\test_metals_cross_asset_signal.py:551:            _make_df(),
Q:\finance-analyzer\tests\test_momentum_fix.py:20:def _make_df(n: int = 60, trend: str = "up") -> pd.DataFrame:
Q:\finance-analyzer\tests\test_momentum_fix.py:127:        df = _make_df(60)
Q:\finance-analyzer\tests\test_momentum_fix.py:138:        df = _make_df(60)
Q:\finance-analyzer\tests\test_momentum_fix.py:145:        df = _make_df(60)
Q:\finance-analyzer\tests\test_momentum_fix.py:155:        df = _make_df(20)
Q:\finance-analyzer\tests\test_monte_carlo_risk.py:561:    def test_very_low_df_extreme_tails(self):
Q:\finance-analyzer\tests\test_oil_loop.py:83:    fake_df = pd.DataFrame({
Q:\finance-analyzer\tests\test_oil_loop.py:92:               return_value=fake_df) as mock_fetch:
Q:\finance-analyzer\tests\test_oil_loop.py:111:def test_fetch_live_prices_returns_empty_when_router_returns_empty_df():
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:9:def _make_df(n=50):
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:32:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:48:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:58:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:67:            _make_df(), ticker="NVDA", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:83:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:106:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:123:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:140:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_orderbook_flow_signal.py:157:            _make_df(), ticker="XAG-USD", config={}, macro={}
Q:\finance-analyzer\tests\test_price_source.py:17:def _empty_df() -> pd.DataFrame:
Q:\finance-analyzer\tests\test_price_source.py:75:        mock_fapi.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:81:        mock_fapi.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:87:        mock_spot.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:93:        mock_alpaca.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:99:        mock_yf.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:111:        mock_yf.return_value = _empty_df()
Q:\finance-analyzer\tests\test_price_source.py:120:        mock_yf.return_value = _empty_df()
Q:\finance-analyzer\tests\test_safety_guards.py:32:    def _make_df(n=60, base=100.0):
Q:\finance-analyzer\tests\test_safety_guards.py:46:        df = self._make_df()
Q:\finance-analyzer\tests\test_safety_guards.py:52:        df = self._make_df()
Q:\finance-analyzer\tests\test_safety_guards.py:58:        df = self._make_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:36:def _make_df(n: int = 250, close_base: float = 100.0,
Q:\finance-analyzer\tests\test_signals_macro_regime.py:55:def _make_df_above_sma(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:72:def _make_df_below_sma(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:89:def _make_golden_cross_df(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:109:def _make_death_cross_df(n: int = 250) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_macro_regime.py:135:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:145:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:154:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:160:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:166:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:175:    def test_very_short_df(self):
Q:\finance-analyzer\tests\test_signals_macro_regime.py:176:        df = _make_df(n=5)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:181:    def test_none_df_returns_hold(self):
Q:\finance-analyzer\tests\test_signals_macro_regime.py:186:    def test_empty_df_returns_hold(self):
Q:\finance-analyzer\tests\test_signals_macro_regime.py:199:        df = _make_df(n=1)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:208:        df = _make_df_above_sma()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:214:        df = _make_df_below_sma()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:220:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:226:        df = _make_df_above_sma(n=100)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:232:        df = _make_df_above_sma(n=50)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:238:        df = _make_df(n=30)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:243:        df = _make_df_above_sma()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:392:        df = _make_golden_cross_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:401:        df = _make_death_cross_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:409:        df = _make_df_above_sma(n=100)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:416:        df = _make_df(n=30)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:423:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:432:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:440:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:450:        df = _make_df(n=5)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:456:        df = _make_golden_cross_df()
Q:\finance-analyzer\tests\test_signals_macro_regime.py:472:        df = _make_df(n=5)
Q:\finance-analyzer\tests\test_signals_macro_regime.py:503:        df = _make_death_cross_df()
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:55:def _make_df(
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:79:def _make_trending_df(
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:113:def _make_flat_df(n: int = 80, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:128:def _make_ha_candles_df(n: int = 20) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:130:    return _make_df(n=n)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:141:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:149:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:154:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:178:        df = _make_df(1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:185:        df = _make_df(9)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:192:        df = _make_df(10)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:260:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:265:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:270:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:275:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:300:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:310:        df = _make_trending_df(n=100, step=1.5)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:315:        df = _make_trending_df(n=100, step=-1.5)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:320:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:335:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:340:        df = _make_trending_df(n=100, step=2.0)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:345:        df = _make_trending_df(n=100, step=-2.0)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:350:        df = _make_df(10)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:367:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:373:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:380:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:386:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:392:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:399:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:404:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:409:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:415:        df = _make_df(80, close_base=100.0, volatility=0.5)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:807:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:813:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:819:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:847:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:858:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:864:        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:878:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:884:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:890:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:917:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:928:        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:941:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:947:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:953:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:959:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:965:        df = _make_flat_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:980:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:987:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:997:        df = _make_df(1)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1004:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1012:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1021:        df = _make_df(10)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1032:        df = _make_df(30)
Q:\finance-analyzer\tests\test_signals_heikin_ashi.py:1044:        df = _make_df(30)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:54:def _make_uptrend_df(n: int = 100, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_fibonacci.py:60:def _make_downtrend_df(n: int = 100, start: float = 200.0, step: float = 0.5) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_fibonacci.py:66:def _make_flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_fibonacci.py:365:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:373:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:385:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:390:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:395:        df = _make_uptrend_df(100)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:422:        df = _make_flat_df(100, price=50.0)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:428:        df = _make_downtrend_df(100, start=200.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:434:        df = _make_uptrend_df(100, start=100.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_fibonacci.py:490:        df = _make_uptrend_df(MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:20:def _make_df(n=100, last_time=None):
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:37:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:43:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:56:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_econ_calendar.py:310:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:26:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:489:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:495:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:521:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:544:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:558:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:569:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:588:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:618:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_candlestick.py:38:def _make_df(rows, columns=("open", "high", "low", "close")):
Q:\finance-analyzer\tests\test_signals_candlestick.py:82:        df = _make_df([(100, 105, 95, 102), (102, 108, 99, 106)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:103:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:117:        df = _make_df([(i, i + 1, i - 1, i + 0.5) for i in range(10, 25)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:122:        df = _make_df([(i, i + 1, i - 1, i - 0.5) for i in range(25, 10, -1)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:127:        df = _make_df([(100, 101, 99, 100) for _ in range(10)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:131:        df = _make_df([(100, 105, 95, 102)])
Q:\finance-analyzer\tests\test_signals_candlestick.py:363:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:372:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:378:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:384:        df = _make_df([(100, 110, 90, 108)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:408:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:421:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:435:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:444:        df = _make_df([(100, 105, 95, 102)] * 10)
Q:\finance-analyzer\tests\test_signals_candlestick.py:465:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:477:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:487:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_candlestick.py:493:        df = _make_df([(100, 105, 95, 102)] * 5)
Q:\finance-analyzer\tests\test_signals_candlestick.py:506:        df = _make_df([
Q:\finance-analyzer\tests\test_signals_candlestick.py:528:        df = _make_df([(100, 100, 100, 100)] * 10)
Q:\finance-analyzer\tests\test_signals_candlestick.py:536:        df = _make_df(rows)
Q:\finance-analyzer\tests\test_signals_calendar.py:40:def _make_df(n: int = 10, start_date: str = "2026-02-16",
Q:\finance-analyzer\tests\test_signals_calendar.py:80:def _make_df_on_date(target_date: str, n: int = 10,
Q:\finance-analyzer\tests\test_signals_calendar.py:114:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:123:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:128:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:134:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:159:        df = _make_df(n=1)
Q:\finance-analyzer\tests\test_signals_calendar.py:165:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:198:        df = _make_df_on_date("2026-02-16", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:205:        df = _make_df_on_date("2026-02-20", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:215:        df = _make_df_on_date("2026-02-17", n=10, red_prior=True)
Q:\finance-analyzer\tests\test_signals_calendar.py:223:        df = _make_df_on_date("2026-02-17", n=10, red_prior=False)
Q:\finance-analyzer\tests\test_signals_calendar.py:230:        df = _make_df_on_date("2026-02-18", n=10)  # Wednesday
Q:\finance-analyzer\tests\test_signals_calendar.py:258:        df = _make_df_on_date("2026-01-30", n=10)  # Jan 30 = 1 day remaining
Q:\finance-analyzer\tests\test_signals_calendar.py:299:        df = _make_df_on_date("2026-06-15", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:305:        df = _make_df_on_date("2026-11-15", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:386:        df = _make_df_on_date("2026-03-16", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:430:        df = _make_df_on_date("2026-01-02", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:447:        df = _make_df_on_date("2026-03-11", n=10)  # Wednesday, mid-month
Q:\finance-analyzer\tests\test_signals_calendar.py:459:                df = _make_df_on_date(target.isoformat(), n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:472:        df = _make_df_on_date("2026-01-02", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:484:        df = _make_df_on_date("2026-07-06", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:495:        df = _make_df_on_date("2026-10-06", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:504:        df = _make_df_on_date("2026-12-14", n=10)
Q:\finance-analyzer\tests\test_signals_calendar.py:510:        df = _make_df(n=10, start_date="2026-01-02")
Q:\finance-analyzer\tests\test_signal_credit_spread.py:14:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_credit_spread.py:43:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:68:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:75:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:85:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:110:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:225:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:236:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:247:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:258:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:269:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:292:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:311:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:324:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_credit_spread.py:336:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:15:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:244:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:268:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:292:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:297:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:303:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:331:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:350:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAG-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:359:        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
Q:\finance-analyzer\tests\test_signal_cot_positioning.py:377:        result = compute_cot_positioning_signal(_make_df(), ticker="BTC-USD")
Q:\finance-analyzer\tests\test_signals_smart_money.py:40:def _make_df(
Q:\finance-analyzer\tests\test_signals_smart_money.py:63:def _make_uptrend_with_bos_df(n: int = 80) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:102:def _make_downtrend_then_reversal_df(n: int = 80) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:122:def _make_fvg_df(bullish: bool = True, n: int = 60) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:193:def _make_flat_df(n: int = 80, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_smart_money.py:216:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:221:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:229:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:234:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:240:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:246:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:254:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:268:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:275:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:309:        df = _make_df(MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_smart_money.py:316:        df = _make_df(MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_smart_money.py:354:        df = _make_uptrend_with_bos_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:360:        df = _make_uptrend_with_bos_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:406:        df = _make_downtrend_then_reversal_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:462:        df = _make_fvg_df(bullish=True, n=60)
Q:\finance-analyzer\tests\test_signals_smart_money.py:512:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:644:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:702:        df = _make_df(80, seed=seed)
Q:\finance-analyzer\tests\test_signals_smart_money.py:711:        df = _make_df(80, seed=seed)
Q:\finance-analyzer\tests\test_signals_smart_money.py:720:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:727:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:740:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:745:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:751:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:757:        df = _make_flat_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:806:        df = _make_uptrend_with_bos_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:816:        df = _make_downtrend_then_reversal_df(n=80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:824:        df = _make_df(80)
Q:\finance-analyzer\tests\test_signals_smart_money.py:843:        df = _make_df(80, seed=42)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:14:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:53:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:67:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:78:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:90:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:111:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:139:        gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:140:        assert gap_df is not None
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:141:        assert "gap" in gap_df.columns
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:142:        assert "avg_corr" in gap_df.columns
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:143:        assert "max_eig_norm" in gap_df.columns
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:147:        gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:148:        assert gap_df is not None
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:149:        assert gap_df["gap"].notna().all()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:150:        assert np.isfinite(gap_df["gap"].values).all()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:170:        gap_df = _compute_complexity_gap_series(closes)
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:171:        if gap_df is not None and len(gap_df) > 0:
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:173:            avg_gap = gap_df["gap"].mean()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:174:            avg_corr = gap_df["avg_corr"].mean()
Q:\finance-analyzer\tests\test_signal_complexity_gap_regime.py:216:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:19:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signals_news_event.py:37:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:43:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:49:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:67:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:78:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:89:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_news_event.py:94:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_oscillators.py:51:def _make_df(
Q:\finance-analyzer\tests\test_signals_oscillators.py:96:def _make_trending_df(
Q:\finance-analyzer\tests\test_signals_oscillators.py:125:def _make_flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_oscillators.py:180:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:190:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:196:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:204:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:210:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:217:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:223:        df = _make_df(n=100, seed=7)
Q:\finance-analyzer\tests\test_signals_oscillators.py:232:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:257:        df = _make_df(n=1)
Q:\finance-analyzer\tests\test_signals_oscillators.py:264:        df = _make_df(n=MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_oscillators.py:271:        df = _make_df(n=MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_oscillators.py:276:        df = _make_df(n=100).drop(columns=["close"])
Q:\finance-analyzer\tests\test_signals_oscillators.py:281:        df = _make_df(n=100).drop(columns=["high"])
Q:\finance-analyzer\tests\test_signals_oscillators.py:286:        df = _make_df(n=100).drop(columns=["low"])
Q:\finance-analyzer\tests\test_signals_oscillators.py:319:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:325:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:346:        df = _make_trending_df(n=200, start=10.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_oscillators.py:356:        df = _make_trending_df(n=200, start=300.0, step=-0.5)
Q:\finance-analyzer\tests\test_signals_oscillators.py:365:        df = _make_trending_df(n=150, start=10.0, step=0.3)
Q:\finance-analyzer\tests\test_signals_oscillators.py:373:        df = _make_trending_df(n=150, start=300.0, step=-0.3)
Q:\finance-analyzer\tests\test_signals_oscillators.py:388:        df = _make_flat_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:393:        df = _make_flat_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:399:        df = _make_flat_df(n=100)
Q:\finance-analyzer\tests\test_signals_oscillators.py:414:        df = _make_df(n=100, volatility=50.0, seed=99)
Q:\finance-analyzer\tests\test_signals_oscillators.py:419:        df = _make_df(n=100, volatility=50.0, seed=99)
Q:\finance-analyzer\tests\test_signals_oscillators.py:425:        df = _make_df(n=80)
Q:\finance-analyzer\tests\test_signals_oscillators.py:802:        df = _make_df(n=150, seed=77)
Q:\finance-analyzer\tests\test_signals_oscillators.py:817:    def test_large_df_completes_without_error(self):
Q:\finance-analyzer\tests\test_signals_oscillators.py:819:        df = _make_df(n=500, seed=101)
Q:\finance-analyzer\tests\test_signals_oscillators.py:824:    def test_1000_bar_df_completes(self):
Q:\finance-analyzer\tests\test_signals_oscillators.py:826:        df = _make_df(n=1000, seed=202)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:15:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:59:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:71:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:80:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:93:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:109:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:121:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:128:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:138:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:222:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:234:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:242:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:250:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:264:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:279:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:302:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signal_hash_ribbons.py:317:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:15:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:28:def _make_trending_up_df(n=100):
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:48:def _make_trending_down_df(n=100):
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:71:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:80:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:86:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:98:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:103:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:109:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:120:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:129:        df = _make_trending_up_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:135:        df = _make_trending_down_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:141:        df = _make_df(n=200, seed=123)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:157:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:173:        df = _make_trending_up_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:178:        df = _make_trending_down_df(n=100)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:187:        df = _make_trending_up_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:193:        df = _make_trending_down_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:199:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_drift_regime_gate.py:206:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:14:def _make_df(n=200, seed=42):
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:27:def _make_trending_df(n=200, direction=1):
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:45:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:54:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:63:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:78:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:83:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:89:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:95:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:147:        df = _make_trending_df(n=200, direction=1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:152:        df = _make_trending_df(n=200, direction=1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:159:        df = _make_trending_df(n=200, direction=-1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:164:        df = _make_trending_df(n=200, direction=-1)
Q:\finance-analyzer\tests\test_signal_cubic_trend_persistence.py:171:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:18:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:31:def _make_ratio_df(n=200, trend="neutral"):
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:65:        df = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:71:        df = _make_ratio_df(10)
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:76:        df = _make_ratio_df(250, "rising")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:86:        df = _make_ratio_df(200, "rising")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:96:        df = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:106:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:107:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:117:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:118:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:128:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:129:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:145:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:152:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:159:        mock_fetch.return_value = _make_ratio_df(200, "falling")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:160:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:172:        mock_fetch.return_value = _make_ratio_df(200, "falling")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:173:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:189:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:190:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:199:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:200:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:206:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:207:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:223:        mock_fetch.return_value = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:224:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:233:        data = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:237:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:244:        data = _make_ratio_df(200, "neutral")
Q:\finance-analyzer\tests\test_signal_copper_gold_ratio.py:248:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_futures_basis.py:19:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_futures_basis.py:280:    def test_df_parameter_ignored(self, mock_fetch):
Q:\finance-analyzer\tests\test_signal_futures_basis.py:283:        df = _make_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:47:def _make_flat_df(n: int = 60, price: float = 100.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:61:def _make_trending_df(n: int = 80, start: float = 100.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:114:def _make_random_df(n: int = 100, seed: int = 42,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:143:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:152:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:158:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:166:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:173:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:182:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:219:        df = _make_random_df(MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:226:        df = _make_random_df(100).drop(columns=["volume"])
Q:\finance-analyzer\tests\test_signals_volume_flow.py:233:        df = _make_random_df(100).drop(columns=["close"])
Q:\finance-analyzer\tests\test_signals_volume_flow.py:258:    def _uptrend_df(self) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_volume_flow.py:259:        return _make_trending_df(n=80, start=50.0, step=1.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:264:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:274:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:282:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:290:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:296:        df = self._uptrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:308:    def _downtrend_df(self) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_volume_flow.py:309:        return _make_trending_df(n=80, start=150.0, step=-1.0,
Q:\finance-analyzer\tests\test_signals_volume_flow.py:314:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:324:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:332:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:338:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:344:        df = self._downtrend_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:358:        df = _make_flat_df(n=80, price=100.0, volume=500.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:364:        df = _make_flat_df(n=80, price=100.0, volume=500.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:370:        df = _make_flat_df(n=80, price=100.0, volume=500.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:388:        df = _make_trending_df(n=80, step=0.5, volume=0.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:395:        df = _make_trending_df(n=80, step=0.5, volume=0.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:401:        df = _make_trending_df(n=80, step=0.5, volume=0.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:417:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:423:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:441:        df = _make_random_df(MIN_ROWS)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:451:        df = _make_random_df(MIN_ROWS - 1)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:523:        df = _make_random_df(60)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:563:        df = _make_random_df(80)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:572:        df = _make_random_df(80)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:582:        df = _make_random_df(80)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:754:        df = _make_trending_df(n=80, step=1.0, volume=2_000.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:779:            df = _make_random_df(100, seed=seed)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:788:            df = _make_random_df(100, seed=seed)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:805:        df = _make_random_df(n)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:815:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:832:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:841:        df = _make_trending_df(n=80, start=50.0, step=0.5)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:849:        df = _make_trending_df(n=80, start=50.0, step=1.0)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:856:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:866:        df = _make_random_df(100)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:882:    def test_same_df_same_result(self):
Q:\finance-analyzer\tests\test_signals_volume_flow.py:883:        df = _make_random_df(100, seed=7)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:890:    def test_df_not_mutated(self):
Q:\finance-analyzer\tests\test_signals_volume_flow.py:892:        df = _make_random_df(100, seed=11)
Q:\finance-analyzer\tests\test_signals_volume_flow.py:915:    def _make_three_day_df(self) -> pd.DataFrame:
Q:\finance-analyzer\tests\test_signals_volume_flow.py:940:        df = self._make_three_day_df()
Q:\finance-analyzer\tests\test_signals_volume_flow.py:952:        df = self._make_three_day_df()
Q:\finance-analyzer\tests\test_signal_engine_core.py:29:def _make_ohlcv_df(n=50, close_start=100.0, trend="flat", volume_start=1000.0):
Q:\finance-analyzer\tests\test_signal_engine_core.py:62:def _make_trap_df(price_trend="up", volume_pattern="declining", n=50):
Q:\finance-analyzer\tests\test_signal_engine_core.py:68:    df = _make_ohlcv_df(n, close_start=100.0, trend=price_trend, volume_start=1000.0)
Q:\finance-analyzer\tests\test_signal_engine_core.py:428:        df = _make_ohlcv_df(50, trend="flat")
Q:\finance-analyzer\tests\test_signal_engine_core.py:462:        df = _make_trap_df(price_trend="up", volume_pattern="declining")
Q:\finance-analyzer\tests\test_signal_engine_core.py:472:        df = _make_trap_df(price_trend="down", volume_pattern="declining")
Q:\finance-analyzer\tests\test_signal_engine_core.py:480:        df = _make_trap_df(price_trend="up", volume_pattern="expanding")
Q:\finance-analyzer\tests\test_signal_engine_core.py:488:        df = _make_trap_df(price_trend="up", volume_pattern="declining")
Q:\finance-analyzer\tests\test_signal_engine_core.py:495:    def test_no_trap_with_short_df(self):
Q:\finance-analyzer\tests\test_signal_engine_core.py:496:        df = _make_ohlcv_df(3)
Q:\finance-analyzer\tests\test_signal_engine_core.py:740:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_engine_core.py:752:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_engine_core.py:763:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_engine_core.py:773:        df = _make_ohlcv_df(50)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:16:def _make_df(n=200, seed=42):
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:29:def _make_trending_df(n=250, direction="up"):
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:47:def _make_mean_reverting_df(n=250):
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:160:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:166:        df = _make_trending_df(100, "down")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:208:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:217:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:225:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:239:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:250:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:256:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:263:        df = _make_df(n=300)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:270:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:286:        df = _make_trending_df(300, "up")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:293:        df = _make_trending_df(300, "down")
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:299:        df = _make_mean_reverting_df(300)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:308:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_hurst_regime.py:319:        df = _make_df(n=300)
Q:\finance-analyzer\tests\test_signal_improvements.py:43:def make_df(n=100, close_base=100.0, volatility=2.0):
Q:\finance-analyzer\tests\test_signal_improvements.py:83:        df = make_df()
Q:\finance-analyzer\tests\test_signal_improvements.py:92:        df = make_df(close_base=100.0)
Q:\finance-analyzer\tests\test_signal_improvements.py:97:        df = make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:20:def _make_df(n=100, trend="up", with_datetime_index=False, hour=22):
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:48:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:82:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:87:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:93:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:99:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:105:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:111:        df = _make_df(trend="up")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:162:        df = _make_df(n=50, with_datetime_index=True, hour=22)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:167:        df = _make_df(n=50, trend="up", with_datetime_index=True, hour=22)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:173:        df = _make_df(n=50, with_datetime_index=True, hour=3)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:197:        df = _make_df(n=50, trend="up")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:203:        df = _make_df(n=50, trend="down")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:217:        df = _make_df(n=30)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:224:        df = _make_df(n=50)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:232:        df = _make_df(n=30, with_datetime_index=True, hour=14)
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:238:        df = _make_df(n=30, with_datetime_index=True, hour=14)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:15:def _make_df(n=100, base_price=50000.0, vol=0.02):
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:33:        df = _make_df(50)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:41:        df = _make_df(5)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:108:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:120:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:129:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:145:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:157:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:166:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:176:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:183:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:192:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:200:        df = _make_df(base_price=3000.0)
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:211:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:220:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:234:        df = _make_df(n=50, vol=0.005)  # Low vol data
Q:\finance-analyzer\tests\test_signal_crypto_evrp.py:247:        df = _make_df(n=50, vol=0.05)  # High vol data
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:16:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:54:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:65:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:73:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:85:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:92:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:111:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:121:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:240:    def test_none_df(self, mock_fetch):
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:253:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_mahalanobis_turbulence.py:266:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_ovx_metals_spillover.py:17:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_ovx_metals_spillover.py:142:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:18:def _make_df(n=200, trend=0.001):
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:44:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:56:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:69:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:85:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:92:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:101:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:107:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:116:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:132:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:172:        df = _make_df(n=250, trend=0.01)
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:179:        df = _make_df(n=50)
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:214:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_gold_real_yield_paradox.py:223:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:16:def _make_df(n=300, trend=0.0):
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:46:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:70:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:91:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:98:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:106:        df = _make_df(n=300, trend=0.1)
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:117:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:125:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:133:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:141:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:157:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:165:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_cross_asset_tsmom.py:235:        df = _make_df(n=300, trend=0.1)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:18:def _make_df(n=100, trend=0.0, seed=42):
Q:\finance-analyzer\tests\test_signal_network_momentum.py:49:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:59:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:80:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:86:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:93:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_network_momentum.py:100:        df = _make_df(n=200, trend=0.1)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:207:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:223:        df = _make_df(n=100, seed=99)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:235:        df = _make_df(n=100, seed=123)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:250:            df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_network_momentum.py:266:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_pipeline.py:15:from conftest import make_ohlcv_df
Q:\finance-analyzer\tests\test_signal_pipeline.py:49:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:67:        df = make_ohlcv_df(n=250, close_base=130.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:78:        df = make_ohlcv_df(n=250, close_base=2000.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:96:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:128:        df = make_ohlcv_df(n=250, close_base=130.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:140:        df = make_ohlcv_df(n=250, close_base=2000.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:153:        df = make_ohlcv_df(n=250, close_base=100.0)
Q:\finance-analyzer\tests\test_signal_pipeline.py:173:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:186:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:196:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:256:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:266:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:278:    def test_enhanced_signals_present_with_df(self, _mock):
Q:\finance-analyzer\tests\test_signal_pipeline.py:281:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signal_pipeline.py:295:    def test_enhanced_signals_hold_without_df(self, _mock):
Q:\finance-analyzer\tests\test_signal_pipeline.py:311:    def test_enhanced_signals_hold_with_short_df(self, _mock):
Q:\finance-analyzer\tests\test_signal_pipeline.py:314:        df = make_ohlcv_df(n=10)  # too short
Q:\finance-analyzer\tests\test_signal_pipeline.py:348:        df = make_ohlcv_df(n=250)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:37:def _make_df(n: int = 100, close_base: float = 100.0,
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:59:def _make_declining_df(n: int = 30, start: float = 100.0,
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:80:def _make_rising_df(n: int = 30, start: float = 100.0,
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:106:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:114:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:119:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:124:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:133:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:144:        df = _make_df(100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:169:        df = _make_df(2)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:182:        df = _make_df(3)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:196:        df = _make_df(50, close_base=100.0, volatility=0.1)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:224:        df = _make_df(50, close_base=100.0, volatility=0.1)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:568:        df = _make_declining_df(n=50, start=100.0, step=-1.0)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:582:        df = _make_rising_df(n=50, start=100.0, step=1.0)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:616:        df = _make_df(50)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:650:    def _make_hourly_df(self, n: int = 50, start_price: float = 100.0):
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:668:        df = self._make_hourly_df(n=50)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:709:        df_50 = self._make_hourly_df(n=50)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:710:        df_100 = self._make_hourly_df(n=100)
Q:\finance-analyzer\tests\test_signals_mean_reversion.py:760:            df = self._make_hourly_df(n=20)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:8:def _make_df(n=200):
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:21:def _make_skewed_df(n=300, skew_direction="negative"):
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:53:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:62:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:68:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:80:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:85:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:91:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:107:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:114:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:134:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:139:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:144:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:154:        df = _make_skewed_df(n=300, skew_direction="negative")
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:161:        df = _make_skewed_df(n=300, skew_direction="positive")
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:185:        df = _make_df(n=60)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:192:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_realized_skewness.py:215:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:13:def _make_df(n=250):
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:31:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:39:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:58:        df = _make_df(n=50)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:66:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:197:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_residual_pair_reversion.py:205:        df = _make_df(n=250)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:17:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:30:def _make_trending_df(n=100, direction="up"):
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:46:def _make_noisy_df(n=100):
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:106:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:114:        df = _make_noisy_df(100)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:130:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:136:        df = _make_trending_df(100, "down")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:151:        df = _make_trending_df(100, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:166:        df = _make_df(150)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:181:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:190:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:196:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:210:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:215:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:221:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:234:        df = _make_trending_df(200, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:240:        df = _make_trending_df(200, "up")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:248:        df = _make_trending_df(200, "down")
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:255:        df = _make_noisy_df(200)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:278:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:284:        df = _make_df(n=MIN_ROWS)
Q:\finance-analyzer\tests\test_signal_shannon_entropy.py:290:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:16:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:37:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:52:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:68:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:82:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:96:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:108:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:122:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:138:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:235:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:250:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_treasury_risk_rotation.py:263:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:12:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:25:def _make_trending_df(n=100, direction="up"):
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:41:def _make_volatile_df(n=100):
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:75:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:81:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:93:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:98:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:104:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:119:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:126:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:133:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:139:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:144:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:153:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:161:        df = _make_df(n=100)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:172:        df = _make_trending_df(n=200, direction="up")
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:179:        df = _make_trending_df(n=200, direction="down")
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:184:        df = _make_volatile_df(n=200)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:189:        df = _make_trending_df(n=300, direction="up")
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:265:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:270:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_statistical_jump_regime.py:275:        df = _make_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:15:def _make_df(n=100, trend=0.0, noise=1.0, seed=42):
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:43:def _make_trending_df(n=100):
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:45:    return _make_df(n=n, trend=0.5, noise=0.3)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:48:def _make_ranging_df(n=100):
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:50:    return _make_df(n=n, trend=0.0, noise=2.0)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:57:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:66:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:72:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:94:        df = _make_df(n=10)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:100:        df = _make_df(n=MIN_ROWS)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:105:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:111:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:120:            df = _make_df(seed=seed)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:129:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:140:        trending_df = _make_trending_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:141:        ranging_df = _make_ranging_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:143:        trending_ratio = _garman_klass_cc_ratio(trending_df).iloc[-1]
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:144:        ranging_ratio = _garman_klass_cc_ratio(ranging_df).iloc[-1]
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:152:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:160:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:210:        df = _make_trending_df(n=200)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:218:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:225:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:231:        df = _make_df(n=100, trend=0.01, noise=0.5)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:255:        df = _make_df(n=100, noise=10.0)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:261:        df = _make_df(n=1000)
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:267:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:274:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vol_ratio_regime.py:282:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_vix_term_structure.py:16:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_vix_term_structure.py:205:    def test_with_df_parameter(self, mock_cached):
Q:\finance-analyzer\tests\test_signal_vix_term_structure.py:207:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:19:def _make_df(n=100):
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:59:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:70:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:82:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:97:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:104:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:120:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:135:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_xtrend_equity_spillover.py:163:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:17:def _make_df(n=100, seed=42):
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:30:def _make_crash_df(n=100):
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:53:def _make_complacent_df(n=100):
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:70:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:80:        df = _make_crash_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:89:        df = _make_complacent_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:100:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:107:        df = _make_crash_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:116:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:124:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:130:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:149:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:158:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:167:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:184:        df = _make_df(n=3)
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:194:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:200:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:208:        df = _make_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:214:        df = _make_crash_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:225:        df = _make_complacent_df()
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:232:            df = _make_df(n=100, seed=seed)
Q:\finance-analyzer\tests\test_signal_williams_vix_fix.py:239:        df = _make_df(n=500)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:23:    def _make_df(self, n=30, close_override=None):
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:36:        df = self._make_df(30)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:48:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:58:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:69:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:79:        df = self._make_df(10)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:86:        df = self._make_df(30, close_override=close)
Q:\finance-analyzer\tests\test_thread_safety_and_nan.py:93:        df = self._make_df(50)
Q:\finance-analyzer\tests\test_train_signal_weights.py:60:        signals_df, returns = result
Q:\finance-analyzer\tests\test_train_signal_weights.py:61:        assert len(signals_df) == 100
Q:\finance-analyzer\tests\test_train_signal_weights.py:62:        assert "rsi" in signals_df.columns
Q:\finance-analyzer\tests\test_train_signal_weights.py:75:        signals_df, _ = result
Q:\finance-analyzer\tests\test_train_signal_weights.py:76:        assert "ml" not in signals_df.columns
Q:\finance-analyzer\tests\test_train_signal_weights.py:77:        assert "funding" not in signals_df.columns
Q:\finance-analyzer\tests\test_vix_fetch.py:28:def _patch_yf_ticker(hist_df):
Q:\finance-analyzer\tests\test_vix_fetch.py:31:    mock_ticker.history.return_value = hist_df

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\main.py'; "'$start=480; $end=508; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 553ms:
480:        """Fetch data + generate signals for one ticker. Fully thread-safe."""
481:        try:
482:            t0 = time.monotonic()
483:            tfs = collect_timeframes(source)
484:            tf_elapsed = time.monotonic() - t0
485:
486:            now_entry = tfs[0][1] if tfs else None
487:            now_df = None
488:            if now_entry and "indicators" in now_entry:
489:                ind = now_entry["indicators"]
490:                now_df = now_entry.get("_df")
491:            else:
492:                now_df = _fetch_klines(source, interval="15m", limit=100)
493:                ind = compute_indicators(now_df)
494:
495:            if ind is None:
496:                logger.info("%s: insufficient data, skipping", name)
497:                return name, None
498:
499:            price = ind["close"]
500:
501:            sig_start = time.monotonic()
502:            action, conf, extra = generate_signal(
503:                ind, ticker=name, config=config, timeframes=tfs, df=now_df
504:            )
505:            sig_elapsed = time.monotonic() - sig_start
506:            total_elapsed = time.monotonic() - t0
507:            logger.info(
508:                "%s: timing: tf=%.1fs sig=%.1fs total=%.1fs",

codex
I’m checking one more thing on the timestamp finding: whether this subsystem ever actually merges Binance and Alpaca candles the way Claude claims. A naive timestamp is one issue; a cross-source DST corruption path is a much stronger claim and needs evidence.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"\\['time'\\]|\\.time\\b|merge\\(|concat\\(|sort_values\\(.*time|set_index\\(.*time|tz_convert|tz_localize\" Q:\\finance-analyzer\\portfolio Q:\\finance-analyzer\\tests" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"from portfolio.onchain_data import _load_onchain_cache|get_onchain_data\\(|No BGeometrics token, using stale cache|time.time\\(\\) - ts|_coerce_epoch\" Q:\\finance-analyzer\\portfolio\\onchain_data.py Q:\\finance-analyzer\\tests\\test_onchain_data.py" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
from pathlib import Path
text = Path(r'Q:\\finance-analyzer\\portfolio\\llm_batch.py').read_text(encoding='utf-8').splitlines()
for start,end in [(305,490)]:
    for i in range(start,end+1):
        if i<=len(text):
            print(f'{i}:{text[i-1]}')
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 559ms:
Q:\finance-analyzer\portfolio\onchain_data.py:12:    data = get_onchain_data()  # returns dict or None
Q:\finance-analyzer\portfolio\onchain_data.py:29:def _coerce_epoch(value) -> float:
Q:\finance-analyzer\portfolio\onchain_data.py:63:        "_coerce_epoch: unparseable value type=%s repr=%.50r — returning 0.0",
Q:\finance-analyzer\portfolio\onchain_data.py:112:        if time.time() - ts > max_age_seconds:
Q:\finance-analyzer\portfolio\onchain_data.py:244:def get_onchain_data():
Q:\finance-analyzer\portfolio\onchain_data.py:260:        cache_ts = _coerce_epoch(raw_ts)
Q:\finance-analyzer\portfolio\onchain_data.py:280:            logger.debug("No BGeometrics token, using stale cache")
Q:\finance-analyzer\tests\test_onchain_data.py:117:# Tests for the main get_onchain_data() aggregator
Q:\finance-analyzer\tests\test_onchain_data.py:129:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:156:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:187:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:285:        result = get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:289:        from portfolio.onchain_data import _load_onchain_cache, _save_onchain_cache
Q:\finance-analyzer\tests\test_onchain_data.py:297:        from portfolio.onchain_data import _load_onchain_cache, _save_onchain_cache
Q:\finance-analyzer\tests\test_onchain_data.py:305:# A-DE-5: _coerce_epoch — defensive timestamp parsing
Q:\finance-analyzer\tests\test_onchain_data.py:316:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:317:        assert _coerce_epoch(1712345678) == 1712345678.0
Q:\finance-analyzer\tests\test_onchain_data.py:320:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:321:        assert _coerce_epoch(1712345678.5) == 1712345678.5
Q:\finance-analyzer\tests\test_onchain_data.py:324:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:325:        assert _coerce_epoch("1712345678") == 1712345678.0
Q:\finance-analyzer\tests\test_onchain_data.py:330:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:333:        assert _coerce_epoch(ts) == pytest.approx(expected)
Q:\finance-analyzer\tests\test_onchain_data.py:338:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:341:        assert _coerce_epoch(ts) == pytest.approx(expected)
Q:\finance-analyzer\tests\test_onchain_data.py:344:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:345:        assert _coerce_epoch("not a date") == 0.0
Q:\finance-analyzer\tests\test_onchain_data.py:346:        assert _coerce_epoch(None) == 0.0
Q:\finance-analyzer\tests\test_onchain_data.py:347:        assert _coerce_epoch({"weird": "shape"}) == 0.0
Q:\finance-analyzer\tests\test_onchain_data.py:348:        assert _coerce_epoch([]) == 0.0
Q:\finance-analyzer\tests\test_onchain_data.py:351:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:352:        assert _coerce_epoch("") == 0.0
Q:\finance-analyzer\tests\test_onchain_data.py:356:        seeding code in get_onchain_data() must NOT crash when reading
Q:\finance-analyzer\tests\test_onchain_data.py:357:        it back. Without _coerce_epoch, time.time() - "2026-04-09T..."
Q:\finance-analyzer\tests\test_onchain_data.py:369:        od.get_onchain_data()
Q:\finance-analyzer\tests\test_onchain_data.py:372:        """P1-14 (2026-05-02): When _coerce_epoch can't parse a value, it
Q:\finance-analyzer\tests\test_onchain_data.py:379:        from portfolio.onchain_data import _coerce_epoch
Q:\finance-analyzer\tests\test_onchain_data.py:384:        assert _coerce_epoch(1712345678) == 1712345678.0
Q:\finance-analyzer\tests\test_onchain_data.py:385:        assert not any("_coerce_epoch" in r.message for r in caplog.records)
Q:\finance-analyzer\tests\test_onchain_data.py:390:            assert _coerce_epoch(bad) == 0.0
Q:\finance-analyzer\tests\test_onchain_data.py:392:                "_coerce_epoch" in r.message and r.levelname == "DEBUG"

 succeeded in 605ms:
Q:\finance-analyzer\tests\test_accuracy_degradation.py:607:            "last_full_check_time": time.time() - 10,  # very recent
Q:\finance-analyzer\tests\test_accuracy_degradation.py:646:            "last_full_check_time": time.time() - 10,
Q:\finance-analyzer\tests\test_accuracy_cache_timestamps.py:67:        assert time.time() - raw2["time_3h"] < 5
Q:\finance-analyzer\tests\test_accuracy_cache_timestamps.py:69:        assert time.time() - raw2["time_1d"] > 7000
Q:\finance-analyzer\tests\test_accuracy_cache_timestamps.py:93:            "time": time.time(),  # Only legacy key, no time_1d
Q:\finance-analyzer\tests\test_agent_completion.py:146:        ai._agent_start = time.time()
Q:\finance-analyzer\tests\test_agent_completion.py:178:        ai._agent_start = time.time() - 60
Q:\finance-analyzer\tests\test_agent_completion.py:221:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_agent_completion.py:235:        ai._agent_start = time.time() - 10
Q:\finance-analyzer\tests\test_agent_completion.py:266:        ai._agent_start = time.time() - 20
Q:\finance-analyzer\tests\test_agent_completion.py:291:        ai._agent_start = time.time() - 20
Q:\finance-analyzer\tests\test_agent_completion.py:311:        ai._agent_start = time.time() - 20
Q:\finance-analyzer\tests\test_agent_completion.py:335:        ai._agent_start = time.time() - 10
Q:\finance-analyzer\tests\test_agent_completion.py:359:        ai._agent_start = time.time() - 10
Q:\finance-analyzer\tests\test_agent_completion.py:374:        ai._agent_start = time.time() - 10
Q:\finance-analyzer\tests\test_agent_completion.py:393:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_agent_completion.py:414:        ai._agent_start = time.time() - 10
Q:\finance-analyzer\tests\test_4h_digest.py:50:        ts = time.time() - 3600
Q:\finance-analyzer\tests\test_4h_digest.py:391:        with patch("portfolio.digest._get_last_digest_time", return_value=time.time() - 100):
Q:\finance-analyzer\tests\test_4h_digest.py:399:        with patch("portfolio.digest._get_last_digest_time", return_value=time.time() - 20000), \
Q:\finance-analyzer\tests\test_analyze.py:491:        old_time = t.time() - 16 * 60  # 16 min ago
Q:\finance-analyzer\tests\test_analyze.py:498:        recent = t.time() - 60  # 1 min ago
Q:\finance-analyzer\tests\test_analyze.py:510:        recent = t.time() - 60
Q:\finance-analyzer\tests\test_analyze.py:523:        recent = t.time() - 60
Q:\finance-analyzer\tests\test_agent_invocation_watchdog.py:156:    ai._agent_start_wall = time.time() - elapsed_s
Q:\finance-analyzer\tests\test_agent_invocation_watchdog.py:284:                       lambda: kill_calls.append(time.time()))
Q:\finance-analyzer\tests\test_agent_invocation_watchdog.py:288:    ai._agent_start_wall = time.time() - 200.0
Q:\finance-analyzer\tests\test_agent_invocation.py:1523:        ai._agent_start_wall = time.time() - 1000
Q:\finance-analyzer\tests\test_agent_invocation.py:1559:        ai._agent_start_wall = time.time() - 1000
Q:\finance-analyzer\tests\test_agent_invocation.py:1591:        ai._agent_start_wall = time.time() - 1000
Q:\finance-analyzer\tests\test_agent_invocation.py:1632:        ai._agent_start_wall = time.time() - 1000
Q:\finance-analyzer\tests\test_agent_invocation.py:1658:        ai._agent_start_wall = time.time() - 1000
Q:\finance-analyzer\tests\test_best_horizon.py:224:            "time": time.time(),
Q:\finance-analyzer\tests\test_best_horizon.py:252:            "time": time.time() - ACCURACY_CACHE_TTL - 1,  # expired
Q:\finance-analyzer\tests\test_batch2_fixes.py:95:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_batch2_fixes.py:138:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_batch2_fixes.py:168:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_batch2_fixes.py:194:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_batch2_fixes.py:208:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_batch2_fixes.py:222:        ai._agent_start = time.time() - 30
Q:\finance-analyzer\tests\test_batch2_fixes.py:247:        ai._agent_start = time.time() - 1000  # Started long ago
Q:\finance-analyzer\tests\test_batch2_fixes.py:295:        # time.time(). The old wall-clock seed made `elapsed = monotonic -
Q:\finance-analyzer\tests\test_avanza_order_lock.py:81:        deadline = time.time() + 2.0
Q:\finance-analyzer\tests\test_avanza_order_lock.py:82:        while not ready_marker.exists() and time.time() < deadline:
Q:\finance-analyzer\tests\test_avanza_order_lock.py:87:        start = time.time()
Q:\finance-analyzer\tests\test_avanza_order_lock.py:91:        elapsed = time.time() - start
Q:\finance-analyzer\tests\test_avanza_order_lock.py:115:        deadline = time.time() + 2.0
Q:\finance-analyzer\tests\test_avanza_order_lock.py:116:        while not ready_marker.exists() and time.time() < deadline:
Q:\finance-analyzer\tests\test_avanza_order_lock.py:121:        start = time.time()
Q:\finance-analyzer\tests\test_avanza_order_lock.py:123:            elapsed = time.time() - start
Q:\finance-analyzer\tests\test_bug_fixes_session4.py:96:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_bug_fixes_session4.py:109:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:184:                with patch("portfolio.avanza_session.time.sleep"):
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:186:                    with patch("portfolio.avanza_session.time.monotonic",
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:219:                with patch("portfolio.avanza_session.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:236:                with patch("portfolio.avanza_session.time.sleep"):
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:237:                    with patch("portfolio.avanza_session.time.monotonic", side_effect=[0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:257:                with patch("portfolio.avanza_session.time.sleep"):
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:258:                    with patch("portfolio.avanza_session.time.monotonic",
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:269:                with patch("portfolio.avanza_session.time.sleep"):
Q:\finance-analyzer\tests\test_avanza_session_cancel_sl.py:270:                    with patch("portfolio.avanza_session.time.monotonic", side_effect=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]):
Q:\finance-analyzer\tests\test_bigbet.py:281:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:302:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:324:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:347:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:377:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:402:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:437:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:474:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:510:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:583:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:615:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:621:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:627:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:635:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:641:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:679:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:699:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:718:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:743:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:766:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:788:        now = time.time()
Q:\finance-analyzer\tests\test_bigbet.py:808:        now = time.time()
Q:\finance-analyzer\portfolio\data_refresh.py:25:    end_time = int(time.time() * 1000)
Q:\finance-analyzer\portfolio\accuracy_stats.py:109:    if time.time() - cache.get("time", 0) >= _SIGNAL_UTILITY_DISK_TTL:
Q:\finance-analyzer\portfolio\accuracy_stats.py:138:            cache["time"] = time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:650:        now = time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:661:                _signal_utility_cache[horizon] = (time.time(), disk_cached)
Q:\finance-analyzer\portfolio\accuracy_stats.py:667:            _signal_utility_cache[horizon] = (time.time(), result)
Q:\finance-analyzer\portfolio\accuracy_stats.py:981:            if time.time() - cache.get("time", 0) < ACTIVATION_CACHE_TTL:
Q:\finance-analyzer\portfolio\accuracy_stats.py:988:            _atomic_write_json(cache_file, {"rates": rates, "time": time.time()})
Q:\finance-analyzer\portfolio\accuracy_stats.py:1001:            if time.time() - ts < ACCURACY_CACHE_TTL:
Q:\finance-analyzer\portfolio\accuracy_stats.py:1017:        cache[f"time_{horizon}"] = time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:1019:        cache["time"] = time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:1175:        now: Override clock for tests. Defaults to time.time().
Q:\finance-analyzer\portfolio\accuracy_stats.py:1182:    t = now if now is not None else time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:1388:            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
Q:\finance-analyzer\portfolio\accuracy_stats.py:1407:        cache["time"] = time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:1447:    now = time.time()
Q:\finance-analyzer\portfolio\accuracy_stats.py:1459:            _regime_accuracy_cache[horizon] = (time.time(), disk_cached)
Q:\finance-analyzer\portfolio\accuracy_stats.py:1478:        _regime_accuracy_cache[horizon] = (time.time(), result)
Q:\finance-analyzer\portfolio\accuracy_stats.py:1729:            if time.time() - cached.get("time", 0) < ACCURACY_CACHE_TTL:
Q:\finance-analyzer\portfolio\accuracy_stats.py:1800:            _atomic_write_json(BEST_HORIZON_CACHE_FILE, {"time": time.time(), "data": result})
Q:\finance-analyzer\portfolio\accuracy_stats.py:1923:            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
Q:\finance-analyzer\portfolio\accuracy_stats.py:1942:        cache["time"] = time.time()
Q:\finance-analyzer\portfolio\agent_invocation.py:354:    entry had duration_s=-1776254571.5 (matches time.monotonic() - time.time()).
Q:\finance-analyzer\portfolio\agent_invocation.py:373:        wall_elapsed = time.time() - _agent_start_wall
Q:\finance-analyzer\portfolio\agent_invocation.py:866:        _agent_start_wall = time.time()  # wall-clock fallback for P2B
Q:\finance-analyzer\portfolio\agent_invocation.py:1161:    # auth_failure entry (-1776254571.5, matching time.monotonic() - time.time()).
Q:\finance-analyzer\portfolio\data_collector.py:286:            if cached and time.time() - cached["time"] < ttl:
Q:\finance-analyzer\portfolio\data_collector.py:309:                _ss._tool_cache[cache_key] = {"data": entry, "time": time.time()}
Q:\finance-analyzer\portfolio\analyze.py:575:    now = time.time()
Q:\finance-analyzer\portfolio\analyze.py:622:    start_time = time.time()
Q:\finance-analyzer\portfolio\analyze.py:648:            elapsed_mins = (time.time() - start_time) / 60
Q:\finance-analyzer\portfolio\analyze.py:743:                t0 = time.time()
Q:\finance-analyzer\portfolio\analyze.py:754:                    c_elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\analyze.py:814:                last_claude_time = time.time()
Q:\finance-analyzer\portfolio\analyze.py:834:        elapsed_mins = (time.time() - start_time) / 60
Q:\finance-analyzer\portfolio\accuracy_degradation.py:351:    elapsed = time.time() - float(state.get("last_full_check_time", 0.0) or 0.0)
Q:\finance-analyzer\portfolio\accuracy_degradation.py:393:    state["last_full_check_time"] = time.time()
Q:\finance-analyzer\portfolio\accuracy_degradation.py:696:    now_ts = now_ts if now_ts is not None else time.time()
Q:\finance-analyzer\portfolio\accuracy_degradation.py:902:    state["last_summary_send_time"] = time.time()
Q:\finance-analyzer\portfolio\daily_digest.py:84:    return not (last and (time.time() - last) < 72000)
Q:\finance-analyzer\portfolio\daily_digest.py:275:            _set_last_daily_digest_time(time.time())
Q:\finance-analyzer\portfolio\alert_budget.py:32:        cutoff = time.time() - self.window_seconds
Q:\finance-analyzer\portfolio\alert_budget.py:39:            self._sent_timestamps.append(time.time())
Q:\finance-analyzer\portfolio\alert_budget.py:43:            self._sent_timestamps.append(time.time())
Q:\finance-analyzer\portfolio\bigbet.py:164:    t0 = time.time()
Q:\finance-analyzer\portfolio\bigbet.py:182:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\bigbet.py:201:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\bigbet.py:204:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\bigbet.py:207:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\bigbet.py:218:            "elapsed_s": round(time.time() - t0, 2),
Q:\finance-analyzer\portfolio\bigbet.py:429:    now = time.time()
Q:\finance-analyzer\portfolio\cumulative_tracker.py:43:    now = time.time()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:86:    if last_ts and (time.time() - last_ts) < MIN_GAP_SECONDS:
Q:\finance-analyzer\portfolio\crypto_scheduler.py:382:        state["last_report_time"] = time.time()
Q:\finance-analyzer\portfolio\crypto_precompute.py:72:    now = time.time()
Q:\finance-analyzer\portfolio\crypto_macro_data.py:234:            now = time.time()
Q:\finance-analyzer\portfolio\crypto_macro_data.py:273:        cutoff = time.time() - (max_age_days * 86400)
Q:\finance-analyzer\portfolio\crypto_macro_data.py:299:            if time.time() - latest_ts < 3600:
Q:\finance-analyzer\portfolio\crypto_macro_data.py:303:            "ts": time.time(),
Q:\finance-analyzer\portfolio\crypto_macro_data.py:348:        now = time.time()
Q:\finance-analyzer\portfolio\crypto_macro_data.py:395:        cutoff = time.time() - (NETFLOW_HISTORY_MAX_DAYS * 86400)
Q:\finance-analyzer\portfolio\crypto_macro_data.py:420:            if time.time() - latest_ts < 21600:  # 6h
Q:\finance-analyzer\portfolio\crypto_macro_data.py:423:        entry = {"ts": time.time(), "netflow": netflow}
Q:\finance-analyzer\tests\test_circuit_breaker.py:42:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 31):
Q:\finance-analyzer\tests\test_circuit_breaker.py:52:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 11):
Q:\finance-analyzer\tests\test_circuit_breaker.py:65:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 11):
Q:\finance-analyzer\tests\test_circuit_breaker.py:78:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 6):
Q:\finance-analyzer\tests\test_circuit_breaker.py:162:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 6):
Q:\finance-analyzer\tests\test_circuit_breaker.py:175:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 59):
Q:\finance-analyzer\tests\test_circuit_breaker.py:182:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 60):
Q:\finance-analyzer\tests\test_circuit_breaker.py:207:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 11):
Q:\finance-analyzer\tests\test_circuit_breaker.py:310:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:322:        with patch("portfolio.circuit_breaker.time.monotonic", return_value=t):
Q:\finance-analyzer\tests\test_circuit_breaker.py:332:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:345:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:363:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:385:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:392:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:405:            with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:412:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:423:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:430:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:444:            with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:461:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\tests\test_circuit_breaker.py:466:        with patch("portfolio.circuit_breaker.time.monotonic",
Q:\finance-analyzer\portfolio\earnings_calendar.py:167:    now = time.time()
Q:\finance-analyzer\portfolio\digest.py:263:    if last and (time.time() - last) < DIGEST_INTERVAL:
Q:\finance-analyzer\portfolio\digest.py:268:        _set_last_digest_time(time.time())
Q:\finance-analyzer\portfolio\fin_evolve.py:1138:    now = time.time()
Q:\finance-analyzer\portfolio\econ_dates.py:155:            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:180:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:224:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:273:            evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
Q:\finance-analyzer\tests\test_claude_gate.py:101:        t0 = time.time()
Q:\finance-analyzer\tests\test_claude_gate.py:109:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\claude_gate.py:487:    t0 = time.time()
Q:\finance-analyzer\portfolio\claude_gate.py:534:    duration = round(time.time() - t0, 2)
Q:\finance-analyzer\portfolio\claude_gate.py:592:    t0 = time.time()
Q:\finance-analyzer\portfolio\claude_gate.py:636:    duration = round(time.time() - t0, 2)
Q:\finance-analyzer\portfolio\fish_monitor_smart.py:106:        self.start_time = time.time()
Q:\finance-analyzer\portfolio\fish_monitor_smart.py:365:        elapsed_hours = (time.time() - self.start_time) / 3600
Q:\finance-analyzer\portfolio\fish_monitor_smart.py:497:        elapsed = (time.time() - self.start_time) / 3600
Q:\finance-analyzer\portfolio\fish_monitor_smart.py:657:                now = time.time()
Q:\finance-analyzer\portfolio\ic_computation.py:237:    t0 = time.time()
Q:\finance-analyzer\portfolio\ic_computation.py:242:        "time": time.time(),
Q:\finance-analyzer\portfolio\ic_computation.py:248:    dt = time.time() - t0
Q:\finance-analyzer\portfolio\ic_computation.py:258:    if time.time() - cache.get("time", 0) > IC_CACHE_TTL:
Q:\finance-analyzer\portfolio\forecast_signal.py:362:        t0 = time.time()
Q:\finance-analyzer\portfolio\forecast_signal.py:366:            entry["chronos_time_ms"] = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\forecast_signal.py:369:        t0 = time.time()
Q:\finance-analyzer\portfolio\forecast_signal.py:373:            entry["prophet_time_ms"] = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\health.py:29:        state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
Q:\finance-analyzer\portfolio\health.py:49:    return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}
Q:\finance-analyzer\portfolio\health.py:60:        state["start_time"] = time.time()
Q:\finance-analyzer\portfolio\health.py:375:    now = time.time()
Q:\finance-analyzer\portfolio\gpu_gate.py:68:        return (time.time() - _GPU_LOCK_FILE.stat().st_mtime) > _STALE_SECONDS
Q:\finance-analyzer\portfolio\gpu_gate.py:198:    deadline = time.time() + timeout
Q:\finance-analyzer\portfolio\gpu_gate.py:201:    remaining = deadline - time.time()
Q:\finance-analyzer\portfolio\gpu_gate.py:211:        while time.time() < deadline:
Q:\finance-analyzer\portfolio\gpu_gate.py:217:                    os.write(fd, f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}".encode())
Q:\finance-analyzer\portfolio\gpu_gate.py:251:        t0 = time.time()
Q:\finance-analyzer\portfolio\gpu_gate.py:255:            elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\fx_rates.py:29:    now = time.time()
Q:\finance-analyzer\portfolio\fx_rates.py:76:    now = time.time()
Q:\finance-analyzer\portfolio\indicators.py:80:    tr = pd.concat(
Q:\finance-analyzer\portfolio\forecast_accuracy.py:43:    now = time.time()
Q:\finance-analyzer\portfolio\iskbets.py:123:    tr = pd.concat(
Q:\finance-analyzer\portfolio\iskbets.py:312:    t0 = time.time()
Q:\finance-analyzer\portfolio\iskbets.py:329:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\iskbets.py:346:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\iskbets.py:349:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\iskbets.py:352:        elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\iskbets.py:363:            "elapsed_s": round(time.time() - t0, 2),
Q:\finance-analyzer\portfolio\futures_data.py:52:            "time": data.get("time", int(time.time() * 1000)),
Q:\finance-analyzer\portfolio\elongir\runner.py:191:        last_report_time = time.time()
Q:\finance-analyzer\portfolio\elongir\runner.py:214:                if time.time() - last_report_time >= cfg.telegram_report_interval:
Q:\finance-analyzer\portfolio\elongir\runner.py:218:                    last_report_time = time.time()
Q:\finance-analyzer\portfolio\golddigger\augmented_signals.py:91:        now = time.time()
Q:\finance-analyzer\portfolio\golddigger\augmented_signals.py:194:        self._state.last_refresh = time.time()
Q:\finance-analyzer\portfolio\golddigger\data_provider.py:117:    now = time.time()
Q:\finance-analyzer\portfolio\golddigger\data_provider.py:320:    now = time.time()
Q:\finance-analyzer\tests\test_crypto_precompute.py:56:            "last_run_epoch": time.time() - 1,
Q:\finance-analyzer\tests\test_crypto_precompute.py:76:            "last_run_epoch": time.time() - 999 * 3600,
Q:\finance-analyzer\portfolio\llama_server.py:291:    start = time.time()
Q:\finance-analyzer\portfolio\llama_server.py:300:    while time.time() < deadline:
Q:\finance-analyzer\portfolio\llama_server.py:305:            remaining = max(0.0, deadline - time.time())
Q:\finance-analyzer\portfolio\llama_server.py:307:            return time.time() - start
Q:\finance-analyzer\portfolio\llama_server.py:309:            return time.time() - start
Q:\finance-analyzer\portfolio\llama_server.py:310:    return time.time() - start
Q:\finance-analyzer\portfolio\llama_server.py:346:        deadline = time.time() + 90
Q:\finance-analyzer\portfolio\llama_server.py:347:        while time.time() < deadline:
Q:\finance-analyzer\portfolio\llama_server.py:384:    deadline = time.time() + timeout
Q:\finance-analyzer\portfolio\llama_server.py:385:    while time.time() < deadline:
Q:\finance-analyzer\portfolio\golddigger\runner.py:271:        _last_session_check = time.time()
Q:\finance-analyzer\portfolio\golddigger\runner.py:283:                if live and page and (time.time() - _last_session_check) > cfg.session_check_interval:
Q:\finance-analyzer\portfolio\golddigger\runner.py:340:                    _last_session_check = time.time()
Q:\finance-analyzer\portfolio\log_rotation.py:16:    if time.time() - _last_rotation > 86400:  # 24 hours
Q:\finance-analyzer\portfolio\log_rotation.py:19:            _last_rotation = time.time()
Q:\finance-analyzer\tests\test_cumulative_tracker.py:40:        mock_last_ts.return_value = time.time() - 30 * 60  # 30 min ago
Q:\finance-analyzer\tests\test_cumulative_tracker.py:50:        mock_last_ts.return_value = time.time() - 60 * 60  # 60 min ago
Q:\finance-analyzer\portfolio\loop_contract.py:816:    age_seconds = time.time() - ACCURACY_SNAPSHOTS_FILE.stat().st_mtime
Q:\finance-analyzer\portfolio\loop_contract.py:1122:    cutoff = time.time() - _TMP_RESIDUE_MIN_AGE_S
Q:\finance-analyzer\portfolio\loop_contract.py:1148:            "age_s": round(time.time() - mtime, 1),
Q:\finance-analyzer\portfolio\loop_contract.py:1394:    now = time.time()
Q:\finance-analyzer\portfolio\loop_contract.py:1750:        return (time.time() - self._last_heal_time) > SELF_HEAL_COOLDOWN_S
Q:\finance-analyzer\portfolio\loop_contract.py:1754:        self._last_heal_time = time.time()
Q:\finance-analyzer\portfolio\loop_contract.py:2161:            critical, time.time(), cooldown_s, state_file=state_file,
Q:\finance-analyzer\portfolio\loop_contract.py:2225:            time.time() - tracker._last_heal_time,
Q:\finance-analyzer\tests\test_daily_digest.py:37:        mock_last.return_value = time.time() - 3600  # 1 hour ago
Q:\finance-analyzer\portfolio\main.py:386:        if not pm or (time.time() - pm.get("_epoch", 0)) > 72000:
Q:\finance-analyzer\portfolio\main.py:389:                result["_epoch"] = time.time()
Q:\finance-analyzer\portfolio\message_throttle.py:41:    return (time.time() - last_sent) >= cooldown
Q:\finance-analyzer\portfolio\message_throttle.py:63:    state["pending_ts"] = time.time()
Q:\finance-analyzer\portfolio\message_throttle.py:108:    state["last_analysis_sent"] = time.time()
Q:\finance-analyzer\portfolio\metals_precompute.py:65:    now = time.time()
Q:\finance-analyzer\portfolio\metals_precompute.py:134:    now = time.time()
Q:\finance-analyzer\portfolio\metals_orderbook.py:84:            "ts": int(time.time() * 1000),
Q:\finance-analyzer\tests\test_data_collector.py:655:        _ss._tool_cache[cache_key] = {"data": cached_entry, "time": time.time()}
Q:\finance-analyzer\tests\test_data_collector.py:678:        _ss._tool_cache[cache_key] = {"data": cached_entry, "time": time.time() - 9999}
Q:\finance-analyzer\portfolio\microstructure_state.py:70:        "ts": depth.get("ts", int(time.time() * 1000)),
Q:\finance-analyzer\portfolio\microstructure_state.py:210:        ms["ts"] = int(time.time() * 1000)
Q:\finance-analyzer\portfolio\microstructure_state.py:226:    age_ms = int(time.time() * 1000) - entry.get("ts", 0)
Q:\finance-analyzer\portfolio\ml_trainer.py:71:    tr = pd.concat(
Q:\finance-analyzer\portfolio\ml_trainer.py:90:    body_top = pd.concat([opn, close], axis=1).max(axis=1)
Q:\finance-analyzer\portfolio\ml_trainer.py:91:    body_bot = pd.concat([opn, close], axis=1).min(axis=1)
Q:\finance-analyzer\portfolio\ml_trainer.py:127:    return pd.concat(frames, ignore_index=True)
Q:\finance-analyzer\tests\test_duration_non_negative.py:39:        ai._agent_start = time.time()  # huge value vs time.monotonic()
Q:\finance-analyzer\tests\test_duration_non_negative.py:74:        ai._agent_start = time.time()  # POISONED — epoch scale
Q:\finance-analyzer\tests\test_duration_non_negative.py:105:        ai._agent_start = time.time()  # POISONED
Q:\finance-analyzer\tests\test_duration_non_negative.py:130:        ai._agent_start = time.time()  # POISONED monotonic
Q:\finance-analyzer\tests\test_duration_non_negative.py:132:        ai._agent_start_wall = time.time() - 200.0
Q:\finance-analyzer\tests\test_duration_non_negative.py:161:        ai._agent_start = time.time()  # POISONED (reproduces the bug)
Q:\finance-analyzer\portfolio\ml_signal.py:72:    tr = pd.concat(
Q:\finance-analyzer\portfolio\ml_signal.py:88:    body_top = pd.concat([opn, close], axis=1).max(axis=1)
Q:\finance-analyzer\portfolio\ml_signal.py:89:    body_bot = pd.concat([opn, close], axis=1).min(axis=1)
Q:\finance-analyzer\portfolio\ml_signal.py:106:    now = time.time()
Q:\finance-analyzer\portfolio\mstr_precompute.py:57:    now = time.time()
Q:\finance-analyzer\portfolio\oil_precompute.py:67:    now = time.time()
Q:\finance-analyzer\portfolio\oil_precompute.py:118:    now = time.time()
Q:\finance-analyzer\portfolio\onchain_data.py:34:    `time.time() - cache_ts < TTL` which crashes with TypeError when ts is
Q:\finance-analyzer\portfolio\onchain_data.py:112:        if time.time() - ts > max_age_seconds:
Q:\finance-analyzer\portfolio\onchain_data.py:211:    result = {"ts": time.time()}
Q:\finance-analyzer\portfolio\onchain_data.py:261:        if time.time() - cache_ts < ONCHAIN_TTL:
Q:\finance-analyzer\portfolio\onchain_data.py:273:                                 time.time() - cache_ts)
Q:\finance-analyzer\portfolio\multi_agent_layer2.py:194:    deadline = time.time() + timeout
Q:\finance-analyzer\portfolio\multi_agent_layer2.py:197:        remaining = max(1, deadline - time.time())
Q:\finance-analyzer\tests\test_fin_evolve.py:1302:            "last_run_epoch": time.time() - fin_evolve._EVOLVE_INTERVAL_SEC - 100,
Q:\finance-analyzer\tests\test_fish_engine_integration.py:77:    engine = FishEngine(time_func=time.time)
Q:\finance-analyzer\tests\test_fish_engine_integration.py:224:            "entry_ts": time.time(),
Q:\finance-analyzer\tests\test_fish_engine_integration.py:287:        engine = FishEngine(time_func=time.time)
Q:\finance-analyzer\tests\test_fish_engine_integration.py:307:        engine = FishEngine(time_func=time.time, trade_log_path=log_path)
Q:\finance-analyzer\tests\test_fish_engine_integration.py:345:        engine = FishEngine(time_func=time.time, trade_log_path=log_path)
Q:\finance-analyzer\tests\test_fish_engine_integration.py:359:        engine = FishEngine(time_func=time.time, trade_log_path=log_path)
Q:\finance-analyzer\tests\test_fish_engine_bug_fixes.py:69:            stale_decision = _make_decision(signal_ts=time.time() - 90.0)
Q:\finance-analyzer\tests\test_fish_engine_bug_fixes.py:85:            fresh_decision = _make_decision(signal_ts=time.time() - 5.0)
Q:\finance-analyzer\tests\test_fish_monitor_smart.py:115:        monitor.start_time = time.time() - 3.5 * 3600
Q:\finance-analyzer\portfolio\mstr_loop\data_provider.py:182:        source_stale_seconds = time.time() - os.path.getmtime(agent_summary_path)
Q:\finance-analyzer\tests\test_gpu_gate.py:69:        _write_lock_raw(isolated_lock, "kronos", 999999999, time.time() - 600)
Q:\finance-analyzer\tests\test_gpu_gate.py:70:        os.utime(isolated_lock, (time.time() - 600, time.time() - 600))
Q:\finance-analyzer\tests\test_gpu_gate.py:82:        _write_lock_raw(isolated_lock, "kronos", os.getpid(), time.time() - 600)
Q:\finance-analyzer\tests\test_gpu_gate.py:83:        os.utime(isolated_lock, (time.time() - 600, time.time() - 600))
Q:\finance-analyzer\tests\test_gpu_gate.py:92:        _write_lock_raw(isolated_lock, "kronos", 999999999, time.time())
Q:\finance-analyzer\tests\test_gpu_gate.py:110:        os.utime(isolated_lock, (time.time() - 600, time.time() - 600))
Q:\finance-analyzer\tests\test_heartbeat_function.py:43:            "start_time": time.time() - 3600,
Q:\finance-analyzer\tests\test_health.py:144:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:159:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:175:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:195:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:224:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:254:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:272:            "start_time": time.time(),
Q:\finance-analyzer\tests\test_health.py:326:            "start_time": time.time(), "cycle_count": 1,
Q:\finance-analyzer\tests\test_health.py:344:            "start_time": time.time(), "cycle_count": 1,
Q:\finance-analyzer\tests\test_health.py:362:            "start_time": time.time(), "cycle_count": 14,
Q:\finance-analyzer\tests\test_health.py:381:            "start_time": time.time(), "cycle_count": 1,
Q:\finance-analyzer\tests\test_health.py:401:            "start_time": time.time(), "cycle_count": 1,
Q:\finance-analyzer\tests\test_health.py:429:            "start_time": time.time(), "cycle_count": 10,
Q:\finance-analyzer\tests\test_health.py:447:            "start_time": time.time(), "cycle_count": 10,
Q:\finance-analyzer\portfolio\qwen3_signal.py:123:    t0 = time.time()
Q:\finance-analyzer\portfolio\qwen3_signal.py:132:    elapsed = time.time() - t0
Q:\finance-analyzer\portfolio\shared_state.py:48:    now = time.time()
Q:\finance-analyzer\portfolio\shared_state.py:69:        _now_evict = time.time()
Q:\finance-analyzer\portfolio\shared_state.py:90:        _loading_timestamps[key] = time.time()
Q:\finance-analyzer\portfolio\shared_state.py:154:    now = time.time()
Q:\finance-analyzer\portfolio\shared_state.py:204:            _loading_timestamps[key] = time.time()
Q:\finance-analyzer\portfolio\shared_state.py:235:            "time": time.time(),
Q:\finance-analyzer\portfolio\shared_state.py:272:            now = time.time()
Q:\finance-analyzer\portfolio\shared_state.py:328:    now = time.time()
Q:\finance-analyzer\tests\test_http_retry.py:45:    return patch("portfolio.http_retry.requests"), patch("portfolio.http_retry.time.sleep")
Q:\finance-analyzer\tests\test_http_retry.py:98:             patch("portfolio.http_retry.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_http_retry.py:113:             patch("portfolio.http_retry.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_http_retry.py:128:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:146:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:172:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:189:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:201:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:213:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:226:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:293:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:306:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:320:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:330:             patch("portfolio.http_retry.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_http_retry.py:345:             patch("portfolio.http_retry.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_http_retry.py:364:             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
Q:\finance-analyzer\tests\test_http_retry.py:381:             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
Q:\finance-analyzer\tests\test_http_retry.py:397:             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
Q:\finance-analyzer\tests\test_http_retry.py:411:             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
Q:\finance-analyzer\tests\test_http_retry.py:429:             patch("portfolio.http_retry.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_http_retry.py:440:             patch("portfolio.http_retry.time.sleep") as mock_sleep:
Q:\finance-analyzer\tests\test_http_retry.py:499:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:606:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:658:        with patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\tests\test_http_retry.py:752:             patch("portfolio.http_retry.time.sleep"):
Q:\finance-analyzer\portfolio\seasonality_updater.py:84:    df = df.set_index("time")
Q:\finance-analyzer\tests\test_indicators_core.py:242:        df = pd.concat([df, crash], ignore_index=True)
Q:\finance-analyzer\tests\test_indicators_core.py:257:        df = pd.concat([df, spike], ignore_index=True)
Q:\finance-analyzer\portfolio\signal_utils.py:52:    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:94:    return time.time() - _cache[tier]["ts"] > cooldowns[tier]
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:158:    now = time.time()
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:713:        _cache[tier]["ts"] = time.time()
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:929:                        _cache[tier]["ts"] = time.time()
Q:\finance-analyzer\tests\test_health_extended.py:121:        state = {"last_heartbeat": old_time, "start_time": time.time(),
Q:\finance-analyzer\tests\test_health_extended.py:142:        state = {"last_invocation_ts": recent, "start_time": time.time(),
Q:\finance-analyzer\portfolio\telegram_poller.py:61:        self._startup_time = time.time()
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:70:    now = time.time()
Q:\finance-analyzer\portfolio\signal_engine.py:1686:    now = time.time()
Q:\finance-analyzer\portfolio\signal_engine.py:1726:        now_ts = _time.time()
Q:\finance-analyzer\portfolio\signals\credit_spread.py:61:    now = time.time()
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:72:    if cached and (time.time() - cached[0]) < _DVOL_CACHE_TTL:
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:78:        now_ms = int(time.time() * 1000)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:107:        _DVOL_CACHE[cache_key] = (time.time(), dvol)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:122:    if cached and (time.time() - cached[0]) < _DVOL_HISTORY_TTL:
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:128:        now_ms = int(time.time() * 1000)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:171:        _DVOL_HISTORY_CACHE[cache_key] = (time.time(), series)
Q:\finance-analyzer\portfolio\trigger.py:189:            "time": time.time(),
Q:\finance-analyzer\portfolio\trigger.py:261:    _flip_now_ts = time.time()
Q:\finance-analyzer\portfolio\trigger.py:311:    _sent_now_ts = time.time()
Q:\finance-analyzer\portfolio\trigger.py:334:        state["last_trigger_time"] = time.time()
Q:\finance-analyzer\portfolio\trigger.py:345:            "time": time.time(),
Q:\finance-analyzer\portfolio\trigger.py:447:    hours_since = (time.time() - last_full) / 3600
Q:\finance-analyzer\portfolio\trigger.py:493:        state["last_full_review_time"] = time.time()
Q:\finance-analyzer\tests\test_iskbets.py:641:        poller._startup_time = time.time() - 120
Q:\finance-analyzer\tests\test_iskbets.py:647:                "date": int(time.time()),
Q:\finance-analyzer\tests\test_iskbets.py:665:        poller._startup_time = time.time() - 120
Q:\finance-analyzer\tests\test_iskbets.py:671:                "date": int(time.time()),
Q:\finance-analyzer\portfolio\signals\forecast.py:319:    t0 = time.time()
Q:\finance-analyzer\portfolio\signals\forecast.py:327:        ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:361:        ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:387:        ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:409:    t0 = time.time()
Q:\finance-analyzer\portfolio\signals\forecast.py:420:                ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:426:        ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:434:        ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:846:    t0 = time.time()
Q:\finance-analyzer\portfolio\signals\forecast.py:849:    chronos_ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\portfolio\signals\forecast.py:864:    t0 = time.time()
Q:\finance-analyzer\portfolio\signals\forecast.py:870:    kronos_ms = round((time.time() - t0) * 1000)
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:281:                t0 = time.time()
Q:\finance-analyzer\tests\test_kronos_backtest_feb27.py:287:                elapsed = time.time() - t0
Q:\finance-analyzer\tests\test_layer2_journal_contract.py:321:    now_t = time.time()
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:58:    now = time.time()
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:63:    now = time.time()
Q:\finance-analyzer\tests\test_llama_server.py:243:    @patch("portfolio.llama_server.time.sleep")
Q:\finance-analyzer\tests\test_llama_server.py:257:    @patch("portfolio.llama_server.time.sleep")
Q:\finance-analyzer\tests\test_llama_server.py:258:    @patch("portfolio.llama_server.time.time")
Q:\finance-analyzer\tests\test_llama_server.py:263:        mocked time.sleep and let time.time() read the real clock — under
Q:\finance-analyzer\tests\test_llama_server.py:266:        time.time also returns a monotonic counter advancing 0.1s per
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:108:    now = time.time()
Q:\finance-analyzer\tests\test_llm_batch.py:509:            "time": time.time() - age_seconds,
Q:\finance-analyzer\tests\test_loop_contract.py:402:        tracker._last_heal_time = time.time() - SELF_HEAL_COOLDOWN_S - 1
Q:\finance-analyzer\tests\test_loop_contract_accuracy_dispatcher.py:344:        forged = _time.time() - 24 * 3600
Q:\finance-analyzer\tests\test_loop_contract_alert_cooldown.py:106:            forged_ts = time.time() - 5 * 3600  # 5 h ago > 4 h default
Q:\finance-analyzer\tests\test_loop_contract_snapshot_freshness.py:45:    target_ts = time.time() - hours * 3600.0
Q:\finance-analyzer\tests\test_loop_contract_invariants.py:152:    old = time.time() - 10 * 60
Q:\finance-analyzer\tests\test_loop_contract_invariants.py:166:    old = time.time() - 10 * 60
Q:\finance-analyzer\portfolio\signals\volume_flow.py:91:            session_id = idx.tz_convert("UTC").date
Q:\finance-analyzer\tests\test_message_throttle.py:20:        mock_load.return_value = {"last_analysis_sent": time.time() - 60}  # 1 min ago
Q:\finance-analyzer\tests\test_message_throttle.py:27:        mock_load.return_value = {"last_analysis_sent": time.time() - 20000}  # >3h ago
Q:\finance-analyzer\tests\test_message_throttle.py:34:        mock_load.return_value = {"last_analysis_sent": time.time() - 600}  # 10 min ago
Q:\finance-analyzer\tests\test_message_throttle.py:84:        mock_load.return_value = {"pending_text": "queued message", "pending_ts": time.time()}
Q:\finance-analyzer\tests\test_metals_loop_autonomous.py:518:        ml._last_auto_telegram = time.time() - 60  # 60s ago (within 1800s cooldown)
Q:\finance-analyzer\tests\test_metals_loop_autonomous.py:531:        ml._last_auto_telegram = time.time() - 60  # recent, would normally throttle
Q:\finance-analyzer\tests\test_metals_loop_autonomous.py:805:        ml._last_auto_telegram = time.time() - 1860
Q:\finance-analyzer\tests\test_metals_risk.py:111:                state[key]["silver79"] = time.time() - 7200  # 2 hours ago
Q:\finance-analyzer\tests\test_metals_risk.py:235:        yesterday = time.time() - 86400
Q:\finance-analyzer\tests\test_metals_risk.py:255:        result = check_portfolio_drawdown(positions, prices, since_ts=time.time())
Q:\finance-analyzer\tests\test_metals_swing_p1_followups.py:70:    used pos_id = f"pos_{int(time.time())}" which collided."""
Q:\finance-analyzer\tests\test_metals_swing_p1_followups.py:74:    # Pin time.time() so both calls see the same second.
Q:\finance-analyzer\tests\test_metals_swing_p1_followups.py:75:    monkeypatch.setattr(mst.time, "time", lambda: 1735000000.0)
Q:\finance-analyzer\tests\test_mstr_loop_data_provider.py:213:    old_time = time.time() - 600  # 10 min old
Q:\finance-analyzer\tests\test_microstructure_state.py:37:        "ts": int(time.time() * 1000),
Q:\finance-analyzer\tests\test_mstr_precompute.py:78:            "last_run_epoch": time.time() - 1,
Q:\finance-analyzer\tests\test_onchain_data.py:290:        test_data = {"mvrv": 1.5, "ts": time.time()}
Q:\finance-analyzer\tests\test_onchain_data.py:298:        test_data = {"mvrv": 1.5, "ts": time.time() - 100000}
Q:\finance-analyzer\tests\test_onchain_data.py:311:    `time.time() - cache_ts` which crashes with TypeError on a string,
Q:\finance-analyzer\tests\test_onchain_data.py:357:        it back. Without _coerce_epoch, time.time() - "2026-04-09T..."
Q:\finance-analyzer\tests\test_per_ticker_accuracy_override.py:78:            cache = {"1d": data, "time": time.time() - mod.ACCURACY_CACHE_TTL - 100}
Q:\finance-analyzer\tests\test_per_ticker_accuracy_override.py:148:        cache = {"1d": cached_data, "time": time.time()}
Q:\finance-analyzer\tests\test_per_ticker_accuracy_override.py:179:        cache = {"1d": cached_data, "time": time.time()}
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:36:            acc_mod._regime_accuracy_cache["1d"] = (time.time(), cached)
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:189:                time.time() - 600.0,
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:207:            assert time.time() - cached[0] < 5.0
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:214:            acc_mod._regime_accuracy_cache["1d"] = (time.time(), {"a": 1})
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:215:            acc_mod._regime_accuracy_cache["3d"] = (time.time(), {"b": 2})
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:226:            acc_mod._regime_accuracy_cache["1d"] = (time.time(), {"a": 1})
Q:\finance-analyzer\tests\test_regime_accuracy_cache.py:227:            acc_mod._regime_accuracy_cache["3d"] = (time.time(), {"b": 2})
Q:\finance-analyzer\tests\test_regime_accuracy.py:242:        stale_time = time.time() - ACCURACY_CACHE_TTL - 1
Q:\finance-analyzer\tests\test_shared_state_cache.py:26:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:39:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:54:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:76:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:90:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:130:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:138:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:149:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:182:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state_cache.py:206:        now = time.time()
Q:\finance-analyzer\tests\test_shared_state.py:103:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:106:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:109:        mock_time.time.return_value = 1059.0  # 59s < 60s TTL
Q:\finance-analyzer\tests\test_shared_state.py:119:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:121:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:124:        mock_time.time.return_value = 1061.0  # 61s > 60s TTL
Q:\finance-analyzer\tests\test_shared_state.py:128:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:130:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:133:        mock_time.time.return_value = 1011.0
Q:\finance-analyzer\tests\test_shared_state.py:138:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:141:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:144:        mock_time.time.return_value = 1060.0  # exactly 60s
Q:\finance-analyzer\tests\test_shared_state.py:157:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:159:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:163:        mock_time.time.return_value = 1100.0  # 100s old, max stale = 60*3 = 180s
Q:\finance-analyzer\tests\test_shared_state.py:167:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:170:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:174:        mock_time.time.return_value = 1179.0
Q:\finance-analyzer\tests\test_shared_state.py:178:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:181:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:185:        mock_time.time.return_value = 1181.0
Q:\finance-analyzer\tests\test_shared_state.py:189:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:192:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:196:        mock_time.time.return_value = 1180.0
Q:\finance-analyzer\tests\test_shared_state.py:218:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:222:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:226:        mock_time.time.return_value = 1070.0
Q:\finance-analyzer\tests\test_shared_state.py:240:        mock_time.time.return_value = 1071.0
Q:\finance-analyzer\tests\test_shared_state.py:246:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:249:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:253:        mock_time.time.return_value = 1070.0
Q:\finance-analyzer\tests\test_shared_state.py:257:        mock_time.time.return_value = 1131.0
Q:\finance-analyzer\tests\test_shared_state.py:276:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:278:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:286:        mock_time.time.return_value = 5000.0
Q:\finance-analyzer\tests\test_shared_state.py:296:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:298:        mock_time.time.return_value = 5000.0
Q:\finance-analyzer\tests\test_shared_state.py:310:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:312:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:318:        mock_time.time.return_value = 5000.0
Q:\finance-analyzer\tests\test_shared_state.py:326:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:329:        mock_time.time.return_value = 5000.0
Q:\finance-analyzer\tests\test_shared_state.py:363:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:365:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:369:        mock_time.time.return_value = 1015.0  # 15s: short expired, long fresh
Q:\finance-analyzer\tests\test_shared_state.py:419:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:424:        mock_time.time.return_value = 0.3  # 0.3s since last call, need 1.0s
Q:\finance-analyzer\tests\test_shared_state.py:433:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:438:        mock_time.time.return_value = 10.2  # 0.2s elapsed, need 0.5s
Q:\finance-analyzer\tests\test_shared_state.py:450:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:455:        mock_time.time.return_value = 5.0  # 5s since last call >> 1.0s interval
Q:\finance-analyzer\tests\test_shared_state.py:462:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:468:        mock_time.time.return_value = 11.0  # exactly 1.0s elapsed
Q:\finance-analyzer\tests\test_shared_state.py:475:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:481:        mock_time.time.return_value = 1000.0
Q:\finance-analyzer\tests\test_shared_state.py:492:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:498:        mock_time.time.return_value = 5.0
Q:\finance-analyzer\tests\test_shared_state.py:505:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:511:        mock_time.time.return_value = 10.3
Q:\finance-analyzer\tests\test_shared_state.py:516:        # last_call is the reserved slot, not time.time() after sleep
Q:\finance-analyzer\tests\test_shared_state.py:524:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:533:        mock_time.time.side_effect = [10.3]
Q:\finance-analyzer\tests\test_shared_state.py:538:        # last_call should be set to the reserved slot, not time.time() after sleep
Q:\finance-analyzer\tests\test_shared_state.py:542:    @patch("portfolio.shared_state.time")
Q:\finance-analyzer\tests\test_shared_state.py:550:        mock_time.time.return_value = 10.3
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:206:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:213:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:228:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:235:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:250:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:257:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:264:            _cache["haiku"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:316:                _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:323:                _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:369:                _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:376:                _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:517:            _cache["haiku"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:518:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:519:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:540:            _cache["haiku"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:541:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:542:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:584:            _cache["haiku"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:585:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:586:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:614:            _cache["opus"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:615:            _cache["sonnet"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signals_claude_fundamental.py:616:            _cache["haiku"]["ts"] = time.time()
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:233:        df.index = df.index.tz_localize("UTC")
Q:\finance-analyzer\tests\test_signal_intraday_seasonality.py:239:        df.index = df.index.tz_localize("US/Eastern")
Q:\finance-analyzer\tests\test_signal_improvements.py:290:                "time": time.time() - 7200,
Q:\finance-analyzer\tests\test_signal_utility_cache.py:116:            _signal_utility_cache["1d"] = (time.time() - 5, value)
Q:\finance-analyzer\tests\test_signal_utility_cache.py:182:            {"time": time.time(), "1d": canned},
Q:\finance-analyzer\tests\test_signal_utility_cache.py:206:        assert time.time() - on_disk["time"] < 5
Q:\finance-analyzer\tests\test_signal_utility_cache.py:213:            {"time": time.time() - acc_mod._SIGNAL_UTILITY_DISK_TTL - 60, "1d": canned},
Q:\finance-analyzer\tests\test_trigger_core.py:132:    state["last_trigger_time"] = time.time() + 99999
Q:\finance-analyzer\tests\test_trigger_core.py:536:            now = time.time()
Q:\finance-analyzer\tests\test_trigger_core.py:552:            now = time.time()
Q:\finance-analyzer\tests\test_trigger_core.py:630:        _set_trigger_time(sf, time.time())
Q:\finance-analyzer\tests\test_trigger_core.py:633:        # (and then set it to time.time() at the end because triggered=True)
Q:\finance-analyzer\tests\test_trigger_core.py:646:            "last_full_review_time": time.time(),  # recent, so no T3 periodic
Q:\finance-analyzer\tests\test_trigger_core.py:657:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:677:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:694:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:705:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:722:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:732:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:749:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:766:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:785:                "last_full_review_time": time.time() - 14401,  # 4+ hours ago
Q:\finance-analyzer\tests\test_trigger_core.py:800:                "last_full_review_time": time.time() - 14401,  # 4+ hours ago
Q:\finance-analyzer\tests\test_trigger_core.py:815:                "last_full_review_time": time.time() - 7200,  # 2 hours ago
Q:\finance-analyzer\tests\test_trigger_core.py:834:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:845:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:866:        before = time.time()
Q:\finance-analyzer\tests\test_trigger_core.py:868:        after = time.time()
Q:\finance-analyzer\tests\test_trigger_core.py:954:        # last.time should be unchanged
Q:\finance-analyzer\tests\test_trigger_core.py:1098:        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
Q:\finance-analyzer\tests\test_trigger_core.py:1107:        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
Q:\finance-analyzer\tests\test_trigger_core.py:1117:        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
Q:\finance-analyzer\tests\test_trigger_core.py:1126:        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {"patient": 1}}
Q:\finance-analyzer\tests\test_trigger_core.py:1171:        with mock.patch("portfolio.trigger.time") as mock_time:
Q:\finance-analyzer\tests\test_trigger_core.py:1183:        with mock.patch("portfolio.trigger.time") as mock_time:
Q:\finance-analyzer\tests\test_trigger_core.py:1317:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_core.py:1391:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:85:        update = _build_update(update_id=42, msg_date=int(time.time()))
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:128:            msg_date=int(time.time()) - 5 * 60,
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:164:            msg_date=int(time.time()) - 5 * 60,
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:190:            msg_date=int(time.time()) - 7 * 24 * 3600,
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:209:        poller._handle_update(_build_update(update_id=200, msg_date=int(time.time())))
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:245:        update = _build_update(update_id=42, msg_date=int(time.time()))
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:273:            msg_date=int(time.time()) - 24 * 3600,
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:296:            msg_date=int(time.time()),
Q:\finance-analyzer\tests\test_telegram_poller_offset.py:329:            msg_date=int(time.time()) - 3 * 24 * 3600,
Q:\finance-analyzer\tests\test_tiered_invocation.py:64:            "last_full_review_time": time.time(),  # recent full review
Q:\finance-analyzer\tests\test_tiered_invocation.py:139:            last_full_review_time=time.time() - 5 * 3600,  # 5h ago (> 4h threshold)
Q:\finance-analyzer\tests\test_tiered_invocation.py:153:            last_full_review_time=time.time() - 5 * 3600,  # 5h ago
Q:\finance-analyzer\tests\test_tiered_invocation.py:167:        # time.time() - 0 >> 4h threshold during market hours → T3
Q:\finance-analyzer\tests\test_tiered_invocation.py:191:        assert time.time() - state["last_full_review_time"] < 5
Q:\finance-analyzer\tests\test_trigger_edge_cases.py:215:        state["last_trigger_time"] = time.time() + 9999
Q:\finance-analyzer\tests\test_trigger_edge_cases.py:224:        state["last_trigger_time"] = time.time() + 9999
Q:\finance-analyzer\tests\test_trigger_edge_cases.py:231:        state["last_trigger_time"] = time.time() + 9999
Q:\finance-analyzer\tests\test_trigger_edge_cases.py:250:        state["last_trigger_time"] = time.time() + 9999
Q:\finance-analyzer\tests\test_trigger_debounce.py:55:        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
Q:\finance-analyzer\tests\test_trigger_debounce.py:60:            "portfolio.trigger.time.monotonic",
Q:\finance-analyzer\tests\test_trigger_debounce.py:68:            "portfolio.trigger.time.monotonic",
Q:\finance-analyzer\tests\test_trigger_full.py:102:        "last_full_review_time": time.time() - last_full_review_offset,
Q:\finance-analyzer\tests\test_trigger_full.py:145:                "last_full_review_time": time.time() - 4 * 3600,  # exactly 4h ago
Q:\finance-analyzer\tests\test_trigger_full.py:159:                "last_full_review_time": time.time() - 7190,  # 10s short of 2h
Q:\finance-analyzer\tests\test_trigger_full.py:173:                "last_full_review_time": time.time() - 14400,  # exactly 4h
Q:\finance-analyzer\tests\test_trigger_full.py:187:                "last_full_review_time": time.time() - 14390,  # 10s short of 4h
Q:\finance-analyzer\tests\test_trigger_full.py:201:        state = {"last_full_review_time": time.time()}
Q:\finance-analyzer\tests\test_trigger_full.py:211:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:340:                "last_full_review_time": time.time() - 8000,  # >2h ago
Q:\finance-analyzer\tests\test_trigger_full.py:356:            "last_full_review_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:388:        before = time.time()
Q:\finance-analyzer\tests\test_trigger_full.py:390:        after = time.time()
Q:\finance-analyzer\tests\test_trigger_full.py:441:        before = time.time()
Q:\finance-analyzer\tests\test_trigger_full.py:443:        after = time.time()
Q:\finance-analyzer\tests\test_trigger_full.py:478:        state = {"last_trigger_time": time.time(), "last_checked_tx_count": {}}
Q:\finance-analyzer\tests\test_trigger_full.py:487:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:501:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:518:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:534:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:548:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:569:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:583:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:598:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_full.py:613:            "last_trigger_time": time.time(),
Q:\finance-analyzer\tests\test_trigger_ranging_dampening.py:62:    state["last_trigger_time"] = time.time() + 99999
Q:\finance-analyzer\tests\test_trigger_grace_period.py:145:        review_time = time.time() - 3600  # 1 hour ago
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:16:        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:21:            "portfolio.trigger.time.monotonic",
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:34:        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:39:            "portfolio.trigger.time.monotonic",
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:53:        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:59:        with patch("portfolio.trigger.time.monotonic", return_value=mono_after):
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:67:            "portfolio.trigger.time.monotonic", return_value=mono_after + 1.0
Q:\finance-analyzer\tests\test_trigger_sustained_monotonic.py:79:                "portfolio.trigger.time.monotonic", return_value=mono + i * 0.01
Q:\finance-analyzer\tests\test_unified_loop.py:140:    @patch("metals_loop.os.path.getmtime", return_value=time.time())
Q:\finance-analyzer\tests\test_unified_loop.py:168:    @patch("metals_loop.os.path.getmtime", return_value=time.time())
Q:\finance-analyzer\tests\test_unified_loop.py:367:        now = time.time()

2026-05-10T15:30:48.097022Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
305:def _flush_fingpt_phase(f_batch: list[tuple[str, str, dict]]) -> dict:
306:    """Execute Phase 3: load finance-llama-8b once, run all queued sentiment
307:    prompts, stash results in sentiment._pending_ab_entries.
308:
309:    Returns a metrics dict on EVERY code path (success, partial, exception):
310:
311:        {
312:          "queries": int,         # prompts sent to llama-server
313:          "received": int,        # non-None text completions back
314:          "parsed": int,          # parsed dicts (non-None) handed to _stash_fingpt_result
315:          "stashed_groups": int,  # distinct (ab_key, sub_key) groups stashed
316:          "exception": str|None,  # exception class name if the bare except fired
317:        }
318:
319:    Per-item failure (None text from the server) bubbles up to the A/B
320:    logger which writes a tagged fingpt:error entry instead of silently
321:    dropping the sample.
322:
323:    The whole phase is wrapped in try/except so fingpt errors never leak
324:    out into the main loop. Shadow signals must not crash anything above
325:    them. The metrics dict is the observability hook so a silent failure
326:    becomes loud — see callers in flush_llm_batch().
327:
328:    2026-05-03 (fix/fingpt-batch-observability): added metrics return +
329:    specific failure-mode warnings (server-returned-all-None, parser-
330:    failed-majority). Previously the phase returned None and a single
331:    bare `except` swallowed every error class with one generic warning,
332:    making silent regressions invisible until the A/B log was inspected
333:    by hand.
334:    """
335:    metrics = {
336:        "queries": 0,
337:        "received": 0,
338:        "parsed": 0,
339:        "stashed_groups": 0,
340:        "exception": None,
341:    }
342:    try:
343:        # fingpt_infer provides the prompt templates, stop tokens, and
344:        # response parsers that were originally used by the retired daemon.
345:        # Imported here (lazy) so a missing Q:\models path degrades gracefully
346:        # — if the import fails, fingpt just doesn't run this cycle.
347:        import platform
348:        import sys
349:        if platform.system() == "Windows":
350:            _models_dir = r"Q:\models"
351:        else:
352:            _models_dir = "/home/deck/models"
353:        if _models_dir not in sys.path:
354:            sys.path.insert(0, _models_dir)
355:        import fingpt_infer  # noqa: E402  (path injection above)
356:
357:        # Flatten the batch into per-prompt requests and keep a parallel meta
358:        # list so we can group results back by (ab_key, sub_key) afterward.
359:        prompts_and_params: list[dict] = []
360:        meta: list[tuple[str, str, dict, int]] = []  # (ab_key, sub_key, ctx, prompt_idx_within_call)
361:        for ab_key, sub_key, ctx in f_batch:
362:            mode = ctx.get("mode", "headlines")
363:            texts = ctx.get("texts") or []
364:            if mode == "cumulative":
365:                headlines_block = "\n".join(f"- {h}" for h in texts[:20])
366:                prompt = fingpt_infer.CUMULATIVE_PROMPT.format(
367:                    count=len(texts),
368:                    headlines_block=headlines_block,
369:                )
370:                prompts_and_params.append({
371:                    "prompt": prompt,
372:                    "n_predict": 30,
373:                    "temperature": 0.1,
374:                    # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
375:                    # Old stop ["\n", "<|eot_id|>"] was designed for the Llama-3
376:                    # chat-format prompt that wiroai-finance-llama-8b doesn't
377:                    # recognize. New CUMULATIVE_PROMPT is a plain-text one-shot
378:                    # template that ends each section with a blank line, so
379:                    # "\n\n" is the natural stop. The <|eot_id|> token was never
380:                    # emitted by this model (it's not chat-tuned) so removing
381:                    # it is a no-op.
382:                    "stop": ["\n\n"],
383:                })
384:                meta.append((ab_key, sub_key, ctx, 0))
385:            else:
386:                # Headlines mode: one prompt per headline. The daemon used
387:                # PROMPT_TEMPLATES[name] for the loaded model; llama_server
388:                # loads finance-llama-8b so we index into that entry directly.
389:                template = fingpt_infer.PROMPT_TEMPLATES.get(
390:                    "finance-llama-8b",
391:                    next(iter(fingpt_infer.PROMPT_TEMPLATES.values())),
392:                )
393:                for i, headline in enumerate(texts):
394:                    prompts_and_params.append({
395:                        "prompt": template.format(headline=headline),
396:                        "n_predict": 20,
397:                        "temperature": 0.1,
398:                        # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
399:                        # Same reason as the cumulative case above. The old
400:                        # stop ["\n", "<|eot_id|>", "[INST]"] cut the few-shot
401:                        # prompt apart at the first newline, which is exactly
402:                        # where the expected answer word appears — so even a
403:                        # correctly-answering model would have been silenced.
404:                        "stop": ["\n\n"],
405:                    })
406:                    meta.append((ab_key, sub_key, ctx, i))
407:
408:        metrics["queries"] = len(prompts_and_params)
409:        if not prompts_and_params:
410:            return metrics
411:
412:        # Single HTTP batch — llama_server holds its own file lock for the
413:        # duration so no other process can swap the model mid-phase.
414:        from portfolio.llama_server import query_llama_server_batch
415:        texts_out = query_llama_server_batch("finance-llama-8b", prompts_and_params)
416:        metrics["received"] = sum(1 for t in texts_out if t is not None)
417:
418:        # 2026-05-03: explicit warning when the server returned None for
419:        # every prompt — this is the "silent failure" mode (model swap
420:        # broke, llama-server crashed mid-batch, file-lock starvation).
421:        # Without this line operators see only the summary "F=0/N" and
422:        # have to dig through agent.log to figure out which layer broke.
423:        if metrics["queries"] > 0 and metrics["received"] == 0:
424:            logger.warning(
425:                "fingpt: server returned None for all %d prompts "
426:                "(likely llama_server unavailable or swap failed)",
427:                metrics["queries"],
428:            )
429:
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 54: character maps to <undefined>

 exited 1 in 614ms:
305:def _flush_fingpt_phase(f_batch: list[tuple[str, str, dict]]) -> dict:
306:    """Execute Phase 3: load finance-llama-8b once, run all queued sentiment
307:    prompts, stash results in sentiment._pending_ab_entries.
308:
309:    Returns a metrics dict on EVERY code path (success, partial, exception):
310:
311:        {
312:          "queries": int,         # prompts sent to llama-server
313:          "received": int,        # non-None text completions back
314:          "parsed": int,          # parsed dicts (non-None) handed to _stash_fingpt_result
315:          "stashed_groups": int,  # distinct (ab_key, sub_key) groups stashed
316:          "exception": str|None,  # exception class name if the bare except fired
317:        }
318:
319:    Per-item failure (None text from the server) bubbles up to the A/B
320:    logger which writes a tagged fingpt:error entry instead of silently
321:    dropping the sample.
322:
323:    The whole phase is wrapped in try/except so fingpt errors never leak
324:    out into the main loop. Shadow signals must not crash anything above
325:    them. The metrics dict is the observability hook so a silent failure
326:    becomes loud — see callers in flush_llm_batch().
327:
328:    2026-05-03 (fix/fingpt-batch-observability): added metrics return +
329:    specific failure-mode warnings (server-returned-all-None, parser-
330:    failed-majority). Previously the phase returned None and a single
331:    bare `except` swallowed every error class with one generic warning,
332:    making silent regressions invisible until the A/B log was inspected
333:    by hand.
334:    """
335:    metrics = {
336:        "queries": 0,
337:        "received": 0,
338:        "parsed": 0,
339:        "stashed_groups": 0,
340:        "exception": None,
341:    }
342:    try:
343:        # fingpt_infer provides the prompt templates, stop tokens, and
344:        # response parsers that were originally used by the retired daemon.
345:        # Imported here (lazy) so a missing Q:\models path degrades gracefully
346:        # — if the import fails, fingpt just doesn't run this cycle.
347:        import platform
348:        import sys
349:        if platform.system() == "Windows":
350:            _models_dir = r"Q:\models"
351:        else:
352:            _models_dir = "/home/deck/models"
353:        if _models_dir not in sys.path:
354:            sys.path.insert(0, _models_dir)
355:        import fingpt_infer  # noqa: E402  (path injection above)
356:
357:        # Flatten the batch into per-prompt requests and keep a parallel meta
358:        # list so we can group results back by (ab_key, sub_key) afterward.
359:        prompts_and_params: list[dict] = []
360:        meta: list[tuple[str, str, dict, int]] = []  # (ab_key, sub_key, ctx, prompt_idx_within_call)
361:        for ab_key, sub_key, ctx in f_batch:
362:            mode = ctx.get("mode", "headlines")
363:            texts = ctx.get("texts") or []
364:            if mode == "cumulative":
365:                headlines_block = "\n".join(f"- {h}" for h in texts[:20])
366:                prompt = fingpt_infer.CUMULATIVE_PROMPT.format(
367:                    count=len(texts),
368:                    headlines_block=headlines_block,
369:                )
370:                prompts_and_params.append({
371:                    "prompt": prompt,
372:                    "n_predict": 30,
373:                    "temperature": 0.1,
374:                    # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
375:                    # Old stop ["\n", "<|eot_id|>"] was designed for the Llama-3
376:                    # chat-format prompt that wiroai-finance-llama-8b doesn't
377:                    # recognize. New CUMULATIVE_PROMPT is a plain-text one-shot
378:                    # template that ends each section with a blank line, so
379:                    # "\n\n" is the natural stop. The <|eot_id|> token was never
380:                    # emitted by this model (it's not chat-tuned) so removing
381:                    # it is a no-op.
382:                    "stop": ["\n\n"],
383:                })
384:                meta.append((ab_key, sub_key, ctx, 0))
385:            else:
386:                # Headlines mode: one prompt per headline. The daemon used
387:                # PROMPT_TEMPLATES[name] for the loaded model; llama_server
388:                # loads finance-llama-8b so we index into that entry directly.
389:                template = fingpt_infer.PROMPT_TEMPLATES.get(
390:                    "finance-llama-8b",
391:                    next(iter(fingpt_infer.PROMPT_TEMPLATES.values())),
392:                )
393:                for i, headline in enumerate(texts):
394:                    prompts_and_params.append({
395:                        "prompt": template.format(headline=headline),
396:                        "n_predict": 20,
397:                        "temperature": 0.1,
398:                        # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
399:                        # Same reason as the cumulative case above. The old
400:                        # stop ["\n", "<|eot_id|>", "[INST]"] cut the few-shot
401:                        # prompt apart at the first newline, which is exactly
402:                        # where the expected answer word appears — so even a
403:                        # correctly-answering model would have been silenced.
404:                        "stop": ["\n\n"],
405:                    })
406:                    meta.append((ab_key, sub_key, ctx, i))
407:
408:        metrics["queries"] = len(prompts_and_params)
409:        if not prompts_and_params:
410:            return metrics
411:
412:        # Single HTTP batch — llama_server holds its own file lock for the
413:        # duration so no other process can swap the model mid-phase.
414:        from portfolio.llama_server import query_llama_server_batch
415:        texts_out = query_llama_server_batch("finance-llama-8b", prompts_and_params)
416:        metrics["received"] = sum(1 for t in texts_out if t is not None)
417:
418:        # 2026-05-03: explicit warning when the server returned None for
419:        # every prompt — this is the "silent failure" mode (model swap
420:        # broke, llama-server crashed mid-batch, file-lock starvation).
421:        # Without this line operators see only the summary "F=0/N" and
422:        # have to dig through agent.log to figure out which layer broke.
423:        if metrics["queries"] > 0 and metrics["received"] == 0:
424:            logger.warning(
425:                "fingpt: server returned None for all %d prompts "
426:                "(likely llama_server unavailable or swap failed)",
427:                metrics["queries"],
428:            )
429:
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 54: character maps to <undefined>

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\tests\\test_shared_state.py'; "'$ranges=@(@(140,210),@(216,260)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 536ms:
140:        """At exactly TTL seconds the condition `now - time < ttl` is False."""
141:        mock_time.time.return_value = 1000.0
142:        _cached("k1", 60, lambda: "old")
143:
144:        mock_time.time.return_value = 1060.0  # exactly 60s
145:        result = _cached("k1", 60, lambda: "new")
146:        assert result == "new"
147:
148:
149:# ===========================================================================
150:# _cached() — error handling / stale data
151:# ===========================================================================
152:
153:
154:class TestCachedStaleDataOnError:
155:    """On error, _cached() returns stale data if within MAX_STALE_FACTOR * TTL."""
156:
157:    @patch("portfolio.shared_state.time")
158:    def test_returns_stale_on_error(self, mock_time):
159:        mock_time.time.return_value = 1000.0
160:        _cached("k1", 60, lambda: "good")
161:
162:        # TTL expired, func raises
163:        mock_time.time.return_value = 1100.0  # 100s old, max stale = 60*3 = 180s
164:        result = _cached("k1", 60, self._failing_func)
165:        assert result == "good"
166:
167:    @patch("portfolio.shared_state.time")
168:    def test_returns_stale_near_max_boundary(self, mock_time):
169:        """Just under MAX_STALE_FACTOR * TTL should still return stale data."""
170:        mock_time.time.return_value = 1000.0
171:        _cached("k1", 60, lambda: "ok")
172:
173:        # age = 179s, max_stale = 180s  →  should still return stale
174:        mock_time.time.return_value = 1179.0
175:        result = _cached("k1", 60, self._failing_func)
176:        assert result == "ok"
177:
178:    @patch("portfolio.shared_state.time")
179:    def test_returns_none_when_stale_exceeds_max(self, mock_time):
180:        """Beyond MAX_STALE_FACTOR * TTL, returns None."""
181:        mock_time.time.return_value = 1000.0
182:        _cached("k1", 60, lambda: "old")
183:
184:        # age = 181s, max_stale = 180s  →  None
185:        mock_time.time.return_value = 1181.0
186:        result = _cached("k1", 60, self._failing_func)
187:        assert result is None
188:
189:    @patch("portfolio.shared_state.time")
190:    def test_returns_none_at_exact_max_stale(self, mock_time):
191:        """At exactly MAX_STALE_FACTOR * TTL, condition `age > max_stale` is False (<=)."""
192:        mock_time.time.return_value = 1000.0
193:        _cached("k1", 60, lambda: "val")
194:
195:        # age = 180s, max_stale = 180s  →  age > max_stale is False → returns stale
196:        mock_time.time.return_value = 1180.0
197:        result = _cached("k1", 60, self._failing_func)
198:        assert result == "val"
199:
200:    def test_returns_none_when_no_cached_data_on_error(self):
201:        """First call fails with no cache → returns None."""
202:        result = _cached("k1", 60, self._failing_func)
203:        assert result is None
204:
205:    @staticmethod
206:    def _failing_func():
207:        raise RuntimeError("data source down")
208:
209:
210:# ===========================================================================
---
216:    """On error, _cached() adjusts the cache timestamp to enforce a retry cooldown."""
217:
218:    @patch("portfolio.shared_state.time")
219:    def test_cooldown_prevents_immediate_retry(self, mock_time):
220:        """After an error, the next call within 60s should still return stale data
221:        without calling func again (because the adjusted timestamp keeps it 'fresh')."""
222:        mock_time.time.return_value = 1000.0
223:        _cached("k1", 60, lambda: "original")
224:
225:        # TTL expired; func fails → stale data returned, timestamp adjusted
226:        mock_time.time.return_value = 1070.0
227:        call_count = 0
228:
229:        def failing():
230:            nonlocal call_count
231:            call_count += 1
232:            raise RuntimeError("fail")
233:
234:        result = _cached("k1", 60, failing)
235:        assert result == "original"
236:        assert call_count == 1
237:
238:        # Adjusted time = now - ttl + RETRY_COOLDOWN = 1070 - 60 + 60 = 1070
239:        # So at 1071 (1s later), age = 1071 - 1070 = 1s < 60s TTL → cache hit
240:        mock_time.time.return_value = 1071.0
241:        fn2 = MagicMock(return_value="should_not_call")
242:        result = _cached("k1", 60, fn2)
243:        assert result == "original"
244:        fn2.assert_not_called()
245:
246:    @patch("portfolio.shared_state.time")
247:    def test_cooldown_expires_after_retry_delay(self, mock_time):
248:        """After the 60s retry cooldown, func is called again."""
249:        mock_time.time.return_value = 1000.0
250:        _cached("k1", 60, lambda: "original")
251:
252:        # Error at t=1070 → adjusted time = 1070 - 60 + 60 = 1070
253:        mock_time.time.return_value = 1070.0
254:        _cached("k1", 60, self._raise)
255:
256:        # At t=1131 → age = 1131 - 1070 = 61s > 60s TTL → retry
257:        mock_time.time.return_value = 1131.0
258:        fn = MagicMock(return_value="recovered")
259:        result = _cached("k1", 60, fn)
260:        assert result == "recovered"
---

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"None results|returns None|cache None|transient\" Q:\\finance-analyzer\\tests\\test_shared_state.py Q:\\finance-analyzer\\tests\\test_shared_state_cache.py" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 530ms:
Q:\finance-analyzer\tests\test_shared_state.py:57:    """_cached() should NOT cache None results (transient API failure)."""
Q:\finance-analyzer\tests\test_shared_state.py:66:        """After func() returns None, the next call should retry func()."""
Q:\finance-analyzer\tests\test_shared_state.py:83:        """If stale data exists and func() returns None, stale data should
Q:\finance-analyzer\tests\test_shared_state.py:88:        # Now func returns None — stale data should survive
Q:\finance-analyzer\tests\test_shared_state.py:180:        """Beyond MAX_STALE_FACTOR * TTL, returns None."""
Q:\finance-analyzer\tests\test_shared_state.py:201:        """First call fails with no cache → returns None."""

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\tests\\test_shared_state.py'; "'$start=50; $end=95; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 584ms:
50:    def test_passes_args_to_func(self):
51:        fn = MagicMock(return_value="ok")
52:        _cached("k1", 60, fn, "a", "b")
53:        fn.assert_called_once_with("a", "b")
54:
55:
56:class TestCachedNoneNotCached:
57:    """_cached() should NOT cache None results (transient API failure)."""
58:
59:    def test_none_result_not_stored_in_cache(self):
60:        """func() returning None should not write to cache."""
61:        result = _cached("k_none", 60, lambda: None)
62:        assert result is None
63:        assert "k_none" not in shared_state._tool_cache
64:
65:    def test_none_result_retried_next_call(self):
66:        """After func() returns None, the next call should retry func()."""
67:        call_count = 0
68:
69:        def flaky():
70:            nonlocal call_count
71:            call_count += 1
72:            if call_count == 1:
73:                return None  # first call fails
74:            return "recovered"
75:
76:        result1 = _cached("k_flaky", 60, flaky)
77:        assert result1 is None
78:        result2 = _cached("k_flaky", 60, flaky)
79:        assert result2 == "recovered"
80:        assert call_count == 2
81:
82:    def test_stale_data_preserved_when_func_returns_none(self):
83:        """If stale data exists and func() returns None, stale data should
84:        be preserved (not overwritten) so it can still be served."""
85:        _cached("k_stale", 60, lambda: "original")
86:        # Force TTL expiry by manipulating cache time
87:        shared_state._tool_cache["k_stale"]["time"] = 0.0
88:        # Now func returns None — stale data should survive
89:        _cached("k_stale", 60, lambda: None)
90:        assert shared_state._tool_cache["k_stale"]["data"] == "original"
91:
92:
93:class TestCachedWithinTTL:
94:    """_cached() returns cached data when within TTL, without calling func again."""
95:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\tickers.py'; "'$start=18; $end=36; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}' in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 517ms:
18:    "XAU-USD": {"binance_fapi": "XAUUSDT"},
19:    "XAG-USD": {"binance_fapi": "XAGUSDT"},
20:    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
21:    "MSTR": {"alpaca": "MSTR"},
22:    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
23:    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
24:    #   Reduces main loop load to stay under 60s cadence. Cycle p50 was 143s with
25:    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
26:    #   because data/metals_loop.py uses it for BTC NAV-premium tracking.
27:}
28:
29:# ── Asset-class subsets ───────────────────────────────────────────────────
30:
31:CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
32:METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
33:STOCK_SYMBOLS = {"MSTR"}
34:
35:# All known tickers (union of all subsets)
36:ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS

codex
I found one potential issue Claude missed in `onchain_data`: the no-token fallback still subtracts a raw `ts` string in `_load_onchain_cache()`. I’m reproducing that path directly before I list it as new.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, r'Q:\\finance-analyzer')
import portfolio.onchain_data as od
from unittest.mock import patch

with tempfile.TemporaryDirectory() as td:
    cache_file = Path(td)/'onchain_cache.json'
    cache_file.write_text(json.dumps({'mvrv': 1.5, 'ts': '2026-04-09T12:00:00Z'}), encoding='utf-8')
    with patch.object(od, 'CACHE_FILE', cache_file), \\
         patch.object(od, '_load_config_token', lambda: None), \\
         patch('portfolio.onchain_data.time.time', return_value=0):
        # direct helper
        try:
            helper = od._load_onchain_cache(max_age_seconds=999999999)
            print('helper', helper)
        except Exception as e:
            print('helper_exc', type(e).__name__, e)
        try:
            result = od.get_onchain_data()
            print('result', result)
        except Exception as e:
            print('result_exc', type(e).__name__, e)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys
sys.path.insert(0, r'Q:\\finance-analyzer')
import portfolio.shared_state as ss
from unittest.mock import patch

call_count = 0
def flaky():
    global call_count
    call_count += 1
    if call_count == 1:
        return 'ok'
    return None

with patch('portfolio.shared_state.time') as mock_time:
    mock_time.time.return_value = 1000.0
    print(ss._cached('k',60,lambda:'ok'))
    mock_time.time.return_value = 1070.0
    print(ss._cached('k',60,lambda:None))
    mock_time.time.return_value = 1071.0
    called = {'n':0}
    def f():
        called['n'] += 1
        return 'new'
    print(ss._cached('k',60,f))
    print('calls', called['n'])
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys
sys.path.insert(0, r'Q:\\finance-analyzer')
import pandas as pd
print(pd.to_datetime(['2026-01-01T12:00:00Z']).dtype)
print(pd.to_datetime([1710000000000], unit='ms').dtype)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 645ms:
ok
None
new
calls 1

2026-05-10T15:31:34.826348Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1047, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpybxkjn9k\\onchain_cache.json'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen os>", line 366, in walk
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpybxkjn9k'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 950, in __exit__
    self.cleanup()
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 954, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 934, in _rmtree
    _shutil.rmtree(name, onexc=onexc)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 781, in rmtree
    return _rmtree_unsafe(path, onexc)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 623, in _rmtree_unsafe
    for dirpath, dirnames, filenames in results:
                                        ^^^^^^^
  File "<frozen os>", line 369, in walk
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 621, in onerror
    onexc(os.scandir, err.filename, err)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 906, in onexc
    _resetperms(path)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 286, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 277, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpybxkjn9k'

 exited 1 in 718ms:
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1047, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpybxkjn9k\\onchain_cache.json'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen os>", line 366, in walk
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpybxkjn9k'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 950, in __exit__
    self.cleanup()
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 954, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 934, in _rmtree
    _shutil.rmtree(name, onexc=onexc)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 781, in rmtree
    return _rmtree_unsafe(path, onexc)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 623, in _rmtree_unsafe
    for dirpath, dirnames, filenames in results:
                                        ^^^^^^^
  File "<frozen os>", line 369, in walk
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\shutil.py", line 621, in onerror
    onexc(os.scandir, err.filename, err)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 906, in onexc
    _resetperms(path)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 286, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\Users\Herc2\AppData\Local\Programs\Python\Python312\Lib\tempfile.py", line 277, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Herc2\\AppData\\Local\\Temp\\tmpybxkjn9k'

 succeeded in 1136ms:
datetime64[us, UTC]
datetime64[ms]

codex
The cache experiment hit a Windows temp-permission quirk. I’m rerunning it inside the writable worktree so I can confirm whether the no-token ISO-cache path is genuinely broken.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, r'Q:\\finance-analyzer')
import portfolio.onchain_data as od
from unittest.mock import patch

base = Path(r'Q:\\finance-analyzer\\.worktrees\\adv-data-external\\.tmp_onchain_check')
if base.exists():
    shutil.rmtree(base)
base.mkdir()
cache_file = base/'onchain_cache.json'
cache_file.write_text(json.dumps({'mvrv': 1.5, 'ts': '2026-04-09T12:00:00Z'}), encoding='utf-8')
with patch.object(od, 'CACHE_FILE', cache_file), \\
     patch.object(od, '_load_config_token', lambda: None), \\
     patch('portfolio.onchain_data.time.time', return_value=0):
    helper = od._load_onchain_cache(max_age_seconds=999999999)
    print('helper', helper)
    result = od.get_onchain_data()
    print('result', result)
shutil.rmtree(base)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 678ms:
helper None
result None
On-chain cache load failed: unsupported operand type(s) for -: 'int' and 'str'
Traceback (most recent call last):
  File "Q:\finance-analyzer\portfolio\onchain_data.py", line 112, in _load_onchain_cache
    if time.time() - ts > max_age_seconds:
       ~~~~~~~~~~~~^~~~
TypeError: unsupported operand type(s) for -: 'int' and 'str'
On-chain cache load failed: unsupported operand type(s) for -: 'int' and 'str'
Traceback (most recent call last):
  File "Q:\finance-analyzer\portfolio\onchain_data.py", line 112, in _load_onchain_cache
    if time.time() - ts > max_age_seconds:
       ~~~~~~~~~~~~^~~~
TypeError: unsupported operand type(s) for -: 'int' and 'str'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$c=Get-Content '"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; "'$ranges=@(@(1756,1770),@(2138,2160)); foreach($r in $ranges){ $start=$r[0]; $end=$r[1]; for($i=$start;$i -le $end;$i++){ if($i -le $c.Length){ '"'{0}:{1}' -f "'$i,$c[$i-1] }}; '"'---' }" in Q:\finance-analyzer\.worktrees\adv-data-external
 succeeded in 647ms:
1756:    Weight per signal = accuracy_weight * ic_mult * regime_mult * normalized_weight
1757:                        * horizon_mult * activity_cap
1758:    where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
1759:    Rare, balanced signals get more weight; noisy/biased signals get less.
1760:
1761:    Signals below the accuracy gate (with sufficient samples) are force-skipped —
1762:    they are noise, not useful contrarian indicators.
1763:
1764:    Regime gating: signals in REGIME_GATED_SIGNALS for the current regime are
1765:    forced to HOLD before vote processing — they produce negative alpha.
1766:
1767:    Correlation deduplication: within defined correlation groups, only the
1768:    highest-accuracy signal gets full weight. Others get 0.3x penalty.
1769:
1770:    Horizon-specific weights: signals with divergent accuracy across horizons
---
2138:                gated_signals.append(signal_name)
2139:                continue
2140:        else:
2141:            _rescued = False
2142:        # Directional accuracy gate: gate individual BUY/SELL direction when
2143:        # direction-specific accuracy is very poor, even if overall accuracy passes.
2144:        # E.g., qwen3 overall=59.8% passes, but BUY=30.0% → gate BUY only.
2145:        if vote == "BUY":
2146:            dir_acc = stats.get("buy_accuracy", acc)
2147:            dir_n = stats.get("total_buy", 0)
2148:        else:
2149:            dir_acc = stats.get("sell_accuracy", acc)
2150:            dir_n = stats.get("total_sell", 0)
2151:        if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
2152:            gated_signals.append(f"{signal_name}_{vote}")
2153:            continue
2154:        # BUG-182: Use direction-specific accuracy as weight when available.
2155:        # A signal with overall 60% accuracy may be 30% for BUY and 75% for SELL.
2156:        # Using overall accuracy over-weights the weak direction.
2157:        _DIR_WEIGHT_MIN_SAMPLES = 20
2158:        if vote == "BUY" and stats.get("total_buy", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
2159:            weight = stats.get("buy_accuracy", acc)  # BUG-185: .get() for cache safety
2160:        elif vote == "SELL" and stats.get("total_sell", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
---

codex
## Verdicts

- [P1] FX sanity check logs an out-of-range USD/SEK rate but still returns it — portfolio/fx_rates.py:47
  Verdict: FALSE-POSITIVE
  Reason: The out-of-range branch only logs at line 48; the only `return rate` is inside the `else` at 49-53, so bad values fall through to stale/hardcoded fallback at 56-71.

- [P1] Stock-headline dedupe keys only on title and suppresses dissemination — portfolio/sentiment.py:233
  Verdict: PARTIAL
  Reason: Lines 233-237 do dedupe Yahoo fallback additions on `title.lower()` only, but only during the NewsAPI→Yahoo merge; `news_keywords.dissemination_score()` still counts unique surviving `source` values at `portfolio/news_keywords.py:197-213`.
  Adjustment: Limited Yahoo-merge scope, not a system-wide Reuters/AP/Bloomberg collapse; severity is overstated.

- [P1] Binance and Alpaca kline timestamps are both timezone-naive — portfolio/data_collector.py:96
  Verdict: PARTIAL
  Reason: Binance is naive at line 96 (`pd.to_datetime(..., unit="ms")`), but Alpaca line 157 parses timestamp strings and preserves timezone offsets when present; `pd.to_datetime("...Z")` yields UTC-aware values.
  Adjustment: Only the Binance path is demonstrably naive here; the claimed cross-source DST corruption path is speculative.

- [P1] `_cached()` not caching `None` causes stale fallback oscillation — portfolio/shared_state.py:95
  Verdict: PARTIAL
  Reason: `_cached()` intentionally skips caching `None` at 95-99 and returns `None` at 103; the stale-data retry cooldown exists only on exceptions at 110-125, so the claimed “next call returns stale without re-fetching” path is backwards.
  Adjustment: The real asymmetry is over-retrying after `None`, not silent stale reuse.

- [P1] Alpha Vantage earnings fetches bypass the daily budget counter — portfolio/earnings_calendar.py:49
  Verdict: PARTIAL
  Reason: The bypass is real and explicitly documented at 49-52, but the current repo only has `STOCK_SYMBOLS = {"MSTR"}` at `portfolio/tickers.py:33`, so this is one untracked AV call per 24h, not a 100-ticker quota explosion.
  Adjustment: Real issue, wildly inflated impact.

- [P1] Headline-filter fallback ignores per-source noise profiles — portfolio/sentiment.py:804
  Verdict: PARTIAL
  Reason: `_filter_relevant_headlines()` uses one global `fallback_n=3` at 682-723, but only after relevance filtering drops everything and after a source-aware credible-outlet escape hatch at 697-709.
  Adjustment: This is a tuning limitation in the fallback count, not a demonstrated P1 bug at the call site.

- [P1] Alpha Vantage budget reset runs outside the lock at midnight — portfolio/alpha_vantage.py:281
  Verdict: FALSE-POSITIVE
  Reason: `_check_budget()` resets `_daily_budget_used` under `_cache_lock` at 164-167, and increments are also under that lock at 279-281.

- [P1] FinGPT enqueue failure silently drops the A/B entry — portfolio/sentiment.py:876
  Verdict: PARTIAL
  Reason: An enqueue exception is only DEBUG-logged at 889-890, but `flush_ab_log()` still writes the row whenever other shadows exist at 472-485, so this loses one shadow model, not the entire A/B entry.
  Adjustment: Shadow-only observability problem, not a P1 correctness failure.

- [P1] Raw OHLCV is discarded when indicators are `None` — portfolio/data_collector.py:296
  Verdict: FALSE-POSITIVE
  Reason: Lines 296-299 intentionally skip that timeframe, `tests/test_data_collector.py:637-645` asserts that contract, and `portfolio/main.py:491-493` refetches the `"Now"` candles if it still needs raw data.

- [P1] Open-interest cache TTL of 300s is too long for momentum signals — portfolio/futures_data.py:22
  Verdict: FALSE-POSITIVE
  Reason: `_OI_TTL = 300` is a policy choice, and the actual futures signal consumes batched `oi_history`/ratio data in `portfolio/signals/futures_flow.py:262-287`; Claude's “1-second updates flip a 3h signal” claim is unsupported by the code.

- [P1] ISO-string on-chain cache timestamps still coerce to `0.0` and burn budget every restart — portfolio/onchain_data.py:112
  Verdict: FALSE-POSITIVE
  Reason: `_coerce_epoch()` already parses numeric strings and ISO timestamps at 46-55 and only returns `0.0` for unparseable garbage at 62-67, so the cited ISO-string restart-burn bug is not present in the tokened fetch path Claude described.

- [P2] EURUSD-based intraday DXY synth exposes a meaningless `value` — portfolio/macro_context.py:141
  Verdict: FALSE-POSITIVE
  Reason: The synth comment is explicit, but `portfolio/signals/dxy_cross_asset.py:71-99` makes decisions from `change_1h_pct` and only passes `value` through as an indicator alongside `source`; no cited downstream consumer treats that synth `value` as a real DXY level.

- [P2] Trading-Hero-LLM permabull bias is uncorrected — portfolio/sentiment.py:757
  Verdict: FALSE-POSITIVE
  Reason: The code explicitly acknowledges that bias at 757-759, and `portfolio/signal_engine.py:2142-2153` applies a direction-specific accuracy gate that can suppress bad BUY/SELL directions; lack of an extra manual de-bias factor is model tuning, not a bug.

## New findings (you, not Claude)

- [P3] No-token on-chain stale-cache fallback still breaks on old ISO-string `ts` — portfolio/onchain_data.py:111
  `get_onchain_data()` normalizes `persistent["ts"]` with `_coerce_epoch()` at 259-261 for the in-memory seeding path, but the no-token fallback at 277-281 calls `_load_onchain_cache()`, which still does raw `time.time() - ts` at 111-112. I reproduced that path with an ISO-string cache file; it logged `TypeError` and returned `None` instead of serving cached on-chain data.

## Summary
- Confirmed: 0
- Partial: 6
- False-positive: 7
- New from you: 1
tokens used
190,315
## Verdicts

- [P1] FX sanity check logs an out-of-range USD/SEK rate but still returns it — portfolio/fx_rates.py:47
  Verdict: FALSE-POSITIVE
  Reason: The out-of-range branch only logs at line 48; the only `return rate` is inside the `else` at 49-53, so bad values fall through to stale/hardcoded fallback at 56-71.

- [P1] Stock-headline dedupe keys only on title and suppresses dissemination — portfolio/sentiment.py:233
  Verdict: PARTIAL
  Reason: Lines 233-237 do dedupe Yahoo fallback additions on `title.lower()` only, but only during the NewsAPI→Yahoo merge; `news_keywords.dissemination_score()` still counts unique surviving `source` values at `portfolio/news_keywords.py:197-213`.
  Adjustment: Limited Yahoo-merge scope, not a system-wide Reuters/AP/Bloomberg collapse; severity is overstated.

- [P1] Binance and Alpaca kline timestamps are both timezone-naive — portfolio/data_collector.py:96
  Verdict: PARTIAL
  Reason: Binance is naive at line 96 (`pd.to_datetime(..., unit="ms")`), but Alpaca line 157 parses timestamp strings and preserves timezone offsets when present; `pd.to_datetime("...Z")` yields UTC-aware values.
  Adjustment: Only the Binance path is demonstrably naive here; the claimed cross-source DST corruption path is speculative.

- [P1] `_cached()` not caching `None` causes stale fallback oscillation — portfolio/shared_state.py:95
  Verdict: PARTIAL
  Reason: `_cached()` intentionally skips caching `None` at 95-99 and returns `None` at 103; the stale-data retry cooldown exists only on exceptions at 110-125, so the claimed “next call returns stale without re-fetching” path is backwards.
  Adjustment: The real asymmetry is over-retrying after `None`, not silent stale reuse.

- [P1] Alpha Vantage earnings fetches bypass the daily budget counter — portfolio/earnings_calendar.py:49
  Verdict: PARTIAL
  Reason: The bypass is real and explicitly documented at 49-52, but the current repo only has `STOCK_SYMBOLS = {"MSTR"}` at `portfolio/tickers.py:33`, so this is one untracked AV call per 24h, not a 100-ticker quota explosion.
  Adjustment: Real issue, wildly inflated impact.

- [P1] Headline-filter fallback ignores per-source noise profiles — portfolio/sentiment.py:804
  Verdict: PARTIAL
  Reason: `_filter_relevant_headlines()` uses one global `fallback_n=3` at 682-723, but only after relevance filtering drops everything and after a source-aware credible-outlet escape hatch at 697-709.
  Adjustment: This is a tuning limitation in the fallback count, not a demonstrated P1 bug at the call site.

- [P1] Alpha Vantage budget reset runs outside the lock at midnight — portfolio/alpha_vantage.py:281
  Verdict: FALSE-POSITIVE
  Reason: `_check_budget()` resets `_daily_budget_used` under `_cache_lock` at 164-167, and increments are also under that lock at 279-281.

- [P1] FinGPT enqueue failure silently drops the A/B entry — portfolio/sentiment.py:876
  Verdict: PARTIAL
  Reason: An enqueue exception is only DEBUG-logged at 889-890, but `flush_ab_log()` still writes the row whenever other shadows exist at 472-485, so this loses one shadow model, not the entire A/B entry.
  Adjustment: Shadow-only observability problem, not a P1 correctness failure.

- [P1] Raw OHLCV is discarded when indicators are `None` — portfolio/data_collector.py:296
  Verdict: FALSE-POSITIVE
  Reason: Lines 296-299 intentionally skip that timeframe, `tests/test_data_collector.py:637-645` asserts that contract, and `portfolio/main.py:491-493` refetches the `"Now"` candles if it still needs raw data.

- [P1] Open-interest cache TTL of 300s is too long for momentum signals — portfolio/futures_data.py:22
  Verdict: FALSE-POSITIVE
  Reason: `_OI_TTL = 300` is a policy choice, and the actual futures signal consumes batched `oi_history`/ratio data in `portfolio/signals/futures_flow.py:262-287`; Claude's “1-second updates flip a 3h signal” claim is unsupported by the code.

- [P1] ISO-string on-chain cache timestamps still coerce to `0.0` and burn budget every restart — portfolio/onchain_data.py:112
  Verdict: FALSE-POSITIVE
  Reason: `_coerce_epoch()` already parses numeric strings and ISO timestamps at 46-55 and only returns `0.0` for unparseable garbage at 62-67, so the cited ISO-string restart-burn bug is not present in the tokened fetch path Claude described.

- [P2] EURUSD-based intraday DXY synth exposes a meaningless `value` — portfolio/macro_context.py:141
  Verdict: FALSE-POSITIVE
  Reason: The synth comment is explicit, but `portfolio/signals/dxy_cross_asset.py:71-99` makes decisions from `change_1h_pct` and only passes `value` through as an indicator alongside `source`; no cited downstream consumer treats that synth `value` as a real DXY level.

- [P2] Trading-Hero-LLM permabull bias is uncorrected — portfolio/sentiment.py:757
  Verdict: FALSE-POSITIVE
  Reason: The code explicitly acknowledges that bias at 757-759, and `portfolio/signal_engine.py:2142-2153` applies a direction-specific accuracy gate that can suppress bad BUY/SELL directions; lack of an extra manual de-bias factor is model tuning, not a bug.

## New findings (you, not Claude)

- [P3] No-token on-chain stale-cache fallback still breaks on old ISO-string `ts` — portfolio/onchain_data.py:111
  `get_onchain_data()` normalizes `persistent["ts"]` with `_coerce_epoch()` at 259-261 for the in-memory seeding path, but the no-token fallback at 277-281 calls `_load_onchain_cache()`, which still does raw `time.time() - ts` at 111-112. I reproduced that path with an ISO-string cache file; it logged `TypeError` and returned `None` instead of serving cached on-chain data.

## Summary
- Confirmed: 0
- Partial: 6
- False-positive: 7
- New from you: 1
