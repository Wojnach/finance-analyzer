# Adversarial review — data-external (2026-05-27)

## Summary

The subsystem is generally hardened — most modules use `_cached()`, circuit
breakers, atomic I/O, and `.get()`-based schema-defensive parsing. The
HTTP retry layer respects 429 with `Retry-After` and adds jitter, and
NewsAPI/BGeometrics/AlphaVantage all have day-budget plumbing.

Real concrete risks found are concentrated in:

1. **CFTC SOCRATA fetcher uses raw `requests` with no retry / no
   rate-limit / no budget cap**, so a transient outage in
   `metals_precompute` / `oil_precompute` cascades into a precompute
   exception that resets the per-source refresh state (P1).
2. **Crypto/MSTR precomputers bypass `price_source` and the shared
   `_binance_limiter`** by calling `requests.get` directly against
   `api.binance.com` and `fapi.binance.com` — these calls do not respect
   the loop's global Binance weight budget, so when crypto_precompute and
   the main loop both fire they can collectively exceed Binance limits
   (P1).
3. **`fetch_news_api`-style behaviour where every retry burns quota
   silently** because `fetch_with_retry` retries inside the request, but
   NewsAPI's quota is tracked at the wrapper level — a 429 retry storm
   counts as 1 call from our side, but the upstream rate-limit hit
   stays. Not the most pressing, but worth observability (P2).
4. **`alternative.me` / yfinance / CFTC schema drift not all surfaces
   the same way** — many sub-modules treat None as HOLD which is the
   right default, but `_aggregate_sentiments`, `fetch_usd_sek`, and
   `_compute_market_health` all have subtle behaviour gaps catalogued
   below.
5. **Hardcoded FOMC/CPI/NFP dates run out at end of 2027** (today is
   2026-05-27 → 19 months runway). Already documented in the source but
   easy to forget (P3).
6. **MSTR holding count, debt, and shares-outstanding are hardcoded**
   constants in `mstr_precompute` that go stale immediately after the
   next 8-K filing — directly distorts the NAV-premium signal used by
   Layer 2 (P1).
7. **DXY EURUSD synth uses an arbitrary constant** which is documented
   but easy to misread (P2).

No hardcoded API keys, no `10m` Binance interval bugs, no Adj-Close
look-ahead on intraday paths, and no `claude -p --bare` silent-success
patterns in scope. Good.

---

### [P1] CFTC SOCRATA fetcher has no retry, no rate-limit guard, no circuit breaker — single transient failure aborts entire precompute
**File:** `portfolio/metals_precompute.py:407` and `portfolio/oil_precompute.py:510`
**Issue:** Both `_fetch_cftc_cot` and `_fetch_cftc_cot_crude` use raw
`requests.get(url, timeout=15)` directly, then call `.raise_for_status()`.
There is no retry, no jitter, no circuit breaker, no rate-limit budget,
and no shared limiter. The same is true for the disaggregated COT
follow-up URL on lines 458 and 579.

A single 503 (SOCRATA returns these on burst load) raises in
`_safe_fetch`, which logs a warning and returns None — fine in
isolation. But the precompute orchestrator's outer `try/except` in
`precompute()` (called from `maybe_precompute_metals/oil`) catches a
DIFFERENT exception path: the FRED fetcher (line 498) and the
disaggregated COT block (line 579) ALSO use raw requests, and a failure
there propagates up out of `_fetch_market_data` into the outer
try/except in `maybe_precompute_metals` (line 79). The whole
precompute is then marked `status: error` and the next-cycle
`refresh_state` is left untouched for the half-fetched sources, so the
next 2h cycle re-fetches everything from scratch (no partial
incremental write).

**Impact:** A single SOCRATA blip burns all 6 metals refresh slots and
the COT history skip. Worse, oil_precompute also writes
`refresh_state[src] = {**old, "last_error_ts": now}` per-source on
failure but never records the next-success window, so once SOCRATA
recovers we still wait the full 7d interval (`_REFRESH_INTERVALS["cot"]
= 7 * 24 * 3600`) from the last `ts`, not from the success retry.

**Fix:** Route all SOCRATA + FRED calls through `fetch_json` (which has
the retry+backoff+429 handling already), wrap with a `CircuitBreaker`
like `binance_spot_cb`, and add a `_socrata_limiter` rate limiter to
`shared_state.py`. The disaggregated COT URL specifically should share
budget with the legacy URL.

**Confidence:** 90

---

### [P1] Crypto/MSTR precomputers bypass `price_source` + shared Binance rate limiter
**File:** `portfolio/crypto_precompute.py:159-200` and `portfolio/mstr_precompute.py:205-214`
**Issue:** `_fetch_market_data` in both modules does
`requests.get("https://api.binance.com/api/v3/ticker/24hr", ...)` and
`requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", ...)`
directly, without:
  - going through `data_collector.binance_klines` / `price_source`,
  - waiting on `shared_state._binance_limiter`,
  - tripping `binance_spot_cb` / `binance_fapi_cb` on failure.

Meanwhile, the main loop, crypto_loop, and metals_loop ALL hit Binance
through the limiter. Three precomputers + three loops means six
concurrent Binance callers, of which two ignore the shared budget
entirely. `CLAUDE.md` and `memory/claude_gate.md` both state Binance is
the project's "live-prices-first" canonical source — running a hidden
quota leak through it is exactly the failure mode the limiter was
introduced to prevent.

Additionally, both precomputers swallow only `status_code != 200`
without distinguishing 429 from 5xx: a 429 returns silently as
"no data" and the value lands as `None` in the output, which
downstream `_build_context` then publishes as `btc_price_usd: None`,
silently disabling the BTC component of `crypto_deep_context.json`.

**Impact:** Burns Binance weight budget invisibly; on quota exhaustion,
the entire BTC/ETH/MSTR deep-context section disappears for 4h with no
log trace tying it to a 429.

**Fix:** Replace direct `requests.get` with `price_source.fetch_klines`
(routed via `_binance_limiter` + circuit breaker) or
`futures_data.get_open_interest`/`funding_rate.get_funding_rate`
(both already go through the limiter and circuit breaker). For 24hr
ticker, add `data_collector.binance_24hr_ticker(symbol)` and reuse.

**Confidence:** 92

---

### [P1] MSTR BTC holdings / debt / shares-outstanding are hardcoded constants — directly distort NAV-premium signal
**File:** `portfolio/mstr_precompute.py:35-37`
**Issue:** `_DEFAULT_BTC_HOLDINGS = 471_107` (annotated "2026-04
estimate; update after next 8-K"), `_DEFAULT_DEBT_USD = 8_500_000_000`,
`_DEFAULT_SHARES_OUTSTANDING = 287_000_000`. These flow directly into
`_compute_nav_premium` (line 85) and the published
`mstr_deep_context.json` premium field.

MSTR's BTC stack grows ~monthly via ATM equity issuance + convertibles.
By session date (2026-05-27) the holdings figure is at minimum 1 month
stale; debt and share count both move every issuance. NAV premium is
the *main* signal Layer 2 uses to decide MSTR vs spot-BTC
(`memory/mstr_loop_notes.md`), so a stale numerator means we
systematically misjudge premium tightness.

Worse, there's no override path from config: `_fetch_market_data` reads
`config.get("mstr", {}).get("btc_holdings", _DEFAULT_BTC_HOLDINGS)`?
No — it doesn't even read that. Holdings are hardcoded as the default
in `out`, never overridden by config in the current code (line 135).

**Impact:** Premium signal drifts ~4-8% per month off-truth as MSTR
buys more BTC. After 6 months untouched, NAV premium computed here
could be off by a full 30%+, which on a leverage-sensitive instrument
makes the signal worse than useless.

**Fix:** Either (a) pull the live SEC 8-K filings on a daily refresh
into a `data/mstr_treasury_state.json` cache + load on each
precompute, or (b) at minimum read overrides from
`config["mstr"]["btc_holdings"]`, `["debt_usd"]`, `["shares_outstanding"]`
so the operator can update without code change. Add a staleness warning
when `_DEFAULT_BTC_HOLDINGS` is used and a `last_updated` annotation
in `mstr_deep_context.json` so Layer 2 knows the premium is stale.

**Confidence:** 88

---

### [P1] `fetch_klines` yfinance fallback path is unguarded against thread races and skips circuit-breaker accounting
**File:** `portfolio/price_source.py:222-243`
**Issue:** When a primary source (Binance/Alpaca) raises, the wrapper
calls `_fetch_yfinance(ticker, interval, period, limit)` as emergency
fallback (line 235). yfinance is NOT thread-safe — `data_collector.py`
acquires `yfinance_lock` (line 277, imported from shared_state) before
calling yfinance, but `price_source._fetch_yfinance` does NOT.

When the loop's 8-worker ThreadPoolExecutor fans out to multiple
tickers and Binance has a transient outage, multiple workers will
simultaneously enter `_fetch_yfinance` from the fallback path
unlocked. This races with `data_collector._fetch_one_timeframe`
which DOES hold `yfinance_lock` for its own yfinance call. The race
manifests as the documented "Tensor on device meta..." / MultiIndex
column corruption (see `fear_greed.py:138-146` comment block) but
intermittently, not on every call.

Separately, when the primary CB trips, this code falls through to
yfinance without recording the primary failure against any circuit
breaker because the inner `_fetch_binance_fapi` call is short-circuited
above (the CB itself caught the failure inside `_binance_fetch` of
`data_collector.py`). That part is fine. But the fallback log just
says "Investigate the primary outage" — it doesn't fire a Telegram
alert, doesn't bump a counter that the dashboard surfaces, and doesn't
fail loudly when the same primary fails 100 cycles in a row.

**Impact:** Silent degradation — Binance has a 10-min outage, yfinance
takes over for a few minutes (with 10-15 min lag), users see no
warning, and trading-decision math is off by 10-15 min on metals
during the outage.

**Fix:** (a) Acquire `yfinance_lock` in `price_source._fetch_yfinance`
identical to the data_collector pattern. (b) Add a separate
`primary_outage_counter` in shared_state that surfaces in the
dashboard `/api/market-health` endpoint and triggers a Telegram
warning when crossing a threshold (e.g. 5 consecutive primary
fallbacks). (c) When the fallback is taken for a non-allowed ticker
(`ticker not in _YFINANCE_LAST_RESORT`), log at ERROR with a 4h cooldown
Telegram alert similar to `fx_rates._fx_alert_telegram`.

**Confidence:** 85

---

### [P1] `crypto_macro_data._fetch_deribit_options` has unbounded loop O(strikes²) per request
**File:** `portfolio/crypto_macro_data.py:139-166`
**Issue:** The max-pain computation iterates `for candidate in all_strikes:
for strike in all_strikes:` — O(N²) where N is the number of unique
strikes on the nearest Deribit expiry. For BTC's quarterly expiries
that's commonly 100-150 strikes, so 10K-22K iterations per fetch,
called every 15 min via `_cached(... OPTIONS_TTL=900)`. That's
tolerable. But Deribit has no rate-limit-aware retry here either
(`fetch_json(...retries=2)`) and the entire result is materialised in
memory across all expiries (`expiry_data` defaultdict scales linearly
with strike count × expiry count, easily hundreds of MB during a
volatile day with 5-10 active expiries).

More important: the inner `_parse_expiry(s)` uses
`datetime.datetime.strptime(s, "%d%b%y")` which is locale-dependent
on Windows. If the Python process inherits a non-English locale
(possible on Swedish Windows 11), `MAR` may not parse and EVERY
quarterly expiry will silently fall into the `expiry_data` map but
fail the nearest-expiry selector, causing the fallback "max OI"
selector to pick a less liquid expiry — silently corrupting max-pain.

**Impact:** Silent max-pain corruption on locale-bearing systems;
possible OOM during extreme expiry surges.

**Fix:** Make `_parse_expiry` locale-independent
(parse the 3-letter month manually against a hardcoded MONTH dict).
Cap the strike count for max-pain math at the most active 50 strikes
around current spot price (cheap and identical accuracy).

**Confidence:** 82

---

### [P1] `seasonality_updater._fetch_hourly_klines` bypasses circuit breaker on Binance FAPI
**File:** `portfolio/seasonality_updater.py:51-86`
**Issue:** Uses `fetch_json` against `BINANCE_FAPI_BASE/klines` directly
with `_binance_limiter.wait()`, but does NOT consult
`binance_fapi_cb` (`portfolio.data_collector.binance_fapi_cb`). If
Binance FAPI is in OPEN state, every other module (including
`data_collector.binance_fapi_klines` and `futures_data._fapi_cb`) will
short-circuit immediately — but `seasonality_updater` will still hit
the API and rack up failed retries against an outage that everyone
else is correctly avoiding.

**Impact:** Wasted retries during Binance FAPI outage; specifically
hurts because seasonality is called from a separate scheduled refresh
path that doesn't see the loop's other CBs trip.

**Fix:** Either route through `data_collector.binance_fapi_klines`
(which already trips the CB) or import `binance_fapi_cb` here and add
the same `allow_request()` guard.

**Confidence:** 84

---

### [P1] `earnings_calendar` AlphaVantage path does NOT consume daily budget — silent quota leak
**File:** `portfolio/earnings_calendar.py:48-62`
**Issue:** The function explicitly waits on `_alpha_vantage_limiter`
but the comment on lines 49-52 acknowledges the call does not
increment `alpha_vantage._daily_budget_used` "because there is no
public increment function exported from that module."

That comment is currently true. Earnings polls all `STOCK_SYMBOLS`
on a 24h cache, but on initial cold start each ticker fires once
(N stock tickers — currently 1 MSTR, but the code is written to
generalise). More importantly, when the per-ticker cache misses
(e.g. after a wallclock midnight crossover before the 24h TTL
expires), a refresh storm can fire multiple AV calls all of which
bypass the 25/day budget tracker in `alpha_vantage.py`.

The result: `refresh_fundamentals_batch` reads
`_daily_budget_used = 0` and proceeds to attempt 25 OVERVIEW calls,
each of which gets a `Note: rate-limit exceeded` because AV's
server-side counter already counted the EARNINGS calls. Cache stays
stale, the wider fundamental-driven signal goes dark, and operators
have no diagnostic.

**Impact:** Daily AV budget can be silently exhausted by earnings
polls long before fundamentals refresh runs. Fundamentals cache then
goes stale, claude_fundamental signal degrades.

**Fix:** Add a public `alpha_vantage.bump_budget()` helper that
acquires `_cache_lock` and increments `_daily_budget_used`. Call it
from `_fetch_earnings_alpha_vantage` after a successful fetch. While
there, also add a `_check_budget`-style precheck so earnings polls
respect the budget too (currently they don't check it before
calling).

**Confidence:** 86

---

### [P2] `_aggregate_sentiments` accepts `s["scores"]` keys without `.get()` defensive — schema drift in any sub-model crashes the cycle
**File:** `portfolio/sentiment.py:695-701`
**Issue:**
```python
pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
```
If any single sub-model returns a row missing `"scores"` or missing
one of those three keys, this raises `KeyError`. The error propagates
through `get_sentiment` (no try/except around `_aggregate_sentiments`),
through the ticker pool, into the cycle log — but does NOT crash the
loop because the ticker pool catches per-worker exceptions. Result:
THIS ticker's sentiment goes silently HOLD for the cycle.

The risk is realistic: FinBERT in particular went through a transient
2026-05-03 race where `_predict_per_text` (line 442) wrote
`scores={"positive":0.33,"negative":0.33,"neutral":0.34}` on errors,
which is the correct safe default, BUT — the in-process model loader
(`bert_sentiment._predict_batched`) emits a Python dict with the same
shape ALWAYS today; the subprocess fallback path in `cryptobert_infer.py`
/ `trading_hero_infer.py` was the original source of truth. If that
subprocess script's output ever drops one of the three keys (e.g. a
binary-classifier model swap), this code raises.

**Impact:** Schema-fragile under sub-model swap. The migration from
CryptoBERT to Trading-Hero in 2026-04 narrowly avoided this — if it
had been a model with positive/negative-only output, the entire
sentiment voter goes dark per cycle.

**Fix:** Use `.get("scores", {}).get("positive", 0.0)` etc. and treat
missing keys as 0.0 in the weighted sum (or as 0.33 if all three are
missing).

**Confidence:** 82

---

### [P2] `crypto_precompute` falls into a per-symbol exception bucket and silently writes None to every field
**File:** `portfolio/crypto_precompute.py:178-197`
**Issue:** Funding rate and OI loops over BTCUSDT/ETHUSDT, with a
single outer try/except. If the first call raises (e.g. a 5xx),
the loop continues — but the inner `except Exception` blocks log
"crypto_precompute: data parse error" at DEBUG level (not warning).
Operators reading `data/agent.log` will miss this.

The published `crypto_deep_context.json` then ships with
`btc_funding_rate: null, btc_open_interest: null` — which downstream
`signals/crypto_macro` and the dashboard treat as "no signal", silently
HOLDing one of the more-active crypto signals.

**Impact:** Funding-rate signal silently dropped on transient Binance
errors with only a DEBUG log line.

**Fix:** Log at WARNING when a precompute sub-source returns None for
a previously-populated field. Bonus: persist a per-source
`last_success_at` and fire a Telegram alert if any source has been
None for >2 consecutive precomputes (4h).

**Confidence:** 80

---

### [P2] `_fetch_dxy_intraday` synth-DXY constant 58.0 is meaningless but written into the `"value"` field
**File:** `portfolio/macro_context.py:137-144`
**Issue:** The comment correctly notes that the constant 58.0 in
`synth = 58.0 * (eurusd ** -0.576)` is arbitrary and only the
*change* fields are meaningful. But the `_dxy_features_from_close`
helper still writes the meaningless absolute level into
`{"value": round(last, 4), "source": "EURUSD=X-synth"}`.

Downstream consumers `signals/dxy_cross_asset` are claimed to read
only change fields. But `dashboard/app.py` and `reporting.py`
include the DXY value in agent_summary — a Layer 2 prompt that reads
"DXY: 65.4" when the actual DXY is ~99 will conclude something is
deeply wrong (or the agent will hallucinate justification for the
bogus reading).

**Impact:** Layer 2 prompt pollution when the synth fallback fires
(currently rare because DX-Y.NYB usually works, but the fallback
path is exactly when an LLM-aware human can least afford to debug).

**Fix:** Write `"value": None` in the synth path (or emit a
`"value_is_synth": True` sentinel) so consumers know to display
"~99 (synth fallback)" or just hide the absolute value.

**Confidence:** 82

---

### [P2] `funding_rate._fetch_funding_rate` short-circuits with HOLD for normal funding — that's intended, but mass-HOLD on `data is None` is also returned, blurring the "no data" vs "actual HOLD" distinction
**File:** `portfolio/funding_rate.py:44-56`
**Issue:** On `data.get(...) is None` (Binance schema drift), returns
None — fine, signal disappears for the cycle. On a successful normal
funding rate (e.g. 0.0001), returns `{"action": "HOLD", "rate": ...}`.
The downstream voter sees `action == "HOLD"` as an active vote;
None means the voter abstains.

In aggregate this means the funding voter's "weak signal during
normal funding" gets counted toward MIN_VOTERS=3 but its information
content is near zero. This is by design (the rate band 0.0-0.0003 is
huge — most of normal funding lives there), but it pulls the
consensus toward HOLD on every cycle without good reason.

**Impact:** Funding voter contributes constant HOLD weight to
consensus 90%+ of the time, masking other signals' decisiveness.

**Fix:** Return None for the neutral band instead of HOLD — let the
voter abstain when funding is uninformative, and only vote when funding
is extreme. This is a semantic change; document in the signal registry
and rebaseline accuracy.

**Confidence:** 78

---

### [P2] `crypto_macro_data._load_ratio_history` and `_load_netflow_history` read JSONL with raw `open()` instead of `load_jsonl`
**File:** `portfolio/crypto_macro_data.py:270-289, 390-411`
**Issue:** Direct `with open(...) as f: for line in f:` — bypasses
`file_utils.load_jsonl` which has atomic-safe reading. If the file
is being concurrently written by `_append_ratio_history` /
`_append_netflow_history` (which use `atomic_append_jsonl`), the
reader can see a half-written line. The reader catches `JSONDecodeError`
and skips the broken line, which is mostly safe — but ALSO catches
`KeyError`, which would hide a schema bug.

**Impact:** Minor — atomic_append_jsonl is mostly OS-atomic at
~4KB block size, so this is rare. But the failure mode is silent
schema-drift suppression, which is exactly the failure CLAUDE.md
warns against.

**Fix:** Switch to `from portfolio.file_utils import load_jsonl`
and use it consistently.

**Confidence:** 80

---

### [P2] `mstr_precompute._fetch_market_data` uses `yf.Ticker.info` which is known-unreliable
**File:** `portfolio/mstr_precompute.py:152-158`
**Issue:** `info = getattr(tk, "info", {}) or {}` then reads
`fiftyTwoWeekHigh`, `shortPercentOfFloat`, `recommendationKey`.
yfinance `.info` is documented-unreliable (frequent silent failures
returning empty dict, frequent schema drift on
`recommendationKey` vs `recommendationMean`), and it makes a separate
slow HTTP call. The code falls back to either `recommendationKey` or
`recommendationMean` but always assigns whichever is non-None — so
`analyst_consensus` can flip between a string ("buy") and a float
(2.1) across cycles, breaking downstream consumers that compare it
as a string.

**Impact:** Schema-flip between cycles in `mstr_deep_context.json`
breaks consumers; also burns yfinance quota.

**Fix:** Normalize to one canonical type — prefer
`recommendationKey` (string) and only fall back to
`recommendationMean` after mapping it to a string bucket
("buy"/"hold"/"sell"). Mark this whole `info` fetch as "advisory
data, may be None" and don't fail-loud on its absence.

**Confidence:** 80

---

### [P2] `econ_dates` hard-codes events through Dec 2027 only
**File:** `portfolio/econ_dates.py:38, 72, 98` and `portfolio/fomc_dates.py:25-34`
**Issue:** Today is 2026-05-27 (~19 months runway). After Dec 2027,
`next_event()` returns None, `is_macro_window()` returns False, and
the entire econ-calendar voter silently goes dark.

The comment in `fomc_dates.py` references the Fed's calendar URL but
there's no scheduled refresh. After Dec 2027 every backtest run
through this code will silently treat the world as event-free.

**Impact:** Silent econ-signal loss starting 2028-01-01.

**Fix:** Either (a) add a yearly auto-fetch of FOMC dates from the
Federal Reserve's public schedule into a `data/fomc_dates_cache.json`
file, or (b) at minimum add a runtime warning that fires when
`now > FOMC_DATES_2027[-1] - 60 days` reminding the operator to
extend the list. NFP dates (first Friday of month) could be computed
algorithmically — only CPI/GDP/FOMC require external maintenance.

**Confidence:** 82

---

### [P2] `bert_sentiment._predict_per_text` silently returns the {0.33, 0.33, 0.34} neutral placeholder on per-text errors
**File:** `portfolio/bert_sentiment.py:442-452`
**Issue:** Per-text fallback writes a zero-confidence neutral
placeholder when forward pass fails. Combined with the
`_aggregate_sentiments` weighted average, multiple such placeholders
PUSH the average toward neutral (which then drowns out a few
decisive headlines — the exact failure pattern documented in the
`portfolio/sentiment.py` `_majority_label` comment block).

The legitimate use case is: one bad headline tokenizer-error
shouldn't kill the batch. But there's no logging-level distinction
between "this one headline tokenizer-errored" (worth WARNING) and
"every headline tokenizer-errored because meta-tensor race"
(definitely worth ERROR + Telegram).

**Impact:** Silent neutral-mass injection on bad batches.

**Fix:** Track placeholder count across the batch and log at WARNING
if `placeholder_count >= len(texts) / 2`. Also short-circuit the
batched path with RuntimeError if `placeholder_count == len(texts)`
so the calling code's try/except in `sentiment.py:982` falls into the
fingpt-only path instead of silently writing a useless neutral
verdict to `sentiment_ab_log.jsonl`.

**Confidence:** 80

---

### [P2] `fetch_usd_sek` returns a stale cached rate without informing the caller
**File:** `portfolio/fx_rates.py:60-65`
**Issue:** When `fetch_with_retry` fails AND the cache exists, the
function returns the cached value with a one-shot Telegram alert if
the cache is >2h old. No caller-side signal — `portfolio_mgr.py` and
`monte_carlo_risk.py` get a float back and can't tell if it's fresh
or 24h stale.

The hardcoded `FX_RATE_FALLBACK = 10.50` is currently roughly correct
(USD/SEK has been 10.20-10.80 through 2026), but if SEK weakens to
say 11.50 over the next quarter and the FX API is down, valuations
go off by ~9.5%. That's not catastrophic but it's also not surfaced
as a confidence reduction in any signal.

**Impact:** Portfolio valuations + Monte Carlo VaR computed against
stale FX without consumer-side awareness.

**Fix:** Return `(rate, source)` tuple where source is one of
"fresh", "stale_cached", "fallback_hardcoded", and downstream callers
can decide whether to fail loud or proceed cautiously. Keep the
backwards-compat `fetch_usd_sek()` returning rate only as a thin
wrapper.

**Confidence:** 80

---

### [P2] `metals_precompute` COT history pruning targets 104 entries for two metals — bug if pruning isn't per-metal-aware
**File:** `portfolio/metals_precompute.py:594-598`
**Issue:** `if len(existing) > 104: prune_jsonl(max_entries=104)` —
but the file contains both silver AND gold records (line 591
`if entry.get("metal") == metal`). 104 entries ≈ 52 weeks × 2 metals,
so after exactly 1 year the prune is per-metal-correct. But
`prune_jsonl` (in `file_utils.py`) presumably keeps the last 104
entries by file order, NOT by metal — which means if gold's COT
fetch fails for a stretch, the file ends up with 100 silver entries
and 4 gold entries after a prune, losing gold history.

Not catastrophic (the loss is "historical" only — `_compute_cot_trend`
only uses the last 8 weeks), but the cap is misleading.

**Impact:** Asymmetric COT history loss possible if one metal fetches
fail; the 8-week trend window is OK but the on-disk history is
asymmetric.

**Fix:** Bump cap to 208 (or split into per-metal files
`cot_history_silver.jsonl` / `cot_history_gold.jsonl`).

**Confidence:** 78

---

### [P3] `crypto_macro_data._fetch_deribit_options` does not validate Deribit instrument-name format on the entire result — drops silently on any 4-part split that fails float-parsing
**File:** `portfolio/crypto_macro_data.py:81-87`
**Issue:** `if len(parts) != 4: continue` and
`try: strike = float(strike_str) except: continue`. Both are reasonable
guards, but Deribit added futures (length 3) and combos (length 5)
historically, and a schema rev could ship "BTC-PERP" or "BTC-FUT-..."
instruments mixed in. The code drops these silently — fine for
max-pain math, but a user reading
`nearest_expiry: "PERP"` or the silent-drop pattern would never know.

**Impact:** Minor observability gap.

**Fix:** Log at DEBUG (once per cycle) the count of dropped
instruments by reason. Bump to WARNING if >50% are dropped.

**Confidence:** 80

---

### [P3] `news_keywords.score_headline` pattern cleanup is fragile
**File:** `portfolio/news_keywords.py:155-159`
**Issue:** `pattern.pattern.replace(r"\b", "").replace("\\", "")` —
manual string mangling of the compiled regex's `.pattern` to recover
the original keyword. This works today but breaks the moment any
keyword in `ALL_KEYWORDS` contains an escaped special char (which is
expected to be escaped via `re.escape`).

For example, if anyone adds `"u.s. recession"` to the keyword list,
the compiled pattern becomes `\bu\.s\.\ recession\b` and the cleanup
produces `u.s. recession` correctly only because `.replace("\\", "")`
happens to strip the backslashes — but if the order changes or a
new escape needs to survive (e.g. `\d`), the function silently
returns the wrong keyword for `matched` reporting.

**Impact:** Cosmetic; only affects what's reported in the keyword
match list, not the score.

**Fix:** Maintain a parallel `_KEYWORD_PATTERN_TO_NAME` dict mapping
compiled patterns back to their original keyword strings.

**Confidence:** 78

---

### [P3] `session_calendar.get_session_info("stock_us")` open-time check is buggy
**File:** `portfolio/session_calendar.py:156-158`
**Issue:** `is_open = (now.weekday() < 5 and now.replace(hour=open_utc,
minute=30, second=0) <= now < session_end)`. The `.replace(...)` uses
`minute=30` (correct: NYSE opens 09:30 ET) but `hour=open_utc` is
either 13 or 14 — so the open boundary is 13:30 UTC summer / 14:30
UTC winter. That's correct. But the close boundary
`session_end = now.replace(hour=close_utc, minute=0, ...)` —
`close_utc` is either 20 (summer) or 21 (winter) — so close is
20:00 UTC summer / 21:00 UTC winter. NYSE closes at 16:00 ET, which
is 20:00 UTC summer / 21:00 UTC winter. Correct.

So the math is right, but the construction `now.replace(...)` on the
right side of `<=` and `<` is bizarre — it constructs a "today at
that hour" anchor. On the rare cycle that fires across midnight
(loop tick at 00:00:05 UTC), `now.weekday() < 5` may be True but
`now.replace(hour=20)` is then 20:00 of today which is AFTER `now`,
giving `is_open = False` correctly. Edge case is correct.

**Impact:** None; just a code-smell.

**Fix:** Compute `session_open` and `session_end` as date-anchored
datetimes once, like the EU branch does, instead of using
`.replace()` in a comparison.

**Confidence:** 80

---

### [P3] `seasonality.compute_hourly_profile` requires DatetimeIndex but doesn't validate timezone
**File:** `portfolio/seasonality.py:44-48`
**Issue:** Uses `df.index.hour` — if the upstream feeder (currently
`seasonality_updater._fetch_hourly_klines`) ever produces a tz-naive
index, the "hour" is whatever local interpretation pandas chose.
`_fetch_hourly_klines` does the right thing today
(`pd.Timestamp(k[0], unit="ms", tz="UTC")` then `set_index("time")`),
but this is implicit contract — no assert.

If a future caller swaps in `data_collector.binance_klines` which
returns `pd.to_datetime(df["open_time"], unit="ms")` without tz,
`compute_hourly_profile` bucketizes by NAIVE hour-of-day which is
silently wrong by however many hours UTC offset.

**Impact:** Latent bug waiting to fire if `_fetch_hourly_klines` is
ever replaced.

**Fix:** Add explicit `if df.index.tz is None: raise ValueError(...)`
at the top of `compute_hourly_profile`. Or convert to UTC unconditionally.

**Confidence:** 82

---

### [P3] `data_refresh.py` does not respect `_binance_limiter` and hardcodes a 200ms intra-batch sleep
**File:** `portfolio/data_refresh.py:30-48`
**Issue:** Loops `download_klines` for 365d / 1h candles, calling
`fetch_with_retry` directly with `time.sleep(0.2)` between batches.
This is a Freqtrade-style bulk download, intended for off-cycle
backtest data — but it bypasses the shared `_binance_limiter` and
fires up to 5 req/sec for the duration of the download. If anyone
runs this concurrently with the main loop, the loop's Binance
budget can be partially consumed and trip the FAPI circuit breaker.

Module docstring is missing; this code is barely referenced from
the rest of the codebase, but if anyone imports `refresh_all` from a
notebook while the loop is live, expect a CB trip.

**Impact:** Only fires if someone runs the script while the loop is
live; the script is documented as a backtest helper.

**Fix:** Either gate behind a "loop must be stopped" check, or route
through the shared limiter + CB, or at minimum add a docstring
"DO NOT RUN while main loop is active".

**Confidence:** 78

---

### [P3] `crypto_precompute._fetch_market_data` imports a `data.crypto_data` module that may not exist (delegation)
**File:** `portfolio/crypto_precompute.py:140-150`
**Issue:** `from data.crypto_data import get_fear_greed,
get_onchain_summary`. The `data/` directory IS the data files
directory, not a Python package — and the project's `data/crypto_loop.py`
is a Python file. `data/crypto_data.py` may or may not exist; I did
not verify in this review pass.

If the import fails (no such module / no such symbol), the `except`
catches and logs at WARNING — but then `out["fear_greed"]` and
`out["onchain_btc"]` stay None and propagate into the published
context. Downstream consumers see a `crypto_deep_context.json` with
permanently-null shared sections.

The right fix is to use `portfolio.fear_greed.get_crypto_fear_greed`
and `portfolio.onchain_data.get_onchain_data`, both of which already
exist and are in scope of this review.

**Impact:** If `data.crypto_data` is missing/refactored, two key
crypto-context fields are silently None forever.

**Fix:** Switch the imports to the portfolio package equivalents:
`from portfolio.fear_greed import get_crypto_fear_greed as
get_fear_greed` and similarly for on-chain.

**Confidence:** 78

---

## Adversarial-angle coverage matrix

| Angle | Result |
|-------|--------|
| 1. Rate-limit/429 handling | Mostly OK (RETRYABLE_STATUS includes 429 + jitter), but quota leak via earnings_calendar (P1) and crypto_precompute bypass of `_binance_limiter` (P1). |
| 2. Stale cache TTL | OK — `_cached` enforces ttl + `_MAX_STALE_FACTOR`; on-chain has good resume logic. |
| 3. API outage = HOLD vs random vote | Funding rate ambiguity (P2); otherwise None=abstain pattern is consistent. |
| 4. API key in code | None found. All consumers route through `api_utils.load_config`. |
| 5. Schema drift | `_aggregate_sentiments` (P2), `mstr_precompute.info` flip (P2), Deribit parse (P3). |
| 6. Timezone | OK — UTC throughout; `seasonality` has implicit-contract bug (P3); `session_calendar` US-stock has a code-smell (P3). |
| 7. `10m` interval bug | None in scope. |
| 8. yfinance quirks (Adj Close, MultiIndex) | OK — flatten patterns present; `_fetch_yfinance` thread-race (P1). |
| 9. price_source CL=F fallback | OK; the fallback is yfinance-allowed for oil. |
| 10. Concurrent refreshers Binance budget | crypto_precompute + mstr_precompute bypass `_binance_limiter` (P1). |
| 11. NewsAPI keyword spam | OK — capped to metals + MSTR only, daily quota enforced. |
| 12. Sentiment model memory leak | OK — `_models` cache is lazy + bounded by `_MODEL_CONFIGS`. |
| 13. Earnings calendar timezone | OK — date-only comparisons. |
| 14. FOMC dates run out | End of 2027 (P2). |
| 15. Precompute race with main loop | Atomic writes used throughout; main risk is shared cache `_loading_keys` not used by precompute path. |
| 16. Fundamentals cache stale-on-failure | OK — fallback to disk; `_is_stale` enforces 5-day max. Risk: AV budget leak via earnings (P1). |

## Relevant absolute paths

- `Q:\finance-analyzer\portfolio\data_collector.py`
- `Q:\finance-analyzer\portfolio\alpha_vantage.py`
- `Q:\finance-analyzer\portfolio\earnings_calendar.py`
- `Q:\finance-analyzer\portfolio\crypto_precompute.py`
- `Q:\finance-analyzer\portfolio\mstr_precompute.py`
- `Q:\finance-analyzer\portfolio\metals_precompute.py`
- `Q:\finance-analyzer\portfolio\oil_precompute.py`
- `Q:\finance-analyzer\portfolio\price_source.py`
- `Q:\finance-analyzer\portfolio\crypto_macro_data.py`
- `Q:\finance-analyzer\portfolio\sentiment.py`
- `Q:\finance-analyzer\portfolio\bert_sentiment.py`
- `Q:\finance-analyzer\portfolio\seasonality_updater.py`
- `Q:\finance-analyzer\portfolio\funding_rate.py`
- `Q:\finance-analyzer\portfolio\fx_rates.py`
- `Q:\finance-analyzer\portfolio\macro_context.py`
- `Q:\finance-analyzer\portfolio\econ_dates.py`
- `Q:\finance-analyzer\portfolio\fomc_dates.py`
- `Q:\finance-analyzer\portfolio\data_refresh.py`
- `Q:\finance-analyzer\portfolio\news_keywords.py`
- `Q:\finance-analyzer\portfolio\session_calendar.py`
- `Q:\finance-analyzer\portfolio\seasonality.py`
- `Q:\finance-analyzer\portfolio\shared_state.py`
- `Q:\finance-analyzer\portfolio\http_retry.py`
- `Q:\finance-analyzer\portfolio\api_utils.py`
