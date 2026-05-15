# Cross-Critique — 7 data-external

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/data_collector.py:288-312` — `_fetch_one_timeframe` returns `{"error": str(e)}` as a "successful" fetch result (P0).** Both reviewers identify the exact same line range. Codex frames the downstream consequence more sharply: "if [signal_engine] doesn't recognize the error schema, they'll iterate over the error dict and silently treat absence of `"indicators"` key as 'no signal'". **Independent rediscovery — confidence very high.** Action: either propagate via the future's exception or use a structured failure tag `{"action": None, "error": ...}` consistent with success shape.

- **`portfolio/data_collector.py:280-294` — race window between dispatcher market-state check and inner `_fetch_klines` (P0/P1).** Claude flagged the conditional lock acquisition (P1). Codex extends with the inner re-check at `:262` that can flip the state mid-cycle; the comment at lines 274-277 explicitly documents yfinance is not thread-safe; the race reintroduces the original bug. **Codex's framing is sharper — promote to P0.** Action: move lock inside `_fetch_klines` itself.

- **`portfolio/fx_rates.py:46-53` — out-of-bounds FX rate silently falls through to fallback (P1).** Both flag. Codex notes the result is the hardcoded 10.50 fallback while real rate may have moved — portfolio_mgr / monte_carlo use wrong SEK valuations. **Both right.**

- **`portfolio/microstructure_state.py` `persist_state` iteration race (P0/P1).** Claude says "per subagent finding". Codex shows the exact line (`for ticker in _snapshot_buffers:` without `_buffer_lock`), the concrete failure mode (`RuntimeError: dictionary changed size during iteration`), and the silent consequence (orderbook_flow signal HOLD-forever via 2-min stale gate). **Codex significantly stronger.** Action: snapshot keys under lock.

- **`portfolio/microstructure_state.py:227` — 2-min stale threshold uses cross-process wall clock (P1).** Both flag. Codex adds NTP-step concern; Claude adds "drift in [115s, 125s] judged inconsistently". Same finding.

- **`portfolio/sentiment.py:288-296` — subprocess fallback timeout=120s (P1).** Both flag. Codex notes `subprocess.run` does kill on timeout in modern Python but the surrounding callers wrap in try/except + neutral return; combined with 33 active signals × 5 tickers, hung BERT can pin up to 120s × N workers. **Cross-validates.**

- **`portfolio/oil_precompute.py` / `metals_precompute.py` CFTC SoQL `$where` interpolation (P1/P2).** Both flag. Codex shows the exact code template `f"...='{commodity_name}'"` and adds the precomputes bypass rate limiting and circuit breakers. **Both right.**

## Codex found, Claude missed

- **`portfolio/crypto_precompute.py:159-167, 179-196, 210-216, 225-232` — `requests.get` bypasses every rate-limiter, circuit-breaker, retry layer (P0).** Six Binance + three yfinance per 4h, no `_binance_limiter`, no `binance_spot_cb`, no `yfinance_lock`. Concurrent with main 60s loop + dashboard, can trip Binance IP-level 429 ban for ALL consumers. **Real P0, Claude missed.**

- **`portfolio/mstr_precompute.py:140-200` — hardcoded MSTR balance-sheet constants drive live NAV-premium calculation (P0).** `_DEFAULT_BTC_HOLDINGS=471,107` (in mstr_precompute) vs `data/crypto_data.py:184 MSTR_BTC_HOLDINGS=499,096` — 6% divergence already exists. Any L2 prompt reading `mstr_deep_context.json` gets stale NAV. **Real P0, Claude missed.** Action: log WARNING with date+source when defaults used; reject precompute output when fetched_at > N days.

- **`data/crypto_data.py:73-85` — `get_fear_greed` IndexError on alternative.me maintenance window swallowed (P0).** `data["data"][0]` on empty list → IndexError → swallowed → returns None. `portfolio/fear_greed.py:106` has explicit defense (with historical "P1-13 / 04-29 DE-P1-2" comment) but `data/crypto_data.py` never got the patch. **Real P0 missed by Claude.**

- **`portfolio/earnings_calendar.py:48-52` — AV daily-budget tracking explicitly bypassed (P1).** Comment admits the bug. Free tier 25/day; restart loops or day-rollover cache misses burn hidden calls. **Real P1, Claude missed.** Action: expose AV budget increment helper.

- **`portfolio/news_keywords.py:80-83` — regex `\b` with multi-word phrases.** Today's keywords don't include non-word chars; if `"S&P 500"` is added it silently never matches. **Narrow P1, useful regression test prevention.**

- **`portfolio/sentiment_shadow_backfill.py:211-213` — `entry_time.tzinfo is None` → `replace(tzinfo=UTC)` assumes legacy log was UTC.** Historical entries pre-UTC migration may have been CET. **Real P1.**

- **`portfolio/session_calendar.py:155-157` — `now.replace(hour=..., minute=30, second=0)` doesn't reset microsecond.** Cosmetic P2 but real.

- **`portfolio/macro_context.py:143` — `synth = 58.0 * (eurusd ** -0.576)` returns synthetic DXY in same dict shape as real path.** A consumer skipping `result["source"]` gets a wrong-DXY threshold. **Real P2.**

- **`portfolio/alpha_vantage.py:266-269, 273-276, 291-293` — failure path runs `_cb.record_failure()` then breaks silently; no Telegram alert.** "Fundamentals haven't refreshed in N days" never surfaces. **Real P2.**

- **`portfolio/onchain_data.py:178, 187` — `latest = data[0] if isinstance(data[0], dict) else data[-1]` — if both aren't dict, AttributeError swallowed.** Real P2.

## Claude found, Codex missed

- **`portfolio/onchain_data.py:86-91` — `_load_config_token` returns None on config error; silent fallback to unauthenticated endpoint (P0).** Cron/restart can burn the 15-req/day BGeometrics budget without operator alert. Codex didn't open onchain_data this deeply. **Real P0.**

- **`portfolio/onchain_data.py:73-74` — `ONCHAIN_TTL = 43200` (12h) but `_coerce_epoch` returns 0.0 for unparseable ts → forces refetch on restart, 30 calls/day vs 15 budget.** Real P1, Codex didn't catch.

- **`portfolio/data_collector.py:74` — `_binance_fetch` default interval="5m"; future callers can silently get wrong candles.** P2 defensive. Codex didn't.

- **`portfolio/fx_rates.py:62-65` — stale cache returned with warning but no critical_errors.jsonl entry.** Codex P1 frames the alert hole on "API returned a number we don't trust" path; Claude flags the parallel hole on the stale-fallback path. **Both flavours real.**

- **`portfolio/data_collector.py:319` — `source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")` — empty-string falls through.** Real P2.

## Disagreements

**Severity on `data_collector.py:280-294` race**: Claude P1, Codex P0. Codex's framing wins because the race re-introduces the *exact original bug* the existing comment was placed to prevent. Use Codex's severity.

**Severity on `microstructure_state.persist_state` race**: Claude P1, Codex P0. Codex shows the exception flow (RuntimeError → silently disables persistence → orderbook_flow HOLD-forever). **Use Codex's severity.**

## What BOTH missed (third pass)

- **`portfolio/precompute_*.py` consistency** — Codex flagged crypto_precompute, mstr_precompute. Neither reviewer audited `silver_precompute.py`, `metals_precompute.py`, `oil_precompute.py` for the same `requests.get` bypass pattern. Probably present (Codex flagged metals/oil for SoQL but not the rate-limiter bypass).

- **News headline poll cadence vs `news_event` signal staleness gate.** Claude flagged news_event substring bugs (in #6 review). Neither reviewer cross-checked whether `headlines_latest.json` updater (data-external) writes at a rate the signal can consume. If updater is 15 min and signal expects 5 min, signal goes HOLD-forever.

- **`portfolio/funding_rate.py` Binance FAPI endpoint** — Codex flagged threshold asymmetry. Neither reviewer checked whether the FAPI endpoint's response shape has changed (Binance has done so before with `10m` removal); if the cached parse logic assumes a field name removed in May 2026, the funding rate signal silently HOLD-only.

- **`portfolio/data_collector.py` Alpaca pagination.** Neither reviewer audited. Alpaca returns paged results for stocks; if pagination logic stops at page 1, the indicators are computed on a truncated dataset.

- **Shared HTTP session lifecycle.** `http_retry.fetch_json` is referenced. Neither reviewer audited whether `requests.Session` objects are reused or recreated per call — connection-pool exhaustion under burst loads is plausible.

## Verdict

P0 list after cross: **6 confirmed** (data_collector error-as-data, data_collector market-state race, crypto_precompute rate-limit bypass, mstr_precompute stale hardcoded constants, crypto_data.py F&G IndexError, onchain_data token silent-None).
P1 list after cross: **~10 confirmed** (microstructure_state persist race, fx_rates oob silent fallback, sentiment subprocess timeout, AV budget bypass, news_keywords regex, sentiment_shadow_backfill TZ assumption, microstructure clock skew, oil/metals precompute SoQL injection-shape, onchain_data 12h TTL, data_collector lock scope).
P2 list after cross: ~8.

Data-external is the **silent-failure surface** — most P0s here mask data quality issues that bias every downstream signal vote and Layer 2 prompt.
