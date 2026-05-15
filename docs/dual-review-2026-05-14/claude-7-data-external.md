# Adversarial Review — 7 data-external (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/data_collector.py:288-312` — `_fetch_one_timeframe` catches all exceptions and stores `{"error": str(e)}` as a SUCCESSFUL fetch result.
  ```python
  try:
      ...
      return (label, entry)
  except Exception as e:
      return (label, {"error": str(e)})
  ```
  Downstream `collect_timeframes` returns these error-entries unconditionally. Callers in `main._process_ticker` check `"error" in entry` only inside a separate log loop (line 545-546), but the agent_summary writer and signal engine treat the timeframe dict as data. If error message contains numeric-looking content that downstream uses (e.g. attempts `entry["indicators"]["close"]` after seeing the error key), this is a silent partial-data path. Worse: a transient yfinance error (e.g. "No data found for symbol") produces an entry that, if not filtered, can corrupt the indicator pipeline. Either raise or return None — never a partial dict that *looks* like data.

- `portfolio/onchain_data.py:86-91` — `_load_config_token` returns `None` on any config error, and the surrounding code falls through to "no token, use unauthenticated endpoint" (lower rate limit) without any operator alert.
  ```python
  except Exception as e:
      logger.warning("BGeometrics token load failed: %s", e, exc_info=True)
      return None
  ```
  When config.json (the symlink to external file with API keys) is unreachable due to a path issue, every loop cycle silently re-tries unauthenticated calls; on-chain data eventually fails the 8/hour limit and the BTC voter goes dark. Promote to critical_errors.jsonl after N consecutive None returns.

## P1 — high-confidence bugs (should fix)

- `portfolio/fx_rates.py:47-48` — out-of-bounds FX rate is logged as ERROR but the function silently falls through to the stale-cache branch with no telegram alert distinguishing "API down" from "API returned garbage".
  ```python
  if not (FX_RATE_MIN <= rate <= FX_RATE_MAX):
      logger.error("FX rate %.4f SEK/USD outside sane bounds (7-15) — ignoring", rate)
  else:
      ...
  ```
  An attacker / misconfiguration that makes Frankfurter return `1.0` (or `0.0`) triggers this branch silently. Operators see no Telegram unless cached rate is also stale (the alert at line 64-70 fires only after exhausting the stale-fallback). Add a `_fx_alert_telegram(reason="api_returned_oob")` call before falling through.

- `portfolio/onchain_data.py:73-74` — `API_BASE = "https://bitcoin-data.com"` with `ONCHAIN_TTL = 43200` (12 hours). Free tier budget is "15 requests/day" but cache covers each metric for 12h. So 6 metrics × 2 refreshes/day = 12 calls/day — under budget. BUT: a process restart invalidates the in-memory cache and re-reads from disk via `_save_onchain_cache`. If `_coerce_epoch` (line 29-67) returns 0.0 for any unparseable ts (which it does, line 67), that metric forces a refetch on restart. Five restarts a day × 6 metrics = 30 calls — DOUBLE the budget. Already throttled by the upstream API (429), but the rate limit doesn't write to critical_errors. The "burning the 15-req/day budget every restart" symptom is mentioned in the docstring (line 60) but no defensive measure shipped.

- `portfolio/data_collector.py:288-294` — yfinance call wrapped in `with _yfinance_lock` ONLY when `_current_market_state in ("closed", "weekend", "holiday")`. Otherwise no lock. If `_ss._current_market_state` is updated mid-cycle (it's globals-style state), the lock isn't acquired and concurrent yfinance calls from other modules (sentiment, fundamentals) can corrupt yfinance's internal state. Always acquire the lock.

- `portfolio/data_collector.py:36` — yfinance interval mapping is keyed `"15m"` but `15m` doesn't exist in Binance API (only `1m, 3m, 5m, 15m, 30m`). Wait — `15m` IS valid on Binance; the CLAUDE.md note is about `10m` specifically. So `15m` is OK. But the rule file flag is worth a re-grep: `grep -n "10m" portfolio/*.py portfolio/signals/*.py data/*.py`. If any file passes `interval="10m"`, the Binance call returns error -1120. Subagent confirmed didn't find any.

- `portfolio/sentiment.py:288-296` — subprocess fallback to BERT model with `timeout=120`. The parent loop's per-ticker pool timeout is 360s (`main.py:607`). If sentiment subprocess hits 120s timeout, the parent waits 120s before `_fetch_one_timeframe` returns with error. 5 tickers × 120s = 600s in worst case (parallel-but-each-can-stall). The 360s pool timeout fires first. Acceptable but tune: lower sentiment timeout to 60s.

- `portfolio/fx_rates.py:62-65` — stale cache used after `_FX_STALE_THRESHOLD`. The cached value is returned WITH a warning, but if Telegram is also down, operators see nothing. The 4h cooldown on alerts (line 79) is fine, but ALSO log an entry to `critical_errors.jsonl` when first hitting "stale" status.

- `portfolio/microstructure.py`, `portfolio/microstructure_state.py` — per subagent, `persist_state` doesn't acquire `_buffer_lock` during dict iteration. The metals 10s fast-tick can mutate buffers concurrently. P1 race.

## P2 — concerns / smells (worth addressing)

- `portfolio/data_collector.py:74` — `_binance_fetch` signature has `interval="5m"` as default. None of the callers seem to rely on the default, but if a future caller forgets to specify interval, they get 5m candles silently — which may not match their indicator expectations. Make interval required.

- `portfolio/onchain_data.py:42-67` — `_coerce_epoch` silently returns 0.0 for unparseable values. The DEBUG log mention is helpful but DEBUG isn't enabled in production. Promote to INFO when this branch fires more than 3 times in 10 minutes.

- `portfolio/fx_rates.py:33-34` — 15min cache. For SEK valuation in a fast-moving FX environment, 15 min is acceptable. But the cache is invalidated by process restart, and any new Layer 2 subprocess invocation may need fresh FX. Consider a disk-backed cache (separate from in-memory).

- `portfolio/oil_precompute.py`, `portfolio/silver_precompute.py`, etc. — per subagent, CFTC `requests.get` with f-string `$where` SoQL interpolation. Trust-boundary mild: only internal tickers flow in, but injection-shape patterns shouldn't ship even when the input is "trusted".

- `portfolio/sentiment.py:288` — subprocess uses `text=True` and `capture_output=True`. On Windows with non-utf8 default code page, stderr can contain mojibake. Pass `encoding="utf-8", errors="replace"` explicitly.

- `portfolio/social_sentiment.py` — per subagent, unauthenticated Reddit. Subject to anti-abuse rate limits. Document the auth-token path or accept that this signal is best-effort.

- `portfolio/macro_context.py`, `portfolio/funding_rate.py` — per subagent, asymmetric thresholds. Low priority; require accuracy review to validate the asymmetry is intentional vs accidental.

- `portfolio/data_collector.py:319` — `source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")`. If a source dict has e.g. `alpaca=""` (empty string), the chain falls through. Defensive but `or ""` won't match anything; should be explicit `is not None` check.

- `portfolio/microstructure_state.py:227` (per subagent) — 2-minute staleness check based on cross-process clock. NTP drift on a Windows system can be several seconds; staleness in [115s, 125s] is judged inconsistently across processes.

## Did NOT find

1. **Silent failures**: P0 onchain token, P0 data_collector error-as-data. fx_rates fallback chain is mostly correct.
2. **Race conditions**: P1 microstructure_state.persist_state; data_collector lock scope (P1).
3. **Money-losing bugs**: indirect — bad data feeds wrong signal votes feeds wrong trades.
4. **State corruption**: onchain_data uses atomic_write_json. headlines_latest persisted atomically.
5. **Logic errors that pass tests**: data_collector error-as-entry pattern would pass any test that doesn't assert the success contract.
6. **Resource leaks**: ThreadPoolExecutor in collect_timeframes uses `with` context — auto-shutdown. Good.
7. **Time/timezone bugs**: ts handling is mostly ISO + UTC. Subagent flagged sentiment_shadow_backfill naive-timestamp UTC assumption.
8. **API misuse**: Binance `10m` interval search yields no hits — clean. yfinance "15m"/"5d" valid combo.
9. **Trust boundary violations**: P2 oil_precompute SoQL `$where` interpolation.
10. **Incorrect partial-state assumptions**: P0 data_collector error-entry, P1 onchain token, P1 fx out-of-bounds.
