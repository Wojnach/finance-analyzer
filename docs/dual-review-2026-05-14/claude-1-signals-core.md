# Adversarial Review — 1 signals-core (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/forecast_accuracy.py:320-327` — `backfill_forecast_outcomes` deletes unprocessed predictions when `max_entries` cap hits.
  ```python
  for entry in entries:
      ...
      modified_entries.append(entry)
      if updated >= max_entries:
          break       # NB: break here drops every entry after this one
  if updated > 0:
      _write_predictions(modified_entries, path)
  ```
  `_write_predictions` (line 374) calls `atomic_write_jsonl(path, entries)` — that REPLACES the whole file with the prefix `modified_entries` accumulated so far. On any cycle where the backfill backlog ≥ `max_entries` (default 500), every prediction past the break point is silently truncated from `data/forecast_predictions.jsonl`. The 2026-05-04 review flagged this; the current code appends `entry` to `modified_entries` before the break (so the trigger row survives) but the entries *after* this in `entries` are still dropped. Fix: extend with the unprocessed tail before writing — `modified_entries.extend(entries[entries.index(entry)+1:])` — or rewrite the loop to seek+truncate only what was processed.

- `portfolio/forecast_signal.py:365,372` vs `portfolio/forecast_accuracy.py:145-148` — schema mismatch: writer emits nested `chronos`/`prophet` payloads; accuracy scorer reads `entry["sub_signals"]` / `entry["raw_sub_signals"]`. 2026-05-04 P1 still unresolved.
  ```python
  # forecast_signal.py:365
  entry["chronos"] = chronos_result
  entry["prophet"] = prophet_result
  ```
  ```python
  # forecast_accuracy.py:145
  sub_signals = entry.get("sub_signals", {})
  if use_raw_sub_signals and entry.get("raw_sub_signals"):
      sub_signals = entry.get("raw_sub_signals", {})
  for sub_name, vote in sub_signals.items():
      ...
  ```
  Every row written by `forecast_signal.run_forecasts` contributes ZERO scored votes — accuracy reports and `accuracy_degradation` forecast checks stay permanently empty. Production effect: cannot detect Chronos/Prophet degradation.

## P1 — high-confidence bugs (should fix)

- `portfolio/signal_engine.py:3069-3074, 3083-3091, 3100-3103, 3116-3120` — direct `ind[...]` access without `.get()` for `rsi`, `macd_hist`, `macd_hist_prev`, `ema9`, `ema21`, `price_vs_bb`. If the indicator builder drops any of these (which it can on a partial-data ticker — e.g. yfinance empty candles or a horizon with `< 26` bars), the function raises `KeyError` instead of degrading to HOLD. The function is wrapped in `try/except Exception` upstream which converts the KeyError to a generic logged failure — silent failure pattern. Every downstream voter (sentiment, ML, etc.) is skipped because we abort the signal computation. Compare to `outcome_tracker.py:30-71` which guards every `indicators.get(...) is None`. Fix: mirror outcome_tracker's safe-default pattern in `generate_signal`.

- `portfolio/accuracy_stats.py:225-228` — `change_pct = outcome.get("change_pct", 0)` followed by `if change_pct is None: null_change_pct_skipped += 1`.
  ```python
  change_pct = outcome.get("change_pct", 0)
  if change_pct is None:
      null_change_pct_skipped += 1
  ```
  `.get("change_pct", 0)` returns `0` when the key is missing, never `None`. The None branch only catches *explicitly-null* values. A row where the key is missing entirely becomes 0, which `_vote_correct` (line 182) maps to `None` (below `_MIN_CHANGE_PCT=0.05`) so the row is skipped — but `null_change_pct_skipped` never increments and the data-quality alert (line 249) never fires. The reviewer comment promises to surface "drift" but the gate is wrong-direction.

- `portfolio/outcome_tracker.py:262-276` — `_fetch_historical_price` does direct `yf.Ticker(...).history()` for any `YF_MAP` ticker (i.e. stocks) with no per-call timeout. `_yfinance_limiter.wait()` rate-limits but the HTTP call itself can hang (yfinance has no default network timeout). Inside `backfill_outcomes` this is called in a tight for-entry-for-horizon loop with no outer wallclock budget — one hung call stalls the entire backfill, which runs on a daily Scheduled Task (`PF-OutcomeCheck`). Fix: wrap yfinance in `subprocess_utils` or pass `session=requests.Session(timeout=15)` via yfinance's internal `_requests` shim.

- `portfolio/ic_computation.py:127-128` — `ic_buy` / `ic_sell` are misnamed: they're not Spearman correlations, they're *mean returns* of BUY-voted and (negated) SELL-voted outcomes.
  ```python
  ic_buy = sum(buy_returns) / len(buy_returns) if buy_returns else 0.0
  ic_sell = -sum(sell_returns) / len(sell_returns) if sell_returns else 0.0
  ```
  The field is consumed by `signal_engine._weighted_consensus` (search: `ic_mult`) under the assumption that "IC" measures correlation strength. Mean return scales with market drift (a bull-leaning ticker inflates `ic_buy` even for noisy signals); correlation does not. Result: BUY signals on perma-bull tickers get a free weight bonus.

- `portfolio/signal_state_since.py:54` — `vote_str = str(vote or "HOLD").upper()` coerces any falsy vote ('', 0, None) to "HOLD". If a signal returns a literal empty string as a valid HOLD reason (some legacy paths do this), the dashboard heatmap "since" timer resets to now every cycle, masking real state-change history. Quote check: prior `prev_entry["vote"]` was stored as "HOLD" via the same coercion, so cycle N reads "HOLD" prev and emits "HOLD" current — that matches. Risk is if any upstream returns lowercase "hold" or "buy". The check `prev_entry.get("vote") == vote_str` is case-strict; coercion is only one-direction. Defensive but inconsistent: also normalize prev.

- `portfolio/signal_engine.py:954-956` — `MIN_VOTERS_METALS = 2` violates `.claude/rules/signals.md`: "MIN_VOTERS = 3 for all asset classes". Doc drift introduced 2026-05-11. If the rule is the policy, code is wrong (lowered metals quorum increases false-positive trades). If the code is the policy, the rule file lies. Either fix; do not leave both.

## P2 — concerns / smells (worth addressing)

- `portfolio/signal_engine.py:97-114` — `_reset_phase_log` eviction `evict_count = len(_phase_log_per_ticker) // 2` halves the dict every time the cap is hit. Once it hits 64, the *next* call drops 32 entries — for a 5-ticker production load this never matters, but a test or probe that warms 65 tickers and then alternates the original 5 will repeatedly hit the eviction. Cheap, but the constant-time amortized claim depends on the workload pattern.

- `portfolio/accuracy_stats.py:148-153` — `load_entries` SQLite-first fallback to JSONL only when `count == 0`. If SQLite has *some* rows but is missing the last day of inserts (per the prior 2026-05-04 finding still flagged as P1), the JSONL tail is ignored. The 2026-05-04 fix never went in.

- `portfolio/signal_engine.py:2807-2811` — `_compute_adx` catches all exceptions, caches `None`, logs a warning. A persistent ADX failure (e.g. corrupted DF column) results in a one-time warning then silently votes HOLD via downstream `adx is None` checks forever. Add a counter so repeat-failure transitions to ERROR.

- `portfolio/outcome_tracker.py:165-166` — SQLite `db.insert_snapshot()` failure is `logger.warning(...)` only. If SQLite goes corrupt, JSONL keeps writing but SignalDB lags; the warning is per-row spam and rotates out of any visible window. Better: track failure count and elevate to critical_errors.jsonl after N.

- `portfolio/ic_computation.py:130-136` — `_rolling_ic` returns `[]` when `len(votes) < window=50`; downstream `icir = 0.0`. Many production signals will sit at <50 votes for weeks after deployment; their `icir` is implicitly "no information" but consumers can't tell that from a real `icir=0` (genuinely flat IC). Distinguish "insufficient data" with a sentinel (or `None`).

- `portfolio/forecast_accuracy.py:357-368` — `_lookup_price_at_time` iterates the FULL price_snapshots_hourly.jsonl every call to find the closest snapshot. Backfill of N predictions × 2 horizons = 2N full scans of an ever-growing file. With current 7+ months of hourly data, this is ~5000 lines × 2N parses per cycle. Build an in-memory index once per backfill call.

## Did NOT find

1. **Silent failures**: covered — see P0/P1 findings above (forecast schema mismatch; null_change_pct gate; yfinance no-timeout). No new silent except besides those.
2. **Race conditions**: signal_log lock at `outcome_tracker.py:282-340` is correct (sidecar lockfile, Windows msvcrt + POSIX fcntl, snapshot+rewrite preserves concurrent appends). ADX cache + phase log + last-signal tracker all hold `threading.Lock()` properly.
3. **Money-losing bugs**: no direct order math in this subsystem.
4. **State corruption**: backfill_outcomes Phase 3 (lines 514-563) snapshots size, releases lock, re-acquires before rewrite, preserves concurrent-tail bytes verbatim — correct pattern. No torn writes seen.
5. **Logic errors that pass tests**: forecast schema mismatch (P0) IS one — `compute_forecast_accuracy` produces empty scores that tests may not assert > 0 on.
6. **Resource leaks**: SignalDB close at backfill exit (line 506) and load_entries' try/finally (line 154-156) both correct. yfinance leaves no handles.
7. **Time/timezone bugs**: ts handling in accuracy_stats uses ISO strings consistently; `datetime.now(UTC)` everywhere.
8. **API misuse**: Binance uses `1h` interval at outcome_tracker.py:219, 240 — correct (not `10m`).
9. **Trust boundary violations**: signal_log entries are JSON-encoded; no string interpolation into shell/sql.
10. **Incorrect partial-state assumptions**: covered by P1 finding on direct `ind[...]` access.
