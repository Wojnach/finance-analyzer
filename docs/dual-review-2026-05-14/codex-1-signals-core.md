# Adversarial Review — 1 signals-core (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/forecast_accuracy.py:254-329` — `backfill_forecast_outcomes` silently truncates predictions file when `max_entries` cap is hit
  The loop iterates ALL entries and appends every one to `modified_entries` (line 320, outside any conditional), but the break at line 322 fires **before** the remaining entries get appended:
  ```python
  modified_entries.append(entry)

  if updated >= max_entries:
      break
  ```
  Then line 327 calls `_write_predictions(modified_entries, path)` which does `atomic_write_jsonl(path, entries)` — overwriting the file with only the entries iterated up to the break point. If `forecast_predictions.jsonl` has 2,000 entries and the 500th update completes, **the file shrinks from 2,000 lines to ~500**. The "max_entries" was clearly intended as an update cap, not a truncation cap. Production impact: any partial backfill silently destroys all subsequent unprocessed predictions — and the function is callable from operator scripts.

- `portfolio/forecast_accuracy.py:142,159` — Missing `change_pct` silently biases all BUY predictions to "wrong"
  ```python
  actual_change = outcome.get("change_pct", 0)
  ...
  predicted_up = vote == "BUY"
  actual_up = actual_change > 0
  ```
  When an outcome dict exists but has no `change_pct` key (or the backfiller wrote a partial entry), the default of `0` makes `actual_up = False`. Every BUY prediction with a missing/zero change_pct is recorded as **incorrect** and every SELL as **correct**. Unlike `accuracy_stats._vote_correct`, this path has no None-guard and no `_MIN_CHANGE_PCT` neutral band. The accuracy_degradation tracker reads this output via `cached_forecast_accuracy`, so a data-quality regression in the forecast outcome writer would silently degrade Chronos accuracy, possibly triggering false "model decay" CRITICAL alerts and ultimately disabling a working model. Same silent-failure shape as the 3-week "Not logged in" outage.

## P1 — high-confidence bugs (should fix)

- `portfolio/forecast_signal.py:97` — Bare `except (ImportError, Exception)` silently downgrades Chronos-2 → v1 on any failure
  ```python
  try:
      from chronos import Chronos2Pipeline
      ...
  except (ImportError, Exception) as e:
      logger.info("Chronos-2 not available (%s), falling back to v1", e)
  ```
  The redundant `(ImportError, Exception)` catches **every** non-system error (OOM, CUDA fault, bad model path, transient HF download failure) and silently drops to v1 with only an INFO log. Operations have no signal that the preferred model is dead. This is the exact "subprocess exits 0 while printing failure" pattern that produced the 3-week Layer 2 outage. The wrong-model path then keeps running and degrading forecast accuracy until somebody hand-checks. Should be `except ImportError` only; surface other errors loudly.

- `portfolio/forecast_signal.py:349-352` — `_load_candles` returns None → DEBUG log, forecast silently disabled
  ```python
  prices = _load_candles(ticker)
  if not prices or len(prices) < 50:
      logger.debug("Skipping %s: insufficient candle data (%d)", ticker, ...)
      continue
  ```
  `_load_candles` already swallows all exceptions at line 60-61 with `logger.debug`. If Binance has an outage or the data_collector hits a config bug, every ticker silently skips with no WARNING, no Telegram alert, no health journal entry. The forecast pipeline then produces zero predictions but logs nothing visible. The `data/forecast_predictions.jsonl` file simply stops growing. Should be `logger.warning` + a health.update_signal_health failure record.

- `portfolio/signal_db.py:288,319,347,387` — Outcome correctness checks fail open when `change_pct == 0`
  Every `signal_accuracy`, `consensus_accuracy`, `per_ticker_accuracy`, and `ticker_signal_accuracy` SQL helper uses:
  ```python
  if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
      stats[sig_name]["correct"] += 1
  ```
  When `change_pct` is exactly 0 (a real outcome can be a flat close), neither branch fires — the vote counts as a sample but **never** as correct, regardless of direction. This systematically depresses every signal's accuracy on flat-close outcomes and is inconsistent with the JSONL path in `accuracy_stats._vote_correct`, which applies a `_MIN_CHANGE_PCT = 0.05` neutral band and returns None (skip). Same SQLite DB feeds the dashboard `/api/accuracy` — discrepancy between two paths reading the same underlying outcomes.

- `portfolio/llm_calibration.py:117`, `portfolio/llm_outcome_backfill.py:121,176,275` — Non-atomic `path.read_text().splitlines()` over actively-appended JSONL
  All four call sites read JSONL files via `read_text(encoding="utf-8").splitlines()` instead of `file_utils.load_jsonl`. These files are written concurrently by `atomic_append_jsonl` (probability log) and outcome backfill. A torn write (e.g., process crash mid-newline) produces a malformed final line — the readers all silently `continue` on `JSONDecodeError`, masking the corruption. `accuracy_stats.py:1573-1581` already documents the same anti-pattern was fixed there per CLAUDE.md rule 4; this leaf code re-introduced it. Migrate to `file_utils.load_jsonl`.

- `portfolio/outcome_tracker.py:273` — Stock outcome backfill compares UTC date to yfinance index date (potential off-by-one)
  ```python
  target_dt = datetime.fromtimestamp(target_ts, tz=UTC)
  ...
  target_date = target_dt.date()
  candidates = h[h.index.date <= target_date]
  ```
  `h.index` is a yfinance `DatetimeIndex`. For US stocks, yfinance daily bars are indexed in NY exchange time, not UTC. `target_date` is the UTC calendar date. When `target_ts` falls in the 03:00-08:00 UTC window (Eastern evening of prior day), the UTC date is one calendar day ahead of the NY date the bar is keyed by — the filter can either silently match an extra future bar (look-ahead) or drop the legitimate target bar. Affects MSTR outcome accuracy specifically.

- `portfolio/forecast_signal.py:161-179, 224-239, 299-315` — Confidence formula can briefly exceed 1.0 before `min(..., 1.0)` clamp on tiny `current_price`
  ```python
  confidence = min((low - current_price) / current_price * 10, 1.0)
  ```
  For a ticker with near-zero `current_price` (illegal state but possible during a Binance maintenance window — see `indicators.py:42-47` already guards this), the numerator dominates and the inner expression overflows. Stage 1 confidence cap is fine, but the divide-by-zero on `current_price = 0` is unguarded and would crash forecast for that ticker (uncaught — bubbles up to `forecast_chronos`'s broad except, silently kills the forecast). The same pattern repeats in v1 / v2 / Prophet code paths.

## P2 — concerns / smells (worth addressing)

- `portfolio/ticker_accuracy.py:130-131` — `direction_probability` silently drops contrarian-informative signals
  ```python
  if accuracy < 0.50:
      continue
  ```
  A signal that's 30% accurate is informative (inverse). The code just skips it. Combined with line 136 `p_up = 1.0 - accuracy` (which uses the combined BUY+SELL aggregate), a signal that's 70% on BUY but 30% on SELL gets treated identically per-direction — directional asymmetry is invisible. This drives Mode B Telegram probabilities and Kelly sizing per CLAUDE.md.

- `portfolio/meta_learner.py:42,402-408` — `_model_cache` accessed without a lock
  ```python
  _model_cache: dict[str, tuple] = {}
  ...
  cached = _model_cache.get(horizon)
  if cached and cached[1] == mtime:
      model = cached[0]
  else:
      model = joblib.load(model_path)
      _model_cache[horizon] = (model, mtime)
  ```
  Five ticker threads call `predict()` concurrently. Two threads simultaneously seeing a cold cache will both invoke `joblib.load`, then race on the dict write — last-writer-wins (benign) but wastes ~200MB of duplicate deserialization and burns startup time. Same pattern in `ml_signal.py:17-27` (`_model_cache` / `_pred_cache` — neither guarded). Documented locking convention elsewhere in the project; these two were missed.

- `portfolio/forecast_signal.py:196` — `pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")` invents timestamps for the context window
  Chronos-2's `predict_df` is fed timestamps anchored on current wall clock — not the actual candle close times the prices came from. If the candle data is stale (data_collector returned cached prices), the prediction is mis-anchored by hours. Probably harmless because Chronos cares about price values not timestamps, but a future model upgrade could regress silently.

- `portfolio/signal_engine.py:3437,3390,3343` — Narrow `except ImportError` for LLM and macro signal blocks
  Each of these protects only against missing optional dependencies; any other failure (e.g., `_cached_or_enqueue` raising a TimeoutError) escapes the signal-level guard and propagates up. The outer dispatch loop only re-catches inside the enhanced-signal section (line 3531-3563); the LLM blocks are run BEFORE that. A transient Ministral / qwen3 / macro_context failure thus aborts the entire `generate_signal()` call for that ticker, losing all subsequent enhanced signals — which is silent at the call site (caller logs "signal cycle failed for X" once).

- `portfolio/ml_signal.py:121, 152-155` — Hourly klines fetched without explicit interval validation; predict uses last row directly
  Calls Binance with `interval=1h, limit=100`. `compute_features` derives `hour_sin / hour_cos` from `pd.to_datetime(df["open_time"], unit="ms")` (naive UTC) — fine for training-inference symmetry. But the inference path silently ignores partial last-bar issues: the most recent 1h kline returned by Binance is the **in-progress** candle until the hour closes. `last_row = features.iloc[[-1]].values` uses that in-progress candle for prediction; ml_classifier was trained on closed bars. This is a subtle look-ahead / regime mismatch on every prediction. Add `df = df.iloc[:-1]` or filter `close_time < now`.

- `portfolio/signal_engine.py:97` (`import json`) and `portfolio/signal_decay_alert.py:12` — unused `import json` (decay_alert never uses it). Minor; flagged because the broader review noted dead imports in the orchestration layer.

## Did NOT find

1. **Silent failures** — found four (Chronos-2 fallback, _load_candles, forecast_accuracy default zero, narrow ImportError). The pattern is widespread but localized to forecast/LLM paths; non-LLM accuracy code is mostly well-guarded.
2. **Race conditions** — `meta_learner._model_cache` and `ml_signal._model_cache` unprotected, but the docstrings explicitly state most paths are single-threaded or use existing locks. `signal_history.update_history` correctly holds `_history_lock` across read-modify-write.
3. **Money-losing bugs** — no wrong-sign PnL, no stop-loss / fee math, no order placement in this subsystem (all signal-analysis layer).
4. **State corruption** — `outcome_tracker.backfill_outcomes` rewrite phase looks correct (head bytes + parsed entries + concurrent_tail_bytes preserved under lock); but the `forecast_accuracy.backfill_forecast_outcomes` truncation (P0) is the same family of bug.
5. **Logic errors that pass tests** — `signal_db.py` `change_pct == 0` family qualifies (P1); also flagged the `ticker_accuracy` direction_probability < 50% drop (P2). Test suites likely mock outcomes with non-zero change_pct.
6. **Resource leaks** — `SignalDB._get_conn` keeps one persistent sqlite3 connection per instance with WAL; close() is paired in `accuracy_stats.load_entries` and `outcome_tracker.backfill_outcomes`. No leaks found in normal paths; subprocess/file-handle leaks not present in this subsystem.
7. **Time/timezone bugs** — found one (yfinance NY date vs UTC date in outcome_tracker, P1). Also: `forecast_signal.forecast_prophet:273` uses `datetime.now(UTC).replace(tzinfo=None)` to strip tz (Prophet requirement) — works but loses the original timestamp; acceptable. `meta_learner._build_features:149` uses `.replace("Z", "+00:00")` which is fine.
8. **API misuse** — searched for Binance 10m interval, none found. All callers use 1h / 5m / 15m. Alpaca pagination not used in this subsystem.
9. **Trust boundary violations** — `signal_registry.load_signal_func` uses `importlib.import_module(entry["module_path"])` but the module_path comes only from the hardcoded `_register_defaults()` table, never user input. The semgrep suppression comment is accurate.
10. **Incorrect assumptions about partial state** — `accuracy_stats.signal_accuracy:225` was a candidate (`outcome.get("change_pct", 0)`) but the downstream `_vote_correct` handles None; the explicit `null_change_pct_skipped` counter at line 249 surfaces drift. `meta_learner._build_features:128` defensively handles non-string `signals` field. `signal_state_since.update_state_since` defensively re-validates every level of nested dict shape.
