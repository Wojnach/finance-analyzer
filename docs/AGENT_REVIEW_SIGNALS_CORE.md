# Adversarial Review: signals-core (Agent Findings)

Reviewer: code-reviewer subagent (fgl whole-codebase audit, fresh pass)
Date: 2026-05-26
Repo: `Q:\finance-analyzer` (live, post-commit 537bca18)
Scope: 16 files, ~12,000 lines.

Prior pass: 2026-05-24 (`AGENT_REVIEW_SIGNALS_CORE.md`). Findings marked
[REPEAT] are still present; [RESOLVED] are confirmed fixed in this pass.

---

## P0 — Money / correctness

portfolio/signal_engine.py:2720,2742: P0: bias double-application STILL not fixed. `norm_weight = act_data["normalized_weight"]` (line 2720) already contains the symmetric `bias_penalty = max(1-bias, 0.1)` baked in by `signal_activation_rates()`; then `_resolve_bias_penalty(signal_bias)` multiplies again at 2742 for the bias-direction branch. Contrarian SELL on a 100%-BUY calendar signal gets the 0.1x legacy penalty (NOT "full weight" as the docstring claims); bias-direction BUY gets `0.1 * 0.2 = 0.02x`. The 2026-05-19 fix at `_resolve_bias_penalty` was a one-site patch; the legacy `bias_penalty` in `accuracy_stats.signal_activation_rates()` is still live. [REPEAT]. Fix: set `bias_penalty=1.0` (and rebake `normalized_weight = rarity_weight`) in `accuracy_stats.py:840-849`; the directional tier at signal_engine.py:2742 is then the sole bias application.

portfolio/accuracy_stats.py:152-164: P0: SQLite is silently preferred even when stale. `load_entries()` returns the SQLite contents when `db.snapshot_count() > 0`; falls back to JSONL only on `count == 0`. If the SQLite dual-write at `outcome_tracker.py:163-166` ever swallows an exception (it does — broad `except Exception` with `logger.warning`), JSONL keeps appending while SQLite stalls. Every downstream accuracy/utility/postmortem path then reads stale data forever — no error surfaced. Fix: compare `db.snapshot_count()` to a cheap JSONL line count (sidecar metadata) and fall back to JSONL when SQLite trails by more than e.g. 100 rows; emit a critical_errors entry on the divergence.

portfolio/outcome_tracker.py:160-166: P0: SQLite dual-write exception silently logged as warning. JSONL append already succeeded at line 157; if SQLite fails (disk full, corruption, locked file from dashboard reader), the loop continues. Combined with accuracy_stats.py:152's SQLite-preferred read, the system enters a "JSONL appends, SQLite stalls, reads are stale" trap. Fix: on dual-write failure, write a critical_errors.jsonl row (category="sqlite_dual_write_failed") so the auto-spawn fix agent escalates instead of the failure being buried as a debug log.

portfolio/signal_decay_alert.py:62-92: P0: silent failure when accuracy_cache.json schema drifts. `recent_data = cache.get(recent_key, {})` returns `{}` if `f"{horizon}_recent"` is missing or renamed; the `for signal in recent_data:` loop is then a no-op and the function returns `[]`. Operators see "no decay detected" forever even when accuracy has collapsed. The Apr-2026 cache-key rename (BUG-133 added per-horizon timestamps) would have triggered this if `_recent` had been touched. Fix: detect schema absence (`if not recent_data and cache: log critical`) and emit a CRITICAL log + critical_errors row when the cache exists but the expected keys don't.

portfolio/accuracy_stats.py:163: P0: JSONL fallback silently truncates to last 50K entries. `load_jsonl_tail(SIGNAL_LOG, max_entries=50000)` is the fallback path; if the SQLite branch is unavailable AND signal_log.jsonl has grown past 50K, every accuracy computation (signal_accuracy, signal_utility, signal_activation_rates, blend_accuracy_data via `_BLEND_DEFAULT_MIN_RECENT_SAMPLES`) loses earlier history. The `all-time` accuracy becomes a rolling 50K window without anyone noticing. Fix: when the SQLite path errors AND JSONL line count > 50000, emit a one-time WARNING with the count and a critical_errors row; or expose `max_entries=None` for full reads in non-hot paths.

---

## P1 — Silent failure / loop crash

portfolio/accuracy_stats.py:970-974: P1: blend_accuracy_data double-counts directional samples. `total_buy/total_sell` summed as `at_v + rc_v` while `recent` is a strict subset of `alltime`. Downstream `_DIRECTIONAL_GATE_MIN_SAMPLES=30` and `_DIR_WEIGHT_MIN_SAMPLES=20` decisions get inflated counts. [REPEAT]. Fix: use `max(at_v, rc_v)` to match the overall total combining rule at line 937, or treat recent as overlay (use rc when rc >= min_recent else at).

portfolio/signal_engine.py:4322: P1: `soft_confidences=extra_info` passes the entire ~70-key dict. Lookup is `soft_confidences.get(f"_soft_conf_{signal_name}")` — works today, but every new contributor sees `soft_confidences` as a parameter named for soft confidences while it's actually the whole context dict. Any future signal whose name shadows an existing `_soft_conf_*` key collides silently (no assert, no warning). Fix: extract only the `_soft_conf_*` keys at the boundary: `soft_confidences={k: v for k, v in extra_info.items() if k.startswith("_soft_conf_")}`.

portfolio/ticker_accuracy.py:38: P1: uncached full signal-log scan per call. `accuracy_by_ticker_signal` calls `load_entries()` (50K SQLite rows) on every invocation. `direction_probability` calls this once per horizon; `get_focus_probabilities` calls it 3 horizons × 5 tickers per Mode-B refresh = 15 cold scans. No L1/L2 cache while `signal_utility` and `regime_accuracy` have aggressive caches. [REPEAT]. Fix: wrap with double-checked locking like `get_or_compute_accuracy`; TTL 600-1800s.

portfolio/signal_history.py:30: P1: cross-process race despite the module-level lock. The `_history_lock` serializes threads within ONE process; main loop, dashboard, crypto_loop, oil_loop, mstr_loop all import this module and write the same `signal_history.jsonl`. Last-writer-wins across processes, persistence scores corrupt during cycle overlap. [REPEAT]. Fix: switch to SQLite WAL (matches signal_log.db pattern) or hold the sidecar lockfile that `atomic_append_jsonl` already provides.

portfolio/signal_db.py:25-37: P1: SQLite connection cached without `check_same_thread=False`. `_get_conn()` lazy-creates `self._conn`. Production callers construct per-call instances (latent), but ThreadPoolExecutor + a future caller that reuses one DB across threads would raise `ProgrammingError`. No lock guards the connection either. [REPEAT]. Fix: pass `check_same_thread=False` + a threading.Lock around `_get_conn()`/`execute()` callsites, OR document the per-thread contract in the class docstring.

portfolio/cusum_accuracy_monitor.py:104: P1: throttle "n > last_alert_n + 10" couples to outcome batch size, not wall-clock time. If `outcome_tracker.backfill_outcomes()` processes a 50-outcome burst, n increments 50× in one second; the +10 throttle fires every 10 outcomes — 5 alerts per backfill. No COOLDOWN_S floor. [REPEAT]. Fix: add `COOLDOWN_S = 24 * 3600` matching `accuracy_degradation.COOLDOWN_PER_SIGNAL_S` and gate `last_alert_ts` time before firing.

portfolio/signal_decay_alert.py:74-77: P1: decay check silently quiet when recent samples are under 50. If outcome backfill is broken (BUG in outcome_tracker, network outage, SQLite stall), `recent_total` stays low → `continue` → no alert. Operators see "no decay detected" while the OUTCOME PIPELINE is dead and the actual decay is invisible. Fix: log INFO once per horizon when zero signals had ≥`_MIN_RECENT_SAMPLES`, and emit a critical_errors entry if the same horizon has been sub-threshold for 7+ days.

portfolio/outcome_tracker.py:464-465: P1: yfinance/Binance fetch errors are silently cached as None. `except Exception: price_cache[cache_key] = None` — a single transient `requests.Timeout` poisons the cache for that `(ticker, minute)` for the lifetime of the function. The next 50 outcome rows for the same ticker minute all skip silently (line 467-469 `if hist_price is None: continue`). Fix: don't cache failures; cache only successful prices. Cache hits should re-fetch on `None`.

portfolio/accuracy_degradation.py:285-302: P1: detector goes dark silently when no 14d-format snapshot exists. `_find_baseline_snapshot` returns None during the 13d transition window (premortem F2). The INFO log at line 408-413 fires once per check, but a 14d gap (snapshot writer regression, disk full, schema rename) produces the same "no baseline" return — INDISTINGUISHABLE from healthy transition. After 14d the detector should NOT be dark. [REPEAT]. Fix: emit a Telegram WARNING after 14d elapsed since the last successful baseline match; track `last_baseline_match_ts` in `degradation_alert_state.json`.

portfolio/signal_engine.py:2278: P1: shadowed `import math as _math` inside `_weighted_consensus`. Line 3097 separately imports `import math as _math_ent` for the entropy guard. Inconsistent aliases mean a future "consolidate imports" patch silently picks one alias and the other use-sites NameError at runtime. [REPEAT]. Fix: hoist `import math` to module top.

portfolio/outcome_tracker.py:155-157: P1: broad `except Exception` on SQLite write swallows partial-rollback state. When line 155 raises, `conn.rollback()` may itself fail (locked, disk full) — the re-raise then propagates with a half-rolled-back state. Caller in `log_signal_snapshot` swallows the exception entirely at line 165. Fix: catch sqlite3.OperationalError explicitly and emit critical_errors; let other exceptions propagate to surface them via the standard tracebacks.

portfolio/accuracy_stats.py:1577-1589: P1: `_load_accuracy_snapshots` silently drops malformed lines via `load_jsonl`. If the snapshot writer wrote a partial line (atomic_append_jsonl race? full-disk?), the line is silently skipped; `_find_baseline_snapshot` returns None; `check_degradation` returns []; operators see clean checks. Fix: log WARNING when load_jsonl drops a line (file_utils may already do this at debug — promote to warning when caller is `_load_accuracy_snapshots`); emit critical_errors if same-day snapshot is missing post-writer-success.

portfolio/train_signal_weights.py:73: P1: `outcome.get("change_pct")` will AttributeError if outcome is somehow a list or string. The defensive `isinstance(outcome, (int, float))` check covers numeric variants but a corrupt JSONL line that decodes to a list slips through. `train_weights()` then crashes mid-iteration. Fix: tighten to `isinstance(outcome, dict)` for the .get path; treat anything else (including int/float OR truly bad data) as missing.

---

## P2 — Perf / observability

portfolio/signal_postmortem.py:148-163: P2: O(entries × tickers × signals²) per call. 5K entries × 5 tickers × ~435 signal pairs = ~11M dict ops per `compute_vote_correlation()`. No caching. [REPEAT]. Fix: cache pairwise agreement matrix with L1+L2 TTL (1h safe, outcomes update daily).

portfolio/signal_engine.py:36-37: P2: `_adx_cache` content-keyed by `(len, first_close, mid_close, last_close, high_max, low_min)` floats. Float equality across distinct dataframes with identical OHLC at round numbers collides; ADX returns a stale value. Low probability per ticker; aggregate exposure is real. [REPEAT]. Fix: include a content hash of >6 bars (e.g. `xxhash(df.values.tobytes())`).

portfolio/accuracy_stats.py:1605-1615: P2: `_find_snapshot_near` indent bug on lines 1611-1613 is harmless today but the double-indented `best = snap; best_delta = delta` lives INSIDE the `if delta <= max_delta_hours` block; if a future refactor flattens it, the `(best_delta is None or delta < best_delta)` short-circuit moves outside the time-window check. Fix: explicit `if delta <= max_delta_hours: if best_delta is None or delta < best_delta: best = snap; best_delta = delta`.

portfolio/signal_engine.py:1017-1020,2403: P2: `MACRO_WINDOW_FORCE_HOLD_SIGNALS` is a single-element frozenset of an already-disabled signal (claude_fundamental). The set + four call sites (signal_engine.py:2403, 4060-4063, plus the two `_topn_accuracy_key`/`_leader_accuracy_key` overlays) is dead code. [REPEAT]. Fix: delete the set + call sites OR mark with `# scaffold:` comment + add a regression test that prevents accidental cleanup.

portfolio/signal_registry.py:382: P2: `_register_defaults()` runs at module import time. If ANY of the 59 default `register_enhanced` calls is missing args or raises, the entire `from portfolio.signal_registry import get_enhanced_signals` import fails — signal_engine then has NO signals and silently degrades to HOLD on every cycle. The dispatch loop at signal_engine.py:3657 iterates `_enhanced_entries.items()` which is empty, no warning. Fix: wrap each register call in try/except + log warning; or pull into a `_register_defaults_safe()` that batches errors and surfaces them at first dispatch.

portfolio/outcome_tracker.py:392-397: P2: JSONL parse swallows `json.JSONDecodeError` silently with `continue`. If signal_log.jsonl has a torn line from a concurrent appender (the sidecar lock should prevent this, but cross-process races exist with crypto_loop/oil_loop), the entry is dropped without any log. Combined with `accuracy_stats.py:152` SQLite preference, the dropped entry never makes it to the accuracy pipeline. Fix: log WARNING with the line number and first 200 chars; emit critical_errors after N dropped lines per cycle.

portfolio/cusum_accuracy_monitor.py:122,142: P2: `alerts_list[-100:]` truncates from the END but the list grows monotonically. After 100 outcomes the OLDEST alert is dropped, but during a sustained regime break (10 signals each firing 5× across 50 outcomes), the alerts list churns and operators reviewing `data/cusum_accuracy_state.json` see only the most recent 100. Fix: bump to 1000 or rotate to a separate JSONL append-only file like signal_decay_alerts.jsonl.

portfolio/signal_engine.py:36,46,82: P2: three module-level dicts (`_adx_cache`, `_last_signal_per_ticker`, `_phase_log_per_ticker`) all carry separate locks and separate eviction caps. The phase log has `_PHASE_LOG_MAX_TICKERS = 64` but `_last_signal_per_ticker` has NO cap. Tests/probes that pass arbitrary ticker names slowly leak. [PARTIAL-REPEAT]. Fix: add the same 64-key LRU prune to `_set_last_signal`.

portfolio/signal_history.py:97: P2: `trimmed.sort(key=lambda e: e.get("ts", ""))` lexically sorts ISO-8601 timestamps. Works when `datetime.now(UTC).isoformat()` is the only writer (consistent `+00:00` suffix in Python 3.11+), but a future ts written with `datetime.now().isoformat()` (naive) has no suffix and sorts BEFORE all UTC entries. Fix: parse with `datetime.fromisoformat(e["ts"])` and sort by datetime, or assert UTC suffix at write time.

portfolio/signal_postmortem.py:138-139,222-223,228-229: P2: three broad `except Exception` paths return `[]` / log warning. `generate_postmortem()` then writes `correlations=[]` to disk — operators reading data/signal_postmortem.json see "no correlations" without distinguishing "no correlations found" from "correlation analysis crashed". Fix: include a `_errors: [...]` field in the report when any branch failed, so consumers can see the analysis was incomplete.

portfolio/accuracy_degradation.py:786: P2: `_record_snapshot_writer_silent_failure` only fires when the snapshot file SIZE doesn't grow. If `save_full_accuracy_snapshot` writes a malformed JSON line (partial write, truncated dict), the file grows but the line is unparseable. `_load_accuracy_snapshots` drops it silently, `_find_baseline_snapshot` returns None, detector goes dark. Fix: read the just-appended line back and json.loads-validate before marking the snapshot as successful.

portfolio/signal_engine.py:1649-1740: P2: `_compute_dynamic_correlation_groups` walks entire 30-day signal log on every TTL miss (~5K snapshots × 5 tickers × ~30 active signals = ~750K rows). TTL hides cost; a cold restart pays it once per horizon per cycle. [REPEAT]. Fix: cache the pandas DataFrame in L1, recompute only when row count grows past a delta threshold.

portfolio/accuracy_stats.py:937: P2: `total = max(at_samples, rc_samples)` combining rule differs from `total_buy/total_sell = at+rc` (P1 above). The asymmetry produces visibly inconsistent rows (`total=10000` and `total_buy=220` impossible by inspection). [REPEAT]. Fix: post P1 directional fix, document combining rules in docstring.

---

## Top 3

1. **outcome_tracker.py:160-166 + accuracy_stats.py:152** — the SQLite dual-write/preferred-read combo silently strands the entire accuracy pipeline on stale data when SQLite errors. JSONL keeps growing, SQLite stops, every gate (47%/50% accuracy, 40% directional, recency blend, IC, utility boost, postmortem) reads from a frozen snapshot. Fix path: emit critical_errors on dual-write failure AND fall back to JSONL on SQLite/JSONL divergence.

2. **signal_engine.py:2720,2742** — bias-penalty double-application is STILL not fixed two reviews later. The 2026-05-19 single-site patch landed only at `_resolve_bias_penalty`; the legacy `bias_penalty` in `accuracy_stats.signal_activation_rates()` continues to symmetric-penalize contrarian votes. Fix is one line: `bias_penalty=1.0` in accuracy_stats.py:840 + rebake `normalized_weight = rarity_weight` so the directional tier at signal_engine.py:2742 is the sole bias application.

3. **signal_decay_alert.py:62-92** — the decay alerter has TWO silent-failure modes: schema-drift in accuracy_cache.json (cache.get returns `{}`, loop is no-op) AND outcome-backfill breakage (recent_total < 50 perpetually, loop continues). Both produce "no decay detected" exactly when decay is most dangerous. Add a CRITICAL log + critical_errors entry on either condition.
