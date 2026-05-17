# Supporting Modules Subsystem Audit — 2026-05-17

Scope: catch-all supporting modules — precomputes, prophecy, fin_evolve, qwen3 trader,
reflection, digests, vector memory, seasonality, analyze, backtester, fish helpers,
tinylora_trainer, etc. Per audit conventions: code-only review, runtime state files
NOT read.

## Critical Findings

portfolio/backtester.py:93,141-147: 🔴 critical: **look-ahead bias in backtest**. `_build_accuracy_data(horizon)` is called once outside the entry loop (L93), then the SAME present-day accuracy table is passed to `_weighted_consensus(...)` for every historical entry (L141). For an entry from 30 days ago, the "new system" uses today's signal-accuracy figures — figures that have absorbed all subsequent outcomes including the one being scored. Result: the reported "Old vs New" delta is structurally inflated. A correct walk-forward backtest must rebuild `accuracy_data` from entries strictly before the current entry's timestamp. As-is, `print_report` prints a number the system literally could not have known at decision time, and the `--days` cutoff doesn't fix it because the accuracy snapshot still reflects post-cutoff data. The 3-line block:

```
    # Pre-build accuracy_data for new consensus (based on the target horizon)
    accuracy_data = _build_accuracy_data(horizon)
```

(comment alone admits the structure: "Pre-build … for new consensus") guarantees every iteration's gate sees future-information accuracy.

portfolio/tinylora_trainer.py:19,32: 🔴 critical: **DST-unaware "CET" timezone hardcoded as UTC+1**. `CET = timezone(timedelta(hours=1))` is a fixed offset — not `zoneinfo.ZoneInfo("Europe/Stockholm")`. During CEST (late Mar – late Oct, ~7 months/year) Sweden is UTC+2, so `datetime.now(CET)` returns a clock that is 1h behind actual local wall time. The training-window check (L42-44, `hour < 8 or hour >= 22`) consequently fires from 09:00 local in summer instead of 08:00, and stops at 23:00 local instead of 22:00. The docstring at L18 even admits "simplified; DST handled separately in production" — but `is_training_allowed` IS the production gate (called from `__main__` L166). Practical impact: TinyLoRA training spawns into market-open windows for an hour every CEST morning, contending with the main loop for the same CPU and (if extended) the same GPU lock that Chronos/Qwen3 use. Fix: `CET = ZoneInfo("Europe/Stockholm")` from the stdlib `zoneinfo` module (already used in daily_digest.py).

## Risk Findings

portfolio/regime_alerts.py:120-123: 🟡 risk: naive-aware datetime mix silently drops history. `datetime.fromisoformat(ts_str)` returns a NAIVE datetime when the original timestamp lacks `+00:00` (some older entries), then `ts >= cutoff` compares against `cutoff = datetime.now(UTC) - timedelta(days=days)` — TypeError gets swallowed by the `except (ValueError, TypeError): continue` at L123. Net effect: entries pre-dating timezone-aware logging silently disappear from `get_regime_distribution`, making it look like that ticker spent zero days in old regimes. The 3-line block:

```
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts >= cutoff:
```

Same bug pattern in `portfolio/weekly_digest.py:36-37` (`_load_jsonl`) and `:73-74` (`_trades_this_week`) — naive timestamps from historical entries get dropped from the weekly summary without warning. Fix: after parsing, `if ts.tzinfo is None: ts = ts.replace(tzinfo=UTC)`.

portfolio/regime_alerts.py:67-78,188-217: 🟡 risk: alert spam from `ranging`/`range-bound` aliasing. `_normalize` collapses the two names for the equality check at L71, but `log_regime_change` (L82-97) and `send_regime_alert` (L149) record the raw input string. Sequence: regime detector returns `"ranging"` → logged → next cycle detector returns `"range-bound"` → check_regime_transition sees they normalize equal, returns None, NO alert sent (good). But `check_and_alert` at L203 reads `_get_last_regime` which returns the raw string. If a third-party caller passes the alternative form back, the journal alternates `ranging`/`range-bound` rows indefinitely. The on-write normalization is missing — store one canonical form.

portfolio/vector_memory.py:117-157,242-244: 🟡 risk: unbounded ChromaDB growth + every-call re-walk. `embed_entries` calls `collection.get()` (L132) which loads ALL existing IDs into RAM on every Layer 2 invocation, then loops the full journal (loaded by `_load_journal_entries` L267 which reads the entire JSONL file line-by-line). With 6 months of 4h digests + per-trigger journals, the dedup set + journal list scales linearly with system age — no pruning, no max-age filter. Vector memory will OOM eventually OR slow Layer 2 dispatch unacceptably as the journal grows. Fix: cap embeddings to last N=2000 entries OR use ChromaDB's `where={"ts": {"$gte": cutoff}}` filter and skip the full `collection.get()` ID load.

portfolio/sentiment_shadow_backfill.py:127,191,321: 🟡 risk: `read_text()` loads entire JSONL file into RAM. `_load_existing_keys` (L127), the main backfill loop (L191), and `compute_model_accuracy` (L321) each call `path.read_text(encoding="utf-8").splitlines()` — loading the full file. The A/B log accumulates indefinitely (rationale section says "Three years' history is already there"). At sustained sentiment-per-cycle write rate, this file will reach hundreds of MB and three full reads per backfill invocation will dominate runtime. Switch to streaming `for line in open(...)` like `tinylora_trainer.collect_training_pairs` does.

portfolio/fish_monitor_smart.py:223-226: 🟡 risk: full-file load to read last line. `metals_log.read_text().strip().split("\n")` slurps the entire metals_signal_log.jsonl on every signal-evaluation tick (every 5min during fishing sessions). For a 100MB+ rolling log this is hundreds of MB/s pure I/O for one line. Use `file_utils.load_jsonl_tail(path, max_entries=1)` — already used by digest.py for the same problem.

portfolio/fish_monitor_smart.py:513: 🟡 risk: naive local datetime in display. `datetime.datetime.now().strftime(...)` uses naive local time. Display-only, but the same value is currently NOT logged — if log_entry at L699 ever adopts the display ts (or copy-paste reuse happens), it would diverge from the timezone-aware `datetime.datetime.now(datetime.UTC).isoformat()` used at L700/L735. Tighten now: `datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S UTC")`.

portfolio/analyze.py:282-289,746-753: 🟡 risk: bypasses claude_gate cost accounting. `run_analysis` and `watch_positions` shell out to `claude -p` directly via `subprocess.run` instead of routing through `portfolio.claude_gate.invoke_claude_text`. `_clean_env()` correctly pops CLAUDECODE (good — no nested-session error), but every `--analyze` and `--watch` run is invisible to `data/claude_invocations.jsonl` token/cost tracking. bigbet.py (L176-182) was explicitly converted to use claude_gate on 2026-05-13 "so the invocation is counted in claude_invocations.jsonl with token+cost accounting (was previously a bypass site, invisible to cost tracking)" — analyze.py is the remaining bypass site. Same `auth-failure check` was added at L296-300, but that's downstream of the invocation, so failed calls still don't get logged to critical_errors with the proper schema.

portfolio/daily_digest.py:30-47: 🟡 risk: read-modify-write race in `_get_last_daily_digest_time`/`_set_last_daily_digest_time`. `_set_last_daily_digest_time` reads the state file, mutates dict, then `atomic_write_json` — but two concurrent callers can both read the pre-update state and one's update is lost. The `should_send_daily_digest` 20-hour gate (L84) blunts the damage (a duplicate send is the worst case), but the pattern repeats in `portfolio/digest.py:48-52`, `portfolio/bigbet.py:36-44`, and reflection write-path. None hold a per-file lock around the read+atomic_write composite. Acceptable for single-loop architecture but documented because the metals_loop subprocess also imports several of these modules.

portfolio/fin_evolve.py:55-58: 🟡 risk: `_atomic_append_jsonl` defeats batch atomicity. The helper iterates entries and calls `atomic_append_jsonl(path, entry)` ONE AT A TIME (L57-58). If the process crashes after writing 4 of 10 entries, journal_outcomes.jsonl is left half-updated. The atomic guarantee is per-entry, not per-batch. `backfill_journal_outcomes` (L424-429) hits this path with `new_outcomes` lists that can exceed 100 entries — partial state will mislead `evolve()` on next run because the "scored set" derived from the file (L342) will think the missing entries are unscored zombies and re-process them. Minor (no data loss, only re-work) but the function's name implies batch atomicity it doesn't deliver.

portfolio/seasonality_updater.py:30-47: 🟡 risk: per-ticker profile partial-update window. `update_seasonality_profiles` loads all profiles (L28), mutates `profiles[ticker]` inside the loop (L38), then writes ALL profiles at the end (L46-47). If the loop crashes mid-iteration after XAG-USD is updated but before XAU-USD finishes, the saved file is fine. BUT if two updaters run concurrently (e.g. metals_loop + main loop both call it), they each read the same baseline and one's XAU update overwrites the other's XAG update — classic lost-update race. Fix: write atomically per ticker, or hold a lock around the read-update-write composite.

portfolio/qwen3_trader.py:45-69,225: 🟡 risk: prompt-file race + missing UTF-8 declaration on stdin read. `_run_native` writes to `qwen3_prompt.txt` in tempdir (L48-50) — a SHARED filename across all qwen3 subprocess invocations. If two `qwen3_signal._call_qwen3` fallbacks fire concurrently (different tickers in different threads with the GPU gate freed between them) they will clobber each other's prompt mid-write. The native binary may then load a Frankenstein prompt blending two tickers' context, producing nonsensical action for both. Use `tempfile.NamedTemporaryFile(delete=False)` with a unique path. Also L226 `json.loads(sys.stdin.read())` lacks explicit encoding — on Windows where stdin defaults to mbcs, non-ASCII chars in the context (Swedish ticker names, reasoning quotes) raise UnicodeDecodeError.

portfolio/metals_precompute.py:108,115,806,840,912,939,962,1000,1060,1148,1217: 🟡 risk: relative paths everywhere. The module hardcodes string literals like `"data/silver_deep_context.json"`, `"data/agent_summary_compact.json"`, `"data/cot_history.jsonl"`, `"data/layer2_journal.jsonl"`, `"config.json"` — all relative to the process CWD. The main loop happens to run from the repo root so this "works", but `crypto_precompute.py:31-32`, `oil_precompute.py:29-32`, `mstr_precompute.py:28-29` repeat the same pattern. If any caller changes CWD (or a future scheduled-task config sets a different working dir), every precompute writes deep-context JSON files to wherever the process started, silently breaking Layer 2's reads. The sibling pattern in `digest.py:23` (`BASE_DIR = Path(__file__).resolve().parent.parent`) is the safe form — adopt it consistently.

## Nit Findings

portfolio/bigbet.py:36-44,594-599: 🔵 nit: read-modify-write on state without per-file lock. Like daily_digest/digest — acceptable in single-process loop but flagged for awareness.

portfolio/prophecy.py:55-141: 🔵 nit: every CRUD operation is read-full-file → mutate → atomic_write_json. For a prophecy file with ~10 active beliefs this is fine, but multiple checkpoint evaluations in `evaluate_checkpoints` (L201-269) all roll into a single write at the end — good. The per-update writes in `add_belief`/`update_belief`/`remove_belief` would race against `evaluate_checkpoints` if both fired in the same cycle. Single-loop assumption keeps this safe; document the assumption.

portfolio/reflection.py:166-172,218-219: 🔵 nit: `maybe_reflect` swallows broad Exception with no exc_info. `logger.warning("reflection failed: %s", e)` loses traceback. Other modules in the same audit (e.g. fin_evolve.py:163-166, crypto_precompute.py:142-143) already use the `exc_info=True` pattern. Apply consistently so future failures don't require log-grep archaeology.

portfolio/shadow_registry.py:251-273: 🔵 nit: `_PROMOTED_CACHE` is a module-level mutable dict mutated without a lock. signal_engine dispatch is multi-threaded (ThreadPoolExecutor 8 workers per CLAUDE.md). Two threads simultaneously running `is_promoted` after TTL expiry could race on the assignment to `_PROMOTED_CACHE["data"]`/`["loaded_at"]`. Race is benign (worst case = one extra file read), but the comment at L246-250 explicitly justifies the cache as a performance optimization, and a wasted concurrent reload defeats the purpose under high dispatch concurrency. Add a threading.Lock or use the same `_init_lock` pattern as bert_sentiment.py.

portfolio/notification_text.py:23-26: 🔵 nit: `format_vote_summary` accepts arbitrary ints with no type validation. Negative counts produce `"-1 buy / 0 sell"` in user-facing notifications. `int(buy_count)` would raise on a non-numeric input. Defensive: `max(0, int(buy_count))`.

portfolio/feature_normalizer.py:32,113: 🔵 nit: module-level `_buffers` dict mutated by `update`/`clear` without locking. ThreadPoolExecutor calling `update("XAG-USD", "rsi", x)` from two threads concurrently could race on `_ensure_buffer` (L37-40) creating two deques and one losing its initial value. `deque.append` itself is thread-safe in CPython but the get-or-create composite is not. Wrap in a Lock OR use `_buffers.setdefault(key, deque(maxlen=...))` for atomic create.

portfolio/seasonality.py:26-73: 🔵 nit: no look-ahead bias detected — `compute_hourly_profile` correctly computes profile FROM the input series only and is the caller's responsibility to pass training data without the target bar. Callers (`seasonality_updater._fetch_hourly_klines`) fetch the latest 500 bars and pass straight in — this DOES include the current/last bar in the profile. For an intraday detrending application this is the standard pattern (the profile represents historical bias, not a leakage signal), but if a downstream signal later applies `detrend_return` to the same bar that built the profile, that's leakage. Add a docstring caveat: "Caller must exclude the bar being evaluated from the input klines, otherwise the seasonal mean is contaminated by the target."

portfolio/decision_outcome_tracker.py:76-77: 🔵 nit: bare `except Exception: continue` on `_fetch_historical_price`. Network blip → outcome silently skipped. fin_evolve.py:163-166 has the same wrap but logs at WARNING with context — apply the same pattern here so a sustained Alpaca/Binance outage shows up in logs rather than silently zeroing out the layer2 backfill accuracy.

portfolio/stats.py:21,58: 🔵 nit: `datetime.fromisoformat(e["ts"])` without timezone normalization. Same family of bug as regime_alerts/weekly_digest. Stats are display-only, so the impact is cosmetic (entries with naive timestamps land in "today" because hour-based comparisons aren't done) — but flag for future-proofing.

portfolio/digest.py:43,256: 🔵 nit: stale agent_summary.json passes through silently. `_build_digest_message` logs WARNING (L142) if summary is empty, but Layer 2 freshness — "is this summary < 5 minutes old?" — is never checked. If the main loop hangs and the summary becomes stale, the digest sends 4h-old numbers labeled as current. Same issue exists for ALL precompute consumers (crypto_deep_context.json, silver_deep_context.json, etc.): Layer 2 reads them and the operator sees `generated_at` only if they look. Wire a freshness gate: if `generated_at` is > 2× the precompute interval, prepend `_(WARNING: data ${age} old)_` to the digest.

portfolio/qwen3_signal.py:13: 🔵 nit: top-level `import subprocess` unused in this module — `subprocess` symbols only referenced on line 110 via `subprocess.TimeoutExpired`. Actually used, keep.

portfolio/fish_instrument_finder.py:48-50: 🔵 nit: `_search_avanza` swallows broad Exception → empty list, masking auth failures from Avanza session expiry. Same Avanza-session-failure-detection pattern called out in 05_avanza_api.md applies here.

---

## Summary

- 2 critical: backtester look-ahead bias (today's accuracy used to gate 30-day-old entries); tinylora_trainer DST-naive CET hardcoded UTC+1
- 9 risks: naive-aware datetime drops in regime_alerts/weekly_digest; vector_memory unbounded growth + full-walk every call; sentiment_shadow_backfill full-file reads; fish_monitor full-file load to read last line; analyze.py bypasses claude_gate cost tracking; daily_digest/digest/bigbet read-modify-write races; fin_evolve per-entry "atomic batch" not actually batched; seasonality_updater concurrent-write race; qwen3_trader shared tempfile name race; metals/crypto/oil/mstr precompute use CWD-relative paths everywhere
- 10 nits: state lock omissions; bare exception handlers; missing freshness gates; thread-unsafe singleton caches; missing UTC normalization in display paths

Auth/CLAUDECODE handling: confirmed clean — bigbet.py routes through `claude_gate.invoke_claude_text` (which calls `_clean_env()`), analyze.py defines its own local `_clean_env()` that correctly pops `CLAUDECODE` and sets `PF_HEADLESS_AGENT`, and fin_evolve doesn't spawn Claude at all. No nested-session-error vector.

Atomicity: prophecy.save_beliefs, reflection writes (atomic_append_jsonl), digest state writes, all precompute outputs use the file_utils atomic helpers. The risk category above is about read-modify-write composite races, not raw write atomicity.

Look-ahead bias: confirmed only in backtester.py (P1 above). seasonality.py itself is safe; whether the caller leaks is undetermined from this scope.

Vector memory: uses cosine distance (`hnsw:space: cosine`) consistently across embed + query. No metric-mismatch bug.

Bigbet circuit breaker: not applicable — bigbet sends alerts only, does not trade. Docstring confirms.
