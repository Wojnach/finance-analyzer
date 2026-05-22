# FGL Adversarial Review — Raw Subagent Outputs (2026-05-22)

Verbatim findings from the 8 fresh-context review subagents. Each reviewed one
subsystem on its `review/sub-<name>` branch (empty-baseline diff). Lead cross-critique
and severity reconciliation is in the synthesis doc; these are the raw inputs.

---

## 1. signals-core — pr-review-toolkit:code-reviewer

- P1 | meta_learner.py:269,290-297 | LightGBM early-stopping AND the accuracy-maximizing threshold search both run on the same `X_test`/`y_test`; `best_iteration_`, `calibrated_threshold`, `calibrated_accuracy`, `test_accuracy` are in-sample. The leaked threshold is what `predict()` uses in production → trades sized on optimistically-biased confidence. | Carve a separate validation slice from the train set for early-stopping + threshold calibration.
- P1 | outcome_tracker.py:157-166,488 + signal_db.py:159-173 | `log_signal_snapshot` appends JSONL then writes SQLite; a crash between leaves a snapshot in JSONL only. `update_outcome` then silently `return False` for that ts during backfill, and since `load_entries()` reads SQLite whenever `snapshot_count()>0`, those backfilled outcomes never reach accuracy/IC/gate computation. | Write SQLite before/atomically-with JSONL; have `update_outcome` insert the missing snapshot row.
- P1 | signal_weight_optimizer.py:90-116 + train_signal_weights.py:90-96 | Walk-forward split uses `iloc[train_end:test_end]` with zero embargo; training rows in the last horizon-window before `train_end` carry forward returns (1d/3d) that resolve inside the test period → look-ahead leakage inflating `avg_oos_corr`. | Insert an embargo gap ≥ one horizon length.
- P2 | signal_engine.py:448 | `_ACCURACY_GATE_HIGH_SAMPLE_MIN = 7000` contradicts its own comment (line 441 "raised → 10000"), SC-P1-2 comments ("10K+"), and `.claude/rules/signals.md` ("10,000+"). Signals with 7K-10K samples are force-HOLD'd at 50% when the documented invariant keeps them at 47%. | Reconcile constant and docs.
- P2 | ic_computation.py:241-262 | `IC_CACHE_FILE` is one shared file but `load_cached_ic` rejects on horizon mismatch and `compute_and_cache_ic` overwrites the whole file; 7 horizons/cycle each evict the others → cache never hits cross-horizon, forcing full Spearman recompute. | Key the cache by horizon.
- P2 | signal_decay_alert.py:27,148 | `check_signal_decay` and `log_decay_alerts` use relative paths `data/...`; under a different CWD the read silently returns `[]` (no decay detected) and the alert log goes to a phantom directory. | Resolve via `Path(__file__)...`.
- P2 | ticker_accuracy.py:86,131 | `direction_probability` applies the 47% gate with `min_samples=5`; a per-ticker signal with 5 noisy samples at 80% feeds Mode B probability + Kelly sizing. signal_engine's own gate requires 30 (`ACCURACY_GATE_MIN_SAMPLES`); MEMORY.md flags this exact "small samples lie" failure. | Raise per-ticker sample floor to ≥30.
- P3 | feature_normalizer.py:35-40 | `_ensure_buffer` non-atomic check-then-insert on module-level `_buffers` while `update()` runs from the 8-worker pool; two threads can each create a deque, discarding samples. | Guard with a lock / `setdefault`.
- P3 | regime_alerts.py:46-79,200-215 | `check_and_alert` loads regime_history up to 3×/call; check-then-`log_regime_change` not atomic across workers → double-log. | Cache lookup, serialize check+append.

SUMMARY: 0 P0, 3 P1, 4 P2, 2 P3.

---

## 2. orchestration — pr-review-toolkit:code-reviewer

- P1 | agent_invocation.py:684-1160 | `invoke_agent` (primary Layer 2 spawn path) never checks `claude_gate.CLAUDE_ENABLED` — documented as the master kill switch blocking ALL Claude invocations "no exceptions". Setting it False silently fails to stop the main Layer 2 agent; only `config.layer2.enabled` does. Operator believes Claude is killed; it isn't. | Check `CLAUDE_ENABLED` / `check_claude_gates` at the top of `invoke_agent`.
- P2 | loop_processes.py:92-106 | `_iter_processes`: `info = p.info` is inside the `try`; the `except` handler references `info.get("pid")`. If `p.info` raises, `info` is unbound → `UnboundLocalError` crashes the whole `process_iter` loop and `/api/loop-processes`. | Initialize `info = {}` before the `try`.
- P2 | escalation_gate.py:202-219 | `should_escalate` creates a `ThreadPoolExecutor(max_workers=1)` per call; on timeout `shutdown(wait=False, cancel_futures=True)` cannot cancel a running future — the hung non-daemon `query_llama_server` thread keeps running, leaking threads and blocking interpreter exit. | Reuse a module-level executor or a daemon worker thread.
- P2 | loop_contract.py:1738,1811-1819,2333-2347 | `ViolationTracker` defaults `state_file=CONTRACT_STATE_FILE`; `_save()` is a non-atomic load→mutate→write RMW. If any non-main loop calls `verify_and_act` without a per-loop tracker, loops interleave RMW on shared `contract_state.json` and clobber escalation counters. | Derive a per-loop state file from `loop_name`.
- P3 | bigbet.py:537-559 | `_update_streak` mutates `condition_streaks` every cycle but `changed` is not set on the not-met `pop` path nor the cooldown-blocked `continue` → streak updates not persisted (masked by 300s staleness check). | Set `changed=True` whenever the dict changes.
- P3 | agent_invocation.py:1086 / multi_agent_layer2.py:178 / analyze.py:282,746 | `claude_gate` docstring forbids direct `subprocess.Popen([claude_cmd,"-p",...])`; these Popen `claude` directly, bypassing cost/usage logging; `analyze.py` also bypasses the tree-kill (leaves Claude Node grandchildren on its 120s timeout). | Relax the docstring for the async-lifecycle case, or route `analyze.py` through the gate.
- P3 | escalation_gate.py:160 | `_log_decision` uses deprecated naive `datetime.utcnow()`; siblings use `datetime.now(UTC)`. | Use `datetime.now(UTC)`.

SUMMARY: 0 P0, 1 P1, 3 P2, 3 P3.

---

## 3. portfolio-risk — pr-review-toolkit:code-reviewer

- P0 | trade_guards.py:103-330 | Cooldown / position-rate guards have a check-then-act TOCTOU race. `check_overtrading_guards()` acquires `_state_lock`, reads, releases; `record_trade()` later acquires the lock independently. Two concurrent callers can both pass the cooldown / `position_rate_limit` check before either records → rate limit bypassed, ticker double-traded in the cooldown window. | Make check+record atomic, or have `record_trade()` re-verify under the lock and return a rejected flag.
- P0 | portfolio_mgr.py:44-62,108-159 | `_rotate_backups()` runs `shutil.copy2` (non-atomic) BEFORE `atomic_write_json`; a kill mid-copy truncates `.bak`, then the next save rotates the truncated `.bak` into `.bak2`, propagating corruption through the recovery chain. | Rotate backups only AFTER the atomic write succeeds; create `.bak` via atomic copy.
- P1 | kelly_sizing.py:296-310 | With no trade history, `avg_win`/`avg_loss` use a fabricated 1.5:1 reward:risk from ATR; combined with a 0.55-0.60 win prob, half-Kelly is still a large bet driven entirely by an assumed never-observed payoff ratio. | Cap at quarter-Kelly when payoff is assumed, or require a configured ratio ≤1.0 for the ATR fallback.
- P1 | kelly_sizing.py:314-315 | `max_alloc` caps recommended size as a % of *cash* not portfolio value; across multiple Kelly recommendations the 30% concentration check elsewhere can still be exceeded. | Cap against total portfolio value, net of existing position.
- P1 | risk_management.py:312 | Drawdown breaker uses strict `>`: at exactly the threshold the breaker does not trip. | Use `>=` for a risk halt.
- P1 | risk_management.py:217-317,251-270 | `check_drawdown` only returns `breached`; nothing here halts trading. Combined with the cash-only fallback when `agent_summary` is empty, an underwater portfolio during a stale-feed window reports a tiny drawdown and the breaker never trips. | When holdings exist but no price feed, return `breached=True` or a distinct `unknown` halt state.
- P1 | cumulative_tracker.py:57-90 | `_get_last_snapshot_ts` reads the last 2KB and uses `split("\n")[-1]`; a partial line at the chunk start → `JSONDecodeError` → returns `None` → duplicate snapshot appended. | Read backwards to a complete line.
- P1 | monte_carlo_risk.py:408 | `compute_portfolio_var` uses `agent_summary.get("fx_rate", FX_RATE_FALLBACK)` — the raw `.get` anti-pattern that risk_management P1-15 fixed. A stale `fx_rate: 1.0` makes every `*_sek` VaR/CVaR ~10× too small. | Route through `_resolve_fx_rate`.
- P1 | warrant_portfolio.py:25-39 | `load_warrant_state()` has no corruption handling and `save_warrant_state` has NO backup rotation (unlike `portfolio_mgr`). A corrupt warrant state silently becomes empty holdings and the next write permanently loses all leveraged positions. | Add backup-rotation + corruption-recovery.
- P1 | warrant_portfolio.py:182-265 | `record_warrant_transaction` does load→mutate→save with no lock; concurrent warrant writes (metals fast-tick + Layer 2) last-writer-wins, dropping a transaction. | Add a per-file lock.
- P1 | exit_optimizer.py:325-332 | Warrant-without-financing-level P&L uses `warrant_move = pct_move * leverage` — wrong for MINIs (ignores path-dependent compounding/decay); overstates upside, understates the loss floor. | Use the financing-level model for MINIs.
- P2 | equity_curve.py:495 | `losses` classifies break-even (pnl_pct==0) as losses, but `compute_trade_metrics` splits profit/loss on `pnl_sek` strict `>0`/`<0` — the two classifications disagree. | One convention (net `pnl_sek`).
- P2 | equity_curve.py:494-501 | Win/loss stats use gross `pnl_pct` while `profit_factor` uses net `pnl_sek`; a trade with a tiny positive move but fees > it counts as a win in `win_rate` and a loss in `profit_factor`. | Use net `pnl_sek` sign everywhere.
- P2 | monte_carlo_risk.py:188-198 | `__init__` allows negative/short positions but `drawdown_probability` / `total_exposure` assume long-only positive exposure; a short makes total value negative and risk silently reports 0.0. | Use `abs(shares)*price` or reject shorts.
- P2 | risk_management.py:43-110 | `_streaming_max` caches raw `f.tell()` offset; a partial last line makes the next call resume mid-line and skip a real entry. | Cache the offset of the last complete newline.
- P2 | cost_model.py:36-47 | `total_cost_pct()` / `round_trip_pct()` exclude the `min_fee_sek` floor; for small orders near the 1000 SEK minimum, break-even math understates true cost. | Make `total_cost_pct` order-value-aware, or rename to `marginal_cost_pct`.
- P3 | portfolio_validator.py:43 | `initial_value_sek` defaults to 500_000 when missing; cash reconciliation then runs against a guessed baseline. | Skip reconciliation when the field is absent.
- P3 | trade_guards.py:381 | The C4 broken-wiring check only fires when `all_warnings == []`; an unrelated warning masks a genuinely broken `record_trade()` wiring. | Decouple the wiring check.

SUMMARY: 2 P0, 9 P1, 5 P2, 2 P3.

---

## 4. metals-core — pr-review-toolkit:code-reviewer

_(see metals section appended below — subagent completed last)_

---

## 5. avanza-api — caveman:cavecrew-reviewer

- P1/P0 | avanza_session.py:336-342 | `api_post` returns `{"raw": body}` on HTTP-200 + unparseable JSON instead of raising. Order callers (`place_order`, `place_stop_loss`) expect `orderRequestStatus`/`status`, miss it, default to "UNKNOWN", log a warning only → a failed order appears to succeed. | On `resp.ok` but unparseable JSON, raise `RuntimeError`; don't return a `{"raw":...}` fallback.
- P1/P0 | avanza_session.py:94-95 | `load_session()` proceeds with only a warning when `expires_at` is unparseable — fail-open. Browser launches with potentially expired cookies. | Fail-closed: raise `AvanzaSessionError` on unparseable `expires_at`.
- P1/P0 | avanza/scanner.py:86-87 | `_marketdata()` catches all exceptions from `api_get()` and returns `{}` silently; caller cannot distinguish API failure from no-data. | Log at error and re-raise, or return `None` with logging.

SUMMARY (reviewer-stated): 0 P0, 3 P1, 0 P2, 0 P3 (reviewer also re-tagged all three as P0; lead reconciles severity in synthesis).

---

## 6. signals-modules — caveman:cavecrew-reviewer

- P0 ×9 | forecast_signal.py:164,167,170 (`_forecast_*`), :225,228,230 (`_forecast_chronos_v2`), :301,304,306 (`forecast_prophet`) | Unguarded division by `current_price` (= `prices[-1]`); if the last price is 0 the forecast confidence math produces inf/nan and crashes or corrupts the signal. | Add `if current_price <= 0: return None` after each `current_price = prices[-1]`.
- P1 | signals/crypto_macro.py:228,281 | `OPTIONS_TTL` is used at line 228 but defined at line 281 — works only via late binding; fragile to refactor. | Move the definition to module top.
- P2 | ml_signal.py:154-155 | `model.predict()` / `predict_proba()` called on possibly all-NaN features from sparse data → model may crash. | Guard: if features all-NaN, return HOLD/0.0.

SUMMARY: 9 P0 (one root cause: unguarded zero-division across 3 forecast models), 1 P1, 1 P2.

---

## 7. data-external — caveman:cavecrew-reviewer

- P1 | onchain_data.py:280-283 | When the BGeometrics token is missing, loads persistent cache with 2× TTL (24h) and returns stale data with no staleness signal — Layer 2 receives 24h-old on-chain metrics as fresh. | Emit a WARNING + age field; return `None` if older than `ONCHAIN_TTL`.
- P1 | funding_rate.py:44-45 | SELL threshold `>0.0003` (0.03%) is ~30-100× higher than real Binance funding (≈+0.0001%); the signal is near-permanently silent. | Lower to ≈0.0001, or calibrate to the live range.
- P1 | data_collector.py:96 | `pd.to_datetime(df["open_time"], unit="ms")` with no `utc=True`; pandas localizes to system tz. On a CET box this produces CET timestamps where downstream assumes UTC → 1-2h signal-time offset. | Add `utc=True`.
- P2 | onchain_data.py:282 | Stateless cache pattern: caller cannot tell 24h-stale from fresh-cached. | Add `_fetched_at_age_seconds` to the returned dict.
- P2 | metals_precompute.py:407 / oil_precompute.py:407 | CFTC COT endpoint uses bare `requests.get()` (no retry); one blip kills COT for the 4-7h refresh window. | Wrap with `fetch_with_retry`.
- P2 | alpha_vantage.py:140-142 | No distinction between rate-limit (4xx, budget burned) and server error (5xx, retryable); quota usage is invisible. | Log rate-limit hits at WARNING + counter; retry 5xx only.
- P2 | fear_greed.py:105-109 | Empty `{"data": []}` (maintenance window) is guarded by `if not data_list` but `data_list[0]` is still reached → `IndexError`. | Index only after a length check.
- P2 | futures_data.py:50 | `float(data["openInterest"])` raises `KeyError` on a malformed Binance payload; no try/except → the worker thread for that symbol freezes. | Wrap in try/except, return `None`.
- P2 | microstructure_state.py:227 | `persist_state()` rebuilds state across multiple independent `_buffer_lock` acquisitions; concurrent metals/main calls can persist an inconsistent snapshot. | Hold the lock once around the whole snapshot.
- P3 | data_collector.py:74-101 | `cb.record_success()` ordering is fragile relative to the post-fetch transforms. | Move success-record after all transforms.
- P3 | fx_rates.py:48 | Out-of-band FX rate path has implicit fall-through control flow. | Add explicit `return None`.
- P3 | macro_context.py:100-102 | `round()` of NaN serializes to invalid JSON `NaN`. | `None` when NaN.
- P3 | metals_precompute.py:292 | `if len(closes) < 5: return None` with no logging — silent abort. | Add a warning log.

SUMMARY: 0 P0, 3 P1, 7 P2, 3 P3.

---

## 8. infrastructure — pr-review-toolkit:code-reviewer

- P1 | file_utils.py:240-248 | `jsonl_sidecar_lock` lock-file creation is non-atomic/racy: `if not lock_path.exists()` then `open(...,"ab")` (TOCTOU); worse, if creation fails (logged, NOT re-raised) the next `open(lock_path,"rb+")` raises `FileNotFoundError` uncaught → crashes every JSONL writer. A disk-full/permission glitch becomes a system-wide JSONL-write crash. | Create via `os.open(...,O_CREAT)` and open the handle with `"a+b"`.
- P1 | file_utils.py:269-292 | `atomic_append_jsonl` is not atomic against a concurrent reader: the append is a plain `open(path,"ab")`; a reader (`load_jsonl_tail`, dashboard) can read a torn final line. `count_jsonl_lines` (used as a write-detection primitive in `agent_invocation`) counts a torn line. | Document reader visibility honestly or lock readers too.
- P1 | gpu_gate.py:225-230 | Same-process GPU-lock re-entry path: on `FileExistsError` with `pid == os.getpid()` it sets `file_acquired=True` and `break`s without creating the file; on context exit `_release_lock()` `unlink`s a lock file another in-process holder still depends on → deletes a live lock. | Track re-entry depth; only the original acquirer unlinks.
- P1 | dashboard/system_status.py:48 + trading_status.py:55-56,278-279 | Session-window constants (`SESSION_OPEN=08:30/CLOSE=21:30`) contradict the module's own docstring ("15:30–21:55"); a future edit trusting the docstring reintroduces the 2026-05-11 user-reported bug. | Fix the stale docstring.
- P2 | file_utils.py:144-207 | `load_jsonl_tail` decodes with `errors="replace"`; a multibyte char split at the seek boundary can mangle a line → that JSON entry silently dropped. | Read back to a char boundary.
- P2 | log_rotation.py:358-364 | `rotate_jsonl` writes the temp file as a FIXED name `filepath.with_suffix(".tmp")`; two overlapping rotation passes race on the same temp path. | Use `tempfile.mkstemp`.
- P2 | log_rotation.py:333-344 | `rotate_jsonl` archive-append decompresses the whole monthly `.gz`, concatenates in RAM, re-compresses non-atomically; a crash mid-write corrupts the archive `.gz`. | Write to temp + `os.replace`.
- P2 | telegram_poller.py:113-121 | `_poll_loop` catches `Exception` and `time.sleep(5)` forever with no backoff and no crash counter; a persistent failure log-spams every 5s and the operator never learns the poller is dead. | Exponential backoff + `critical_errors` after N failures.
- P2 | health.py:64-86 | `heartbeat()` does load→mutate→`atomic_write_json` under a `threading.Lock` (intra-process only); `health_state.json` is also written by `check_agent_silence` and possibly the dashboard process → cross-process lost updates to `cycle_count`/`errors`. | Cross-process file lock, or single-writer.
- P3 | digest.py:168 | `l2_failures = max(0, invoked - l2_analyses)` mixes two files written at different times → skewed "Failed" digest line. | Cosmetic.
- P3 | config_validator.py:43-57 | `validate_config` only rejects empty *string* required keys; a required key that is `null`/`0`/`{}` passes. | Reject falsy non-string values too.
- P3 | memory_consolidation.py:371 | `output_path.write_text(md)` is a non-atomic raw write of `trading_insights.md` (Layer 2 reads it every invocation) — violates the atomic-I/O rule. | Use `atomic_write_text`.
- P3 | journal.py:24-40 / journal_index.py:373-383 / vector_memory.py:269-281 | Three raw `open(JOURNAL_FILE)` line-readers instead of `file_utils.load_jsonl`. | Use the canonical reader.

SUMMARY: 0 P0, 4 P1, 4 P2, 4 P3.
Reviewer note: dashboard `auth.py`/`cf_access.py` are solid (`hmac.compare_digest`, JWT signature-verified, fail-closed); `process_lock.py` uses kernel-released OS locks so PID reuse cannot grant a false lock.
