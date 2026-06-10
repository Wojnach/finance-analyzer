# PLAN — Audit-fix campaign 2026-06-10/11

> Supersedes the 2026-06-01 adversarial-review plan (completed — findings merged into
> `docs/IMPROVEMENT_AUDIT_2026-06-10.md` workflow).

Source: `docs/IMPROVEMENT_AUDIT_2026-06-10.md` — 138 findings across 13 sections, every
P0/P1 batch adversarially skeptic-verified (33 confirmed, 3 refuted). This plan executes
the fixes batch by batch under the /fgl protocol.

Process: worktree `.worktrees/audit-fixes`, branch `fix/audit-batches-20260610`.
Merge to main after EACH batch (session-loss safety — user constraint 2026-06-10).
Push periodically. Per batch: implement → targeted tests → `caveman:cavecrew-reviewer`
on the batch diff → fix P1/P2 findings → commit → merge.

## Status

- **B1 ops-automation — DONE**, merged `31df4c77` (pre-fgl; retro adversarial review scheduled).
  Pickups task action bat-wrapped (argv redirection bug killed every scheduled run),
  run-hidden.vbs optional wait mode, pickup retry semantics, check_critical_errors
  auto-resolve fix, fix_agent per-category state persist, accuracy_degradation 24h dedup.
- **B2 live-incident signals — DONE**, merged `0d0d2e94` (pre-fgl; retro review scheduled).
  crypto_macro expiry-proximity quarterly-only BUY (was permanent BUY on daily Deribit
  expiries — primary June accuracy-collapse driver), options metrics on monthly/quarterly
  chain, consecutive_negative day units, netflow 7d staleness alert (BGeometrics endpoint
  confirmed 404 — feed dead), fallback weight neutralization, econ_calendar event_free
  regime gate, accuracy consensus/shadow separation, SE gate n_eff autocorrelation correction.

## Remaining batches

### B3 prophecy
- alerts.log_critical level "error" → "critical" so startup check sees prophecy alerts (P1).
- outcomes: validate Claude-supplied spot_at_prediction (sanity band vs live/recent price;
  reject + alert on breach) — prevents permanent scoring crash/poison.
- outcomes: 8 of 13 enabled instruments structurally unscoreable — reconcile scoring
  coverage with enabled set; alert on unscoreable instruments instead of silence.
- publish: reconcile against enabled instruments (hallucinated accepted / missing unflagged).
- cost: cumulative_30d_usd = real 30-day window, same-day re-run dedupe.
- prophecy-daily.bat: check prep exit code before invoking claude; kill switch fail-closed
  (missing sentinel dir = treat as disabled + alert); model read from prophecy_config.json
  instead of hardcode.
- SECURITY: pass --allowedTools restricting the headless agent (Read, WebSearch, WebFetch,
  Write scoped to data/prophecy_runs/) — prompt-injection from fetched web content currently
  inherits repo-wide Bash(*)/Write(*).
- Weekend/holiday horizon scoring window fix (4h window scored as 1-2d move).

### B4 metals real-money
- avanza_session: pin ALL Playwright traffic to one dedicated long-lived worker thread
  (module-level single-thread executor wrapping api_get/api_post/api_delete) — kills the
  greenlet thread-affinity class that silently disabled grid fisher + stop management
  (confirmed P1). Alternative considered and rejected: adding greenlet error to
  is_browser_dead_error only heals after breakage.
- get_open_orders: propagate errors (raise or sentinel) — silent [] made reconcile
  misclassify live orders as filled/cancelled.
- grid_fisher: global halt runs EOD sweep + cancels armed buys before returning; halted
  flag clears next session day. EOD cancel verification before marking CANCELLED.
  Hardcoded 21:55 → todayClosingTime from API (memory rule). Reconcile buy+sell same-tick
  ambiguity: detect, log, take conservative path. Session-loss halt threshold: fixed
  global value, not scaled by instrument count.
- avanza_session min-order guard: exempt position-closing SELLs (sub-1000 SEK inventory
  must be exitable; EOD flat retried forever).
- metals_loop trailing stops: cancel previous stop before placing replacement on
  add-to-position.
- avanza_orders CONFIRM flow: single getUpdates consumer (route confirmations through
  telegram_poller offsets or disable the second consumer) — currently poller eats CONFIRM
  replies and confirmed orders silently expire.

### B5 orchestration
- autonomous._detect_regime: read extra['_regime'] (currently always falls back to
  'range-bound' — and autonomous IS the production path during freeze).
- invoke_agent: handle exited-but-unobserved prior agent (run completion accounting before
  spawn); fail-CLOSED on config read error for layer2.enabled; call
  claude_gate.check_claude_gates('layer2') at top so one master switch governs;
  persist {pid, start_ts, tier} at spawn + startup reap of stale agent processes.
- trigger: arm flip cooldown only after budget-floor gate passes; persist wall-clock not
  monotonic in _update_sustained.
- main: stop double-logging every False invoke_agent return as 'skipped_busy'.
- Comment sweep: 60s → 600s cadence where stale (market_timing.py; CLAUDE.md text in B11).

### B6 signal-core
- btc_proxy: disabled signal still votes live on MSTR — route through DISABLED_SIGNALS
  force-HOLD properly.
- MIN_VOTERS_METALS=2 vs Stage-4 dynamic floor 3 — honor the metals floor.
- Metals seasonal BUY multiplier applied after global 0.80 cap — apply before cap (cap is
  the documented invariant).
- _compute_applicable_count: ministral counts for non-crypto tickers too.
- outcome_tracker.backfill_outcomes: hold jsonl sidecar lock across read-process-rewrite
  (rotation race).
- blend_accuracy_data: recent window double-count in directional totals.
- Circuit-breaker comments + .claude/rules vs constants (2pp/45%/7000) — sync docs to code.
- Soft-vote dampening scale-invariance: document current behavior; add scale factor only
  if cheap and test-covered (judgment call at implement time).

### B7 portfolio-risk
- _DEFAULT_STATE: deep-copy per load (shared mutable holdings/transactions refs).
- compute_probabilistic_stops: metals branch annualization; MSTR gets stock session key.
- Drawdown peak: sanity bound vs current equity (glitched row must not permanently
  inflate peak); _streaming_max: skip non-numeric rows with warning instead of TypeError
  fail-open.
- monte_carlo_risk: fx_rate sanity band (mirror P1-15 fix).
- trade_guards: corrupt state file → warning + critical entry (not silent reset);
  check_overtrading_guards uses its portfolio arg / correct new-position counting;
  cooldown + loss-escalation apply to ENTRIES only — exits (SELL of existing position)
  pass through. Rationale: blocking exits hardest after losses inverts risk management.

### B8 swing-loops
- mstr_loop execution: SHORT entries priced/sized off the BEAR cert quote (currently BULL).
- mstr_loop state: live cash sync — implement real sync from Avanza buying power at phase
  start, or refuse to enter live phase with a loud error (no silent cash-starved start).
- mstr_loop loop.py: EOD-flatten backstop independent of bundle.is_usable().
- mstr_loop config: validate PHASE env (unknown → hard error at startup).
- crypto_loop: reject 0.0 prices like oil_loop does.
- oil_grid_signal: stamp bar timestamp, not now() — grid must see data age; oil fast-tick
  cadence 10s → 60s (feed lags 10-15 min; 10s is pure waste).
- oil_loop header: document real route (yfinance-only) — CLAUDE.md text in B11.

### B9 infra
- file_utils: msvcrt lock acquisition — bounded blocking retry with backoff + clear error
  naming the holder file; atomic_write_jsonl takes jsonl sidecar lock; prune_jsonl uses the
  same recovery decoder as last_jsonl_entry (don't drop concatenated-object lines);
  last_jsonl_entry: type-check decoded value is dict before returning.
- http_retry: cap retry_after sleep (90s) + total-deadline option; fetch_json: reject
  unknown kwargs with TypeError (silent **kwargs swallow).
- telegram_poller: ack offset only after successful command handling (current design
  permanently acks failed commands server-side) — bounded: skip-after-3-attempts to avoid
  poison-message loops.
- message_store: sync docstring routing table with SEND_CATEGORIES.

### B10 dashboard
- house_blueprint: sanitize rendered markdown HTML (bleach if importable, else
  markupsafe.escape fallback) — stored XSS from scraped content.
- Redact ?token= and /go/<slug> from access logs (custom log filter) or document residual
  risk; auth cookie: set secure flag only on https requests so LAN http sessions work
  (currently cookie dropped, forcing ?token= on every request — worse for log hygiene).
- App-wide MAX_CONTENT_LENGTH (e.g. 1 MB).
- cf_access: JWKS negative cache 60s.
- /api/avanza_account worker queue: bound + in-flight dedup.

### B11 docs
- CLAUDE.md fact sweep: active signals (rebuild list from signal_registry +
  DISABLED_SIGNALS at implement time — audit says 15 not 21), MIN_VOTERS metals=2, signal
  module counts, removed per-ticker overrides, applicable-signal counts, dashboard route
  counts (re-grep), test surface (~440 files / ~10.4K tests), loop cadence 600s, oil price
  route. Keep structure, fix facts.
- README: refresh from March state (instruments, signal count, test count).
- TRADING_PLAYBOOK: telegram sender claim, '30 signals', retired Forecast/Kronos refs,
  config.json raw-open sample → safe pattern.
- SYSTEM_OVERVIEW: header vs body counts.
- trading_status._in_session docstring (08:30-21:30).

### B12 hygiene
- .gitignore: zz*.txt, data/arxiv_*.xml, _livecheck/, phone-*.png, '0', 'nul',
  data/*_task.log; delete the untracked debris from disk.
- Prune orphaned .worktrees/* (git worktree prune + delete dirs not in `git worktree list`,
  EXCEPT audit-fixes while campaign live) — ~326 MB.
- `git add scripts/win/RC_DISABLED_DO_NOT_REENABLE.md` (kill-switch doc must be tracked).
- Move data/test_metals_swing_trader.py → tests/ (currently never runs in the suite);
  fix imports/paths as needed.
- Move one-off live-trading debug scripts in data/ (gold_sell_debug.py etc.) →
  scripts/archive/ with a README warning about hardcoded account/orderbook IDs.
- Optional if cheap: docs/SCHEDULED_TASKS.md inventory generated from
  scripts/win/install-*.ps1.

## End-of-campaign ops (main checkout, after final merge)

1. Full suite: `.venv/Scripts/python.exe -m pytest tests/ -n auto` — compare against the
   42 known pre-existing failures recorded in B2.
2. Append resolution entries for the 16 accuracy_degradation criticals (verdict: genuine
   regime shift; crypto_macro structural bias fixed in B2; cite commit).
3. Inspect scripts/pickups handler for LLM-CRYPTOTRADER-72H re: token freeze, then
   force-run: `.venv/Scripts/python.exe scripts/process_pending_pickups.py --force LLM-CRYPTOTRADER-72H`.
4. User admin actions: re-run install-pending-pickups-task.ps1 (action line changed);
   decide PF-FixAgentDispatcher (flag data/fix_agent.disabled now in place).
5. Restart loops: PF-DataLoop + PF-MetalsLoop (schtasks via cmd.exe; fallback taskkill
   PID + Start-Process per memory 2026-06-03).
6. Push via `cmd.exe /c "cd /d Q:\finance-analyzer && git push"` (if classifier blocks,
   ask user to run `! git push`).
7. Update docs/SESSION_PROGRESS.md + .remember handoff.

## Deferred (explicitly out of campaign scope — flagged for future sessions)

- signal_engine.py split (4,698 lines: policy constants / regime overlays / consensus /
  dispatch).
- Swing-loop scaffolding dedup (crypto/oil/metals ~90% copy-paste).
- Relocating executable loops out of data/ (metals_loop.py 7,904 lines).
- Dashboard app.py monolith split.
- run-hidden.vbs default-wait migration for all PF-* tasks (only pickups uses wait now).
- Replacement source for exchange-netflow (BGeometrics endpoint 404).

## Premortem

(to be filled by fresh-context premortem agent, then edited — /fgl step 3)
