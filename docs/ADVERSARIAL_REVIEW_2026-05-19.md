# Adversarial Review Synthesis — 2026-05-19

**Trigger commit:** `2daa4fd9` (main HEAD at review start)
**Review style:** /fgl whole-codebase audit, 8 subsystems × dedicated reviewer subagent + 1 independent cross-cutting pass from the main thread
**Empty-baseline diff technique:** orphan branch `review-baseline-empty` lets each reviewer see the entire subsystem as additions, not just recent changes
**Worktree:** `Q:\finance-analyzer-reviews\2026-05-19` (cleaned up after merge — see commit log)

---

## Tallies

| Subsystem | Agent | P0 | P1 | P2 | P3 | Total |
|-----------|-------|----|----|----|----|-------|
| signals-core | pr-review-toolkit:code-reviewer | 5 | 11 | 6 | 2 | 24 |
| orchestration | pr-review-toolkit:code-reviewer | 2 | 9 | 5 | 2 | 18 |
| portfolio-risk | pr-review-toolkit:code-reviewer | 4 | 11 | 9 | 4 | 28 |
| metals-core | pr-review-toolkit:code-reviewer | 11 | 13 | 11 | 15 | 50 |
| avanza-api | caveman:cavecrew-reviewer | 3 | 7 | 5 | 3 | 18 |
| signals-modules | pr-review-toolkit:code-reviewer | 4 | 16 | 14 | 8 | 42 |
| data-external | pr-review-toolkit:code-reviewer | 5 | 16 | 8 | 3 | 32 |
| infrastructure | caveman:cavecrew-reviewer | 4 | 6 | 10 | 7 | 27 |
| **main thread cross-cut** | self | 4 | 6 | 4 | 2 | 16 |
| **TOTAL** | | **42** | **95** | **72** | **46** | **255** |

42 P0 findings is high but expected — the empty-baseline technique surfaces issues that diff-mode reviewers skip. Many P0s are concentration-of-concern findings the team already knew about (`memory/signal_engine_audit_findings.md`, `memory/dynamic_corr_bug.md`) — confirms the audit. Several are genuinely new.

---

## TOP 10 — Highest-conviction fix-first findings

Picked by: cross-agent corroboration (independently flagged by ≥2 sources), money-at-risk, accuracy-impact, ease-of-fix ratio. These are the must-fix items.

### T1. Bias-penalty double-application in signal_engine

`portfolio/signal_engine.py:2646-2668` (Agent 1, P0).
`_weighted_consensus` multiplies weight by `normalized_weight = rarity * bias_penalty` from `signal_activation_rates`, THEN multiplies again by `_resolve_bias_penalty(signal_bias)`. Same bias applied twice. Extreme-bias signal (bias=0.91) gets effective 0.02× weight instead of intended 0.2×. Docstring at line 2533 explicitly claims single application — it is wrong.

**Fix:** drop one path. Prefer keeping `_resolve_bias_penalty` (tiered cascade) and removing the bias term from `signal_activation_rates.normalized_weight`. Then audit any consumer that reads `normalized_weight` for the same expectation.

### T2. Utility-boost lets failed signals bypass accuracy gate

`portfolio/signal_engine.py:4140-4162` (Agent 1, P0; corroborated by `memory/signal_engine_audit_findings.md`).
`u_score = u.get("avg_return", 0.0)` is the raw percent value (0.5 means +0.5pp). `boost = min(1.0 + u_score, 1.5)` produces up to 1.5× boost. Signal with 40% accuracy becomes `0.40 * 1.5 = 0.60` in the gate check — passes both the 47% standard gate and the 50% high-sample gate.

**Fix:** divide `avg_return` by 100 before computing boost, or change boost to `1 + u_score/2` capped at 1.10.

### T3. Stop-loss limits too tight on 5x silver MINI warrants

`portfolio/grid_fisher.py:1493` + `portfolio/fin_snipe_manager.py:537` (Agent 4, P0; corroborated by `metals-avanza.md` rule "5x leverage certificates need -15%+ stops" + `memory/feedback_mini_stoploss.md`).
- grid_fisher stop sell-price floor `stop_price * 0.995` = 0.5% buffer. Gap past trigger → unfilled limit.
- fin_snipe_manager `HARD_STOP_CERT_PCT = 0.05` (5% cert) = ~1% underlying for 5x cert — inside normal intraday wick.
- fin_snipe_manager `HARD_STOP_SELL_BUFFER_PCT = 0.01` = 1% below trigger → 30s wicks bypass the limit entirely.

**Fix:** widen all three to ≥3% buffer; raise cert stop-distance to 15% (3% underlying).

### T4. Warrant state file has zero concurrency protection

`portfolio/warrant_portfolio.py:42-49` (Agent 3, P0).
`save_warrant_state` writes `portfolio_state_warrants.json` with no threading.Lock and no file lock. metals_loop + dashboard + GoldDigger + ad-hoc scripts all do read-modify-write on this leveraged-positions state. Lost transactions silent.

**Fix:** wrap with `portfolio/process_lock.py`-style cross-process advisory lock (file-level, not threading.Lock — that's per-process only).

### T5. Cross-process race on portfolio_state.json

`portfolio/portfolio_mgr.py:108-113` (Agent 3, P0).
Same class as T4. `_save_state_to` uses `threading.Lock`. Layer 2 subprocess + dashboard + main loop are separate processes. RMW window unprotected. Same class of bug as Mar 3 stop-loss incident (coordination assumption that doesn't hold cross-process).

**Fix:** file-level lock (msvcrt.locking / fcntl.flock) or move state to SQLite WAL.

### T6. Layer 2 off-hours skip silently disables decisions

`portfolio/main.py:972-979` (Agent 2, P1).
When `layer2.enabled=True` but `_is_agent_window()` is False (weekend, US holiday), trigger logged as `skipped_offhours` and autonomous_decision is NOT called. Crypto + metals trade 24/7. Weekend XAG-USD F&G crossing → no journal, no Telegram, no recommendation.

**Fix:** in the outside-window branch, call `autonomous_decision` to maintain pipeline.

### T7. BULL vs BEAR warrant direction not honored in P&L math

`portfolio/warrant_portfolio.py:92-101` + `portfolio/risk_management.py:382` (Agent 3, P1).
`implied_pnl_pct = underlying_change * leverage` — leverage treated as positive scalar. BEAR cert with positive underlying_change should produce NEGATIVE P&L. Stop-loss direction check has same bug — `triggered = current_price < stop_price` assumes long. Silent P&L corruption + wrong stop direction for any BEAR position.

**Fix:** read `direction = holding.get("direction", "LONG")` and sign appropriately.

### T8. Binance error responses returned as garbage candles

`portfolio/data_collector.py:74-101` (Agent 7, P0).
`_binance_fetch` does `r.raise_for_status()` then `pd.DataFrame(data, ...)`. Binance returns 200 OK with `{"code": -1121, "msg": ...}` JSON on bad symbol or `10m` interval (CLAUDE.md warns). Dict gets DataFrame'd → garbage 1-row frame → ValueError or silent bad signals.

**Fix:** `if isinstance(data, dict) and "code" in data: raise ConnectionError(data["msg"])`.

### T9. Account ID not filtered in get_positions — pension positions leak into ISK trading

`portfolio/avanza_session.py:676, 682-717` (Agent 5, P0; corroborated by `memory/feedback_isk_only.md`).
`get_positions()` returns all positions across all accounts. Callers assume ISK-only. Pension account (2674244) holdings can be sized as ISK cash.

**Fix:** filter response by `account_id == ALLOWED_ACCOUNT_IDS` before returning.

### T10. Silver fast-tick races slow loop on module globals

`data/metals_loop.py:1407-1503` (Agent 4, P0).
Fast-tick (10s thread) mutates `_silver_fast_prices`, `_silver_alerted_levels`, `_silver_session_low/high`, `_silver_consecutive_down`, `_silver_prev_price` with NO lock. Slow loop (60s) reads/clears same globals. `_silver_fast_prices.clear()` racing velocity-window read.

**Fix:** wrap fast-tick state in `threading.Lock`.

---

## Cross-cutting patterns (recurring across subsystems)

### Pattern 1 — Cross-process state files without cross-process locks

T4 (warrants), T5 (portfolio_state), Agent 8 (journal.py:28 race), Agent 7 (microstructure_state:205-213 dict-mutation-during-iteration). `threading.Lock` is widespread; `file_utils.jsonl_sidecar_lock` is the right primitive but not consistently used outside `atomic_append_jsonl`. **Recommendation:** add a `state_file_lock(path)` context-manager wrapping `msvcrt.locking`/`fcntl.flock` and migrate all RMW sites.

### Pattern 2 — Quotas not persisted across restarts

Agent 7 (alpha_vantage daily 25/day, onchain_data BGeometrics 15/day, earnings_calendar bypassing AV counter). Module-level integers reset on every process restart. Loop crash-restart cadence (10s→5min backoff after crash) plus quota-bypass = quota exhausted by lunch. **Recommendation:** persist `{"date": ..., "used": N}` to `data/quota_state.json` via `atomic_write_json`.

### Pattern 3 — Per-ticker accuracy gate divergence

Agent 1 (ticker_accuracy.py:131 uses flat 0.47, ignores tiered 0.50 for 7K+ samples). signal_engine has tiered logic; ticker_accuracy doesn't import the same constants. Mode B Telegram probabilities and Kelly sizing both flow through ticker_accuracy. **Recommendation:** centralize the tier table in one module, import everywhere.

### Pattern 4 — Wall-clock `time.time()` used for elapsed/cooldown

Agent 2 (shared_state rate limiter), Agent 8 (health.py uptime, message_throttle cooldown). NTP jumps after Windows sleep cause silent breakage. **Recommendation:** `time.monotonic()` for elapsed; keep wall-clock only for ISO timestamp display.

### Pattern 5 — Hardcoded calendars exhaust

Agent 7 (fomc_dates, econ_dates exhausted Dec 2027), Agent 4 (`EOD_HOUR_CET = 17.0` hardcoded no DST). 18 months until production-affecting calendar starvation. **Recommendation:** fetch from Fed/BLS JSON during a weekly maintenance job, or use a library (`pandas_market_calendars`).

### Pattern 6 — Hardcoded FX fallback

Agent 7 (`FX_RATE_FALLBACK = 10.50`), Agent 4 (iskbets.py:743 hardcoded 10.5). Stale by 5-7% vs current 10.85-11.30 range. Portfolio valuation, equity curve, Monte Carlo VaR all use this. **Recommendation:** kill trades when no fresh FX available; persist last-good rate to disk.

### Pattern 7 — Cache-vs-source divergence between SQLite and JSONL

Agent 1 (signal_log.db vs signal_log.jsonl produce different accuracy numbers because SQL skips `_MIN_CHANGE_PCT` neutral filter; outcome_tracker writes JSONL first then SQLite — crash in between leaves SQLite missing entries). **Recommendation:** write SQLite first (canonical), JSONL second (audit trail); document which is source-of-truth; add explicit reconciliation pickup on startup.

### Pattern 8 — `datetime.utcnow()` + naive datetime drift

Agent 2 (`escalation_gate.py:160`), Agent 7 (`crypto_macro_data.py:108` `date.today()` local-time). Mixed naive/aware datetime comparisons. **Recommendation:** sweep for `utcnow()` and `date.today()`, replace with `datetime.now(UTC)` everywhere.

### Pattern 9 — Silent except-pass burying signal failures

Agent 1 (signal_engine.py:3701 broad Exception → just `_signal_failures.append`), Agent 7 (sentiment subprocess JSON-parse, futures_data oi_usdt drop), Agent 6 (`vwap_zscore_mr` silent return HOLD on any exception). Pattern: signal compute fails → silent HOLD → no Telegram alert. **Recommendation:** raise N consecutive-failure threshold for individual signals; route into `data/critical_errors.jsonl` so the fix-agent dispatcher catches it.

### Pattern 10 — Hardcoded leverage assumptions / USD-vs-cert mix-ups

Agent 4 (iskbets, fin_snipe_manager USD-ATR used for cert-distance stops — Telegrams understate risk by 5×), Agent 3 (warrant_portfolio leverage as scalar — no direction). **Recommendation:** establish a "cert-space vs underlying-space" convention; require explicit `* leverage` or `/ leverage` at every conversion boundary; lint for it.

---

## Per-subsystem highlights (P0/P1 only)

### signals-core (Agent 1) — 5 P0, 11 P1

The most consequential subsystem because every trade decision flows through it.

Critical paths:
- **Bias penalty double-application** (T1) — accuracy-corruption P0
- **Utility boost accuracy-gate bypass** (T2) — already in audit memory
- **`accuracy_stats.py:225-228`** — diagnostic counter only catches explicit None, not absent keys → schema corruption invisible
- **`outcome_tracker.py:159-166`** — JSONL written before SQLite; crash between leaves accuracy data invisible to `load_entries()` which prefers SQLite
- **`ticker_accuracy.py:131`** — flat 0.47 gate, ignores tiered 0.50 → Mode B Telegrams & Kelly sizing hit by sub-tier signals (Pattern 3)
- **`ic_computation.py:25-28`** — single-file IC cache cannot hold multiple horizons (each overwrites the previous)
- **`forecast_accuracy.py:142, 158-161`** — neutral outcomes counted as correct for SELL votes (inflated SELL accuracy)
- **`signal_engine.py:1467`** — silent skip when cross_acc==0.0 (legitimate 0% accuracy looks like missing data)
- **`signal_engine.py:2928-2932`** — RVOL gate silently skipped when volume data missing (rule says "RVOL<0.5 forces HOLD regardless")
- **`signal_weights.py`** — `SignalWeightManager` is dead code (writes file no one reads)

### orchestration (Agent 2) — 2 P0, 9 P1

- **`main.py:95-109`** — Linux/WSL singleton lock not released; duplicates `process_lock.py` (T4 of cross-cutting pass)
- **`main.py:616-617`** — ticker pool `future.result()` not exception-wrapped → single bad ticker crashes whole cycle
- **`main.py:972-979`** — off-hours Layer 2 skip without autonomous fallback (T6)
- **`shared_state.py:269-279`** — rate limiter wall-clock vulnerable to NTP jumps (Pattern 4)
- **`shared_state.py:200-209`** — `_cache_lock` held during `enqueue_fn` → potential deadlock under llama_server contention
- **`trigger_buffer.py:121-138, 160-196`** — RMW on shared JSON without file lock (Pattern 1)
- **`escalation_gate.py:202-219`** — thread-pool leak: every timeout creates fresh ThreadPoolExecutor + hung thread
- **`agent_invocation.py:355`** — `_no_position_skip` always reads T1 context regardless of tier
- **`agent_invocation.py:81`** — completion watchdog 30s interval → T1 (180s) can overrun by 16%

### portfolio-risk (Agent 3) — 4 P0, 11 P1

- **F01-F03** — cross-process lock gaps (T4, T5 + portfolio_state)
- **F04** — `risk_management.py:676` `total_fees = max(state_fees, computed_fees)` silently hides double-counting
- **F10-F12** — Monte Carlo math errors: 252-vs-365 mismatch on crypto vol scaling, t-copula correlation under-binding (Pearson ≠ input matrix), VaR raw `fx_rate` bypass of cached fallback chain (Pattern 6)
- **F13** — ATR cap at 15% silently softens stops to barrier-violating distances
- **F14-F15** — direction ignored in stop-loss and warrant P&L math (T7)
- **F16-F17** — Kelly without quarter-Kelly cap; metals Kelly silently passes when `cert_loss_frac ≥ 1` (full ruin territory)
- **F05-F09** — equity_curve / Sharpe / Sortino: inconsistent loss classifier (`pnl_sek > 0` for wins, `pnl_pct <= 0` for losses), 365 annualization with daily_rf from 365 (should be 252 BD), `_daily_returns` injects 0.0 on `prev_val <= 0`

### metals-core (Agent 4) — 11 P0, 13 P1

The subsystem trading real Avanza warrants — most P0s here are direct money-loss paths.

- **#1, #3, #5** — stop-loss limit-price too tight (T3)
- **#2** — grid_fisher knockout guard silently inert because catalog ships `barrier=0` → `None` → check skipped
- **#4** — `HARD_STOP_CERT_PCT = 5%` for 5x cert = ~1% underlying = inside normal wick
- **#6, #22** — iskbets USD-ATR stops used as cert-price stops; Telegram alerts understate risk by 5× (Pattern 10)
- **#7** — silver fast-tick races slow loop (T10)
- **#8-9** — ORB predictor pagination off-by-1ms can double-count bars; DST-shifted morning-window inconsistent with UTC candle hours
- **#10** — exit_optimizer uses `_TRADING_MINUTES["warrant"] = 820` to annualize vol of 24/7 underlying → 1.75× too tight quantiles
- **#11** — `metals_avanza_helpers:294` `today_str` from local OS time (not Stockholm) → `validUntil` can be yesterday
- **#42** — `stop_loss_price` updated even when `place_stop_loss` fails → state shows stop but no broker order (naked position)
- **#33** — emergency two-phase cancel-then-place leaves position naked on Avanza rate-limit
- **#32** — `EOD_HOUR_CET = 17.0` hardcoded with no DST handling (metals close is 21:55, not 17:00 — currently a "summary trigger" but a refactor could flatten at 17:00)

### avanza-api (Agent 5) — 3 P0, 7 P1

- **`avanza_session.py:676`** — `get_positions()` doesn't filter by account_id (T9)
- **`avanza_session.py:714`** — extracted `account_id` never validated against `ALLOWED_ACCOUNT_IDS`
- **`avanza/client.py:65`** — config-side `account_id` has no whitelist check; rogue config could trade on pension via TOTP path
- **`avanza_session.py:89` vs `:124`** — `<=` vs `<` asymmetry on expiry check
- **`avanza_orders.py:368, 373`** — order ID lost when API returns success without orderId → orphaned order
- **`avanza_session.py:336-342`** — `api_post()` returns `{"raw": body}` on non-JSON; callers expect structured response (silent failure on HTML login page)

### signals-modules (Agent 6) — 4 P0, 16 P1

- **`credit_spread.py:285` + `gold_real_yield_paradox.py:265`** — relative `"config.json"` path (loop runs from `C:\Windows` → silent HOLD)
- **`futures_basis.py:209-212`** — votes on XAU/XAG via thin perp futures (no metals exclusion)
- **`signal_engine.py:3405-3410`** — on-chain BTC voter sets votes without confidence; tie=HOLD = no-data=HOLD (likely root of "100% accuracy on 5 samples")
- **`signal_utils.majority_vote(count_hold=False)`** — confidence = winner/active. 1-of-5 composite sub-signals firing → confidence 1.0 (=unanimous). Affects 30+ composites.
- **`copper_gold_ratio.py:251-252` + `treasury_risk_rotation.py:185`** — hardcoded asset-class inversion violates "never invert sub-50% signals"
- **`statistical_jump_regime.py:204`** — SELL on any negative slope in low-vol regime (no deadband)
- **`finance_llama.py:204-214` + `cryptotrader_lm.py:150-158`** — confidence masquerade: HOLD with conf=0.50 (not 0.0) when parser fails — looks like real vote
- **`vwap_zscore_mr.py:124-125`** — silent `except: return HOLD/0.0` masks bugs (Pattern 9)
- **`news_event.py:47-50`** — single `headlines_latest.json` overwritten across 8 ticker threads (lock prevents corruption but not data loss)
- **Verifications:** DISABLED_SIGNALS gate works; P1.10 news_event lock in place; `_validate_signal_result` schema enforcement robust.

### data-external (Agent 7) — 5 P0, 16 P1

- **`microstructure_state.py:205-213`** — `persist_state()` double-records OFI history every cycle → variance crushed → orderbook_flow false signals (Pattern 1 variant)
- **`data_collector.py:74-101`** — Binance error JSON → DataFrame garbage (T8)
- **`alpha_vantage.py:31-32`** + **`earnings_calendar.py:49-52`** — daily 25/day quota reset per restart + earnings bypasses counter (Pattern 2)
- **`fx_rates.py:67-71`** — `FX_RATE_FALLBACK = 10.50` stale (Pattern 6)
- **`data_collector.py:96`** — Binance timestamps tz-naive but UTC (mixed naive/aware comparison errors)
- **`data_collector.py:316-344`** — `as_completed` TimeoutError doesn't actually cancel running futures (pool thread starvation)
- **`alpha_vantage.py:149-154`** — only checks `Note`, misses newer `Information` rate-limit field
- **`earnings_calendar.py:172-178`** — negative-caches `None` for 24h → earnings gate stays disabled on transient network blip
- **`fomc_dates.py:25-34`, `econ_dates.py:38-103`** — calendars exhaust Dec 2027 (Pattern 5)
- **`funding_rate.py:44-49`** — asymmetric thresholds (+0.03% SELL, -0.01% BUY) embed structural long bias
- **`macro_context.py:138-144`** — DXY synth path returns garbage `value` (constant 58.0)
- **`onchain_data.py:208-243`** — no daily quota tracking on 15/day BGeometrics budget (Pattern 2)
- **`indicators.py:155-207`** — regime cache key missing `horizon` parameter
- **`crypto_macro_data.py:108`** — `date.today()` local-time → Deribit expiry off by a day around midnight CET
- **`crypto_macro_data.py:218-220`** — `compute_gold_btc_ratio` reads from `agent_summary_compact.json` (cached, violates "Live prices first" CLAUDE.md rule)

### infrastructure (Agent 8) — 4 P0, 6 P1

- **`journal.py:28`** — `load_recent()` reads JOURNAL_FILE without `jsonl_sidecar_lock` → race with concurrent `atomic_append_jsonl`
- **`health.py:29`** — `uptime_seconds` uses `time.time() - state["start_time"]` (Pattern 4)
- **`api_utils.py:28-31`** — config cache deadlock: if `stat().st_mtime` raises OSError once, cache stale forever
- **`telegram_poller.py:290`** — `_log_inbound()` catches all exceptions silently → trade executed but never logged (audit-trail loss)
- **`http_retry.py:51-57`** — `Retry-After` header honored for 429, ignored for 502/503/504 → exponential backoff hammers server during recovery
- **`log_rotation.py:363`** — `os.fsync` after `os.replace` doesn't guarantee directory entry durability (call fsync BEFORE replace)
- **`config_validator.py:46-57`** — silent OPTIONAL_KEYS pass → missing Telegram token crashes at runtime instead of at startup

### main thread cross-cut — 4 P0, 6 P1

- **M01-M03** — raw `open(... "w") + json.dump` in `data/{crypto_monitor, metals_history_fetch, silver_monitor}.py` — violates mandatory atomic-I/O rule (CLAUDE.md). Replace all three with `atomic_write_json`.
- **M04** — dashboard `_get_dashboard_token() is None` opens all routes (config corruption / mid-edit window)
- **M05** — main.py reimplements singleton-lock; use `process_lock.py` instead (also fixes Agent 2 P0-1)
- **M06** — Layer 2 subprocess inherits all parent env via `os.environ.copy()` + targeted `pop()`; consider env allowlist
- **M07** — `escalation_gate.py:160` `datetime.utcnow().isoformat() + "Z"` produces naive ISO with TZ marker — false TZ claim (Pattern 8)
- **M08** — signal_engine and signal_history use 8 different module-level threading.Locks but no documented composition order
- **M09** — Layer 2 prompts embed unsanitized trigger reasons (defense-in-depth concern, not exploitable today)
- **M10** — `os.environ.get("NO_TELEGRAM")` truthiness check spread across 5 files; pick one canonical check and centralize

---

## Cross-critique: agent disagreements & false positives

### False positives flagged

- **Agent 8 `subprocess_utils.py:290`** — flagged as P1 shell injection. Verified `ps_cmd` is hardcoded with no user input. Downgrade to P3 (`shell=True` is a smell on Windows but no actual injection vector). Already documented in `data/fgl-logs/codex-critique-of-claude-infrastructure.err`.
- **Agent 4 #37** — `calculate_morning_range` UTC-hour-vs-DST-window. Agent re-reviewed and withdrew the P1 (correct by accident, kept as P3 documentation).
- **Agent 4 #44** — `fin_snipe.py:106-107` "Hammer pattern, ~1500 calls/day". Verified: `fin_snipe_manager` uses 60s cycle, not 5s. Actual ~1440/day per endpoint. Still excessive; downgrade to P3 "should cache 30-60s" rather than P2 "rate-limit risk".

### Cross-agent corroboration (same defect, multiple angles)

- T4 + T5 (cross-process state locks): Agent 3 (F01-F03), Agent 8 (journal.py:28), Agent 7 (microstructure_state:205-213), Agent 2 (trigger_buffer.py RMW), Main M01-M03 (raw json.dump). Five sources → highest confidence.
- T3 (stop-loss too tight): Agent 4 (#1, #3, #4, #5, #6, #22), `metals-avanza.md` rule, `memory/feedback_mini_stoploss.md`. Three sources → highest confidence.
- T7 (BULL/BEAR direction): Agent 3 (F14, F15), Agent 4 (no direct match — but Agent 4 deliberately scoped to fishing-warrant flow which is long-only). Agent 3 finding stands alone but the system is short-warrant-curious (BEAR certs exist in `data/fin_fish.py`).
- Pattern 2 (quotas reset on restart): Agent 7 (alpha_vantage + earnings_calendar + onchain) all independent. Three sources.
- Pattern 4 (wall-clock for elapsed): Agent 2 + Agent 8 independent. Two sources.

### Gaps neither agent caught

Reading the synthesis as a whole: nobody enumerated **the dashboard's data writers**. Dashboard reads from `data/*.json` files but several endpoints accept POST (e.g. `/api/validate-portfolio`). The agents partitioned by subsystem so dashboard `app.py`/`house_blueprint.py` writes weren't audited. Carry forward to next session.

Also missing: `tests/` audit. 415 test files; the prompt scoped this out, but **test isolation patterns** (xdist safety) is a known fragile area per `docs/TESTING.md`.

---

## Implementation Roadmap (Priority Stack)

**Sprint 1 (must fix before next live trading):**
1. T3 (stop-loss limits too tight) — money-at-risk on every metals trade
2. T7 (BULL/BEAR direction in P&L + stops) — silent P&L corruption for BEAR certs
3. T9 (account filter in get_positions) — pension/ISK isolation
4. T1 + T2 (signal-engine accuracy corruption) — consensus quality
5. T4 + T5 (cross-process state locks) — data-loss class

**Sprint 2 (silent-failure paths):**
6. T6 (off-hours autonomous fallback) — 24/7 coverage gap
7. T8 (Binance error → garbage candle) — bad data → bad signals
8. T10 (silver fast-tick lock) — race condition

**Sprint 3 (quality + reliability):**
9. Pattern 2 (persist quota state) — silent API outage by lunch
10. Pattern 4 (monotonic clocks everywhere) — NTP-jump robustness
11. Pattern 5 (calendar refresh) — 18-month timer
12. Pattern 1 (cross-process lock utility) — eliminates whole class of bugs
13. M01-M03 (atomic-I/O in data/ scripts)

**Backlog (P2 + P3 by file):** carry into `docs/IMPROVEMENT_BACKLOG.md` from per-agent files.

---

## Reviewer Scorecard

| Agent | Subsystem | Coverage | Severity calibration | False-positive rate |
|-------|-----------|----------|---------------------|---------------------|
| 1 signals-core | depth: excellent | calibrated, cites code | ~0% |
| 2 orchestration | breadth: 18 files / depth: ok | calibrated, cross-references existing fixes | ~0% |
| 3 portfolio-risk | depth: excellent, math focus | well-calibrated | ~0% |
| 4 metals-core | depth: excellent (50 findings) | one self-withdrawn P1 → P3 | ~2% (1/50) |
| 5 avanza-api | tight, caveman-style one-liners | well-calibrated | ~0% |
| 6 signals-modules | breadth: all 60 modules enumerated | calibrated | ~0% |
| 7 data-external | depth: excellent | calibrated | ~0% |
| 8 infrastructure | breadth: 32 files | one over-flagged P1 (shell injection) → P3 | ~3% (1/27) |
| main thread cross-cut | cross-cutting focus, less depth per file | calibrated | n/a |

Single common failure mode across all agents: **none reviewed test fixtures** for the patterns they flagged in production code. Cross-cutting gap to address next pass.

---

## Notes for Next Implementation Session

- All agent outputs land in `docs/adversarial_review/*.md` — read raw files for full citations.
- Main-thread pass at `docs/adversarial_review/main_thread_pass.md`.
- This synthesis is opinionated about T1–T10 priority; the per-agent files are the canonical source-of-truth for finding text.
- Carry P2 + P3 items into `docs/IMPROVEMENT_BACKLOG.md` mechanically — do not let them drift.
- Run `tests/` after each batch per `docs/GUIDELINES.md`; never push red.
- Restart loops after touching anything in `data/metals_loop.py` or `portfolio/main.py --loop` per the protocol.

The full review process took ~14 minutes of parallel agent runtime + ~8 minutes of main-thread cross-cutting work. 255 findings against ~75 files, all severity-tagged, all citing file:line.
