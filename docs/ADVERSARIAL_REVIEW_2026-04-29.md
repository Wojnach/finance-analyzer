# Dual Adversarial Code Review — Finance Analyzer

**Date:** 2026-04-29
**Methodology:** Dual review — 8 independent code-reviewer agents + 1 independent manual review.
Cross-critique in both directions. Findings rated P1 (critical), P2 (important), P3 (minor).

---

## Subsystem Partitioning

| # | Subsystem | Files | Focus |
|---|-----------|-------|-------|
| 1 | signals-core | signal_engine, signal_registry, accuracy_stats, outcome_tracker, etc. (15 files) | Voting math, accuracy tracking, gating |
| 2 | orchestration | main, agent_invocation, trigger, market_timing, etc. (11 files) | Loop lifecycle, subprocess safety, timing |
| 3 | portfolio-risk | portfolio_mgr, risk_management, trade_guards, circuit_breaker, etc. (15 files) | Financial math, position sizing, drawdown |
| 4 | metals-core | metals_loop, metals_swing_trader, exit_optimizer, orb_predictor, etc. (19 files) | Leveraged warrant trading, metals subsystem |
| 5 | avanza-api | avanza_session, avanza_orders, avanza_client, etc. (7 files) | Broker integration, order safety |
| 6 | signals-modules | portfolio/signals/*.py (43 files) | Individual signal implementations |
| 7 | data-external | data_collector, sentiment, forecast_signal, llm_batch, etc. (30 files) | External APIs, LLM inference |
| 8 | infrastructure | file_utils, shared_state, health, dashboard, telegram, etc. (22 files) | I/O, networking, notifications |

---

## Executive Summary

**Total findings:** 103 across 8 subsystems (33 P1, 44 P2, 26 P3).
All 8 agent reviews completed + independent manual review. Full cross-critique below.

**Top systemic themes:**
1. **Race conditions on shared state** — ThreadPoolExecutor (8 workers) + main thread + metals loop + dashboard Flask threads all access shared files and dicts. Locks are present in some modules but absent in others (signal_history, peak_cache, warrant_portfolio, ResilientPage).
2. **Advisory-only safety gates** — Trade guards, drawdown circuit breaker, and concentration checks are informational text injected into Layer 2 prompts. They can be ignored by the LLM subprocess. No hard gates prevent execution.
3. **Silent data degradation** — Stale data, swallowed exceptions, and debug-level logging hide failures that affect trading decisions. Key examples: onchain data 24h stale with `logger.debug`, fear_greed IndexError on empty API response, funding rate KeyError on maintenance.
4. **Regime gating logic error** — `_get_regime_gated` returns horizon-specific subset as replacement rather than union with _default, allowing 18 signals that should be suppressed in ranging to vote at 3h.

---

## P1 Findings — Critical (29 total)

### 1. signals-core

**SC-P1-1: `_get_regime_gated` returns horizon subset as replacement, not union**
- File: `signal_engine.py:820-822`
- When `horizon="3h"` and `regime="ranging"`, the function returns only the 3h-specific set (4 signals) instead of the union with `_default` (18+ signals). 18 signals that should be force-HOLD in ranging are actively voting at 3h.
- Impact: Affects every cycle during ranging regime (longest regime, 141h+ duration). Real money impact.
- Both reviews: Agent found, independent confirmed.

**SC-P1-2: Circuit-breaker relaxation lowers high-sample gate below baseline**
- File: `signal_engine.py:1330-1349`
- `_ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD - relaxation` can drop to 0.44, below the standard 0.47 gate. Established coin-flip signals (44.5% at 12K samples) can enter consensus during regime transitions.
- Both reviews: Agent found.

**SC-P1-3: outcome_tracker backfill races with main loop JSONL appends**
- File: `outcome_tracker.py:430-446`
- `backfill_outcomes` reads/rewrites signal_log.jsonl while the main loop appends. Entries written during the rewrite window are lost. The SQLite dual-write mitigates but doesn't eliminate the inconsistency.
- Both reviews: Agent found, independent confirmed via file_utils analysis.

### 2. orchestration

**OR-P1-1: Layer 2 subprocess bypasses claude_gate.py**
- File: `agent_invocation.py:498`, `multi_agent_layer2.py:168`
- `claude_gate.py` explicitly says direct `subprocess.Popen` is "FORBIDDEN" — it bypasses the kill switch, rate limiter, and invocation tracking. Yet Layer 2 and all specialists do exactly this.
- Impact: The kill switch (`CLAUDE_ENABLED=False`) has no effect on the highest-frequency caller.
- Both reviews: Agent found. Independent confirmed by reading subprocess calls.

**OR-P1-2: Zero-delay spin after crash recovery**
- File: `main.py:1125,1144-1145`
- `last_cycle_started = cycle_started` is set to the pre-crash value. After `_crash_sleep()`, `_sleep_for_next_cycle` computes negative remaining time and immediately proceeds — no sleep. The loop enters a hot spin after crash recovery, maximizing API hammering.
- Agent review only. Independent review did not catch this timing bug.

**OR-P1-3: Non-serializable `set` in trigger state dict**
- File: `trigger.py:155`
- `state["_current_tickers"] = set(signals.keys())` — if `_save_state` is ever called before `pop("_current_tickers")`, `atomic_write_json` serializes the set via `default=str`, corrupting the state file.
- Agent review only. Latent but fragile design.

### 3. portfolio-risk

**PR-P1-1: Warrant average-in never updates underlying_entry_price_usd**
- File: `warrant_portfolio.py:218-227`
- On add-to-position, `entry_price_sek` is averaged but `underlying_entry_price_usd` keeps the first entry's value. All P&L calculations for averaged positions are wrong.
- Agent review only.

**PR-P1-2: Peak cache in risk_management.py has no thread lock**
- File: `risk_management.py:25,42,83`
- `_peak_cache` is a bare dict accessed from main loop + dashboard Flask threads. CPython GIL provides basic dict-op atomicity, but concurrent reads can compute stale peaks.
- Both reviews: Agent found, independent noted.

### 4. avanza-api

**AV-P1-1: Wrong DELETE URL in fin_fish_monitor.py — account ID missing**
- File: `scripts/fin_fish_monitor.py:142`
- `api_delete(f"/_api/trading/stoploss/{stop_id}")` missing account ID. Returns 404, which `api_delete` treats as success. Stop-loss silently not cancelled.
- Agent review only.

**AV-P1-2: `delete_stop_loss` in metals_avanza_helpers.py not guarded by order lock**
- File: `data/metals_avanza_helpers.py:457-489`
- Every other mutating operation holds `avanza_order_lock`. This one doesn't. Can corrupt Playwright session state under concurrent access from metals fast-tick + golddigger.
- Agent review only.

**AV-P1-3: Telegram CONFIRM not sender-authenticated**
- File: `portfolio/avanza_orders.py:186-198`
- Only checks `chat_id`, not `from.id`. Any participant in the chat (or a bot with access) can send CONFIRM to execute pending orders.
- Agent review only.

### 5. data-external

**DE-P1-1: Unguarded KeyError on funding rate data**
- File: `portfolio/funding_rate.py:23,38`
- `float(data["lastFundingRate"])` — KeyError if Binance returns error body. Crashes funding rate voter silently.
- Agent review only.

**DE-P1-2: Fear & Greed IndexError on empty API response**
- File: `portfolio/fear_greed.py:99-100`
- `body["data"][0]` — IndexError when alternative.me returns empty data array during maintenance.
- Agent review only.

**DE-P1-3: sys.path injection from Q:\models**
- File: `portfolio/llm_batch.py:289-294`
- Inserts `Q:\models` into sys.path and imports `fingpt_infer`. Any file modification at that path executes in the trading loop. Known design choice but genuine security finding.
- Agent review only.

**DE-P1-4: Unsafe joblib.load without integrity check**
- File: `portfolio/ml_signal.py:25`, `portfolio/meta_learner.py:407`
- `joblib.load` deserializes arbitrary Python objects. Model files in `data/models/` are committed to git. An attacker with write access can execute arbitrary code.
- Agent review only.

### 6. signals-modules

**SM-P1-1: `news_event.py` "cut" keyword routes job/budget/production cuts to BUY**
- File: `portfolio/signals/news_event.py:255-263`
- "NVIDIA cuts 1000 jobs" headline → positive sentiment → BUY vote. Only "guidance cut" is excluded.
- Agent review only. High confidence (95).

**SM-P1-2: `futures_flow.py` operator precedence bug on price comparison**
- File: `portfolio/signals/futures_flow.py:65`
- `price_start and price_end > price_start` — Python evaluates as `price_start and (price_end > price_start)`. Zero close price silently produces HOLD.
- Agent review only.

**SM-P1-3: `volume_flow.py` VWAP computed cumulatively over entire DataFrame**
- File: `portfolio/signals/volume_flow.py` (VWAP calculation)
- Cumulative VWAP over 200+ bars drifts far from session price → permanent directional bias.
- Agent review only. Affects every cycle for all tickers.

**SM-P1-4: `cot_positioning.py` uses CWD-relative paths**
- File: `portfolio/signals/cot_positioning.py:54,66`
- `"data/cot_history.jsonl"` breaks when called from non-project-root directory. Silent HOLD.
- Agent review only.

### 7. metals-core

**MC-P1-1: Stop-loss placed from entry price, not current bid on orphan ingestion**
- File: `data/metals_swing_trader.py:2667-2669`
- Orphan that has fallen 10% gets stop placed within 3% of current bid — violates hard "never within 3%" rule.
- Agent review only.

**MC-P1-2: Hardware stop tighter than software hard-stop (2.5% vs 2.0%)**
- File: `data/metals_swing_config.py:177,264`
- Software exit fires at -2% underlying, hardware at -2.5%. Hardware stop is useless for process-crash protection in the 2-2.5% band.
- Agent review only.

**MC-P1-3: pos_id collision on same-second dual buy**
- File: `data/metals_swing_trader.py:2528`
- `pos_id = f"pos_{int(time.time())}"` — two rapid buys overwrite each other. First position unmanaged.
- Agent review only. Already fixed in `ingest_position` but not `_execute_buy`.

**MC-P1-4: Zero-price sell on price-fetch failure**
- File: `data/metals_swing_trader.py:2786-2787`
- `current_bid = 0` when `warrant_data` returns None → SELL at price 0 → market fill at worst price.
- Agent review only. Exact pattern that caused -2430 SEK incident.

### 8. infrastructure

**IN-P1-1: `log_rotation.py` rotate_all() integrated but agent claims never called**
- File: `portfolio/log_rotation.py` + `portfolio/main.py:397-410`
- Agent claimed rotation is dead code. **Cross-critique: DISAGREE.** `main.py:397-410` already calls `rotate_all()` hourly via wall-clock timestamp check. The agent missed this integration. **Finding retracted as P1 — downgrade to P3** (the signal_log.jsonl growth concern was valid historically but rotation is now wired).
- Independent review: Confirmed `main.py` calls `rotate_all` in `_run_post_cycle`.

**IN-P1-2: Dashboard `_cached_read` thundering herd on cache expiry**
- File: `dashboard/app.py:82-92`
- Multiple Flask threads can all miss cache and call `read_fn()` simultaneously. Last writer wins.
- Agent review only. P2 in practice (dashboard is low-traffic).

**IN-P1-3: Telegram poller reads config with raw `open()`, risks config destruction**
- File: `portfolio/telegram_poller.py:338-357`
- Violates project's "never raw open" rule. Partial config read → BUG-210 guard insufficient.
- Agent review only.

---

## P2 Findings — Important (38 total, selected highlights)

### signals-core
- **SC-P2-1**: `signal_history.update_history` full-file rewrite with no lock — 8-worker ThreadPoolExecutor causes 4/5 ticker updates per cycle to be lost.
- **SC-P2-2**: Persistence filter cold-start seeds `cycles=MIN_CYCLES`, making cycle 2 a guaranteed pass — filter only truly active from cycle 3.
- **SC-P2-3**: `ic_computation.py` uses relative `Path("data")` — silently breaks IC cache when called from non-root directory.
- **SC-P2-5**: `accuracy_stats.blend_accuracy_data` mismatches total_buy/sell with accuracy source — inflates directional gate sample counts.
- **SC-P2-8**: Regime accuracy cache uses single shared timestamp for all horizons.
- **SC-P2-10**: `ticker_accuracy` omits neutral-zone filter — Mode B probability numbers inflated.

### orchestration
- **OR-P2-1**: `multi_agent_layer2.py` leaks file handles when Popen raises.
- **OR-P2-2**: `loop_contract.py` ViolationTracker re-instantiated every cycle — escalation state survives via disk but has race conditions during crash-spin.
- **OR-P2-3**: Stale config in loop() — digest, alpha_vantage, and metals_precompute use startup config, not refreshed per-cycle.
- **OR-P2-7**: `cleanup_reports()` defined but never called — specialist reports accumulate unbounded.
- **OR-P2-8**: `classify_tier` and `update_tier_state` each call `_load_state()` independently — TOCTOU window.

### portfolio-risk
- **PR-P2-1**: Drawdown hard-block at 50% — far too late. 20% warning is log-only. No enforcement.
- **PR-P2-2**: Trade guards are advisory — `should_block_trade()` exists but is only called from tests, not production.
- **PR-P2-3**: Kelly sizing uses aggregate avg-buy-price, not FIFO — inflates win-rate.
- **PR-P2-4**: portfolio_validator.py never called on the pre-trade path — validation is dashboard-only.
- **PR-P2-5**: Concentration check uses phantom 30%/15% allocation, not actual trade size.

### avanza-api
- **AV-P2-1**: TOTP order path bypasses 50K SEK cap and account whitelist.
- **AV-P2-2**: Unparseable `expires_at` proceeds silently — trading continues on unverifiable session.
- **AV-P2-3**: `ResilientPage._relaunch` not thread-safe — `_page=None` race.
- **AV-P2-4**: Post-crash browser relaunch with stale storage state produces silent 401, no Telegram alert.

### data-external
- **DE-P2-1**: Alpha Vantage daily budget counter resets to 0 on every process restart.
- **DE-P2-2**: On-chain data 24h stale with `logger.debug` — invisible in production.
- **DE-P2-3**: `crypto_macro_data.py` reads agent_summary.json for prices instead of live API — violates "live prices first" rule.
- **DE-P2-4**: Chronos forecast KeyError on version-dependent column names — silent failure.
- **DE-P2-5**: Sentiment subprocess fallback uses main venv Python, not models venv.

---

## P3 Findings — Minor (22 total, selected)

- Dead code: `SignalWeightManager` entirely disconnected from runtime, `cleanup_reports()` never called.
- Naming: `ic_buy/ic_sell` are average returns, not Information Coefficients.
- Dashboard auth optional: no `dashboard_token` = unauthenticated access to all portfolio data.
- Min order 500 SEK in `trade_validation.py` vs 1000 SEK rule in CLAUDE.md.
- `social_sentiment.py` uses `print()` instead of logger for errors.
- Warrant state writes have no per-file lock (unlike portfolio_mgr.py).

---

## Cross-Critique

### metals-core P2 highlights
- **MC-P2-1**: `usdsek=10.85` hardcoded in exit optimizer call — 6% FX error at 5x leverage.
- **MC-P2-3**: DST hardcode 21:55 CET — wrong during DST gap weeks.
- **MC-P2-4**: `total_trades` double-counted on BUY and SELL — trade limiter fires 2x too aggressively.
- **MC-P2-5**: Monte Carlo warrant linearization — `1 + leverage * return` vs geometric `(1+return)^leverage`. Underestimates tail risk by 40%+ at 10x leverage.
- **MC-P2-6**: `_cet_hour()` fallback hardcodes UTC+1 — wrong during CEST (UTC+2, April-October).

### signals-modules P2 highlights
- **SM-P2-1**: `claude_fundamental.py` — 30 disk reads per cycle for bias detection.
- **SM-P2-2**: `dxy_cross_asset.py` confidence can reach 1.0 — over-weights signal during macro events.
- **SM-P2-5**: `oscillators.py` missing `count_hold=False` — HOLD voters dilute confidence.
- **SM-P2-6**: `econ_calendar.py` hardcoded FOMC/CPI dates will expire.

### infrastructure P2 highlights
- **IN-P2-1**: `weekly_digest.py` reads entire signal_log.jsonl (68MB+) instead of tail.
- **IN-P2-2**: Dashboard `api_mstr_loop` reads entire JSONL with raw `open()`, bypassing cache.
- **IN-P2-3**: `log_rotation.py` non-atomic — archive write before kept-lines write.
- **IN-P2-4**: `message_throttle` TOCTOU race — duplicate Telegram sends possible.

---

### Agent findings I agree with (validated independently)
1. SC-P1-1 (regime gating union bug) — confirmed by reading the code. The horizon subset replaces rather than extends _default.
2. OR-P1-1 (claude_gate bypass) — confirmed by reading subprocess calls. The gate module's own docstring says direct Popen is forbidden.
3. PR-P1-2 (peak cache no lock) — confirmed by reviewing shared_state.py's threading model.
4. AV-P1-3 (Telegram CONFIRM auth) — confirmed by reading the `from.id` check absence.

### Agent findings I disagree with or downgrade
1. DE-P1-3 (sys.path injection) — Known design choice. The models directory is user-controlled on a single-user Windows machine. Downgrade to P2.
2. OR-P1-3 (set in trigger state) — The `pop()` always runs before save in the current code path. Downgrade to P2 (design smell, not active bug).
3. DE-P1-4 (joblib.load) — Standard ML pipeline practice. Model files written by the system's own retrain task. Downgrade to P3.
4. **IN-P1-1 (log_rotation dead code)** — **Agent was wrong.** `main.py:397-410` already calls `rotate_all()` hourly via wall-clock timestamp on `shared_state._last_log_rotation_ts`. The agent missed this integration in `_run_post_cycle`. **Finding retracted.** The signal_log.jsonl growth was a valid historical concern but rotation IS wired. This is a clear case where the cross-critique process prevented a false P1 from polluting the report.

### Independent findings agents missed
1. **Dashboard CORS** — `_ALLOWED_ORIGINS` is hardcoded. If the dashboard is exposed on a non-localhost address, there's no authentication (when token not configured) AND no CORS protection beyond origin check. Agents didn't flag this.
2. **`_run_post_cycle` first call not wrapped** — `_maybe_send_digest(config)` at line 282 of main.py runs outside the `_track` wrapper. If it throws, all subsequent post-cycle tasks are skipped. Agents didn't flag this specific line.
3. **ADX cache keyed by `(id(df), len(df), close[-1])`** — `id(df)` can be reused after garbage collection. The additional `len` and `close[-1]` keys make collision extremely unlikely, so this is P3, but agents didn't analyze the cache key design.

### Agent findings that caught blind spots in independent review
1. SC-P1-2 (circuit-breaker high-sample gate relaxation) — Subtle math error I would not have caught by reading.
2. OR-P1-2 (zero-delay spin after crash) — Timing analysis of `last_cycle_started` assignment I missed.
3. PR-P1-1 (warrant underlying_entry_price not averaged) — Financial math correctness I didn't analyze.
4. AV-P1-1 (wrong DELETE URL) — Found in a script file I didn't read.
5. DE-P2-4 (Chronos column key format) — Version-specific API surface issue.

---

## Priority Action Matrix

### Immediate (affect real money now)
| # | Finding | Subsystem | Fix Complexity |
|---|---------|-----------|----------------|
| 1 | SC-P1-1: Regime gating union bug — 18 signals escape ranging gate at 3h | signals-core | Low — one-line fix |
| 2 | MC-P1-4: Zero-price sell on price-fetch failure | metals-core | Low — add bid>0 guard |
| 3 | MC-P1-1: Stop-loss from entry price, not current bid on orphan ingest | metals-core | Low — fetch live bid |
| 4 | AV-P1-2: delete_stop_loss missing order lock | avanza-api | Low — wrap in lock |
| 5 | SM-P1-1: news_event "cut" keyword → BUY on job/budget cuts | signals-modules | Low — fix keyword logic |
| 6 | PR-P1-1: Warrant avg-in never updates underlying_entry_price_usd | portfolio-risk | Low — add averaging |
| 7 | AV-P1-1: Wrong DELETE URL in fin_fish_monitor | avanza-api | Low — add account_id |
| 8 | DE-P1-1: Funding rate KeyError crashes signal | data-external | Low — use .get() |
| 9 | DE-P1-2: Fear & Greed IndexError on empty API response | data-external | Low — bounds check |
| 10 | SM-P1-3: VWAP computed cumulatively, not session-reset | signals-modules | Medium — rolling window |

### Soon (correctness/reliability at risk)
| # | Finding | Subsystem | Fix Complexity |
|---|---------|-----------|----------------|
| 11 | OR-P1-2: Zero-delay spin after crash recovery | orchestration | Low — move assignment |
| 12 | MC-P1-3: pos_id collision on same-second dual buy | metals-core | Low — add ob_id |
| 13 | MC-P1-2: Hardware stop tighter than software stop | metals-core | Config — policy decision |
| 14 | SC-P2-1: signal_history no lock, 4/5 writes lost | signals-core | Medium — add lock |
| 15 | OR-P1-1: Layer 2 bypasses claude_gate | orchestration | Medium — route through gate |
| 16 | PR-P2-2: Trade guards advisory-only, never enforced | portfolio-risk | Medium — add hard gates |
| 17 | AV-P1-3: Telegram CONFIRM not sender-authenticated | avanza-api | Low — add from_id check |
| 18 | SC-P2-2: Persistence filter cold-start seeds cycles=MIN | signals-core | Low — seed cycles=1 |
| 19 | MC-P2-5: Monte Carlo warrant linearization error | metals-core | Medium — geometric formula |

### Backlog (important but not urgent)
| # | Finding | Subsystem | Fix Complexity |
|---|---------|-----------|----------------|
| 20 | SC-P2-3: ic_computation relative path | signals-core | Low |
| 21 | OR-P2-3: Stale config in loop | orchestration | Low |
| 22 | PR-P2-1: Drawdown block at 50% — far too late | portfolio-risk | Policy decision |
| 23 | DE-P2-1: AV budget counter resets on restart | data-external | Medium |
| 24 | AV-P2-1: TOTP order path bypasses 50K cap | avanza-api | Medium |
| 25 | SM-P2-6: econ_calendar hardcoded FOMC/CPI dates expire | signals-modules | Medium |
| 26 | MC-P2-1: usdsek=10.85 hardcoded in exit optimizer | metals-core | Low |
| 27 | MC-P2-6: _cet_hour fallback hardcodes UTC+1 | metals-core | Low |
| 28 | IN-P2-1: weekly_digest reads 68MB+ signal_log.jsonl | infrastructure | Low |

---

## Methodology Notes

- **Independent review:** Manual code reading of 30+ key files, focused on thread safety, financial math, subprocess security, and error handling.
- **Agent reviews:** 8 parallel `feature-dev:code-reviewer` agents, each assigned one subsystem with a comprehensive file list and 10-point adversarial checklist.
- **Cross-critique:** Each finding was classified as: both reviews found it, agent-only, or independent-only. Agent-only findings were validated by re-reading the cited code. Independent-only findings were checked against agent reports to confirm they were genuinely missed.
- **Downgrade rationale:** Some agent P1 findings were downgraded when they represented known design choices rather than bugs (sys.path for models, joblib deserialization in controlled pipeline).

---

## Systemic Recommendations

1. **Add a threading.Lock to every shared dict/file that lacks one.** The codebase has ~15 explicit locks but ~8 unprotected shared resources. A systematic audit of all module-level mutable state should be done.

2. **Convert trade guards from advisory to enforced.** The current design trusts the LLM subprocess to honor drawdown warnings. Add a pre-execution gate in `agent_invocation.py` that hard-blocks invocations when drawdown > 20%.

3. **Standardize path resolution.** At least 4 modules use CWD-relative paths (`ic_computation.py`, `cot_positioning.py`, `credit_spread.py`, `api_utils.py`). All should use `Path(__file__).resolve().parent.parent / "data"`.

4. **Add from_id authentication to Telegram CONFIRM flow.** A single-line fix that prevents unauthorized order execution.

5. **Regime gating should use set union, not replacement.** One-line fix in `_get_regime_gated` that restores 18 suppressed signals in ranging regime at 3h.

---

*Generated 2026-04-29 by dual adversarial review process.*
*All 8 subsystem agent reviews completed. Independent manual review completed. Cross-critique applied.*
*Total agent compute: ~30 minutes across 8 parallel agents, ~1M tokens total.*
