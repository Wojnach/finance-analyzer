# Changelog

## 2026-04-21 (auto-improve: safety-critical fixes)

10 bug fixes addressing findings from 3 consecutive adversarial reviews.

- **BUG-209: OHLCV zero/negative price validation (portfolio/indicators.py)**: `compute_indicators()` now rejects zero/negative close prices (returns None). Previously these produced RSI=50, MACD=0, ATR=0 — poisoning all 33 downstream signals during Binance maintenance windows.
- **BUG-210: Config wipe guard (portfolio/telegram_poller.py)**: Refuse to overwrite config.json when loaded config has <5 keys. Prevents catastrophic API key destruction from transient file access issues.
- **BUG-211: Max order size limit (portfolio/avanza_session.py)**: Added MAX_ORDER_TOTAL_SEK = 50,000 guard in `_place_order()`. Prevents total account exposure from a single malformed LLM call.
- **BUG-212: Rate limiter sleep-outside-lock (portfolio/shared_state.py)**: `_RateLimiter.wait()` now sleeps outside the lock, preventing priority inversion across 8 worker threads.
- **BUG-213: _loading_timestamps cleanup (portfolio/shared_state.py)**: `_cached()` success path now cleans up `_loading_timestamps` (previously leaked until 120s eviction).
- **BUG-214: Drawdown circuit breaker wired in (portfolio/agent_invocation.py)**: `check_drawdown()` now called before every Layer 2 invocation — first-ever automated risk gate on the primary trading path. Advisory at >20%, hard-block at >50%.
- **BUG-215: Thread-safe FX cache (portfolio/fx_rates.py)**: Added `threading.Lock` to `_fx_cache` dict accessed from 8-worker ThreadPoolExecutor.
- **BUG-216: Monte Carlo random seeds (3 files)**: `seed=42` → `seed=None` in monte_carlo.py, monte_carlo_risk.py, and data/metals_risk.py. Production risk metrics now use system entropy.
- **BUG-217: Metals sell exception safety (data/metals_swing_trader.py)**: `_execute_sell()` exceptions now caught per-position with `sell_failed_at` marking, preventing one failed sell from aborting the entire exit loop.
- **BUG-218: econ_calendar disabled (portfolio/tickers.py)**: Force-HOLD — all 4 sub-signals were structurally SELL-only, never BUY. Removes systematic SELL bias from consensus.
- **Dashboard timing attack fix (dashboard/app.py)**: Token comparison switched from `==` to `hmac.compare_digest()`.
- **Journal atomic write (portfolio/journal.py)**: `CONTEXT_FILE.write_text()` replaced with `atomic_write_text()`.
- **file_utils: added `atomic_write_text()` utility**.
- **21 new tests** in `tests/test_safety_guards.py` covering all fixes.
- Theme: Safety Debt Paydown, Risk Enforcement, Thread Safety.

## 2026-04-20 (outcome-tracking repair)

- **Fix: fin_evolve.by_command dynamically enumerates all /fin-* commands
  (portfolio/fin_evolve.py)**: Replaced hardcoded `for cmd in ("fin-silver",
  "fin-gold"):` with enumeration of commands present in scored data.
  fin-crypto, fin-mstr, fin-btc, fin-eth were previously logged to
  `fin_command_log.jsonl` but invisible in `system_lessons.by_command`.
- **Fix: live-price API fallback in _find_price_at (portfolio/fin_evolve.py)**:
  MSTR hourly snapshots only cover ~42% of slots (US-market-only), which
  silently blocked outcome scoring when +1d/+3d landed off-hours. Added
  opt-in fallback to `outcome_tracker._fetch_historical_price` (Binance
  for crypto, yfinance for stocks). Exception-safe, logs at WARNING so a
  sustained upstream outage surfaces in production logs.
- **Fix: multi-ticker fin-crypto backfill (portfolio/fin_evolve.py)**:
  `backfill_outcomes()` now detects the `tickers` list + per-ticker nested
  blocks (entry["btc"], entry["eth"], entry["mstr"]) and scores each block
  independently. `_collect_scored_fin_entries()` expands fin-crypto entries
  into per-ticker virtual records for the lesson aggregator.
- **Impact**: After the first backfill run, total scored verdicts jumped
  705 → 937 (+232). MSTR appears in `system_lessons.by_ticker` for the
  first time (n_total=74, n_evaluable=5, accuracy_3d=1.0). fin-crypto now
  visible in `by_command` with 42 scored entries. 94 new tests added to
  `tests/test_fin_evolve.py`.

## 2026-04-17 (momentum-entries session)

- **Feature: entry-side upside-momentum detector (metals_loop.py)**:
  Mirrors the exit-side `_silver_fast_tick` in the opposite direction and
  is NOT gated on having an active position. Writes momentum candidates
  to `data/metals_momentum_state.json` when silver rises ≥0.8%/3min or
  gold rises ≥0.4%/3min with RVOL ≥1.5 and a 5-minute dedup window.
  8 new tests in `tests/test_metals_entry_fasttick.py`.
- **Feature: swing-trader momentum-entry path (metals_swing_trader.py,
  metals_swing_config.py)**: On a fresh LONG momentum candidate, relax
  `MIN_BUY_CONFIDENCE` (0.60 → 0.50) and `MIN_BUY_VOTERS` (3 → 2). All
  other entry gates (RSI zone, MACD improving, regime confirm, TF
  alignment, strict majority) unchanged — those defend against false
  breakouts which is exactly what you want active under momentum. After
  `_execute_buy` the candidate is marked consumed so the same burst does
  not re-trigger. 13 new tests in `tests/test_metals_swing_momentum.py`.
- **Fix: adaptive sizing on Kelly-fallback path (metals_swing_trader.py)**:
  Fallback branch `alloc = cash * POSITION_SIZE_PCT / 100` had no
  `MIN_TRADE_SEK` floor; with `cash=2822 SEK × 30% = 847 SEK` entries
  were silently rejected by the `alloc < MIN_TRADE_SEK` skip branch.
  Now mirrors Kelly-primary behaviour: floor at 1000 SEK, cap at 95% of
  cash, skip only when `cash × 0.95 < MIN_TRADE_SEK`. 4 new tests in
  `tests/test_metals_swing_sizing.py`.
- **Deprecation: fish engine permanent (metals_loop.py)**: Replace the
  stale "re-enable after 6 bugs fixed" comment with the 2026-04-15
  12,257 SEK loss context and a two-step revival guard — future
  operators must flip `FISH_ENGINE_ENABLED` AND remove
  `_assert_fish_engine_allowed` to re-activate. The 6 integration bugs
  ARE fixed; the strategy itself has no measurable edge. Swing trader's
  new momentum-entry path supersedes it for upside breakouts. 5 new
  tests in `tests/test_fish_engine_deprecated.py`.
- **Root cause context**: 2026-04-17 silver breakout moved +1.3% in
  20 min on RVOL 2.2x while the metals loop's fast-tick machinery was
  dark (gated on active silver position) and the swing trader's
  snapshot gates required two regime-confirm cycles (by which time the
  move was half over). Four architectural gaps — sizing floor, absent
  entry detector, no momentum entry path, stale fish-engine comment —
  compounded into a silently-rejected full-conviction trade.
- **Plan**: `docs/plans/2026-04-17-momentum-entries-plan.md`
- **Test delta**: +30 new tests, 1 regression test fixed; 92/92 touched
  tests green.
- Theme: Breakout Entry Detection, Adaptive Sizing, Deprecation Hygiene.

## 2026-04-16 (autonomous improvement session)
- **BUG-200: Layer 2 auth-failure bypass in bigbet (P1)**: `bigbet.invoke_layer2_eval` called `subprocess.run(["claude", "-p", ...])` directly, bypassing `detect_auth_failure` — when OAuth expired, claude exited 0 with "Not logged in" on stdout but the bypass meant no `critical_errors.jsonl` entry. Startup check never surfaced the issue. Fixed: scan stdout+stderr before trusting parse, return `(None, "")` + record critical entry with `caller="bigbet_layer2"` on auth failure.
- **BUG-201: iskbets default-approve safety gap (P0)**: `iskbets.invoke_layer2_gate` had the same bypass, but `_parse_gate_response` defaults `approved=True` when it can't find a DECISION line — "Not logged in" output would have been interpreted as APPROVED for a warrant trade. Fixed: auth failure detection overrides default-approve to `approved=False` before the parser sees the output.
- **BUG-201b: analyze CLI auth-failure silence (P2)**: `analyze.run_analysis` (manual CLI path) had the same bypass. Doesn't run autonomously but still records critical entry + prints user-visible re-login hint so `check_critical_errors.py` surfaces the issue in the next session.
- **BUG-202: LAYER2_JOURNAL_GRACE_S too wide (P1)**: `loop_contract.py:42` used 60-minute grace before the journal-activity contract fires. Three consecutive overnight auth-silent outages (Apr 14–16) fell inside the grace window and went undetected. Tightened to 18 min (T3's 15-min subprocess cap + 3 min slack for spawn/Telegram/journal flush). Pin test added.
- **BUG-203: Wall-clock elapsed in agent_invocation (P3)**: `_agent_start = time.time()` + `elapsed = time.time() - _agent_start` was NTP-jump-susceptible. Switched to `time.monotonic()` for elapsed math. Wall clock still used for log-entry timestamps.
- **BUG-204: Silent exception in qwen3 GPU reaper (P3)**: `try: kill_orphaned_llama() except Exception: pass` made VRAM leaks invisible if the reaper itself broke. Promoted to `logger.debug(exc_info=True)`.
- **BUG-205: Silent exception in dashboard market_health (P3)**: Same treatment for the optional `market_health` enrichment in `dashboard/app.py`. Added module-level logger.
- **New tests**: 6 new tests — `test_auth_failure_bypass` (5: bigbet/iskbets/analyze paths + healthy-output negatives), `test_grace_window_is_18_minutes` pin. All pass.
- **Lint cleanup**: `scripts/verify_kronos.py` — SIM105, F541, E741 fixed.
- **Root cause**: Three consecutive overnight Layer 2 outages — detection infrastructure (`detect_auth_failure`, `record_critical_error`, journal-activity contract) was sound but bypass sites prevented escalation and the grace window hid the silent-stall signal.
- Theme: Auth-Failure Detection, Silent-Failure Visibility.

## 2026-04-15 (autonomous improvement session)
- **BUG-196: Relative path fragility (P2)**: 6 modules (`microstructure_state`, `fear_greed`, `seasonality`, `linear_factor`, `signal_weight_optimizer`, `train_signal_weights`) used `Path("data/...")` relative to CWD instead of `Path(__file__).resolve().parent.parent / "data"`. Now all 6 use the absolute pattern, matching the 40+ other modules.
- **BUG-197: Dead timestamp code (P3)**: `agent_invocation.py:691` computed `ts_str_clean` but never used it — Python 3.12 `fromisoformat()` handles `+00:00` and `Z` natively. Simplified to direct `fromisoformat()` call.
- **BUG-198: Signal registry import spam (P2)**: `signal_registry.py:78-89` re-attempted broken imports on every call (35×/cycle). Now caches failures with 5-min cooldown via sentinel value. Warning logged once per cooldown.
- **BUG-199: Trigger sustained gate duplication (P3)**: Extracted shared `_update_sustained()` helper from signal flip (section 2) and sentiment reversal (section 5) in `trigger.py`. Reduces 40 lines to 20, ensures changes apply to both paths.
- **Lint cleanup**: 9 unused imports removed (F401 → 0), 8 ruff violations fixed (UP035, SIM102, SIM105, SIM118, E731, I001). Portfolio ruff violations 67 → 59 (remaining: 54 E402 intentional, 5 SIM115 intentional).
- **New tests**: 12 new tests — `_update_sustained` unit tests (8), signal registry import caching (4). All pass.
- Theme: Path Safety, Code Quality, DRY, Lint Compliance.

## 2026-04-12 (autonomous improvement session)
- **BUG-185: Directional accuracy KeyError safety (P2)**: `signal_engine.py:861,863` accessed `stats["buy_accuracy"]` without `.get()`. Falls back to overall accuracy if key is missing (cache corruption defense).
- **BUG-186: Blended accuracy `correct` field (P3)**: `accuracy_stats.py:650` `correct` count was always from all-time data even when `accuracy` was a 70/30 recent/alltime blend. Now derives `correct = round(blended * total)` so `correct/total ≈ accuracy`.
- **BUG-187: Circuit breaker dead code (P3)**: `circuit_breaker.py:89-92` HALF_OPEN probe branch was unreachable — probe is always sent via OPEN→HALF_OPEN transition. Replaced with comment explaining probe lifecycle.
- **BUG-188: Redundant acc_horizon (P3)**: `signal_engine.py:1826` duplicate computation removed (already computed at line 1813).
- **BUG-189: Windows taskkill rc=128 (P2)**: `agent_invocation.py:175` treated rc=128 (process already exited) as failure, blocking Layer 2 for the cycle. Now treated as success. Wait timeout increased 10s→15s for Claude CLI Node.js teardown.
- **BUG-190: Digest tail read (P3)**: `digest.py:65,99` used `load_jsonl()` (full file read) for invocations and journal. Replaced with `load_jsonl_tail()` (reads last 512KB only).
- **New tests**: 13 new tests — directional accuracy missing keys (2), blend accuracy consistency (6), circuit breaker probe lifecycle (5). All pass.
- Theme: Safety, Reliability, Performance.

## 2026-04-08 (autonomous improvement session)
- **BUG-176: Concentration check uses cash-only allocation (P1)**: `risk_management.py:585` computed proposed allocation as `cash * alloc_pct` instead of `min(total_value * alloc_pct, cash)`. Now uses portfolio-proportional sizing, capped at available cash.
- **BUG-177: Sortino ratio unit inconsistency (P3)**: `equity_curve.py:244` used inline `r / 100` while Sharpe used pre-computed `daily_rets_dec`. Math was correct but confusing. Now uses `daily_rets_dec` consistently.
- **BUG-178: No timeout on main loop ThreadPoolExecutor (P1)**: `main.py:514` `as_completed(futures)` could hang indefinitely on stuck signal computation. Added 120s timeout with graceful degradation — timed-out tickers are cancelled, cycle continues with partial results.
- **BUG-179: No timeout on data_collector ThreadPoolExecutor (P1)**: `data_collector.py:325` same issue for per-ticker timeframe collection. Added 60s timeout. Partial results returned for completed timeframes.
- **BUG-180: ADX cache full-clear eviction (P2)**: `signal_engine.py:982` cleared all 200 cache entries on overflow. Now evicts oldest 50% (LRU-style using Python dict insertion order), keeping recent entries warm.
- **BUG-181: Fishing context stale data on failure (P2)**: `agent_invocation.py:432` exception handler left old `fishing_context.json` on disk. Now writes neutral context (direction_bias=neutral, confidence=0) on failure to prevent stale bias misleading fish engine.
- **BUG-182: GPU lock breaks stale without PID check (P2)**: `gpu_gate.py:126` broke stale locks based only on file mtime (>300s). Now validates owning PID is dead via `psutil.pid_exists()` before breaking, preventing legitimate long-running model loads from being interrupted.
- **ARCH-29: Trade guard `should_block_trade()` helper**: Added convenience function to `trade_guards.py` that returns True if any warning has severity="block". Gives Layer 2 a clean go/no-go signal.
- **New tests**: 16 new tests — ThreadPoolExecutor timeout (2), ADX LRU eviction (1), should_block_trade (6), GPU PID check (4), fishing context fallback (2), main helpers regression (1). All pass.
- Theme: Reliability, Defensive Programming, Risk Accuracy.

## 2026-04-07 (autonomous improvement session)
- **BUG-171: Silent exception swallowing (P2)**: ~14 `except Exception: pass` patterns across 10 modules. Cleanup/teardown paths converted to `contextlib.suppress(Exception)`. Operational paths (llama_server, avanza/scanner) get `logger.debug()` for traceability.
- **BUG-172: fin_fish.py deprecated datetime (P3)**: `datetime.timezone.utc` → `datetime.UTC` (UP017).
- **BUG-173: orchestrator.py deprecated import (P3)**: `typing.Callable` → `collections.abc.Callable` (UP035).
- **BUG-174: Unused import (P3)**: Removed unused `pathlib.Path` import from golddigger_strategy.py.
- **REF-45: Collapsible if statements**: 9 SIM102 violations fixed across 8 files — nested ifs combined with `and`.
- **REF-46: If-with-same-arms**: 3 SIM114 fixes — indicators.py, crypto_macro.py branches combined with `or`.
- **REF-47: Suppressible exceptions**: 2 SIM105 fixes in bot runners — `try/except/pass` → `contextlib.suppress`.
- **REF-48: Test file cleanup**: 91 auto-fixes in 34 test files (50 F401, 21 I001, 8 UP017, 4 SIM300, etc).
- **REF-49: Needless bool**: SIM103 in daily_digest.py — return negated condition directly.
- **Lint progress**: Portfolio ruff violations 74 → 56 (remaining: E402 intentional, SIM115 intentional). Test violations 253 → 170 (remaining: F841 test vars, SIM117 style, E741 naming).
- Theme: Code Quality, Silent Failure Prevention, Lint Compliance.

## 2026-04-04 (autonomous improvement session)
- **BUG-168: llama_server.py dead assignment (P3)**: `_ensure_model()` assigned `_local_model = name` without `global` declaration, creating and discarding a local variable. Fix: removed the dead assignment — PID file is the real cross-process guard.
- **BUG-169: Regime cache thread safety (P3)**: `_regime_cache` in `shared_state.py` accessed without lock from 8 concurrent ThreadPoolExecutor threads. The check-then-clear pattern was racy. Fix: added `_regime_lock`, wrapped access in `indicators.detect_regime()`. Computation remains outside lock.
- **BUG-170: fear_greed.py non-atomic write (P3)**: Streak file used `write_text()` instead of `atomic_write_json()`. Also replaced raw `json.loads()` reads with `load_json()` for consistency. Fix: uses atomic_write_json + load_json throughout.
- **REF-39**: Ruff auto-fixes — 2 I001 (import sort), 1 UP015 (redundant open mode), 1 UP017 (datetime.UTC alias), 1 F401 (unused timezone import).
- **REF-40**: Prefixed 2 unused loop variables with `_` in `llm_batch.py`.
- **REF-41**: Renamed ambiguous variable `l` to `line`/`lo` in `log_rotation.py`, `heikin_ashi.py`, `mean_reversion.py` (6 occurrences).
- **REF-42**: Merged duplicate `isinstance()` call in `calendar_seasonal.py`.
- **REF-43**: Converted 2 lambda assignments to `def` functions in `avanza_control.py`, `avanza_session.py`.
- **REF-44**: Converted `except Exception: pass` to `contextlib.suppress()` in `main.py`.
- **BUG-171 verified**: LLM batch Ministral/Qwen3 parse asymmetry is intentional — Ministral wraps in `{"original": ..., "custom": ...}` (legacy LoRA structure), Qwen3 returns flat dict. Signal engine handles both via `ms.get("original") or ms`.
- **New tests**: 2 thread-safety tests for regime cache (concurrent access, cycle invalidation). All 212 indicator/signal tests pass.
- **Lint**: Reduced ruff violations from 75 to 59 (16 fixed). Remaining: 42 E402 (intentional), 9 SIM102, 5 SIM115, 3 SIM114.
- Theme: Thread Safety, I/O Safety, Lint Cleanup, Code Quality.

## 2026-03-31 (after-hours research session)
- **BUG-152: Trending-up regime gating incomplete (P1)**: Signal audit found 5 signals with 0-11% accuracy in trending-up (trend 0%, ema 11%, volume_flow 10%, macro_regime 11%, momentum_factors low). Plus claude_fundamental at 5.9%. All produce massive false SELL consensus during uptrends. Fix: gate all 6 at `_default` horizon in trending-up. 3h left ungated (short-term signals work in trends).
- **BUG-153: low_activity_timing correlation group mixes quality tiers (P1)**: Group contained {calendar 62.8%, econ_calendar 86.8%, forecast 36.1%, futures_flow 33.3%}. If forecast became leader, it suppressed excellent signals. Fix: removed forecast and futures_flow from the group.
- **BUG-154: claude_fundamental no regime gating (P2)**: 62.2% ranging (good), 5.9% trending-up (catastrophic), 30.4% trending-down (bad). Fix: gate in both trending-up and trending-down at `_default`.
- **BUG-155: bb no trending-down gating (P2)**: 21.7% accuracy in trending-down (46 samples) — false reversal signals. Fix: gate bb in trending-down at `_default`.
- **BUG-156: forecast actively harmful (P2)**: 36.1% 1d_recent, 38.3% 3h_recent. Verified: blended accuracy (70% recent + 30% all-time) = ~40% at both horizons, below 45% gate. No code change needed — existing accuracy gate catches it.
- **New tests**: 11 new tests across 3 classes (TestTrendingUpRegimeGating, TestTrendingDownRegimeGating, TestCorrelationGroupSplit), 1 existing test updated. 132 total pass.
- **Research deliverables**: daily_research_review.json, daily_research_signal_audit.json, daily_research_quant.json, daily_research_ticker_deep_dive.json, daily_research_macro.json, morning_briefing.json.
- Theme: Regime-Aware Signal Protection, Correlation Group Quality, Breakout Readiness.

## 2026-03-30 (autonomous improvement session)
- **BUG-150: Cross-horizon averaging bias (P1)**: `_compute_dynamic_horizon_weights()` used running `(old+new)/2` formula instead of true arithmetic mean. With 3+ cross horizons, the last-processed horizon got ~57% weight (should be 33%). Fix: accumulate sum+count, divide once. Also resolved 1 pre-existing test failure (`test_3h_boosts_news_event`).
- **REF-18/ARCH-28: Extract `_build_llm_context()` helper**: Ministral and Qwen3 signal blocks had ~80 lines of identical code (timeframe summary, EMA gap, context dict). Extracted shared `_build_llm_context(ticker, ind, timeframes, extra_info)` function. Qwen3 extends with `asset_type`. Net -47 lines.
- **REF-19: Remove dead funding code**: `main.py:341-342` checked `funding_action`/`funding_rate` in extra dict, but funding signal is disabled and those keys are never set. Removed.
- **REF-20: outcome_tracker module logger**: Replaced 5 function-local `import logging as _logging` patterns with a single module-level `logger = logging.getLogger("portfolio.outcome_tracker")`.
- **New tests**: 9 new tests — cross-horizon true mean (2), `_build_llm_context` helper (5), module logger (2). All pass. 1 pre-existing failure resolved.
- Theme: Signal Weight Accuracy, Code Deduplication, Dead Code Removal.

## 2026-03-29 (after-hours research session)
- **BUG-149: Regime gating not horizon-aware (P1)**: `REGIME_GATED_SIGNALS` applied uniformly across all prediction horizons. Trend had 61.6% accuracy on 3h in ranging markets — short-term trends exist within ranges. Fix: restructured to `{regime: {horizon: frozenset(signals)}}` with `_get_regime_gated()` helper. Trend is now only gated on 1d/default in ranging (40.7%), NOT on 3h (61.6%). Mean reversion is gated on 3h in trending (45.5%), NOT on 1d (65.4%).
- **BUG-150: Stale HORIZON_SIGNAL_WEIGHTS (P2)**: Static weights from March 27 were outdated. Updated with March 29 accuracy audit: 8 new 3h entries (smart_money 1.2, volatility_sig 1.2, momentum_factors 1.2, qwen3 1.2, trend 1.2, oscillators 0.7, bb 0.6, mean_reversion 0.7), 6 new 1d entries (ministral 1.3, macd 1.2, bb 1.2, volatility_sig 0.5, ema 0.6, trend 0.6, heikin_ashi 0.7). Tightened fear_greed 1d from 0.5 to 0.4 (25.9%), forecast from 0.6 to 0.5.
- **BUG-151: EMA missing 1d penalty (P2)**: EMA at 40.8% 1d_recent had no horizon penalty despite being near accuracy gate. Added 0.6x penalty.
- **FEAT-4: Dynamic horizon weight computation**: `_compute_dynamic_horizon_weights()` reads accuracy_cache.json and computes cross-horizon ratio multipliers automatically. Formula: `this_horizon_acc / cross_horizon_acc`, clamped [0.4, 1.5], with ±10% deadband. 1-hour cache TTL. Falls back to static dict when cache unavailable. Eliminates need for manual weight updates each session.
- **FEAT-5: macro_external correlation group**: New group `{fear_greed, sentiment, news_event}` — all depend on external data quality and fail together. Secondary signals get 0.3x correlation penalty.
- **New tests**: 10 new tests — horizon-aware regime gating (5), dynamic weight computation (6), macro_external correlation group (1). Updated 3 existing tests. All 115 signal_engine_core tests pass.
- **Research deliverables**: daily_research_review.json, daily_research_signal_audit.json, daily_research_quant.json, daily_research_ticker_deep_dive.json, daily_research_macro.json.
- Theme: Signal Accuracy Optimization, Adaptive Weighting, Horizon-Specific Intelligence.

## 2026-03-29 (autonomous improvement session)
- **BUG-143: Unanimity penalty uses pre-gated vote counts (P1)**: `buy`/`sell` counts in `signal_engine.py` were computed from raw votes BEFORE regime gating. The unanimity penalty (Stage 5) used stale counts — e.g., in ranging regime with 2 gated signals, penalty was ~33% too aggressive. Fix: apply `REGIME_GATED_SIGNALS` gating before counting (idempotent with `_weighted_consensus`).
- **BUG-144: Forecast regime discount is dead code (P1)**: `forecast.py` reads `context.get("regime")` to apply 0.5x confidence discount in trending markets (Chronos mean-reversion bias). But `generate_signal()` never included `regime` in `context_data`. The discount was never applied. Fix: add `"regime": regime` to context_data dict.
- **BUG-145: meta_learner SQLite connection leak (P2)**: `_load_data()` opened `sqlite3.connect()` but if `pd.read_sql_query()` threw, `conn.close()` was never called. Wrapped in try/finally.
- **BUG-146: meta_learner datetime import style (P3)**: Modernized from `datetime, timezone` to `datetime, UTC` (Python 3.11+, matches codebase REF-16).
- **BUG-147: meta_learner duplicated SIGNAL_NAMES (P2)**: Maintained its own 30-element copy instead of importing from `portfolio.tickers`. Any signal addition that updated `tickers.py` without updating `meta_learner.py` would silently train the model on wrong features. Fix: `from portfolio.tickers import SIGNAL_NAMES`.
- **BUG-148: meta_learner predict() disk I/O on every call (P2)**: `joblib.load()` deserialized 600KB model file on every prediction. Added module-level `_model_cache` dict with mtime-based staleness detection. Required prerequisite for future meta-learner integration (FEAT-3).
- **Test fixes**: Fixed 8 pre-existing failures in `test_confidence_penalties.py` caused by unanimity penalty (Stage 5) interacting with test fixtures using 83% buy/sell agreement. Updated `_base_extra` default to 67% agreement.
- **New tests**: 14 new tests — regime gating before vote counts (3), regime in context_data (2), meta_learner SQLite cleanup (2), SIGNAL_NAMES import (2), model cache (5). All pass.
- Theme: Signal Accuracy, Dead Code Activation, Resource Safety, Feature Coupling.

## 2026-03-27 (autonomous improvement session)
- **BUG-133: Accuracy cache cross-horizon staleness (P1)**: Cache shared a single timestamp for all horizons. Writing 3h data made stale 1d data appear fresh. Now uses per-horizon timestamps (`time_1d`, `time_3h`, etc.) with backwards-compatible fallback to legacy `time` key.
- **BUG-134: Regime accuracy hardcoded to 1d (P1)**: `signal_engine.py` always called `load_cached_regime_accuracy("1d")` regardless of prediction horizon. 3h/4h/12h predictions now use horizon-matched regime accuracy.
- **BUG-135: Utility boost hardcoded to 1d (P1)**: `signal_utility("1d")` was called regardless of horizon. Signals profitable at 1d but not at 3h were incorrectly boosted. Now uses `acc_horizon`.
- **BUG-136: Utility boost in-place mutation (P2)**: `accuracy_data[sig_name]["accuracy"] *= boost` mutated potentially cached data. Replaced with dict copy using spread operator.
- **BUG-137: SQLite resource leak (P2)**: `load_entries()` didn't close SignalDB on exception. Added try/finally.
- **BUG-139: load_json crashes on PermissionError (P2)**: `load_json()` now catches `OSError` (including `PermissionError`), returning default gracefully when files are locked by antivirus or other processes.
- **ARCH-23: Shared accuracy blend function**: Extracted `blend_accuracy_data()` into `accuracy_stats.py`. Both `signal_engine.py` and `backtester.py` use the shared function, eliminating 30 lines of duplicated blending logic.
- **ARCH-24: Pre-loaded entries parameter**: Added optional `entries=` parameter to 8 accuracy functions. `print_accuracy_report()` now loads entries once and passes to sub-functions, eliminating up to 21 redundant reads of the 68MB signal_log.jsonl.
- **New tests**: 20 new tests — per-horizon cache timestamps (6), blend_accuracy_data (7), load_json OSError (4), entries= parameter (3). All pass.
- Theme: Accuracy Correctness, Data Integrity, Code Deduplication.

## 2026-03-26 (autonomous improvement session)
- **BUG-128: Avanza offset file atomicity**: `avanza_orders.py` Telegram offset file now uses `atomic_write_json()` instead of `write_text()`, preventing corruption on crash. Read path handles both legacy plain-text and new JSON format for backwards compatibility.
- **BUG-129: Playwright thread safety**: `avanza_session.py` global Playwright state (`_pw_instance`, `_pw_browser`, `_pw_context`) now protected by `threading.Lock` to prevent concurrent access corruption.
- **BUG-130: Dashboard TTL cache**: Added thread-safe in-memory TTL cache to `dashboard/app.py` (5s default, 60s for config). Eliminates redundant disk I/O on concurrent API requests.
- **BUG-131: Safe Telegram truncation**: `message_store.py` now truncates at the last newline boundary before the 4096-char limit instead of at an arbitrary character position. Prevents splitting Markdown tags mid-formatting.
- **SYSTEM_OVERVIEW.md**: Updated test count (5,994 across 159 files), added fixture documentation, tracked BUG-128 through BUG-132, ARCH-21/22.
- **New tests**: 9 new tests — avanza offset format compatibility (3), message truncation safety (3), dashboard cache behavior (3). All pass.
- Theme: Crash Safety, Thread Safety, Performance Caching, Markdown Integrity.

## 2026-03-25 (autonomous improvement session)
- **BUG-122: Health module 68MB memory spike (x2)**: `check_outcome_staleness()` and `check_dead_signals()` in `health.py` both used `f.readlines()` on the 68MB signal_log.jsonl to check 20-50 entries. Replaced with `load_jsonl_tail()` — reads ~512KB instead of 68MB. Eliminates ~150MB memory spike per health cycle.
- **BUG-123: Untracked files break worktrees**: `portfolio/metals_ladder.py`, `portfolio/process_lock.py`, `portfolio/subprocess_utils.py`, `portfolio/notification_text.py` were imported by tracked modules but never committed. Any worktree or fresh clone hit `ModuleNotFoundError`. Now tracked in git along with 5 test files.
- **BUG-124: fin_snipe_manager raw config read**: `_notify_critical()` used raw `open()/json.load()` for config.json. Replaced with `load_json()` for crash-safe fallback.
- **BUG-125: onchain_data non-atomic cache write**: `_save_onchain_cache()` used `write_text()`. Replaced with `atomic_write_json()` to prevent corrupt cache on crash.
- **BUG-126: main.py silent exception handlers**: Two `except Exception: pass` in safeguard Telegram alerts. Added `logger.debug()` for visibility.
- **BUG-127: crypto_scheduler silent exception**: Fundamentals cache read failure silently swallowed. Added `logger.debug()`.
- **REF-9: Raw JSONL append consolidation**: Replaced 5 remaining raw `open("a")/f.write(json.dumps())` patterns with `atomic_append_jsonl()` in `crypto_macro_data.py`, `analyze.py`, `bigbet.py`, `iskbets.py`. Provides fsync durability.
- **REF-10: fin_evolve.py import cleanup**: Removed 5 underscore-prefixed import aliases (`_load_json`, `_atomic_write_json`, etc.) — legacy from removed fallback wrappers. Updated 13 call sites and fixed 2 test assertions.
- Also synced 5 modified portfolio files from main that were never committed (ministral/qwen3 signal/trader, signal_engine).
- Theme: Memory Optimization, I/O Safety, Git Hygiene, Observability.

## 2026-03-23 (autonomous improvement session)
- **BUG-111: Accuracy tracking corruption**: `outcome_tracker._derive_signal_vote("rsi")` used hardcoded 30/70 thresholds while `signal_engine` uses adaptive `rsi_p20`/`rsi_p80` percentiles. Accuracy backfill recorded different RSI votes than actually cast, corrupting signal accuracy tracking. Fixed to use adaptive thresholds with [15,85] clamp, matching signal_engine exactly.
- **BUG-112: Backfill memory optimization**: `backfill_outcomes()` loaded entire 68MB signal_log.jsonl (~150K entries, ~75MB parsed JSON) into memory to process only 2,000 entries. Refactored to streaming approach: count lines (binary scan), skip head without parsing, parse only tail, stream head bytes verbatim on rewrite. Memory: 75MB → 2MB.
- **BUG-113: majority_vote HOLD confidence**: When HOLD won (neither BUY nor SELL majority), `majority_vote()` returned misleading non-zero confidence with `count_hold=True`. HOLD is the absence of a signal — confidence is now always 0.0.
- **BUG-114: Forecast extraction observability**: `_extract_json_from_stdout()` had 3 fallback strategies for parsing JSON from contaminated subprocess stdout but never logged which succeeded. Added debug-level logging for each fallback path.
- **COVERAGE-2: outcome_tracker tests**: Added 85 new tests (81 in `test_outcome_tracker_core.py` + 4 streaming tests in `test_outcome_tracker_backfill.py`). Coverage for `_derive_signal_vote` (all 11 signal branches) and `log_signal_snapshot`.
- Theme: Accuracy Tracking Correctness, Memory Optimization, Signal Robustness. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-22 (autonomous improvement session)
- **BUG-107: Digest zero-division**: `digest.py` and `daily_digest.py` P&L calculations crashed when `initial_value_sek` was 0 or missing. Added `or INITIAL_CASH_SEK` fallback (same fix as BUG-103, missed in these two modules).
- **BUG-108: Alpha Vantage budget thread safety**: `_daily_budget_used` counter was read/incremented without lock protection. Wrapped in existing `_cache_lock`.
- **BUG-109: Signal log performance**: `digest.py` read entire 68MB `signal_log.jsonl` to get last 500 entries. Added `load_jsonl_tail()` to `file_utils.py` — seeks to last 512KB instead of reading entire file.
- **BUG-110: Stale import path**: `digest.py` imported `load_jsonl` from `portfolio.stats` re-export instead of canonical `portfolio.file_utils`.
- **COVERAGE-1: reporting.py tests**: Added 50 tests for `reporting.py` (1,109 lines, previously ZERO coverage). Covers `write_agent_summary`, `_write_compact_summary`, `_cross_asset_signals`, `_macro_headline`, `_portfolio_snapshot`, `write_tiered_summary`, `_get_held_tickers`.
- Theme: Digest Safety, Budget Tracking, Reporting Tests. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-19 (autonomous improvement session)
- **REF-16: Python 3.11 modernization**: ruff auto-fix applied 1,910 fixes across 268 files. Key changes: `datetime.timezone.utc` → `datetime.UTC` (199), `Optional[X]` → `X | None` (149), unsorted imports (75), `Dict`/`List`/`Tuple` → builtins (44), redundant open modes (10), deprecated typing imports (8), duplicate set value (1). Zero behavioral change.
- **REF-17: Manual ruff fixes**: 28 fixes across 20 files. `raise ImportError(...)` → `raise ... from None` (B904), 17 unused loop variables prefixed with `_` (B007), 2 needless bool returns simplified (SIM103).
- **BUG-81**: `avanza_client.py` `raise ImportError` now chains with `from None` for clean tracebacks.
- **BUG-83: Silent exception logging**: Added `logger.debug()` to 5 remaining `except Exception: pass` handlers in gpu_gate.py, telegram_notifications.py, signal_engine.py, reporting.py (x2).
- **BUG-84: ADX caching**: `_compute_adx()` now cached per DataFrame identity (`id(df)`), eliminating ~140 redundant computations per loop cycle. Cache auto-clears on overflow (200 entries max).
- Theme: Python Modernization & Final Bug Sweep. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-18 (autonomous improvement session)
- **REF-13: ruff lint cleanup**: Auto-fixed 112 violations (94 unused imports, 15 empty f-strings, 2 reimports) across 59 files. Manually fixed 3 Python 3.11 f-string backslash compatibility issues in `autonomous.py` and 1 unused import in `risk_management.py`.
- **REF-14 + BUG-75/76/77: Dead variable removal**: Removed 15 unused variable assignments across 13 modules: `signal_engine.py`, `trigger.py`, `telegram_poller.py`, `smart_money.py`, `autonomous.py`, `alpha_vantage.py`, `avanza_session.py`, `bigbet.py`, `daily_digest.py`, `equity_curve.py`, `http_retry.py`, `portfolio_validator.py`.
- **BUG-71/73: Config IO hardening**: Replaced raw `json.load(open(...))` in golddigger and elongir config loading with `load_json()` from `file_utils`. Corrupt config now raises `ValueError` instead of cryptic `JSONDecodeError`.
- **BUG-72: Golddigger Telegram routing**: Replaced direct `requests.post()` Telegram call with `send_or_store()` from `message_store`. Gains JSONL message logging, Markdown escaping, and 4096 char handling.
- **BUG-74: Golddigger data cache IO**: Replaced local `_load_json_safe()` body with `load_json()` from `file_utils`.
- **BUG-79: Silent exception logging**: Added `logger.debug()` to `avanza_tracker.py`'s silent import exception handler.
- Theme: Lint Cleanup & Subsystem IO Hardening. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-17
- **Model/runtime hardening session**: landed the `feat/model-upgrades` work on `main`, moving both local trading LLMs onto the native CUDA llama.cpp path and tightening the Windows loop launcher.
- **Qwen3 upgrade**: `qwen3_trader.py` now runs through `llama-completion` on CUDA 13.1, gained batch-mode support for multi-ticker runs, and exposes explicit native asset validation via `load_model()` so missing GGUF/binary paths fail clearly.
- **Ministral-3 upgrade**: `ministral_trader.py` now prefers native `llama-completion` inference for Ministral-3-8B and falls back cleanly to the legacy Ministral-8B path when native arch/load errors occur.
- **GPU gate**: added exclusive GPU/VRAM coordination with VRAM usage logging across all four GPU-backed models (Ministral, Qwen3, Chronos, Kronos), then tightened the wait timeout from 60s to 15s after measuring real lock hold times.
- **Loop crash fix**: `scripts/win/pf-loop.bat` now sets `PYTHONPATH=Q:\finance-analyzer` before launching the loop to prevent `ModuleNotFoundError` crash loops in detached Windows contexts, and `scripts/restart_loop.py` mirrors the same safeguard for manual restarts.

## 2026-03-14 (autonomous improvement session)
- **IO safety sweep complete**: Replaced all 37+ raw `json.loads(path.read_text())` calls across 23 portfolio modules with `load_json()` from `file_utils` — eliminates TOCTOU race conditions and crash-on-corrupt-file.
- **REF-8**: Added `atomic_write_jsonl()` helper to `file_utils.py` for safe full-file JSONL rewrites.
- **BUG-48**: Replaced 3 non-atomic writes: `prophecy.py` (→ `atomic_write_json`), `signal_history.py` and `forecast_accuracy.py` (→ `atomic_write_jsonl`).
- **BUG-49**: Replaced manual JSONL parse loops with `load_jsonl()` in `analyze.py`, `signal_history.py`, `focus_analysis.py`, `equity_curve.py`.
- **TEST-11**: Added `tests/test_io_safety_sweep.py` (34 tests) — static analysis scan verifying no raw file reads remain, plus functional tests for `atomic_write_jsonl` and `load_json` edge cases.
- Modules touched: accuracy_stats, alpha_vantage, analyze, autonomous, avanza_client, avanza_orders, avanza_session, avanza_tracker, bigbet, daily_digest, equity_curve, focus_analysis, forecast_accuracy, forecast_signal, iskbets, journal, local_llm_report, main, onchain_data, perception_gate, prophecy, signal_history, telegram_notifications, signals/claude_fundamental.

## 2026-03-11
- **NewsAPI configured**: Added API key to `config.json → newsapi_key` (free tier, 100 req/day). Enhances stock sentiment headlines in `sentiment.py` and news_event signal #26 alongside Yahoo Finance fallback.
- **Config validator updated**: Added `newsapi_key`, `alpha_vantage.api_key`, `golddigger.fred_api_key`, `bgeometrics.api_token` to `OPTIONAL_KEYS` in `config_validator.py` — warns at startup if missing.
- **API inventory documented**: Full external API integration table added to `docs/SYSTEM_OVERVIEW.md` section 6 (12 services, all configured). Avanza manual auth status documented.
- **TODO.md updated**: Alpha Vantage and NewsAPI marked as done. Avanza credential automation added as pending item.

## 2026-03-05 (autonomous improvement session)
- Hardened dashboard JSONL consumers:
  - `/api/telegrams` now ignores non-object JSONL entries instead of propagating malformed shapes.
  - `/api/decisions` now ignores non-object JSONL entries instead of assuming dict records.
- Added dashboard API test coverage for:
  - malformed JSONL resilience in `/api/telegrams` and `/api/decisions`,
  - `/api/metals-accuracy` success/missing/auth behavior.
- Improved `portfolio.accuracy_stats.load_entries()` JSONL fallback to skip malformed lines instead of failing the entire accuracy read.
- Upgraded static dashboard export tool:
  - supports token-protected dashboards (reads `dashboard_token` from `config.json`),
  - exports frontend-required routes `/api/metals-accuracy` and `/api/lora-status`.

## 2026-03-05
- **BUG-61 through BUG-67**: Replaced 15 silent `except Exception: pass` handlers with logged warnings/debug messages across 6 modules: `autonomous.py`, `fx_rates.py`, `outcome_tracker.py`, `journal.py`, `forecast.py`, `main.py`.
- **BUG-69**: Fixed `_run_post_cycle()` in `main.py` to use module-level `DATA_DIR` constant instead of re-deriving path.
- **BUG-70**: Removed redundant `import time as _time` from `run()` in `main.py` — `time` already imported at module level.
- Added `tests/test_silent_exceptions.py` with 9 tests verifying each previously-silent handler now logs.

## 2026-03-04
- Aligned `tests/test_shared_state.py` with current LRU fallback cache eviction semantics in `portfolio/shared_state.py`.
- Verified targeted batch tests for shared-state eviction, forecast circuit reset, JSONL pruning, and signal registry isolation.
- Confirmed full-suite failures are mostly pre-existing integration/runtime-environment issues (Freqtrade strategy path, metals autonomous expectations, trigger/report timing).
- Removed transient working document `docs/SESSION_PROGRESS.md` per auto-improve workflow.
