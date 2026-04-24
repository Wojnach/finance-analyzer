# Full Dual Adversarial Review #7 — 2026-04-24

## Methodology

**Dual review protocol:** 8 parallel code-reviewer agents (one per subsystem)
conducted independent adversarial reviews, then the lead reviewer cross-critiqued
every finding against direct code reads. Each finding is classified:

- **CONFIRMED** — Verified by both agent and lead reviewer with code evidence
- **FALSE POSITIVE** — Agent claim refuted by direct code inspection
- **DOWNGRADED** — Real observation but severity overstated
- **NOVEL** — Found only by lead reviewer, not by agent

**Subsystems reviewed:**
1. signals-core (signal_engine, accuracy_stats, signal_db, outcome_tracker, etc.)
2. orchestration (main.py, agent_invocation, trigger, market_timing, loop_contract, bigbet)
3. portfolio-risk (portfolio_mgr, risk_management, trade_guards, kelly_sizing, monte_carlo)
4. metals-core (fin_snipe_manager, fin_fish, exit_optimizer, iskbets, microstructure)
5. avanza-api (avanza_session, avanza_orders, avanza_control, portfolio/avanza/ package)
6. signals-modules (38 enhanced signal plugins in portfolio/signals/)
7. data-external (data_collector, fear_greed, sentiment, futures_data, onchain_data, etc.)
8. infrastructure (file_utils, shared_state, health, telegram, dashboard, log_rotation, bots)

**Total findings:** 53 raw agent findings -> 4 false positives, 4 downgrades,
2 novel lead-reviewer findings = **47 confirmed findings** (4 P0, 12 P1, 31 P2/P3).

---

## P0: CRITICAL — Fix Immediately (Financial Risk)

### P0-1: Regime gating drops `_default` set when horizon-specific key exists
- **File:** `portfolio/signal_engine.py:721-728`
- **Subsystem:** signals-core
- **Agent finding:** CONFIRMED
- **Lead cross-check:** CONFIRMED — independently verified identical data structure,
  compared with `_get_horizon_disabled_signals` (line 453) which correctly unions
  `default_set | horizon_set`. Same structure, different semantics = confirmed bug.
- **Issue:** `_get_regime_gated()` returns ONLY the horizon-specific frozenset when
  `horizon in regime_dict`, dropping the 13-signal `_default` set entirely. In a
  ranging regime at 3h, only 4 signals are gated (mean_reversion, bb,
  claude_fundamental, sentiment) while 13 signals with documented sub-45% accuracy
  in ranging (trend, momentum_factors, ema, heikin_ashi, structure, econ_calendar,
  forecast, news_event, volatility_sig, candlestick, smart_money, oscillators,
  funding) vote freely.
- **Impact:** Consensus contaminated by 13 unreliable signals in ranging regime at
  3h/4h horizons — the primary intraday decision horizons. Can produce false
  BUY/SELL on metals and crypto.
- **Fix:**
  ```python
  def _get_regime_gated(regime, horizon=None):
      regime_dict = REGIME_GATED_SIGNALS.get(regime, {})
      if not regime_dict:
          return frozenset()
      default_set = regime_dict.get("_default", frozenset())
      if horizon and horizon in regime_dict:
          return default_set | regime_dict[horizon]  # union, not replace
      return default_set
  ```

### P0-2: Stop-loss at 5% cert distance = 1% underlying at 5x leverage
- **File:** `portfolio/fin_snipe_manager.py:507-541`
- **Subsystem:** metals-core
- **Agent finding:** CONFIRMED
- **Issue:** `_compute_stop_plan()` sets stop trigger at `position_avg * (1 - 0.05)`
  where `position_avg` is the **certificate price in SEK**. A 5% cert drop on a
  5x leveraged warrant corresponds to only a 1% underlying move — well within
  normal intraday silver noise (1-2% wicks are routine).
- **Impact:** Stop-losses will repeatedly trigger on normal intraday noise,
  producing a string of -5% cert losses that compound. The project rule requires
  stops to be >=3% of current bid, and the metals rule requires -15%+ cert stops
  for 5x warrants.
- **Fix:** Use `HARD_STOP_CERT_PCT = 0.15` (15% cert = 3% underlying at 5x), or
  compute stop distance on the underlying price and verify it's >=3%.

### P0-3: `delete_stop_loss` treats HTTP 404 as failure — aborts sell flow
- **File:** `data/metals_avanza_helpers.py:482-484`
- **Subsystem:** avanza-api
- **Agent finding:** CONFIRMED
- **Lead cross-check:** CONFIRMED — compared with `avanza_session.cancel_stop_loss`
  (line 906) which explicitly handles `or http_status == 404` as success.
- **Issue:** `success = 200 <= http_status < 300` excludes 404. The stop-loss
  DELETE endpoint returns 404 when the stop has already triggered, expired, or
  been cancelled by the broker. The cancel-before-sell sequence interprets 404 as
  failure and aborts the sell, leaving the position open with no stop protection.
- **Impact:** If a stop-loss triggers during the brief window before a manual sell,
  the cancel returns 404, the sell is aborted, and the position remains open
  naked. This is worse than the original problem the stop was meant to prevent.
- **Fix:** `success = (200 <= http_status < 300) or http_status == 404`

### P0-4: `check_drawdown` blind with open positions when prices stale (CR-1 recurrence)
- **File:** `portfolio/risk_management.py:126-139`
- **Subsystem:** portfolio-risk
- **Agent finding:** CONFIRMED
- **Lead cross-check:** This is a recurrence of CR-1 from the April 17 review.
  The WARNING is logged but the circuit breaker returns `breached=False` with
  `current_value = cash_sek` when `agent_summary` is empty/stale. A portfolio
  with 60% in a crashing leveraged warrant appears to have 0% drawdown.
- **Impact:** Circuit breaker disabled during the exact scenario it exists for —
  correlated API failure + market crash.
- **Fix:** When `holding_count > 0` and agent_summary is empty, return
  `breached=True` (fail-safe) or at minimum preserve the last known drawdown.

---

## P1: HIGH — Fix This Sprint

### P1-1: `bigbet.py` bypasses `claude_gate` serialization lock
- **File:** `portfolio/bigbet.py:175-181`
- **Subsystem:** orchestration
- **Agent finding:** CONFIRMED
- **Issue:** `invoke_layer2()` calls `subprocess.run(["claude", ...])` directly
  instead of routing through `claude_gate.invoke_claude()`. This bypasses the
  `_invoke_lock` mutex, allowing concurrent Claude CLI processes when bigbet
  fires while the main agent is running. Additionally, auth failure scanning
  concatenates stdout+stderr (line 191) instead of scanning separately as
  `claude_gate` requires.
- **Fix:** Replace with `invoke_claude()` from `claude_gate`, or at minimum
  acquire `_invoke_lock` before the subprocess call.

### P1-2: `active_voters` stale after persistence filter — trades below quorum
- **File:** `portfolio/signal_engine.py:2906, 3101, 3124`
- **Subsystem:** signals-core
- **Agent finding:** CONFIRMED
- **Lead cross-check:** CONFIRMED — independently verified. `active_voters` at
  line 2906 is computed from pre-persistence `votes`. Persistence filter at 3101
  can suppress multiple BUY/SELL signals to HOLD. The min_voters guard at 3124
  still uses the stale count. A 5-voter pre-persistence consensus that drops to
  2 actual voters post-persistence passes the min_voters=3 gate.
- **Impact:** Trades emitted from fewer voters than the quorum requires.
- **Fix:** Recompute `active_voters` from `consensus_votes` after
  `_apply_persistence_filter` returns, before line 3110.

### P1-3: `avanza/client.py` account_id not validated against whitelist
- **File:** `portfolio/avanza/client.py:63-65`
- **Subsystem:** avanza-api
- **Agent finding:** CONFIRMED
- **Issue:** `AvanzaClient.get_instance()` reads `account_id` from config without
  checking `ALLOWED_ACCOUNT_IDS`. If misconfigured, all orders route to the
  wrong account (potentially pension account 2674244).
- **Fix:** After reading `account_id` from config, assert membership in
  `ALLOWED_ACCOUNT_IDS` and raise on mismatch.

### P1-4: `fear_greed.py` unguarded IndexError/KeyError on malformed API response
- **File:** `portfolio/fear_greed.py:98-107`
- **Subsystem:** data-external
- **Agent finding:** CONFIRMED
- **Issue:** `body["data"][0]` with no guard for empty list or missing key.
  The alternative.me API can return `{"data": []}` during maintenance.
- **Fix:** `raw = body.get("data"); if not raw: return None; data = raw[0]`

### P1-5: `trade_guards.check_overtrading_guards` not atomic
- **File:** `portfolio/trade_guards.py:97-275`
- **Subsystem:** portfolio-risk
- **Agent finding:** CONFIRMED
- **Issue:** `_load_state()` at line 97 and `_save_state()` at line 275 have no
  lock between them. With 8 parallel ThreadPoolExecutor workers, two concurrent
  BUYs can both pass the position-rate-limit guard and both record, exceeding
  the intended limit.
- **Fix:** Wrap check+record in a file-level lock.

### P1-6: `multi_agent_layer2.py` specialist timeout — no tree-kill, unhandled TimeoutExpired
- **File:** `portfolio/multi_agent_layer2.py:205-209`
- **Subsystem:** orchestration
- **Agent finding:** CONFIRMED
- **Issue:** `proc.kill()` only kills direct child on Windows, leaving Node.js
  grandchildren as zombies. If `proc.wait(timeout=5)` also times out, the
  exception is unhandled and crashes the multi-agent path.
- **Fix:** Use `_kill_process_tree()` from `claude_gate` and wrap the second
  `wait()` in try/except.

### P1-7: `loop_contract.py` read-modify-write race on CONTRACT_STATE_FILE
- **File:** `portfolio/loop_contract.py:872-880`
- **Subsystem:** orchestration
- **Agent finding:** CONFIRMED
- **Issue:** `ViolationTracker._save()` and `check_layer2_journal_activity()` both
  read-modify-write the same file independently. The second writer clobbers keys
  from the first, losing the `layer2_last_violation_trigger_ts` dedup key.
- **Impact:** Violation dedup fails -> Telegram spam with repeated alerts.

### P1-8: `avanza_session.py` expired session — naive/aware datetime comparison
- **File:** `portfolio/avanza_session.py:82-90`
- **Subsystem:** avanza-api
- **Agent finding:** CONFIRMED
- **Issue:** `datetime.fromisoformat(expires_at)` produces a naive datetime when
  no timezone in the stored string. Comparing against `datetime.now(UTC)` raises
  `TypeError`. Only `ValueError` is caught, so `TypeError` propagates, and the
  session is used despite being expired.
- **Fix:** Force UTC if `exp.tzinfo is None`.

### P1-9: `agent_invocation.py` cleanup not in `finally` — double-completion risk
- **File:** `portfolio/agent_invocation.py:948-959`
- **Subsystem:** orchestration
- **Agent finding:** CONFIRMED
- **Issue:** If an exception occurs mid-completion handling, `_agent_proc` is left
  non-None. Next cycle, `poll()` returns the same exit code again, causing
  double-processing — duplicate trades in the overtrading guard.
- **Fix:** Move cleanup block into `finally`.

### P1-10: `outcome_tracker.py` non-atomic JSONL rewrite, SQLite divergence risk
- **File:** `portfolio/outcome_tracker.py:430-446`
- **Subsystem:** signals-core
- **Agent finding:** CONFIRMED
- **Issue:** `os.replace` on Windows is not atomic when the target is locked by
  another process (e.g., the main loop's `atomic_append_jsonl`). If `os.replace`
  fails after SQLite was already updated, the two backends diverge permanently.
- **Fix:** Use `file_utils.atomic_write_jsonl()` for the rewrite, or deprecate
  JSONL rewriting and use SQLite as the single source of truth.

### P1-11: `telegram_poller` uses raw `open()` — Rule 4 violation
- **File:** `portfolio/telegram_poller.py:199-212`
- **Subsystem:** infrastructure
- **Agent finding:** CONFIRMED
- **Issue:** Reads `config.json` with `json.load(open(...))` instead of
  `file_utils.load_json()`. `PermissionError` propagates uncaught.
- **Fix:** Replace with `file_utils.load_json(config_path)`.

### P1-12: `realized_skewness` z-score normalization — sub-signal 1 always HOLD
- **File:** `portfolio/signals/realized_skewness.py:55-63`
- **Subsystem:** signals-modules
- **Agent finding:** CONFIRMED
- **Issue:** `lookback = min(SKEW_LOOKBACK, len(returns))`, then
  `_compute_rolling_skewness(returns, lookback)` rolls a window of `lookback`
  over `lookback` bars, producing exactly 1 non-NaN value. `std()` of a single
  point is NaN or 0, triggering the `< 1e-8` guard => perpetual HOLD.
  Sub-signal 1 is silently dead on all intraday timeframes.
- **Fix:** Use `NORM_WINDOW` (60) as the rolling skewness window, not `lookback`.

---

## P2: MEDIUM — Fix When Convenient

### P2-1: Utility boost can cross accuracy gate threshold
- **File:** `signal_engine.py:3058-3073`
- **Subsystem:** signals-core
- **Issue:** A signal at 0.46 accuracy with positive avg_return can be boosted
  above the 0.47 gate threshold. Undermines gate intent.
- **Fix:** Cap boost so it cannot cross the gate threshold.

### P2-2: `ic_computation.py` misleading field names
- **File:** `portfolio/ic_computation.py:119-122`
- **Subsystem:** signals-core
- **Issue:** `ic_buy`/`ic_sell` are average returns (range [-10,+10]), not rank
  correlations (range [-1,+1]). No live code reads these for gating decisions.
- **Fix:** Rename to `avg_return_buy`/`avg_return_sell`.

### P2-3: `market_timing.py` no Swedish holiday check for agent window
- **File:** `portfolio/market_timing.py:255`
- **Subsystem:** orchestration
- **Issue:** Layer 2 fires on Swedish-only holidays (Midsummer, Epiphany, etc.)
  when Avanza is closed. Orders would fail at API level but tokens are wasted.
- **Fix:** Add `or is_swedish_market_holiday(now)` to the guard.

### P2-4: `trigger.py` BUY/SELL flip bypasses ranging dampening
- **File:** `portfolio/trigger.py:239-241`
- **Subsystem:** orchestration
- **Issue:** Direction flips in ranging regime bypass the confidence check that
  was added for HOLD->BUY/SELL transitions.

### P2-5: `crypto_scheduler.py` local-timezone JSONL timestamp
- **File:** `portfolio/crypto_scheduler.py:310`
- **Subsystem:** orchestration
- **Issue:** Uses `datetime.now().astimezone()` instead of `datetime.now(UTC)`.
  Breaks UTC consistency for cross-file timestamp comparisons.

### P2-6: `main.py` stale config for post-cycle tasks
- **File:** `portfolio/main.py:1081-1128`
- **Subsystem:** orchestration
- **Issue:** `_run_post_cycle(config, ...)` uses config loaded once at startup.
  Post-cycle tasks don't pick up config changes without restart.

### P2-7: ATR stop formula only for long positions
- **File:** `portfolio/risk_management.py:244`
- **Subsystem:** portfolio-risk
- **Issue:** `stop_price = entry * (1 - 2*atr_pct/100)` — wrong for shorts.
  Currently safe because `SHORT_ENABLED=False`.

### P2-8: `monte_carlo_risk.py` default fx_rate=10.0 stale
- **File:** `portfolio/monte_carlo_risk.py:502-505`
- **Subsystem:** portfolio-risk
- **Issue:** Default USD/SEK rate of 10.0 understates SEK VaR by ~7% when actual
  rate is ~10.7. Log warning when fallback used.

### P2-9: `compute_stop_levels` proceeds with atr_pct=0
- **File:** `portfolio/risk_management.py:244`
- **Subsystem:** portfolio-risk
- **Issue:** When `atr_pct=0` (signal absent), `stop_price = entry_price`,
  creating a perpetually-triggered stop. Missing the `atr_pct <= 0` guard.

### P2-10: `equity_curve._pair_round_trips` doesn't subtract fees
- **File:** `portfolio/equity_curve.py:380-399`
- **Subsystem:** portfolio-risk
- **Issue:** `pnl_sek = (sell_price - buy_price) * matched` ignores `fee_sek`.
  Reported profit_factor is inflated.

### P2-11: `warrant_portfolio.py` P&L ignores financing level for MINIs
- **File:** `portfolio/warrant_portfolio.py:52-113`
- **Subsystem:** metals-core
- **Issue:** Linear leverage approximation is wrong for MINI warrants near
  financing level. Dashboard and journal show incorrect unrealized P&L.

### P2-12: `iskbets.py` defaults exit price to `highest_price`
- **File:** `portfolio/iskbets.py:800-851`
- **Subsystem:** metals-core
- **Issue:** `_handle_sold` without an exit price defaults to session high,
  inflating recorded P&L on stop exits.

### P2-13: `avanza_orders.py` CONFIRM race across multiple pending orders
- **File:** `portfolio/avanza_orders.py:120-135`
- **Subsystem:** avanza-api
- **Issue:** "CONFIRM" Telegram reply confirms most-recent-timestamped pending
  order, not necessarily the one the user saw. With concurrent metals_loop and
  golddigger, could confirm the wrong instrument.

### P2-14: `avanza_session.py` valid_until defaults to local date
- **File:** `portfolio/avanza_session.py:609`
- **Subsystem:** avanza-api
- **Issue:** `date.today()` uses local time. Between 00:00-01:00 UTC, this is
  yesterday's CET date and Avanza rejects the order.

### P2-15: `avanza/account.py` returns zeroed AccountCash on account-not-found
- **File:** `portfolio/avanza/account.py:93-94`
- **Subsystem:** avanza-api
- **Issue:** Returns `AccountCash(buying_power=0.0)` on miss; callers can't
  distinguish from zero balance. Should return `None`.

### P2-16: `avanza_session.py` potential double browser on recovery
- **File:** `portfolio/avanza_session.py:219-226`
- **Subsystem:** avanza-api
- **Issue:** Concurrent 401 recovery can launch two Chromium processes, leaking one.

### P2-17: `earnings_calendar.py` AV calls bypass daily budget counter
- **File:** `portfolio/earnings_calendar.py:48-53`
- **Subsystem:** data-external
- **Issue:** EARNINGS endpoint calls not reflected in Alpha Vantage budget tracker.
  With only MSTR active, currently 1 hidden call/day. Risk grows if stocks re-added.

### P2-18: `data_collector.py` error dict masquerades as valid data
- **File:** `portfolio/data_collector.py:311-313`
- **Subsystem:** data-external
- **Issue:** Exception returns `{"error": str(e)}` — passes `is not None` checks
  but has no `indicators` key. Downstream signals raise KeyError in wrong location.

### P2-19: `onchain_data.py` skips `_coerce_epoch` in cache load
- **File:** `portfolio/onchain_data.py:95-107`
- **Subsystem:** data-external
- **Issue:** `_load_onchain_cache` does raw arithmetic on `ts` without calling
  `_coerce_epoch`. If `ts` is ISO string from older cache, raises TypeError.

### P2-20: `forecast_signal.py` no row-count validation before `iloc[h-1]`
- **File:** `portfolio/forecast_signal.py:217-219`
- **Subsystem:** data-external
- **Issue:** If Chronos-2 returns fewer rows than `max(horizons)`, `iloc[h-1]`
  raises IndexError. Caught by outer try/except but silently produces no signal.

### P2-21: `shared_state._cached` leaks `_loading_timestamps` on KeyboardInterrupt
- **File:** `portfolio/shared_state.py:104-108`
- **Subsystem:** infrastructure
- **Issue:** Missing `_loading_timestamps.pop(key, None)` in KeyboardInterrupt
  handler. Key persists up to 120s, suppressing fetches for that window.

### P2-22: `log_rotation.rotate_text` truncates active log non-atomically
- **File:** `portfolio/log_rotation.py:319-325`
- **Subsystem:** infrastructure
- **Issue:** Truncation via `open(filepath, "w")` races with Python's
  RotatingFileHandler. Crash mid-truncate leaves zero-byte log.

### P2-23: `shared_state._cached` uses stale timestamp for cache entries
- **File:** `portfolio/shared_state.py:48-100`
- **Subsystem:** infrastructure
- **Issue:** `now = time.time()` captured before lock acquisition. If `func()` takes
  10-30s (LLM inference), cached timestamp is 30s in the past, causing premature
  re-fetch.

### P2-24: `health.py` reports healthy while circuit breakers unavailable
- **File:** `portfolio/health.py:254-263`
- **Subsystem:** infrastructure
- **Issue:** Missing `circuit_breakers` key in summary when import fails.
  Monitoring sees `None` = can't distinguish from "all healthy".

### P2-25: `gpu_gate.py` stale lock threshold too long
- **File:** `portfolio/gpu_gate.py:26`
- **Subsystem:** infrastructure
- **Issue:** `_STALE_SECONDS = 300` is 2x the LLM inference timeout. Crashed
  GPU jobs block all GPU signals for up to 3 minutes after the timeout.

### P2-26: `message_throttle.py` race between check and send
- **File:** `portfolio/message_throttle.py:57-65`
- **Subsystem:** infrastructure
- **Issue:** No lock between `should_send_analysis` check and `_send_now` update.
  Two threads can both pass and both send within the same cooldown window.

### P2-27: `mean_reversion` correlated sub-signal double-counts
- **File:** `portfolio/signals/mean_reversion.py:571`
- **Subsystem:** signals-modules
- **Issue:** `ibs_rsi2_combined` sub-signal votes alongside its component
  sub-signals (IBS and RSI(2)), inflating confidence when both agree.

### P2-28: `complexity_gap_regime` self-referential peer universe
- **File:** `portfolio/signals/complexity_gap_regime.py:57`
- **Subsystem:** signals-modules
- **Issue:** Peer universe includes `GC=F`/`SI=F` — the instruments being traded.
  Signal measures gold's own correlation with itself.

### P2-29: `calendar_seasonal` double-BUY in January
- **File:** `portfolio/signals/calendar_seasonal.py:157,183`
- **Subsystem:** signals-modules
- **Issue:** `_sell_in_may` and `_january_effect` both vote BUY in January,
  counting one seasonal thesis twice.

### P2-30: Hull MA level-based, not crossover — persistent trending bias
- **File:** `portfolio/signals/heikin_ashi.py:288-292`
- **Subsystem:** signals-modules
- **Issue:** `HMA(9) > HMA(21)` is a level comparison, not crossover. Votes BUY
  on every bar during a sustained uptrend. Inconsistent with `trend.py`'s
  `_golden_cross` which uses actual crossover detection.

### P2-31: `social_sentiment.py` errors via print() not logger
- **File:** `portfolio/social_sentiment.py:110,122`
- **Subsystem:** data-external
- **Issue:** Reddit errors go to stdout, not the logging system.

---

## FALSE POSITIVES

### FP-1: Alligator forward shift "look-ahead bias"
- **File:** `portfolio/signals/heikin_ashi.py:318-320`
- **Agent claim:** "Look-ahead bias" / "stale data bug" from Alligator shift
- **Verdict:** FALSE POSITIVE (consistent with April 17 review HI-3). Pandas
  `shift(8)` shifts data DOWN (backward in time). At the last bar, `jaw.iloc[-1]`
  = the SMMA value from 8 bars ago, displaced forward to the current position.
  This is the correct Williams Alligator implementation.

### FP-2: Persistence filter cold-start seed bypasses filter
- **File:** `portfolio/signal_engine.py:264-268`
- **Agent claim:** Seeding at `_PERSISTENCE_MIN_CYCLES` instead of 1 bypasses filter
- **Verdict:** FALSE POSITIVE. Cold start returns `votes` unfiltered (line 268).
  On cycle 2, `prev["cycles"]` is 2 (seed), incremented to 3 (>=2), passes. With
  seed=1, cycle 2 would increment to 2 (>=2), also passes. Both produce identical
  behavior. The filter is intentionally bypassed on cold start.

### FP-3: `SignalDB` thread safety
- **File:** `portfolio/signal_db.py:25-37`
- **Agent claim:** Shared connection across threads
- **Verdict:** FALSE POSITIVE in practice. Each call to `log_signal_snapshot` and
  `backfill_outcomes` creates a new `SignalDB` instance. Thread sharing does not
  occur in the current codebase. (Pattern is fragile but not currently broken.)

### FP-4: `price_targets.py` fill_probability_buy math
- **File:** `portfolio/price_targets.py:98-107`
- **Agent claim:** "Incorrect mathematical transformation"
- **Verdict:** FALSE POSITIVE. Agent self-retracted during review — the reflection
  principle and sign conventions are mathematically correct.

---

## DOWNGRADES

### DG-1: `_get_regime_gated` called twice with divergent results
- **Original severity:** HIGH -> **Actual:** LOW
- **Reason:** Idempotent — double-gating HOLD->HOLD is a no-op. Only logging
  impact: debug output from `_weighted_consensus` won't show already-HOLD signals.

### DG-2: `kelly_metals.py` MAX_POSITION_FRACTION=0.95
- **Original severity:** MEDIUM -> **Actual:** LOW (design concern)
- **Reason:** User prefers 5x, not 10x. Half-Kelly at realistic parameters
  produces ~8% of cash, well below the 95% cap.

### DG-3: `realized_skewness` academic interpretation
- **Original severity:** MEDIUM -> **Actual:** LOW (design concern)
- **Reason:** Signal is in shadow mode with zero accuracy samples. Academic
  validity is a concern for promotion decision, not production risk.

### DG-4: `trigger.py` grace flag + disk state not atomic
- **Original severity:** HIGH -> **Actual:** LOW
- **Reason:** Requires exception between lines 189-191 during first cycle after
  restart. Extremely unlikely, and consequence is one spurious T3 trigger.

---

## Cross-Cutting Themes

### Theme 1: Union vs Replace Semantics in Hierarchical Configs
`_get_regime_gated` (P0-1) and `_get_horizon_disabled_signals` (correct) have
identical data structures but different override semantics. This is a systemic
pattern risk: any new hierarchical config (per-ticker, per-regime, per-horizon)
must decide union vs replace. Recommend:
- Add a `_hierarchical_merge()` utility used by all such lookups
- Add a module-load test asserting `_get_regime_gated("ranging", "3h")` is a
  superset of `_get_regime_gated("ranging", None)`

### Theme 2: Stale State After Sequential Filtering
`active_voters` is computed before persistence filtering (P1-2), creating a stale
count. The same pattern applies to `core_active` (line 2900) — also pre-persistence.
The `extra_info["_voters"]` export (line 2906) is consumed by Stage 4's dynamic_min
check. Recommend: all derived counts recomputed from `consensus_votes` after the
final filter step.

### Theme 3: File-Based IPC Without Coordination
`contract_state.json` (P1-7), `trigger_state.json`, `portfolio_state.json` are all
read-modify-written by multiple callers. `portfolio_mgr.py` uses proper locking;
`loop_contract.py` and `trigger.py` do not. Recommend: adopt the same
file-level lock pattern from `portfolio_mgr.py` for all shared state files.

### Theme 4: `proc.kill()` vs Tree-Kill on Windows
`bigbet.py` (P1-1) and `multi_agent_layer2.py` (P1-6) both use `proc.kill()` for
Claude CLI subprocesses on Windows, which only kills the direct child. `claude_gate.py`
correctly uses `taskkill /T /F` for tree-killing. Any new subprocess caller must
use the gate's `_kill_process_tree()` helper.

### Theme 5: Error Dicts Masquerading as Valid Data
`data_collector.py` (P2-18) returns `{"error": str(e)}` on failure, which passes
`is not None` checks. Similar patterns exist in other fetchers. Recommend: adopt
a `Result[T, E]` pattern or consistently return `None` on failure.

---

## Regression Check Against April 17 Review

| April 17 Finding | Status |
|-------------------|--------|
| CR-1: Drawdown blind when prices stale | **RECURRED** — now P0-4. Not fixed. |
| CR-2: Shared state deadlock | FALSE POSITIVE then, still FALSE POSITIVE now |
| CR-3: Momentum exit on pre-entry history | Fixed (MOMENTUM_EXIT_MIN_HOLD_SECONDS=300) |
| CR-4: Non-atomic outcome backfill in signal_db | **Still open** — P1-10 scope includes this |
| CR-5: GPU lock PID recycling | Still open (accepted risk, P2-25 reduces window) |
| CR-6: Concurrent order lock reads | Still open (accepted risk for now) |
| CR-7: blend_accuracy_data reconstructs correct | DOWNGRADED, not causing live issues |

**2 regressions from prior review (CR-1, CR-4) that were not fixed.** CR-1 is now
critical enough to warrant immediate action.

---

## Recommended Fix Priority

**Immediate (today/tomorrow):**
1. P0-1: One-line fix in `_get_regime_gated` — union instead of replace
2. P0-3: One-line fix in `metals_avanza_helpers.py` — add 404 to success
3. P0-4: Fail-safe drawdown circuit breaker when prices stale
4. P1-2: Recompute `active_voters` from `consensus_votes`

**This sprint:**
5. P0-2: Adjust stop-loss cert distance to 15%
6. P1-1: Route bigbet through claude_gate
7. P1-3: Add account_id whitelist check
8. P1-4: Guard fear_greed API response
9. P1-5: Add lock to trade_guards
10. P1-12: Fix realized_skewness normalization window

**Backlog:**
- All P2 findings — prioritize by subsystem when working in that area
- P1-6 through P1-11 — fix during next relevant subsystem work

---

*Review completed 2026-04-24 by 8 parallel code-reviewer agents + lead reviewer
cross-critique. 53 raw findings -> 47 confirmed, 4 false positives, 4 downgrades.
4 P0, 12 P1, 31 P2.*
