# Adversarial Review — Codex

**Reviewer:** Codex (GPT-5 family, via `codex@openai-codex` plugin v0.116.0)
**Date:** 2026-04-05
**Method:** 8 parallel `/codex:adversarial-review --scope branch --base empty-baseline` runs,
one per subsystem, each reading a subsystem-scoped diff from the review worktrees.

Each subsystem's review is presented verbatim per the `/codex:adversarial-review` contract.

---

## Subsystem 1 — signals-core

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: the live vote stack can pass quorum on non-independent raw votes, regime gates can be bypassed by stale ticker history, sub-45% signals are treated inconsistently across paths, collapse adaptation is too slow, and the training pipeline can save leaked/overfit weights.

Findings:
- [high] Quorum is checked on raw vote count, not on the post-gate effective voters (portfolio/signal_engine.py:1343-1538)
  `active_voters` and `core_active` are computed before `_weighted_consensus`, but `_weighted_consensus` can later drop votes via the 45% accuracy gate, correlation-group gating, and top-N exclusion. The same stale raw count is also stored in `_voters` and reused by the dynamic min-voter stage, so three or five same-theme votes can satisfy quorum even when only one effective signal still contributes weight. In ranging/high-vol conditions this defeats the intended protection and assumes raw signals are independent when the code already knows many are not.
  Recommendation: Recompute quorum from the signals that actually survive weighting/gating, or use an effective-voter metric based on correlation clusters / summed normalized weight instead of raw BUY+SELL count.
- [high] Regime gates are bypassed using all-time per-ticker stats (portfolio/signal_engine.py:1318-1338)
  The regime-gating exemption uses `accuracy_by_ticker_signal_cached(acc_horizon)`, which is all-history per-ticker accuracy with no recent window and no regime filter. A signal that used to work on one ticker can therefore exempt itself from today's regime gate even after it has degraded in the current ranging/high-vol regime, which is exactly when the gate is supposed to protect the system. This makes regime behavior sticky and slow to fail safe after a local signal-quality collapse.
  Recommendation: Only grant regime-gate exemptions from recent same-regime performance, or require recent-vs-all-time divergence checks before allowing a stale per-ticker exemption.
- [high] Below-45% signals are force-HOLDed here but treated as counter-signals elsewhere (portfolio/signal_engine.py:605-608)
  `_weighted_consensus` hard-skips any signal below 45% accuracy once it has enough samples. In the same branch, `ticker_accuracy.direction_probability()` interprets the same low-accuracy vote as opposite-direction evidence (`SELL` becomes `1 - accuracy` P(up)). After a ranging-regime flip or signal collapse, one path can emit HOLD while another amplifies the reverse direction from the same history, so the architecture has no single answer to whether sub-50% signals are noise or inverse alpha.
  Recommendation: Pick one policy for sub-50% signals and apply it everywhere: either abstain, or invert only when the estimate is statistically below 50% with enough recent samples.
- [high] Recent-collapse handling has a hard 50-sample blind spot, and the online learner is not consumed live (portfolio/accuracy_stats.py:559-569)
  `blend_accuracy_data()` ignores recent performance entirely until a signal accumulates 50 recent samples, then abruptly switches to a 70/30 or 90/10 blend. That already makes collapse detection slow, and repo usage shows the supposedly faster MWU path only writes `signal_weights.json` from `outcome_tracker.py`; `signal_engine.py` never reads those weights. The result is that a suddenly broken signal keeps voting off stale all-time history until enough bad outcomes pile up to cross the hard sample cliff.
  Recommendation: Replace the 50-sample cliff with continuous recency weighting (EWMA/Bayesian shrinkage or a change-point detector), and either wire `SignalWeightManager` into live consensus or remove the dead adaptation path.
- [high] The training pipeline leaks time structure and saves the model before out-of-sample validation can reject it (portfolio/train_signal_weights.py:58-143)
  `_load_signal_history()` creates one row per `(ts, ticker)`, then `train_weights()` feeds those rows into walk-forward windows sized as if they were hours/days. With many tickers per cycle, a `720`-row '30 day' train window is only a small number of market cycles and can split contemporaneous rows across train/test; then the full model is saved before any walk-forward result is checked. That is a direct path to overfitting recent cross-ticker noise and pushing an unvalidated linear-factor model into the live confirmation/dampen path.
  Recommendation: Window by unique timestamps or higher-level time buckets, keep all rows from a timestamp in the same fold, and block `model.save()` unless walk-forward produces enough windows and clears an explicit OOS-performance floor.

Next steps:
- Make quorum and dynamic min-voter checks depend on post-gate effective voters rather than raw signal count.
- Remove stale all-time exemptions and unify the below-50% signal policy across consensus and probability code.
- Fix training to use true time-based folds and refuse to publish models that do not pass out-of-sample checks.
- Either consume `signal_weights.json` in live weighting or delete the MWU path so failure-response behavior is explicit.

---

## Subsystem 2 — orchestration

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: under load or failure this loop can lose triggers, feed stale context into Claude, break the single-agent invariant, and let contract recovery both stall the live loop and give Claude unreviewed write/shell authority. The contract layer also misses crash/hang paths entirely.

Findings:
- [critical] Contract self-heal can block the live loop and run a writable Claude session without approval (portfolio/loop_contract.py:625-653)
  Any critical contract violation escalates into `_trigger_self_heal()`, which invokes Claude inline from the loop path. That call relies on `invoke_claude()` defaults, so the recovery session gets `Read,Edit,Bash,Write` and waits synchronously for up to 180 seconds. In practice, one bad cycle can both freeze the 60s Layer 1 cadence and give an unreviewed model shell/edit authority over the live trading system.
  Recommendation: Make self-heal opt-in and out-of-band; default to read-only diagnostics, require explicit operator approval for any Bash/Edit/Write action, and never block the market loop on an LLM repair attempt.
- [high] Skipped Layer 2 invocations still consume triggers and overwrite shared context (portfolio/main.py:594-622)
  `run()` writes the fresh summary, writes tiered context, and updates tier state before it knows whether `invoke_agent()` actually accepted the work. If Layer 2 is already running, main just records `skipped_busy` and stops; there is no queue or replay path. Because `check_triggers()` has already advanced its baseline in `trigger_state.json`, the event can be lost, and the shared `agent_context_t*.json` / `agent_summary_compact.json` files the older agent reads may already have been replaced. This also advances `last_full_review_time` even when the T3 review never ran.
  Recommendation: Only commit trigger/tier state after a subprocess is accepted, snapshot context into invocation-specific files, and queue pending reasons while another Layer 2 run is in flight.
- [high] Windows timeout recovery can orphan the old Claude process and allow a second one to start (portfolio/agent_invocation.py:163-200)
  On the timeout path, a failed `taskkill` logs that the old process may still be running, then clears `_agent_proc` anyway. The next trigger will therefore see 'no running agent' and can spawn a second Claude subprocess against the same journals, summaries, and portfolio state. Under stress, this breaks the single-agent invariant and can duplicate or conflict with decisions.
  Recommendation: Do not clear `_agent_proc` on a failed kill; enter a latched 'zombie suspected' state and require confirmed exit or operator intervention before allowing any new Layer 2 invocation.
- [high] Multi-agent synthesis can read stale specialist reports from a previous invocation (portfolio/multi_agent_layer2.py:33-216)
  Specialist outputs are fixed global filenames (`data/_specialist_*.md`). Synthesis always reads all report paths, but failed or timed-out specialists are not excluded and the cleanup routine is not wired into the caller. If one specialist fails to refresh its file, the next synthesis can read a leftover report for another ticker or session and treat it as current analysis.
  Recommendation: Use a unique temp directory per invocation, delete it before launch and after synthesis, and only pass report paths from specialists that completed successfully in the current run.
- [high] Loop-contract enforcement is bypassed on the crash and hang paths it is supposed to protect (portfolio/main.py:885-917)
  The main loop only calls `verify_and_act()` when `run()` returns a `report`. Any exception inside `run()` sets `report = None` and skips contract verification entirely; a deadlock or hang inside `run()` never reaches this block at all. That means the invariant system does not fire on partial cycles, ThreadPool hangs, or other hard failures, so the most important failures get only generic crash handling.
  Recommendation: Emit cycle state incrementally and verify from a `finally` path, or move invariant enforcement into an external watchdog that can detect hangs/crashes independently of `run()` returning.

Next steps:
- Stop mutating trigger and tier state until a Layer 2 invocation is actually accepted, and persist per-invocation context snapshots instead of shared files.
- Replace fixed specialist artifacts with invocation-scoped temp files and only synthesize from reports produced in that invocation.
- Make self-heal asynchronous and read-only by default, and retain zombie-agent state until a timed-out Claude process is confirmed dead.
- Audit `portfolio/market_timing.py` separately; it only models US DST and still hardcodes a 07:00 UTC EU open window.

---

## Subsystem 3 — portfolio-risk

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: state writes are not safe under concurrent updates, the new risk controls are advisory or dead-on-arrival, and several sizing/risk modules can produce materially wrong capital or P&L numbers.

Findings:
- [critical] Portfolio state writes are atomic but not concurrency-safe (portfolio/portfolio_mgr.py:39-61)
  `load_state()`/`save_state()` expose an uncoordinated read-modify-write cycle. Under concurrent writers, two fill handlers can load the same snapshot, each mutate cash/holdings/transactions, and the later `save_state()` will overwrite the earlier one because the underlying helper only replaces the file; it does not lock or compare-and-swap. That can silently drop a cash debit/credit or one side of a position update, which is exactly how you end up with fills recorded without matching cash movement or with positions double-counted/missing. This is an inference about the API contract under the concurrent-writer scenario you asked to review, not a proven live call path in this branch.
  Recommendation: Replace separate load/save calls with a locked or versioned `update_state()` transaction that retries on conflict, and require idempotent fill IDs so retries/concurrent writers cannot clobber prior updates.
- [high] Overtrading guards never get updated from actual fills (portfolio/trade_guards.py:52-219)
  `check_overtrading_guards()` only reads persisted guard state, but `record_trade()` is the sole mutator for cooldown timestamps, loss streaks, and new-position timestamps. Repo-wide search found no call sites for `record_trade()`, so the guard state stays at its default empty values and cooldown/escalation/rate-limit logic will keep approving repeated trades. Even if a guard does fire, the function only emits warnings, so nothing here can actually stop execution.
  Recommendation: Invoke `record_trade()` in the same post-fill commit path as every successful BUY/SELL, and treat cooldown/rate-limit breaches as blocking failures in the order-placement path rather than informational warnings.
- [high] Concentration and correlation limits are reporting-only (portfolio/risk_management.py:550-607)
  `check_concentration_risk()` turns a >40% position into a warning object instead of a reject, and this module never escalates that warning to a hard stop. Repo-wide search shows these flags are only surfaced through reporting, so a trade can still be approved with a known concentration, correlation, or regime violation attached as metadata. That defeats the control at the exact point where it matters: pre-trade gating.
  Recommendation: Move hard portfolio-limit checks into the execution gate and return a blocking verdict for concentration/correlation breaches; keep summary reporting as a secondary sink, not the enforcement mechanism.
- [high] Kelly sizing fabricates realized edge by reusing buys across sells (portfolio/kelly_sizing.py:55-104)
  `_compute_trade_stats()` computes one weighted-average buy price from all BUYs for a ticker, then scores every SELL against that same average without respecting execution order or remaining shares. A SELL can therefore be benchmarked against inventory bought later, and the same buy inventory is effectively reused across multiple sells. Those distorted win/loss stats flow directly into `recommended_size()`, so the system can size up on an edge that was never actually realized.
  Recommendation: Reconstruct realized round trips with FIFO or remaining-share matching and only use buys that existed before each sell; reusing the round-trip pairing logic from `equity_curve.py` would be a safer base.
- [high] t-copula VaR/CVaR math does not match the documented model (portfolio/monte_carlo_risk.py:272-290)
  After generating `T_samples`, the code computes `U = t.cdf(T_samples)` and then immediately applies `t.ppf(U)`, which is an identity transform even though the comment says this step should recover normal quantiles. The result is that df=4 Student-t marginals are fed straight into `sigma * sqrt(T)`, so the simulated marginal variance no longer matches the supplied volatility and VaR/CVaR are materially distorted. This is a model-implementation bug, not a tuning disagreement.
  Recommendation: If the intent is a t-copula with GBM marginals, transform `U` with `norm.ppf`; if the intent is true t marginals, standardize them to unit variance and recalibrate/document the volatility assumption accordingly.
- [high] Warrant positions bypass cash accounting and can report impossible SEK P&L (portfolio/warrant_portfolio.py:52-246)
  `record_warrant_transaction()` only appends a transaction and mutates unit counts; it never debits cash or fees from any portfolio ledger, so warrant exposure can be recorded without consuming capital. Separately, `warrant_pnl()` ignores the passed `fx_rate` entirely and linearly applies `underlying_change * leverage`, which can drive `current_implied_sek` negative for a long warrant. That means the system can both hide warrant exposure from concentration/drawdown/VaR checks and publish economically impossible SEK P&L.
  Recommendation: Track warrant fills in the same atomic cash ledger as other positions, include fees/capital usage, use FX in SEK valuation, and clamp long-warrant value at zero/knockout instead of allowing negative implied prices.

Next steps:
- Implement a single atomic portfolio update API with locking/version checks and idempotent fill identifiers, then route every fill write through it.
- Wire trade guards and concentration/correlation checks into the actual pre-trade execution path so they can block orders, and add tests that repeated fills and over-limit trades are rejected.
- Rework Kelly/VaR/warrant math from realized lot matching and corrected marginal models, then add regression tests for partial sells, FX changes, leveraged warrants, and tail-risk scenarios.

---

## Subsystem 4 — metals-core

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: the silver fast-tick state is not durable or reset correctly, Avanza session expiry can stall monitoring before health state flips, persisted OFI is stale-by-design, and the ORB backtest materially overstates edge with look-ahead trade scoring.

Findings:
- [high] Silver fast-tick loses its entry anchor on normal position saves (data/metals_loop.py:351-396)
  `_silver_init_ref()` relies on a persisted `underlying_entry`, but `_load_positions()` never restores that field and `_save_positions()` rewrites the state file with only `active/units/entry/stop` plus sell metadata. The separate `_silver_persist_ref()` write therefore gets dropped by the next ordinary position save. After a restart, or after any later `_save_positions()` call, the silver monitor falls back to current XAG as the reference price, so threshold/P&L alerts are anchored to the wrong baseline and can miss real drawdowns.
  Recommendation: Make `underlying_entry` a first-class persisted field in both load/save paths, and merge existing state rather than rewriting a narrowed schema.
- [high] Mid-cycle Avanza expiry can short-circuit the loop before session health updates (data/metals_loop.py:4836-4902)
  The main cycle fetches active-position prices first and immediately `continue`s to sleep on a fetch error, while the explicit session-health poll happens later and only every 20 loops. If Avanza expires after startup, a 401 during `fetch_price()` can skip the rest of the cycle for up to ~20 minutes: no trigger evaluation, no holdings reconciliation, and no `session_healthy` flip/expiry alert. This is especially risky because the cycle is the only place stop/exit logic runs.
  Recommendation: Run `_check_session_and_alert()` before any Avanza I/O each cycle, and treat auth failures from `fetch_price`/`fetch_account_cash` as immediate session-death signals instead of generic price errors.
- [medium] Silver fast-tick state survives sells and contaminates the next position (data/metals_loop.py:4795-4798)
  The fast-tick rearm path only initializes silver monitoring when `_silver_underlying_ref` is `None`. There is a `_silver_reset_session()` helper, but repo inspection shows no call site on silver position close/reopen. After a silver sell, the old `_silver_underlying_ref`, alert dedupe set, and session highs/lows remain live; when a new silver position is reactivated, it inherits the previous trade's anchor and muted alert levels. That can suppress fresh threshold alerts or fire them against the wrong entry reference.
  Recommendation: Call `_silver_reset_session()` on every silver position close and before any silver reactivation/new fill, then initialize a fresh reference for the new position.
- [medium] Cross-process microstructure state is stale-by-design (portfolio/microstructure_state.py:115-127)
  `load_persisted_state()` rejects any microstructure state older than 120 seconds, but the producer only persists every fifth snapshot from the 60s main loop, i.e. roughly every 2.5-5 minutes. A reader in another process will therefore see `None` for much of the day or operate on a value already outside its freshness contract. That makes the persisted rolling OFI/spread state unreliable exactly when it is supposed to bridge processes.
  Recommendation: Either persist on every accumulation or raise the freshness TTL above the worst-case write interval; also include per-snapshot age so consumers can degrade gracefully instead of dropping the signal entirely.
- [medium] ORB backtest credits impossible trades using end-of-day extrema (portfolio/orb_backtest.py:173-210)
  `_simulate_trades()` books a win whenever the day's low is below the buy target and the day's high is above the sell target, but it never checks which happened first. A day that rallies to the high before ever touching the buy level is still scored as 'buy low, sell high'. The fallback exit at `(actual_high + actual_low) / 2` also uses full-day information that is unavailable at trade time. That is classic look-ahead bias and materially inflates the ORB model's hit rate, win rate, and P&L.
  Recommendation: Rework the backtest to use candle-by-candle sequencing (or at minimum enforce low-before-high ordering via timestamps) and model unfilled exits with only information available at the exit decision time.

Next steps:
- Fix silver monitor state lifecycle first: persist `underlying_entry` correctly and reset fast-tick state on every silver close/reopen.
- Move Avanza session validation ahead of all Avanza API calls in the main cycle and fail closed on auth errors.
- Rebuild the ORB validation/trade simulation without end-of-day look-ahead before using its reported edge operationally.

---

## Subsystem 5 — avanza-api

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: the order layer can overcommit exits, silently turn malformed input into live SELLs, execute the wrong pending order on a generic Telegram confirm, and report stop-loss deletion success when the stop may still be active. The TOTP auth singleton also has no credible recovery path after expiry.

Findings:
- [critical] Sell and stop-loss placement never enforce the position-size invariant (portfolio/avanza_session.py:346-528)
  `place_sell_order()` funnels into `_place_order()` and `place_stop_loss()` posts a new hardware stop directly, but neither path checks current holdings or sums existing active stop-loss volume for the same account/orderbook. A manual SELL placed while a stop is already live, or a second stop added on top of the first, can exceed the position size. The likely impact is an unprotected/stale exit or an extra sell attempt when the remaining stop later fires.
  Recommendation: Before any SELL or stop-loss POST, fetch the current position and active stop-losses for that account/orderbook and reject requests where `requested_sell_volume + active_stop_volume > position_volume`. Apply the same guard in the parallel `portfolio.avanza.trading` path so the invariant is enforced uniformly.
- [critical] Any non-`BUY` side in the no-page facade becomes a live SELL (portfolio/avanza_control.py:313-325)
  `place_order_no_page()` only special-cases `BUY`; every other value falls through to `_place_sell_order()`. A typo, enum drift, or missing value therefore fails open into a sell order instead of being rejected. In a trading facade this is a dangerous default because bad caller input can liquidate a position.
  Recommendation: Validate `side` explicitly against `BUY` and `SELL` and raise on anything else. Do not default unknown values to a trading action.
- [high] Telegram confirmation is global, not order-specific or atomic (portfolio/avanza_orders.py:99-203)
  `_check_telegram_confirm()` reduces approval to a single boolean for any plain `CONFIRM` message from the chat, and `check_pending_orders()` updates disk only after `_execute_confirmed_order()` returns. If multiple orders are pending, the wrong one can be executed because the confirmation is not tied to an order id. If two pollers run at once, both can observe the same pending record and submit the live order before either save wins.
  Recommendation: Require `CONFIRM <order-id>` (or another per-order nonce), atomically mark the order as `executing` and persist that state before contacting Avanza, and protect the pending-order/Telegram-offset store with a process-safe lock or transactional backend.
- [high] `delete_stop_loss_no_page()` reports success even when Avanza said the delete failed (portfolio/avanza_control.py:361-374)
  `_api_delete()` returns an `ok` flag, but this wrapper ignores it and returns `(True, result)` for any non-exception response. A 403/422/500 from Avanza will therefore be treated as a successful cancellation. Downstream code can place a replacement order or stop while the original stop is still active, recreating the exact over-volume exit condition this layer is supposed to prevent.
  Recommendation: Return `result.get("ok", false)` instead of unconditional success, and only treat 2xx/404 as deleted. Surface the HTTP status to callers and block follow-on exit placement when deletion failed.
- [high] The TOTP singleton has no real reauth path after session expiry (portfolio/avanza/auth.py:74-120)
  `AvanzaAuth.get_instance()` permanently reuses the first authenticated client until `AvanzaAuth.reset()` is called. I could not find any caller in this tree that resets it on auth failure, and the public client wrapper does not automatically cascade that reset. The likely result is that once the underlying TOTP session/token expires, subsequent calls keep handing back the stale client instead of reauthenticating.
  Recommendation: Make the client-level reset path also clear `AvanzaAuth`, and on auth-specific failures catch, reset both singletons, and retry once. If automatic reauth is not allowed, fail fast with one explicit, centralized error path instead of silently reusing the dead singleton.

Next steps:
- Add one shared pre-trade risk gate for SELL/stop-loss flows that validates side, position size, and outstanding stop-loss exposure before any live POST.
- Make confirmation and cancellation flows fail closed: order-specific approvals, atomic state transitions, and truthful delete success handling.
- Unify auth recovery so TOTP expiry cannot leave the process stuck on a stale singleton.

---

## Subsystem 6 — signals-modules

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: the new signal pack has calendar logic that misapplies to unsupported assets, event timing that can be hours wrong, static calendar data that silently decays to inert HOLDs, and an LLM-fundamental path that masks model failures as fresh neutral output. Trend-family votes are also materially over-counted.

Findings:
- [high] US equity seasonality is emitted for any ticker (portfolio/signals/calendar_seasonal.py:47-369)
  `compute_calendar_signal()` only accepts a DataFrame, so it cannot distinguish equities from metals, crypto, or any other 24x7 market. The sub-signals it always runs are explicitly equity/US-calendar rules (`Monday historically bearish for equities`, `Sell in May`, January effect, US pre-holiday, Santa rally, FOMC drift). If this module is registered in a generic engine, unsupported assets inherit structurally wrong BUY/SELL votes with no skip reason or applicability guard.
  Recommendation: Require `context`/`ticker` input, whitelist supported asset classes, and disable or replace unsupported sub-signals for crypto, metals, and other non-US-equity instruments.
- [high] Economic-event windows are mis-timed for timezone-aware bars (portfolio/signals/econ_calendar.py:30-36)
  `_get_current_date()` relabels the last bar timestamp as UTC with `replace(tzinfo=UTC)` instead of converting it. For any timezone-aware feed, that discards the original offset, so a bar stamped in New York or another local market timezone is shifted by several hours before the 4h/24h/48h event windows are evaluated. Around FOMC/CPI this can flip `SELL`/`HOLD` at exactly the wrong time in both live trading and backtests.
  Recommendation: Convert aware timestamps with `astimezone(UTC)`/`tz_convert('UTC')`, define explicit handling for naive timestamps, and reject stale timestamps instead of silently reinterpreting them.
- [high] Expired calendar data degrades to an indistinguishable neutral signal (portfolio/signals/econ_calendar.py:165-210)
  When the static econ/FOMC schedule runs out, the module only logs a warning and then majority-votes the default HOLDs. Callers that consume the returned payload see an apparently healthy neutral signal instead of a stale-data failure, so the protection this module is supposed to provide around major events simply disappears once the date lists age out. That is especially risky because the repository context says this signal depends on hard-coded event lists.
  Recommendation: Expose freshness/error state in the returned indicators and fail closed or disable the module once the last known event date is behind `ref_date`.
- [high] LLM refresh failures are cached as fresh and collapse to silent HOLD (portfolio/signals/claude_fundamental.py:543-773)
  The parsers trust model fields with raw `float(...)` conversion, so one malformed `confidence`/`conviction` value can raise and abort the whole tier refresh. `compute_claude_fundamental_signal()` marks `_cache[tier]["ts"]` as fresh before the background thread succeeds, and `_bg_refresh()` only logs on failure; the result is that a model outage or garbage response suppresses retries for the full cooldown while `_get_best_result()` falls back to `_DEFAULT_HOLD` with no degradation signal in the payload. This hides precisely the dependency failure the user asked to surface.
  Recommendation: Validate model fields defensively per ticker, keep refresh-in-progress/error state separate from freshness, and only advance the cache timestamp after a successful parse and cache write.
- [medium] Trend confidence is inflated by counting the same factor multiple times (portfolio/signals/trend.py:69-556)
  This module majority-votes MA ribbon, price vs MA200, Supertrend, Parabolic SAR, Ichimoku, and ADX-derived direction as if they were independent features, but most of them are just different transforms of the same closing-price trend. In a ranging regime, ADX contributes only one HOLD while the other trend-followers can still align on the latest drift above or below the same average cluster, so the module can emit a directional vote with overstated confidence. The same trend family is then repeated again in `heikin_ashi.py` and `momentum.py`, which compounds vote inflation at engine level.
  Recommendation: Add a hard regime veto for low-trend markets and downweight or orthogonalize correlated trend followers instead of feeding them into an equal-weight majority vote.

Next steps:
- Add explicit asset-class applicability gates and stale-data/error indicators to the calendar-based modules.
- Fix timezone conversion before computing event proximity windows and add tests with timezone-aware timestamps.
- Separate LLM refresh state from cache freshness so model outages and malformed outputs are surfaced instead of returning silent HOLD.
- Rework trend-family voting so correlated close-trend transforms do not count as independent evidence.

---

## Subsystem 7 — data-external

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: provider failures are repeatedly flattened into false freshness or false absence. This change can disable the earnings safety gate for a full day, overcount fear streaks by poll frequency, overshoot Alpha Vantage budget, publish partial on-chain snapshots as fresh, and feed stale FX/provider-switch data downstream without a machine-readable health signal.

Findings:
- [critical] Transient earnings-provider failures disable the HOLD gate for 24 hours (portfolio/earnings_calendar.py:30-191)
  `_fetch_earnings_date()` returns `None` both for a real "no upcoming earnings" case and for Alpha Vantage/yfinance failures. `get_earnings_proximity()` then caches that `None` for the full 24h TTL, and `should_gate_earnings()` converts it to `False`. A timeout or rate-limit near the open therefore suppresses the earnings gate for the rest of the trading day with no structured health signal, which defeats the risk control this module is supposed to provide.
  Recommendation: Separate provider-error results from true empty results, avoid caching failures for the full TTL, and propagate a degraded status that blocks or flags stock signals instead of silently returning `False`.
- [high] Fear/greed streaks are counting fetches, not days (portfolio/fear_greed.py:33-67)
  `update_fear_streak()` increments `streak_days` every time a fetch lands in the same regime and never checks whether that regime was already counted for the current calendar day or provider timestamp. The surrounding comments say signal_engine uses this as a sustained-days gate, but any intraday caller will turn hours into "days" and trigger contrarian logic far earlier than intended.
  Recommendation: Persist the last counted date/provider reading and increment the streak at most once per new daily observation.
- [high] Alpha Vantage daily-budget accounting ignores failed requests (portfolio/alpha_vantage.py:149-297)
  The refresh loop caps work using `_daily_budget_used`, but that counter is only incremented after a successful normalization. Real Alpha Vantage quota is consumed by every outbound request, including responses that come back rate-limited (`"Note"`) or otherwise unusable. Once the provider starts rejecting calls, this code can keep retrying on later scheduler runs because the local budget still looks available; the short circuit breaker only pauses that for 5 minutes.
  Recommendation: Count every attempted request against the daily budget, persist the budget state across runs, and expose a rate-limited/budget-exhausted status so downstream code knows fundamentals are stale.
- [high] Partial BGeometrics refreshes are stamped fresh and overwrite the cache (portfolio/onchain_data.py:164-260)
  `_fetch_all_onchain()` assigns a fresh `ts` before any requests and saves the snapshot whenever any one of the six metrics succeeds. Failed metrics are just omitted, and that partial result overwrites the persistent cache. `interpret_onchain()` then silently skips missing fields, so provider outages degrade into a fresh-looking but incomplete snapshot that can neutralize MVRV/SOPR/NUPL signals for the full TTL.
  Recommendation: Track freshness and errors per metric, merge partial refreshes with the previous cache instead of replacing it wholesale, and emit explicit coverage/health metadata when required metrics are missing.
- [high] FX fallback returns stale or hardcoded prices as an ordinary live rate (portfolio/fx_rates.py:20-55)
  `fetch_usd_sek()` always returns a bare float, even when the live fetch failed and the value is a stale cache entry or the hardcoded `10.85` fallback. Logging and Telegram are out-of-band and can fail independently, so downstream valuation code has no programmatic way to tell that FX data is degraded. That makes stale FX silently poison every SEK-denominated figure that consumes this function.
  Recommendation: Return structured metadata (`rate`, `source`, `age_secs`, `is_stale`) or raise/block on the hardcoded fallback so affected downstream outputs can be suppressed or clearly marked.
- [medium] Stock timeframe caches survive provider switches and keep provider-native timestamps (portfolio/data_collector.py:115-306)
  Stock data flips between Alpaca and yfinance based on `_current_market_state`, but `_fetch_one_timeframe()` caches only by ticker+label. That means a weekend/closed-session yfinance frame can be served back after market open instead of refreshing from Alpaca, for up to the label TTL. The problem is compounded by the fact that neither path normalizes timestamps to a common timezone before caching, so cached frames can carry different provider-native time representations under the same key.
  Recommendation: Normalize all candle timestamps to UTC before caching, and include backend/market-state in the cache key or explicitly invalidate cached stock frames when the provider source changes.

Next steps:
- Distinguish provider failure from true empty data everywhere a safety gate or cache currently collapses both to `None`/`False`.
- Add machine-readable freshness/health metadata to provider outputs so downstream code can block, downgrade, or annotate poisoned signals instead of treating them as live.
- Add regression tests for rate-limit exhaustion, intraday fear-greed polling, earnings-provider outages, partial on-chain refreshes, FX fallback, and stock provider-switch transitions.

---

## Subsystem 8 — infrastructure

# Codex Adversarial Review

Target: branch diff against empty-baseline
Verdict: needs-attention

No-ship: the infrastructure layer still has blocking failure modes. Under normal faults it can drop queued alerts, break GPU exclusivity after 5 minutes, lose log/journal entries during rotation, duplicate Telegram sends on retry, return success before state files are crash-durable, and hang past a subprocess timeout.

Findings:
- [high] Analysis throttle clears the queue even when delivery failed (portfolio/message_throttle.py:102-112)
  `_send_now()` ignores the boolean result from `send_or_store()` and always writes `last_analysis_sent` while deleting `pending_text`. If Telegram is down, credentials are bad, or `send_or_store()` returns `False`, the only queued analysis is discarded and the cooldown is still reset. The impact is silent alert loss for an entire cooldown window. Only clear the pending payload and advance `last_analysis_sent` after a confirmed send, and leave the message queued on failure.
  Recommendation: Make `_send_now()` transactional: check the return value from `send_or_store()`, preserve `pending_text` on failure, and update `last_analysis_sent` only after confirmed delivery.
- [high] GPU lock is broken by any holder that runs longer than 5 minutes (portfolio/gpu_gate.py:111-158)
  The cross-process lock is treated as stale purely from the lock file mtime, but the holder never refreshes that mtime while inside the critical section. Any real model load/inference that lasts more than 300 seconds will be declared stale and unlinked by a second process, so the separate `Q:/models/.venv-llm` worker can enter while the first process is still using the GPU. That defeats the exclusivity guarantee and can cause VRAM contention or crashed inference jobs. Use a real OS lock or add a heartbeat plus PID-liveness check before breaking a lock.
  Recommendation: Replace the file sentinel with an OS-backed interprocess lock, or continuously heartbeat the lock file and verify the recorded PID is dead before treating it as stale.
- [high] JSONL rotation can drop live writes during the rotation window (portfolio/log_rotation.py:148-233)
  `rotate_jsonl()` snapshots the current file into `keep_lines`/`archive_buckets`, then later replaces the live file with that old snapshot. Any line appended after the read loop starts but before `os.replace()` runs is absent from both outputs if it was not in the initial snapshot, so journal/messages/log entries can disappear during a normal daily rotation. Because the module is intended to run from the live loop, concurrent writers are the expected case, not an edge case. Rotate by renaming the live file first under an exclusive writer lock, then process the detached file.
  Recommendation: Coordinate rotation with writers using a lock or rename-and-reopen protocol so appends cannot race the read/rewrite path.
- [high] HTTP retry helper replays non-idempotent POSTs with no dedupe (portfolio/http_retry.py:27-62)
  `fetch_with_retry()` retries `POST` on timeouts and connection errors exactly like `GET`. The Telegram send paths in this diff use it for `sendMessage`, so an ambiguous failure where Telegram accepted the first POST but the client timed out will produce a second alert on retry. There is no idempotency key or dedupe record to suppress duplicates, so double-notify is a built-in behavior under transient network faults. Retries for non-idempotent operations need to be opt-in and paired with deduplication.
  Recommendation: Do not automatically retry non-idempotent methods by default; require an explicit opt-in plus a dedupe/idempotency mechanism for Telegram and any other state-changing POSTs.
- [high] Subprocess timeout can still hang indefinitely when job assignment fails (portfolio/subprocess_utils.py:130-140)
  `_run_with_job_object()` ignores whether `AssignProcessToJobObject()` actually succeeded, then on timeout kills only the direct child and immediately calls `proc.communicate()` with no secondary timeout. If the process was never in the job and a descendant kept inherited stdio handles open, the cleanup wait can block forever, so the supposed timeout does not bound execution. This also leaves orphaned descendants outside the job. Check the assignment result, fail closed when the job cannot be applied, and use bounded tree-kill cleanup.
  Recommendation: Validate `AssignProcessToJobObject()` success, terminate the full process tree or job on timeout, and bound the post-kill `communicate()` call with a second timeout/fallback.
- [medium] `atomic_write_json()` returns before the replacement is crash-durable (portfolio/file_utils.py:20-24)
  The function writes JSON to a temp file and calls `os.replace()`, but it never `fsync()`s the temp file or the parent directory. After a power loss or host crash, callers can get a successful return and still lose the new state or revert to the previous file because the data and rename were only in volatile OS buffers. This is especially risky for config/state files and the pending-message queue. If the intent is durable atomic persistence, the temp file must be synced before replace and the directory synced after it.
  Recommendation: Flush and `os.fsync()` the temp file before `os.replace()`, then sync the parent directory where the platform supports it; otherwise narrow the function contract so it does not claim durable atomic writes.

Next steps:
- Make notification delivery transactional and idempotent: preserve queued alerts on send failure and stop retrying Telegram `POST`s without a dedupe key.
- Rework file persistence/rotation around explicit writer coordination, and make atomic writes actually crash-durable with file and directory syncs.
- Replace the GPU sentinel with a liveness-checked interprocess lock, and make subprocess timeout cleanup fail closed when Job Object protection is unavailable.
