# FGL Adversarial Review — Subsystem Findings (verbatim from 8 review subagents)

**Date:** 2026-05-30 · **Baseline:** main @ `1730651f` · whole-subsystem diff vs
orphan branch `fgl/empty-baseline` · clean worktree `Q:/fa-fgl-review`.

Each subsystem was reviewed by a fresh Claude Code subagent with no shared
context (6× `pr-review-toolkit:code-reviewer` for broad subsystems, 2×
`caveman:cavecrew-reviewer` for the tight/security-focused ones). Findings below
are the reviewers' own output; cross-critique + de-dup + severity corrections are
in `02-SYNTHESIS.md`. Line numbers reference the worktree (== main HEAD).

Tally (pre-cross-critique): **76 findings — P0:11 · P1:24 · P2:28 · P3:13.**

---

## 1. signals-core — 19 findings (P0:1 P1:6 P2:7 P3:5)
*pr-review-toolkit:code-reviewer.* Voting engine heavily defended but carries a
fragile multiplier cascade with unverified stage interactions. **`accuracy_degradation`
recurring error root-caused as MIS-MEASUREMENT**, not real degradation.

- `accuracy_degradation.py:498-499` — **P1** — Baseline/current window asymmetry from late backfill: current side recomputes 1d accuracy from a fully-backfilled DB, baseline side reads a snapshot **frozen 14d ago** whose last 24-72h of outcomes were never backfilled into it → structural bias manufacturing >15pp deltas. Fix: recompute baseline accuracy from the live DB over `[now-28d, now-14d]`, or only compare fully-matured windows.
- `accuracy_degradation.py:158 vs 499` — **P1** — Horizon/window double-mismatch: `signals_recent` is 1d-horizon over a 14d *entry* window (outcomes up to 14d old), not a 14d-of-prediction window; 12-alert message dominated by `per_ticker::*` keys, not global collapse.
- `signal_engine.py:3118-3126` — **P1** — Stage-4 dynamic MIN_VOTERS silently overrides metals quorum: returns 5 for ranging/unknown with no asset-class awareness, force-HOLDing metals that legitimately passed the `MIN_VOTERS_METALS=2` gate. Negates the documented metals relaxation in the most common regime. Fix: floor `dynamic_min` at the ticker's `min_voters`.
- `signal_db.py:288,319,347,387` — **P1** — SQL accuracy methods omit the ±0.05% neutral-zone filter the Python `_vote_correct` applies, and count `change_pct==0.0` as a loss for every non-HOLD vote → SQL-path accuracy diverges silently from the live gate. Fix: replicate the neutral band in SQL, or route all accuracy through Python.
- `signal_engine.py:2820-2826` — **P1** — Weighted-consensus denominator excludes HOLD/abstain weight, so two same-direction soft votes (cap 0.30 each) yield `conf=1.0` — full confidence from near-noise before penalties. Fix: floor the denominator with neutral mass.
- `ic_computation.py:130-134` — **P1** — ICIR from overlapping rolling windows (step 1, window 50) → adjacent ICs share 49/50 obs → `ic_std` far too small → `icir` inflated → clears `_IC_STABILITY_MIN=0.10` and applies up to 1.5× weight boosts to unstable IC. Fix: non-overlapping blocks or Newey-West correction.
- `train_signal_weights.py:54` — **P1** — `load_jsonl` of 68MB+ `signal_log.jsonl` with no tail cap (OOM risk); per-(ts,ticker) rows then windowed as a time series → train/test boundaries split mid-timestamp, mixing instruments. Fix: `load_jsonl_tail` cap + per-ticker windows.
- `signal_engine.py:3045-3057,4565-4577` — **P2** — Unbounded multiplier-stacking between clamps (regime ×1.10, volume ×1.15, market-health, linear-factor ×1.10, seasonal ×1.15) fights the calibration-compression stage; boost-after-compression re-inflates before the 0.80 cap. Fix: collect all multipliers, apply once, clamp; assert monotonicity.
- `signal_engine.py:2466-2497` — **P2** — Top-N/leader selection ranks raw accuracy ignoring sample count; a 30-sample 0.55 signal outranks a 5000-sample 0.54 and can rescue a whole correlation group. Fix: empirical-Bayes shrink toward 0.5 or min-sample floor for leadership.
- `signal_engine.py:3046-3050` — **P2** — Regime penalties read raw `regime` string, bypassing `_normalize_regime` used elsewhere → `"Ranging"`/`"trending_up"` skip the 0.75×/0.80× penalty while still gated elsewhere. Fix: normalize once at entry.
- `accuracy_stats.py:225,344,353,536,601,1353,1763` — **P2** — Many accuracy fns pass `change_pct=0` default (not None) into `_vote_correct`, making an absent outcome indistinguishable from a real 0.00% move. Fix: `.get("change_pct")` (default None) uniformly.
- `signal_engine.py:4054-4061,4233` — **P2** — Regime-gate exemption uses a different horizon mapping (`base_hz`) than the main blend (`acc_horizon`); for `horizon="12h"` a signal can be exempted on 1d evidence yet scored on 12h. Fix: one `acc_horizon`, reuse.
- `signal_engine.py:4339-4348` — **P2** — `use_best_horizon` overlay replaces a high-sample blended accuracy (incl. `total`) with a different-horizon record → a 1d-gated 10K-sample 49% signal lifted to its 3h 56% and voting at the relaxed tier. Config-gated off by default. Fix: borrow scalar within same horizon family, keep original sample count.
- `outcome_tracker.py:29-109` — **P2** — `_derive_signal_vote` fallback is stale vs the live engine (old dead-zone semantics) and silently mislabels accuracy if `_raw_votes` is ever absent. Fix: drop the divergent fallback or assert `_raw_votes` presence.
- `meta_learner.py:438-445` — **P3** — `predict()` re-reads metrics JSON from disk twice per call; module is dead relative to the live loop. Fix: cache metrics by mtime.
- `signal_engine.py:4565-4577` — **P3** — Hardcoded metals seasonal month→multiplier table applied only to BUY (asymmetric), no config hook. Fix: move to `seasonality_profiles`, document/symmetrize.
- `signal_weights.py` (whole) — **P3** — Dead MWU code: `signal_weights.json` is never read by `signal_engine` (per outcome_tracker C6); orphan that confuses operators. Fix: delete or mark unused.
- `cusum_accuracy_monitor.py:84-146` — **P3** — Full-state-file RMW per outcome under a thread-only lock; loop + PF-OutcomeCheck race → last-writer-wins drops counters. Advisory only. Fix: batch or cross-process lock.
- `signal_engine.py:3152-3157` — **P3** — Ensemble-entropy HOLD count derived from static `_total_applicable`, not the post-gating vote universe; `max(0,…)` clamp hides it but entropy understates disagreement feeding the 0.6×/0.8× cap.

---

## 2. orchestration — 14 findings (P0:2 P1:5 P2:5 P3:2)
*pr-review-toolkit:code-reviewer.* One structural silent-fail gap fully explains
the recurring `layer2_journal_activity` violation; auth-failure detection + tiered
timeouts + crash recovery are robust.

- `main.py:989` — **P0** — `skipped_busy` re-logged whenever `invoke_agent()` returns False, including after legitimate internal skips that wrote their own status; `skipped_busy` is the latest `invocations.jsonl` row and is **excluded** from `_LEGITIMATE_SKIP_STATUSES` → **root cause of ~20×/week `contract_violation`**. Fix: have `invoke_agent` signal "already logged terminal status"; only log `skipped_busy` otherwise; add internal skips to the whitelist.
- `agent_invocation.py:1622` — **P0** — `status=="failed"` (exit≠0, no auth marker) sends `*L2 FAILED*` Telegram but writes NO journal stub (unlike `incomplete`/timeout) and `failed` is absent from `_KNOWN_FAILURE_STATUSES` → a real crash-exit looks identical to the skipped_busy false positives, burying genuine silent crashes. Fix: write a `failed` journal stub + add to `_KNOWN_FAILURE_STATUSES`.
- `agent_invocation.py:1080` — **P1** — `specialist_quorum_fail` returns with no journal entry and no fallback → trigger gets no decision AND triggers the same unsuppressed double-log. Fix: fall back to single-agent/autonomous or write a stub.
- `agent_invocation.py:870` — **P1** — `_agent_log_start_offset` is set outside the lock-protected critical section that sets `_agent_start`/`_agent_proc`; a watchdog completion-scan can use a stale offset and misattribute auth-log slices. Fix: set offset inside the same locked section.
- `main.py:862` — **P1** — First-of-day Tier-3 review never fires: `check_triggers` persists `last_trigger_date=today` before `classify_tier` reads it, so the "first real trigger of day → T3" branch is always False. Fix: capture date before `check_triggers`, or set it only in `update_tier_state`.
- `trigger.py:285` — **P1** — `_check_recent_trade` writes `last_trigger_time=0` (int) while elsewhere it's an ISO string → type inconsistency; a future `_parse_iso` reader gets None → silently "no trigger". Fix: consistent type.
- `agent_invocation.py:1203` — **P1** — `journal_written` detection via line-count delta + last-ts fallback can mislabel a genuine write as `incomplete` if two writes share the same second AND prune masks the count delta → false `*L2 INCOMPLETE*` + spurious stub. Fix: content-hash/monotonic marker.
- `agent_invocation.py:801` — **P2** — Stack-overflow auto-disable is a process-global; after the 5th overflow Layer 2 is permanently skipped with one INFO log + `skipped_stack_overflow` rows, no recurring alert. Fix: throttled periodic re-alert.
- `loop_contract.py:304` — **P2** — Contract reads `last_trigger_time` stamped at END of cycle (0-900s after the agent spawned) → `trigger_age_s` understates true age, compressing grace. Fix: stamp at trigger-detection time.
- `main.py:832` — **P2** — Buffered (deferred) trigger reasons can be stranded in `trigger_buffer.json` across a restart with no max-age hard flush. Fix: add max-age flush.
- `multi_agent_layer2.py:201` — **P2** — Specialists append `status:"invoked"` to the shared `invocations.jsonl`; if synthesis fails to spawn, a specialist `invoked` row suppresses the contract until it ages out, masking the failure. Fix: distinct status/caller for specialist rows.
- `agent_invocation.py:818` — **P2** — Auth-cooldown scan `break`s at the first non-`skipped*` status; with multi-agent, a specialist `invoked` row is hit first → never inspects the prior `auth_error`, defeating the cooldown during the storm it was built for. Fix: skip `invoked`/specialist rows.
- `claude_gate.py:343` — **P3** — Daily-invocation counter filters on `entry["timestamp"]` but `_log_trigger` writes `ts` → Layer-2 rows invisible to the rate-limit counter. Fix: accept both keys.
- `main.py:53` — **P3** — Singleton lockfile PID rewrite (`seek(0)+truncate()`) can race a reader of the lockfile PID (diagnostic-only). Fix: write PID before truncate.

---

## 3. portfolio-risk — 11 findings (P0:1 P1:4 P2:4 P3:2)
*pr-review-toolkit:code-reviewer.* Leveraged warrant valuation can go negative;
the atomic-RMW helper exists but the real write path bypasses it.

- `warrant_portfolio.py:100-103` — **P0** (validated) — `current_implied_sek = entry_price_sek*(1+implied_pnl_pct)` has no zero floor; a 5× warrant whose underlying passes −20% yields a **negative** per-unit value → negative `total_value_sek`/`pnl_sek` (loss exceeding capital), distorting portfolio value, drawdown, and VaR. Fix: `max(0.0, …)` + knockout flag.
- `warrant_portfolio.py:198-279` — **P1** — `record_warrant_transaction` does an unguarded load→mutate→save (unlike `portfolio_mgr.update_state`); fast-tick silver monitor + main metals cycle interleave → a transaction and holdings delta silently lost; oversell clamp reads stale `current_units`. Fix: module-level lock / atomic RMW.
- `risk_management.py:251-270` — **P1** — When `agent_summary` is empty but holdings exist, `check_drawdown` falls back to cash-only value, ignoring unrealized losses → anti-conservative; breaker never trips while positions underwater on a stale feed. Fix: fail-safe to breach, or value holdings at last-known cost.
- `monte_carlo.py:357` — **P1** — `conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)`: a real `0.0` confidence is falsy → silently replaced; same `or`-trap on `action` (356). Skews `p_up` → drift → every VaR/stop probability. Fix: explicit `None` checks.
- `risk_management.py:739-791` — **P1** — `check_concentration_risk` only ever returns `severity:"warning"`, never `"block"`, and is advisory — no path prevents a BUY at >40% concentration. Fix: emit `block` + executor honors it, or document advisory-only.
- `trade_guards.py:126-330` — **P2** — check and record are separately locked; two near-simultaneous BUYs both pass `len(recent)>=limit` (limit=1) before either records → 2 positions in a 1-position window. Fix: fold rate-limit reservation into one locked op.
- `portfolio_mgr.py:136-159` — **P2** — `update_state` (full-cycle lock) is called nowhere in production; the trade path uses bare `load_state()`/`save_state()` (non-atomic RMW) and the in-process lock can't guard the 3 processes writing the same file. Fix: route all writers through `update_state` + cross-process file lock.
- `equity_curve.py:398-400` — **P2** — Sell-fee proration uses `sell_shares` (full sell) even when a SELL splits across FIFO lots; if the sell exceeds available buys the loop exits early and a fraction of the sell fee is dropped → realized P&L slightly overstated. Fix: prorate over actually-matched quantity.
- `warrant_portfolio.py:200-208` — **P2** — Warrant txns record `units`/`price_sek` but no `total_sek` and SELL records no realized P&L → the clamped-oversell audit trail can't reconstruct cash impact; warrant book is unreconcilable. Fix: record `total_sek` post-clamp on both legs.
- `risk_management.py:84-110` — **P3** — `_streaming_max` reads/streams the file outside `_peak_cache_lock`; duplicated work, peak monotonic so correctness holds. Quality only.
- `monte_carlo_risk.py:408` — **P3** — `compute_portfolio_var` reads `fx_rate` directly, bypassing the validated `_resolve_fx_rate` chain; a stale `fx_rate:1.0` makes `*_sek` VaR ~10× too small. Fix: reuse `_resolve_fx_rate`.

---

## 4. avanza-api — 4 findings (P0:2 P1:1 P2:1)
*caveman:cavecrew-reviewer.* Real-money path: silent session expiry + a
stop-loss reachable via the wrong endpoint in the legacy page path.

- `avanza_session.py:89` — **P0** — `load_session()` does not eagerly re-auth; on expiry it raises `AvanzaSessionError` that callers swallow, then `load_json(SESSION_FILE)` returns None next cycle → multi-day silent auth outage (the 2026-05-23 incident). Fix: verify expiry at `_get_playwright_context()` and fail-closed.
- `avanza_control.py:356-376` — **P0** — Legacy page-based `place_stop_loss(page,…)` routes through `_place_page_stop_loss` which can hit the regular order endpoint `/_api/trading-critical/rest/order/new` instead of `/_api/trading/stoploss/new` → instant fill at worst price on trigger (Mar-3 incident). Fix: unified dispatch that detects stop-vs-regular and forces the stop endpoint.
- `avanza_resilient_page.py:126` — **P1** — `page.goto(..., wait_until="domcontentloaded")` has no explicit timeout; a hung login/BankID/Cloudflare interstitial blocks the first loop cycle and deadlocks subsequent requests. Fix: explicit `timeout=5000`.
- `avanza_session.py:337-342` — **P2** — `api_post()` returns `{"raw": body}` on a 2xx with non-JSON body (e.g. HTML maintenance page); caller `.get("status")` → None → silently treated as failure, never surfaced. Fix: return `{"status":"FAILED","error":"unexpected_response_body"}`.

---

## 5. signals-modules — 5 findings (P0:0 P1:1 P2:3 P3:1)
*pr-review-toolkit:code-reviewer.* Contract well-defended (raise→HOLD, NaN→0.0
sanitized, no `except:pass` anywhere). Real exposure = dead signals + constant-price
history methodology bugs, not crashes.

- `gs_kalman_zscore_regime.py:70-103` — **P1** (validated) — **Dead active signal**: reads `context["gs_ratio_series"]`/`silver_close`/`gold_close`, none of which the engine's `context_data` (signal_engine.py:3702) ever supplies → always HOLD, polluting its accuracy stats and occupying a voter slot. Fix: self-fetch the counterpart metal close (pattern in `copper_gold_ratio.py:77`) or populate the context keys.
- `mstr_mnav_discount.py:175-182` — **P2** — `historical_ratios` uses the single current `btc_price` for every past MSTR close → velocity + z-score sub-signals collapse to MSTR-price proxies (not mNAV dynamics). `_mnav_level` (primary) is correct. Fix: date-aligned BTC close series.
- `stablecoin_supply_ratio.py:209-216` — **P2** — Same constant-price flaw: SSR history holds current crypto price fixed → `ssr_level` z-score reflects only supply variation. Fix: date-aligned historical crypto price.
- `momentum_factors.py:349-358` — **P2** — Seasonality detrend reads the already-mutated prior bar inside the loop → geometric compounding drift (the exact bug fixed in `mean_reversion.py` P1-6). Metals-only, try/except-guarded. Fix: capture `original_close` before the loop, reconstruct from it.
- `calendar_seasonal.py:208-223` — **P3** — Hardcoded approximate US holiday `(month,day)` tuples drift off-by-days year-to-year → pre-holiday BUY fires on the wrong day. Fix: weekday-rule computation or a holidays library.

---

## 6. data-external — 14 findings (P0:1 P1:4 P2:6 P3:3)
*pr-review-toolkit:code-reviewer.* LIVE-FIRST mostly honored, but one silent
stale-masking path + non-persisted budgets + a tz-naive/aware split.

- `price_source.py:240-262` — **P0** — `fetch_klines` catches ALL primary-source exceptions and silently substitutes 10-15-min-stale yfinance with no staleness/source tag → a dead Binance FAPI feed drives metals/crypto signals on phantom lagged data (violates the cardinal LIVE-FIRST rule). Fix: tag `df.attrs["source"]/["stale"]`, consumers down-weight/HOLD, or raise for real-time-critical tickers.
- `http_retry.py:40-43,68-78` — **P1** — `fetch_with_retry` retries POST (and arbitrary methods) on 429/5xx/timeout with no idempotency guard → double-submit on lost-response timeouts. GET-only here but shared with order placement. Fix: only retry idempotent methods by default; opt-in for POST.
- `shared_state.py:312-344` / `alpha_vantage.py:30-32,162-168` — **P1** — Daily budgets are in-memory globals reset to 0 on every restart; crash-loop restarts re-grant the full 25/day AV + ~90/day NewsAPI budgets → real quota blown, key throttled/banned. Fix: persist counter+reset-date, reload on startup.
- `data_collector.py:96,157` — **P1** — Timestamp dtype mismatch: Binance tz-naive, Alpaca tz-aware UTC, yfinance tz-aware local → downstream compare/merge raises or silently off-by-tz. Fix: normalize all to tz-aware UTC at fetch boundary.
- `fx_rates.py:36-39` — **P1** — USD/SEK from frankfurter.app (ECB daily fix, none on weekends) presented as fresh via a 15-min cache / 2h stale threshold; portfolio SEK valuations + knockout math ride a daily-fixed rate labeled fresh. Fix: real-time FX source or model the true daily cadence + label.
- `data_collector.py:96,124-125` — **P2** — Indicators computed on the still-forming last Binance candle (`close.iloc[-1]`) → intra-bar mutation; "closed" value differs from outcome-tracking backfill. Fix: drop final row if `close_time>now`, or document intra-bar intent.
- `data_collector.py:181-184` — **P2** — `fetch_vix` can propagate `float(NaN)` into regime classification (`current>=30` all False → silently "complacent"). Fix: NaN-check before classify.
- `price_source.py:60-68,239` — **P2** — yfinance alias map only covers metals; a Binance-form crypto symbol (`BTCUSDT`) on spot failure isn't aliased → emergency fallback dies. Fix: add `BTCUSDT→BTC-USD` etc.
- `onchain_data.py:277-286,107-119` — **P2** — Missing BGeometrics token serves ≤24h-old cache as current at DEBUG only. Fix: WARNING + stamp age so the on-chain voter discounts it.
- `crypto_macro_data.py:275-286,397-408` — **P2** — JSONL reads via raw `open().read()` loop, bypassing `file_utils` (writes correctly use `atomic_append_jsonl`). Fix: route reads through a file_utils reader.
- `sentiment.py:233-235` — **P2** — NewsAPI budget counted only when the call returns non-empty; a legit zero-article 200 still consumes quota but isn't counted → real quota drains faster than the internal counter. Fix: count every completed request.
- `futures_data.py:36-41` — **P3** — Every `None` treated as a transient breaker failure; a fatal 403 (bad/expired key) counted as flakiness and recovered on a timer → masks permanent auth problem (the multi-week-outage shape). Fix: distinguish fatal 4xx, surface loud + non-recovering.
- `social_sentiment.py:32,65,110,121` — **P3** — Reddit fetchers use bare `requests.get` (no retry/rate-limit/breaker) and report via `print()` not `logger` → failures invisible, 429 not backed off. Fix: route through `fetch_json`, use module logger.
- `metals_cross_assets.py:59-61` — **P3** — `_yf_download` swallows every fetch exception into an empty frame at WARNING → "no data" indistinguishable from "flat market". Fix: structured per-feature availability flag.

---

## 7. infrastructure — 7 findings (P0:2 P1:3 P2:2)
*caveman:cavecrew-reviewer.* (Severity corrections applied in synthesis — see notes.)

- `health.py:161` — **P0→P1** (corrected) — naive `last_heartbeat` parses OK then the line-165 subtraction raises `TypeError` (the `except` wraps only `fromisoformat`); **it raises, not "returns inf"** as written, and only triggers on a corrupted/manually-edited ts (normal path writes aware `datetime.now(UTC)`). Latent → P1. Fix: `.replace(tzinfo=UTC)` on naive parse / catch TypeError on subtraction.
- `health.py:202` — **P0→P1** (corrected) — same naive-datetime issue in `check_agent_silence`. Same fix.
- `file_utils.py:289-292` — **P1→P3** (corrected) — `atomic_append_jsonl` is wrapped in `jsonl_sidecar_lock`, which **serializes appenders cross-process**, so the "two processes fsync-race, one lost" scenario cannot occur through the helper. Residual risk only if a writer bypasses the helper. Fix: covered by B1 (route all writers through the helper).
- `alert_budget.py:44-46` — **P1** — `should_send()` prunes then checks `len()` then appends in separate statements under the lock; if the decision and append aren't one critical section, two threads can both return True and exceed the rate (a dropped/over-sent CRITICAL alert). Fix: append-before-check or single locked op.
- `shared_state.py:88-99` — **P1→P3** (corrected) — claimed `_loading_keys` leak on interrupt between lock-exit and `try:`; the window is ~zero statements and self-heals via the 120s stuck-key eviction. Over-stated. Fix: optional try/finally for tidiness.
- `gpu_gate.py:214` — **P2** — `os.open(O_CREAT|O_EXCL)` then a separate write; if the write fails the lock file exists empty → next `_read_lock` `int(parts[1])` IndexError on empty split → corrupted lock spins forever. Fix: write+flush+close before releasing, or truncate stale empty locks.
- `log_rotation.py:498-504` — **P2** — Rotation builds a tmp, fsyncs, then `os.replace`; a crash between leaves a tmp behind with no cleanup → next rotation can't overwrite the busy tmp on Windows → rotation silently fails, file grows. Fix: use `tempfile.mkstemp` (auto-clean on exception) like `file_utils`.

---

## 8. metals-core — 11 findings (P0:0 P1:4 P2:5 P3:2)
*pr-review-toolkit:code-reviewer.* Live order helpers are correct (right stop
endpoint, cross-process order lock, fail-closed CSRF, robust pre-sell stop-cancel).
Exposure is concentrated in **EOD-flat reachability** (a halted/restarted grid leaves
leveraged inventory overnight with no operator alert) + a naked-position re-arm window.

- `grid_fisher.py:1751-1763` — **P1** — EOD leave-open on halt: `should_halt_global` returns BEFORE the EOD block, so once the grid halts (session-loss limit), `eod_market_flat()` can never run → leveraged inventory carries overnight. Fix: gate only NEW buys on halt; always run liquidation/protection.
- `grid_fisher.py:1538-1579` — **P1** — Naked-position window: old stop cancelled BEFORE new stop placed; on placement failure inventory is unprotected until next-tick rearm. Fix: place-then-cancel (only cancel old on confirmed new-stop SUCCESS).
- `grid_fisher.py:1906-2004` + `metals_loop.py:7236-7268` — **P1** — EOD market-flat fires only in a 5-min window (21:50-21:55); a loop stall/restart/session-down early-return misses it → overnight hold, and repeated `eod_market_sell_failed` only go to the decisions JSONL (no Telegram/critical_errors). Fix: widen/repeat window + after-close reconcile-and-flat + escalate failures.
- `metals_loop.py:7219-7234` — **P1** — Fishing EOD-sell guard `_eod_fishing_sold_today` is set unconditionally even when `emergency_sell` fails → failed EOD sell never retried, position open overnight silently. Fix: set the guard only when every position confirmed sold.
- `metals_loop.py:4994-5005` — **P2** — `_cancel_stop_orders` uses the 1-segment stop-loss DELETE `/_api/trading/stoploss/{orderId}` instead of the canonical 2-segment `/{accountId}/{stopId}` (see `delete_stop_loss:481`); 404 fallback also wrong-verb → cascading cancels silently fail → orphaned/stacked stops can block sells. Latent behind `STOP_ORDER_ENABLED=False`. Fix: reuse `delete_stop_loss` (2-segment).
- `metals_swing_trader.py:3165-3210` — **P2** — `_execute_sell` sends `pos["units"]` with no clamp to live holdings; `_reconcile_swing_positions` prunes only fully-absent positions, never clamps a smaller live holding → repeated oversell attempts Avanza rejects (`short.sell.not.allowed`), position stuck. No real oversell (broker blocks). Fix: clamp to reconciled live holding.
- `metals_swing_trader.py:3073` + `metals_swing_config.py:323` — **P2** — Swing EOD exit effectively disabled (`EOD_EXIT_MINUTES_BEFORE=0` → only at/after 21:55 when the loop no longer calls swing) → swing warrants carry overnight on the broker trailing stop only. Documented user override; flagging to keep it conscious.
- `fin_snipe_manager.py:529-563` — **P2** — `_compute_stop_plan` sets hard-stop at `entry_avg*(1-5%)` with only a 3%-from-bid guard; the MINI knockout `barrier_level` is in the snapshot but never checked against the trigger. The memory rule "never place a stop near a MINI barrier" is implicit, not enforced. Fix: explicit barrier-margin guard.
- `metals_loop.py:3736-3927` — **P2** — `emergency_sell` sends `pos["units"]` without re-clamping to live holdings, relying on the `short.sell.not.allowed` recovery branch instead of a pre-check. Latent behind `EMERGENCY_SELL_ENABLED=False`. Fix: clamp to freshly-fetched live holding.
- `silver_monitor.py:55-66` — **P3** — Non-atomic `json.load(open(...))` of `metals_positions_state.json` (violates file_utils invariant). DEPRECATED module, read-only. Fix: delete or route through `file_utils.load_json`.
- `grid_fisher.py:1062-1076` — **P3** — `_safe_session_call` logs order timeouts/errors only to the grid decisions JSONL, no Telegram/critical_errors → persistent session failure during placement/EOD-flat invisible to operator. Fix: escalate repeated failures.

---

## ⚠ LIVE-EVIDENCE CORRECTION — `contract_violation` root cause (supersedes signals-core/orchestration static analysis)

Both the orchestration subagent AND the orchestrator's own pass independently
concluded the recurring `layer2_journal_activity` violation was driven by the
`skipped_busy`-clobber mechanism (main.py:989 / loop_contract.py:353). **Live data
refutes that as the dominant cause.** `last_invocation_status` on the last 40
real violations (`data/contract_violations.jsonl`, 116 total logged):

| status | count | share |
|--------|------:|------:|
| **success** | **29** | **72%** |
| invoked | 6 | 15% |
| incomplete | 2 | 5% |
| skipped_busy | 1 | 3% |
| auth_error | 1 | 3% |
| skipped_auth_cooldown | 1 | 3% |

So the `skipped_busy`-clobber story (both static analyses' headline) explains only
**~3%** of fires. The **dominant** driver is `status="success"`: the agent ran,
exited 0, and journaled — yet the contract still fired. That is a **contract-window
/ timestamp-lag** defect, not a silent failure:

- `last_trigger_time` is stamped at *end of cycle* (loop_contract.py:304, health.py),
  and the journal-vs-trigger test is `journal_ts >= last_trigger - 5s`.
- With back-to-back triggers, trigger B updates `last_trigger_time` to "now" while
  the newest journal entry is still agent-A's (older). The contract reads
  "newest journal older than newest trigger" → fires, even though the system IS
  journaling — just one invocation behind the latest trigger.
- The dedup (precondition 5) only suppresses re-fires of the *same* trigger, not
  this cross-trigger lag.

**Both static-analysis conclusions were real but ~3% of the picture.** This is the
single most important result of the review: two independent agents agreed on a
plausible mechanism, and only validation against the live journal exposed it as a
minority case. Treat the `success`-lag defect as the primary fix and the
`skipped_busy`/`failed`-stub gaps as real-but-secondary.

**Recommended fix (primary):** in `check_layer2_journal_activity`, compare the
newest journal ts against the trigger that *actually drove the most recent
invocation* (correlate via `invocations.jsonl` reasons/ts), not the latest
end-of-cycle `last_trigger_time`; OR require the journal to post-date the
*invocation* row, not the trigger stamp; AND stamp `last_trigger_time` at
trigger-detection time. **Secondary:** the `skipped_busy`-clobber + `failed`-no-stub
gaps (orchestration P0×2) still need fixing — they account for the residual ~10%
and one of them can mask a genuine crash.
