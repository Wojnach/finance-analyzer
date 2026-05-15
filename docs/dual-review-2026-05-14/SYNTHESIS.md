# Dual Adversarial Review — Synthesis (2026-05-14 / 2026-05-15)

## Methodology

Eight subsystems reviewed by two independent reviewers each (main-thread Claude
and a Claude subagent with isolated context posing as Codex; the real Codex CLI
was quota-locked at session start and re-attempted on the next session). Each
subsystem was presented as a single commit on top of an empty baseline branch,
so the full subsystem appears as one diff. Cross-critiques in
`cross-N-{subsys}.md` reconcile agreements, surface what each reviewer missed,
and add a third-pass list of things neither reviewer noted.

This synthesis collapses all 24 review docs into a single prioritized
punch-list. **Severities reflect cross-critique outcomes**, not either
reviewer's initial call — where they disagreed, the cross-critique winner is
adopted here.

## P0 — money-losing or data-corrupting (must fix)

Ordered by severity within group; numbered for reference.

### Trading-direction & leverage math

1. **`portfolio/risk_management.py:344-394` — `compute_stop_levels` LONG-only AND ignores leverage.**
   `current_price < stop_price` only fires LONG; BEAR certs never trip stops.
   `atr_pct` is underlying ATR, not leveraged-cert ATR — a 5x cert is already
   20-40% down before a 2*ATR stop on the underlying. Compound bug, two fixes.
   Subsystem 3.

2. **`portfolio/exit_optimizer.py:60-74, 303-340, 568, 741-750` — `Position` lacks `direction`; warrant PnL math LONG-only.**
   For BEAR certs, `pct_move = (exit - entry)/entry` has the wrong sign — EV
   inverted, "best" candidate is actually worst. Candidate filter at `:568`
   prunes winning SHORT candidates. Production paths via
   `fin_snipe_manager._summarize_market` and `reporting.py:684` silently
   produce LONG-only plans. Subsystem 4.

3. **`portfolio/kelly_metals.py:215-221` — Kelly leverage division saturates to MAX_POSITION_FRACTION=0.95 on any positive edge.**
   With XAG defaults (avg_loss=2.43%, leverage=5), `cert_loss_frac=0.1215`. Any
   half_kelly ≥ 0.122 (full_kelly ≥ 24%, which a 55%/3:2 R:R produces routinely)
   saturates → 4.75× notional in a 5x warrant. Subsystem 3.

4. **`portfolio/trade_validation.py:67-81` — `max_cash_pct=50%` is on order_value, not leverage-adjusted notional.**
   A 50% allocation in a 10x warrant is 5x bankroll notional and still passes
   "safe". Gate that all manual trades pass through is leverage-blind. Subsystem 3.

5. **`portfolio/grid_fisher_config.py:50` GRID_STOP_PCT=3.5 violates `5x certs need ≥15% stops` rule.**
   `metals_swing_config.STOP_LOSS_WARRANT_PCT = 30.0` for same family — grid_fisher
   is 10x tighter. XAG 1min ATR commonly exceeds 0.3% underlying = 1.5% on 5x cert,
   so 3.5% cert stop is barely 2× ATR. Every fill rotates to a sell+stop where stop
   fires within minutes. Subsystem 4.

### Silent-failure / fail-open patterns (3-week-outage shape)

6. **`portfolio/iskbets.py:287-356` — Layer 2 gate defaults to APPROVE on empty Claude output.**
   `approved=True` initializer; only `auth_error` triggers SKIP. Empty output from
   `success=True` falls through with approved=True. Real money positions opened
   off an LLM that returned no text. Same shape as the documented 3-week outage. Subsystem 4.

7. **`portfolio/agent_invocation.py:572` — `_kill_overrun_agent` clears `_agent_proc=None` even on `kill_ok=False`.**
   Next cycle spawns a second `claude -p` while the first is still alive; both
   append to `agent.log` and race on portfolio_state. Auth-error scan's
   `_agent_log_start_offset` reset by the new spawn — `"Not logged in"` from the
   old process becomes invisible. Recreates the 3-week silent-auth outage. Subsystem 2.

8. **`portfolio/multi_agent_layer2.py:163-185, 210` — specialists bypass `claude_gate`.**
   No `_invoke_lock`, no tree-kill (Node.js descendants leak), no
   `invocations.jsonl` entry (rate-limiter/cost blind), auth scan only
   post-wait. Documented forbidden direct-Popen pattern. Subsystem 2.

9. **`portfolio/forecast_accuracy.py:254-329` — `backfill_forecast_outcomes` truncates predictions file when `max_entries` hits.**
   Loop appends every entry to `modified_entries` then `break`s; rewrite drops
   tail. 2026-05-04 flagged, fix never landed. Predictions destroyed silently. Subsystem 1.

10. **`portfolio/forecast_signal.py:365,372` vs `portfolio/forecast_accuracy.py:145-148` — schema mismatch.**
    Writer emits nested `chronos`/`prophet` payloads; scorer reads
    `sub_signals`/`raw_sub_signals`. Every Chronos prediction contributes ZERO
    scored votes — model degradation invisible. Subsystem 1.

11. **`portfolio/telegram_poller.py:361` — `atomic_write_json` on the symlinked `config.json` severs the symlink.**
    Embeds API keys in the repo working tree on next commit. Re-introduces the
    Mar 15 leak failure mode. Subsystem 8.

12. **`portfolio/file_utils.py:228-238` — `jsonl_sidecar_lock` path alias bug.**
    Different aliases (relative vs absolute) → different lock files → torn-write
    contract broken. Latent until a path-mismatch drift; would re-trigger the
    2026-05-11 signal_log torn-write incident. Subsystem 8.

13. **`portfolio/claude_gate.py:662 vs 777` — `invoke_claude_text` return-type mismatch (signature says 3, body returns 4).**
    Any unpacking caller raises `ValueError: too many values to unpack`. Subsystem 8.

### Broker-side blast radius

14. **`portfolio/avanza/trading.py:80-92` — unified `place_order` bypasses whitelist + MAX_ORDER ceiling.**
    Legacy `avanza_session.place_*_order` has whitelist + 50K SEK cap +
    `avanza_order_lock`; new unified path has none of them. Migration silently
    drops safety nets. Subsystem 5.

15. **`portfolio/avanza_session.py:720-811` — `place_stop_loss` has NO MAX_ORDER guard; `place_trailing_stop` has NO trail_percent validation.**
    A 250K SEK stop leg is accepted and fires a 250K SEK sell on trigger.
    `trail_percent=-5` (sign error) accepted. Subsystem 5.

16. **`portfolio/avanza/account.py:64-94` — `get_buying_power` silent zeros on account-not-found.**
    The exact pattern the C7 fix in `avanza_session.py` was written to eliminate.
    Regression in the unified package. Subsystem 5.

17. **`portfolio/grid_fisher.py:1576-1582` — EOD market-sell falls back to 0.01 SEK floor when bid=0 AND avg_entry=0.**
    Liquidates 1000-unit position at 1 öre. Subsystem 4.

### Risk math

18. **`portfolio/monte_carlo_risk.py:419` — `compute_portfolio_var` bypasses `_resolve_fx_rate` sanity band.**
    Stale `fx_rate=1.0` understates SEK VaR by 10x. `_resolve_fx_rate` exists
    specifically to defend against this (P1-15 / 2026-05-02) — VaR path
    bypasses it. Subsystem 3.

### Signal-modules

19. **`portfolio/signals/mahalanobis_turbulence.py:99` + `complexity_gap_regime.py:92` — `_cached` argument-order bug.**
    `_cached(key, ttl, func, *args)` called as `_cached(key, func, ttl=...)`.
    TypeError on every call. Disabled today; crashes ticker cycle on re-enable. Subsystem 6.

20. **`portfolio/signals/copper_gold_ratio.py:248-265` — sub_signals dict recorded PRE-inversion while action is POST-inversion.**
    For XAU/XAG, recorded sub_signals contradict recorded action. Per-sub-signal
    accuracy tracker measures opposite of what voted. Subsystem 6.

### Data-external

21. **`portfolio/data_collector.py:288-312` — `_fetch_one_timeframe` returns `{"error": str(e)}` as success.**
    Signal engine iterates the error dict and silently skips. Subsystem 7.

22. **`portfolio/data_collector.py:280-294` — race between dispatcher market-state check and inner `_fetch_klines`.**
    The yfinance not-thread-safe lock can be bypassed mid-cycle; re-introduces
    the bug the existing lock-comment defends against. Subsystem 7.

23. **`portfolio/crypto_precompute.py:159-232` + `mstr_precompute.py:140-200` + `data/crypto_data.py:73-85` — bypass rate-limiters/circuit-breakers, hardcoded stale balance-sheet constants, F&G IndexError swallowed.**
    Triple-pattern across precompute modules: unauthenticated `requests.get`,
    `yf.Ticker(...).history` without yfinance_lock, missing `get_fear_greed` defense
    that `portfolio/fear_greed.py` already has (with explicit historical comment).
    NAV-premium math drives Layer 2 prompts with stale BTC holdings (471,107 vs
    499,096 across two files). Subsystem 7.

24. **`portfolio/microstructure_state.py:208` — `persist_state` iterates `_snapshot_buffers` without `_buffer_lock`.**
    Metals 10s fast-tick mutates dict concurrently → RuntimeError → silently
    disables persistence → orderbook_flow signal HOLD-forever via 2-min stale gate. Subsystem 7.

25. **`portfolio/onchain_data.py:86-91` — `_load_config_token` returns None on config error; silent fallback to unauthenticated endpoint.**
    BGeometrics 15-req/day budget burns silently. Subsystem 7.

### Stale journal tail / mis-direction

26. **`portfolio/agent_invocation.py:1338-1342` — fishing_context derived from possibly-stale journal tail.**
    Race between `journal_written` check and `last_jsonl_entry` read; can read
    metals_loop / autonomous append as if it were the agent's entry. Feeds
    grid_fisher direction_bias with wrong direction. Subsystem 2.

---

## P1 — high-confidence bugs (should fix)

### Subsystem 1: signals-core
- `forecast_signal.py:97` bare `except (ImportError, Exception)` silently downgrades Chronos-2 → v1
- `forecast_signal.py:349-352` `_load_candles` returns None → DEBUG log, no health alert
- `signal_db.py:288,319,347,387` `change_pct == 0` fails open for both BUY and SELL
- `signal_engine.py:3069-3091` direct `ind[...]` access without `.get()` → KeyError → silent ticker skip
- `accuracy_stats.py:225` `null_change_pct_skipped` counter never fires (gate wrong-direction)
- `outcome_tracker.py:262-276` yfinance call without per-call timeout
- `ic_computation.py:127-128` `ic_buy`/`ic_sell` are mean returns, not Spearman correlations; weighting math wrong
- `signal_engine.py:954-956` MIN_VOTERS_METALS = 2 violates `MIN_VOTERS = 3 for all asset classes` rule
- `llm_calibration.py:117` + `llm_outcome_backfill.py:121,176,275` non-atomic `read_text().splitlines()` on JSONL
- `outcome_tracker.py:273` UTC date vs yfinance NY exchange date (off-by-one for MSTR)
- `meta_learner.py:42` + `ml_signal.py:17-27` unlocked `_model_cache` racing under ThreadPool
- `ml_signal.py:121, 152-155` in-progress 1h candle used for inference (look-ahead)

### Subsystem 2: orchestration
- `agent_invocation.py:597-619` auth-error cooldown fails open on load_jsonl raise
- `analyze.py:282-289` direct `subprocess.run(["claude", "-p", ...])` bypasses claude_gate
- `main.py:608-661` ticker pool zombie thread accumulation (`shutdown(wait=False)`)
- `trigger.py:165-199` corruption-masking startup grace
- `agent_invocation.py:946-953` text-mode log_fh with CRLF translation breaks byte offset
- `main.py:541` `ind['rsi'], ind['macd_hist']` direct subscript in log format
- `journal.py:23-40` `load_recent` raw `open()` + `json.loads` per line
- `loop_contract.py:1004` `_JOURNAL_UNIQUENESS_WINDOW_S = 600` < T3 timeout 900s
- `agent_invocation.py:1411-1423` `_agent_timeout` not cleared in cleanup
- `agent_invocation.py:562` `_agent_tier=None` produces literal `"layer2_tNone_timeout"`

### Subsystem 3: portfolio-risk
- `equity_curve.py:467-505` profit_factor uses pnl_sek (net) but wins/losses use pnl_pct (gross)
- `equity_curve.py:314-426` FIFO matcher pre-queues buys; sell-at-t=5 matches buy-at-t=10 if list-ordered
- `kelly_sizing.py:84-100` & `kelly_metals.py` Kelly fitted on samples of 2 and 30 — too noisy
- `risk_management.py:273-276` bold-vs-patient detected by substring "bold" in path
- `trade_validation.py:84-92` spread check passes if `bid > ask` (crossed market)
- `risk_management.py:255-270` cash-only fallback when summary empty → spurious massive drawdown
- `monte_carlo.py:88-97` `drift_from_probability` calibrated for T=1/252; used for 3-day horizons
- `trade_guards.py:126-128` `_state_lock` released between read and write
- `risk_management.py:88-110` `_streaming_max` stale offset never reset on consecutive failures
- `kelly_metals.py:198-205` source attribution wrong when defaults used

### Subsystem 4: metals-core
- `metals_avanza_helpers.py:330-408` `place_stop_loss` returns `(False, "")`; grid_fisher writes None, no retry
- `grid_fisher.py:1217-1256` cancel-stop rollback gap on `_safe_session_call` swallow
- `exit_optimizer.py:54` `usdsek=10.85` silent fallback
- `metals_swing_config.py:323` + `crypto/oil_swing_config` `EOD_EXIT_MINUTES_BEFORE=0` regression
- `fish_engine.py:213-232` + `metals_swing_trader.py:2796-2797` 21:55 CET hardcoded; DST gap unhandled
- `oil_grid_signal.py:51-63` RSI uses pure EWM not Wilder's
- `metals_risk.py:45` `MAX_TRADES_PER_SESSION` comment says 17:25 but warrants are 21:55
- `iskbets.py:282-301` substring match on "SKIP" matches `"NO_SKIP"`
- `grid_fisher.py:1545-1552` duplicate-sell guard via stale `eod_sell_order_id` across sessions
- `grid_fisher.py:266-296` DST timing correct but venue close hardcoded; not reading `todayClosingTime`
- `fish_engine.py:647-655` Layer 2 weight-2 duplicate vote bypasses MIN_VOTES

### Subsystem 5: avanza-api
- `avanza_session.py:88-95` expires_at parse failure → "proceeding with caution"
- `avanza_session.py:653-664` `get_open_orders` returns empty list on double-failure → duplicate orders
- `avanza/scanner.py:78` BankID path drops `itype_str` filter
- `avanza/tick_rules.py:124-126` last-tick fallback hides API shape drift
- `avanza/account.py:74-90` per-call rate (no caching at API layer)
- `avanza_session.py:759-762` sell_price=0 → ValueError instead of skip-and-retry
- `avanza/streaming.py:79-85` channel string from raw account IDs, no whitelist
- `avanza_session.py:1064-1070` cancel busy-wait holds `_pw_lock` 3s
- `avanza/account.py:139-140` client-side transaction filter leaks pension into ISK caches
- `avanza_session.py:752` `valid_until` 8-day default; Friday stops expire Sunday

### Subsystem 6: signals-modules
- `hurst_regime.py:283-302` same vote double-counted in majority
- `vwap_zscore_mr.py:124-125` bare except → HOLD with NO logging
- `futures_flow.py:118, 135, 162, 287` direct `[-1]["longShortRatio"]` access without KeyError handler
- `williams_vix_fix.py` 3/4 sub-indicators BUY-only structural long-bias
- `cot_positioning.py:213-217` `commercial_change` is actually `-noncomm_net_change`
- `realized_skewness.py` rolling window = data length → z-scores inflated
- `gold_overnight_bias.py:35-41, 161` DST blindness on London fix times
- `gold_overnight_bias.py:118-140` `_fix_proximity_vote` BUY-only
- `metals_cross_asset.py:220, 224` ≥3-of-4 gate but vote-HOLD on degraded source

### Subsystem 7: data-external
- `microstructure_state.py:227` 2-min stale check on cross-process wall clock
- `fx_rates.py:46-53` out-of-bounds FX returns None → silent 10.50 fallback
- `sentiment.py:288-296` subprocess fallback 120s timeout pins workers
- `earnings_calendar.py:48-52` AV daily-budget bypassed (admitted in comment)
- `oil_precompute.py:503-577` + `metals_precompute.py:404-458` CFTC SoQL `$where` interpolation
- `news_keywords.py:80-83` `\b` + multi-word phrase regex edge case
- `sentiment_shadow_backfill.py:211-213` legacy log naive timestamps assumed UTC
- `onchain_data.py:73-74` 12h TTL but `_coerce_epoch` returns 0.0 forces refetch on restart

### Subsystem 8: infrastructure
- `log_rotation.py:432-440` rotate_text copy+truncate race loses appends
- `file_utils.py:240-258` Win32 `msvcrt.locking(LK_LOCK)` 10s timeout, not pure-blocking
- `http_retry.py:44-49` Retry-After header ignored; only Telegram JSON shape read
- `http_retry.py:34` non-GET/POST drops `json_body`
- `feature_normalizer.py:35-40` `_ensure_buffer` race drops first samples
- `llama_server.py:419` Plex-active swap abort fires after `_stop_server` kills model
- `llama_server.py:179-180` `pid` unbound on exception path → NameError
- `subprocess_utils.py:214-225` PowerShell single-quote backtick escapes inert
- `claude_gate.py:597-607` auth detection bypassed on Popen-time exception
- `api_utils.py:30-35` raw `open()` + `json.load` violates atomic-I/O rule

---

## P2 — concerns / smells (worth addressing)

P2 list across all 8 cross-critique docs runs to ~70 items. Highlights:

- **Confidence calibration drift across signal modules** — modules cap at varying
  values (0.6 in calendar_seasonal, 0.7 in copper_gold_ratio, 1.0 elsewhere)
  but the engine treats them symmetrically.
- **Structural long-bias in several signal modules**:
  `williams_vix_fix`, `calendar_seasonal` (6/8 sub-signals BUY-only),
  `gold_overnight_bias` proximity vote BUY-only. Each individually defensible
  but combine to a long-only ensemble bias.
- **Caches without eviction or locks**: `copper_gold_ratio._CACHE` (unlocked),
  `portfolio_mgr._FILE_LOCKS` (no eviction), `meta_learner._model_cache`
  (unlocked), `ml_signal._pred_cache` (unlocked).
- **DST handling inconsistency**: documented in `gold_overnight_bias`,
  `intraday_seasonality`, `orb_predictor`; probably present in others.
- **Equity-curve annualization extrapolates from <60d of data** (`equity_curve.py:184-189`)
  → dashboard shows +1190% on 7d of +5%.
- **Defensive defaults that hide data drift**: `seasonality.py:62-66` (count=0
  treated equal to count=20), `_KEYWORD_PATTERNS` (substring matching),
  `macro_context.synth DXY` returns plausible-but-fake number.

Full P2 inventory: see `cross-1-signals-core.md` through `cross-8-infrastructure.md`.

---

## Recommended fix sequencing

### Wave 1 — Stop bleeding (one-day fix sprint)
- P0 #1, #2, #3, #4, #5 (LONG-only stops, Position direction, Kelly leverage saturation, max_cash_pct leverage-blind, GRID_STOP_PCT) — every metals trade today is exposed to one or more.
- P0 #6 (iskbets fail-open APPROVE) — silent autonomous trading on empty LLM output.
- P0 #11 (telegram_poller symlink severing) — one stray `/mode` command embeds API keys in repo.
- P0 #14, #15 (Avanza unified place_order whitelist + place_stop_loss MAX_ORDER) — one migration silently uncaps every order safety.

### Wave 2 — Stop hiding (data-integrity sprint)
- P0 #7, #8 (`_agent_proc=None` race + multi_agent gate bypass) — re-arms the 3-week silent-auth-outage failure mode.
- P0 #9, #10 (forecast accuracy truncation + schema mismatch) — model-degradation invisible right now.
- P0 #12, #13 (sidecar lock alias + claude_gate return-type) — torn writes and unpack failures latent across the loop.
- P0 #21, #22, #23, #24, #25 (data-external silent-failure cluster) — every signal's input quality is currently uncheckable.

### Wave 3 — Stop drifting (math + audit sprint)
- P0 #16, #17, #18 (get_buying_power zero + EOD 1-öre floor + fx_rate VaR bypass) — risk-reporting accuracy.
- P0 #19, #20 (`_cached` argument-order + copper_gold_ratio inversion) — re-enablement gate for disabled signals.
- P0 #26 (fishing_context stale journal) — wrong direction_bias on grid_fisher.
- All P1s in subsystems 1-8 (~80 items).

### Wave 4 — Backlog
- P2 inventory ~70 items.
- Tests for every P0 fix.

---

## Coverage notes

**Reviewer agreement on P0**: 17 of 26 P0 findings were independently rediscovered
by both reviewers. The other 9 are single-reviewer P0s that survived
cross-critique scrutiny (the other reviewer didn't dispute, just didn't open
that file).

**Reviewer blind spots**:
- Claude (main thread) didn't open `trade_validation.py`, `crypto_precompute.py`,
  `mstr_precompute.py`, `data/crypto_data.py`, several Codex-found files.
- Codex-substitute (Claude subagent) didn't surface schema-vs-writer mismatches
  in forecast pipeline, didn't audit ic_computation math, didn't notice
  `MIN_VOTERS_METALS=2` rule drift.
- Both missed: ADX cache key from `b66375cb`, `signal_decay_alert.py` (zero
  findings), `portfolio_mgr.py` SHARES vs CASH atomicity, Alpaca pagination,
  shared HTTP session lifecycle.

**Codex CLI quota note**: at session start the real Codex was rate-limited
until 17:27 PM. Substitute reviews from a Claude subagent with isolated context
were used as second-opinion. Real Codex runs were re-triggered on quota
recovery and stored to `codex-raw/`. Where real Codex output added findings
beyond the substitute, the synthesis above was updated accordingly.

## Real Codex findings (delta over substitute review)

Real Codex review on **subsystem 1 / signals-core** completed and surfaced
three findings that neither Claude review nor the substitute caught:

- **`portfolio/signal_engine.py:3655-3656` — local LLM actions logged AFTER gate replaces them with HOLD.** `_gate_local_model_vote()` may downgrade qwen3/ministral BUY/SELL to HOLD, but the probability log records the gated HOLD with the model's *original confidence*. Calibration and Brier analysis poisoned. **Real P2 — should be promoted to P1 if calibration is used in production sizing.** Add to subsystem 1 P1 list.
- **`portfolio/accuracy_stats.py:1923-1926` — `write_ticker_accuracy_cache` uses single `time` key for all horizons.** A fresh `1d` write refreshes the TTL for stale `3h`/`4h` blocks. `signal_engine` consumes stale per-ticker accuracy for horizon-specific gates. **Real P2, narrow but real.**
- **`portfolio/accuracy_stats.py:1388-1390` — same single-`time`-key bug in `write_regime_accuracy_cache`.** Stale regime accuracy → wrong signal weights/gates. **Real P2, same root cause.**

The forecast truncation P0 and schema-mismatch P0 (subsystem 1 above) are
independently confirmed by real Codex.

Real Codex review on **subsystem 2 / orchestration** also completed and
surfaced six findings neither reviewer caught:

- **`portfolio/agent_invocation.py:274` — `_extract_ticker` regex only matches `flipped|crossed|broke`, but `check_triggers` emits `"{ticker} consensus ..."` and `"{ticker} moved ..."`.** Stock-trigger reasons fall through to default `XAG-USD` ticker → wrong trade-guard, wrong specialist prompts, wrong decision feedback. **Real P1 — silent instrument misrouting.**
- **`portfolio/agent_invocation.py:849` — `cleanup_reports()` unused; stale specialist reports leak across runs.** Synthesis agent reads previous run's report for a different ticker as if current. **Real P1.**
- **`portfolio/market_health.py:244-246` — `ftd_day_offset` persisted as array index; with fixed 90-day fetch, never exceeds failure window, so `FTD_CONFIRMED` never promotes to `confirmed_uptrend` after 10 days.** Market health permanently understated. **Real P2.**
- **`portfolio/reflection.py:80` — `total_pnl_pct = (cash - initial) / initial` ignores holdings value.** After a BUY, cash drops but holdings carry value; reflection reports false large loss → Layer 2 generates false "down X% — reduce size" insights whenever positions open. **Real P1, poisons Layer 2 prompts.**
- **`portfolio/autonomous.py:710` — Mode B Telegram appends raw probability with `%` sign: `0.62` displayed as `0.62%` instead of `62%`.** Misleading operator. **Real P3 but operator-visible.**
- **`portfolio/trigger.py:271` — `flip_cooldowns` use wall-clock timestamps; backward NTP jump makes `_flip_now_ts - last_flip_ts` negative → suppresses every sustained flip until clock catches up.** Note: commit `7a303961` ("clock skew guard for trigger") on main may already address; review branch baseline 2026-05-14 predates that fix. Re-audit on current main. **P3 if fixed, P1 if not.**

Real Codex quota exhausted before reviewing subsystems 3-8 (next reset
22:27 local). The codex-substitute reviews (Claude subagent with isolated
context) covered those subsystems; see `codex-N-{subsystem}.md` and
`cross-N-{subsystem}.md`. Re-run on next session: see
`codex-real-findings.md` for instructions.

---

## Out of scope

- Implementation. This is review-only. Each Wave is a follow-up session.
- Test runs. No code changed; nothing to regress.
- Live-trading impact assessment. Some P0s are gated by `DRY_RUN=True` or
  `DISABLED_SIGNALS` lists in current config. Sequencing above considers
  "exposed surface area when re-enabled".
