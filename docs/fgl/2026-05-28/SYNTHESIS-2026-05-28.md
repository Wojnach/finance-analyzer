# FGL Adversarial Review — Synthesis (2026-05-28)

**Method.** Codebase partitioned into 8 subsystems. Each subsystem checked out as a
single reviewable diff against an empty git baseline (`fgl-baseline`) in its own
worktree under `Q:\fa-fgl\<sub>`. One fresh review subagent per subsystem ran in
parallel (5× `pr-review-toolkit:code-reviewer` on the large/critical ones, 3×
`caveman:cavecrew-reviewer` on the tighter ones). The orchestrating agent ran an
**independent** deep pass (`_independent.md`) on the highest-stakes hot paths and
**cross-critiqued** every subagent P0, verifying or downgrading by reading the code.

Scope: ~63K LOC of trading-critical core (8 subsystems × ~10–16 files). Not every one
of the 335 project `.py` files was reviewed — the partition targeted the money-path and
reliability-path code. Per-subsystem detail in the sibling files in this directory.

**Severity key.** P0 = real-money loss / loop crash / data corruption / silent-failure
that exits 0. P1 = wrong behavior under realistic conditions. P2 = latent/fragile.
P3 = minor. The agents' raw severities are kept, with a **Recalibrated** column where
the orchestrator verified the code (some agents — notably the caveman infra/data-external
runs — ran hot on severity).

**Totals (recalibrated by orchestrator; raw agent counts in parentheses).**

| Subsystem | P0 | P1 | P2 | P3 | Reviewer |
|---|---|---|---|---|---|
| orchestration | 2 | 3 | 5 | 3 | pr-review |
| portfolio-risk | 2 | 9 | 8 | 6 | pr-review |
| avanza-api | 3 (5) | 3 | 3 (1) | — | caveman |
| signals-core | 0 | 4 | 9 | 4 | pr-review |
| signals-modules | 0 | 3 | 5 | 4 | pr-review |
| data-external | 1 (4) | 11 (9) | 3 | 2 | caveman |
| infrastructure | 0–3 (6) | 4 | 2 | — | caveman |
| metals-core | 1 gated | 4 | 5 | 3 | pr-review |
| independent | — | 4 | — | 1 | orchestrator |

Recalibration deltas: 2 infra P0s disproven (`file_utils.py:240` false positive; `health.py:156`
→P2), avanza/data-external/infra severities cooled where "silent-degradation" was tagged P0.
The genuinely sharpest, money/reliability-critical items are the §1 incidents + §4 list.

---

## 1. Headline: the two live, currently-firing incidents are now root-caused

These were visible in `data/critical_errors.jsonl` at session start (37 unresolved
entries over 7 days). The review pinned both to specific code.

### A. `avanza_account_mismatch: Session expired` — daily since 2026-05-23
**Root cause (CONFIRMED by direct read):** `portfolio/avanza_session.py:139`
`_get_playwright_context()` early-returns the cached `_pw_context` whenever it is
non-None and only calls `load_session()` on first creation. A long-running loop process
holds the Playwright context across the ~24h BankID session expiry and **never
re-validates**. `is_session_expiring_soon()` exists (`avanza_session.py:123`) but the
reuse path never consults it. Compounded by `avanza_account_check.py:225` treating
expiry as a transient outage (not a fatal that forces re-auth) and `avanza_session.py:426`
returning `None` on API error so callers can't distinguish "empty balance" from "401".
**Fix:** on every `_get_playwright_context` reuse, if `is_session_expiring_soon()` →
`close_playwright()` and rebuild; classify `AvanzaSessionError` as fatal → trigger the
documented re-login (`scripts/avanza_login.py`) instead of looping on a dead session.

### B. `contract_violation: Layer 2 ... no journal entry ... failing silently` — many/day
**Root cause (CONFIRMED, orchestration agent + independent pass):** this is **mostly a
false positive**, not a real silent failure. `loop_contract.py:410` compares
`journal_ts >= last_trigger` with **zero tolerance**, but the journal `ts` is stamped at
cycle-start (whole-second `...Z`) while `last_trigger_time` is written later at
microsecond precision. 233 violations in 7 days, **all** with
`last_invocation_status=success, journal_written=True`. The check fires on perfectly
successful, journaled, telegrammed runs — spamming Telegram, polluting
`critical_errors.jsonl`, arming the fix-agent dispatcher, and training operators to
ignore the one check meant to catch a real outage.
**Secondary REAL gap (independent IND-1, extended by orchestration agent):** the
timeout-kill path (`agent_invocation.py:1444-1457` / `_kill_overrun_agent:655-737`)
writes **no** journal stub (unlike the `incomplete` path at 1583-1598), and `timeout`
/ `failed` are **absent from the contract's `_KNOWN_FAILURE_STATUSES`** — so a genuine
hang/timeout WOULD be an undetected silent gap hidden inside the false-positive noise.
**Fix:** add a skew tolerance window (or trigger-reason match) to `loop_contract:410`;
write a `status="timeout"` journal stub on the kill path; register `timeout`/`failed`
as known failure statuses.

---

## 2. Cross-cutting themes (span ≥3 subsystems — fix once, benefit everywhere)

### Theme 1 — "empty / None / stale silently read as valid / live / flat" (MOST PERVASIVE)
A failed fetch, expired session, or locked file returns an empty/default value a
downstream consumer treats as authoritative. Confirmed in:
- `avanza_session.py:426` API error → None (empty-balance vs 401 indistinguishable). **P0**
- `avanza_account_check.py:225` expiry treated as transient. **P0**
- `file_utils.py:86-90` `load_json` OSError → default `{}` masks a locked state file as empty (IND-3). **P2**
- `sentiment.py:854` empty headlines → "unknown" 0.0-conf vote, doesn't disable voter, dilutes consensus. **P1** (agent said P0)
- `metals_cross_assets.py:56` `fetch_klines` exception → empty DataFrame, no staleness flag. **P1** (agent said P0)
- `price_source.py:214` silent yfinance fallback → ~10-min-stale price as live. **P1**
- `fx_rates.py:44` out-of-band rate not cached → P&L uses fallback 10.50, 10-15% off. **P1**
**Systemic fix:** an explicit `UNKNOWN`/`STALE` sentinel distinct from empty/zero/flat;
consumers (voters, sizing, valuation, position checks) fail-closed (HOLD/skip/halt) on
UNKNOWN rather than trading a phantom zero.

### Theme 2 — dead safety/logic code on a reliability-critical path
Controls that look protective but have no production caller, so nobody is actually
protected/served:
- `trade_validation.py` + `trade_risk_classifier.py` — cash/size/spread/deviation caps + HIGH-risk scoring: **tests-only, no execution-path caller** (portfolio-risk). **P1**
- `portfolio_validator.py` — negative-cash / holdings-mismatch / fee-mismatch checks wired only to the read-only dashboard, never to `save_state`/`update_state`; corrupt state persists silently (portfolio-risk). **P1**
- `signal_weights.py` + `signal_weight_optimizer.py` — confirmed dead (MWU removed, walk-forward results never consumed) yet living on the signal path (signals-core). **P2**
**Fix:** wire them in at the execution/save boundary, or delete them so the map matches reality.

### Theme 3 — concurrency: write-atomic but not read-modify-write-serialized, esp. cross-process
- `portfolio_mgr.py:29,136-159` `update_state` uses a `threading.Lock` — in-process only; portfolio_state.json is mutated by 3 OS processes (L1 loop, L2 `claude -p` subprocess, dashboard) → cross-process lost updates (IND-2). `save_state`/`load_state` bypass even that lock.
- `file_utils.py:53-71` `atomic_write_json` has no JSON-state equivalent of `jsonl_sidecar_lock`.
- `alpha_vantage.py:280` budget counter incremented inside lock, reset outside → quota race (data-external). **P1**
- (metals-core: state RMW between loop thread + 10s fast-tick thread — pending.)
**Fix:** a cross-process sidecar lock for JSON-state RMW; single-writer discipline for portfolio state.

### Theme 4 — severity inflation in two agents (calibration note, not a code bug)
The caveman infra + data-external runs labeled silent-degradation and crash-window edges
as P0. Two infra P0s were disproven on read (see §4). Most data-external "P0 loop-killer"
items are silent-dilution = P1. Synthesis severities are recalibrated accordingly.

---

## 3. Consolidated findings by subsystem (de-duped, recalibrated)

### orchestration — 2 P0 / 3 P1 / 5 P2 / 3 P3
- **P0** `loop_contract.py:410` zero-skew journal/trigger comparison → 233 false CRITICALs (incident B).
- **P0** `agent_invocation.py:1444-1457` timeout path writes no journal stub; `timeout`/`failed` not in `_KNOWN_FAILURE_STATUSES` → real hang would be silent.
- **P1** `main.py:855` `update_tier_state` runs before `invoke_agent` → resets T3 heartbeat even when the invocation is skipped/blocked/busy.
- **P1** `trigger.py:296-343` consensus/flip baseline persisted before invocation → a crossing that hits `skipped_busy`/a gate is consumed and never re-fires (lost trigger).
- **P1** `agent_invocation.py:1095-1100` L2 cmd omits `--output-format json` → max-turns-exhausted partial-journal run misclassified `success`.
- Verified-correct: CLAUDECODE/CLAUDE_CODE_ENTRYPOINT scrub before Popen; no `--bare`; auth scan on both completion+timeout paths; crash-backoff floor; DST/holiday math.

### portfolio-risk — 2 P0 / 9 P1 / 8 P2 / 6 P3
- **P0** `kelly_sizing.py:91-104` one global avg buy price scored against every SELL (no FIFO, look-ahead) → wrong win-rate/payoff into Kelly → mis-sized recs. Reuse `equity_curve._pair_round_trips`.
- **P0** `risk_management.py:374,465,897` ATR stop `entry*(1-2*atr/100)` has no distance floor / no knockout-barrier awareness → on low-ATR silver certs the stop lands ~2% away, violating the "never stop within 3% of bid" rule (Mar-3 instant-fill class). [Matches `.claude/rules/metals-avanza.md`: 5x certs need -15%+ stops.]
- **P1** `monte_carlo.py:359-362` fabricated directional drift injected into VaR/exit GBM → understated stop-hit prob + downside VaR. Run VaR zero-drift.
- **P1** dead controls `trade_validation.py` / `trade_risk_classifier.py` / `portfolio_validator.py` (Theme 2).
- **P1** `monte_carlo_risk.py:408` raw `agent_summary.get("fx_rate")` (pre-P1-15 pattern) → SEK VaR ~10x understated if fx missing/1.0.
- **P1** `equity_curve.py:188,557` annualizes over tiny `years` with no min-window guard → Calmar/annualized blow-up on a few days of data.
- **P1** `kelly_sizing.py:296-310` returns non-zero Kelly from a fabricated ATR-edge with zero realized trades.
- Verified-correct: 50% drawdown block IS wired in `agent_invocation.py`; `check_drawdown` has NaN/Inf fail-safe; `_pair_round_trips` correct FIFO; `_resolve_fx_rate` rejects 1.0.
- **Cross-critique:** agent's "`update_state` concurrency-safe" is true only in-process; cross-process lost-update stands (Theme 3 / IND-2).

### avanza-api — raw 5 P0 / 3 P1 / 1 P2 → recalibrated 3 P0 / 3 P1 / 3 P2
- **P0** `avanza_session.py:139` session not re-verified on context reuse (incident A). CONFIRMED.
- **P0** `avanza_session.py:620` order-placement failure swallowed, result dict misread as success → blind to rejected orders.
- **P0** `place_stoploss_once.py:171` not idempotent → re-run/restart creates duplicate stops (overfill risk; violates "check existing orders before placing").
- **P1** `avanza_session.py:426` API error → None, empty-vs-401 indistinguishable (Theme 1). [agent P0 → keep visibility but it's the same root as :139.]
- **P1** `avanza_account_check.py:225` expiry treated as transient (incident A contributor). [agent P0]
- **P1** `avanza_session.py:256` non-JSON 200 (HTML) raises uncaught, browser not torn down → context leak.
- **P2** `avanza_order_lock.py:85` stale lock from a crash hangs all orders → needs mtime staleness break.
- **P2/P3** (downgraded) TOTP in config.json = the documented external-secrets design, not a leak; confirm-token case-sensitivity is a nit.

### signals-core — 0 P0 / 4 P1 / 9 P2 / 4 P3
- **P1** `signal_engine.py:4119` 3d/5d/10d consensus horizons collapse onto 1d accuracy stats for gate/weights/IC/utility (TODO P1.12) → wrong-horizon edge on every long-horizon vote.
- **P1** `forecast_accuracy.py:159` forecast correctness uses `change>0` with no neutral-move filter (unlike every other accuracy path) → biased forecast numbers, apples-to-oranges degradation diff.
- **P1** `ticker_accuracy.py:131` Mode-B/Kelly `direction_probability` published off 5-sample recent windows → real-money sizing on noise. [Matches user rule: "small samples lie."]
- **P1** `ic_computation.py:130` ICIR std over overlapping cross-ticker windows → inflated ICIR → admits unstable IC weight boosts.
- **P2** `accuracy_stats.py:1017` + `outcome_tracker.py:567` utility-cache invalidation never clears `accuracy_cache.json` → up-to-1h-stale accuracy keeps a freshly-degraded signal voting.
- Verified-correct (my prime suspects, all clean): no look-ahead in outcome backfill; empty-recent-window defended (all-time fallback <30 samples, neutral 0.5); accuracy gate only force-HOLDs (never inverts); `accuracy_degradation` dedups Telegram re-fires (24h cooldown + identity hash).
- **Observational follow-up:** `critical_errors.jsonl` still shows ~daily `accuracy_degradation` entries in same-second pairs — reconcile whether the alert-set key changes daily (bypassing the 24h identity dedup) or there's a double-emit. Low severity.

### signals-modules — 0 P0 / 3 P1 / 5 P2 / 4 P3
- **P1** `sentiment_extremity_gate.py:144` documented crypto-only but has NO ticker guard → runs on XAU/XAG/MSTR using the crypto alt.me F&G index (wrong sentiment series).
- **P1** `cryptotrader_lm.py:139` (the OVERDUE-shadow module) degrades to HOLD on failure (good) but `query_llama_server` blocks on a 300s file lock + 240s HTTP timeout → a cold model swap can occupy a worker ~240s and blow the T1 180s ticker-pool budget (BUG-178 hang class). Add a short signal-local timeout; confirm it's in the fail-closed shadow-throttle set.
- **P1** `btc_gold_correlation_regime.py:93` joins per-ticker intraday df against counterpart klines fetched at `interval="1d"` → timestamps misalign, `dropna()` empties the frame → signal silently never fires on non-daily timeframes.
- **P2** `crypto_evrp.py:204` mislabeled (ranks DVOL percentile, not eVRP; `rv_hist` dead code).
- **P2** `statistical_jump_regime.py:158` registered with no `max_confidence` cap (peers use 0.7) → single voter to ~1.0.
- Verified-correct: no crash-the-voter paths (canonical HOLD on bad input + engine catches raises → force-HOLD); no look-ahead; no inverted directions (incl. deliberate safe-haven inversions in credit_spread / metals_cross_asset / btc_gold_correlation match docstrings).

### data-external — raw 4 P0 / 9 P1 → recalibrated 1 P0 / 11 P1 / ...
- **P0→P1** `sentiment.py:854` empty headlines → "unknown" 0.0-conf vote (doesn't disable voter; dilutes consensus) — Theme 1.
- **P0→P1** `metals_cross_assets.py:56` `fetch_klines` exception swallowed → empty DataFrame, no staleness flag — Theme 1.
- **P0 (keep)** `alpha_vantage.py:280` budget counter incremented-in-lock/reset-outside → ThreadPool race blows the 25/day quota → fetches silently fail for the rest of the day.
- **P0→P1** `onchain_data.py:29` `_coerce_epoch` returns 0.0 on ISO parse failure → forces an API call every restart (verify the "crash" path; likelier a churn/quota bug).
- **P1** `fx_rates.py:44` out-of-band rate not cached → P&L 10-15% wrong for 2h+ (Theme 1).
- **P1** `crypto_macro_data.py:224` `_load_ratio_history` unbounded JSONL load → MemoryError on >30M file.
- **P1** `price_source.py:214` silent yfinance fallback → ~10-min-stale as live (Theme 1).
- **P1** `crypto_data.py:63` `_WARNED` guard never resets → after first error, all later errors silent until restart.

### infrastructure — raw 6 P0 / 4 P1 / 2 P2 → recalibrated 0–3 P0 (see cross-critique)
- **DISPUTED (FALSE POSITIVE)** `file_utils.py:240` "sidecar-lock creation race → corruption": the seed is idempotent in outcome — any ≥1-byte lock file is lockable; content never matters. No corruption. **Drop.**
- **DOWNGRADE P0→P2** `health.py:156` "staleness falsely reports not-stale": a naive `last_heartbeat` makes the *uncaught* subtraction at `health.py:165` RAISE `TypeError` (try/except wraps only `fromisoformat`) — a crash, not a silent false-negative; and current `heartbeat()` writes tz-aware ISO so it's latent. Still worth hardening (normalize tz / wrap subtraction).
- **P1 (plausible, verify)** `process_lock.py:103` truncate-then-write not atomic → crash leaves empty lock file, loses PID for stale detection (BUG-182 needs PID). Wrap in tempfile+os.replace.
- **P1 (plausible, verify)** `log_rotation.py:349` SQLite `-wal`/`-shm` not checkpointed before rotation (signal_log.db dual-write) → reopened-DB corruption risk. `PRAGMA wal_checkpoint(TRUNCATE)` first.
- **P1 (plausible, verify)** `log_rotation.py:510` text-file rotation has no sidecar lock → writer appending mid-rotation loses data.
- **P1** `subprocess_utils.py:214` PowerShell filter interpolation in `kill_orphaned_by_cmdline` — low exploitability (own-process cmdline) → really P2; still quote the filter.
- **P2** `journal.py:28` `load_recent` reads+parses whole file → use `load_jsonl_tail`.
- Verified-correct: no dashboard auth bypass — `cf_access.py` does real RS256 JWT verification (aud/exp/iat + email-claim match; 2026-05-13 header-spoof P0 fixed); cookie/query/bearer use `hmac.compare_digest`.

### metals-core — 1 P0 (flag-gated) / 4 P1 / 5 P2 / 3 P3
- **P0 (latent, flag-gated)** `metals_swing_trader.py:2974-2977` SHORT warrant exit-trigger P&L is **inverted**. A held BEAR cert is bought/sold like any long, so its true P&L is `(bid/entry-1)` (what `_execute_sell:3169` computes), but `_check_exits` flips the sign for SHORT → once `SHORT_ENABLED=True`, winning BEAR positions trip HARD_STOP and never reach TAKE_PROFIT (force-sell winners, ride losers). Gated off today but the code documents enabling. **Fix per GUIDELINES: keep disabled at the gate with an explicit TODO until exits are made direction-consistent.**
- **P1** `metals_loop.py:4474` direction-blind barrier distance `(und-barrier)/und`. The live catalog has 47 SHORT warrants with barrier ABOVE underlying → negative distances → `metals_execution_engine._summary_filters` silently filters out every SHORT-warrant BUY rec. Direction-aware formula already exists at `metals_swing_trader.py:2490-2498` — reuse it.
- **P1** `metals_loop.py main()` a fatal crash exits **code 0** (`main()` returns None → `sys.exit(None)`) — the exact exit-0-on-failure class CLAUDE.md's STARTUP CHECK warns about; a supervisor can't tell crash from clean stop. Return a non-zero code on fatal.
- **P1** `metals_loop.py:7166-7848` no per-cycle isolation: the whole `while True` body is one try with its only `except` OUTSIDE the loop → an uncaught raise in any unwrapped step kills the loop instead of skipping one cycle (violates the "loop runs 100%" priority). Wrap the cycle body.
- **P1** `grid_fisher.py:1871-1916` (`eod_market_flat`) deletes the broker stop + cancels sells BEFORE placing the replacement sell; on sell failure the position is left **naked** (no stop, no sell) until a later tick — naked overnight if the loop then dies. Unlike `rotate_on_buy_fill`, it neither re-arms nor logs a critical naked-position entry. Re-arm a stop / log critical on failure.
- Verified-correct: every stop-loss path uses `/_api/trading/stoploss/new` (only emergency/EOD market-equivalent sells use the regular order API — the documented pattern); stop placement direction is correct everywhere (certs always held long → SELL stop below bid right for BULL and BEAR); grid-fisher exit/stop/P&L and the execution engine are otherwise fully direction-aware.

### independent pass (orchestrator) — see `_independent.md`
IND-1 timeout-stub gap (folded into orchestration §B). IND-2 cross-process lost update
(Theme 3). IND-3 load_json-OSError-as-empty (Theme 1). IND-4 unset dashboard_token =
open incl. POST (P3). Plus the cross-critique log that recalibrated the infra/avanza P0s.

---

## 4. Recommended fix order (impact × certainty, money/reliability first)

1. **Incident A — Avanza session reuse** (`avanza_session.py:139` + `account_check.py:225`): the live trading-halt. Re-validate on reuse, classify expiry as fatal → re-login.
2. **Incident B — contract false-positive + real timeout gap** (`loop_contract.py:410`, `agent_invocation.py:1444-1457`): stop the CRITICAL spam AND close the genuine silent-timeout hole; both are small, surgical.
3. **Stop-loss safety** (`risk_management.py` ATR floor + barrier awareness): money; aligns with an existing documented rule already burned once (Mar 3).
4. **Kelly FIFO / look-ahead** (`kelly_sizing.py:91-104`): money; wrong sizing on every rec.
5. **Theme 1 sentinel + fail-closed consumers**: the highest-leverage systemic fix; convert silent-degradation P1s into safe halts.
6. **alpha_vantage quota race** + **fx out-of-band caching**: cheap, stops quota lockout + 10-15% P&L error.
7. **Theme 2 dead controls**: wire in `portfolio_validator` at the save boundary + `trade_validation` at execution, or delete.
8. **Metals reliability**: `metals_loop.py main()` exit-0-on-crash → return non-zero (a supervisor-blindness bug of the exact class CLAUDE.md warns about); wrap the `while True` cycle body so one bad cycle is skipped, not fatal; `grid_fisher.eod_market_flat` re-arm/log-critical on replacement-sell failure (naked-position window). Keep `SHORT_ENABLED=False` with a TODO until `_check_exits` P&L sign is direction-consistent (`metals_swing_trader.py:2974`).
9. **Theme 3 cross-process JSON-state lock**; **signals-core horizon-collapse (P1.12)**; **cryptotrader_lm worker-budget timeout**; infra plausible-P1s (`process_lock` truncate-write, `log_rotation` wal/text) after independent verification.

These are review findings only — no code was changed in this session.
