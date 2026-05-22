# FGL Adversarial Review — finance-analyzer — 2026-05-22

**Type:** Full-codebase adversarial review. Read-only — no production code changed.
**Method:** Codebase partitioned into 8 subsystems. One fresh-context Claude Code review
subagent per subsystem (5× `pr-review-toolkit:code-reviewer`, 3× `caveman:cavecrew-reviewer`),
each reviewing its subsystem as an empty-baseline git diff (`review/sub-<name>` vs
`review/empty`). In parallel, the lead agent ran an independent fresh-eyes pass over the
highest-risk core files. Findings were then cross-critiqued: every subagent finding was
independently confirmed, downgraded, or rejected; corroboration across reviewers raised
confidence.

**Raw inputs:** `docs/reviews/2026-05-22-fgl-raw/subagent-reviews.md` (8 subagent outputs),
`docs/reviews/2026-05-22-fgl-raw/independent-pass.md` (lead independent pass, IND-1…17).

**Scope reviewed:** ~127K LOC — 203 core files across signals-core (25), orchestration (22),
portfolio-risk (22), metals-core (36), avanza-api (19), signals-modules (68), data-external
(32), infrastructure (28 + dashboard).

---

## 1. Executive summary

The system is **heavily defended** — most files show extensive prior adversarial-review
scar tissue (atomic I/O, monotonic clocks, watchdogs, fail-safe gates, auth-error
detection). The dashboard auth path and `process_lock` were independently confirmed
**solid**. No reviewer found a loop-crashing P0.

The findings cluster into a small number of **systemic weaknesses** that recur across
subsystems — those are the real story, not the individual line items:

| Theme | Where it recurs | Why it matters |
|-------|-----------------|----------------|
| **`threading.Lock` used for cross-process state** | `portfolio_mgr`, `warrant_portfolio`, `health`, `avanza_orders` pending file, `loop_contract` | The Layer 2 `claude -p` subprocess + metals loop + dashboard are separate processes. A `threading.Lock` does nothing across processes. Read-modify-write interleaving → lost transactions, dropped pending orders, corrupted state. `atomic_write_json` makes each *write* atomic but not the *RMW*. |
| **Incomplete 252→365 annualization fix** | `metals_risk.py`, `metals_execution_engine.py`, `risk_management.py:462` | Prior batches fixed 252→365 in monte_carlo / exit_optimizer / price_targets but missed three more call sites. Crypto/metals trade 24/7; vol/VaR/stop-hit probabilities are understated ~21%. |
| **`file_utils` JSONL locking gaps** | `atomic_write_jsonl` (no lock), `jsonl_sidecar_lock` (racy creation, can crash), reader visibility | The one primitive ~20 writers depend on has a lock-bypass path and a creation race that can crash every JSONL writer. |
| **Silent failure / stale data served as fresh** | `onchain_data`, `avanza_session.api_post`, `avanza/scanner`, `_kill_overrun_agent` wedge, `telegram_poller` | Functions that fail without surfacing it — the exact class behind the 3-week Mar–Apr Layer 2 auth outage. |
| **Live order placement without idempotency / barrier safety** | `grid_fisher._safe_session_call`, `metals_loop.emergency_sell`, `fin_snipe_manager` stops | Timeouts and missing guards on the paths that place real Avanza orders. |

**Reconciled severity counts** (after cross-critique — see §6 for downgrades):

| | P0 | P1 | P2 | P3 |
|---|----|----|----|----|
| Total reconciled | 1 | 21 | 20 | 14 |

The headline number from the raw subagent outputs was "2 P0 + 9 P0" — the lead pass
**rejected/downgraded 13 of those P0 claims** (see §6). The genuine P0 count is **1**.

---

## 2. Fix first — top 15

Ordered by risk × likelihood. IDs: `M#`=metals-core, `O#`=orchestration, `PR#`=portfolio-risk,
`SC#`=signals-core, `DE#`=data-external, `INF#`=infrastructure, `AV#`=avanza-api,
`IND#`=lead independent pass.

1. **[P0] `grid_fisher.py:1026-1037` — duplicate LIVE order on session-call timeout.**
   `_safe_session_call` returns `None` (treated as "not placed") on its 30s timeout but
   never cancels the in-flight `place_buy_order`/`place_stop_loss` call. The original
   request can still land at Avanza; next tick re-places it → **duplicate real order,
   doubled notional, per-instrument/global cap breach**. Fix: on timeout mark the ob_id
   "uncertain", force an open-order reconcile before any new placement, re-place only if
   reconcile confirms nothing landed.

2. **[P1] Cross-process RMW race cluster** (`IND-2`, `PR` warrant_portfolio, `INF` health,
   `IND-13` avanza_orders, `O` loop_contract). `portfolio_mgr.update_state`,
   `warrant_portfolio.record_warrant_transaction`, `health.heartbeat`,
   `avanza_orders` pending-file, `loop_contract.ViolationTracker._save` all do
   load→mutate→write guarded only by a `threading.Lock` (or nothing). Two processes
   interleave → lost transactions, **a confirmed pending order silently dropped**,
   warrant holdings desynced from reality, clobbered escalation counters. Fix: an OS
   file lock (the `jsonl_sidecar_lock` pattern) around each RMW, or a single-writer
   guarantee per file.

3. **[P1] `M` `metals_loop.py:2075` — `emergency_sell` with bid=0.** When no live bid is
   available the EOD path submits a `NORMAL` (limit) sell at `price=0`; the code then
   appends to `sold` regardless, **faking a successful EOD flat** while the position is
   actually still open overnight. Fix: guard `bid<=0` inside `emergency_sell` — refetch a
   quote, or skip + alert; never submit price 0; use an aggressive-limit like
   `eod_market_flat`.

4. **[P1] `O` `agent_invocation.invoke_agent` ignores `claude_gate.CLAUDE_ENABLED`.**
   The documented master kill switch ("blocks ALL Claude invocations, no exceptions")
   does not stop the primary Layer 2 spawn path — only `config.layer2.enabled` does. An
   operator who flips the kill switch during an incident is not protected. Fix: check
   `CLAUDE_ENABLED`/`check_claude_gates` at the top of `invoke_agent`.

5. **[P1] `IND-7` `agent_invocation._kill_overrun_agent` — silent Layer 2 wedge.** If
   `taskkill` genuinely fails or `wait()` times out, `_agent_proc` stays set; every
   later `invoke_agent` + watchdog tick re-attempts the kill, fails, returns False →
   Layer 2 permanently dead with only `logger.error`, **no Telegram / `critical_errors`
   escalation**. Same silent-outage class as Mar–Apr. Fix: escalate after N failed kills.

6. **[P1] `IND-1`/`INF` `file_utils` JSONL locking.** (a) `atomic_write_jsonl`
   full-rewrite takes **no** `jsonl_sidecar_lock` → loses any append landing between its
   read and `os.replace`. (b) `jsonl_sidecar_lock` lock-file creation is a TOCTOU race
   and, if creation fails, the next `open(lock_path,"rb+")` raises an uncaught
   `FileNotFoundError` → **crashes every JSONL writer**. Fix: `os.open(...,O_CREAT)` +
   open with `"a+b"`; take the lock in `atomic_write_jsonl`.

7. **[P1] `IND-10` `trigger.py` — first-of-day T3 full review is unreachable.**
   `check_triggers` writes `last_trigger_date=today` before `main.py:849` calls
   `classify_tier` (which reloads state); the `last_trigger_date != today → return 3`
   branch can therefore never fire. The intended daily full review is dead. Fix: pass the
   pre-save state into `classify_tier`, or store `last_trigger_date` as the *previous*
   trigger's date.

8. **[P1] `M` `fin_snipe_manager.py:61,536` — 5% cert stop on 5× MINIs + no barrier
   check.** `HARD_STOP_CERT_PCT=0.05` on a 5× cert ≈ 1% underlying move → whipsaw
   stop-out on normal silver intraday range; contradicts the documented "5× certs need
   −15%+ stops" rule. Separately, the stop trigger is never checked against the knockout
   `barrier_level` (corroborated by `IND-16` for `metals_swing_trader._set_stop_loss`).
   Fix: leverage-aware stop width; clamp any stop trigger to a safe margin inside the
   live knockout barrier.

9. **[P1] `SC` ML train/test hygiene.** `meta_learner.py:269,290-297` runs LightGBM
   early-stopping **and** the accuracy-maximising threshold search on the same test set
   → `calibrated_threshold` (used by `predict()` in production) and the reported
   accuracy are in-sample/optimistic. `signal_weight_optimizer`/`train_signal_weights`
   walk-forward split has **no embargo** → 1d/3d forward returns leak across the
   train/test boundary. Both inflate confidence that downstream **sizes real trades**.
   Fix: separate validation slice for calibration; embargo gap ≥ one horizon.

10. **[P1] `SC` `outcome_tracker`/`signal_db` — backfilled outcomes can vanish.** A
    crash between the JSONL append and the SQLite write leaves a snapshot in JSONL only;
    `update_outcome` then `return False` for that ts, and since `load_entries()` reads
    SQLite once it has any rows, those outcomes never reach accuracy/IC/gates. Fix:
    write SQLite first/atomically; `update_outcome` should insert the missing row.

11. **[P1] Incomplete 252→365 annualization** (`M` metals_risk, metals_execution_engine;
    `IND-17` risk_management:462). 24/7 instruments annualized with 252 trading days →
    vol ~21% low → Monte-Carlo VaR and stop-hit probabilities optimistic. Fix: 365
    (and 24h/day) on the crypto/metals paths.

12. **[P1] `PR` `monte_carlo_risk.py:408` — raw `fx_rate` `.get`.**
    `agent_summary.get("fx_rate", FX_RATE_FALLBACK)` is the exact anti-pattern
    `risk_management._resolve_fx_rate` (P1-15) was built to kill; a stale
    `fx_rate:1.0` makes every `*_sek` VaR/CVaR ~10× too small. Fix: route through
    `_resolve_fx_rate`.

13. **[P1] `DE` silent stale / dead signals.** `onchain_data` serves 24h-stale cache as
    fresh when the token is missing (no staleness marker). `funding_rate` SELL threshold
    `>0.0003` is 30–100× above real Binance funding → the signal is permanently silent.
    `data_collector.py:96` builds tz-naive timestamps (`pd.to_datetime(...,unit="ms")`,
    no `utc=True`) → on a CET box every signal timestamp is offset. Fix: staleness
    field + `None` past TTL; recalibrate the funding threshold; add `utc=True`.

14. **[P1] `PR` `warrant_portfolio` — no corruption recovery.** `load_warrant_state`
    has no corruption handling and `save_warrant_state` has no backup rotation (unlike
    `portfolio_mgr`). A corrupt warrant file silently becomes empty holdings and the
    next write **permanently loses all leveraged positions**. Fix: mirror
    `portfolio_mgr._load_state_from` backup+recovery.

15. **[P1] `PR` `risk_management.check_drawdown` blind to stale feeds.** When holdings
    exist but `agent_summary` is empty, it falls back to cash-only value → an underwater
    portfolio reads a tiny drawdown and the circuit breaker never trips. (Documented
    in-code as a known trade-off; still a real blind spot.) Fix: return `breached=True`
    or a distinct `unknown` halt state when holdings exist with no price feed.

---

## 3. Systemic themes (root-cause view)

### 3.1 Cross-process state needs OS locks, not `threading.Lock`
Five independent files use `threading.Lock` (or nothing) to guard read-modify-write of a
file that ≥2 processes touch. This is the single most corroborated finding of the review
(5 reviewers, 5 files). The codebase already has the right primitive —
`file_utils.jsonl_sidecar_lock` (a real cross-process OS lock) — but state-file RMW paths
don't use it. **Recommendation:** add a `with file_lock(path):` context manager generalised
from `jsonl_sidecar_lock` and wrap every state-file RMW (`portfolio_mgr.update_state`,
`warrant_portfolio.record_warrant_transaction`, `health.heartbeat`, `avanza_orders`
pending file, `loop_contract` tracker). This is one focused change that closes five
findings.

### 3.2 The 252→365 fix was never finished
Three more 252-hardcoded annualization sites survived the prior batches. **Recommendation:**
a single grep sweep (`252`, `trading_days`, `sqrt(252`) and a shared
`annualization_factor(asset_class)` helper so this can't regress a fourth time.

### 3.3 Silent failure is still the dominant risk class
Despite the auth-outage hardening, new silent-failure paths exist: `_kill_overrun_agent`
wedge, `onchain` stale cache, `avanza_session.api_post` `{"raw":body}`, `scanner`
swallowed exceptions, `telegram_poller` infinite no-escalation loop. **Recommendation:**
any `except` that returns a degraded value must (a) log at WARNING+, and (b) after N
repeats, write a `critical_errors.jsonl` entry — the system already has the dispatcher
to act on those.

### 3.4 `file_utils` is load-bearing and slightly under-built
~20 writers depend on it. `atomic_write_jsonl` bypasses the lock; `jsonl_sidecar_lock`
creation can crash callers; readers can see torn lines. **Recommendation:** treat
`file_utils` JSONL paths as a small, well-tested unit and close all four gaps together.

---

## 4. Per-subsystem findings (reconciled severity)

Severity below is the **lead's reconciled verdict**, which may differ from the subagent's
original tag (see §6).

### signals-core — 0 P0 · 3 P1 · 4 P2 · 2 P3
- **P1** `meta_learner.py:269,290-297` — test-set leakage in early-stopping + threshold calibration.
- **P1** `outcome_tracker.py:157-166` + `signal_db.py:159-173` — JSONL/SQLite crash-gap drops backfilled outcomes.
- **P1** `signal_weight_optimizer.py:90-116` + `train_signal_weights.py:90-96` — walk-forward split has no embargo.
- **P2** `signal_engine.py:448` — `_ACCURACY_GATE_HIGH_SAMPLE_MIN=7000` contradicts comment/`signals.md` ("10,000"). *Lead-confirmed by direct read.*
- **P2** `ic_computation.py:241-262` — single-file IC cache thrashes across 7 horizons.
- **P2** `signal_decay_alert.py:27,148` — relative `data/` paths → silent no-op under a different CWD.
- **P2** `ticker_accuracy.py:86,131` — per-ticker accuracy gate `min_samples=5` vs engine's 30; "small samples lie".
- **P3** `feature_normalizer.py:35-40` — non-atomic buffer creation under the 8-worker pool.
- **P3** `regime_alerts.py` — non-atomic check-then-log → double regime-change log.

### orchestration — 0 P0 · 2 P1 · 3 P2 · 4 P3
- **P1** `agent_invocation.invoke_agent` — ignores `claude_gate.CLAUDE_ENABLED` master kill switch.
- **P1** `IND-7` `agent_invocation._kill_overrun_agent` — permanent kill-failure wedges Layer 2 with no escalation.
- **P2** `loop_processes.py:92-106` — `UnboundLocalError` if `p.info` raises → crashes `/api/loop-processes`.
- **P2** `escalation_gate.py:202-219` — per-call `ThreadPoolExecutor`; hung non-daemon worker leaks/blocks exit.
- **P2** `loop_contract.py` — shared `contract_state.json` RMW clobbers escalation counters (see §3.1).
- **P3** `IND-8` multi-agent `wait_for_specialists` blocks the main loop 30s; orphan specialist procs.
- **P3** `IND-9` `auth_error` completion status sends no Telegram alert (failed/incomplete do).
- **P3** `bigbet.py:537-559` — streak mutations not persisted on the not-met/cooldown paths.
- **P3** `IND-11/12` `trigger.py` stale docstrings (monotonic-reset claim wrong; "10-minute cadence").

### portfolio-risk — 0 P0 · 8 P1 · 5 P2 · 2 P3
- **P1** `kelly_sizing.py:296-310` — fabricated 1.5:1 payoff ratio dominates position size with no empirical backing.
- **P1** `kelly_sizing.py:314-315` — size capped vs raw *cash*, not portfolio value net of existing position.
- **P1** `risk_management.py:217-270` — `check_drawdown` cash-only fallback blinds the breaker on a stale feed.
- **P1** `cumulative_tracker.py:57-90` — 2KB tail read can hit a partial line → duplicate snapshot.
- **P1** `monte_carlo_risk.py:408` — raw `fx_rate` `.get` → stale 1.0 → VaR ~10× understated.
- **P1** `warrant_portfolio.py:25-39` — no corruption recovery / backup rotation → silent loss of all warrant positions.
- **P1** `warrant_portfolio.py:182-265` — unlocked RMW (see §3.1).
- **P1** `exit_optimizer.py:325-332` — `pct_move*leverage` P&L model is wrong for path-dependent MINIs.
- **P2** `portfolio_mgr.py:44-62` — `_rotate_backups` uses non-atomic `shutil.copy2` (*downgraded from subagent P0 — see §6*).
- **P2** `trade_guards.py:103-330` — check-then-act TOCTOU (*downgraded from subagent P0 — see §6*).
- **P2** `risk_management.py:312` — drawdown breaker uses strict `>` (no trip at exactly the threshold).
- **P2** `equity_curve.py:494-501` — win/loss split mixes gross `pnl_pct` and net `pnl_sek` conventions.
- **P2** `monte_carlo_risk.py:188-198` — short positions silently report zero risk.
- **P3** `cost_model.py` — `total_cost_pct` excludes the min-fee floor; misleads break-even math near 1000 SEK.
- **P3** `portfolio_validator.py:43` — reconciliation runs against a guessed `initial_value_sek`.

### metals-core — 1 P0 · 3 P1 · 4 P2 · 3 P3
- **P0** `grid_fisher.py:1026-1037` — `_safe_session_call` timeout → duplicate live order (top finding #1).
- **P1** `metals_loop.py:2075` — `emergency_sell` bid=0 submits a 0-price order / fakes EOD flat.
- **P1** `fin_snipe_manager.py:61,536` — 5% cert stop on 5× MINIs (whipsaw) + no knockout-barrier check on stops.
- **P1** `IND-16` `metals_swing_trader._set_stop_loss` — fixed warrant-% stop with no barrier-distance clamp.
- **P2** `grid_fisher.py:698-724` — `reconcile_against_live` attributes a fill to the wrong tier when two vanish at once.
- **P2** `metals_risk.py:135,189` + `metals_execution_engine.py:141,148` — 252-day annualization for 24/7 metals.
- **P2** `metals_loop.py:2447-2498` — cascading-stop refresh cancels all stops then can leave the position unprotected if every replacement fails.
- **P2** `metals_execution_engine.py:38,53` — hardcoded `21:55` close (rule says query `todayClosingTime`).
- **P3** `grid_fisher.py:1643-1648` — EOD market-sell branch doesn't cancel still-armed buy tiers → re-open after flat.
- **P3** `metals_swing_trader.py:3210` — limit SELL placed before the hardware stop is cancelled (brief double-live window).
- **P3** `metals_swing_trader.py:2450` — `_check_entry_criteria` has no market-open lower bound (currently masked by the loop guard).

### avanza-api — 0 P0 · 0 P1 · 2 P2 · 1 P3
*(subagent tagged all three P0/P1; lead downgraded — see §6.)*
- **P2** `avanza_session.py:336-342` — `api_post` returns `{"raw":body}` on 200+unparseable JSON; a genuinely-succeeded order is then recorded as FAILED → re-order / double-position risk.
- **P2** `avanza_session.py:94-95` — `load_session` proceeds (fail-open) on an unparseable `expires_at`.
- **P3** `avanza/scanner.py:86-87` — `_marketdata` swallows all exceptions, returns `{}` (catalog-refresh degradation).

### signals-modules — 0 P0 · 1 P1 · 1 P2 · 1 P3
- **P1** `signals/crypto_macro.py:228,281` — `OPTIONS_TTL` used before definition (works only via late binding).
- **P2** `ml_signal.py:154-155` — `model.predict()` on possibly all-NaN features.
- **P3** `forecast_signal.py` (9 sites) — unguarded `/current_price` (*downgraded from subagent 9×P0 — see §6*).

### data-external — 0 P0 · 3 P1 · 6 P2 · 3 P3
- **P1** `onchain_data.py:280-283` — 24h-stale cache served as fresh when the token is unset.
- **P1** `funding_rate.py:44-45` — SELL threshold 30–100× above real funding → signal permanently silent.
- **P1** `data_collector.py:96` — tz-naive `pd.to_datetime` → CET/UTC offset in every signal timestamp.
- **P2** `metals_precompute.py:407` / `oil_precompute.py:407` — CFTC COT fetch has no retry.
- **P2** `alpha_vantage.py:140-142` — no rate-limit vs server-error distinction; quota usage invisible.
- **P2** `fear_greed.py:105-109` — `data_list[0]` reached after the empty-list guard → `IndexError`.
- **P2** `futures_data.py:50` — `KeyError` on a malformed Binance OI payload freezes the worker thread.
- **P2** `microstructure_state.py:227` — `persist_state` snapshots across multiple lock acquisitions → inconsistent state.
- **P2** `onchain_data.py:282` — stateless cache exposes no staleness age to callers.
- **P3** `fx_rates.py:48`, `macro_context.py:100-102` (NaN→invalid JSON), `metals_precompute.py:292` (silent abort).

### infrastructure — 0 P0 · 4 P1 · 4 P2 · 4 P3
- **P1** `file_utils.py:240-248` — `jsonl_sidecar_lock` creation race + uncaught `FileNotFoundError` crashes JSONL writers.
- **P1** `IND-1` `file_utils.py:295-313` — `atomic_write_jsonl` rewrites without the sidecar lock → lost appends.
- **P1** `gpu_gate.py:225-230` — same-process re-entry path `unlink`s a lock another in-process holder depends on.
- **P1** `dashboard/trading_status.py:278-279` — session-window docstring contradicts the constants (reintroduces a fixed bug).
- **P2** `health.py:64-86` — cross-process `health_state.json` RMW under a thread-only lock (see §3.1).
- **P2** `log_rotation.py:358-364` — fixed `.tmp` name races between overlapping rotations.
- **P2** `log_rotation.py:333-344` — archive `.gz` rewritten in RAM non-atomically → corrupt archive on crash.
- **P2** `telegram_poller.py:113-121` — infinite 5s retry loop, no backoff, no escalation.
- **P3** `config_validator.py` (falsy non-string keys pass), `memory_consolidation.py:371` (non-atomic write), `journal*/vector_memory` (raw readers), `digest.py:168` (skewed metric).

### Lead-only cross-subsystem findings
- **P1** `IND-13` `avanza_orders` pending-file cross-process RMW race → silently dropped confirmed order.
- **P2** `IND-3` `jsonl_sidecar_lock` — Windows `LK_LOCK` raises after ~10s, not truly blocking.
- **P2** `IND-4` `portfolio_mgr` `load_state`+`save_state` RMW not atomic even intra-process.
- **P2** `IND-5` `portfolio_mgr._validated_state` doesn't type-check `cash_sek`/`initial_value_sek`.
- **P2** `IND-14` `avanza_orders._check_telegram_confirm` shares the `getUpdates` stream with `telegram_poller` → a CONFIRM reply can be eaten.
- **P3** `IND-6` atomic writers don't fsync the parent directory; **P3** `IND-15` pending-orders file never pruned.

---

## 5. Corroboration matrix

Findings independently surfaced by ≥2 reviewers — these have the highest confidence:

| Root cause | Reviewers that hit it | Confidence |
|------------|----------------------|-----------|
| `threading.Lock` for cross-process state | lead (`IND-2`), portfolio-risk (warrant_portfolio), infrastructure (health), lead (`IND-13` avanza_orders), orchestration (loop_contract) | **Very high — 5 hits** |
| `file_utils` JSONL locking gaps | lead (`IND-1`,`IND-3`), infrastructure (×2) | **High — 4 hits** |
| Incomplete 252→365 annualization | lead (`IND-17`), metals-core (×2 files) | **High — 3 hits** |
| Knockout-barrier not checked when placing a stop | lead (`IND-16`), metals-core (fin_snipe_manager) | **High — 2 hits** |
| Silent stale/swallowed data | data-external (onchain), avanza-api (api_post, scanner), infrastructure (telegram_poller), lead (`IND-7`) | **High — thematic** |
| Drawdown breaker blind on stale feed | portfolio-risk (check_drawdown), lead (independent read agreed) | **High — 2 hits** |
| `signal_engine` 7000 vs 10000 gate constant | signals-core, lead (direct read confirmed) | **Confirmed** |

---

## 6. Cross-critique — downgraded / rejected claims

The lead pass independently judged every subagent P0. **13 of 15 raw P0 claims did not
survive** — recorded here for honesty and so they aren't re-escalated:

- **REJECT→P3 — signals-modules "9× P0" `forecast_signal` division by `current_price`.**
  The subagent counted one root cause as 9 separate P0s. The crash requires
  `prices[-1] == 0` — real market data for BTC/ETH/XAU/XAG/MSTR never yields a zero
  close. `Forecast (Chronos)` is on the **DISABLED** signal list (CLAUDE.md), and the
  signal engine wraps every signal call in try/except, so even a triggered
  `ZeroDivisionError` HOLDs that signal rather than crashing the loop. Real but
  defensive-only: **one P3** ("add `if current_price<=0: return None`").

- **DOWNGRADE→P2 — portfolio-risk P0 `trade_guards` TOCTOU race.** Verified the callers:
  `trade_guards.record_trade` is invoked **only** from `agent_invocation._record_new_trades`,
  which runs in the main-loop process inside `_completion_lock` — serialized.
  `check_overtrading_guards` has no concurrent hot-path caller; the 8-worker pool computes
  signals, not trades. The race is a latent API-design weakness (a future concurrent
  caller would hit it), not a live P0.

- **DOWNGRADE→P2 — portfolio-risk P0 `_rotate_backups` non-atomic copy.** Rotating
  backups *before* the new write is correct semantics (`.bak` = last-known-good prior
  state). The real defect is only that `shutil.copy2` is non-atomic, so a crash
  mid-rotation can truncate **one** backup tier — but the 3-tier chain plus the
  genuinely-atomic main-file write give recovery redundancy. Worth fixing; not a P0.

- **DOWNGRADE→P2/P2/P3 — avanza-api "3× P0".** The cavecrew reviewer first tagged these
  3 P1, then re-tagged them P0. (a) `api_post` `{"raw":body}`: callers default the
  missing status field to `"UNKNOWN"`, which `_execute_confirmed_order` treats as
  **FAILED** and alerts the user — fail-safe-ish, not "appears to succeed". Residual
  risk is a re-order if the order actually landed → **P2**. (b) `load_session`
  fail-open just defers an auth failure that the next API call detects → **P2**.
  (c) `scanner._marketdata` is warrant-catalog refresh — degraded scan, not a
  money path → **P3**.

The two P0 claims that **survived** scrutiny: none from the raw set as-is — the single
reconciled P0 (`grid_fisher` duplicate order) was raised by metals-core as **P1** and the
lead **escalated** it to P0, because it places a duplicate *live* order and Avanza
session-call timeouts are not rare.

---

## 7. What is solid (verified, no action)

- **Dashboard auth** (`dashboard/auth.py`, `cf_access.py`) — `hmac.compare_digest`
  throughout, Cloudflare-Access JWT signature-verified and fail-closed on missing
  config, header/claim email cross-check, last-known-good config cache. No fail-open.
- **`process_lock.py`** — non-blocking OS locks the kernel releases on process death;
  PID reuse cannot grant a false lock.
- **`agent_invocation`** Layer 2 lifecycle — monotonic clock with wall-clock fallback,
  30s completion watchdog, auth-error scan on *both* happy and timeout paths,
  stack-overflow auto-disable, count-delta journal/telegram detection. (Residual risks
  are §2 #5 and the `auth_error` alert gap only.)
- **`avanza_orders`** per-order `confirm_token` — correctly closes the stale-CONFIRM,
  wrong-order, and no-pending-yet races; adds sender authentication. (Residual risks are
  the cross-process file race and the shared `getUpdates` stream.)
- **`file_utils` atomic JSON writers** — `mkstemp` + fsync + `os.replace`, symlink
  resolution for `config.json`. (Gaps are JSONL-specific only.)

---

## 8. Recommended next session (implementation priority)

1. **One change closes five findings:** generalise `jsonl_sidecar_lock` into a
   `file_lock(path)` context manager; wrap every state-file RMW (§3.1).
2. Fix the `grid_fisher` duplicate-order P0 (§2 #1) — reconcile-before-replace.
3. Fix `file_utils` JSONL locking (§2 #6) as one well-tested unit.
4. `invoke_agent` → honour `CLAUDE_ENABLED`; `_kill_overrun_agent` → escalate on
   repeated kill failure.
5. Finish the 252→365 sweep with a shared `annualization_factor()` helper.
6. `trigger.classify_tier` first-of-day fix; `warrant_portfolio` backup/recovery.
7. Defer ML-hygiene fixes (embargo, calibration split) to a dedicated batch — they
   change reported accuracy numbers and need careful before/after validation.

All findings should be triaged into `docs/IMPROVEMENT_BACKLOG.md`. No production code was
changed by this review.

---

## Appendix — methodology

- 8 subsystem branches built from an empty orphan baseline (`review/empty`) so each
  reviewer saw its whole subsystem as a unified diff.
- 8 fresh-context review subagents run concurrently in the background; lead independent
  pass run in parallel over `file_utils`, `portfolio_mgr`, `agent_invocation`, `trigger`,
  `risk_management`, `avanza_orders` (+ targeted `signal_engine`, `metals_swing_trader`).
- Cross-critique: every subagent finding confirmed / downgraded / rejected by the lead;
  corroboration across reviewers tracked in §5.
- Worktree and `review/*` branches were created for the review and removed afterward.
- Raw reviewer outputs preserved in `docs/reviews/2026-05-22-fgl-raw/`.
