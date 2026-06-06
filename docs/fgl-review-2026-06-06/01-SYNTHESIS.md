# Adversarial Review — Synthesis (2026-06-06)

8-subsystem adversarial pass over the finance-analyzer codebase. Method: empty-baseline
git worktree branches (`review/sub-*` vs `review/empty-baseline`, full subsystems as
additive diffs) + one fresh review subagent per subsystem (6× `pr-review-toolkit:code-reviewer`,
2× `caveman:cavecrew-reviewer`) run in parallel, plus an independent orchestrator pass
(`00-own-pass.md`) on the foundational + integration-seam files. This doc cross-critiques
the subagent findings (confirming, rejecting, re-grading) and ranks the survivors by money/
reliability impact.

Per-subsystem raw findings: `02-*.md` … (one file per subsystem). Orchestrator pass: `00-own-pass.md`.

Severity: **P0** money loss / data corruption / outage / silent-failure-masking / secret leak.
**P1** real bug, wrong result. **P2** robustness/race-under-load. **P3** maintainability.

---

## Cross-critique — rejected / re-graded subagent findings

Independent verification (orchestrator read the cited code) overturned three claims. This
is the point of the two-pass design: a per-subsystem reviewer pattern-matches a smell; the
orchestrator checks the actual control flow.

| Claim | Verdict | Evidence |
|---|---|---|
| infra **P0**: `file_utils.atomic_append_jsonl` `seek(-1,SEEK_END)` is UB on new/empty files | **REJECTED** (false positive) | `file_utils.py:343` guards `if f.tell() > 0:` *before* the backward seek. Empty files never reach it. |
| infra **P1**: `shared_state.py:123` `return` inside lock → deadlock, "context-manager exit skipped" | **REJECTED** (false positive) | The `return` (line 124) is inside `with _cache_lock:` (line 111). Python runs `__exit__` on return — the lock IS released. |
| infra **P0**: telegram `/mode` overwrites `config.json`, destroying API keys if `load_json` returns `{}` | **RE-GRADED → P1** | Real *design* smell (writing notification mode into the external secrets symlink) but the size-guard makes key-destruction unconfirmed. Fix is "stop writing mode into config.json", not "add a lock". Verify before P0. |
| infra **P1**: newsapi quota reset clock-skew | **RE-GRADED → P2** | Robustness only; needs a backward clock jump to trigger. |
| orchestration **P0** vs OWN-1: `CLAUDE_ENABLED` master switch bypass | **CONFIRMED, graded P1** | Independently confirmed (`agent_invocation.py:853` gates only on `config.layer2.enabled`). Defeats an incident control but does not itself mis-trade → P1, not P0. |
| metals **P0**: grid global-cap unguarded breach | **N/A — cap is guarded** | `00-own-pass.md`: per-tier `projected > global_cap_sek` re-check (grid_fisher.py:1417-1424) + fail-closed `_effective_global_cap`. Real residual is only a *cross-process* race, blocked by the singleton loop lock. |

Everything else below survived cross-critique.

---

## Confirmed P0 — fix first (real money / outage / silent failure)

1. **Naked overnight leveraged exposure — grid EOD flatten strips the stop before the sell is confirmed.**
   `portfolio/grid_fisher.py:1940-1973`. EOD-flat sets `stop_loss_id = None` (line 1948) — *even
   when `cancel_stop_loss` failed* — before confirming the replacement `bid*0.99` limit sell, which
   can fail to fill in the illiquid close auction. On sell failure (`continue` at 1966/1976) the
   position is left with **no stop and no sell** = naked overnight leveraged warrant. This is the
   exact invariant the subsystem exists to forbid. → Confirm the sell (or re-arm the stop) before
   clearing `stop_loss_id`; re-arm on sell failure. *(verify-recommended; highest stakes finding.)*

2. **Session-roll abandons unprotected inventory.** `portfolio/grid_fisher.py:517 + 1923`.
   `roll_session_if_new_day` clears `eod_sell_order_id` while `inventory_units > 0` (prior EOD sell
   never filled); the stop was already nulled the prior day → new session holds unprotected inventory
   until the next EOD window. → On roll with open inventory + unconfirmed EOD sell, set
   `stop_needs_rearm = True`. *(verify-recommended.)*

3. **Silent failed stop-loss deletion → overfill. CONFIRMED by orchestrator.**
   `portfolio/avanza_control.py:401`. `delete_stop_loss_no_page` checks `result.get("errorCode")`,
   but `api_delete` (`avanza_session.py:377`) returns `{"http_status", "ok"}` — **there is no
   `errorCode` key**. An HTTP 500 returns `{"http_status":500,"ok":False}` → the check is falsy →
   function returns `True`. Caller believes the stop was cancelled. Per the metals rule "cancel
   existing stops BEFORE placing a sell (prevents overfill)", a silent failed-delete leaves the old
   stop + the new sell both live → overfill / double-exit. → Check `result.get("ok") is True`.

4. **`http_retry` unbounded `retry_after` sleep → 60s-loop stall.** `portfolio/http_retry.py:53-62`.
   A server-supplied 429 `retry_after` is used verbatim as the sleep, then `wait += random.uniform(0,
   wait)` is added on top, with no ceiling and no cumulative budget. A large/garbage value blocks a
   worker for minutes → loop stall (system-reliability = #1 priority). → Clamp each wait to ~10-20s,
   cap cumulative sleep, validate `retry_after` is a sane positive number.

5. **`_cached` serves stale-as-live on the fetch-error path.** `portfolio/data_collector.py:74-101`
   + `shared_state.py:109-125`. On a fetch exception the prior value is returned with its `time`
   refreshed and **no staleness marker** — futures/funding/onchain/cross-asset/treasury/dxy all flow
   through this. A consumer cannot distinguish a fresh quote from post-failure cache. Violates "LIVE
   PRICES FIRST" and is the same class as the 3-week auth outage (default value masking a degraded
   state). → Attach a `_stale`/`_as_of` marker on the error path (as `price_source` already does via
   `df.attrs`). *(Note: line 123 itself is NOT a deadlock — see cross-critique.)*

6. **Sequential specialist wait collapses the multi-agent window.** `portfolio/multi_agent_layer2.py:218`.
   `wait_for_specialists` waits on the 3 specialists sequentially against a shared deadline; if #1
   consumes the window, #2/#3 get the `max(1, …)` 1-second floor and are force-killed, and synthesis
   proceeds on their empty `.md` reports (only quorum==0 is blocked). → Poll all three concurrently
   until the shared deadline. *(verify-recommended.)*

7. **No in-band wall-clock timeout on the Layer-2 subprocess.** `portfolio/agent_invocation.py:1224`.
   Enforcement relies on the per-cycle completion check + a 30s daemon watchdog whose only liveness
   guarantee is a broad `except Exception` (a `BaseException` kills it). Watchdog death + a stalled
   `run()` → an unbounded hung Claude. → Add a watchdog-liveness assert+respawn, or a hard
   `communicate(timeout=tier_timeout)` on the subprocess.

---

## Confirmed P1 — fix this session

- **Unlocked warrant book → money-state corruption.** `portfolio/warrant_portfolio.py:199-280`.
  `record_warrant_transaction` does unlocked load→mutate→save on `portfolio_state_warrants.json`
  while the two main books are lock-serialized. Concurrent metals-loop/grid fills race →
  last-writer-wins drops a transaction, corrupting units/avg-entry. → `update_warrant_state(mutate_fn)`
  mirroring `portfolio_mgr` + the sidecar lock. (Same class as OWN-2: lock-contract not uniform.)
- **Corruption→default reset overwrites a quarantined book.** `portfolio/portfolio_mgr.py:228-234`.
  On unrecoverable corruption `_load_state_from` returns a fresh 500K default; `update_state` then
  writes that reset book over the quarantined file → money created/destroyed mid-session. → Refuse to
  write when the load came from the corrupt→default branch.
- **Drawdown breaker fails toward trading.** `portfolio/risk_management.py:252-271`. Cash-only
  fallback when `agent_summary` is empty under-reports drawdown for underwater holdings — the one
  guard meant to *halt* fails open. → Use `avg_cost` fallback, or treat stale-feed-with-holdings as halt.
- **VaR understated ~10× by unsanitized fx_rate.** `portfolio/monte_carlo_risk.py:408,485-491`.
  `fx_rate = agent_summary.get("fx_rate", FALLBACK)` with no sanity band; a stale `1.0` understates
  every `*_sek` VaR/CVaR ~10×. → Reuse `_resolve_fx_rate` / clamp to [7,15].
- **`CLAUDE_ENABLED` master kill-switch bypass** (OWN-1 + orchestration). `agent_invocation.py:853`
  gates trade spawns only on `config.layer2.enabled`. → Honor the master switch on the money path.
- **Post-cap confidence inflation.** `portfolio/signal_engine.py:4679-4698`. Seasonal BUY multiplier
  (1.10-1.15×, Jan-Apr) runs *after* the `min(conf, 0.80)` cap → metals BUY returns 0.88-0.92,
  feeding inflated conviction to Layer 2 / sizing. → Re-apply the cap after the multiplier.
- **`crypto_evrp` level-vote direction contradicts its documented edge** (active signal).
  `signals/crypto_evrp.py:195-201`. Votes SELL for eVRP>10 / BUY for eVRP<-10 — opposite of the
  docstring's stated edge. → Reconcile; if the docstring is the validated edge, the vote is inverted.
- **`OrderLockBusyError` abandons a confirmed order.** `portfolio/avanza_orders.py:207-214`. Lock
  contention is caught by the broad handler → `status="error"`, the user's confirmation is discarded.
  → Catch separately, preserve `status="confirmed"` for retry.
- **Barrier-blind stop triggers.** `data/metals_swing_trader.py:2759-2760` + `grid_fisher.py:1546-1547`.
  Stop trigger anchored to a fixed warrant % with no knockout-barrier validation → can land below the
  knockout on near-barrier certs. → Apply the existing `_tier_skip_for_knockout` guard to stops.
- **Meta-learner threshold tuned on the test set** (look-ahead). `portfolio/meta_learner.py:289-310`.
  Calibrated threshold maximises accuracy on the only OOS slice, then reported + deployed → optimistic
  + overfit. → Tune on a separate validation split. *(re-graded P2→P1: it inflates a deployed gate.)*
- **AV earnings calls bypass the 25/day budget.** `portfolio/earnings_calendar.py:48-61`. → Share one
  AV daily counter. (Quota burn breaks OVERVIEW refreshes.)
- **`get_open_orders` fails open.** `avanza_session.py:653-669` + `grid_fisher.py:1664`. Returns `[]`
  (not None) on total endpoint failure; grid's degraded guard only trips on None → a transient orders
  failure makes reconcile mark every resting buy cancelled/filled. → Fail closed (raise/return None).

## Confirmed P2 (selected)
- OWN-2: `atomic_write_jsonl` bypasses `jsonl_sidecar_lock` (latent lost-append; callers single-writer today).
- Grid reconcile partial-fill overwrites `tier.qty` → inventory under-count + `avg_entry_price` poisoning (`grid_fisher.py:730-757`).
- `futures_data.py:62-121` bracket-key parse crashes whole batch on a partial row (hardening from `funding_rate.py` not mirrored).
- `fx_rates.py:61-72` returns stale USD/SEK indistinguishable from live (age flag missing).
- `kelly_sizing.py:83-88` break-even round-trips bucketed as losses → Kelly under-sizes.
- `signal_db.py:262-301` SQL accuracy lacks the `_MIN_CHANGE_PCT` neutral band → diverges from the live Python path.
- `realized_skewness.py:228-239` (active) missing the 0.7 confidence cap → single sub-signal yields conf 1.0.
- `circuit_breaker.py:98` `_half_open_probe_sent` set but never read (dead flag, double-probe regression risk).

## P3 (maintainability) — see per-subsystem files
Dead code (`_accuracy_tier_mult` documented in CLAUDE.md but never called — `signal_engine.py:525`),
`forecast_signal.py:97` redundant `(ImportError, Exception)`, ic_cache single-horizon thrash,
OWN-3 POSIX dir-fsync gap (prod-safe on Windows), crypto_macro_data raw `open()+json.loads` JSONL reads.

---

## Systemic themes (the through-line across subsystems)

1. **Fail-toward-trading / stale-as-live is the dominant risk shape.** Drawdown cash-only fallback,
   VaR fx_rate=1.0, `_cached` post-error serve, `fx_rates` stale serve, `get_open_orders` fail-open —
   all *act* (or *under-report risk*) when an input is silently degraded. The system is excellent at
   *atomic writes* but weak at *refusing to act on known-stale inputs*. This is the same failure class
   as the 3-week auth outage. **Recommendation:** a shared `StaleInput` convention (every cache/feed
   returns an `as_of`/`stale` flag; risk + sizing treat stale-with-exposure as halt-or-de-risk).

2. **The stop-loss lifecycle is the metals subsystem's weakest seam.** Three independent P0/P1s
   (EOD strip-before-confirm, session-roll abandon, silent failed-delete) all share one root: stop
   state is mutated *before* the broker confirms the protective replacement. **Recommendation:** a
   single invariant — never clear/replace a stop without a confirmed protective order in place; assert
   "position ⇒ (live stop OR confirmed exit)" every tick and alert+re-arm on violation.

3. **Lock-contract completeness.** The sidecar-lock + portfolio_mgr lock are correct where applied
   but not uniformly (warrant book unlocked, `atomic_write_jsonl` bypasses the lock). Audit every
   state/JSONL rewrite path for lock coverage.

4. **Doc/behavior drift.** Accuracy-tier boost is dead code; SQL vs Python vs forecast accuracy use
   three different neutral-band conventions; CLAUDE.md describes behavior the live gate doesn't run.
   Dashboards and offline tuning can disagree with production. **Recommendation:** one accuracy
   primitive (`_vote_correct`) called by every path; delete or wire the tier boost.

## Suggested fix order
P0 #3 (avanza errorCode — confirmed, one-line, money) → P0 #1/#2 (grid stop lifecycle — verify then
fix together under theme #2) → P0 #4 (http_retry clamp — one-line, loop stall) → P0 #5 (stale marker
— enables theme #1) → P1 warrant-lock + portfolio_mgr reset-guard → remaining P1 → P2/P3.
**No live config/weight/threshold changes** without human approval (per protocol). All fixes additive
and test-covered.
