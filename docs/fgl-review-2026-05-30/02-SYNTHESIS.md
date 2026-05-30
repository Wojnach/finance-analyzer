# FGL Adversarial Review — SYNTHESIS

**Date:** 2026-05-30 · **Baseline:** main @ `1730651f`
**Method:** codebase partitioned into 8 subsystems; one fresh review subagent per
subsystem (6× `pr-review-toolkit:code-reviewer`, 2× `caveman:cavecrew-reviewer`)
run in parallel against a whole-subsystem diff vs an orphan `fgl/empty-baseline`
branch in a clean worktree; plus an independent orchestrator pass on the
foundations + cross-cutting themes; then cross-critique against source and live
runtime data.

**Raw output:** 87 findings (76 subagent + 11 orchestrator/foundation+meta).
**After de-dup + cross-critique + severity correction: ~60 distinct, P0:5 · P1:~22 · P2:~25 · P3:~16.**

---

## 0. Executive summary

The system is **mature and heavily defended** — atomic I/O primitives are correct,
the signal engine sanitizes/fails-closed, auth-failure detection exists, crash
recovery is robust, and the live Avanza order helpers use the right stop endpoint
with a cross-process lock. The bugs that remain are concentrated in **five
patterns**, not scattered randomly:

1. **Hot write-paths bypass the correct atomic primitives** (state-loss races).
2. **Silent stale-data fallbacks** mask a dead source (violates "live prices first").
3. **Producer/consumer contract drift** — two files that must agree on a vocabulary
   silently diverge.
4. **EOD-flat reachability** — a halted/stalled metals loop can leave leveraged
   warrant inventory open overnight with no operator alert.
5. **Reconstructed-history methodology bugs** — sub-signals built with the current
   price held constant across the lookback window.

The headline operational pain — the `contract_violation` alert firing ~20×/week —
was **mis-diagnosed by two independent static analyses** and corrected by live
data (see §2). That correction is the most important single result here.

---

## 1. P0 master list (production-breaking — fix first)

| # | Subsystem | Location | Problem | Fix |
|---|-----------|----------|---------|-----|
| P0-1 | avanza-api | `avanza_session.py:89` | Session expiry not eagerly checked; `AvanzaSessionError` swallowed by callers → multi-day **silent auth outage** (the real 2026-05-23 incident, still unresolved in `critical_errors.jsonl`). | Verify expiry at `_get_playwright_context()`, fail-closed + alert before any order path. |
| P0-2 | avanza-api | `avanza_control.py:356-376` | Legacy page-based `place_stop_loss` can route through the **regular order endpoint** instead of `/_api/trading/stoploss/new` → instant fill at worst price on trigger (Mar-3 incident class). | Unified dispatch that detects stop-vs-regular and forces the stop endpoint; delete/guard the legacy path. |
| P0-3 | portfolio-risk | `warrant_portfolio.py:100-103` | No knockout floor: a 5× warrant past −20% underlying yields a **negative** per-unit value → negative portfolio value, corrupting drawdown + VaR (which feed the circuit breaker). *Source-validated.* | `current_implied_sek = max(0.0, …)` + surface a knockout flag at 0. |
| P0-4 | data-external | `price_source.py:240-262` | `fetch_klines` silently substitutes 10-15-min-stale yfinance on ANY primary failure, **no staleness tag** → dead Binance FAPI feed drives metals/crypto trades on phantom data. | Tag source/age on the frame; consumers down-weight/HOLD on fallback; raise for real-time-critical tickers. |
| P0-5 | orchestration | `agent_invocation.py:1622` | `status="failed"` (exit≠0, no auth marker) writes **no journal stub** and isn't in `_KNOWN_FAILURE_STATUSES` → a genuine Layer-2 crash is invisible, buried among the false `contract_violation` noise. The one **true silent-failure** in the contract story. | Write a `failed` journal stub (mirror the `incomplete` branch) + add `"failed"` to `_KNOWN_FAILURE_STATUSES`. |

**Borderline P0 (rated P1 by reviewer, flagged for elevation):** metals-core EOD
leave-open cluster (`grid_fisher.py:1751`, `1538`, `1906`; `metals_loop.py:7219`) —
leveraged warrant inventory held overnight unprotected, with failures only in a
JSONL (no Telegram). Severe blast radius (real money, gap risk), gated by the halt/
stall precondition. Treat as **P1-critical**, fix alongside P0.

---

## 2. ⚠ Cross-critique correction: the `contract_violation` root cause

**This is the review's flagship finding.** The recurring critical error
`layer2_journal_activity :: trigger fired but no journal entry written` fires
~20×/week. Two independent analyses — the `orchestration` subagent (caller-side)
and the orchestrator's own pass (producer/consumer-side) — **converged** on the
same mechanism: `main.py:989` re-logs `skipped_busy` after a legitimate internal
skip, and `skipped_busy` is excluded from the contract whitelist. Convergence felt
like near-certainty.

**Live data refuted it as the dominant cause.** `last_invocation_status` on the
last 40 real violations (`data/contract_violations.jsonl`):

```
success: 29 (72%) │ invoked: 6 (15%) │ incomplete: 2 (5%)
skipped_busy: 1 (3%) │ auth_error: 1 (3%) │ skipped_auth_cooldown: 1 (3%)
```

The `skipped_busy`-clobber story explains **~3%** of fires. The dominant driver
(72%) is **`status="success"`** — the agent ran, exited 0, and journaled — and the
contract fired anyway. That is a **contract-window / timestamp-lag** defect:

- `last_trigger_time` is stamped at *end of cycle* (loop_contract.py:304, health.py),
  and the test is `journal_ts >= last_trigger - 5s`.
- Under back-to-back triggers, trigger B advances `last_trigger_time` to "now" while
  the newest journal is still agent-A's (older) → "newest journal older than newest
  trigger" → fires, even though the system is journaling one invocation behind.
- Dedup (precondition 5) only suppresses re-fires of the *same* trigger, not this
  cross-trigger lag.

**Lesson (process):** two independent reviewers agreeing on a plausible mechanism
is *not* proof. Validation against the live journal — which /fgl mandates
("evidence over opinion") — was the only thing that caught it.

**Fix priority for this defect:**
1. *(primary, ~72%)* In `check_layer2_journal_activity`, require the newest journal
   to post-date the *invocation row that actually ran* (correlate via
   `invocations.jsonl`), not the latest end-of-cycle `last_trigger_time`; and stamp
   `last_trigger_time` at trigger-detection time.
2. *(secondary, ~3%)* Fix the `skipped_busy` clobber (have `invoke_agent` signal
   "already logged terminal status").
3. *(P0-5 above, real silent crash)* Add the `failed` journal stub.

### Other cross-critique severity corrections

| Finding | Reviewer | Correction | Why |
|---------|----------|-----------|-----|
| `health.py:161,202` false-healthy heartbeat | infra **P0×2** | → **P1** | The `except` wraps only `fromisoformat`; a naive parse **raises `TypeError`** at the subtraction (it does *not* "return inf / healthy forever" as written). Triggers only on a corrupted/manually-edited ts; normal path always writes aware `datetime.now(UTC)`. Latent. |
| `file_utils.py:289` append loses writes | infra **P1** | → **P3** | `atomic_append_jsonl` holds `jsonl_sidecar_lock`, which **serializes appenders cross-process** — the "two processes fsync-race, one lost" scenario can't occur through the helper. Residual risk only if a writer bypasses it (covered by Theme B1). |
| `shared_state.py:88` `_loading_keys` leak | infra **P1** | → **P3** | Un-covered window between lock-exit and `try:` is ~zero statements and self-heals via the 120s stuck-key eviction. |
| `skipped_busy` clobber as THE root cause | orchestration **P0** / own pass | → **P1** (real, ~3%) | Live data: dominant cause is `success`-lag, not this. Real but minority. |

Findings spot-validated against source and kept at stated severity: P0-3 (warrant
negative value), P0-4 (price_source), P0-1/P0-2 (avanza), signals-modules P1
(gs_kalman dead signal — confirmed `context_data` never supplies its keys).

---

## 3. Cross-cutting meta-themes (the highest-leverage fixes)

Each theme is one structural fix that closes a *class* of findings no
single-subsystem reviewer could see as systemic.

### Theme B1 — Hot write-paths bypass the correct atomic-RMW primitives
`portfolio_mgr.update_state` (full-cycle lock) exists but is **called nowhere in
production**; the trade path uses bare `load_state()`/`save_state()`.
`warrant_portfolio.record_warrant_transaction` (P1) and
`cusum_accuracy_monitor.update_cusum` (P3) do unguarded load→mutate→save.
`atomic_write_jsonl` skips the sidecar lock that `atomic_append_jsonl`/`prune_jsonl`
hold (own-pass A2). And the in-process `threading.Lock` doesn't guard the **three
processes** (main loop, metals loop, dashboard) that write the same files.
**One fix:** make a locked, cross-process RMW the only sanctioned mutate path;
route warrant + cusum + jsonl-rewrite writers through it. Closes ~5 findings.
*(Members: portfolio-risk P1/P2×2, signals-core P3, own-pass A2.)*

### Theme B2 — Silent stale-data fallback masks a dead source (violates "live prices first")
`price_source` yfinance fallback (**P0-4**), `onchain_data` 24h cache (P2),
`futures_data` fatal-4xx-as-transient (P3 — the 3-week-outage shape),
`metals_cross_assets` empty-frame swallow (P3), `fx_rates` daily-fixed-as-fresh
(P1). **One fix:** a `{value, source, age, stale}` envelope from every fetcher;
fatal-vs-transient typing enforced at the single `fetch_json` boundary; consumers
down-weight/HOLD on stale. Closes ~5 findings + hardens against the exact silent
auth-outage class CLAUDE.md calls out.

### Theme B3 — Producer/consumer contract drift
invoke_agent skip statuses vs `loop_contract._LEGITIMATE_SKIP_STATUSES` (§2);
`claude_gate` reads `timestamp` but `_log_trigger` writes `ts` (orchestration P3);
`SignalDB` SQL accuracy omits the ±0.05% neutral band the Python path applies
(signals-core P1). **One fix:** centralize the status vocabulary and the
scoring/neutral-zone logic in one module each; consumers import from the producer
rather than re-declaring constants.

### Theme B4 — EOD-flat / overnight-hold reachability for leveraged inventory
Four metals P1s converge: halt short-circuits EOD-flat (`grid_fisher.py:1751`);
naked-position re-arm window (`1538`); 5-min EOD window with no catch-up (`1906`);
fishing EOD guard set even on failed sell (`metals_loop.py:7219`); plus session
failures logged only to JSONL (no Telegram). **One fix:** a single end-of-day
reconcile-and-flat path that runs regardless of halt/restart, repeats/widens its
window, and escalates persistent sell failures to Telegram + critical_errors.

### Theme B5 — Reconstructed-history methodology bug
`mstr_mnav_discount` (P2) and `stablecoin_supply_ratio` (P2) build their historical
ratio series with the **current** price held constant across the lookback, so their
velocity/z-score sub-signals collapse to price-only proxies; `momentum_factors`
seasonality detrend compounds across bars (the bug already fixed in
`mean_reversion.py`). **One fix:** a shared "date-aligned historical series"
helper; reconstruct per-bar with the contemporaneous price.

---

## 4. Remediation roadmap (impact × ease)

**Sprint 1 — money/safety/silent-failure (do now):**
- P0-3 warrant knockout floor (1-line clamp, high impact). 
- P0-4 price_source staleness tag (medium).
- P0-1/P0-2 avanza session eager-check + stop-endpoint dispatch (the session one is
  an **active unresolved** critical error since 2026-05-23).
- Theme B4 EOD-flat-always + Telegram escalation (real-money overnight risk).

**Sprint 2 — reliability/observability:**
- §2 contract-window fix (primary `success`-lag) + P0-5 `failed` stub + `skipped_busy`
  clobber → kills the ~20×/week alert noise AND unmasks real crashes.
- Theme B1 single atomic-RMW path.
- monte_carlo `0.0`-truthiness (P1), drawdown cash-only fallback (P1).

**Sprint 3 — signal quality (no live-config changes without approval):**
- signals-core: accuracy_degradation mis-measurement (P1×2), metals Stage-4 quorum
  override (P1), SQL-vs-Python neutral zone (P1), ICIR overlapping windows (P1).
- signals-modules: disable/repair `gs_kalman` dead signal (P1); Theme B5.

**Sprint 4 — Theme B2 stale-data envelope + remaining P2/P3 hygiene.**

---

## 5. What was checked and found clean (negative results matter)
- `file_utils.atomic_write_json/_text` — correct (same-vol tmp, fsync, `os.replace`,
  BaseException cleanup, symlink-resolve).
- `process_lock` — no stale-lock deadlock (OS releases on death); only a fragile
  metadata-truncate-under-lock (P3).
- `shared_state._cached` — dogpile guard, refuses to cache `None`, stuck-key eviction.
- Signal plugin contract — raise→HOLD, NaN→0.0 sanitized, no `except:pass` anywhere
  in `portfolio/signals/`; cross-asset secondaries self-fetch + guard div-by-zero.
- Avanza live order helpers — right stop endpoint, cross-process order lock,
  fail-closed CSRF, robust pre-sell stop-cancel with server poll-verify.

---

## 6. Methodology / reproducibility
- 8 subsystems: signals-core, orchestration, portfolio-risk, metals-core,
  avanza-api, signals-modules, data-external, infrastructure.
- Empty-baseline diff trick: orphan branch `fgl/empty-baseline` (empty tree via
  `git commit-tree`) → `git diff fgl/empty-baseline -- <files>` renders each whole
  subsystem as additions for the diff-oriented reviewers.
- Clean worktree `Q:/fa-fgl-review` @ HEAD isolated reviewers from the dirty main
  working tree (data/*.json churn).
- Each subagent: self-contained adversarial prompt, strict `path:line — Pn —
  problem. fix.` output, no fix/scope-creep.
- Orchestrator: independent foundation read + 4 source validations + 1 live-data
  validation (the §2 correction).
