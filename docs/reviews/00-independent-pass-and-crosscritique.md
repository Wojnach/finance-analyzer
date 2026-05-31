# Independent Pass + Cross-Critique — 2026-05-31

Main-thread (Opus) independent adversarial reading of the 3 highest-blast-radius
subsystems, run *in parallel* with the subagents so their claims can be checked
against a second reading rather than trusted blind. Then a verdict on each
agent's headline finding.

---

## A. Independent findings (verified against live code + data)

### A1. `contract_violation` is overwhelmingly a FALSE POSITIVE — and already mostly fixed; the live problem is a STALE critical-errors backlog
Evidence (queried `data/critical_errors.jsonl` + `data/invocations.jsonl` live):
- 139 historical `contract_violation` rows. `last_invocation_status` distribution:
  `timeout 36, success 33, auth_error 23, invoked 17, failed 14, error 7,
  incomplete 5, skipped_busy 3, skipped_auth_cooldown 1`.
- The last 12 violations are **all `status='success'`** with the journal `ts`
  written **20–180s BEFORE** the trigger `ts` (e.g. 2026-05-28 10:44: jrnl
  10:30:15 vs trigger ~10:30:38). That is a timestamp-provenance race (journal
  stamped with signal-snapshot time; trigger stamped with independent `now()`),
  **not** a silently-dead agent.
- Precondition 6 (loop_contract.py:413-419), deployed 2026-05-30, suppresses
  exactly the `success`+`journal_written` case. **Zero `contract_violation` rows
  have fired since 2026-05-29 18:22** — i.e. since the patch, the dominant cause
  is closed.
- Yet `scripts/check_critical_errors.py` still reports 31 unresolved over 7 days,
  dominated by stale 2026-05-28/29 `contract_violation` entries that nobody
  appended a resolution line for. These (a) drown genuine criticals so an
  operator can't triage, and (b) keep `PF-FixAgentDispatcher` spawning fix agents
  on an already-fixed category (Claude-budget burn).
- **Action:** append `resolution` lines for the pre-2026-05-30 `contract_violation`
  entries (fix is deployed), and make `check_critical_errors.py` treat a category
  as auto-resolved when a known fix-commit timestamp post-dates all of that
  category's unresolved entries (or when the category has not re-fired in N days).
  This is the cheapest highest-value operational fix in the whole review.
- Converges with `02-orchestration` ROOT-CAUSE HYPOTHESIS (timestamp-provenance,
  not silent death) — independently reached. The orchestration agent adds the
  code-fix angle (P0-#1 auth_error stub, P1-#1 naive/aware TypeError); I add the
  empirical "already fixed, backlog is stale" operational angle.

### A2. `outcome_tracker._fetch_historical_price` forward-shift — CONFIRMED P0, fix refined
Read `outcome_tracker.py:185-252` + caller `:454-496`. Confirmed:
- `base_price = tickers[ticker]["price_usd"]` is the **exact** live price at
  `entry_ts` (line 461).
- exit = `_fetch_historical_price(ticker, entry_ts + h_seconds)` returns
  `data[0][4]` = **close** of the 1h bar *containing/after* `target_ts` ≈ price
  0–60min (avg ~30min) AFTER `target_ts` (line 484).
- So the realized horizon is **asymmetrically overstated by up to ~1h**:
  negligible at 1d (~+2-4%), but **~+17% avg / +33% worst at 3h, up to +100% at
  1h**. Those short-horizon accuracy/IC numbers gate signals and feed the Mode-B
  probabilities the user trades.
- **Refined fix** (vs the agent's "bar that ends at target_ts"): use
  `interval="1m", startTime=target_ts, limit=1` and take the bar **open**
  (`data[0][1]`) — that is the price *at* `target_ts` to the minute, no shift,
  no ambiguity about which 1h bucket Binance returns.
- This is the single most important correctness finding in the review.

### A3. signals-modules lookahead spot-checks — regime gates are CLEAN
- `absorption_ratio_regime.py:189-190` uses `ar_series.iloc[:-1]` for mean/std —
  correctly EXCLUDES the scored point. Good.
- `vol_ratio_regime.py` uses `close.shift(1)` and `min_periods=window` — no
  forward leak.
- `eth_btc_ratio_roc_zscore` / `mahalanobis_turbulence` have mild *in-sample*
  contamination (current point inside the z-score window) — not future leakage,
  generally acceptable. Worth a closer look in the dedicated re-review.

---

## B. Cross-critique of agent headline findings

| Subsystem | Headline finding | Verdict | Reasoning |
|---|---|---|---|
| signals-core | outcome backfill forward-shift (P0) | **CONFIRMED** | verified caller convention (A2); refined the fix |
| signals-core | accuracy_degradation = regime false-alarm (P1) | **CONFIRMED (logic)** | matches seeded 4× fire; consistent with the broader "false-positive criticals" theme; recommend regime-gating the alert |
| signals-core | MSTR cross-ticker consensus cache keyed by ticker only, overwritten across 7 horizons (P0) | **PLAUSIBLE, verify** | matches CLAUDE.md (MSTR uses btc_proxy); needs a runtime check that the 1d slot is actually what btc_proxy reads |
| orchestration | timestamp-provenance root cause | **CONFIRMED** | independently reached (A1) |
| orchestration | auth-detection scans rotation-fragile 16-line log slice (P0-#2) | **CONFIRMED-plausible** | read the scan window logic at agent_invocation.py:576-637; the rotation/offset reset is genuinely fragile; this is the one that can reopen the 3-week-outage class |
| orchestration | `check_layer2_journal_activity` unguarded naive/aware TypeError drops whole cycle (P1-#1) | **CONFIRMED-plausible** | the compares at loop_contract.py:442 etc. are aware-side; an LLM-authored naive journal ts would raise; main.py blanket-except swallows |
| portfolio-risk | warrant_portfolio unlocked cross-process read-modify-write (P0-1) | **CONFIRMED-class** | matches the documented 3-process model; atomic write ≠ lost-update protection; high-leverage book → highest impact |
| portfolio-risk | drawdown breaker computed but not enforced (P0-3) | **NEEDS VERIFICATION** | agent explicitly "found no consumer in the 13 files"; enforcement may live in an execution module out of scope — must confirm before treating as live P0 |
| portfolio-risk | kelly_metals → 95% of buying power into one 5x cert (P1-1) | **CONFIRMED-math, verify wiring** | the leverage division is mathematically backwards (leverage should shrink cash deployed); BUT confirm kelly_metals is actually wired to live sizing vs advisory — cross-ref metals-core |
| avanza-api | avanza_account_mismatch = session-expiry mislabeled | **CONFIRMED** | matches seeded error text exactly; same theme as A1 |
| avanza-api | stop-loss API invariant | **VERIFIED HOLDS** | all live paths use `/_api/trading/stoploss/new` — the Mar-3 incident fix is in place |
| data-external | stale-but-silent caches/fallbacks (theme) | **ACCEPT** | consistent pattern across fx/onchain/price_source/metals_cross |
| infrastructure | foundational primitives clean; 1 P1 heartbeat blind spot | **ACCEPT** | atomic write / locks / auth independently consistent with what I read |
| signals-modules | crypto_macro OPTIONS_TTL use-before-def NameError (only finding) | **REFUTED** | runtime global lookup inside function body; defined at module level at import; crypto_macro is a live 54.5%-acc signal — see 06 doc |

---

## C. Cross-cutting themes (emergent — only visible across subsystems)

### C1. The critical-error / alerting layer is the noisiest, least-trustworthy part of the system
**All three seeded critical-error categories are false-positive or mislabeled:**
- `contract_violation` → timestamp-provenance race (A1)
- `accuracy_degradation` → regime shift, not degradation (signals-core P1)
- `avanza_account_mismatch` → session expiry mislabeled (avanza-api root cause)
Consequence: 31 "unresolved" criticals that are mostly noise → operator can't
triage real failures, and the auto-spawn fix-agent dispatcher burns Claude budget
on phantom problems. **This is the #1 systemic finding.** Fixes: (a) accurate
category naming (session-expiry ≠ account-mismatch), (b) regime-gate the accuracy
alert, (c) auto-resolve stale entries when a fix post-dates them, (d) make the
journal the single source of truth for every Layer-2 terminal status (write a
stub on auth_error too — orchestration P0-#1).

### C2. Atomic writes are solid; cross-process read-modify-write is NOT
`file_utils.atomic_write_json` is verified correct (infra). But every money-state
mutation does load→mutate→write with only an in-process `threading.Lock`, while
CLAUDE.md documents 3+ independent OS processes (main loop, metals loop, Layer 2
subprocess, dashboard) writing the same files. Lost-update class confirmed in:
warrant_portfolio (P0-1), portfolio_mgr (P0-2), trade_guards TOCTOU (P2-4), and
signal_engine shared caches. **Fix:** a single cross-process file-lock primitive
(`file_utils.jsonl_sidecar_lock` already exists) wrapped around every money-state
read-modify-write.

### C3. Silent-failure / fail-open is the dominant bug shape
auth_error writes no journal stub; grid_fisher swallows AvanzaSessionError into a
generic placement failure; data caches serve stale silently; risk gates compute
but don't enforce. The system consistently prefers "look healthy / do nothing"
over "fail loud", which is exactly what hid the 3-week 2026 auth outage. Every
fix above should err toward a loud, accurately-named critical.
