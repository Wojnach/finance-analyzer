# FGL Independent Adversarial Pass (main-thread, not subagent)

Date: 2026-05-28. Reviewer: orchestrating agent (own deep read of the highest-stakes
hot paths, run in parallel with the 8 subsystem subagents for cross-critique).

Scope chosen: the paths where I most wanted an independent verdict — the live
`contract_violation` (Layer-2 silent-failure) incident, the atomic-I/O foundation that
every subsystem depends on, and the dashboard auth surface. Findings below are
confirmed against current `main` code.

---

## IND-1 — P1 (proximate root cause of the recurring `contract_violation` incident)

`portfolio/agent_invocation.py:1448-1457` (timeout branch of `_check_agent_completion_locked`)
and `_kill_overrun_agent` `portfolio/agent_invocation.py:655-737`.

The "incomplete" path (exit 0, no journal) was explicitly fixed to append a **stub journal
entry** (`agent_invocation.py:1583-1598`) so the `loop_contract.layer2_journal_activity`
watchdog is satisfied. The **timeout path does NOT**: the early return at 1448-1457 sets
`journal_written: False` and writes nothing; `_kill_overrun_agent` only calls
`_log_trigger(..., "timeout")` (→ `invocations.jsonl`) and `_scan_agent_log_for_auth_failure`.
It never appends to `layer2_journal.jsonl`.

Consequence: every timed-out Layer 2 run (T1 180s / T2 600s / T3 900s) leaves the
journal-activity watchdog with nothing to reconcile → the recurring
`contract_violation: "Layer 2 trigger fired Nm ago but no journal entry has been written
since. Agent may be failing silently."` — 37 such entries 2026-05-21..28 in
`data/critical_errors.jsonl`. The alert is largely a **false "silent failure"** caused by
the asymmetry, masking any genuine silent failure in the same noise.

Fix: mirror the incomplete-path stub write in the timeout branch / `_kill_overrun_agent`
with `status="timeout"`; AND have `loop_contract` consult `invocations.jsonl` status
(timeout/failed/incomplete already recorded) before escalating to "failing silently".
Separate follow-up (not this gap): investigate WHY T2/T3 agents time out so often.

## IND-2 — P1 — cross-process lost-update on shared JSON state

`portfolio/portfolio_mgr.py:29,35-41,136-159`. `update_state` serializes read-modify-write
with a `threading.Lock` — in-process only. `portfolio_state.json` / `_bold.json` are mutated
by **three separate OS processes**: `main.py` (Layer 1 loop), the Layer 2 `claude -p`
subprocess (writes trades), and the dashboard. Across processes the threading lock is a
no-op → lost updates (last os.replace wins; a Layer 2 trade can be silently clobbered by a
concurrent Layer 1 save). `file_utils.atomic_write_json` (`file_utils.py:53-71`) provides
**write-atomicity** (os.replace) but not RMW serialization, and `jsonl_sidecar_lock` exists
for JSONL but has **no JSON-state equivalent**. Worse, `save_state`/`load_state`
(`portfolio_mgr.py:121-133`) bypass even the threading lock — any `load_state → mutate →
save_state` caller has zero serialization.

Fix: gate JSON-state RMW with a cross-process sidecar lock (reuse the `jsonl_sidecar_lock`
/ `process_lock` pattern keyed on the state path); funnel all mutations through
`update_state`; or stop letting the Layer 2 subprocess write portfolio state directly
(return decisions to Layer 1 to apply under one writer).

CROSS-CRITIQUE of portfolio-risk subagent: it concluded "`update_state` is
concurrency-safe under a per-file lock." True within one process, **false across the
L1 / L2-subprocess / dashboard boundary** — the dominant real-world contention here.

## IND-3 — P2 — `load_json` swallows OSError → default masks a locked critical file as empty state

`portfolio/file_utils.py:86-90`. On OSError (Windows file lock held by another writer /
antivirus) `load_json` returns `default`. `portfolio_mgr._load_state_from` only enters
backup-recovery when load returns `None` **and** the file exists; a *locked* file returns
`default`, so for callers whose default is `{}`/`[]` a transient lock reads as
authoritative "no holdings / flat". Same failure class as the Avanza
"empty-because-auth-failed misread as flat" risk (see avanza-api review).

Fix: distinguish transient-lock from missing — on OSError, brief retry then raise or
return an explicit `UNKNOWN` sentinel callers must not treat as empty.

## IND-4 — P3 — unset `dashboard_token` = fully open incl. POST mutations

`dashboard/auth.py:123-125`. When `dashboard_token` is unset, `require_auth` allows ALL
requests (backwards-compat), including POST endpoints (`/api/validate-portfolio`) on the
dual-stack all-interfaces bind. Fine only if a token is guaranteed in prod (config is
external, unverifiable here). Fix: fail closed, or bind loopback-only when no token set.

NOT A BUG (verified, to suppress false positives): `cf_access.py` does real RS256 JWT
signature verification against CF JWKs with aud/exp/iat + email-claim/header match (the
2026-05-13 header-spoof P0 is properly fixed); cookie/query/bearer paths all use
`hmac.compare_digest`. No auth-bypass finding on the hardened paths.

---

## Cross-critique log (subagent claims I checked)
- portfolio-risk: "update_state concurrency-safe under per-file lock" → only within-process; cross-process lost-update stands (IND-2).
- infrastructure (caveman) P0 `file_utils.py:240` "sidecar lock creation race → corruption": **FALSE POSITIVE / DOWNGRADE.** The seed is idempotent in *outcome* — whether the race yields a 1-byte or 2-byte lock file, any ≥1-byte file is lockable (content never matters; only a lockable byte must exist, which the docstring states explicitly). No corruption.
- infrastructure (caveman) P0 `health.py:156` "heartbeat staleness falsely reports not-stale": **OVERSTATED → P2.** A naive `last_heartbeat` makes the *uncaught* subtraction at `health.py:165` raise `TypeError` (the try/except wraps only `fromisoformat`, not the subtraction) — a crash to the caller, NOT a silent "not stale". And current `heartbeat()` writes tz-aware ISO, so it's latent. Real fix is still worth it: normalize tz / wrap the subtraction. Reclassify P2.
- infrastructure (caveman) other P0s (`process_lock.py:103` truncate-then-write non-atomic; `log_rotation.py:349` SQLite -wal/-shm not checkpointed; `log_rotation.py:510` text rotation no sidecar lock): plausible, not yet independently verified — keep as P1 "verify" in synthesis.
- avanza-api P0 `avanza_session.py:426` (API error → None, caller can't tell empty-balance from 401) and P0 `avanza_account_check.py:225` (expiry treated as transient): **CROSS-CONFIRM my IND-3** — the "empty-because-auth-failed misread as flat/empty" theme is real and spans file_utils.load_json + the avanza layer. These two + `avanza_session.py:139` (expiry not re-verified on context reuse) are the credible root cause of the daily `avanza_account_mismatch: Session expired` critical-error stream since 2026-05-23. HIGH confidence, promote in synthesis.
- avanza-api P1 "TOTP plaintext in config.json" and P1 "confirm token case-sensitive": severity hot — TOTP-in-config is the documented design (config.json is the external secrets file, never committed), so P2/P3 hardening at most; case-sensitive confirm token is P3. Downgrade in synthesis.
- avanza-api P0 `avanza_session.py:139` (session not re-verified on context reuse): **CONFIRMED by direct read** — `_get_playwright_context` early-returns the cached `_pw_context` and only calls `load_session()` on first creation; `is_session_expiring_soon()` (line 123) exists but is never consulted on reuse. A long-running loop holds a stale context across the ~24h BankID expiry → the daily `avanza_account_mismatch: Session expired` stream since 2026-05-23. **HEADLINE root cause.**

### Correction to IND-1 (orchestration agent found the dominant mechanism)
The orchestration subagent root-caused the live `contract_violation` more precisely than my IND-1:
**primary** = `loop_contract.py:410` compares `journal_ts >= last_trigger` with ZERO tolerance, but journal `ts` is stamped at cycle-start (whole-second `...Z`) while `last_trigger_time` is written later at microsecond precision → 233 *successful* runs (status=success, journal_written=True) in 7d trip the CRITICAL. My IND-1 (timeout path writes no stub) is the **secondary** real gap, which the agent extended: `timeout`/`failed` are also absent from `_KNOWN_FAILURE_STATUSES`, so a genuine hang/timeout would still be an undetected silent gap. BOTH fixes needed: (a) add skew tolerance / trigger-reason match to the contract; (b) write a timeout stub AND register timeout/failed as known failure statuses. Net: the alert is mostly false-positive noise that *also* masks the real silent-failure case — worst of both.

## CROSS-CUTTING THEME (confirmed across ≥4 subsystems) — "empty/None/stale read as valid/live/flat"
The single most pervasive systemic risk. A failed fetch / expired session / locked file returns an empty or default value that a downstream consumer treats as authoritative:
- `avanza_session.py:426` API error → None, caller can't tell empty-balance from 401 (P0).
- `avanza_account_check.py:225` session expiry treated as transient outage (P0).
- `file_utils.py:86-90` `load_json` OSError → default `{}` masks a locked state file as empty (IND-3, P2).
- `sentiment.py:854` empty headlines → "unknown" 0.0-confidence vote, doesn't disable the voter, dilutes consensus (data-external P0).
- `metals_cross_assets.py:56` `fetch_klines` exception swallowed → empty DataFrame, no staleness flag (data-external P0).
- `price_source.py:214` silent yfinance fallback → ~10-min-stale price passed as live (data-external P1).
- `fx_rates.py:44` out-of-band rate not cached → P&L silently uses fallback 10.50, 10-15% off (data-external P1).
Systemic fix direction: introduce an explicit `UNKNOWN`/`STALE` sentinel distinct from "empty/zero/flat", and make consumers (voters, sizing, valuation, position checks) fail-closed (skip/HOLD/halt) on UNKNOWN rather than treating it as a tradeable zero.
