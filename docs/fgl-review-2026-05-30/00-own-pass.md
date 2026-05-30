# FGL Adversarial Review — Independent Pass (main thread)

**Date:** 2026-05-30
**Reviewer:** Claude (orchestrator), independent of the 8 subsystem subagents
**Scope:** crown-jewel foundations (atomic I/O, locking, shared state, Layer-2
invocation→journal path) + cross-cutting themes that span subsystem boundaries
and are therefore invisible to any single-subsystem reviewer.

Severity scale: P0 production-breaking · P1 serious bug · P2 robustness/limited
blast radius · P3 quality.

---

## A. Foundation findings (read directly)

### A1 — `contract_violation` root cause (P0) — CONFIRMED by code + dual analysis
**Files:** `portfolio/main.py:964,989`, `portfolio/agent_invocation.py:426-434`,
`portfolio/loop_contract.py:344-388`.

The recurring critical error `layer2_journal_activity :: trigger fired but no
journal entry written` (~20×/week, 2026-05-26..29) is a **false positive caused
by invocation-status clobbering**:

1. `invoke_agent()` has ≥6 internal early-returns that each write their OWN
   status row to `invocations.jsonl` via `_log_trigger` — `skipped_gate`,
   `blocked_drawdown_{patient,bold}`, `blocked_drawdown_unavailable`,
   `blocked_trade_guards`, `skipped_no_position`, `specialist_quorum_fail` — then
   `return False`.
2. Back in `main.py:989`, the caller unconditionally re-logs
   `_log_trigger(reasons, "invoked" if result else "skipped_busy")`. The
   escalation path at `main.py:964` re-logs the **dynamic** string
   `skipped_busy_{why}`.
3. `skipped_busy` (and any `skipped_busy_*`) is the *most-recent*
   `invocations.jsonl` entry, and it is **deliberately excluded** from the
   contract's `_LEGITIMATE_SKIP_STATUSES` (loop_contract.py:353 — excluded on
   purpose because `skipped_busy` also covers real failure paths).
4. The contract finds no journal entry within grace → fires `CRITICAL`
   `contract_violation`.

So a perfectly legitimate skip (perception gate, drawdown block, no-position
skip) is reported as a silent Layer-2 failure. The genuinely dangerous case
(`status="failed"`, exit≠0, no stub) is buried in the same noise.

**Cross-validation:** the `orchestration` subagent reached this independently
(its P0 #1) from the opposite direction (caller side). I reached it from the
producer/consumer-contract side. Convergence → near-certain.

**Fix:** have `invoke_agent` return a status string (or a sentinel meaning
"already logged a terminal status"); only emit the outer `skipped_busy` when the
inner path did NOT already log. Belt-and-braces: add the internal skip statuses
to `_LEGITIMATE_SKIP_STATUSES` and add `"failed"` to `_KNOWN_FAILURE_STATUSES`
(so real crashes still surface, distinctly).

### A2 — `atomic_write_jsonl` bypasses the sidecar lock (P2)
**File:** `portfolio/file_utils.py:295-313`.

`atomic_append_jsonl` (269) and `prune_jsonl` (379) both take
`jsonl_sidecar_lock`, but `atomic_write_jsonl` (full rewrite → `os.replace`) does
**not**. A concurrent `atomic_append_jsonl` that lands between a caller's
"compute entries" and `atomic_write_jsonl`'s `os.replace` is silently clobbered —
the exact `signal_log_reconciliation` divergence class the sidecar lock was
introduced to kill, just on a different writer. **Fix:** wrap
`atomic_write_jsonl`'s tmp-write+replace in `jsonl_sidecar_lock(path)`.

### A3 — in-memory rate/quota counters reset on crash-restart (P2) — CONVERGES with data-external
**Files:** `portfolio/shared_state.py:312-344` (NewsAPI), `alpha_vantage.py`
(25/day), limiters at `shared_state.py:297-305`.

Every daily budget counter is a module global with no disk persistence. The main
loop auto-restarts on crash (exponential backoff per CLAUDE.md). A crash loop, or
several restarts in a day, silently re-grants the full NewsAPI (100/day) and
Alpha Vantage (25/day) budgets → real provider quota blown → key throttled/banned
→ a *data* outage that looks like "API flaky". **Fix:** persist
`{count, reset_date}` via `atomic_write_json`, reload on startup. (The
`data-external` subagent flagged the same independently — keep as one finding.)

### A4 — `process_lock` metadata rewrite truncates the locked byte range (P3)
**File:** `portfolio/process_lock.py:101-105`.

After `msvcrt.locking(fileno, LK_NBLCK, 1)` locks byte [0,1), `_write_lock_metadata`
does `seek(0); truncate(); write(...)`. Truncating the very byte range that is
range-locked is fragile on Windows (works in practice today, but a future
`LK_LOCK`-blocking variant or a larger lock range would deadlock/raise). Lock
correctness does not depend on the metadata, so impact is low. **Fix:** write
metadata to a sidecar `.meta` file, or lock a byte *past* the metadata region.

### A5 — `jsonl_sidecar_lock` silently yields with NO lock if neither msvcrt nor fcntl (P3)
**File:** `portfolio/file_utils.py:251-258`.

If both `_msvcrt` and `_fcntl` import as None, the `with` block `yield`s without
acquiring anything — a silent correctness downgrade (torn lines return). Can't
happen on the Windows target, but it's a silent-failure shape. **Fix:** raise (as
`process_lock._lock_file` correctly does) rather than yield unlocked.

### A6 — no parent-dir fsync after `os.replace` (P3, Windows-N/A)
**File:** `portfolio/file_utils.py:67`. The docstring promises power-loss
durability (H34); on POSIX that needs an fsync of the parent directory fd after
the rename. Moot on the Windows production host (NTFS `os.replace` ordering), but
the durability claim is stronger than the code on Linux/WSL test runs.

---

## B. Cross-cutting meta-themes (span ≥3 subsystems — synthesis-level)

These are the highest-value output: each is a *pattern* repeated across
subsystems that no single-subsystem reviewer could see as systemic.

### B1 — Non-atomic read-modify-write of state files (the atomic primitives exist but hot paths bypass them)
The codebase HAS correct atomic helpers (`file_utils.atomic_write_json`,
`portfolio_mgr.update_state` with a per-file lock) — but several live write paths
**don't use them**:
- `portfolio_mgr.update_state` (full-cycle lock) is called **nowhere** in
  production; the real trade path uses bare `load_state()`/`save_state()`
  (portfolio-risk P2, main.py:449/796) → non-atomic RMW.
- `warrant_portfolio.record_warrant_transaction` does unlocked load→mutate→save
  (portfolio-risk P1) → concurrent fast-tick monitor + main metals cycle can lose
  a transaction; the oversell clamp then reads stale units.
- `cusum_accuracy_monitor.update_cusum` rewrites the whole state file per outcome
  under a *thread*-only lock (signals-core P3) → loop + PF-OutcomeCheck race,
  last-writer-wins drops counters.
- Even where the in-process `threading.Lock` is held, it does NOT guard against
  the **three processes** (main loop, metals loop, dashboard) that write the same
  files — only `os.replace` atomicity + the sidecar lock protect cross-process.

**Synthesis recommendation:** make `update_state`-style locked RMW the *only*
sanctioned mutate path, add a cross-process file lock around it, and route warrant
+ cusum writes through it. This single structural fix closes a whole class of
silent state-loss bugs.

### B2 — Silent stale-data fallbacks masking a dead source (violates "live prices first")
The project's #3 critical rule is "Live prices first. Never base analysis on
cached/precomputed data." Yet the same anti-pattern recurs:
- `price_source.fetch_klines` silently substitutes 10–15-min-stale yfinance on
  ANY primary failure, no staleness tag (data-external **P0**).
- `onchain_data.get_onchain_data` serves ≤24h-old cache at DEBUG when the token
  is missing (data-external P2).
- `futures_data._fetch_json` counts a fatal 4xx (bad key) as a transient
  breaker trip → permanent auth failure looks like flakiness (data-external P3) —
  *the exact shape of the 3-week 2026 silent auth outage*.
- `metals_cross_assets._yf_download` swallows outages into empty frames → "no
  data" indistinguishable from "flat market" (data-external P3).

**Synthesis recommendation:** a single `StaleData`/`source+age` envelope returned
by every fetcher, with signal consumers down-weighting or HOLDing on stale/fallback
data, and fatal-vs-transient typing enforced at the one `fetch_json` boundary.

### B3 — Producer/consumer contract drift (status strings & schemas diverge across files)
Two files that must agree on a vocabulary silently drift:
- invoke_agent skip statuses vs `loop_contract._LEGITIMATE_SKIP_STATUSES` (B1/A1
  above) — the headline P0.
- `claude_gate._count_today_invocations` filters on `entry["timestamp"]` but
  `_log_trigger` writes `entry["ts"]` (orchestration P3) → daily Claude-usage
  counter under-counts.
- `SignalDB` SQL accuracy methods omit the ±0.05% neutral band that the Python
  `_vote_correct` applies (signals-core P1) → any dashboard/report calling the
  SQL path shows different accuracy than the live gate uses.

**Synthesis recommendation:** centralize the status vocabulary and the
neutral-zone/scoring logic in one module each; make the consumers import from the
producer rather than re-declaring constants.

---

## C. Cross-critique seeds (adversarial pass over the subagents' own findings)
(Full cross-critique in the synthesis doc; seeds captured here.)

- **infra P1 "`_cached` `_loading_keys` leak on interrupt"** → **OVERSTATED,
  downgrade to P3.** The un-covered window is between the `with _cache_lock`
  exit (shared_state.py:89) and the `try:` (line 91) — effectively zero
  statements — and any leak self-heals via the 120s stuck-key eviction
  (lines 68-74). The `try/except` already covers `KeyboardInterrupt` and
  `Exception` around `func()`.
- **infra P1 "atomic_append_jsonl loses appends despite fsync (two processes)"**
  → **largely MITIGATED, downgrade to P3.** `atomic_append_jsonl` holds
  `jsonl_sidecar_lock`, which *serializes* appenders cross-process — only one
  holds the lock at a time, so the "both fsync concurrently, one lost" scenario
  cannot occur *through the helper*. Real residual risk only if a writer bypasses
  the helper (which B1 already targets).
- **portfolio-risk P0 (negative warrant value)**, **data-external P0
  (price_source)**, **infra P0×2 (false-healthy heartbeat)**, **avanza-api P0×2**
  → spot-validated as REAL, keep at stated severity.
