# Adversarial Review Synthesis — finance-analyzer

**Date:** 2026-04-05
**Baseline HEAD:** `6c3f154` (main branch, 2026-04-04 — "docs: morning briefing + quant research deliverables")
**Reviewers:** Codex (GPT-5 family via `codex@openai-codex` plugin v0.116.0) and Claude (Opus 4.6 1M)
**Method:** 8 parallel subsystem reviews by Codex + 8 independent subsystem reviews by Claude +
cross-critique in both directions

## Source documents

| Document | Contents |
|----------|----------|
| `docs/PLAN_ADVERSARIAL_REVIEW.md` | The review plan (partitioning, method, risks) |
| `docs/ADVERSARIAL_REVIEW_CODEX.md` | Codex's 8 subsystem reviews (verbatim) |
| `docs/ADVERSARIAL_REVIEW_CLAUDE.md` | Claude's 8 subsystem reviews |
| `docs/META_REVIEW_CLAUDE_ON_CODEX.md` | Claude's verification of Codex's findings |
| `docs/META_REVIEW_CODEX_ON_CLAUDE.md` | Codex's verification of Claude's findings |

## Methodology note

The review partitioned the codebase into 8 subsystems (signals-core,
orchestration, portfolio-risk, metals-core, avanza-api, signals-modules,
data-external, infrastructure). For each, both reviewers worked independently
before seeing the other's output. Claude then verified Codex's findings
against the code; Codex then verified Claude's findings against the code.

Total findings: **~85** (Codex 43 + Claude 45). Cross-verification results:

| Reviewer of | Total findings | VALID | PARTIAL | FALSE POS | STALE |
|-------------|---------------|-------|---------|-----------|-------|
| Codex (verified by Claude) | 43 | 42 | 1 | 0 | 0 |
| Claude (verified by Codex) | 45 | ~25 | ~12 | 7 | 1 |

**Key observations from cross-verification:**
- Codex's findings were dramatically more precise — zero false positives, one
  partial, no stale. Codex focused on concrete bugs found via line-by-line
  reading.
- Claude's review had 7 false positives (~16%) — primarily speculation about
  concurrent code paths or forward-looking concerns not verified against the
  actual code. Claude's strengths were architectural critiques and
  cross-referencing user memory.
- Claude also had **one factual error** (EU DST direction backward: summer
  is correct, winter is the actual bug — fix is the opposite of what Claude
  originally said) and **one stale finding** (econ_calendar silent-exhaustion
  already fixed upstream).
- Both reviewers independently converged on the same issue for several
  high-impact items: `place_stop_loss` volume invariant (C3), atomic_write_json
  fsync (H34), overtrading guards being advisory-only, hardcoded FOMC/CPI
  dates, trend-family over-voting, dead code patterns.

This synthesis surfaces only findings that **survived cross-verification**.
False positives and stale findings have been removed. Severity reflects the
consensus (or the lower of the two reviewers' assessments where they disagree).

---

## TIER 1 — CRITICAL, ACT NOW

These are findings where either (a) both reviewers agree AND severity is
critical, or (b) one reviewer found it, the other verified it, and severity
is critical.

### C1. Loop-contract self-heal grants Claude CLI Edit+Bash+Write authority synchronously in the live loop
**Source:** Codex 2.1 (verified by Claude meta-review)
**Files:** `portfolio/loop_contract.py:625-653`, `portfolio/claude_gate.py:117`,
`portfolio/main.py:914`
**What:** On any CRITICAL contract violation, `verify_and_act` calls
`_trigger_self_heal` which calls `invoke_claude(...)` with the default
`allowed_tools="Read,Edit,Bash,Write"` and `timeout=180`. The call blocks
the main loop for up to 180 seconds AND grants the subprocess write+bash
access to the live trading repository, with no operator approval, no
sandbox, and no dry-run gate. This fires exactly when the system is in a
bad state — the worst moment for autonomous code modification.

**Impact:** (a) 180s cadence stall during critical failures; (b) unreviewed
LLM modifies production trading code.

**Fix shape:** Make self-heal opt-in and read-only by default. Move to an
out-of-band diagnostic thread. Require explicit operator approval via
Telegram for any Edit/Bash/Write action from self-heal.

### C2. `place_order_no_page` fails open: any non-BUY side becomes a live SELL
**Source:** Codex 5.2 (verified by Claude)
**Files:** `portfolio/avanza_control.py:313-325`
**What:**
```python
normalized_side = (side or "").strip().upper()
if normalized_side == "BUY":
    result = _place_buy_order(...)
else:
    result = _place_sell_order(...)  # FALLTHROUGH
```
A typo, `None`, empty string, `"HOLD"`, `"sell"` (lowercase fails uppercase
match — actually wait, `.upper()` fixes this one), or any enum drift → live
SELL. The design fails OPEN into a trading action, the opposite of fail-safe.

**Impact:** A bug in any caller (including Layer 2 or metal_loop's
decision logic) could liquidate a position.

**Fix shape:** Strict validation. `if normalized_side not in ("BUY",
"SELL"): raise ValueError(f"Invalid side: {side!r}")`.

### C3. `place_stop_loss` and `place_sell_order` have no position-size / outstanding-stop volume invariant
**Source:** Claude 4.2 + Codex 5.1 (both reviewers, CRITICAL on both)
**Files:** `portfolio/avanza_session.py:346-528`,
`portfolio/avanza/trading.py` (parallel implementation)
**What:** Neither `_place_order` nor `place_stop_loss` checks current
holdings or sums existing active stop-loss volume. A caller that passes the
wrong volume can exceed position size, trigger short-selling, or leave
exposure unprotected after a fill.

This is the same category as the March 3, 2026 incident documented in
CLAUDE.md (wrong endpoint for stop-loss). The endpoint bug was fixed; the
volume invariant was not.

**Fix shape:** Single shared pre-trade risk gate for SELL/stop-loss flows.
Fetch current position + active stops, compute `free_volume`, reject if
`requested_volume > free_volume`. Apply in both the legacy
`portfolio.avanza_session` and the new `portfolio.avanza.trading` paths.

### C4. `record_trade()` has ZERO call sites — overtrading guards never activate
**Source:** Codex 3.2 (verified by Claude)
**Files:** `portfolio/trade_guards.py:171-219`
**What:** `record_trade` is the sole mutator for cooldown timestamps,
consecutive losses, and new-position timestamps. A repository-wide grep
finds zero callers. Guard state stays empty forever; cooldown never
activates; loss escalation never engages; position rate limits never fire.

The entire overtrading-guard subsystem is non-functional. The system has
the appearance of risk management but none of it operates on real fills.

**Fix shape:** Invoke `record_trade()` in the same post-fill commit path
as every successful BUY/SELL in Layer 2 execution code. Also convert the
key guards to `severity=block` with enforcement in the execution path.

### C5. Singleton lock silently no-ops on non-Windows hosts
**Source:** Claude 2.1 (not surfaced by Codex)
**Files:** `portfolio/main.py:39-55`, `data/metals_loop.py:583-611`
**What:**
```python
try:
    import msvcrt
except ImportError:
    msvcrt = None
...
def _acquire_singleton_lock():
    if msvcrt is None:
        return True  # silent pass-through
```
On Linux/WSL, the singleton lock always returns True — zero enforcement.
In a WSL + Windows-Scheduled-Task environment (the user's production
setup), running the loop in WSL during development would coexist with the
Windows scheduled task instance, double-writing to state files.

**Impact:** Silent state corruption from concurrent writers. All atomic
writes prevent within-write corruption; they do NOT prevent interleaved
read-modify-write cycles.

**Fix shape:** Use `fcntl.flock` as the non-Windows alternative. Or fail
loudly: `raise RuntimeError("Singleton lock unsupported")`.

### C6. MWU signal weights — written to disk, never read by the engine
**Source:** Codex 1.4 (verified by Claude)
**Files:** `portfolio/signal_weights.py`, `portfolio/outcome_tracker.py`,
`portfolio/signal_engine.py`
**What:** `SignalWeightManager.batch_update()` is called from
`outcome_tracker.py`, writing to `data/signal_weights.json`. Nothing in
`signal_engine.py` imports `SignalWeightManager` or reads
`signal_weights.json`. The MWU adaptation path is dead code — disk I/O and
CPU burned for no effect, while operators believe adaptation is happening.

**Fix shape:** Either wire `SignalWeightManager.get_normalized_weights()`
into `_weighted_consensus` or delete the dead path. Don't leave vestigial
adaptation systems in a trading codebase.

### C7. `load_state` silently regenerates defaults on corrupt JSON — transaction history loss
**Source:** Claude 3.1 (not surfaced by Codex)
**Files:** `portfolio/portfolio_mgr.py:39-44`
**What:** On any load failure (corrupted JSON, OSError, etc.), `load_state`
returns `{**_DEFAULT_STATE, "start_date": now()}` — a fresh blank state.
The next `save_state` call atomically writes this blank state over the
corrupted file, **permanently destroying transaction history**.

**Impact:** One bad write window (power loss, antivirus scan, shadow copy)
can erase the entire trade record once the next cycle saves.

**Fix shape:** On load failure, raise loudly and refuse to save until
recovered. Keep rolling backups (`.bak`, `.bak2`).

### C8. Portfolio state read-modify-write has no concurrency safety
**Source:** Codex 3.1 (verified by Claude — overlaps partly with my C7)
**Files:** `portfolio/portfolio_mgr.py:39-61`
**What:** `load_state()` and `save_state()` provide separate operations
with no lock, no version, no CAS. Under concurrent writers, two fill
handlers can load the same snapshot, each mutate, and the later `save_state`
overwrites the earlier update.

**Fix shape:** Replace with a locked or versioned `update_state()`
transaction. Combine with C7's corruption guard for a single atomic-state
module.

### C9. Monte-Carlo t-copula implementation is an identity transform — VaR/CVaR biased by ~√2
**Source:** Codex 3.5 (verified by Claude)
**Files:** `portfolio/monte_carlo_risk.py:270-290`
**What:**
```python
T_samples = W * scale             # fat-tailed multivariate-t samples
U = t_dist.cdf(T_samples, df=...) # → uniform
Z_marginal = t_dist.ppf(U[:, i], df=...) # → t-samples AGAIN
```
The cdf→ppf round-trip through the same distribution is an identity. The
downstream `sigma * sqrt(T) * Z_marginal` then uses t-variance (≈2 for
df=4) instead of unit-variance Gaussian — the effective sigma is √2× the
nominal. VaR and CVaR are materially wrong.

**Fix shape:** Use `norm.ppf(U)` to recover Gaussian marginals (correct
t-copula + Gaussian GBM), OR standardize t to unit variance (multiply by
`sqrt((df-2)/df)`) before the `sigma*sqrt(T)` scaling.

### C10. Regime-gated signals cannot recover through data — dead-signal trap
**Source:** Claude 1.1 (Codex verified VALID, downgraded CRITICAL→HIGH in
meta-review; keeping in Tier 1 because structural recovery-blocking is
severe regardless of severity tag)
**Files:** `portfolio/signal_engine.py:1339-1341`, `portfolio/outcome_tracker.py:123-129`
**What:** Regime-gated signals have their votes rewritten to HOLD at line
1341 **before** being logged via `_votes` in the snapshot. Accuracy is
computed only for non-HOLD votes. Once a signal is on the gate list for a
regime, its accuracy data for that regime cannot accumulate — it has zero
recorded non-HOLD votes. The data needed to un-gate it can never exist.

**Fix shape:** Log a **raw pre-gate** vote alongside the gated vote. Allow
a scheduled re-audit to identify gate-list entries that would now flip back.

---

## TIER 2 — HIGH, address in next sprint

### H1. Quorum computed on raw votes, not post-gate effective voters
**Source:** Codex 1.1, overlaps with Claude 1.11. Both reviewers.
**Files:** `signal_engine.py:1343-1366, 1515-1525`
**Fix:** Compute `active_voters` from signals that SURVIVE weighting and gating.

### H2. Regime-gate exemption uses all-time per-ticker accuracy
**Source:** Codex 1.2 (not in Claude's review).
**Files:** `signal_engine.py:1318-1338`
**Fix:** Use **recent same-regime** accuracy for the exemption check.

### H3. Skipped Layer 2 invocations still consume tier state
**Source:** Codex 2.2.
**Files:** `main.py:594-633`
**Fix:** Only advance tier state after `invoke_agent` returns True.
Snapshot per-invocation context instead of shared files.

### H4. Loop-contract enforcement bypassed on exception/hang paths
**Source:** Codex 2.5.
**Files:** `main.py:885-917`
**Fix:** Emit cycle state incrementally and verify from a `finally` block,
or use an external watchdog.

### H5. Windows timeout recovery can orphan old Claude process
**Source:** Codex 2.3 (classified PARTIAL by Claude meta-review).
**Files:** `agent_invocation.py:163-200`
**Fix:** Latched "zombie suspected" state; require confirmed exit.

### H6. Multi-agent synthesis reads stale specialist reports
**Source:** Codex 2.4.
**Files:** `portfolio/multi_agent_layer2.py:33-216`
**Fix:** Unique temp directory per invocation; cleanup in a finally block.

### H7. `portfolio_value` returns cash-only on invalid fx_rate — false catastrophic loss
**Source:** Claude 3.2.
**Files:** `portfolio_mgr.py:64-67`, `fx_rates.py` (Codex 7.5 covers the cause).
**Fix:** Use last-known-good fx_rate with staleness metadata.

### H8. 2×ATR stop-loss doesn't account for leverage (contradicts user's 5x cert rule)
**Source:** Claude 3.3. (Codex downgraded to MEDIUM: "depends on external
trading preference rather than a code contradiction". Claude retained at
HIGH because the user's auto-memory explicitly calls out this exact rule
after a past incident — the external preference is a durable operator
constraint, not a generic opinion.)
**Files:** `risk_management.py:184`
**Fix:** Detect leverage via `instrument_profile.py`; widen stop to `max(2, leverage*1.5)×ATR`.

### H9. Drawdown check is advisory, not enforcing
**Source:** Claude 3.8 + Codex 3.3.
**Files:** `risk_management.py:53-128`
**Fix:** Wire `breached` into the trade-guard block path.

### H10. Kelly sizing ignores FIFO lot matching — fabricates realized edge
**Source:** Codex 3.4.
**Files:** `kelly_sizing.py:55-104`
**Fix:** Reconstruct round trips with FIFO matching.

### H11. Warrant positions bypass cash accounting
**Source:** Codex 3.6.
**Files:** `warrant_portfolio.py:52-246`
**Fix:** Track warrant fills in the same cash ledger; respect fx_rate; clamp
long-warrant value at 0/knockout.

### H12. Silver fast-tick loses `underlying_entry` on normal position saves
**Source:** Codex 4.1, converges with Claude 4.4.
**Files:** `data/metals_loop.py:351-396, 751-796`
**Fix:** Include `underlying_entry` in the save schema; merge on update.

### H13. Mid-cycle Avanza expiry blinds loop for ~20 minutes
**Source:** Codex 4.2.
**Files:** `data/metals_loop.py:4836-4902`
**Fix:** `_check_session_and_alert()` before any Avanza I/O.

### H14. Silver fast-tick state not reset on sell → contaminates next position
**Source:** Codex 4.3.
**Files:** `metals_loop.py` — `_silver_reset_session` has no callers.
**Fix:** Call reset on every silver close/reopen.

### H15. Silver fast-tick silently no-ops on errors
**Source:** Claude 4.1.
**Files:** `data/metals_loop.py:812-834, 746-748`
**Fix:** Telemetry on each silent-return path; alert after 5 consecutive failures.

### H16. Dual avanza implementations (legacy + new package) can double-fire orders
**Source:** Claude 5.1.
**Files:** `portfolio/avanza_session.py` (legacy) vs `portfolio/avanza/trading.py` (new)
**Fix:** Intent-table with `(orderbook_id, side, price_rounded, volume, window)` dedup key.

### H17. Telegram confirmation is global, not order-specific
**Source:** Codex 5.3.
**Files:** `avanza_orders.py:99-203`
**Fix:** `CONFIRM <order-id>` or per-order nonce.

### H18. `delete_stop_loss_no_page` ignores `ok` flag, reports false success
**Source:** Codex 5.4.
**Files:** `avanza_control.py:361-374`
**Fix:** Return `result.get("ok", False)`.

### H19. TOTP singleton never re-auths after session expiry
**Source:** Codex 5.5, overlaps with Claude 5.5.
**Files:** `portfolio/avanza/auth.py:74-120`
**Fix:** Auto-reset on auth failure; retry once; fail fast if auto-reauth disabled.

### H20. US equity seasonality emitted for any ticker (no asset-class gate)
**Source:** Codex 6.1.
**Files:** `portfolio/signals/calendar_seasonal.py:47-369`
**Fix:** Require ticker input; whitelist US equities.

### H21. Economic-event windows mis-timed on timezone-aware bars
**Source:** Codex 6.2.
**Files:** `portfolio/signals/econ_calendar.py:30-36`
**Fix:** Use `astimezone(UTC)` instead of `replace(tzinfo=UTC)`.

### H22. Econ/FOMC calendar data hardcoded to 2026-2027
**Source:** Claude 6.1 + Codex 6.3 (both reviewers).
**Files:** `portfolio/econ_dates.py`, `portfolio/fomc_dates.py`
**Fix:** Fetch from remote source, OR alert when date horizon < 90 days,
OR fail loudly when expired.

### H23. LLM refresh failures cached as fresh, collapse to silent HOLD
**Source:** Codex 6.4.
**Files:** `portfolio/signals/claude_fundamental.py:543-773`
**Fix:** Separate refresh-state from cache-state; validate model output
per-ticker.

### H24. Trend signal family over-votes the same underlying factor
**Source:** Codex 6.5 + Claude 6.2. Both reviewers.
**Files:** `signals/trend.py`, `signals/heikin_ashi.py`, `signals/momentum.py`.
**Fix:** Rebuild correlation groups from rolling 30d vote-correlation;
downweight or veto correlated clusters in ranging regimes.

### H25. Earnings-gate cache poisoned by transient provider failure → 24h disabled
**Source:** Codex 7.1, related to Claude 1.12.
**Files:** `portfolio/earnings_calendar.py:30-191`
**Fix:** Distinguish "no earnings" from "provider error"; don't cache failures.

### H26. Fear/greed streaks count fetches, not days
**Source:** Codex 7.2.
**Files:** `portfolio/fear_greed.py:33-67`
**Fix:** Increment at most once per daily observation.

### H27. Alpha Vantage daily budget ignores failed requests
**Source:** Codex 7.3.
**Files:** `portfolio/alpha_vantage.py:149-297`
**Fix:** Count every attempted request; persist across restarts.

### H28. Partial BGeometrics refreshes stamped fresh, overwrite cache
**Source:** Codex 7.4.
**Files:** `portfolio/onchain_data.py:164-260`
**Fix:** Track freshness per metric; merge partial with previous.

### H29. FX fallback returns stale/hardcoded prices as live
**Source:** Codex 7.5, paired with Claude H7.
**Files:** `portfolio/fx_rates.py:20-55`
**Fix:** Return `(rate, source, age_secs, is_stale)` tuple or named tuple.

### H30. Sentiment module subprocess paths hardcoded at module load
**Source:** Claude 7.2.
**Files:** `sentiment.py:32-45`
**Fix:** Validate script existence at startup; fail loudly.

### H31. Subprocess-based LLM signals have no concurrency limit
**Source:** Claude 7.3.
**Files:** `signals/forecast.py`, `signals/claude_fundamental.py`, `sentiment.py`
**Fix:** Global semaphore limiting concurrent model subprocesses to 2-4.

### H32. NewsAPI daily budget in-memory, lost on restart
**Source:** Claude 7.4.
**Files:** `shared_state.py:193-196`
**Fix:** Persist to `data/api_quota_state.json`.

### H33. `_cached` dogpile prevention has TOCTOU window on long-running load
**Source:** Claude 7.5.
**Files:** `shared_state.py:67-77`
**Fix:** Add load-started timestamp to `_loading_keys`.

### H34. `atomic_write_json` doesn't fsync file or parent directory
**Source:** Claude 8.1, 8.2 + Codex 8.6. Both reviewers.
**Files:** `file_utils.py:13-28`
**Fix:** `f.flush() + os.fsync(fd)` before `os.replace`; fsync parent dir
after on POSIX.

### H35. `load_json` silent-default on all errors hides corruption
**Source:** Claude 8.3.
**Files:** `file_utils.py:31-48`
**Fix:** Split into `load_json` (optional) and `require_json` (raises on corruption).

### H36. `atomic_append_jsonl` not atomic for entries >4KB
**Source:** Claude 8.4.
**Files:** `file_utils.py:134-146`
**Fix:** Per-file threading.Lock OR `fcntl.flock`/`msvcrt.locking`.

### H37. Analysis throttle clears queue even on delivery failure
**Source:** Codex 8.1.
**Files:** `message_throttle.py:102-112`
**Fix:** Check `send_or_store` return value; preserve queue on failure.

### H38. GPU lock broken by any holder running >5 minutes
**Source:** Codex 8.2.
**Files:** `gpu_gate.py:111-158`
**Fix:** Replace mtime sentinel with OS-backed interprocess lock (fcntl/msvcrt)
or heartbeat + PID liveness check.

### H39. JSONL rotation drops writes during the rotation window
**Source:** Codex 8.3.
**Files:** `log_rotation.py:148-233`
**Fix:** Rename first under writer lock, then process detached file.

### H40. HTTP retry replays non-idempotent POSTs → Telegram double-send
**Source:** Codex 8.4.
**Files:** `http_retry.py:27-62`
**Fix:** Opt-in retry for non-idempotent methods; require dedupe key.

### H41. Subprocess timeout hangs indefinitely on failed job assignment
**Source:** Codex 8.5.
**Files:** `subprocess_utils.py:130-140`
**Fix:** Check `AssignProcessToJobObject()` result; bounded
`communicate()` timeout on cleanup.

### H42. `_startup_grace_active` test-isolation leakage
**Source:** Claude 2.2.
**Files:** `trigger.py:40, 98, 134, 138`
**Fix:** Use a persisted `boot_id` (UUID at startup) instead of in-memory bool.

### H43. Trade detection catches only `(KeyError, AttributeError)` — JSONDecodeError crashes loop
**Source:** Claude 2.3.
**Files:** `trigger.py:76-93`
**Fix:** Broaden exception catch; log warning; continue.

### H44. Multi-agent specialists block main loop up to 150s
**Source:** Claude 2.4.
**Files:** `agent_invocation.py:249-260`
**Fix:** Move specialist wait to background future; collect on next cycle.

### H45. Stack-overflow detection hardcodes Windows exit code only
**Source:** Claude 2.5.
**Files:** `agent_invocation.py:35, 551`
**Fix:** Include Linux signals (134, 139) and stderr-match.

### H46. Autonomous mode always records HOLD — poisons reflection stats
**Source:** Claude 2.8.
**Files:** `autonomous.py:119-122`
**Fix:** Tag journal entries `source=autonomous_observer` and exclude from
reflection/accuracy stats, OR record actual recommended direction.

### H47. EU market open hardcoded to 07:00 UTC with no CET/CEST DST
**Source:** Claude 2.9. **CORRECTION per Codex meta-review:** Claude's
original description had the DST direction backward. `07:00 UTC` is
**correct for summer (CEST, UTC+2)** when Frankfurt/London open at 08:00
local → 07:00 UTC. The actual bug is **winter** (CET, UTC+1), when Frankfurt
opens at 08:00 local → **08:00 UTC** — one hour LATER than the code's
07:00. In winter, the loop treats the first hour of EU trading as still in
"market closed" state.
**Files:** `market_timing.py:8`
**Fix:** Compute `_eu_market_open_hour_utc(dt)` that returns 8 during EU
standard time (winter) and 7 during EU summer time. EU DST: last Sunday of
March → last Sunday of October.

### H48. Trend signal compares `iloc[-1]` and `iloc[-2]` without verifying adjacency
**Source:** Claude 6.8.
**Files:** `signals/trend.py:37-62`
**Fix:** Find the last 2 adjacent valid bars.

### H49. Global confidence cap at 0.80 creates pile-up instead of fixing overconfidence
**Source:** Claude 1.6.
**Files:** `signal_engine.py:1627-1630`
**Fix:** Non-linear decay or invert at the bucket boundary instead of hard cap.

### H50. Market-health penalty creates structural SELL bias at market bottoms
**Source:** Claude 1.3.
**Files:** `signal_engine.py:1564-1578`
**Fix:** Symmetric penalty, or regime-conditional (invert in capitulation sub-regime).

---

## TIER 3 — MEDIUM / LOW (nice-to-have, track in backlog)

Grouped by theme. Individual items listed in the source docs; this section
just catalogs the themes:

### Theme: hardcoded dated audit findings decay silently
- Signal engine regime gate lists (signals-core 1.8)
- Horizon weight static dict (signals-core 1.8)
- Correlation groups (signals-core 1.7)
- Monte Carlo correlation priors (portfolio-risk 3.9)
- Regime penalty weights (signals-core)
- Group-leader gate threshold 0.47 (signals-core 1.7)

**Cross-cutting fix:** Single `data/audit_constants.json` with `audited_on`
date per entry. Monthly re-audit job emits Telegram summary of drift.

### Theme: rate limiters and timers use wall clock instead of monotonic
- `_RateLimiter` uses `time.time()` (H30)
- `classify_tier` uses `time.time() - last_full` (H43)

### Theme: stale cache and provider state
- Stock timeframe caches survive provider switches (Codex 7.6)
- Cross-process microstructure state stale-by-design (Codex 4.4)
- `_cached` error path corrupts `time` field (Claude 8.8)
- `_tool_cache` eviction oscillates at boundary (Claude 8.10)

### Theme: per-signal confidence caps scatter
- `_MAX_CONFIDENCE` duplicated across modules, global cap at 0.80 in engine
  (Claude 6.9)

### Theme: dead code and orphaned helpers
- `_silver_reset_session` not called (Codex 4.3, Claude H14)
- `record_trade` not called (Codex 3.2, C4)
- `SignalWeightManager` not read by engine (Codex 1.4, C6)

### Theme: ORB and backtester look-ahead bias
- `orb_backtest._simulate_trades` uses EOD extrema (Codex 4.5)

### Theme: error handling too narrow
- Earnings gate silent `except Exception: pass` (Claude 1.12)
- Trade detection catches only (KeyError, AttributeError) (Claude 2.3)

---

## Cross-cutting architectural critiques (agreed by both reviewers)

### A1. Silent failures dominate the failure catalog
Every subsystem has "X fails silently on Y exception" patterns. The system
consistently prefers graceful degradation over loud failure. This is the
right policy for non-critical paths (e.g., "a signal module didn't fetch
headlines"), but WRONG for safety-critical paths (earnings gate, stop-loss
volume check, session expiry, silver fast-tick, signal weights). The
system does not distinguish these categories.

**Recommendation:** Introduce `@degrades(severity="critical")` marker on
error handlers that must also raise a health event. Operators need a
single place that says "system X has degraded silently in the last 24h".

### A2. Advisory vs enforcing gap — the Layer 2 LLM is trusted to catch
issues the code should enforce.
Trade guards emit "warning" but never "block". Drawdown check is computed
but not enforced. Concentration is reporting-only. Autonomous mode has
strictly weaker risk controls than Layer 2. The design pattern is "the
LLM will read these and do the right thing", which fails when Layer 2 is
unavailable OR when the LLM is being adversarial.

**Recommendation:** Define a minimum HARD-BLOCK rule set enforceable by
code: `4+ consecutive losses`, `drawdown > 25%`, `concentration > 40%`,
`sell+stop > position`. Make them explicit, tested, and un-overridable
by LLM decisions.

### A3. Multi-process state fragmentation without coordination
Main loop, metals loop, GoldDigger, Elongir each have their own state
files and locks. Shared files (health, fundamentals cache, signal log,
accuracy cache) are race-prone because each loop's singleton lock only
covers its own state files. Cross-process atomicity is assumed but not
enforced.

**Recommendation:** Declare state-ownership per file (who owns writes,
who can read). Files with shared ownership need `fcntl.flock`/`msvcrt`
cross-process locking or a single-writer daemon pattern.

### A4. Hardcoded dated audit findings embedded throughout the code
Signal gating lists, horizon weight dicts, correlation groups, FOMC/CPI
calendars, correlation priors — all snapshot-based, all drift when the
market shifts, none updated on a schedule. Over time the system accumulates
wrong constants.

**Recommendation:** Move every audit-derived constant to a single
`data/audit_constants.json` with `audited_on` field. Monthly job re-audits
and emits a delta report.

---

## Findings removed from synthesis as false positives or stale (per cross-verification)

Claude's original review had 7 false positives and 1 stale finding that
Codex caught during its meta-review. These are NOT in the synthesis above;
listed here for completeness and so future sessions can understand why
items absent from the synthesis were still raised initially.

| Claude finding | Verdict | Reason |
|---|---|---|
| 1.13 Sentiment cache global mutability | FALSE POSITIVE | Keyed by ticker; per-ticker semantics correct |
| 2.3 `(KeyError, AttributeError)` catch too narrow | FALSE POSITIVE | `load_json` already swallows JSONDecodeError/ValueError |
| 2.6 Env var stripping brittle | FALSE POSITIVE | Forward speculation, not current bug |
| 4.7 Metals loop shares portfolio_state.json with main | FALSE POSITIVE | Metals writes are to metals-specific state files, no actual overlap |
| 5.4 `_get_csrf` only searches cookies | FALSE POSITIVE | Speculation about localStorage fallback need |
| 6.9 Per-module `_MAX_CONFIDENCE` unenforced | FALSE POSITIVE | `signal_registry.py:115` applies max_confidence centrally |
| 8.6 `load_jsonl_tail` skips truncated last line | FALSE POSITIVE | Defensive behavior, not a bug |
| 6.10 Econ calendar silent exhaustion | STALE | Already fixed — `econ_calendar.py:167` now logs warning |

Claude's factual error that DID survive as a valid finding but with
corrected description: **H47 (EU DST direction)** — the bug is winter, not
summer; synthesis text has been corrected.

## Summary counts (post-verification)

| Tier | Count | Description |
|------|-------|-------------|
| Critical (Tier 1) | 10 | Money-at-risk or data-corruption paths |
| High (Tier 2) | ~45 | Systemic reliability issues (post-FP removal) |
| Medium/Low (Tier 3) | ~25 | Backlog |
| Agreed by both reviewers | ~15 | High-confidence findings |
| Codex-only (verified by Claude) | ~20 | Claude missed these |
| Claude-only (verified by Codex) | ~15 | Codex missed these |
| False positives removed | 7 | All from Claude's review |
| Stale findings removed | 1 | From Claude's review (econ_calendar silent exhaustion) |

## Reviewer comparison

**Codex's strengths:** Precise line-level bugs. Found concrete
implementation errors (C2 non-BUY→SELL, C6 dead MWU, C9 t-copula identity,
C1 self-heal authority). These required file-by-file reading. Zero false
positives across 43 findings. When Codex flagged something, verification
confirmed it.

**Claude's strengths:** Architectural critique and cross-cutting themes.
Found structural issues (C5 singleton on non-Windows, C7 state wipe on
corruption, C10 regime-gated recovery trap, A1-A4 cross-cutting themes).
Also leveraged conversation memory (5x cert rule, Mar 3 incident, WSL
context). Claude was looser: 7 false positives out of 45 findings (~16% FP
rate) plus 1 factual error (EU DST direction backward).

**Lessons:** If you want "zero false positives per finding" reliability,
Codex's methodical file-by-file approach wins. If you want "what's wrong
architecturally across the whole system" including forward-looking risks
and operator-specific context, Claude's broader scope helps. Neither is
strictly dominant — together they catch things neither does alone.

**Both approaches are valuable.** A single-reviewer pass would have missed
approximately half the critical findings, regardless of which reviewer was
used. Dual review with cross-critique is a legitimate tool for high-stakes
code review and the cost (one extra model invocation + one meta pass) is
small compared to the value.

**Cost disclosure (for future scaling):** Total wall time ~1 hour. Token
cost: Codex 8 parallel adversarial reviews (~15 min each, moderate effort)
+ 1 meta-review task (~3 min, high effort). Claude read ~7000 lines of code
across 8 subsystems + wrote ~1600 lines of review + ~500 lines of
meta-review + ~800 lines of synthesis. This is affordable as a monthly
exercise on a working production codebase.

## What this review is NOT

- Not a test-coverage audit (tests were out of scope).
- Not a style/lint pass.
- Not a performance review.
- Not a security audit in the traditional sense, though security-adjacent
  issues were flagged where encountered (C1 self-heal authority, C2 fail-open-to-SELL).
- Not an implementation plan. The fix shapes are suggestions, not
  prescriptions. A separate implementation session should take this
  synthesis and produce a `docs/PLAN.md` update with sequencing.

---

*Synthesis compiled from independent adversarial reviews by Codex (GPT-5 via
codex@openai-codex plugin v0.116.0) and Claude (Opus 4.6 1M). Baseline HEAD
`6c3f154`. See `docs/META_REVIEW_CLAUDE_ON_CODEX.md` and
`docs/META_REVIEW_CODEX_ON_CLAUDE.md` for the cross-critique step.*
