# Improvement Plan — Auto Session 2026-04-24

Based on deep exploration by 6 parallel agents (orchestration, signal engine, data/portfolio,
metals/Avanza, reporting/infra, tests/docs) plus manual verification of all P0/P1 findings.

Previous session (2026-04-23) fixed: BUG-219 pnl_pct pass-through (P0), rate limiter slot
reservation (P1), drawdown NaN guard (P1), cache None prevention (P2), orphan subprocess
logging (P2).

## Exploration Summary

### Confirmed Bugs

- **BUG-220: outcome_tracker base_price=None → phantom 0% changes**
  `outcome_tracker.py:364-389` — When `base_price = tickers[ticker].get("price_usd")`
  returns None (ticker present in signal_log but not in current price feed), `change_pct`
  stays at its default `0.0` and the outcome is stored with `"change_pct": 0.0`. This
  pollutes accuracy stats with phantom zero-change entries that count as correct outcomes
  (BUY with 0% change = miss, but HOLD with 0% = hit — biasing HOLD accuracy upward).
  - **Severity**: P2 — accuracy data pollution, not loop-crashing
  - **Fix**: Skip the entry when `base_price is None or base_price <= 0` (continue to
    next horizon). Already has `hist_price is None` guard at line 382; this mirrors it
    for base_price.

- **BUG-221: daily_digest timezone exception uncaught**
  `daily_digest.py:68` — `zoneinfo.ZoneInfo(tz_name)` where `tz_name` comes from
  `config["notification"]["daily_digest_tz"]`. A bad config value (typo, empty string)
  raises `ZoneInfoNotFoundError` which propagates uncaught, crashing the digest check
  and potentially disrupting the main loop's post-cycle tasks.
  - **Severity**: P2 — config-driven crash, `tzdata` is installed so normal values work
  - **Fix**: Wrap in try/except, fall back to UTC on failure, log warning.

- **BUG-222: fin_snipe_manager _notify_critical swallows all send failures**
  `fin_snipe_manager.py:98-102` — The `try/except` around `send_or_store` catches all
  exceptions silently (only has a `pass` in the original pattern). If Telegram delivery
  fails for critical alerts (session_expired, naked_position), the failure is invisible.
  - **Severity**: P3 — single-threaded context reduces impact, but critical alerts must
    not silently fail
  - **Fix**: Add `logger.warning()` in the except block.

### Architecture Improvements

- **ARCH-1: outcome_tracker uses raw signal_log.jsonl parsing**
  `outcome_tracker.py` loads the full JSONL file on every run (can be 50K+ lines).
  The SignalDB SQLite backend exists and is populated in parallel, but outcome_tracker
  still reads raw JSONL as primary source.
  - **Impact**: P3 — performance, not correctness. Defer to backlog.

- **ARCH-2: Signal count documentation drift**
  `CLAUDE.md` says "33 active signals (36 modules registered, 3 force-HOLD)" but actual
  state in `tickers.py` is 45 signal names, 16 in DISABLED_SIGNALS, giving 29 active.
  `SYSTEM_OVERVIEW.md` similarly outdated.
  - **Impact**: P2 — documentation accuracy affects Layer 2 decisions
  - **Fix**: Update docs in Phase 4.

### Verified Non-Bugs (False Positives from Agents)

- **max_confidence caps**: Agent reported caps not enforced. VERIFIED: `signal_engine.py:2706-2707`
  correctly reads `entry.get("max_confidence", 1.0)` and passes to `_validate_signal_result()`.
- **EWMA neutral weight**: Agent reported neutral weight never applied. VERIFIED: lines 280-287
  correctly use `ewma_weight` in the fallback path.
- **trigger.py:138 first-run default**: `prev_count = last_checked_tx.get(label, current_count)`
  defaults to current count on first run. This is CORRECT — avoids false trigger on startup.
- **metals_loop check_session_alive**: Agent reported undefined. VERIFIED: imported at line 342
  from `portfolio.avanza_control`.
- **fin_snipe_manager race condition**: Agent reported P1 dict race on `_critical_alert_last`.
  VERIFIED: module is single-threaded (no threading imports, not imported by metals_loop).
  Downgraded to P3.

---

## Implementation Batches

### Batch 1: BUG-220 outcome_tracker base_price guard (2 files, ~10 lines)

**Files**: `portfolio/outcome_tracker.py`, `tests/test_outcome_tracker.py`

1. In `outcome_tracker.py:364-389`, add guard after line 364:
   ```python
   base_price = tickers[ticker].get("price_usd")
   if base_price is None or base_price <= 0:
       continue  # skip — no base price to compute change_pct
   ```
   Move `base_price` fetch inside the horizon loop so each ticker is checked once,
   and skip ALL horizons for that ticker when base_price is missing.

2. Write test: `test_outcome_backfill_skips_none_base_price` — mock tickers dict with
   `{"price_usd": None}` entry, verify outcome is NOT stored with 0% change.

### Batch 2: BUG-221 daily_digest tz guard + BUG-222 fin_snipe alert logging (3 files, ~15 lines)

**Files**: `portfolio/daily_digest.py`, `portfolio/fin_snipe_manager.py`,
`tests/test_daily_digest.py`

1. In `daily_digest.py:67-68`, wrap `ZoneInfo(tz_name)` in try/except:
   ```python
   try:
       tz = zoneinfo.ZoneInfo(tz_name)
   except (KeyError, zoneinfo.ZoneInfoNotFoundError):
       logger.warning("Unknown timezone %r, falling back to UTC", tz_name)
       tz = UTC
   now_local = datetime.now(tz)
   ```

2. In `fin_snipe_manager.py`, add logging in the `_notify_critical` except block
   (around line 103) so failed critical alerts are visible.

3. Write test: `test_daily_digest_bad_timezone_fallback` — verify bad tz string
   doesn't crash, falls back to UTC behavior.

### Batch 3: Documentation updates (3 files, ~50 lines)

**Files**: `CLAUDE.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPROVEMENT_PLAN.md`

1. Update signal counts in `CLAUDE.md`:
   - "36 modules registered, 3 force-HOLD" → "45 modules registered, 16 force-HOLD"
   - "33 active signals" → "29 active signals"
   - Update the signal list sections to reflect actual state

2. Update `docs/SYSTEM_OVERVIEW.md` signal inventory section.

3. This plan document itself gets committed.

---

## Backlog (deferred — not this session)

- **ARCH-1**: Migrate outcome_tracker from JSONL to SignalDB SQLite queries
- **TEST-HYGIENE-2**: `test_llama_server_job_object.py` tests unimplemented feature
- **fin_snipe_manager**: Consider adding threading.Lock if module ever gets called
  from threaded context
- **outcome_tracker**: JSONL open-then-parse is O(n) on every run; cap at last N entries
  or use seek-from-end
- **Loop contract**: `_check_l2_journal_activity` grace period could be configurable
  per-tier instead of hardcoded

## Dependency Ordering

Batch 1 → Batch 2 → Batch 3

No cross-dependencies between batches, but sequential ordering keeps commits clean
and makes bisection straightforward. Each batch gets its own test run and commit.
