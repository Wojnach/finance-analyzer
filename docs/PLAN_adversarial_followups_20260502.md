# Plan — Adversarial Review Follow-ups (2026-05-02)

## Context

Three recent dual adversarial reviews:
- `docs/ADVERSARIAL_REVIEW_2026-04-17.md`
- `docs/ADVERSARIAL_REVIEW_2026-04-24.md`
- `docs/ADVERSARIAL_REVIEW_2026-04-29.md`
- `docs/ADVERSARIAL_REVIEW_2026-05-01.md` (synthesis, committed yesterday in `eadd094b`)

Audit: which findings were actioned, which are still outstanding, which are
intentionally deferred, which are outdated/false positives.

## Disposition Matrix (2026-05-01 synthesis)

### P0 — Critical

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| P0-1 | `ticker_accuracy.py` no neutral filter | OUTSTANDING | Confirmed: lines 53-62 still lack `_vote_correct`. **WILL FIX.** |
| P0-2 | `ic_computation.py:19` relative `Path("data")` | OUTSTANDING | Confirmed: still relative. **WILL FIX.** |
| P0-3 | `signal_history.py` no lock | OUTSTANDING | Confirmed: `update_history` is read-modify-write with no lock. **WILL FIX.** |
| P0-4 | `avanza_client.py` TOTP order bypasses lock | OUTSTANDING | Confirmed: `_place_order` line 326 has no lock. **WILL FIX.** |
| P0-5 | drawdown circuit breaker bare except | OUTSTANDING | Confirmed: `agent_invocation.py:402` swallows all exceptions. **WILL FIX.** |
| P0-6 | `equity_curve.py:384` excludes fees from pnl | OUTSTANDING | Confirmed: line 384 has gross pnl. Reported `fee_sek` separately but `pnl_sek` is not net. Defer to next batch (gold-plating concern: changes downstream profit_factor calculations). |

### P1 — Important (15 findings)

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| P1-1 | `_rescued` not reset per loop iteration | OUTSTANDING | Confirmed but very low-impact — the gate-fail/else paths almost always reset _rescued. Defensive 1-line fix safe. Defer (cosmetic). |
| P1-2 | persistence cold-start seeds at MIN_CYCLES | FALSE POSITIVE | Already disproven by 04-24 review FP-2. Cycle-2 behavior identical with seed=1 vs seed=MIN. No-op fix. |
| P1-3 | Layer 2 bypasses claude_gate | INTENTIONALLY_DEFERRED | Architectural — known design choice (subprocess.Popen pattern). |
| P1-4 | `volume_flow.py` VWAP cumulative not session | OUTSTANDING | Confirmed: `_compute_vwap` line 61 uses cumsum over entire df. Defer (semantic redesign). |
| P1-5 | `heikin_ashi.py:317` Alligator shift stale | FALSE POSITIVE | Disproven by 04-24 FP-1 — `pandas .shift(8)` is correct Williams Alligator. |
| P1-6 | `mean_reversion.py:462` seasonality compounds | OUTSTANDING | Confirmed: line 472 reads modified df. Metals-only path. Defer. |
| P1-7 | `macro_regime.py:203` yield threshold 1.5 too high | OUTSTANDING | Confirmed: `change_5d > 1.5` = 150bps. Defer (config decision). |
| P1-8 | metals stop too tight | OUTSTANDING | HARD_STOP_UNDERLYING_PCT=2.0. Already noted in user feedback memory. Defer. |
| P1-9 | metals exit optimizer hardcoded usdsek=10.85 | OUTSTANDING | Confirmed: line 2823. **WILL FIX.** |
| P1-10 | `avanza_orders.py` CONFIRM races | OUTSTANDING | Defer (user-facing UX change). |
| P1-11 | `kelly_sizing.py:139` uses system-wide accuracy | OUTSTANDING | Confirmed. Defer (per-ticker accuracy already computed elsewhere; needs plumbing). |
| P1-12 | `trade_guards.should_block_trade` never called | OUTSTANDING | Confirmed: only called from tests. Defer (architectural). |
| P1-13 | `fear_greed.py` IndexError on empty data | OUTSTANDING | Confirmed: `body["data"][0]` unguarded. **WILL FIX.** |
| P1-14 | `onchain_data.py` skips _coerce_epoch | OUTSTANDING | Confirmed but already wrapped in try/except — fails to None gracefully. Defer. |
| P1-15 | FX fallback to 1.0 false circuit breaker | OUTSTANDING | Confirmed: `risk_management.py:99`. Defer (related to P0-6 — broader rework needed for fx caching). |

## Disposition Matrix (2026-04-29 dual review)

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| SC-P1-1 | `_get_regime_gated` regime/horizon union bug | OUTSTANDING | **Confirmed: signal_engine.py:881-882 still uses replace, not union. Same bug as 04-24 P0-1. NEVER FIXED.** **WILL FIX.** |
| SC-P1-2 | Circuit-breaker high-sample relaxation | OUTSTANDING | Defer (signal-engine behavior change). |
| SC-P1-3 | outcome_tracker backfill races | OUTSTANDING | Defer (file I/O semantics). |
| OR-P1-1 | Layer 2 bypasses claude_gate | DEFERRED | Same as 05-01 P1-3. |
| OR-P1-2 | Zero-delay spin after crash | OUTSTANDING | Defer. |
| OR-P1-3 | Set in trigger state dict | DOWNGRADED | Per same review's cross-critique (DG-3 equivalent). |
| PR-P1-1 | Warrant avg-in not updating underlying_entry | OUTSTANDING | Defer (warrant subsystem). |
| PR-P1-2 | Peak cache no thread lock | OUTSTANDING | Defer. |
| AV-P1-1 | Wrong DELETE URL in fin_fish_monitor | OUTSTANDING | Defer (script, not loop). |
| AV-P1-2 | delete_stop_loss missing order lock | OUTSTANDING | Defer (related to P0-4 fix). |
| AV-P1-3 | Telegram CONFIRM not sender-authenticated | OUTSTANDING | Defer (security). |
| DE-P1-1 | Funding rate KeyError | OUTSTANDING | Defer. |
| DE-P1-2 | Fear&Greed IndexError | OUTSTANDING | Same as 05-01 P1-13 — fix in this batch. |
| DE-P1-3 | sys.path injection | DOWNGRADED | Per review (P2). |
| DE-P1-4 | joblib.load unsafe | DOWNGRADED | Per review (P3). |
| SM-P1-1 | news_event "cut" keyword | OUTSTANDING | Defer (signal logic). |
| SM-P1-2 | futures_flow operator precedence | OUTSTANDING | Defer (signal logic). |
| SM-P1-3 | volume_flow VWAP cumulative | OUTSTANDING | Same as 05-01 P1-4. |
| SM-P1-4 | cot_positioning CWD-relative path | OUTSTANDING | Defer. |
| MC-P1-1 | Stop-loss from entry not bid on orphan | OUTSTANDING | Defer (orphan ingest path). |
| MC-P1-2 | HW stop tighter than SW stop | OUTSTANDING | Defer (config). |
| MC-P1-3 | pos_id collision (dup of partial fix) | OUTSTANDING | Defer (rare race). |
| MC-P1-4 | Zero-price sell on price fetch fail | OUTSTANDING | Defer (metals_swing_trader is large). |
| IN-P1-2 | Dashboard cache thundering herd | DOWNGRADED | Per review (P2). |
| IN-P1-3 | telegram_poller raw open() | OUTSTANDING | Defer. |

## Plan: Top 5 Fixes To Ship

Selected by severity, blast radius, and ease of testing:

1. **`_get_regime_gated` regime/horizon union** (signal_engine.py:881-882)
   - REJECTED on closer audit: replace-semantics is intentional (BUG-149,
     2026-03-29). Funding 74.2% @3h_ranging means we WANT it to vote at
     3h ranging even though it's in `_default` (which targets 1d/3d/5d
     where funding is 29.9%). Trend 61.6% @3h vs 40.7% @1d ranging — same
     pattern. Adversarial reviewer compared to `_get_horizon_disabled_signals`
     (which DOES union) but the two structures have different intent.
   - Outcome: docstring updated to document the intentional semantics, and
     a 4-test regression class added to prevent re-flagging.

2. **`ic_computation.py` relative path** (line 19)
   - Silently disables IC weights when CWD differs from repo root
   - 1-line fix + targeted test

3. **`signal_history.py` add lock** (`update_history`)
   - 5-worker thread pool corrupts history JSONL
   - Add `threading.Lock` + targeted test

4. **`agent_invocation.py` drawdown bare except** (line 402)
   - Default to BLOCK on check failure (fail-safe instead of fail-open)
   - Wrap call narrowly + targeted test

5. **`fear_greed.py` IndexError guard** (line 100)
   - Crashes signal on empty API response (alternative.me maintenance)
   - 4-line guard + targeted test

## Out of scope (this batch)

- P0-1 (ticker_accuracy neutral filter): worth doing but adds 1-line per call site. Defer to next batch — needs a separate test for direction_probability.
- P0-4 (avanza_client TOTP lock): deeper integration — needs to wrap multiple call sites. Defer.
- P0-6 (equity_curve fees): correctness fix but cascades to profit_factor / Sharpe / Sortino — needs tests across `equity_curve` math. Defer.
- All P1 findings except P1-9/P1-13 (already in top 5): defer to backlog.

## Test plan

For each fix:
- Write a targeted test in the appropriate `tests/test_*.py`.
- Run that test file in isolation to confirm pass.
- After all 5 fixes, `pytest --collect-only` to verify no import-time breakage.

## Worktree

`/mnt/q/finance-analyzer-adversarial-followups` on branch
`fix/adversarial-followups-20260502`.

## Commit cadence

One commit per finding with severity in subject:
- `fix(signal_engine): P0 — regime gating union (was replace) at horizon override`
- `fix(ic_computation): P0 — absolute DATA_DIR path (was relative)`
- `fix(signal_history): P0 — add threading.Lock to update_history`
- `fix(agent_invocation): P0 — fail-safe drawdown circuit breaker`
- `fix(fear_greed): P1 — guard IndexError on empty API response`
