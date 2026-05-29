# Improvement Plan — 2026-05-29 Auto-Session

**Branch:** `improve/auto-session-2026-05-29`
**Sources:** 6 parallel exploration agents + FGL adversarial review (2026-05-28) synthesis + manual verification.
**Pre-existing fixes:** 17 bugs from commit `77b43289` (2026-05-28 bug-hunt) + heartbeat fix `e0e84271`.

---

## 1. Bugs & Problems Found

### BATCH 1 — Critical (money/reliability/silent-failure)

| # | File | Line(s) | Bug | Impact | FGL Ref |
|---|------|---------|-----|--------|---------|
| B1-1 | `portfolio/loop_contract.py` | 372 | `_KNOWN_FAILURE_STATUSES` missing `"timeout"`, `"failed"`, `"stack_overflow"` — genuine hang/timeout is invisible to the contract checker | Real silent failures hidden inside false-positive noise | §B |
| B1-2 | `portfolio/loop_contract.py` | 410 | Zero-skew journal/trigger timestamp comparison — journal stamps at whole-second, trigger at microsecond precision → 233 false CRITICAL violations in 7 days | Operator alert fatigue, fix-agent dispatcher waste | §B |
| B1-3 | `portfolio/agent_invocation.py` | timeout-kill path | Timeout-kill writes no journal stub (unlike `incomplete` path at 1583-1598) | A genuine T1/T2/T3 hang leaves zero trace in journal | §B |
| B1-4 | `portfolio/risk_management.py` | 373-374 | ATR stop `entry*(1-2*atr/100)` has no distance floor — on low-ATR silver certs the stop lands ~2% away, violating "never stop within 3% of bid" rule | Money — repeat of Mar 3 instant-fill incident class | §3 |
| B1-5 | `portfolio/kelly_sizing.py` | 91-104 | One global avg_buy_price scored against every SELL (no FIFO, look-ahead) → wrong win-rate/payoff into Kelly → mis-sized recs | Wrong sizing on every rec | §4 |
| B1-6 | `portfolio/signal_engine.py` | blend_accuracy_data | `at.get("accuracy")` can return None from corrupt cache → TypeError crash in `abs(rc_acc - at_acc)` | Consensus computation crash, forced HOLD for all tickers | Agent BUG-5/7 |
| B1-7 | `data/metals_loop.py` | main() | Fatal crash exits code 0 (`main()` returns None → `sys.exit(None)`) — supervisor can't distinguish crash from clean stop | Exact exit-0-on-failure class CLAUDE.md warns about | §8 |
| B1-8 | `portfolio/signals/sentiment_extremity_gate.py` | 144+ | No ticker guard — runs on XAU/XAG/MSTR using crypto alt.me F&G index (wrong sentiment series) | Wrong signal for metals/stocks | FGL signals-modules P1 |

### BATCH 2 — Reliability

| # | File | Line(s) | Bug | Impact |
|---|------|---------|-----|--------|
| B2-1 | `data/metals_loop.py` | while-True body | Entire cycle body in one try — uncaught raise kills the loop instead of skipping one cycle | Violates "loop runs 100%" priority |
| B2-2 | `portfolio/forecast_accuracy.py` | 154 | `sub_name.split("_", 1)[1]` for horizon suffix — multi-part names (e.g. "daily_ma_1h") parsed wrong | Sub-signals silently under-counted in accuracy |
| B2-3 | `dashboard/app.py` | 73 | Path leaked in 500 error response | Info disclosure |
| B2-4 | `dashboard/app.py` | 1073 | Exception string reflected in POST validation error | Info disclosure |
| B2-5 | `portfolio/message_store.py` | ~190 | `set(tg_cfg.get("muted_categories", []))` — if config value is string, set() iterates chars | Mute gate misbehaves on bad config |
| B2-6 | `portfolio/signal_engine.py` | 2776 | Dead-zone soft votes × favorable accuracy/regime/IC can exceed strong-vote weights | Soft votes dominate consensus contrary to design |
| B2-7 | `portfolio/signal_registry.py` | 19 | No validation of max_confidence bounds — out-of-range values propagate | Normalization break in consensus |
| B2-8 | `portfolio/trade_guards.py` | 166 | ValueError from corrupt timestamp silently swallowed with no warning log | Cooldown check skipped silently |

### BATCH 3 — Signal Quality & Polish

| # | File | Line(s) | Bug | Impact |
|---|------|---------|-----|--------|
| B3-1 | `portfolio/signal_engine.py` | 1591-1593 | Horizon weights cached 1h — not invalidated after accuracy backfill | Stale weights for up to 60 min post-backfill |
| B3-2 | `portfolio/signal_engine.py` | 4142 | 3d/5d/10d horizons collapse to 1d accuracy when per-horizon data missing | Wrong-horizon edge on long-horizon votes |
| B3-3 | `portfolio/equity_curve.py` | 244-246 | Sharpe/Sortino std recomputed redundantly (O(n) waste) | Performance only |
| B3-4 | `portfolio/signal_utils.py` | 100 | `count_hold` parameter in majority_vote never used | Dead code confusion |

---

## 2. Architecture Improvements

### Deferred (too risky for autonomous session)

| Item | Why Deferred |
|------|-------------|
| **Cross-process JSON-state lock** (Theme 3) | Needs design doc, multi-process testing, careful integration — not tactical |
| **Theme 1 sentinel + fail-closed** | Systemic change across all data consumers — large blast radius |
| **Theme 2 dead controls wiring** | `portfolio_validator` at save boundary, `trade_validation` at execution — needs integration testing with live portfolios |
| **Avanza session re-validation on reuse** | Requires manual BankID testing, Playwright session lifecycle |

---

## 3. Execution Plan

### Batch 1: Critical Bug Fixes (8 files)
**Files:** `loop_contract.py`, `agent_invocation.py`, `risk_management.py`, `kelly_sizing.py`, `signal_engine.py`, `metals_loop.py` (main exit code), `sentiment_extremity_gate.py`, + tests

**Order:**
1. Write failing tests for B1-1, B1-2, B1-3, B1-6, B1-7, B1-8
2. Fix loop_contract (B1-1 + B1-2) — surgical, 2 changes
3. Fix agent_invocation timeout journal stub (B1-3) — ~20 lines
4. Fix risk_management ATR floor (B1-4) — add 3% min distance
5. Fix kelly_sizing FIFO (B1-5) — reuse equity_curve._pair_round_trips
6. Fix signal_engine None-handling (B1-6) — use _safe_accuracy guard
7. Fix metals_loop exit code (B1-7) — sys.exit(1) on crash
8. Fix sentiment_extremity_gate ticker guard (B1-8) — add CRYPTO_TICKERS check
9. Run tests, commit

### Batch 2: Reliability (8 files)
**Files:** `metals_loop.py` (cycle wrap), `forecast_accuracy.py`, `dashboard/app.py`, `message_store.py`, `signal_engine.py` (soft-conf cap), `signal_registry.py`, `trade_guards.py`, + tests

**Order:**
1. Write tests for B2-1 through B2-8 where missing
2. Fix metals_loop per-cycle isolation (B2-1) — wrap cycle body
3. Fix forecast_accuracy rsplit (B2-2)
4. Fix dashboard info disclosure (B2-3, B2-4)
5. Fix message_store type safety (B2-5)
6. Fix signal_engine soft-conf cap (B2-6)
7. Fix signal_registry validation (B2-7)
8. Fix trade_guards warning log (B2-8)
9. Run tests, commit

### Batch 3: Signal Quality & Polish (4 files)
**Files:** `signal_engine.py` (cache invalidation), `equity_curve.py`, `signal_utils.py`, + tests

**Order:**
1. Fix horizon weights invalidation (B3-1)
2. Add horizon-aware accuracy fallback warning (B3-2) — TODO comment for full fix
3. Remove redundant Sharpe computation (B3-3)
4. Remove dead count_hold parameter (B3-4)
5. Run tests, commit

---

## 4. Impact Assessment

| Change | Could Break |
|--------|-------------|
| loop_contract status set | Other callers checking status strings — grep verified isolated |
| ATR floor 3% | Existing stops on low-ATR instruments may widen — DESIRED behavior |
| Kelly FIFO | Kelly recommendations change (more accurate) — no live execution depends on this |
| Signal engine None guard | No side effects — adds safety net |
| Metals loop exit code | Supervisor sees crash as non-zero — DESIRED for restart logic |
| Sentiment gate ticker guard | Metals/stocks lose one signal voter — reduces false votes |
| Soft-conf cap | Weak soft votes can't dominate — DESIRED correction |

---

## 5. Dependency / Ordering

Batch 1 MUST complete before Batch 2 (metals_loop touched in both).
Batch 2 and 3 are independent.
All batches must pass tests before merge.
