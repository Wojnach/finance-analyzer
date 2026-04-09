# Session Progress — Metals loop reliability + XAU SHORT support 2026-04-09

## Status: COMPLETE (merged + pushed, loop restart + smoke test pending)

### 2026-04-09 afternoon session | fix/metals-loop-reliability-apr09 → main
**Context**: investigation of "why isn't the metals loop trading" uncovered
5 real bugs + 1 feature gap. Shipped all 8 fixes in a single merge per user
direction (`All fixes in one merge` decision in plan).

Commits (branch FF-merged into main):
- `fff09aa feat(avanza): add page-based fetch_positions helper`
- `fe8adfc fix(metals): swing trader reliability overhaul + XAU SHORT support`
- `c7dd15b fix(infra): TNX FRED fallback + wmic → tasklist`
- Plus: deleted stale `HOLD` file at repo root (untracked orphan, 0 bytes,
  Apr 1, not referenced by any code)

Fixes shipped:
1. **Fix 1** (cash sync): `_sync_cash` now sets `cash_sync_ok` flag, entries
   refuse when False, Telegram alerts on transition
2. **Fix 2** (position reconciliation): new `_reconcile_swing_positions()`
   runs unconditionally first cycle, then every 3 cycles. Prunes swing state
   positions not on Avanza. 10-cycle failure streak watchdog. Today's 08:25
   phantom cascade would have been prevented by this
3. **Fix 3** (entry_ts hardening): corrupt `entry_ts` → drop position via
   new `corrupt_ids` list, no sell attempt, Telegram alert
4. **Fix 3b** (cascade root cause, found by Plan-agent review):
   `to_remove.append(pos_id)` at line 865 was dead code — the list was
   populated but never iterated for deletion. Only `_execute_sell`'s
   internal `del` removed positions, and that was skipped on place_order
   failure. Result: any failed SELL left the position in state forever,
   triggering the 08:25 cascade. Now: actual deletion loop + `sell_failed_at`
   timestamp with 5-min cooldown to prevent tight re-fire loops
5. **Fix 4** (fill verification): positions carry `fill_verified` +
   `buy_order_id`. New `_verify_recent_fills()` runs 15-90s after entry,
   calls the new `fetch_page_positions`, and rolls back unfilled orders
   (cancel buy via `delete_order_live`, cancel stop-loss, restore cash).
   Stop-loss placed IMMEDIATELY on buy success (protected during window)
6. **Fix 5** (HOLD cleanup): deleted stale orphan file
7. **Fix 6** (TNX FRED fallback): `_fetch_treasury` now falls back to FRED
   DGS10 when yfinance `^TNX` returns None. Reuses
   `portfolio/golddigger/data_provider.fetch_us10y` (already 1h-cached +
   circuit-breakered)
8. **Fix 7** (wmic → tasklist): `llama_server._is_llama_server_process`
   replaced deprecated `wmic.exe` with `tasklist /FI "PID eq N"`. Eliminates
   the FileNotFoundError spam
9. **Fix 8** (XAU SHORT support, canary-gated): direction-aware
   `_check_exits` math — `und_change_pct` flipped for SHORT, new
   `trough_underlying` tracking, SIGNAL_REVERSAL + MOMENTUM_EXIT
   direction-aware, exit_optimizer skipped for SHORT (Option A per plan).
   Ships disabled via `SHORT_ENABLED=False` + empty `SHORT_CANARY_WARRANTS`.
   User flips both to enable on a single canary warrant

Tests: 10 new tests in `TestReliabilityFixes` class + 1 LONG TAKE_PROFIT
regression test. 66 swing-trader tests pass + 120 total across files I
touched. No new ruff violations; removed one pre-existing F541.

Plan file: `/root/.claude/plans/structured-rolling-sparrow.md`
Plan-agent critique caught Fix 3b (missing `to_remove` deletion) and the
Fix 4 stop-loss-before-verification risk — both incorporated.

### What's next
- **Restart loops**: `schtasks /end /tn PF-MetalsLoop && schtasks /run /tn PF-MetalsLoop`
- **5-min smoke test**: watch `data/metals_loop_out.txt` for:
  - `cash_sync_ok=True` at startup (or explicit "cash sync failed" alert)
  - Phantom reconciliation fires on first cycle
  - No TNX stale errors (FRED fallback picks up)
  - No wmic FileNotFoundError
  - No SELL cascade
- **1-hour observation**: monitor Telegram for unexpected alert volume
- **To enable SHORT later** (separate step, after 24h observation):
  1. Run the metals loop interactively, inspect `trader.warrant_catalog`
     for lowest-leverage XAU SHORT MINI warrant keys (`direction == "SHORT"`,
     `underlying == "XAU-USD"`, sorted by `leverage` ascending then
     `spread_pct` ascending)
  2. Set `SHORT_ENABLED = True` and populate `SHORT_CANARY_WARRANTS` with
     that single key in `data/metals_swing_trader.py`
  3. Restart metals loop
  4. Watch for first SHORT entry on Telegram; manually verify on Avanza

---

# Session Progress — Auto-Improve 2026-04-09

## Status: COMPLETE

### What was done
1. **BUG-183**: Removed dead code after return in `metals_swing_trader.py:_regime_confirmed()` — unreachable lines referencing undefined `signal_data` (F821)
2. **BUG-184**: Renamed duplicate `test_btc_leads_eth` to `test_btc_leads_eth_sell` — BUY test case was silently shadowed (F811)
3. **REF-50**: 64 ruff auto-fix violations across 24 files (I001 import sorting, F401 unused imports, F541 f-strings, SIM114 same-arms if/elif, UP017 datetime.UTC)
4. **REF-51**: 9 unused vars/imports manually removed from `metals_loop.py` (F841×6, F401×3)
5. **Documentation**: Updated SYSTEM_OVERVIEW.md (signal count 32→34, new bug/ref entries, violation counts)

### Metrics
- Ruff violations: 382 → 309 (73 fixed, 19% reduction)
- Remaining violations are intentional (E402 lazy imports, F841 test vars, SIM117 cosmetic)
- Test count: ~6,449

### What's next
- SIM105 conversions in metals_loop.py (22 try/except/pass → contextlib.suppress) — deferred due to no test coverage for the monolith
- ARCH-18: metals_loop.py (6,574 lines) decomposition — large effort, separate session
- E741 ambiguous variable names in tests (40) — cosmetic, low priority

### 2026-04-09 08:30 UTC | main
12555a0 fix(metals): address codex review of f6b491c — 5 issues fixed
data/metals_swing_trader.py
data/metals_warrant_refresh.py
data/test_metals_swing_trader.py
tests/test_metals_swing_trader_notifications.py

### 2026-04-09 09:01 UTC | chore/reduce-tickers-apr09
3f859c7 docs: plan for reducing ticker universe to crypto + metals + MSTR
docs/PLAN_REDUCE_TICKERS.md

### 2026-04-09 09:01 UTC | main
4336f45 docs(guidelines): split verify+ship into codex review, test, ship, restart loops
docs/GUIDELINES.md

### 2026-04-09 09:11 UTC | chore/reduce-tickers-apr09
2e18ba0 chore(tickers): reduce main loop universe to 5 (crypto + metals + MSTR)
portfolio/tickers.py
tests/test_alpha_vantage.py
tests/test_consensus.py
tests/test_signal_pipeline.py

### 2026-04-09 09:15 UTC | chore/reduce-tickers-apr09
8cbd3fc chore(tickers): fix test_earnings_calendar + futures_data __main__ for reduced set
portfolio/futures_data.py
tests/test_earnings_calendar.py

### 2026-04-09 09:19 UTC | chore/reduce-tickers-apr09
b21c727 test: fix remaining ticker-dependent tests after reduction
tests/test_analyze.py
tests/test_gpu_skip.py
tests/test_iskbets.py

### 2026-04-09 09:29 UTC | 
29ca004 docs: plan for reducing ticker universe to crypto + metals + MSTR
docs/PLAN_REDUCE_TICKERS.md

### 2026-04-09 09:29 UTC | 
766987d chore(tickers): reduce main loop universe to 5 (crypto + metals + MSTR)
portfolio/tickers.py
tests/test_alpha_vantage.py
tests/test_consensus.py
tests/test_signal_pipeline.py

### 2026-04-09 09:29 UTC | 
ccd5144 chore(tickers): fix test_earnings_calendar + futures_data __main__ for reduced set
portfolio/futures_data.py
tests/test_earnings_calendar.py

### 2026-04-09 09:29 UTC | 
e582cf3 test: fix remaining ticker-dependent tests after reduction
tests/test_analyze.py
tests/test_gpu_skip.py
tests/test_iskbets.py

### 2026-04-09 09:35 UTC | main
eff076f docs(claude-md): sync signal count + ticker universe to current reality
CLAUDE.md

### 2026-04-09 11:27 UTC | chore/fingpt-daemon-and-cadence
040428f docs: plan for fingpt warm daemon + main loop 60s → 600s cadence
docs/PLAN_FINGPT_DAEMON.md

### 2026-04-09 11:32 UTC | chore/fingpt-daemon-and-cadence
879c7b1 feat(sentiment): warm fingpt daemon — reuse persistent model load
portfolio/sentiment.py
scripts/fingpt_daemon.py
tests/test_fingpt_daemon.py

### 2026-04-09 11:34 UTC | chore/fingpt-daemon-and-cadence
89b2a51 chore(cadence): bump main loop intervals 60/120 → 600s, time-based log rotation
portfolio/main.py
portfolio/market_timing.py
tests/test_market_timing.py

### 2026-04-09 11:46 UTC | chore/fingpt-daemon-and-cadence
9378311 fix: address code review — bounded daemon reads, wall-clock trigger gate
portfolio/sentiment.py
portfolio/trigger.py
tests/test_fingpt_daemon.py
tests/test_market_timing.py

### 2026-04-09 11:56 UTC | fix/metals-loop-reliability-apr09
d401b83 feat(avanza): add page-based fetch_positions helper
data/metals_avanza_helpers.py
portfolio/avanza_control.py

### 2026-04-09 11:57 UTC | fix/metals-loop-reliability-apr09
b379441 fix(metals): swing trader reliability overhaul + XAU SHORT support
data/metals_swing_trader.py
data/test_metals_swing_trader.py

### 2026-04-09 11:57 UTC | fix/metals-loop-reliability-apr09
56b87f4 fix(infra): TNX FRED fallback + wmic → tasklist
portfolio/llama_server.py
portfolio/macro_context.py

### 2026-04-09 11:59 UTC | hotfix/fingpt-daemon-prefer-small-model
0592d3a hotfix(fingpt): prefer 1.2B sentiment model over 8B finance-llama
scripts/fingpt_daemon.py

### 2026-04-09 11:59 UTC | 
fff09aa feat(avanza): add page-based fetch_positions helper
data/metals_avanza_helpers.py
portfolio/avanza_control.py

### 2026-04-09 11:59 UTC | 
fe8adfc fix(metals): swing trader reliability overhaul + XAU SHORT support
data/metals_swing_trader.py
data/test_metals_swing_trader.py

### 2026-04-09 11:59 UTC | 
c7dd15b fix(infra): TNX FRED fallback + wmic → tasklist
portfolio/llama_server.py
portfolio/macro_context.py
