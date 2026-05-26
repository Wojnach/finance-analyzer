# Adversarial Review Synthesis — 2026-05-26

**Inputs:**
- 8 subagent reviews (`docs/AGENT_REVIEW_*.md`) — fresh pass on 2026-05-26
- 1 independent main-thread pass (`docs/ADVERSARIAL_REVIEW_INDEPENDENT_2026-05-26.md`)
- Cross-reference to 2026-05-24 prior pass — `[REPEAT]` tag tracks unfixed items

**Total findings:** ~144 (deduplicated to ~120 unique across all 9 inputs)
**P0:** ~30 · **P1:** ~60 · **P2:** ~30

---

## Executive Summary

Three structural classes of bug dominate this pass:

1. **Silent-failure clusters around Layer 2 / `claude -p` invocation.** The Mar-Apr 2026
   auth outage taught the system to detect `Not logged in`, but the detection lives only
   on the main-loop path. Metals loop (data/metals_loop.py:6765) and any future direct
   `subprocess.Popen([claude_cmd...])` are blind spots. Log rotation (10MB hourly) can
   also race against a 900s T3 invocation and silently invalidate the detector's
   start-offset.
2. **Direction-blindness in portfolio-risk after the SHORT/BEAR cert work.** P&L,
   stop-loss math, monte-carlo, drawdown, ATR proximity — five separate sites still
   assume `long-only`. Any held SHORT/BEAR cert reports inverted P&L AND trips spurious
   "stop triggered" on profitable green prints.
3. **Cross-process race conditions in portfolio state.** `portfolio_mgr`, `trade_guards`,
   `warrant_portfolio` all use process-local `threading.Lock` while main loop + Layer 2
   subprocess + dashboard mutate the same JSON files from separate processes. Atomic
   write is necessary but insufficient — read-modify-write cycles need cross-process
   locks (`file_utils.jsonl_sidecar_lock` exists; not used).

A fourth theme: **the [REPEAT] count is growing.** Avanza-API: 5/5 repeats from May 24.
Orchestration: ~10 repeats. Signals-core: 8 repeats. Portfolio-risk: ~15 repeats. The
fgl review pipeline is generating findings faster than the implementation pipeline can
close them.

---

## Top 20 Highest-Impact Items (Cross-Subsystem Dedup, Ranked)

| # | Severity | Path | Finding | Source |
|---|---------|------|---------|--------|
| 1 | P0 | `data/metals_loop.py:6765` | Direct claude Popen with NO auth scan — mirror of Mar-Apr outage on a different loop | INDEPENDENT (NEW) |
| 2 | P0 | `portfolio/agent_invocation.py:1117` + `log_rotation.py:62-67` | Auth detector blind to log rotation during T3 invocation | ORCHESTRATION (NEW) |
| 3 | P0 | `portfolio/main.py:973-980` | Off-hours bare-elif has NO autonomous fallback — silently skipped triggers | ORCHESTRATION [REPEAT 5x] |
| 4 | P0 | `portfolio/warrant_portfolio.py:96` | warrant_pnl hard-codes LONG math — 41 SHORT certs report inverted P&L | PORTFOLIO-RISK [REPEAT] |
| 5 | P0 | `portfolio/risk_management.py:374,382,465,484,897` | Stop-loss math LONG-only across 5 sites — SHORT cert green prints trip stop | PORTFOLIO-RISK [REPEAT] |
| 6 | P0 | `portfolio/portfolio_mgr.py:108-159` + `warrant_portfolio.py:42-48,265` + `trade_guards.py:32` | Process-local `threading.Lock` for cross-process state — drops trades | PORTFOLIO-RISK [REPEAT] |
| 7 | P0 | `portfolio/accuracy_stats.py:152-164` + `outcome_tracker.py:160-166` | SQLite/JSONL dual-write divergence strands accuracy on stale data | SIGNALS-CORE [REPEAT] |
| 8 | P0 | `portfolio/signal_engine.py:2720,2742` + `accuracy_stats.py:840` | Bias-penalty double-application — contrarian votes neutered | SIGNALS-CORE [REPEAT 3x] |
| 9 | P0 | `portfolio/signal_decay_alert.py:62-92` | Decay alert two silent-failure modes (schema drift + low recent_total) | SIGNALS-CORE (NEW) |
| 10 | P0 | `portfolio/agent_invocation.py:402-423` | `_no_position_skip` reads missing field — gate always returns True, kills all entries | ORCHESTRATION [REPEAT] |
| 11 | P0 | `portfolio/microstructure_state.py:205-213` | `persist_state` doesn't persist deques — OFI z-score = 0.0 for ~10 min after restart | DATA-EXTERNAL (NEW) |
| 12 | P0 | `portfolio/crypto_precompute.py:154-235` | 5 raw `requests.get` + thread-unsafe yfinance — 429 ban risk | DATA-EXTERNAL (NEW) |
| 13 | P0 | `portfolio/earnings_calendar.py:64,177` + `alpha_vantage.py:150` | Missing `"Information"` rate-limit key — silent 24h cache-of-None on AV rate limit | DATA-EXTERNAL [REPEAT] |
| 14 | P0 | `portfolio/agent_invocation.py:1021` | `specialist_timeout_s=30s` default below specialist's own 90-120s → quorum_fail | ORCHESTRATION [REPEAT] |
| 15 | P0 | `portfolio/main.py:1051,1086` | `_run_cycle_id` resets on restart — IC cache + safeguards effectively dead | ORCHESTRATION [REPEAT] |
| 16 | P0 | `portfolio/loop_contract.py:335` | In-flight suppression doesn't match `invoked_<why>` prefix — false-positive storms | ORCHESTRATION [REPEAT] |
| 17 | P0 | `portfolio/portfolio_validator.py` | No per-transaction overdraft replay — only end-state cash check | PORTFOLIO-RISK (NEW) |
| 18 | P0 | `portfolio/monte_carlo_risk.py:204,228,408` | `_trading_days=365` hardcoded + raw fx_rate bypass — 20% sigma understatement | PORTFOLIO-RISK [REPEAT] |
| 19 | P0 | `dashboard/auth.py:175` | Bearer token path skips `_refresh_cookie()` — silent token expiry after 1y | INFRASTRUCTURE (NEW) |
| 20 | P0 | `portfolio/health.py:165` | `fromisoformat()` naive datetime subtracted from UTC-aware — `/api/health` crash | INFRASTRUCTURE (NEW) |
| 21 | P0 | `data/metals_loop.py:1088` | Silver fast-tick `_has_active_silver()` only reads legacy `POSITIONS`, not swing trader state — all -3% to -12.5% alerts silently no-op for swing-managed silver | METALS-CORE (NEW) |
| 22 | P0 | `data/metals_swing_trader.py:2760` | Stop-sell buffer 1% (`sell_price = trigger * 0.99`) — same wick-bypass class fixed in grid_fisher/fin_snipe (widened to 3%); swing trader is primary entry path, never widened | METALS-CORE (NEW) |
| 23 | P0 | `data/metals_loop.py:1051` | `_sleep_for_cycle` returns immediately on overrun — fast-tick produces 0 silver ticks on LLM-heavy cycles, corrupts velocity deque | METALS-CORE (NEW) |
| 24 | P1 | `portfolio/signals/cross_asset_tsmom.py:148-171` | Bond/equity momentum returns same direction across tickers — polarity reversed for ~3 of 5 (TLT up = BUY metals but should be SELL crypto) | SIGNALS-MODULES (NEW) |
| 25 | P1 | `portfolio/signals/treasury_risk_rotation.py:182-185` | Inverts `action` for safe-haven tickers but leaves `sub_signals` unchanged — journal shows "subs BUY, action=SELL" mismatch | SIGNALS-MODULES (NEW) |
| 26 | P1 | `portfolio/signals/econ_calendar.py:44` | `.replace(tzinfo=UTC)` reinterprets wall-clock without converting tz-aware — CET timestamps silently shifted 1-2h, corrupting FOMC/CPI proximity math | SIGNALS-MODULES (NEW) |

---

## Cross-Cutting Themes

### Theme 1: Auth-detection coverage gaps

The Mar-Apr 2026 outage triggered `detect_auth_failure()` in `claude_gate.py`. Coverage:

- ✅ `agent_invocation.try_invoke_agent` — covered (line 619)
- ❌ `data/metals_loop.py:6765` — direct Popen, NO scan (Item #1)
- ❌ JSON-envelope mode (`--output-format json`) — line-prefix `{` rejects scan (Independent P0 #2)
- ⚠ `claude_gate.invoke_claude` timeout path (line 610) — never scans on `timed_out=True` (Orchestration P1)
- ⚠ Log rotation race during T3 — offset invalid after rotate (Item #2)

**Recommendation:** Extract the scanner into `claude_gate.scan_log_segment(path, start_offset, expected_inode)` that all callers MUST invoke after Popen completes. Make direct subprocess.Popen of `claude` a CI lint failure.

### Theme 2: SHORT/BEAR direction-blindness

Five sites in `risk_management.py` + `warrant_portfolio.py:96` + `monte_carlo.py:305` all
assume LONG. The system has 41 SHORT certs in `metals_warrant_catalog.json`. Holding
any of them causes:
- Inverted P&L in dashboard/journal/telegram
- Spurious `stop_triggered=true` on green prints (force-sell into strength)
- Wrong `p_stop_hit` in Monte Carlo
- Wrong `check_atr_stop_proximity` warnings every cycle

**Recommendation:** Plumb `direction_sign` (LONG=+1, SHORT=-1) through holdings schema as
a first-class field. Refactor the 5+ sites to multiply by sign and use `(price - stop) * sign < 0`
for triggered.

### Theme 3: Cross-process state corruption

`threading.Lock` is process-local. Main loop, Layer 2 subprocess, dashboard, bigbet,
iskbets, fix-agent dispatcher are SEPARATE processes. They all mutate:
- `portfolio_state.json` / `portfolio_state_bold.json`
- `portfolio_state_warrants.json`
- `trade_guards_state.json`
- `trigger_state.json`
- `health_state.json`

`atomic_write_json` is safe at the write boundary, but the read-modify-write cycle is
not protected. `file_utils.jsonl_sidecar_lock` already provides the right primitive —
just isn't used by `portfolio_mgr._save_state_to`, `warrant_portfolio.save_warrant_state`,
or `trade_guards._save_state`.

**Recommendation:** A small refactor session that converts all 5+ state save sites to
wrap read-modify-write in `jsonl_sidecar_lock`. ~50 LOC change, high-confidence fix.

### Theme 4: Five unrate-limited loops to Binance

`main.py`, `data/metals_loop.py`, `mstr_loop`, `data/oil_loop.py`, `data/crypto_loop.py`
all hit Binance directly. No shared token bucket. `portfolio/http_retry.py` is per-call
backoff. A schtasks-restart storm (sleep wake at 05:45 + 06:00 AutoImprove triggers all
loop starts in ~10s) can burst 50+ requests/min into Binance.

**Recommendation:** Either consolidate to a single process with a scheduler, or introduce
`shared_state.binance_token_bucket(weight)` that all 5 loops must acquire. Either way
is architectural — won't be done in one session, but it's getting urgent.

### Theme 5: [REPEAT] backlog growth

```
Subsystem          | Total | [REPEAT] | NEW
-------------------+-------+----------+-----
avanza-api         |   5   |   5      |   0
portfolio-risk     |  30   |  ~15     | ~15
signals-core       |  29   |   8      |  21
data-external      |  30   |  ~18     |  12
orchestration      |  30   |  ~10     |  20
infrastructure     |   7   |   0      |   7
metals-core        |  30   |  ~10     |  20
signals-modules    |  28   |  14      |  14
independent        |  13   |   0      |  13
-------------------+-------+----------+-----
TOTALS             | 202   |  80      | 122
```

About 39% of findings carry forward as unfixed. Avanza-API is at 100%. Without a
dedicated [REPEAT]-closing session, the backlog continues to compound — and the
fix-agent dispatcher is being asked to handle more categories than it has rules
for.

**Recommendation:** Block one weekly fgl session for [REPEAT]-only closure. Target:
zero P0/P1 [REPEAT] entries by the third such session.

---

## Per-Subsystem Highlights

### signals-core (29 findings · 5 P0 / 13 P1 / 11 P2)
Top: SQLite/JSONL accuracy divergence (P0), bias-penalty double-application 3rd review
(P0), decay alert dual silent-failure modes (P0). All three feed every downstream
accuracy gate.

### orchestration (30 findings · 6 P0 / 12 P1 / 12 P2)
Top: off-hours autonomous fallback missing (P0 [REPEAT 5x]), auth detector log-rotation
race (P0 NEW), specialist timeout misconfiguration (P0 [REPEAT]). Mar-Apr outage class
not fully closed.

### portfolio-risk (30 findings · 8 P0 / 16 P1 / 6 P2)
Top: warrant_pnl LONG-only (P0), state-file process-local locks (P0), risk_management
stop math LONG-only across 5 sites (P0). Direction-blindness is the dominant theme.

### data-external (30 findings · 9 P0 / 12 P1 / 10 P2)
Top: microstructure deque persistence missing (P0 NEW), crypto_precompute 5 raw
requests (P0 NEW), earnings_calendar 24h cache-of-None on rate limit (P0 [REPEAT]).
Data freshness silently degrades after first failure.

### avanza-api (5 findings · 2 P1 / 3 P2 · 100% [REPEAT])
All 5 findings carry forward from May 24. Subsystem under-served by current
implementation cadence.

### infrastructure (7 findings · 2 P0 / 2 P1 / 3 P2)
Top: Bearer token cookie refresh (P0), `/api/health` crash from naive datetime (P0),
Telegram retry doubling (P1). All NEW this pass.

### metals-core (30 findings · multiple P0/P1)
Top: silver fast-tick blind to swing-trader state (`metals_loop.py:1088`) — all -3% to
-12.5% threshold alerts and 3-min velocity flush silently no-op for swing-managed silver,
which is the canonical post-Apr-2026 entry path. Swing-trader stop-sell buffer at 1%
(line 2760) was never widened to 3% like grid_fisher/fin_snipe_manager. Fast-tick zero
on cycle overrun (line 1051) gaps velocity deque without reset. Ministral prompt
mislabeled as "cryptocurrency trader" on metals decisions (`metals_llm.py:443`).
Small-sample accuracy inflation pattern (XAG 82% on 34 samples) still uncorrected at
`metals_signal_tracker.py:643`. Stale warrant catalog (5+ days on disk per git status,
TTL is 6h) falls back silently with DEBUG-only log.

### signals-modules (28 findings · multiple P1)
Top NEW: `cross_asset_tsmom.py:148-171` — `_compute_bond_momentum` /
`_compute_equity_momentum` return same direction across all tickers, polarity reversed
for ~3 of 5 (TLT up is BUY for metals but should be SELL for crypto; SPY up is BUY for
crypto but should be SELL for metals). `treasury_risk_rotation.py:182-185` — final
`action` inverted for safe-haven tickers but `sub_signals` dict unchanged, journal sees
"all subs BUY but action=SELL" for XAU/XAG. `econ_calendar.py:44` — `.replace(tzinfo=UTC)`
reinterprets wall-clock without converting tz-aware timestamps, CET timestamps shifted
1-2h, corrupting FOMC/CPI proximity. Zero of 14 prior 2026-05-24 findings remediated;
all 14 are [REPEAT].

### independent (13 findings · 3 P0 / 6 P1 / 3 P2 · 100% NEW)
Cross-cutting + architectural. Notably: 5-loop unrate-limited Binance fan-out, JSON-
envelope auth scan blind spot, metals_loop direct Popen.

---

## Recommended Action Items for Next Session

### Tier 0 — Ship this week (cheap, high-impact)
1. `dashboard/auth.py:175` — add `_refresh_cookie()` to Bearer path (1 line)
2. `portfolio/health.py:165` — `.replace(tzinfo=UTC)` on `fromisoformat` (1 line)
3. `portfolio/http_retry.py:55` — cap jitter at `min(wait, 60)` (1 line)
4. `portfolio/main.py:973-980` — invoke `autonomous_decision()` in bare-elif (5 LOC)
5. `portfolio/agent_invocation.py:1021` — raise specialist_timeout_s default (1 line)

### Tier 1 — One-session refactor (medium effort, structural)
6. Cross-process state locks — wrap `portfolio_mgr` + `warrant_portfolio` + `trade_guards`
   save sites in `jsonl_sidecar_lock` (~50 LOC, well-bounded)
7. Auth scanner extraction — single helper in `claude_gate`, all subprocess.Popen callers
   route through it OR a CI lint that bans direct Popen of `claude` (lint = 30 LOC,
   wiring = ~100 LOC across metals_loop + agent_invocation + multi_agent_layer2)
8. SQLite/JSONL divergence detection — emit critical_errors when SQLite trails JSONL by
   >100 rows (~20 LOC in outcome_tracker + accuracy_stats)

### Tier 2 — Multi-session architectural
9. Direction-aware risk/stop math — refactor 5+ sites in risk_management + warrant_portfolio
   + monte_carlo to read `direction_sign` from holdings (~200 LOC, needs careful test pass)
10. Shared Binance token bucket — `shared_state.binance_token_bucket()` + retrofit all
    5 loops (~150 LOC, requires schtasks coordination test)

### Tier 3 — Backlog-closing
11. Dedicated [REPEAT]-only session — target 100% of Avanza-API (5 trivial fixes) plus
    the top-10 P1 [REPEAT] entries from portfolio-risk + orchestration. One day's work
    closes ~25 backlog items.

---

## Notes on Methodology

This synthesis intentionally does NOT propose code changes — that's the next plan
document's job (`docs/RESEARCH_PLAN.md` or a dedicated `docs/FGL_2026-05-26_FIX_PLAN.md`).

Two of eight subagent reviews (metals-core, signals-modules) were still running when
the main thread reached its synthesis budget. Their findings will be added in a
follow-up commit if novel; otherwise the orchestration/portfolio-risk findings cover
the dominant overlap.

The independent pass cross-validated the subagent findings on 4 items (auth detection,
warrant direction blindness, state-file races, accuracy divergence). All 4 are
ranked in the Top 5 by independent severity, confirming the subagents' priorities are
not artifacts of prompt anchoring.

---

## Historical Continuity

| Past Incident | Finding(s) From This Pass That Could Recur It |
|---------------|-----------------------------------------------|
| Mar-Apr 2026 silent auth outage (3 weeks) | Items #1, #2, Theme 1 |
| Mar 3 stop-loss instant-fill | Item #4, #5 — wrong direction = wrong stop math |
| Feb 18-19 CLAUDECODE env outage (34h) | None this pass — defenses hold |
| Bold strategy -7% loss Feb 11-18 | Item #17 — no per-tx overdraft check still |
| BUG-178 silent ticker hangs | Orchestration P0 #15 — `_run_cycle_id` reset still |

---

*End of synthesis. Source documents:*
- `docs/AGENT_REVIEW_AVANZA_API.md` (5 findings)
- `docs/AGENT_REVIEW_PORTFOLIO_RISK.md` (30 findings)
- `docs/AGENT_REVIEW_SIGNALS_CORE.md` (29 findings)
- `docs/AGENT_REVIEW_ORCHESTRATION.md` (30 findings)
- `docs/AGENT_REVIEW_DATA_EXTERNAL.md` (30 findings)
- `docs/AGENT_REVIEW_INFRASTRUCTURE.md` (7 findings)
- `docs/AGENT_REVIEW_METALS_CORE.md` (pending)
- `docs/AGENT_REVIEW_SIGNALS_MODULES.md` (pending)
- `docs/ADVERSARIAL_REVIEW_INDEPENDENT_2026-05-26.md` (13 findings + cross-cutting)
