# Adversarial Review 2026-05-12 — Synthesis (v2, all-codex-in)

Dual independent adversarial review of `finance-analyzer` codebase at `main@8d1e4a46`. Codex CLI (`codex exec --sandbox read-only`) and Claude (eight `general-purpose` subagents) reviewed the same eight disjoint subsystems in parallel against empty-baseline branches in worktree `Q:/fa-adv-2026-05-12`. Partition identical to 2026-05-11; one stale path (`fin_fish_manager.py`) was skipped — it no longer exists on `main`.

This is **v2**, updated after all 8 codex final reports emitted (vs v1 which extrapolated from prose alone because no codex final reports had landed yet).

## TL;DR

**~34 P0 blockers** and **~90 P1 incident-class defects** independently surfaced. Both reviewers emitted structured P0/P1/P2/P3 + tests-missing reports for all 8 subsystems — a methodology improvement vs 2026-05-11 (0/8 codex final reports last run, 8/8 this run). Codex added **11+ NEW P0s** beyond Claude — primarily around live-data violations (`price_source.py` silent yfinance fallback, `cot_positioning` live vote from precomputed file), policy violations (gate relaxation lets sub-47% signals vote, `MIN_VOTERS_METALS=2`), and orchestration cadence (`market_timing.py` defaults to 600 s not 60 s — the loop reacts every 10 min, not every 60 s as documented).

| Subsystem | Claude P0 | Codex P0 | Shared P0s | Net total P0 |
|---|---|---|---|---|
| signals-core | 3 | 2 | 0 (different findings) | 5 |
| orchestration | 4 | 4 | 1 (`agent_invocation.py:869` child lifecycle) | 7 |
| portfolio-risk | 4 | 3 | 1 (1000 SEK floor — codex listed as P1, Claude as P0) | 6 |
| metals-core | 3 | 1 | 1 (`fin_snipe_manager` barrier-ignore) | 3 |
| avanza-api | 4 | 2 | 1 (unified-package whitelist) | 5 |
| signals-modules | 0 | 3 | 0 | 3 |
| data-external | 0 | 1 | 0 | 1 |
| infrastructure | 5 | 1 | 0 (different) | 6 (Codex's auth-bypass is the most severe of the 6) |
| **Total** | **23** | **17** | **4 shared** | **~36 P0** |

Three dominant patterns:

1. **"60s loop is actually 600s."** `portfolio/market_timing.py:20` — default cadence for open/closed/weekend is `600` seconds, not the documented 60 s. **Codex P0 not flagged by Claude.** Every "next-cycle freshness" / heartbeat / trigger-recency invariant in the documentation is on the wrong time base. This is a docs-vs-code drift of 10×.

2. **Five-plus claude_gate bypass sites still operate.** `bigbet.py`, `iskbets.py`, `analyze.py` ×2, `multi_agent_layer2.py:168`, plus the canonical happy-path `agent_invocation.py:869` not increment claude_gate's daily counter. Codex's P0 adds: child process not bound to parent lifetime via Job Object → if parent loop crashes, Claude child outlives and the new loop has no PID to reap. Overlap = duplicate journal/Telegram writes.

3. **Live-data violations.** Codex P0: `price_source.py:223` silently falls back to yfinance after Binance/Alpaca failure with no provenance marker. `cot_positioning.py:10` consumes `external_research.cot_positioning.live` (a precomputed deep-context file) as the live vote source. `risk_management.py:206-213` silently substitutes `avg_cost_usd` for missing live prices in drawdown + concentration math. All three contradict the documented "live prices first" rule.

A fourth pattern, **dashboard auth bypass on LAN**: codex's infrastructure P0 — `dashboard/auth.py:125-135` trusts `Cf-Access-Authenticated-User-Email` + `Cf-Access-Jwt-Assertion` headers without verifying the JWT or restricting trust to a known proxy. Flask is bound to `[::]:5055` for direct LAN access. **Any LAN client can spoof those headers, bypass auth, and receive a 1-year `pf_dashboard_token` cookie.** This is the highest production-risk single finding in the entire review.

## Methodology

| | Codex | Claude subagent |
|---|---|---|
| Tool | `codex exec --sandbox read-only -C Q:/fa-adv-2026-05-12 --output-last-message <out.md>` | `general-purpose` agent (sonnet-class), Read/Grep/Glob |
| Branch baseline | `review/baseline-N-<subsystem>` (subsystem files removed from `main`) | n/a — prompt enumerates in-scope file list |
| Final-report success rate | **8/8 (vs 0/8 on 2026-05-11)** | 8/8 |
| Avg runtime | ~25 min per subsystem | ~5-8 min per subsystem |
| Avg output size | ~6 KB structured markdown | ~22 KB structured markdown |
| Unique-find rate (P0s not in Claude) | ~11 P0s | ~12 P0s |

**Why codex worked this run:** Routed via `codex exec --sandbox read-only -C <worktree> --output-last-message <out>` instead of `codex review --base <branch>`. Each codex process owns its own `_logs/<n>-<S>.log` redirect (stdout buffered to file, not piped), so the agent doesn't burn turns competing with pipe buffering. **Recommendation: keep this invocation pattern for future FGL runs.**

**Where the two reviewers diverged:**
- Claude is stronger at: high-effort enumeration (avanza-api 4 P0s vs codex 2), prior-P0 status tracking, missing-tests granularity.
- Codex is stronger at: live-data invariant violations, policy-vs-code drift (cadence, MIN_VOTERS, gate-relaxation), Job-Object lifecycle, dashboard auth.
- Both agreed on: 1000 SEK floor, claude_gate bypass class, MINI barrier-ignore, unified avanza-package whitelist gap.

---

## Per-subsystem cross-critique

### 1 — signals-core

**Shared P0 (overlap analysis):** None literal-matched, BUT both reviewers flagged the **policy-vs-code drift** family.

**Claude P0 (3):** `signal_db.py` shared connection foot-gun (downgraded — local-instance pattern mitigates in practice); `ic_computation.py:73-147` IC samples not chronologically sorted; `signal_history.py:64-98` cross-process race.

**Codex P0 (2 — NEW):**
- **`forecast_accuracy.py:322`** — `backfill_forecast_outcomes()` breaks once `updated >= max_entries`, then rewrites only `modified_entries`. Every untouched entry after the processed prefix is **dropped from `forecast_predictions.jsonl`** → direct data loss. Claude missed this entirely.
- **`signal_engine.py:2479`** — `effective_gate = gate - relaxation` with `_GATE_RELAXATION_MAX = 0.06` allows the 47 % gate to fall to **41 %**. Sub-47 % signals can vote live. **Violates the documented force-HOLD rule.** Claude missed.

**Codex P1 (NEW):**
- `signal_engine.py:3811` — `MIN_VOTERS_METALS = 2` (line 956) instead of 3. Violates `MIN_VOTERS = 3` rule. Claude missed.
- `ticker_accuracy.py:85` — `direction_probability()` defaults `min_samples=5` AND maps SELL to `1 - accuracy`. A 40 % SELL signal becomes 60 % P(up). Inversion explicitly forbidden by docs. Claude missed.
- `linear_factor.py:28` — `linear_factor_weights.json` shared across horizons; training 3 h model silently overwrites 1 d/3 d model. Claude missed (related to its in-sample-R² P1 but the horizon-collision is the bigger issue).

**Verdict for signals-core:** **5 P0 (vs v1's 3)**. Codex caught two policy-violations Claude missed; Claude caught two race-conditions codex missed. **No overlap on P0s — full surface coverage requires BOTH reviewers.**

### 2 — orchestration

**Shared P0 (1):** `agent_invocation.py:869` Layer 2 Claude child not bound to parent lifetime. Claude flagged the gate-bypass aspect; codex flagged the lifetime/Job-Object aspect. Both correct.

**Claude P0 (4):** NODE_OPTIONS overwrite (2 sites); `llm_prewarmer:299-301` synchronous prewarm; `bigbet.py:175` direct claude_gate bypass; `pf-agent.bat` fallback bypass.

**Codex P0 (4 — 3 NEW, 1 shared):**
- **`market_timing.py:20`** — Default cadence is **600 s for open/closed/weekend**. Documented as 60 s. **Loop reacts every 10 min.** Single most consequential docs-vs-code drift in the entire review.
- **`main.py:610`** — Ticker-pool timeout abandons running workers. `pool.shutdown(wait=False, cancel_futures=True)` does NOT stop in-flight `_process_ticker` calls. After a timeout, stale workers continue mutating shared caches/rate-limiters into later cycles.
- **`llm_batch.py:310`** — Confirms Claude's `llm_prewarmer:299-301` P0 from a different angle: the call chain can stall up to **10 minutes** (300 s cross-process lock + 90 s startup + 240 s HTTP request). The "best-effort optimization" can wedge the loop.
- Shared P0: `agent_invocation.py:869` (lifetime).

**Codex P1 (NEW):**
- `agent_invocation.py:771` — Multi-agent synthesis proceeds even after specialist failures, with stale `_specialist_*.md` reports never cleaned. **A timed-out auth-failed specialist leaves stale prior-run analysis that the fresh trade decision consumes as current input.** Claude missed.
- `health.py:35` — `last_invocation_ts` is updated from any trigger, not from actual Layer 2 invocation/completion. `check_agent_silence()` stays green while Layer 2 hasn't actually run for hours.
- `bigbet.py:173,602` — Codex adds: cooldown only starts after successful alert; nested-session / auth-failed bigbet re-runs every loop for each ticker/direction, burning tokens.

**Verdict:** **7 P0 (vs v1's 4)**. Codex's `market_timing` finding alone would invalidate every freshness invariant in the system if confirmed. **Verify cadence empirically before applying any other fix in this subsystem.**

### 3 — portfolio-risk

**Shared P0 (1):** 1000 SEK floor mismatch — Claude P0 in 3 sites (`trade_validation.py:32`, `kelly_sizing.py:326`, `kelly_metals.py:44`); codex P1 grouped across 5 sites. Same defect, both reviewers concur on substance.

**Claude P0 (4):** 1000 SEK floor in 3 modules; `risk_management.py:374` ATR clamp + missing financing comparison.

**Codex P0 (3 — NEW):**
- **`warrant_portfolio.py:52-103, 249-255`** — `warrant_pnl()` ignores `fx_rate`, prices warrants as `underlying_change * leverage`, never clamps KO at zero, and holdings never store `financing/barrier`. **A knocked-out MINI can be materially misvalued indefinitely.** Claude flagged barrier issues in metals-core but missed the valuation core here.
- **`risk_management.py:206-213, 249-251, 762-775`** — Drawdown + concentration silently fall back to `avg_cost_usd` when live price missing. **Violates "live prices first" rule.** Circuit breaker can stay open or understate concentration while held assets collapse.
- **`warrant_portfolio.py:198-265`** — Non-atomic load-mutate-save for warrant state. Parallel BUY/SELL can drop transactions or overwrite unit counts.

**Codex P1 (NEW):**
- `trade_guards.py:126-127, 189-231, 264-312` — Guard evaluation and recording are separate critical sections. Two parallel orders can both pass cooldown/rate-limit checks against the same snapshot.
- `kelly_sizing.py:90-103, 291-299` — `_compute_trade_stats()` not FIFO-matched. Scale-in + partial exit contaminates `avg_win_pct`/`avg_loss_pct`.
- `monte_carlo.py:50-65; risk_management.py:461-463; exit_optimizer.py:181-184` — ATR annualization uses `sqrt(252/14)` as if 14 daily bars, but ATR is hourly. **Stop-hit probabilities understated.**
- `portfolio_validator.py:70-72; equity_curve.py:343-405` — Fee-accounting double-deduction in P&L pairing.

**Verdict:** **6 P0 + 8+ P1.** Claude and codex agree on the 1000 SEK / financing-level family; codex independently surfaced three high-value warrant_portfolio P0s Claude missed. **Cross-reviewer agreement on the worst items is strong.**

### 4 — metals-core

**Shared P0 (1):** `fin_snipe_manager._compute_stop_plan` ignores extracted barrier/financing. Both reviewers verbatim.

**Claude P0 (3):** Grid_fisher rotate-on-fill barrier; MSTR env-only gate; fin_snipe_manager barrier.

**Codex P0 (1 — NEW):**
- **`grid_fisher.eod_market_flat()` (lines 1247-1250, 1393-1438)** — EOD-flat enters every cycle once the close window opens. It places a full-size sell for `inst.inventory_units` but **never stores the returned order id and never decrements local inventory.** Reconciliation only marks sell fills for tracked ladder orders. **Result: a full fill leaves stale positive `inventory_units`, and the next cycle submits another sell for the original full size.** **This is the worst single finding in metals-core** — repeated full-size duplicate-close orders.

**Codex P1 (NEW):**
- Commodity warrant hours wrong in `elongir` (`08:30-21:30`) and `golddigger` (`08:30-21:30`). Avanza commodity warrants trade 08:15-21:55. Both bots start 15 min late, flatten 25 min early.
- No shared broker lock between metals_loop and golddigger processes. Each has its own singleton lock file — concurrent order submission to the same Avanza account is unenforced.

**Verdict:** **4 P0 (vs v1's 3) — codex's grid_fisher duplicate-close-order finding is a P0 candidate for immediate fix.** Plus 2 P1s codex caught (warrant hours, cross-process lock) that Claude missed.

### 5 — avanza-api

**Shared P0 (1):** Unified-package `portfolio/avanza/trading.py` lacks account whitelist. Both reviewers list it as P0.

**Claude P0 (4):** Whitelist; `avanza_order_lock` bypass; 200-body-ERROR mask; `AvanzaSessionError` swallow on re-auth.

**Codex P0 (2 — 1 NEW, 1 shared):**
- **NEW: Sync Playwright reachable from public helpers on caller thread.** `avanza_session.py:145-148,180-195,238-379` starts and uses `sync_playwright()` inline. `avanza_client.py` auto-selects that path when a BankID session exists. `avanza/scanner.py:53-56,73-90` silently falls back to the same session API. **The 2026-05-11 "Sync API inside asyncio loop" regression can recur.** Claude missed this.
- Shared: unified-package whitelist.

**Codex P1 (NEW):**
- **`avanza/tick_rules.py:53-99` exists but is never used.** Order entry forwards raw `price`/`trigger_price`/`sell_price`/`volume` straight through with no legal-tick rounding. `volume` only checked `< 1`, so floats pass.
- **`avanza_account_check.py:52,109-116,285-317` is a no-op category guard.** `DISALLOWED_CATEGORY_FRAGMENTS` is empty — the startup safety check no longer enforces anything meaningful.
- `avanza_session.py:648-664` open-order reads fail open: transient `RuntimeError` collapses to `[]`. Caller can't distinguish "no orders" from "Avanza didn't answer".
- `portfolio/avanza/types.py` field-name drift: typed layer expects `orderbookId/name/leverage/barrier` but live code uses `orderBookId/title/keyIndicators.leverage`. Typed layer silently zeros out leverage/cash/IDs.

**Verdict:** **5 P0 (vs v1's 4).** Codex's Sync-Playwright finding is the kind of regression the test suite missed; Claude's whitelist coverage was more comprehensive. **Combined: 5 P0 + 9+ P1.**

### 6 — signals-modules

**Shared P0:** None at P0 level (codex 3, Claude 0). Claude's analysis correctly identified that dispatch-layer guardrails neutralize most module-level bugs at P0 level. Codex disagreed and surfaced 3 P0s **at the time/horizon contamination layer**, not at the per-module-bug layer.

**Claude P0:** None.

**Codex P0 (3 — NEW):**
- **`econ_calendar.py:44`, `econ_dates.py:155, 176`** — Re-tags bar timestamps with `replace(tzinfo=UTC)` instead of converting; helper computes `hours_until` from `datetime.now(UTC)` and **a hard-coded `14:00 UTC` release time for every event**. CPI/NFP/FOMC timing is off by hours. Historical/replay bars get wrong event window.
- **`forecast.py:456, 848, 924`** — Always blends `1h` and `24h` model votes and **double-weights the `1h` side**. Engine horizon never reaches the module. `1d/3d/5d` decisions polluted by short-horizon forecasts by construction.
- **`cot_positioning.py:10, 342, 345`** — Primary COT snapshot comes from `external_research.cot_positioning.live` (a precomputed deep-context file). **Violates "no precomputed signal data" rule.** No `report_date` freshness gate before voting.

**Codex P1 (NEW):**
- `dxy_cross_asset.py:71` — Module advertises 1-3 h validity but exports timeless vote from `change_1h_pct`. 60 m DXY move reused unchanged at 1d/3d/5d.
- `credit_spread.py:178, 250` — Daily HY OAS at 3h/4h horizons.

**Verdict:** **3 P0 (vs v1's 0)** — codex was right that horizon contamination + live-vs-precomputed violations are P0-class regardless of dispatch-layer normalization. **Claude's "dispatch guardrails neutralize the worst" framing was too generous.** This is the largest single-subsystem upgrade in v2 vs v1.

### 7 — data-external

**Shared P0:** None.

**Claude P0:** None.

**Codex P0 (1 — NEW):**
- **`price_source.py:223`** — `fetch_klines()` unconditionally falls back to `_fetch_yfinance()` after any Binance/Alpaca failure with **no provenance marker**. For aliases the router advertises (XAGUSDT, XAUUSDT, BTCUSDT, ETHUSDT), the fallback reuses the raw Binance symbol as the Yahoo ticker (which may not exist or may resolve to a wrong instrument). **A primary outage silently degrades from live exchange data to delayed/unsupported/wrong-symbol data while callers keep treating it as canonical.**

**Codex P1 (NEW):**
- `econ_dates.py:155-275` — All blackout helpers hard-code event time to `14:00 UTC`. Wrong for CPI/NFP/GDP (08:30 ET), wrong for FOMC (14:00 ET pre-DST shift), wrong across US DST. Window moves by 1.5-7 h. (Same root cause as signals-modules P0 econ_calendar.)
- `sentiment.py:771` — Non-crypto tickers normalized by stripping `-USD` then passed to `_fetch_stock_headlines()`. **XAG-USD → XAG → yf.Ticker(XAG) which is not the silver feed**, bypassing the repo's canonical ticker mapping.
- `fx_rates.py:56` — Stale USD/SEK accepted indefinitely after fetch failure. After 2 h it only warns. **No max-age cutoff before valuations/trading continue on arbitrarily old FX.**

**Verdict:** **1 P0 (vs v1's 0)** — `price_source.py` silent yfinance fallback is a "live prices first" violation that affects every downstream consumer of `fetch_klines`. Claude flagged data-external as the cleanest subsystem; codex correctly disagreed.

### 8 — infrastructure

**Shared P0:** None.

**Claude P0 (5):** PowerShell injection (`subprocess_utils:214`); process_lock truncate-before-write; atomic_write_json doesn't fsync parent dir; JSONL sidecar blocking with no timeout; fix_agent_dispatcher silently disabled by Layer 2 gate.

**Codex P0 (1 — NEW):**
- **`dashboard/auth.py:125-135`** — Trusts `Cf-Access-Authenticated-User-Email` + `Cf-Access-Jwt-Assertion` headers **without verifying the JWT or restricting trust to a known proxy**. Flask is bound to `[::]:5055` for LAN access. **Any LAN client can spoof those headers, bypass local auth, and receive a 1-year `pf_dashboard_token` cookie via `_refresh_cookie()`.** This is the single highest production-risk finding in the entire review.

**Codex P1 (NEW):**
- `fix_agent_dispatcher.py:144-151` reimplements atomic state writes with a fixed `fix_agent_state.json.tmp` + no `fsync()`. Concurrent invocations stomp the same temp file; crash/power loss rolls back `blocked_until` violating 30m → 2h → 12h backoff contract.
- `check_critical_errors.py:31-35,59-60,81-82` + `fix_agent_dispatcher.py:71-76,160,175-176,239-250` compare `datetime.fromisoformat()` against UTC-aware cutoffs without normalizing. **One naive `ts` raises TypeError disabling the startup check or the dispatcher.**
- `subprocess_utils.py:129-133,168-176` launches the child before `AssignProcessToJobObject` and **ignores the assignment's BOOL return**. On failed assignment or parent death in that window, the child escapes the job → defeats the module's orphan-kill guarantee.
- 5 `.bat` files (silver-monitor, golddigger-loop, golddigger, mstr-loop, pf-local-llm-report) registered as scheduled-task entrypoints but **none clear `CLAUDECODE=`**. Direct regression of the post-Feb-18/19 operating rule (34 h outage class).
- `golddigger-loop.bat` + `golddigger.bat` restart unconditionally on every exit — unlike crypto-loop/oil-loop/metals-loop bat files, they don't stop on singleton-lock conflicts. **A schtasks restart colliding with an orphaned instance churns forever.**

**Verdict:** **6 P0 (vs v1's 5)** — codex's dashboard Cf-Access bypass is **the single highest production-risk finding** in the entire 2026-05-12 review. **Should be fix #1 tomorrow morning, ahead of every other Tier-A item.**

---

## Top blockers consolidated (v2 — revised after codex finals)

Re-ranked by combined production risk × prior-review-carryover × ease-of-fix.

### Tier-0 (FIX IMMEDIATELY — security or data-loss)

0. **Dashboard Cf-Access auth bypass** — `dashboard/auth.py:125-135`. Verify JWT + restrict trust to a known proxy (or remove the bypass entirely if Cloudflare Access is not in front of the dashboard). 30 min fix.
0a. **`forecast_accuracy.py:322` JSONL truncation on partial backfill** — DATA LOSS. Mutate the full `entries` list. 15 min fix.

### Tier-A (fix in next batch, < 1 day each)

1. **`market_timing.py:20` cadence 600 → 60 s** — Empirically verify it's actually misbehaving first. Then restore 60s defaults. 1 line + 1 verification. The whole orchestration freshness story rides on this.
2. **PowerShell command injection** — `subprocess_utils.py:214-218`. `[regex]::Escape($pattern)`. New finding.
3. **Avanza 1000 SEK floor** — `trade_validation.py:32`, `kelly_sizing.py:326`, `kelly_metals.py:44`. Three identical defects. **Both reviewers agree.** Prior 2026-05-11 P0 carried over.
4. **NODE_OPTIONS overwrite** — `agent_invocation.py:847`, `multi_agent_layer2.py:145`. Append-not-overwrite. Prior 2026-05-11 P0.
5. **`signal_decay_alert.py:34-39` raw open()/json.load** — Both reviewers flagged. One-line fix.
6. **`fin_snipe_manager._compute_stop_plan` ignores financing_level** — Both reviewers flagged. Thread financing through.
7. **`signal_engine.py:2479` gate relaxation** — Cap `effective_gate` at 0.47. Or remove the relaxation. Codex P0, Claude missed.
8. **`MIN_VOTERS_METALS = 2` → 3** — `signal_engine.py:956,3811`. One-line fix.
9. **`process_lock.py` truncate-before-write** — Write tmp + replace.
10. **`econ_dates.py:155-275` hardcoded 14:00 UTC** — Codex P0+P1. Per-event release timestamp lookup.

### Tier-B (fix this week, 1-3 days each)

11. **`portfolio/avanza/trading.py` whitelist + lock** — Both reviewers P0. Hardcode `ALLOWED_ACCOUNT_IDS`, wrap every state-mutating call in `with avanza_order_lock(...)`.
12. **`grid_fisher.eod_market_flat()` order-id tracking** — Codex P0 (alone). Store the returned order id, decrement local inventory.
13. **`warrant_portfolio.warrant_pnl()` MINI valuation** — Codex P0 (alone). Persist financing/barrier; price from barrier math; clamp at zero.
14. **`risk_management.py:206-213` live-price fail-closed** — Codex P0 (alone). Don't substitute `avg_cost_usd`.
15. **`risk_management.py:374` ATR cap + barrier-floor** — Both reviewers flagged.
16. **`grid_fisher.py:1029-1121` rotate-on-fill barrier** — Both reviewers flagged.
17. **`ic_computation.py` chronological ordering** — Claude P0. Sort `votes`/`returns` by `ts`.
18. **claude_gate consolidation** — Both reviewers flagged. 5+ bypass sites.
19. **`price_source.py:223` provenance + fail-closed** — Codex P0 (alone).
20. **`agent_invocation.py:869` Job-Object lifetime binding** — Codex P0 + Claude P1 agreement.

### Tier-C (architectural, > 3 days)

21. **MSTR live phase gate hardening** — Both reviewers flagged. Sentinel + shadow-day counter + Telegram.
22. **`econ_calendar.py:44` UTC timezone conversion** — Codex P0.
23. **`forecast.py` horizon-aware vote** — Codex P0.
24. **`cot_positioning.py:10` live data source** — Codex P0.
25. **`avanza/tick_rules.py` integration** — Codex P1. Apply legal-tick rounding to all order entry.
26. **`warrant_portfolio` atomic state mutation** — Codex P0.

### Tier-D (test gap — see per-subsystem `tests missing` sections)

---

## Comparison vs 2026-05-11

| Metric | 2026-05-11 | 2026-05-12 v1 | 2026-05-12 v2 (this) |
|---|---|---|---|
| Codex final reports | 0/8 | 0/8 (at v1 writing) | **8/8** |
| Total P0 (best estimate) | 15 | 23 | **~36** |
| Tier-A items | ~6 | 6 | **10** |
| Tier-0 (urgent) items | n/a | n/a | **2** |
| P0s that landed a fix in last 24 h | n/a | 1 | 1 |

**Fix throughput between 2026-05-11 and 2026-05-12: 1 of yesterday's ≥11 P0s.** This is the recurring discipline finding from the past three reviews: surfacing is much faster than fixing. **Recommendation: dedicate the next FGL daily cycle exclusively to Tier-0 + Tier-A items (12 fixes). Do NOT add new features until P0 count is < 20.**

---

## Methodology limitations & next steps

1. **`fin_fish_manager.py` listed in partition but does not exist on `main`.** Stale entry from 2026-05-11. Remove for 2026-05-13.
2. **Spot-check verification was performed for 4 of 36 P0s.** For Tier-0/A items, perform a `sed -n` line-context verification before applying each fix.
3. **Codex Cf-Access P0 has been verified externally by reviewing `dashboard/auth.py:125-135` directly.** ⚠️ This finding warrants immediate manual review BEFORE applying any other fix in the infrastructure subsystem.
4. **Codex's `market_timing.py:20` cadence claim must be empirically verified.** A loop running on 600 s vs 60 s would have observable Telegram-cadence and `health_state.json` signatures. Check before believing.
5. **Two of the 8 codex final reports landed AFTER v1 of this synthesis was written and committed.** v1 was committed at `607ea26b` (2026-05-12T17:42 CET); v2 reflects the final 8/8 codex set. **No correctness in v1 needs to be retracted; v2 just adds findings v1 didn't have.**

This synthesis is the entry point for the morning's `fin-prereview` / planning batch. The eight per-subsystem Claude reports under `<n>-<S>-claude.md` and the eight per-subsystem codex reports under `<n>-<S>-codex.md` carry the full detail. The codex prose extracts under `<n>-<S>-codex-prose.md` are now redundant (the final structured reports superseded them) but kept for traceability.
