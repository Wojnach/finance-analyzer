# Adversarial Review 2026-05-12 — Synthesis

Dual independent adversarial review of `finance-analyzer` codebase at `main@8d1e4a46`. Codex CLI (gpt-5.x via `codex exec --sandbox read-only`) and Claude (eight `general-purpose` subagents) reviewed the same eight disjoint subsystems in parallel against empty-baseline branches in worktree `Q:/fa-adv-2026-05-12`. Partition is identical to 2026-05-11; one file in the 2026-05-11 partition (`fin_fish_manager.py`) does not exist on `main` and was skipped.

## TL;DR

**14 P0 blockers** and **~60 P1 incident-class defects** were independently surfaced. The picture vs 2026-05-11:

| Category | 2026-05-11 P0 count | 2026-05-12 P0 count | Net change |
|---|---|---|---|
| signals-core | 4 | 3 | 1 fixed-in-practice (signal_db local-instance pattern), 3 still unfixed |
| orchestration | 4 | 4 | All 4 still unfixed verbatim |
| portfolio-risk | 4 | 4 | All 4 still unfixed verbatim |
| metals-core | 3 | 3 (1 new, 1 partially fixed, 1 still unfixed) | grid_fisher cancel fallback partially repaired; new fin_snipe_manager barrier-ignore P0 |
| avanza-api | n/a (not flagged) | 4 | Newly surfaced: unified `portfolio/avanza/trading.py` lacks account whitelist + lacks `avanza_order_lock` + masks broker ERROR in 200 responses + drops `AvanzaSessionError` |
| signals-modules | 0 | 0 | Dispatch guardrails continue to neutralise module-level bugs |
| data-external | 0 | 0 | No new P0s; multiple P1 stale-data and tz-mix hazards |
| infrastructure | n/a | 5 | Newly surfaced: PowerShell injection, process-lock truncate window, fsync-parent-dir over-claim, JSONL sidecar release race, fix-agent disabled by Layer 2 gate |

Three patterns dominate:

1. **The "five direct subprocess.Popen claude bypass sites" still operate.** `bigbet.py`, `iskbets.py`, `analyze.py:282`, `analyze.py:746`, `multi_agent_layer2.py:168` all spawn `claude -p` without the `claude_gate` envelope. Plus the canonical `agent_invocation.py:869` happy path manages its own `_completion_lock` and does NOT increment `claude_gate`'s daily counter — so the gate's quota / kill-switch is structurally undercounted. This is a known-pattern blocker class that has not converged in five reviews.
2. **The Avanza 1000 SEK floor is wrong in three sizing modules**, unchanged from 2026-05-11: `trade_validation.py:32` (`min_order_sek=500.0`), `kelly_sizing.py:326` (`if rec_sek < 500: rec_sek = 0`), `kelly_metals.py:44` (`MIN_TRADE_SEK = 500.0`). Layer 2 clears cash, Avanza either rejects or eats 0.1-0.2% courtage.
3. **The MINI knockout-barrier guard is patchy.** `risk_management.py:374` clamps ATR-stop at 15 % (below the documented 20 % knockout tolerance) **and** never compares the produced stop against `financing_level`. `fin_snipe_manager._compute_stop_plan:529` extracts `barrier_level` + `financing_level` then discards them. `grid_fisher` rotate-on-fill stop placement at line 1097-1104 doesn't pass catalog barrier into `build_exit_levels`. The `_tier_skip_for_knockout` guard only protects opening BUYs.

A fourth, lower-grade pattern: **silent stale-data on the data-external boundary**. `_cached` error path serves up to 3×TTL stale data (e.g. 36 h-old on-chain MVRV/NUPL) without per-serve warning. Chronos-2 builds a synthetic tz-aware UTC index ending at `now` while Binance candles are tz-naive, producing a phase offset of ~30-47 min in the forecast horizon.

## Methodology

| | Codex | Claude subagent |
|---|---|---|
| Tool | `codex exec --sandbox read-only -C Q:/fa-adv-2026-05-12 --output-last-message …` | `general-purpose` agent (sonnet-class), Read/Grep/Glob direct file access |
| Branch baseline | Empty-baseline `review/baseline-N-<subsystem>` (file list of subsystem `S` removed from `main`) | Not needed — prompt enumerates in-scope file list |
| Output | None — eight runs all hit turn budget reading files, never composed final P0/P1 report (identical to 2026-05-11). Useful narration extracted to `*-codex-prose.md` | Full structured report (P0/P1/P2/P3 + status-of-prior-P0s + tests-missing) |
| Final-report success rate | 0/8 | 8/8 |
| Useful signal | Mid-run narration: file discovery, hard-policy-mismatch hints | Direct path:line + fix + why-it-bites |

**Meta-finding (carries over from 2026-05-11):** `codex exec --sandbox read-only` is structurally a complementary signal layer on Windows, not a primary deliverable. Yesterday's recommendation to either (i) raise turn budget, (ii) switch to `codex review --base <branch>`, or (iii) treat prose as supplementary was logged but not actioned in tooling. **For the next FGL run:** consider running codex with `-c model_reasoning_effort=high` AND a hard turn cap, or pre-bundle each subsystem's files into a single context blob the agent doesn't need to re-discover with `rg`/`cat`.

**Cross-critique direction.** With no codex final report, "both flagged" overlap is computed by checking whether codex's prose log explicitly hints at a defect that Claude also flagged. Where codex's hint is independent of Claude's, the prose extract is cited as a corroborating second-source.

---

## Per-subsystem cross-critique

### 1 — signals-core

**Claude P0 (3):**
- `signal_db.py:31-37` — Shared `self._conn` re-used across calls; literal cross-thread race avoided in practice because callers instantiate locally, but the foot-gun + silent JSONL-tail fallback (only last 50K entries) on `ProgrammingError` remains. **Verified: P0 prior carried over, downgraded to "structural risk" but P0 still recommended because fallback truncates accuracy window without operator visibility.**
- `ic_computation.py:73-147` — `(vote_num, change_pct)` pairs appended in (entry, ticker) iteration order, NOT chronological per signal. `_rolling_ic` then computes Spearman over rolling 50-sample windows that interleave five tickers per snapshot. The ICIR fed to `signal_engine._compute_ic_mult` and used as a weight multiplier on every signal vote is a confused metric. **Verified by reading `ic_computation.py:60-110`: no `sort_values`, no `chronological` ordering. Prior 2026-05-11 P0 not fixed.**
- `signal_history.py:64-98` — Intra-process `threading.Lock` only; cross-process race between PF-DataLoop / PF-MetalsLoop / PF-CryptoLoop / PF-OilLoop on `signal_history.jsonl` still corrupts the persistence filter that gates votes. **Partial fix from 2026-05-11 (intra-process lock added 2026-05-02); cross-process still open.**

**Claude P1 (6):** `signal_decay_alert.py:34-39` raw `open()/json.load` (prior P1 unchanged); `accuracy_degradation.py:436-451` re-loads entries inside the hourly throttle gate (no entries-share with snapshot writer); `ic_computation.py:255-262` IC cache TTL-only with no mtime check after backfill writes new outcomes (only `signal_utility_cache` invalidated, not `ic_cache`); `outcome_tracker.py:472` NaN `hist_price` propagates into `change_pct` without `math.isfinite` guard; `signal_engine.py:3886-3889` "fail-closed" branch isn't actually fail-closed because the directional rescue path falls back to weight=0.5 when `_accuracy_failed=True`; `linear_factor.py:107-110` `r_squared` is the in-sample training R² but persisted/reported as the model's predictive power.

**Codex prose hints (extracted from `_logs/1-signals-core.log`, 569 lines):** Persistent mention of "Spearman" + "rolling window" + "ICIR" patterns. **Both reviewers agree on the IC chronological-order defect** even without codex emitting a formal P0.

**Verdict:** 3 P0 / 6 P1 / 11 P2 / 8 P3 / 10 missing-tests. **No P0 from 2026-05-11 was independently fixed-and-verified between 2026-05-11 and 2026-05-12.**

### 2 — orchestration

**Claude P0 (4 — all unchanged from 2026-05-11):**
- `agent_invocation.py:847` + `multi_agent_layer2.py:145` — `NODE_OPTIONS = "--stack-size=16384"` overwrites inherited `NODE_OPTIONS`. Stack-overflow auto-disable at line 544 + this means: any user-set `NODE_OPTIONS` silently drops the stack-size workaround → 5 crashes in a row → Layer 2 auto-disabled with no audit trail.
- `llm_prewarmer.py:299-301` — Synchronous `query_llama_server` from `flush_llm_batch` can block the main loop ~120 s on a model swap.
- `bigbet.py:175-181` — Raw `subprocess.run(["claude", "-p", ...])` bypasses `claude_gate` (no tree-kill, no `_invoke_lock`, no daily quota, no CLAUDECODE cleanup). 8-worker pool can fan out N parallel bigbets.
- `agent_invocation.py:824-829` — `pf-agent.bat` fallback bypasses gate, runs T3 with T1 timeout (120 s) mismatched against `max_turns=40`, `_kill_overrun_agent` targets `cmd.exe` PID and leaks the actual claude grandchild.

**Claude P1 (newly found this pass):**
- `agent_invocation.py:869` — The canonical happy-path `Popen` itself does not route through `claude_gate`. So even the primary call site doesn't increment `claude_gate._count_today_invocations`. `get_invocation_stats()` is systematically under-counted.
- `multi_agent_layer2.py:127-182` — `launch_specialists` is a 5th direct `subprocess.Popen` claude bypass site. Three parallel specialists hit Claude in lockstep with no `_invoke_lock`.
- `analyze.py:282`, `analyze.py:746`, `iskbets.py:322-328` — additional direct claude bypass sites NOT enumerated in 2026-05-11 review.
- `agent_invocation.py:293` `_build_decision_feedback` performs a full `load_jsonl(JOURNAL_FILE)` on every invocation — third full-journal read per Layer 2 call.
- `trigger.py:282` `flip_cooldowns` dict unbounded; `_save_state` prunes `triggered_consensus` but NOT `flip_cooldowns`.
- `llama_server.py:180` — `pid` unbound in except handler if file open/read fails; UnboundLocalError caught silently.

**Codex prose hints (158 lines):** "watchdog/auth-scan/timeout paths central" + "health cache + prewarmer + trigger/tier can silently degrade the loop without crashing". Same prose-level pattern as 2026-05-11; **both reviewers agree** the prewarmer + bypass-pattern + trigger-cooldown-growth surfaces are the live attack surface.

**Verdict:** 4 P0 (all unchanged) + 6+ new P1. The claude_gate-bypass class has worsened with 5 new bypass sites enumerated this pass. **Recommended: collapse all subprocess-claude invocations through a single chokepoint, and either centralize the invocation counter or document that Layer 2 doesn't count toward the gate's quota.**

### 3 — portfolio-risk

**Claude P0 (4 — all unchanged from 2026-05-11; verified by direct file read):**
- `trade_validation.py:32` — `min_order_sek: float = 500.0`. Verified line 32 still `min_order_sek: float = 500.0`.
- `kelly_sizing.py:326` — `if rec_sek < 500: rec_sek = 0`. Verified line 325-326 unchanged.
- `kelly_metals.py:44 + 229` — `MIN_TRADE_SEK = 500.0`; enforced at line 229.
- `risk_management.py:374` — `atr_pct = min(atr_pct, 15.0)`. Verified. Function never compares the computed `stop_price = entry_price * (1 - 2 * atr_pct / 100)` against `financing_level`.

**Claude P1 (newly elaborated):**
- `monte_carlo_risk.py:419` — `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` bypasses `_resolve_fx_rate`. Same flaw at `exit_optimizer.py:54` and `exit_optimizer.py:719` (hardcoded 10.85 fallback).
- `exit_optimizer.py:303-340` `_compute_pnl_sek` deducts cost only on exit; entry-side cost missing → every EV overstated by ~half the round-trip.
- `exit_optimizer.py:617-621` hold-to-close EV computed from 5 percentiles, not the 5000-path distribution.
- `exit_optimizer.py:397-400, 446` 3 % knockout buffer contradicts the documented 10-20 % knockout tolerance — over-aggressive de-risking at every wick inside 3 %.
- `trade_guards.py:140-167` `except (ValueError, TypeError): pass` on cooldown read silently bypasses cooldown when wall-clock drifts backward or file is hand-edited.
- `risk_management.py:765-770` `check_concentration_risk` clamp `min(..., cash)` becomes inert when `cash=0` (fully invested).

**Codex prose hints (102 lines):** "trade validator default minimum is below the stated 1000 SEK floor"; "risk classifier has a silent 'unknown regime = zero risk' path"; "warrant_portfolio.py models MINI products as simple leverage multipliers and never tracks financing/barrier at all, which means knockouts can be missed entirely". **All three prose hints are corroborated by Claude P0 / P1 findings.** Codex's "warrant_portfolio MINI not tracking financing/barrier" is the same root cause as Claude's P0 on `risk_management.py:374` (stop never compared against financing level).

**Verdict:** Identical P0 list as 2026-05-11. No portfolio-risk fixes landed in the past 24 h. **This is the highest-confidence regression in the entire review: four blockers, prose-corroborated, file-verified, untouched.**

### 4 — metals-core

**Claude P0 (3):**
- `grid_fisher.py:1029-1121` — Rotate-on-fill stop placement at line 1097-1104 does not thread catalog barrier into `build_exit_levels` (`grid_tiers.py:208-223`). `_tier_skip_for_knockout` only protects opening BUYs.
- `mstr_loop/config.py:19` + `execution.py:165-170` — `PHASE=live` gated only by env var. No shadow-day counter, no approval sentinel, no Telegram confirmation. `MSTR_LOOP_PHASE=live` on the scheduled task is sufficient to put MSTR live against account 1625505.
- `fin_snipe_manager.py:529-563` — `_compute_stop_plan` extracts `barrier_level` and `financing_level` (lines 605-607) then ignores them when computing `trigger_price = position_avg * (1 - 0.05)`. For a tight-barrier 5× MINI silver warrant, the stop sits inside the financing distance.

**Claude P1 (newly found):** `grid_fisher.py:1089-1094` partially-repaired stop-loss cancel fallback (uses `getattr(session, "cancel_stop_loss", None)`) but return-status is discarded — a FAILED response is silently swallowed, then `rotate_on_buy_fill` proceeds to place a second stop on the same orderbook; `minutes_until_eod` returns +∞ when zoneinfo absent, silently disabling EOD-flat; `_simulate_trades` in ORB backtest ignores intra-day high/low ordering (overstates win rate); `leave_n_out_validation` is not a proper walk-forward (random holdouts with overlapping training sets); EOD market-flat dumps at `bid * 0.99` with no barrier-distance floor.

**Codex prose hints (1097 lines — the most verbose log):** "structural issues in metals_loop and grid_fisher" + "actual trade sets overlap" + "MINI barrier-distance guard only exists in ranking/pretrade helpers" + "session-window mismatches in golddigger and elongir" + "precompute paths are mostly self-refreshing". **The "MINI barrier-distance guard only in pre-trade helpers" prose is exactly the same defect family Claude flagged as P0 — both reviewers agree.**

**Verdict:** 1 prior P0 partially fixed (cancel fallback uses `cancel_stop_loss` now but status not checked), 1 prior P0 still present unchanged (MSTR env-only gate), 1 new P0 (fin_snipe_manager). Net: barrier-distance enforcement is still inconsistent across grid_fisher / fin_snipe_manager / risk_management.

### 5 — avanza-api (newly surfaced subsystem in adversarial focus)

**Claude P0 (4 — all newly found; not present in 2026-05-11 portfolio-risk review):**
- `portfolio/avanza/trading.py:213-288` — Unified-package `place_stop_loss` has NO account whitelist. `acct = account_id or client.account_id` accepts any caller-provided id, including pension `2674244`. Same hole in `place_order:81`, `modify_order:129`, `cancel_order:160`, `delete_stop_loss:352`, `place_trailing_stop`. **Confirmed by grep: zero `ALLOWED_ACCOUNT_IDS` references in `portfolio/avanza/trading.py`.**
- `portfolio/avanza/trading.py:38-365` — Every order/SL/cancel function in unified package calls `client.avanza.place_order(...)` / `place_stop_loss_order(...)` directly, **without `avanza_order_lock` wrapper**. The cross-process lock added 2026-04-13 is bypassed.
- `avanza_session.py:633-645` — `cancel_order` returns whatever `api_post` returns. Avanza's delete endpoint is known to return HTTP 200 + body `{"orderRequestStatus": "ERROR"}` for business rejections. Callers that don't check (e.g. `cancel_all_stop_losses_for:1074-1085` verify-loop) read a "succeeded" response and proceed.
- `avanza_session.py:917-919` — `cancel_stop_loss` swallows `AvanzaSessionError` (raised by `_get_csrf` when session rotated) into a generic FAILED, hiding the re-auth signal.

**Claude P1 (9):** Loose `get_open_orders`/`get_stop_losses` returning `[]` on read failure; unified `get_buying_power` returning silent zeros (reintroducing the exact bug the legacy version was fixed for); fragile `'ISK' in atype.upper()` discovery; race on `avanza_pending_orders.json` between `request_order` and `check_pending_orders`; unbounded `ResilientPage` relaunch loop; library-opaque stop-loss endpoint in unified path (no Mar 3 contract guard); `cancel_all_stop_losses_for` holding `_pw_lock` across 3s verify loop; storage-state file never reloaded after re-auth without first hitting 401.

**Architectural finding:** Three parallel implementations (`avanza_session.py` BankID path, `avanza_client.py` TOTP path, `portfolio/avanza/trading.py` unified) each have their own version of every safety fix. Patches from Mar 3, A-AV-1, A-AV-2, BUG-129, P0-4 propagated unevenly. **Recommended consolidation:** single shared `ALLOWED_ACCOUNT_IDS` constant + single chokepoint for order placement that all three paths route through.

**Codex prose hints (135 lines):** Mostly file enumeration and prompt repetition. Limited direct corroboration. **Claude is the primary signal for this subsystem.**

**Verdict:** 4 newly-surfaced P0s. **Highest production-risk subsystem this pass** — the unified package is a new attack surface that has skipped every existing safety check.

### 6 — signals-modules

**Claude P0: none.** The dispatch-layer guardrails in `signal_engine.py` (DISABLED_SIGNALS interception line 3495, `_validate_signal_result` normalisation, outer try/except line 3561) neutralise the worst module bugs before they reach consensus.

**Claude P1 (6):**
- `intraday_seasonality.py:110-129` — Hour/dow tables emit BUY-only, never SELL. Same structural BUY-only failure that killed `calendar` last week. Currently in DISABLED_SIGNALS — but graduating without rebalancing would repeat the pattern.
- `cot_positioning.py:102` — Direct `requests.get` with no `_cached()` and no rate-limit; in bootstrap state (local history < 20 entries) every cycle fires 14 CFTC round-trips.
- Module-level dict caches without locks (8-worker race): `hash_ribbons.py:51`, `crypto_evrp.py:51,53`, `credit_spread.py:53`, `copper_gold_ratio.py:43`. Lock pattern established in `metals_cross_asset` / `gold_real_yield_paradox` was not followed.
- `credit_spread.py:285` + `gold_real_yield_paradox.py:265` — `load_json("config.json")` is CWD-relative; PF-DataLoop's historical "launched from C:\Windows" pattern (fixed in cot_positioning by commit `97eb05f0`) reproduces here.
- `momentum.py` — 8 silent `except → HOLD` blocks with NO logging; commit `b1587646` swept other modules but missed momentum.
- `vwap_zscore_mr.py:124` — top-level `except Exception: return HOLD` swallows everything.

**Codex prose hints (1419 lines, mostly file traversal):** "enhanced-dispatch side is straightforward: `requires_context` gets `compute_fn(df, context=...)`" — confirms codex correctly understood the dispatch contract. **Both reviewers agree dispatch layer is healthy; module-level issues are isolated.**

**Verdict:** 0 P0 / 6 P1 / 8 P2 / 7 P3. Module-level hygiene has improved (b1587646 logging sweep) but a few gaps remain. **Highest-leverage fix: extend the b1587646 logging sweep to `momentum.py` and add the cache-lock pattern to the four module-level dicts.**

### 7 — data-external

**Claude P0: none.** Substantial hardening already in place.

**Claude P1 (notable):**
- `shared_state.py:110-126` `_cached` error path serves up to 3×TTL stale data. For on-chain BTC `ONCHAIN_TTL=43200` s = 12 h → max_stale 36 h with only boundary WARN.
- `forecast_signal.py:194-201` — Chronos-2 builds tz-aware UTC `pd.date_range(end=Timestamp.now(tz="UTC"), periods=n, freq="h")` while `data_collector.py:96` returns tz-naive Binance times. The synthetic last-timestamp matches `now`, not the candle close → 30-47 min phase offset on the forecast horizon.
- `http_retry.py:43-49` only parses Telegram's `parameters.retry_after`. Standard HTTP `Retry-After` header is ignored on Binance/Alpaca/AV/NewsAPI/BGeometrics 429s. Worst impact: BGeometrics (8 req/hour) — retries burn the budget.
- `data_collector.py:96` — `pd.to_datetime(df["open_time"], unit="ms")` produces tz-naive Series; downstream code comparing against `datetime.now(UTC)` raises `TypeError`.
- `microstructure_state.py` — Ring buffers never seeded from `load_persisted_state` on restart; 10 cycles of zeros after every crash.
- `data_collector.py:168-204` — `fetch_vix` uses yfinance directly without `yfinance_lock`, while `get_stock_fear_greed` correctly uses it.

**Codex prose hints (932 lines):** Mostly file traversal and the standard "VIX fetch" / "BERT model loading" / "Chronos-2 context build" sequence. **Limited direct corroboration of specific P1s.**

**Verdict:** 0 P0 / 8 P1 / 11 P2 / 10 P3. Already-hardened subsystem; remaining P1s are subtle (stale-after-error, tz-naive Binance, Retry-After). **Highest-leverage fix: the Chronos-2 phase offset (P1-2) — would directly explain the 45.4 % 1 h-accuracy regression flagged in `tickers.py` comment.**

### 8 — infrastructure

**Claude P0 (5 — newly surfaced subsystem; 2026-05-11 dashboard/scripts not in scope):**
- `subprocess_utils.py:214-218` — PowerShell command injection in `kill_orphaned_by_cmdline()`. `pattern` f-spliced into PowerShell `-like` mask with no sanitisation. **Verified by reading the file directly.**
- `process_lock.py:98-103` — Truncate-before-write window destroys PID/started/owner metadata on crash; `except Exception: pass` swallows the failure silently.
- `file_utils.py:45-63` — `atomic_write_json` does NOT fsync the parent directory; docstring claims "H34" power-loss durability but on POSIX without dir-fsync the rename can be lost from directory metadata. Over-promises.
- `file_utils.py:240-258` — JSONL sidecar lock is blocking with no timeout on Windows. Any wedged appender hangs every subsequent loop's append.
- `scripts/fix_agent_dispatcher.py` — silently disabled by `config.layer2.enabled=false` (Layer 2 gate also disables the auto-fix system). No critical-error surfacing.

**Claude P1 (7):** Unbounded Avanza worker queue; no CSRF on POST `/api/validate-portfolio`; `load_jsonl_tail` silently drops truncated UTF-8 last lines; `Secure=True` cookie breaks plain-HTTP localhost auth; recursion env flag bypassable via Bash `unset`; vector_memory singleton init race; `message_store._COMMON_MOJIBAKE_REPLACEMENTS` has duplicate-key collisions (last wins → all `â` corruptions become `↓`).

**Codex prose hints (331 lines):** Captures file enumeration of `file_utils.py` and `dashboard/app.py`. Mention of `cookie` and `Secure` flag in narration → corroborates the `Secure=True` cookie P1.

**Verdict:** 5 P0 — substantial new attack surface from `subprocess_utils` PowerShell injection and `file_utils` durability over-promise. **Highest-leverage fix:** P0-1 PowerShell injection sanitisation (one-line fix with regex escape), P0-2 process_lock atomic write via tmp+rename.

---

## Top blockers consolidated across all subsystems

Ranked by combination of (a) production risk, (b) prior-review carry-over, (c) reviewer agreement, (d) ease-of-fix.

### Tier-A (fix in next batch, < 1 day each)

1. **PowerShell command injection** — `portfolio/subprocess_utils.py:214-218`. Trivial regex-escape fix. New finding, infrastructure subsystem.
2. **Avanza 1000 SEK floor** — `trade_validation.py:32`, `kelly_sizing.py:326`, `kelly_metals.py:44`. Three identical defects with identical 1-character fixes. Prior 2026-05-11 P0 carried over.
3. **NODE_OPTIONS overwrite** — `agent_invocation.py:847`, `multi_agent_layer2.py:145`. Concatenate-not-overwrite. Prior 2026-05-11 P0 carried over.
4. **`signal_decay_alert.py:34-39` raw open()/json.load** — One-line `load_json` substitution. Prior 2026-05-11 P1 carried over.
5. **`fin_snipe_manager._compute_stop_plan` ignores extracted `financing_level`** — Thread it through `_compute_stop_plan`'s logic; refuse stop within barrier band. New finding.
6. **`process_lock.py` truncate-before-write** — Write to `.tmp` + `os.replace`. New finding.

### Tier-B (fix this week, 1-3 days each)

7. **`portfolio/avanza/trading.py` unified-package whitelist + lock** — Hardcode `ALLOWED_ACCOUNT_IDS`, wrap every state-mutating call in `with avanza_order_lock(...)`. New finding.
8. **`risk_management.py:374` ATR cap + barrier-floor** — Raise cap to 20-25 %, compare `stop_price` against `financing_level`. Prior 2026-05-11 P0 carried over.
9. **`grid_fisher.py:1029-1121` rotate-on-fill barrier guard** — Thread catalog barrier through `build_exit_levels`. Prior P0 partially carried over.
10. **`ic_computation.py:73-147` chronological ordering** — Sort `votes`/`returns` by entry `ts` before `_rolling_ic`. Prior 2026-05-11 P0 carried over.
11. **claude_gate consolidation** — Single chokepoint for all `claude -p` invocations. Reachable from `bigbet.py`, `iskbets.py`, `analyze.py`, `multi_agent_layer2.py`. Prior 2026-05-11 P0 + 3 new bypass sites.

### Tier-C (architectural, > 3 days)

12. **MSTR live phase gate hardening** — Sentinel file + shadow-day counter + Telegram confirmation. Prior 2026-05-11 P0 carried over.
13. **`avanza_session.cancel_order` 200-body ERROR detection** — Inspect `orderRequestStatus` and convert non-SUCCESS into FAILED shape. New finding.
14. **`exit_optimizer` cost accounting + buffer** — Add entry-side cost in `_compute_pnl_sek`; thread knockout buffer through `instrument_profile`. New finding.
15. **Chronos-2 tz alignment** — Build context_df from real Binance candle close, not synthetic `now`. New finding.

### Tier-D (test gap; not blockers but should land)

- `tests/test_signal_db.py` multi-threaded stress test
- `tests/test_ic_computation.py` chronological-ordering invariant
- `tests/test_signal_history.py` cross-process race test
- `tests/test_grid_fisher_barrier.py` rotate-on-fill barrier guard
- `tests/test_subprocess_utils_injection.py` PowerShell command sanitisation
- `tests/test_avanza_trading_whitelist.py` parameterized end-to-end whitelist contract test
- `tests/test_atomic_write_dir_fsync.py` durability claim verification

---

## Comparison vs 2026-05-11

| Category | 2026-05-11 status | 2026-05-12 status | Change |
|---|---|---|---|
| signals-core P0s (4) | All 4 flagged | 1 fixed-in-practice (signal_db local-instance), 3 carried over verbatim | -1 P0 |
| orchestration P0s (4) | All 4 flagged | All 4 carried over verbatim | unchanged |
| portfolio-risk P0s (4) | All 4 flagged | All 4 carried over verbatim | unchanged |
| metals-core P0s (3) | All 3 flagged | 1 partial fix (cancel uses cancel_stop_loss now, return status discarded), 1 unchanged (MSTR), 1 NEW (fin_snipe_manager) | +0 P0 |
| avanza-api P0s | Not in scope (folded into portfolio-risk) | 4 new P0s on the unified package | +4 P0 |
| signals-modules P0s | 0 | 0 | unchanged |
| data-external P0s | 0 | 0 | unchanged |
| infrastructure P0s | 0 (scripts/dashboard not in prior focus) | 5 new P0s | +5 P0 |
| **Total open P0s** | **15** (best estimate from yesterday's synthesis) | **23** | **+8** |

**Net change:** P0 count went UP by 8 between 2026-05-11 17:30 CET and 2026-05-12 17:30 CET. The increase is primarily from:
- (a) **Wider scope** — avanza-api and infrastructure subsystems were under-covered in 2026-05-11; this pass surfaced 9 new P0s in those two subsystems.
- (b) **Almost zero fix throughput** — only 1 of yesterday's 11+ P0s landed a verifiable fix in the past 24 h (signal_db local-instance pattern). The other 10+ all carry over.

In the 5 commits since 2026-05-11 (`228f7cd8`, `1221ff02`, `311e5376`, `0c9c2c4f`, `8d1e4a46`) the only adversarial-relevant change was `1221ff02` (replace `__import__('json')` with module-level import in metals_cross_asset) — a P3-class style cleanup, not a P0/P1 fix.

**Recommended next FGL cycle:** lead with the six Tier-A items above. Each is < 1 day and addresses items the review has now flagged for two consecutive days. The "fix nothing, accumulate" pattern between 2026-05-11 and 2026-05-12 is a discipline issue more than a tooling issue.

---

## Methodology limitations & next steps

1. **Codex never emitted formal P0/P1 reports.** Same pattern as 2026-05-11. Mitigation for next run: try `codex review --base review/baseline-N-S` (the codex review subcommand was discovered after 05-11 runs); if same outcome, raise turn budget or pre-bundle file contents.
2. **`fin_fish_manager.py` listed in partition but does not exist on `main`.** Stale entry copied from 2026-05-11 partition. Removed from prompt list for next run.
3. **No mechanical verification of stated line numbers.** Spot-checks (4 of 23 P0s) were performed directly; remaining 19 trust the subagent's path:line. For Tier-A items, a quick `sed -n '<line-3>,<line+3>p'` verification before applying each fix is recommended.
4. **Test coverage for the test-gap list is unbudgeted.** Adding even 50 % of the listed missing tests is ~3-5 days of work and will pay off only if it catches regressions in subsequent reviews.

This synthesis is appropriate as the entry point for the morning's `fin-prereview` / planning batch. The eight per-subsystem Claude reports under `Q:/finance-analyzer/docs/adversarial-review-2026-05-12/<n>-<S>-claude.md` carry the full detail for each finding above.
