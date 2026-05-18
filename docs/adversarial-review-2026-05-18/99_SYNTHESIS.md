# Adversarial Review 2026-05-18 — Synthesis

Full multi-subsystem adversarial review of the finance-analyzer codebase. Done per /fgl protocol with worktree on branch `review/full-adversarial-2026-05-18` from `main` at SHA `167af164`.

## Scope and methodology

- Codebase partitioned into 8 subsystems.
- 8 reviewer subagents spawned in parallel against the worktree (background mode).
  - 6 × `caveman:cavecrew-reviewer` (tight diff format)
  - 2 × `pr-review-toolkit:code-reviewer` (broader scope: avanza-api, signals-modules)
- Own independent adversarial pass run alongside.
- Findings cross-critiqued (a few subagent claims demoted or escalated after source verification).

## Reviewer outcomes

| # | Subsystem | Reviewer | P1 | P2 | P3 | Status |
|--|--|--|--|--|--|--|
| 1 | orchestration | caveman:cavecrew-reviewer | 3 | 17 | 0 | Done |
| 2 | portfolio-risk | caveman:cavecrew-reviewer | 2 | 9 | 0 | Done |
| 3 | signals-core | caveman:cavecrew-reviewer | 6 | 9 | 0 | Done |
| 4 | data-external | caveman:cavecrew-reviewer | 3 | 16 | 0 | Done |
| 5 | infrastructure | caveman:cavecrew-reviewer | 0→8* | 16 | 0 | Done (own pass escalated 8 RISK→P1) |
| 6 | metals-core | caveman:cavecrew-reviewer | 3 | 14 | 2 | Done |
| 7 | avanza-api | pr-review-toolkit:code-reviewer | 8 | 10 | 13 | Done (strongest review) |
| 8 | signals-modules | pr-review-toolkit:code-reviewer | — | — | — | **TIMED OUT** (stalled at 6min no progress) |

**Totals across 7 completed subsystems: 33 P1, 91 P2, 15 P3 + own pass (4 verified P1, 3 own P2, 3 own P3).**

Signals-modules (38 plugin files in `portfolio/signals/*.py`) review missing — must be re-run separately. Filed as TODO at bottom.

## Top P1 issues (ordered by blast radius)

### Crash + correctness (orchestration / signals-core)

1. **`agent_invocation.py:1395-1396` — NameError on watchdog completion check** (own-verified)
   `_check_agent_completion_locked` reads `_journal_count_before` / `_telegram_count_before` but they were only set as locals inside `invoke_agent` at line 1059-1060 with no `global` declaration. The function's `global` decl at line 1339-1340 lists the `_ts_` versions but NOT the `_count_` versions. **First post-completion check after 2026-05-17 raises `NameError`.** Likely silently degraded since the count-delta heuristic was added.
   Fix: add `global _journal_count_before, _telegram_count_before` in both `invoke_agent` and `_check_agent_completion_locked`. Initialize module-scope to `0`.

2. **`accuracy_stats.py:930` — Sample-count inflation across overall vs directional gates** (signals-core)
   `total = max(at_samples, rc_samples)` for overall; `total_buy = at_v + rc_v` for directional. Asymmetric: passes overall, fails directional. Affects every accuracy-gated signal vote.

3. **`accuracy_stats.py:1001` — Cache TTL wall-clock** (signals-core)
   `time.time()` for staleness; NTP/DST backward jump → stale cache appears fresh forever. Use `time.monotonic()`.

4. **`ic_computation.py:128` — IC sign inversion asymmetry** (signals-core)
   `ic_sell` negates sum; `ic_buy` does not. Both +1%/-1% return give ic_buy=+1.0, ic_sell=+1.0 → sign convention broken.

5. **`signal_engine.py:1475-2471` — DISABLED_SIGNAL_OVERRIDES not re-checked at consensus gate** (signals-core)
   Rescued signal computed at module-dispatch level but then accuracy-gated at consensus → inconsistent "what voted" contract.

### Silent failures (avanza-api — repeat of March outage class)

6. **`avanza_session.py` request timeouts missing** (P1-1 from avanza)
   `ctx.request.get/post/delete` with no timeout, under `_pw_lock` RLock. Single hung POST blocks all Avanza calls → main loop + metals + grid all stall. Exact silent-outage failure mode.

7. **`avanza_session.py` HTML-200-on-dead-session not detected** (P1-2 + P1-8 from avanza)
   `verify_session()` returns True if Avanza serves SPA shell with HTTP 200 on expired cookies. `_session_client` cached True forever. Same class as the 3-week March 2026 silent Layer 2 outage.

8. **`avanza_session.py` no Telegram alert on `AvanzaSessionError`** (P1-3 from avanza)
   `check_session_expiry()` has no scheduled caller. Operator never paged. Loop just stops trading silently. Repeat of March 2026 grudge.

9. **`avanza_orders.py:53` — `_HEX_TOKEN_RE` no length bound** (P1-4 from avanza)
   Confirm tokens 6 hex chars but regex `^[0-9a-f]+$` accepts any length. Brute-force CONFIRM attempts feasible.

10. **`http_retry.py:50,54,63,67` + `avanza_orders.py:274-278` — Telegram bot token in URL → logged plaintext** (P1-5 from avanza)
    `fetch_with_retry` logs full URL with bot token on retry/warning/error paths. Tokens are in `data/agent.log`. Same incident class as Mar 15 API-key leak grudge.

11. **`http_retry.py:36-37` — 4xx returned as success** (P1-6 from avanza)
    `if resp.status_code not in RETRYABLE_STATUS: return resp`. Caller calls `.json()` on error body.

12. **`avanza_session.py:1086-1109` — `cancel_all_stop_losses_for` FAILED with non-empty snapshot** (P1-7 from avanza)
    Naive rearm path could attempt stops on already-encumbered volume.

### Data integrity (infrastructure / data-external)

13. **`file_utils.py:371-415 prune_jsonl` — no sidecar lock** (own-verified)
    Reads + atomically replaces JSONL with no `jsonl_sidecar_lock` held. Concurrent `atomic_append_jsonl` between read and `os.replace` is LOST. Same divergence class the contract invariant was supposed to detect. Affects `invocations.jsonl`, `signal_log.jsonl`, others (28 callers).

14. **`shared_state.py:88-89` — `_loading_keys.add()` outside try** (infrastructure)
    If `enqueue_fn` raises before try entry, key marked loading forever → deadlock on next cache miss.

15. **`alpha_vantage.py:231 + 281` — quota race** (data-external)
    Budget check unlocked; increment locked. Two threads pass check → double-count quota → API ban risk.

16. **`crypto_macro_data.py:165` — max pain inverted** (data-external)
    Code finds minimum; comment says maximize. Signal direction reversed for options-based macro signal.

17. **`econ_dates.py:155` — event time hardcoded 14:00 UTC** (data-external)
    FOMC actual 19:00 UTC, CPI 13:30 UTC. All macro events off by ±5h. **Likely root cause** of econ_calendar signal accuracy drop 71.2%→33.4% in current `critical_errors.jsonl` entry (2026-05-18T14:29:20).

### Portfolio + risk

18. **`risk_management.py:760-761` — silent None on zero total_value** (own-verified)
    All-cash or net-negative portfolio returns None silently; caller can't distinguish "concentration safe" from "skipped".

19. **`equity_curve.py:366-369` — orphan SELL records** (portfolio-risk)
    SELL with no matching BUY silently skipped. P&L incomplete.

20. **`portfolio_mgr.py:154-159` — backup rotated AFTER mutate_fn** (portfolio-risk)
    Mid-execution crash leaves only previous-cycle backup → recovery hides current broken state.

### Metals + EOD-flat invariant

21. **`grid_fisher.py:129-135, 1545-1552` — `eod_sell_order_id` never cleared on fill** (metals-core)
    Position carried overnight against EOD-flat invariant. Direct violation of grid-fisher spec.

22. **`metals_loop.py:1099-1131` — fast-tick fallback to stale price with no staleness check** (metals-core)
    Memory rule `feedback_live_prices_first.md` directly violated. False entry/exit signals during Binance hiccups.

23. **`metals_loop.py:251-268` — warrant `_TRADING_MINUTES=820` used regardless of evaluation time** (metals-core)
    Pre-market (06:00 CET) Monte-Carlo simulations annualize wrong.

### Infrastructure data races

24. **`health.py:23-41`, `digest.py:47-52`, `journal.py:28`, `message_throttle.py:44-66`, `regime_alerts.py:90`, `prophecy.py:72`, `gpu_gate.py:216`** — Multiple TOCTOU read-modify-write races + missing sidecar locks (infrastructure, own escalations).

## Cross-cutting themes

### Theme A — Silent failure pattern (8+ findings)
The system has a documented grudge: claude CLI exited 0 while printing "Not logged in". Multiple new findings of the same pattern in avanza-api, data-external, infrastructure. Suggests insufficient discipline at boundary checks.
**Recommended hardening:** every external boundary must check (a) content-type, (b) shape, (c) explicit success marker, NOT just HTTP status.

### Theme B — Wall-clock time in cooldowns / TTLs (5+ findings)
- `accuracy_stats.py:1001` (TTL)
- `shared_state.py:334` (newsapi daily reset)
- `trigger.py:376-378` (flip cooldown)
- `main.py:607` (ticker pool timeout)
- `telegram_poller.py:221` (restart bypass)

NTP adjustments, DST flips, and suspend/resume all cause incorrect behavior.
**Recommended:** project-wide audit; switch internal cooldowns to `time.monotonic()`; wall-clock reserved for human-readable timestamps and scheduling that genuinely needs calendar.

### Theme C — Sidecar lock not used for full-file rewrites (file_utils.prune_jsonl, regime_alerts append, journal.load_recent)
The sidecar lock was added for `atomic_append_jsonl` ↔ `log_rotation` interaction in 2026-05-11. Same lock should wrap any read+rewrite cycle on a JSONL file. **Audit + fix all such call sites.**

### Theme D — Stale-cache / fail-open silent paths (7+ findings)
- `_session_client = True` cached forever (avanza-api)
- `_estimate_volatility` default 20% if missing (metals)
- `_fx_cache` fallback 10.50 (fx_rates)
- Price-source fallback (price_source.py:235)
- `avg_cost_usd` fallback (risk_management.py:752-758)
- `_underlying_prices` stale fallback (metals_loop.py:1099)
- `_effective_global_cap` 60s cache (grid_fisher.py)

Pattern: silent fallback without telemetry. **Each fallback should log + post a non-spammy alert.**

### Theme E — Concentration / position-sizing depends on stale prices
After-hours stock trading + fallback to `avg_cost_usd` for missing prices means 5-20% blind spots in concentration checks. Caused by external boundary problems above. Cluster of: risk_management.py:752-758, warrant_portfolio.py:237-246, equity_curve.py:413.

## Cross-critique (vs subagents)

- Subagent claimed `signal_engine.py:3933` is a ZeroDivisionError P1. After verification: today gated by `active_voters < min_voters` at 3929. Demoted to P3 robustness ("future refactor risk"). Documented in own pass.
- Subagent claimed `agent_invocation.py:1395` NameError P1. After verification: REAL NameError. Confirmed P1. Found additional related local-shadowing issue at line 1061-1062 for the `_ts_` versions (own P2).
- Infrastructure reviewer marked all 24 findings as RISK with no P1. Own pass escalated 8 to P1 based on blast radius (deadlock potential, JSONL torn lines, in-memory/on-disk skew). The reviewer was under-graded for system-reliability impact.
- All other subagent findings spot-checked plausible; not exhaustively verified.

## Recommended actions (ordered)

### Immediate (this week)
1. **Add `global` decls** to fix the agent_invocation NameError (#1) — single-line fix.
2. **Wrap `prune_jsonl` in `jsonl_sidecar_lock`** (#13) — 1-line edit.
3. **Add request timeouts** to `avanza_session.py` `ctx.request.*` (#6) — order-of-magnitude reliability win.
4. **Scrub bot tokens from `http_retry.py` log lines** (#10) — secret leak.
5. **Fix `_HEX_TOKEN_RE` anchor** (#9) — `^[0-9a-f]{6}$`.
6. **Fix `crypto_macro_data.py:165` max pain inversion** (#16) — flip comparator.
7. **Fix `econ_dates.py:155` hardcoded times** (#17) — likely root cause of current critical_errors entry.

### Soon (2 weeks)
8. Detect HTML-on-200 in `verify_session()` + `api_get` (#7-#8). Wire Telegram alert on `AvanzaSessionError` (#8).
9. Fix `accuracy_stats.py:930` sample-count asymmetry (#2) + `ic_computation.py:128` IC sign (#4).
10. Wall-clock → monotonic project-wide audit (Theme B).
11. Sidecar-lock all JSONL read+rewrite call sites (Theme C).
12. Fix `risk_management.py:760` silent None (#18). Fix `equity_curve.py:366-369` orphan SELL (#19).

### Backlog (research / non-urgent)
13. `eod_sell_order_id` clearing path for fast fills (#21).
14. Warrant `_TRADING_MINUTES` parameterize by evaluation time (#23).
15. Concentration check using stale prices (Theme E).
16. RE-RUN signals-modules adversarial review (stalled subagent).

## Notes / housekeeping

- Critical errors as of session start: `accuracy_degradation` 2026-05-18T14:29:20 with 5 signals dropped >15pp incl `econ_calendar 71.2%→33.4%`. **Likely root cause = #17 (hardcoded UTC times)**. Suggest resolving the critical_errors entry with this finding attached.
- Per /fgl rules + user instruction, no implementation in this session. All findings are advisory until a follow-up implementation session.
- Signal-modules subsystem (38 files) **NOT REVIEWED**. Re-run with a dedicated background agent or split per-module.
- Worktree: `Q:/fa-review-2026-05-18` on branch `review/full-adversarial-2026-05-18`. To be cleaned up post-commit.

## Per-subsystem detail reports

- `01_orchestration.md`
- `02_portfolio_risk.md`
- `03_signals_core.md`
- `04_data_external.md`
- `05_infrastructure.md`
- `06_metals_core.md`
- `07_avanza_api.md`
- `08_own_independent.md`
