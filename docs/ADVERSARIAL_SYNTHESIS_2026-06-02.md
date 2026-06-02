# Adversarial Review Synthesis — 2026-06-02

**Inputs (9):**
- 8 subagent reviews (`docs/AGENT_REVIEW_*_2026-06-02.md`) — fresh pass, fresh Claude Code subagents
  (7× `pr-review-toolkit:code-reviewer`, 1× `caveman:cavecrew-reviewer` for avanza-api)
- 1 independent main-thread pass (`docs/ADVERSARIAL_REVIEW_INDEPENDENT_2026-06-02.md`)
- Cross-reference to the 2026-05-26 pass — `[REPEAT]` tracks unfixed items, `[RESOLVED]` confirms fixes.

**Method:** the whole codebase was partitioned into 8 subsystems, each committed onto an
**empty-baseline branch** (`review-<sub>-2026-06-02` rooted at git's empty tree) inside a
throwaway worktree, so every reviewer saw its subsystem as a clean add-only diff. Reviewers
were seeded with the prior pass's P0s per subsystem and the live `accuracy_degradation`
alerts, and instructed to tag `[REPEAT]`/`[RESOLVED]`/`[NEW]`.

**Totals (raw, pre-dedup):** ~92 findings · **P0 ≈ 24** (incl. 3 dormant landmines) ·
**P1 ≈ 31** · **P2 ≈ 32**. After cross-subsystem dedup: **~22 distinct P0**.

---

## Executive Summary

Four structural classes dominate this pass:

1. **The atomic-I/O foundation has a crack (NEW, highest-leverage).** `file_utils.jsonl_sidecar_lock`
   — the one cross-process primitive every subsystem is told to trust — uses
   `msvcrt.locking(LK_LOCK,1)`, which gives up after ~10 s and raises `OSError`. `log_rotation.rotate_jsonl`
   holds that lock across a 50 MB gzip, so under load concurrent `atomic_append_jsonl` calls
   raise and **silently lose journal appends** (signal_log, critical_errors, claude_invocations…).
   Separately, `rotate_text` truncates `loop_out.txt` with no lock at all while the loop holds it
   open via `>>`. The thing that's supposed to make everything else safe is itself unsafe — and
   it must be fixed *before* leaning on it harder (see Theme 1).

2. **SHORT/BEAR direction-blindness is now LIVE, not latent.** The warrant catalog holds **37 of 84
   SHORT certs** with an explicit `direction:"SHORT"` field, but `warrant_pnl` (line 96), the entire
   `exit_optimizer`, and `risk_management`'s stop math are unconditionally LONG, and
   `record_warrant_transaction` never persists `direction`. Hold one BEAR cert → inverted P&L
   everywhere + spurious knockout/stop on rallies + an exit engine that only proposes losing exits.
   This is a half-wired feature; per GUIDELINES rule 6 it should be **gated off with a TODO** until
   `direction_sign` is plumbed end-to-end.

3. **The accuracy-degradation safety nets are unplugged — during a live collapse.** Four signals
   tripped `accuracy_degradation` (>15pp below 50%) this session. The drop is *real and correctly
   measured* (signals-core proved the diff is apples-to-apples and `blend` doesn't double-count),
   the 44% gate IS force-HOLDing them (live trading protected) — but the two automated detectors
   meant to catch this early are non-functional: `cusum_accuracy_monitor.update_cusum` is **dead
   code** (never called from `backfill_outcomes`), and `signal_decay_alert` silently no-ops on
   schema drift. And one collapsing signal, `statistical_jump_regime`, has a *concrete code bug*
   (neutral-state counter increments on both jump directions → fabricated regimes).

4. **Cross-process state corruption remains the dominant correctness debt (`[REPEAT]`).** Five+ OS
   processes mutate the same JSON state under process-local `threading.Lock`; `atomic_write_json`
   prevents torn files but not lost updates. Confirmed independently in portfolio-risk, signals-core,
   and infrastructure. This is the Bold-7%-loss class.

**Backlog signal:** the money-math and accuracy-engine P0s from 05-26 carry forward **unfixed**
(warrant direction, stop direction, bias double-apply, SQLite staleness, cross-process locks), while
the infra/auth P0s from 05-26 were **genuinely closed** (Bearer cookie refresh, health datetime,
auth-detector log-rotation reset). The fix pipeline is closing operational/auth bugs but not the
deeper correctness debt.

---

## Top 20 Highest-Impact Items (cross-subsystem dedup, ranked)

| # | Sev | Path | Finding | Source | Status |
|---|-----|------|---------|--------|--------|
| 1 | P0 | `file_utils.py:295` + `log_rotation.rotate_jsonl` | sidecar lock 10s ceiling held across 50 MB gzip → concurrent `atomic_append_jsonl` raises → silent lost appends in every journal | INFRA + INDEP | NEW |
| 2 | P0 | `warrant_portfolio.py:96` + `exit_optimizer.py` + `record_warrant_transaction` | SHORT/BEAR blindness, LIVE (37/84 certs SHORT): inverted P&L, losing-only exits, no `direction` persisted | PORTFOLIO-RISK | REPEAT+NEW |
| 3 | P0 | `portfolio_mgr`/`warrant_portfolio`/`trade_guards` | process-local lock for cross-process state → lost-update drops trades | PORTFOLIO-RISK | REPEAT |
| 4 | P0 | `cusum_accuracy_monitor.py:69` | online degradation detector is dead code — the safety net for THIS collapse observes zero outcomes | SIGNALS-CORE | NEW |
| 5 | P0 | `data/metals_loop.py:1086,1417` | silver fast-tick reads only legacy POSITIONS → all −3%→−12.5% exit alerts dead for swing-managed silver | METALS-CORE | REPEAT |
| 6 | P0 | `data/metals_swing_trader.py:2760` | stop-sell limit only 1% below trigger → naked 5x position on a wick | METALS-CORE | REPEAT |
| 7 | P0 | `data/metals_loop.py:6765,7723` | raw claude Popen, log tail never scanned → exit-0 "Not logged in" silently recorded as success | METALS-CORE | REPEAT |
| 8 | P0 | `log_rotation.py:573` `rotate_text` | unlocked non-atomic gzip+truncate of loop_out.txt while loop holds `>>` → unbounded growth or lost auth-detector lines | INFRA | NEW |
| 9 | P0 | `main.py:985-992` | LIVE: weekend/off-hours crypto+metals triggers dropped, no autonomous fallback | ORCHESTRATION | REPEAT |
| 10 | P0 | `signal_engine.py:2866` + `accuracy_stats.py:849` | bias penalty applied twice → skewed-signal in-direction votes under-weighted | SIGNALS-CORE | REPEAT |
| 11 | P0 | `accuracy_stats.py:152` | `load_entries` serves stale SQLite, no freshness check → hides/fakes recent accuracy | SIGNALS-CORE | REPEAT |
| 12 | P0 | `avanza_session.py:143` | `load_session()` raises on expiry, `_get_playwright_context()` doesn't catch → propagates to retry loops = silent auth outage | AVANZA-API | NEW |
| 13 | P0 | `earnings_calendar.py:174` + `alpha_vantage.py:94,150` | AV `"Information"` throttle unrecognized → 24h cache-of-None → earnings BUY-gate disabled all day | DATA-EXTERNAL | REPEAT |
| 14 | P0 | `microstructure_state.py:205-213` | rolling deques not persisted/restored → ~5-10 min of 0.0 OFI/spread z-scores after every metals_loop restart | DATA-EXTERNAL | REPEAT |
| 15 | P0 | `portfolio_validator.py:69-90` | end-state-only cash check; no chronological overdraft replay (the Feb-2026 Bold gap) | PORTFOLIO-RISK | REPEAT |
| 16 | P0 | `monte_carlo_risk.py:204,228,408` | `trading_days=365` hardcoded + raw `fx_rate` bypass → SEK VaR/CVaR understated up to ~10x | PORTFOLIO-RISK | REPEAT |
| 17 | P0 | `data/metals_loop.py:1051` | cycle-overrun drops all fast-ticks; velocity deque has no time anchor → false "RAPID DROP" + mis-windowed flushes | METALS-CORE | REPEAT |
| 18 | P1 | `signals/statistical_jump_regime.py:97-110` | neutral-state counter increments on both jump directions → fabricated regimes (concrete cause of its collapse) | SIGNALS-MODULES | NEW |
| 19 | P0* | `agent_invocation.py:402-423` + `reporting.py:1191` | DORMANT: `_no_position_skip` reads `signals` from a file that never has it → kills every new entry when `no_position_skip_enabled` flips on | ORCHESTRATION | REPEAT |
| 20 | P0* | `agent_invocation.py:1079` | DORMANT: `specialist_timeout_s=30s` < specialists' 90-120s → multi-agent quorum-fail when `multi_agent` flips on | ORCHESTRATION | REPEAT |

\* dormant = guarded by a config flag that is currently off; arms the moment the flag flips.

---

## Cross-Cutting Themes

### Theme 1 — Fix the lock before leaning on it (ORDERING MATTERS)
The recommended remedy for the cross-process state debt (Theme 4) is "wrap read-modify-write in
`jsonl_sidecar_lock`." But infrastructure found that *lock* gives up after ~10 s and is held across a
50 MB gzip in `rotate_jsonl`. So the sequence must be: **(a)** make `jsonl_sidecar_lock` a true
blocking lock (retry-to-deadline) and move the gzip/archive write outside the held region; **(b)**
*then* retrofit the state save sites onto it. Doing (b) first would expand the blast radius of (a).

### Theme 2 — Two automated degradation detectors are non-functional
`cusum_accuracy_monitor` (online, 3-7 obs) is dead code; `signal_decay_alert` silently returns `[]`
on schema drift; and `accuracy_degradation._summary_diffs` (the Telegram daily) fires WITHOUT the
sample-floor/2-SE gate the contract path enforces → it shouts false drops on tiny samples while the
real online detector is asleep. The once-daily `check_degradation` path is currently the *only*
working detector — and it caught this collapse, so the floor held, but the early-warning layer is gone.

### Theme 3 — SHORT/BEAR direction-blindness (5 sites, now LIVE)
`warrant_pnl:96`, `exit_optimizer` (P&L + candidate generation + knockout), `risk_management` stop
math (337/377/385/469/912), `record_warrant_transaction` (no `direction` field), `monte_carlo`. 37/84
certs are SHORT. Plumb a `direction_sign` (LONG=+1, SHORT=−1) onto the holding at write time; sign-flip
every P&L/stop/knockout site; until then, gate SHORT-cert entries off with a TODO.

### Theme 4 — Cross-process state corruption (`[REPEAT]`, dominant correctness debt)
`portfolio_mgr`, `warrant_portfolio`, `trade_guards`, `cusum_accuracy_monitor`, `message_throttle`,
`prophecy` all read-modify-write shared state under process-local locks. ~60 LOC sidecar-lock
retrofit (after Theme 1) closes the largest single risk surface.

### Theme 5 — Per-process rate limiters under-protect a 5-loop fan-out
`shared_state._RateLimiter` is module-level = per-process; 5 loop processes each assume the full
Binance/Yahoo budget → up to 5× the intended request rate to one IP. Plus raw un-throttled
`requests.get`/yfinance in `crypto_data`/`social_sentiment`/`sentiment` (missing `yfinance_lock`).
Needs a file/OS-backed token bucket or process consolidation (architectural, multi-session).

### Theme 6 — Silent Telegram delivery losses
weekly_digest routes through the `layer1_messages`-gated `send_telegram` (never sent);
`message_throttle._send_now` drops the message + resets the 3h cooldown on send failure with no retry.
The operator is blinded with no error surfaced.

---

## Cross-Critiques (what this pass refuted)

- **Prior P0 `dashboard/auth.py:175` (Bearer skips cookie refresh) was OVER-RATED → `[RESOLVED]`.**
  Bearer clients re-send the token statelessly and never depend on the rolling cookie; the cookie/query
  paths DO refresh. Confirmed by both the independent pass and the infrastructure subagent.
- **The independent pass's own hypothesis — that `econ_calendar.py:44`'s tz bug *causes* the accuracy
  collapse — was REFUTED by signals-modules.** Event proximity is computed from real `datetime.now(UTC)`,
  not the tz-shifted `ref_date`. The collapse is genuine Fed-pause BUY-failure (650 BUY / 0 SELL recent
  is pause behavior: only `post_event_relief` BUY fires). The tz finding drops to P2/RESOLVED.
- **signals-modules vs signals-core on the collapse are complementary, not contradictory.** signals-core
  proved the *measurement* is sound (drop is real); signals-modules proved one *module* has a counter
  bug. Both hold: real drop + one buggy module + two dead detectors.
- **`momentum_factors` global recent is healthy (54.4%);** the alert is the per-ticker `XAG-USD` slice.
  Separately, the *metals* tracker (`metals_signal_tracker.py:506`) has its own all-time-not-rolling
  measurement bug — distinct from the (sound) main pipeline.
- **tz `.replace(tzinfo=UTC)` is NOT a 30-site bug cluster.** Most sites attach UTC to UTC-naive values
  correctly; recorded as a hygiene watch-item, not a finding.

---

## [REPEAT] Backlog Tracking

```
Subsystem        | P0+P1 | [REPEAT] | NEW   | notable RESOLVED
-----------------+-------+----------+-------+------------------
portfolio-risk   |  12   |   6      |  6    | —
orchestration    |   6   |   5      |  1    | auth-detector log-rotation, crash recovery
signals-core     |   7   |   4      |  3    | —
signals-modules  |   7   |   2      |  5    | econ_calendar tz (reclassified)
metals-core      |   9   |   6      |  3    | —
data-external    |   8   |   3      |  5    | fx locale-flip (RESOLVED)
infrastructure   |   6   |   0      |  6    | auth Bearer + health datetime (RESOLVED)
avanza-api       |   5   |   1      |  4    | —
-----------------+-------+----------+-------+------------------
TOTALS           |  60   |  27      | 33    |
```

~45% of P0/P1 carry forward unfixed. The money-math subsystems (portfolio-risk, metals-core) have the
highest [REPEAT] density; infrastructure and avanza-api are mostly NEW (those subsystems either got
fixed or got freshly scrutinized). **Recommendation: block one [REPEAT]-only closure session** targeting
the SHORT/BEAR theme (#2/#3) and the lock/state themes (#1/#3 above) — they cover ~10 of the 27 repeats.

---

## Recommended Action Items (next session)

### Tier 0 — Ship this week (cheap, high-impact, well-bounded)
1. `file_utils.py:295` — make `jsonl_sidecar_lock` retry-to-deadline (not 10s give-up); move `rotate_jsonl`
   gzip OUTSIDE the held lock. **(P0, foundational — do FIRST)**
2. `cusum_accuracy_monitor` — wire `update_cusum` into `backfill_outcomes`'s per-outcome loop, or delete
   the dead module + its tests. **(P0)**
3. `data/metals_loop.py:1086` — union swing-trader silver positions into `_has_active_silver()` so exit
   fast-ticks fire. **(P0)**
4. `data/metals_swing_trader.py:2760` — widen stop-sell buffer to ~3%. **(P0)**
5. `main.py:990` — fall through to `autonomous_decision(...)` in the off-hours `else`. **(P0, LIVE)**
6. `data/metals_loop.py:7723` — `detect_auth_failure(tail, "metals_loop_complete")` in the completion branch. **(P0)**
7. `avanza_session.py:143` — try/except around `load_session()` in `_get_playwright_context()`. **(P0)**
8. `log_rotation.py:573` `rotate_text` — switch the loop to a Python `RotatingFileHandler` (drop bat `>>`), or lock it. **(P0)**

### Tier 1 — One-session refactor (medium, structural)
9. SHORT/BEAR direction: persist `direction` in `record_warrant_transaction`; sign-flip `warrant_pnl` +
   `exit_optimizer` + `risk_management` stops; until done, gate SHORT-cert entries off with a TODO. **(P0, ~200 LOC)**
10. Cross-process state locks: wrap `portfolio_mgr`/`warrant_portfolio`/`trade_guards` read-modify-write in
    the (now-fixed) sidecar lock. **(P0, ~60 LOC — AFTER item 1)**
11. `accuracy_stats.load_entries` — SQLite/JSONL freshness compare + `critical_errors` on divergence. **(P0)**
12. `signal_engine.py:2866` / `accuracy_stats.py:849` — set `normalized_weight = rarity_weight` only; make
    `_resolve_bias_penalty` the sole bias application + regression test. **(P0)**

### Tier 2 — Multi-session / architectural
13. Shared cross-process Binance/Yahoo token bucket (file/OS-backed) or loop consolidation. **(Theme 5)**
14. Monte-Carlo: per-asset `trading_days`; route fx through `_resolve_fx_rate`. **(P0 understatement)**
15. `portfolio_validator` chronological overdraft replay. **(P0, Feb-Bold gap)**

### Tier 3 — Module + hygiene
16. `statistical_jump_regime` neutral-state counter + SMA-slope vote double-count. **(collapse cause)**
17. `cross_asset_tsmom` / `treasury_risk_rotation` / `copper_gold_ratio` / `xtrend_equity_spillover`
    polarity + journal-mismatch fixes (`[REPEAT]`, zero remediated since 05-26).
18. `metals_signal_tracker.py:506` apply the rolling `ACCURACY_WINDOW`; `:643` add min-sample gate.

---

## Historical Continuity

| Past Incident | This pass's finding(s) that could recur it |
|---------------|---------------------------------------------|
| Mar-Apr 2026 silent auth outage (3 wk) | #7 metals_loop no auth scan; #12 avanza session-expiry propagation (main-loop/agent_invocation paths RESOLVED) |
| Mar 3 stop-loss instant-fill | #2/#6 wrong-direction & thin-buffer stop math |
| Feb 18-19 CLAUDECODE outage (34h) | None — defenses verified intact (`_clean_env` pops CLAUDECODE) |
| Bold −7% loss Feb 11-18 | #15 validator no per-tx overdraft replay still |
| BUG-178 silent ticker hangs | #17 cycle-overrun fast-tick drop; orchestration `_run_cycle_id` reset |
| Journal/state corruption class | #1 sidecar-lock lost appends; #3 cross-process lost-update |

---

## Methodology Notes

This synthesis does NOT propose code changes — that is a fix-plan's job. All 8 subagents completed
(metals-core ran longest at ~11 min; signals-core ~7 min). Each reviewed a clean add-only diff of its
subsystem via an empty-baseline branch in a throwaway worktree (cleaned up after). The independent pass
cross-validated 4 items in code directly (warrant direction, bias double-apply, SQLite staleness, stop
math) and **refuted 2** (Bearer over-rated; its own econ_calendar-tz hypothesis), confirming the
subagents' priorities are not prompt-anchoring artifacts. The live `accuracy_degradation` alert was
forensically traced to a genuine market regression + one buggy module + two dead detectors — not a
measurement artifact.

*Source documents:* `docs/AGENT_REVIEW_{SIGNALS_CORE,ORCHESTRATION,PORTFOLIO_RISK,METALS_CORE,AVANZA_API,SIGNALS_MODULES,DATA_EXTERNAL,INFRASTRUCTURE}_2026-06-02.md` + `docs/ADVERSARIAL_REVIEW_INDEPENDENT_2026-06-02.md`.
