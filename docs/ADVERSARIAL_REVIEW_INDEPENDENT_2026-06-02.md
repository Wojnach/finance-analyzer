# Independent Adversarial Review — 2026-06-02

Main-thread pass, run in parallel with (and deliberately orthogonal to) the 8 subsystem
subagents. Purpose: (1) re-verify prior-pass P0s directly in current code for
`[REPEAT]`/`[RESOLVED]` continuity, (2) forensically root-cause the live accuracy-collapse
alert, (3) verify cross-cutting architectural themes that no single-subsystem reviewer can
see, (4) cross-critique both prior findings and this pass's subagents (a fresh context is
less anchored to either's optimism).

Repo: `Q:\finance-analyzer` (live, post-commit 8ca546dc). Live startup state: 4 unresolved
`accuracy_degradation` critical errors (the collapse below) + 1 OVERDUE pickup
(`LLM-CRYPTOTRADER-72H`).

---

## 1. Continuity — prior 2026-05-26 P0s re-verified in code

| Prior P0 | Current code | Verdict |
|---|---|---|
| `warrant_portfolio.py:96` warrant_pnl LONG-only | line 96 `implied_pnl_pct = underlying_change * leverage`, no `direction` param; catalog has 37/84 SHORT certs | **[REPEAT]** confirmed |
| `risk_management.py` stop math LONG-only | lines 337/377/385/469/912 all `entry*(1-…)` + `current < stop` | **[REPEAT]** confirmed (5 sites) |
| `signal_engine.py` bias double-application | `accuracy_stats.py:849` `normalized_weight = rarity_weight * bias_penalty` AND `signal_engine.py:2866` `weight *= _resolve_bias_penalty(...)` | **[REPEAT]** confirmed — prior fix never landed |
| `accuracy_stats.py:152` SQLite-stale-preference | `load_entries()` returns SQLite whenever `count>0`, no freshness check | **[REPEAT]** confirmed |
| `portfolio_mgr`/`warrant`/`trade_guards` process-local locks | `threading.Lock` still; `jsonl_sidecar_lock` exists, wired only to JSONL appends | **[REPEAT]** confirmed |
| `dashboard/auth.py:175` Bearer skips cookie refresh | Bearer path (224-225) returns without `_refresh_cookie`, BUT cookie/query paths (207/217) do refresh | **[RESOLVED / OVER-RATED]** — see §4 |
| `health.py:165` naive-datetime crash | writers use aware `now(UTC)`, parse try/except, route try/except to 500 | **[RESOLVED]** |
| Mar-Apr auth-detector coverage | `claude_gate` scans stdout+stderr, overrides `auth_error` on exit 0; `agent_invocation` direct Popen scans on completion AND timeout; log-rotation offset-reset present (agent_invocation.py:611-621) | **[RESOLVED]** for main-loop + agent_invocation paths |

Net: the **money-math and accuracy-engine P0s carry forward unfixed** (warrant direction,
stop direction, bias double-apply, SQLite staleness, cross-process locks), while the
**infra/auth P0s from 05-26 were genuinely closed**. The fix pipeline is closing
operational/auth bugs but not the deeper correctness debt — consistent with the prior
synthesis's "[REPEAT] backlog growth" theme.

---

## 2. Accuracy-collapse forensic (the live critical-errors)

Probed `data/accuracy_cache.json` directly for the 4 alerting signals:

```
statistical_jump_regime  1d_recent  acc=0.348  total=388  (buy=197 sell=191)  ← balanced, both fail
econ_calendar            1d_recent  acc=0.355  total=650  (buy=650 sell=0)    ← 100% BUY, zero SELL
momentum_factors (global)1d_recent  acc=0.544  total=559                       ← HEALTHY globally
```

Findings:
- **The drop is real and the measurement is sound.** Cross-validated by signals-core:
  `_diff_against_baseline` compares recent-14d-now vs the recent-14d block in the 14d-ago
  snapshot (apples-to-apples) with a 200-sample floor + 2-SE gate; `blend_accuracy_data`
  uses `max(at,rc)` for the main total and the degradation path reads raw `signal_accuracy`,
  not the blend. So the prior-suspected "blend double-count" is **NOT** the cause.
- **`momentum_factors` global recent is healthy (54.4%)** — the alert is the **per-ticker
  `XAG-USD::momentum_factors`** slice (63.4%→37.1%), an XAG-local regression, not a global
  module failure.
- **`statistical_jump_regime` has a concrete code bug** (signals-modules: neutral-state
  counter increments on both jump directions → fabricated regimes; + SMA-slope vote
  double-count collapsing the 3-vote composite to a single trend follower). Genuine drop
  AND a logic defect — fix the module, don't just gate it.
- **`econ_calendar` is genuine, not a bug** — the 650-BUY/0-SELL skew is Fed-pause behavior:
  with no near-term FOMC/CPI, only `post_event_relief` BUY fires, and BUYs are failing
  market-wide (matches the prior session's logged "BUY_ACCURACY_COLLAPSE 15-33% BUY vs
  56-87% SELL"). My initial hypothesis that the `econ_calendar.py:44` tz bug was *causing*
  the collapse was **refuted** by signals-modules (proximity is computed from wall-clock
  `now(UTC)`, not the tz-shifted ref_date) — recorded honestly in §4.

**The real systemic risk the collapse exposes:** the two automated safety nets meant to
catch exactly this are non-functional — `cusum_accuracy_monitor.update_cusum` is **dead
code** (never called from `backfill_outcomes`; signals-core P0), and `signal_decay_alert`
silently no-ops on schema/type drift (signals-core P1). The 44% accuracy gate IS correctly
force-HOLDing the collapsed signals (verified: 34.8%/35.5% recent are below the gate), so
live trading is protected — but detection relies on the once-daily degradation path alone.

---

## 3. Cross-cutting architectural themes (verified horizontally)

### T1 — Cross-process state corruption is the dominant structural debt
`threading.Lock` is process-local, but ≥5 OS processes (main loop, Layer 2 subprocess,
dashboard, metals_loop, bigbet/iskbets) mutate the same JSON/JSONL. Confirmed across THREE
independent subsystem reviews: portfolio-risk (`portfolio_mgr`/`warrant_portfolio`/
`trade_guards` state), signals-core (`cusum_accuracy_monitor` state, `signal_history`),
infrastructure (`message_throttle`/`prophecy`). The correct primitive exists
(`file_utils.jsonl_sidecar_lock`, cross-process via msvcrt/fcntl) but is wired only into
JSONL appends — never the JSON read-modify-write paths. This is the Bold-7%-loss
lost-update class. **One bounded refactor (~60 LOC) wrapping every state read-modify-write
in the sidecar lock closes the largest single risk surface.**

### T2 — `jsonl_sidecar_lock` itself has a 10s ceiling (the lock you trust can fail)
infrastructure P0: `msvcrt.locking(LK_LOCK,1)` retries only ~10s then raises
`OSError(EDEADLOCK)`; `rotate_jsonl` holds this lock across a 50 MB gzip. So the very
primitive recommended for T1 can time out under load and make concurrent
`atomic_append_jsonl` *lose appends*. **T2 must be fixed (retry-to-deadline + gzip-outside-
lock) before T1's refactor leans on it harder** — ordering matters.

### T3 — SHORT/BEAR direction-blindness is now LIVE, not latent
The catalog holds 37/84 SHORT certs with an explicit `direction:"SHORT"` field, yet
`warrant_pnl`, the entire `exit_optimizer`, and `risk_management` stop math are
unconditionally LONG, and `record_warrant_transaction` never persists `direction`. Holding
one BEAR cert → inverted P&L in dashboard/journal/Telegram + spurious knockout/stop on
rallies + exit engine that only proposes losing exits. This is a half-wired feature
(SHORT entry exists, exits/P&L/stops assume LONG) — per GUIDELINES rule 6 it should be
**disabled at the gate with a TODO** until `direction_sign` is plumbed end-to-end.

### T4 — Per-process rate limiters under-protect a 5-loop Binance/Yahoo fan-out
`shared_state._RateLimiter` instances (`_binance_limiter=600`, `_yfinance_limiter=30`, …)
are module-level = **per-process** token buckets. The 5 loop processes (DataLoop,
MetalsLoop, CryptoLoop, MstrLoop, OilLoop) each believe they own the full budget → combined
up to 5× the intended request rate against a single Binance/Yahoo IP. Confirmed by
data-external (raw un-throttled `requests.get`/yfinance in `crypto_data`/`social_sentiment`/
`sentiment`, no shared budget, missing `yfinance_lock`). Architectural; needs a file/OS-
backed token bucket or process consolidation.

### T5 — tz `.replace(tzinfo=UTC)` antipattern: 30+ sites, mostly benign (downgraded)
Repo-wide grep found 30+ `.replace(tzinfo=UTC)`. Most operate on values stored UTC-naive,
where `.replace` is the correct re-attachment. The dangerous case (input tz-aware non-UTC)
was checked for `econ_calendar` and found benign. **Recording as a hygiene watch-item, not a
bug cluster** — prefer `.astimezone(UTC)` defensively in new code. (This corrects a
tempting over-generalization; see §4.)

---

## 4. Cross-critiques (refuting findings — including my own)

- **Prior P0 `dashboard/auth.py:175` (Bearer skips cookie refresh) was OVER-RATED.** Bearer
  clients are CLI/scripts that re-send the token statelessly every request — they never
  depend on the rolling cookie, so "silent token expiry after 1y" cannot affect them. The
  cookie/query paths DO refresh. Independently confirmed `[RESOLVED]` by the infrastructure
  subagent. Lesson: a "P0" predicated on a code path's assumed dependency should be checked
  against who actually exercises that path.
- **My own hypothesis (econ_calendar tz bug *causes* the accuracy collapse) was REFUTED** by
  signals-modules. Event proximity (`hours_until`) is derived from real `datetime.now(UTC)`,
  not the tz-shifted `ref_date`; the tz handling only touches a 1-day filter edge case. The
  collapse is genuine Fed-pause BUY-failure. Kept the observation (650 BUY/0 SELL) but
  dropped the causal claim — this is exactly the value of an independent pass cross-checked
  by a focused module reader.
- **signals-modules vs signals-core on the collapse are NOT in conflict.** signals-core
  proved the *measurement* sound (drop is real); signals-modules proved one *module*
  (`statistical_jump_regime`) has a counter bug. Both true: real drop + one buggy module.
  Synthesis must not present these as contradictory.
- **econ_calendar prior P1 (tz) reclassified to P2/RESOLVED**, and the prior
  cross_asset_tsmom / treasury_risk_rotation polarity findings **confirmed [REPEAT]** (zero
  of the 14 signals-modules findings from 05-26 were remediated — the implementation
  pipeline has not touched this subsystem).

---

## 5. Independent Top 10 (cross-subsystem, severity-ranked)

| # | Sev | Path | Finding | Status |
|---|-----|------|---------|--------|
| 1 | P0 | `file_utils.py:295` + `log_rotation.rotate_jsonl` | sidecar lock 10s ceiling across 50 MB gzip → silent lost appends in every journal | NEW |
| 2 | P0 | `cusum_accuracy_monitor.py:69` | online degradation detector is dead code — the safety net for THIS collapse observes zero outcomes | NEW |
| 3 | P0 | `warrant_portfolio.py:96` + `exit_optimizer.py` + `record_warrant_transaction` | SHORT/BEAR blindness, LIVE (37/84 certs SHORT) — inverted P&L + losing-only exits | REPEAT+NEW |
| 4 | P0 | `portfolio_mgr`/`warrant_portfolio`/`trade_guards` | process-local locks for cross-process state → lost-update drops trades | REPEAT |
| 5 | P0 | `log_rotation.py:573` `rotate_text` | unlocked non-atomic gzip+truncate of loop_out.txt while loop holds `>>` → unbounded growth or lost auth-detector log lines | NEW |
| 6 | P0 | `main.py:985-992` | LIVE: weekend/off-hours crypto+metals triggers dropped with no autonomous fallback | REPEAT |
| 7 | P0 | `signal_engine.py:2866` + `accuracy_stats.py:849` | bias penalty applied twice → skewed-signal in-direction votes under-weighted | REPEAT |
| 8 | P0 | `accuracy_stats.py:152` | `load_entries` serves stale SQLite, no freshness check → hides/fakes recent accuracy | REPEAT |
| 9 | P1 | `statistical_jump_regime.py:97-110` | neutral-state counter increments both jump directions → fabricated regimes (collapse cause) | NEW |
| 10 | P1 | `portfolio_validator.py:69-90` | end-state-only cash check; no chronological overdraft replay (Feb-2026 Bold gap) | REPEAT |

(`avanza_session.py:143` session-expiry exception propagation — NEW P0 in avanza-api — is
the standout in an otherwise chronically-unfixed subsystem and belongs in any Tier-0 list.)
