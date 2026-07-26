# FGL Adversarial Review 2026-07-26 — Synthesis

Baseline `main @ 597176b2`. 7 of 9 subsystem reviewers reported (metals-core and
infrastructure outstanding at time of writing) + orchestrator independent pass.
**~40 findings. Documentation only — no code changed.**

---

## 1. Executive summary

The codebase is genuinely well-hardened where it has been attacked before. Three
things I expected to find broken were verified **correctly fixed**: the warrant
knockout floor, the Layer-2 silent-outage vector, and the confidence-cap clamp
ordering. Reviewers also disproved eight of their own candidate findings against
live data. Where the system is exposed is not old logic — it is **new safety
machinery with a hole in the one path nobody could execute at the time it was
written.**

Five of the nine P0/P1-critical findings are _wiring_ defects, not logic defects:
the correct implementation exists and production never reaches it.

## 2. P0 master list

| #       | Finding                                                                                                                                                                                                                                                              | Location                                                   | Status                                                                                                                                                               |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0-1    | **Promoted LLM path bypasses context enrichment** — phi4_mini, promoted on BTC/ETH/XAU/XAG, skips the shadow-branch gate that builds the rich context, receiving bare `context_data`. Reproduced prompt: `Asset: XAG-USD (cryptocurrency)`, `Price: $0.00`, 9× `N/A` | `signal_engine.py:4021,4177-4178,3986`                     | **VERIFIED.** Supersedes the same-day `asset_type` fixes (b5d2026b, 597176b2), which covered shadow + backtest only. Dormant behind `local_llm.disabled` + loops-off |
| P0-2    | **Bold equity shows −100% wipeout** for a strategy whose state file doesn't exist — value from empty state (0.0) vs `initial` from the 500K constant                                                                                                                 | `risk_management.py:661-676` → `/api/equity-curve`         | **VERIFIED LIVE.** 343/343 history rows. Actively wrong on the dashboard now                                                                                         |
| P0-3    | **`update_state()` has zero production callers** — the lock-holding write wrapper is unused; `pf.py` does unlocked read-mutate-write on cash                                                                                                                         | `portfolio_mgr.py`, `scripts/pf.py:398,465`                | **VERIFIED.** The "atomic-RMW bypass" meta-theme was never actually closed                                                                                           |
| P0-4    | **`record_warrant_transaction()` has zero production callers** — so VaR/CVaR structurally excludes leveraged exposure and `/api/warrants` is always empty                                                                                                            | `warrant_portfolio.py:183-282`                             | **ACCEPTED.** Knockout math is correct but never invoked                                                                                                             |
| P0-5    | **Broker retry without idempotency key** — browser death after Avanza accepts an order resubmits it; the sibling timeout path explicitly refuses to, for this exact reason                                                                                           | `avanza_session.py:514-546`                                | **ACCEPTED** (static-traced; Avanza intentionally down)                                                                                                              |
| P0-6    | **`get_open_orders` returns `[]` on shape drift** instead of raising, feeding the metals spike-rollback safety decision → restores a full-volume stop over a possibly-live sell. Shape has already drifted twice in 2026                                             | `avanza_session.py:1047-1092` → `metals_loop.py:5578-5601` | **ACCEPTED**                                                                                                                                                         |
| P0-7    | **Real-money oil leg has no freshness gate** — `bar_ts` is computed and forwarded _specifically_ for staleness judgment; grid_fisher never reads it (`grep bar_ts grid_fisher.py` → 0). BZ=F is always yfinance (10-15 min lag)                                      | `grid_fisher.py:2083-2086` vs `oil_grid_signal.py:170-177` | **VERIFIED**                                                                                                                                                         |
| P0-8 | **grid_fisher EOD-flat has no retry** — once `eod_sell_order_id` is set every later tick skips the instrument; a cancelled/partial sell never resets it and the stop was already nulled unconditionally → naked leveraged inventory carried **overnight** | `grid_fisher.py:2231,2280-2288` | **ACCEPTED.** Reachable-but-unexercised (grid_fisher has never had a production fill) |
| P0-9 | **golddigger never persists or cancels its hardware stop id** — a stop from a closed position keeps resting and can fire against a later unrelated position | `golddigger/runner.py:192-212`, `state.py:12-22` | **ACCEPTED** |
| P0-10 | **golddigger EOD flatten gets one ~60s window** — unreachable outside session hours, no retry next day; a failed flatten carries a 20x-leveraged position overnight | `golddigger/bot.py:88-135,187` | **ACCEPTED** |
| P0-11 | **Legacy metals exit protection permanently disabled** — `STOP_ORDER_ENABLED`/`EMERGENCY_SELL_ENABLED` hardcoded False, `emergency_sell()` a no-op; any ob_id outside a 3-entry allowlist lands there with zero exit protection (cited against a real incident: +5.78% -> -1.27%, no exit) | `metals_loop.py:430,436` | **ACCEPTED** |
| P0-12 | **Pytest write-guard can drop production journal appends** — keys on `PYTEST_CURRENT_TEST`, which children inherit. Blast radius includes `grid_fisher`'s **naked-leveraged-position** alarm                                                                         | `file_utils.py:396` + `grid_fisher.py:1813-1830`           | **VERIFIED. Escalated to P0 by the infrastructure reviewer** on blast-radius grounds; proof-of-mechanism at `tests/test_portfolio.py:993-999`                                                                                                        |

## 3. Meta-themes (each = one structural fix closing a class of bugs)

**M1 — Safeguards built, plumbed, then never consumed.** The dominant theme, with
five independent instances: `bar_ts` forwarded for staleness and never read
(P0-7); `update_state` written and never called (P0-3);
`record_warrant_transaction` written and never called (P0-4); the knockout floor
implemented twice and never invoked (P0-4); `_build_llm_context` correct but
skipped on the promoted path (P0-1). Also `_accuracy_tier_mult` — defined,
unit-tested, documented in CLAUDE.md as live, **never called**. _Structural fix:
a "reachability" check — assert in tests that each safety mechanism has at least
one production caller; treat an unreferenced safety function as a failing test,
not dead code._

**M2 — New fixes hole exactly where they couldn't be executed.** Every safety
mechanism added in the last 8 days has a gap on an unrunnable path: the
`asset_type` fix missed the promoted path (loops off), the voter-state fix missed
the second endpoint, `tune_instrument`'s overlap gate no-ops when span data is
missing, the pytest guard drops production writes. All were "verified" — on the
paths that could be run. _Structural fix: when the stack is down, verification
must include a path-coverage argument, not just a passing observation. Name the
paths not exercised._

**M3 — Two sources of truth that disagree.** `/api/control/registry` says phi4 is
SHADOW while `/api/system_status` says GATED_REMOTE_DOWN, rendered in the same
card (VERIFIED LIVE). Registry-vs-legacy `DISABLED_SIGNALS` divergence across 14
modules. `signal_db` SQL accuracy vs `accuracy_stats` Python accuracy (no neutral
band). Three ATR-stop formulas. `is_enabled` vs `is_globally_disabled` overlay
precedence — with a docstring asserting they agree. _Structural fix: one function
per question; delete or delegate the duplicates._

**M4 — Raw sample counts treated as independent evidence.** 338 "3d" outcomes
inside a 6.7-day span. `accuracy_degradation` invented an effective-N correction
for its own SE gate and **never shared it** with the accuracy gates that decide
vote weight and eligibility; its divisor is also a fixed K=20, calibrated for
vote persistence, under-correcting outcome-window overlap by 7× at 1d, 22× at 3d,
36× at 5d. MSTR sub-daily horizons all resolve to one daily close. _Structural
fix: one effective-N function, horizon-aware, applied to every min-samples gate
and weight multiplier — not just the SE._

**M5 — Rejections and near-misses leave no trace.** CSRF and rate-limit
rejections return before `_audit()` on a public write surface, so an attack probe
is forensically invisible. `health.py` reports time-since-_trigger_ as
time-since-_Layer-2-ran_ (192/217 rows were autonomous), and
`scripts/health_check.py` has no threshold branch at all — the operator-facing
Layer-2 silence check _structurally cannot fail_. _Structural fix: audit/alarm on
every path via try/finally; never derive a liveness signal from a proxy event._

**M6 — Fail-open where fail-safe was intended.** `with-herc.sh:51`'s busy-guard
treats a failed SSH query as "not busy" and shuts the machine down, while the
orchestrator's *other* same-day script explicitly treats missing evidence as busy.
`claude_gate._load_config_layer2_enabled()` fails OPEN on a kill switch while the
equivalent in `agent_invocation.py` was hardened to fail CLOSED. `tune_instrument`'s
overlap gate no-ops when span data is missing. `fin_fish.fetch_fx_rate()` returns a
hardcoded 10.0 indistinguishable from a live rate, feeding a real order price.
*Structural fix: for every guard, write down what it does on missing evidence — and
make "unknown" mean "refuse", never "proceed".*

## 4. Notable non-findings (recorded so they aren't re-litigated)

- Warrant knockout floor: **correctly implemented** in two places — prior P0 stale.
- Layer-2 silent outage: every failure mode writes both an `invocations.jsonl` row
  and a `layer2_journal.jsonl` stub; auth-scan runs on both completion and
  timeout-kill paths.
- Confidence-cap clamp ordering: traced end-to-end, no re-inflation bug.
- MIN_VOTERS_METALS=2 quorum: threaded consistently, no per-class bypass.
- Registry promotion/blacklist separation: holds on every path (both statically
  and by live execution).
- `price_source` fail-closed: default genuinely closed; no caller substitutes a
  stale value.
- Dashboard zero-denominator paths: correctly guarded.
- `registry_defaults.py`: regenerated and diffed — **zero semantic drift**.
- XSS, polling races, storage guards, per-URL error map: all clean.
- **Orchestrator's own negative-multiplier hypothesis: REJECTED** by the
  `u_score > 0` guard (see `00-own-pass.md`).

## 5. Remediation roadmap — ordered by "arms on next restart"

Several P0s are dormant _only_ because the stack is intentionally down. Ordering
by that, not by severity label:

**Tier 1 — before any loop restart (arms immediately on `rm data/local_llm.disabled`
or `systemctl --user start pf-*`):**

1. P0-1 promoted-LLM context enrichment (enrich by signal identity, not branch).
2. P0-7 oil-leg freshness gate (read `sig["bar_ts"]`, skip on stale).
3. Pytest-guard discriminator (`+ "pytest" in sys.modules`, log at WARNING).
4. P0-6 `get_open_orders` raise-on-shape-drift.

**Tier 2 — before any real-money trading resumes:**
4b. P0-8/9/10 EOD-flat + stop-id lifecycle (grid_fisher **and** golddigger — both
    carry leveraged inventory overnight on a failed flatten).
4c. P0-11 legacy `POSITIONS` unmanaged-position path (startup assertion at minimum).
4d. `KillSignal=SIGINT` on pf-metalsloop/dataloop/mstrloop so cleanup actually runs
    (precedent already in pf-golddigger.service).
4e. golddigger price-plausibility floor (98/126 dry-run trades at 0.001 SEK). 5. P0-5 idempotency: re-query instead of resubmitting a mutation. 6. P0-3 route `pf.py` through `update_state`. 7. P0-4 wire `record_warrant_transaction`, or make VaR read the real source. 8. Avanza CONFIRM path → BankID functions (currently cannot place a trade at all).

**Tier 3 — data honesty, fix now (affects what you're looking at today):** 9. P0-2 Bold −100% (one-line loader swap). 10. M3 voter-state contradiction on #silver. 11. M5 audit-on-every-path + honest `last_invocation_ts`.

**Tier 4 — measurement validity (blocks the Silver-tuning project):** 12. M4 shared horizon-aware effective-N across all gates. 13. Registry/legacy divergence (finish Phase 4.3) — **prerequisite for using
`tune_instrument --write` at all**, since overlays currently corrupt what the
legacy readers report. 14. `tune_instrument` span-None → hard SKIP; overlay precedence + validation.

## 6. Coverage and honesty notes

- **All 9 subsystems reported.** metals-core and infrastructure arrived after the
  first synthesis draft and are incorporated above; `grid_fisher.py` got a full
  read, while `metals_loop.py` (8011 lines) was covered via two focused
  sub-reviews with spot-verification rather than a single linear read.
- **Declared gaps:** ~2300 lines of `dashboard/app.py`; most of 62 JS files
  (selective XSS check only); `price_targets.py`; `kelly_sizing.py` /
  `kelly_metals.py` (unassigned — partition gap to fix next FGL);
  `reporting.py`, `journal.py`, `digest.py`, `crypto_scheduler.py`,
  `llm_batch.py`, `llama_server.py` (orchestration's declared gap); ~55
  disabled/pending signal modules.
- **Live-evidence limits:** loops intentionally stopped and Avanza intentionally
  down, so avanza-api and metals findings are static-traced, not runtime-observed.
- **Security incident during the review:** a reviewer printed the live
  BGeometrics `api_token` into its own tool-call transcript, self-reported it, and
  did not propagate the value. Low blast radius (free-tier read-only market data,
  and that endpoint is dead) but the credential should be treated as exposed;
  rotation is free. **Surfaced to the user for decision — not actioned here.**
