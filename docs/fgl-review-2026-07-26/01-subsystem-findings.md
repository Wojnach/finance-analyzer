# FGL Adversarial Review 2026-07-26 — Subsystem Findings

Baseline `main @ 597176b2`. Nine parallel review subagents. Findings below are as
reported, each annotated with the orchestrator's **verification status**:

- **[VERIFIED]** — orchestrator independently reproduced the claim against source
  and/or live data.
- **[ACCEPTED]** — traced through source, consistent, not independently reproduced.
- **[LATENT]** — real code path, cannot fire in the current intentionally-down
  configuration.
- **[REJECTED]** — checked and found wrong; kept with the reason, because a review
  that hides its false positives can't be trusted.

Documentation only — no code changed.

---

## 3. portfolio-risk — 6 findings (P0:3 P1:2 P2:1)

_pr-review-toolkit:code-reviewer._ Atomic-I/O primitive sound; the flagship prior
P0 (negative warrant value / no knockout floor) is genuinely **fixed** in both
warrant P&L implementations — that prior finding is now stale. But the
"atomic-RMW bypass" meta-theme is only half-closed, and live data exposed an
actively-wrong number on the dashboard.

- `portfolio/risk_management.py:661-663,672-676` — **P0** — **[VERIFIED]** —
  `log_portfolio_value()` loads Bold state with raw `load_json(bold_path,
default={})` instead of `portfolio_mgr.load_bold_state()`. With
  `portfolio_state_bold.json` absent, `bold={}` → value 0.0, **but**
  `bold_initial` falls back to the `INITIAL_VALUE_DEFAULT` 500_000 constant →
  `bold_pnl_pct = -100.0`. Orchestrator reproduced: **343/343 rows** of
  `data/portfolio_value_history.jsonl` carry `bold_value_sek: 0.0` /
  `bold_pnl_pct: -100.0` while Patient correctly reads 500000.0; Bold state file
  confirmed absent. Served raw to the dashboard by `/api/equity-curve`
  (`dashboard/app.py:1264-1272`) — i.e. the UI has been showing a **total
  wipeout for a strategy that does not exist**, a direct violation of the
  project's first rule. Sibling `check_drawdown:264-265` uses the safe
  `_load_state_from` and is unaffected, so this is a single-function defect, not
  a systemic loader problem. Fix: use the safe loader, or `default=None` and skip
  the metric when state is genuinely absent instead of manufacturing a wipeout.

- `scripts/pf.py:55-78,354-421,432-496` — **P0** — **[VERIFIED]** — the manual
  trade CLI hand-rolls `load_json` / `_atomic_write_json` (no `fsync` before
  `os.replace`) and performs unlocked read→mutate→write on
  `portfolio_state.json` (`cash_sek -= alloc_sek` at 398, `+= net_proceeds` at
  465, unlocked writes at 415/482), with no backup rotation, quarantine, or
  schema validation. Orchestrator confirmed the root cause: **`portfolio_mgr.
update_state()` — the wrapper that holds the lock across load→mutate→write —
  has zero production callers** (grep matches only its own definition). This is
  the open half of the atomic-RMW meta-theme. Two concurrent `pf.py` runs, or one
  racing any other writer, silently lose a trade's cash delta. Fix: route
  `pf.py` through `portfolio_mgr.update_state`.

- `portfolio/warrant_portfolio.py:183-282` + `dashboard/app.py:1700-1708` +
  `portfolio/multi_agent_layer2.py:54-58` + `monte_carlo_risk.compute_portfolio_var`
  — **P0** — **[ACCEPTED]** — `record_warrant_transaction()`, the only writer of
  `data/portfolio_state_warrants.json`, has **zero production callers** (only
  `tests/`), and the file does not exist. Consequences: `/api/warrants` always
  empty; Layer 2's risk sub-agent lists that file among its three data sources
  and will never see warrant data; `compute_portfolio_var` iterates only
  `portfolio_state["holdings"]`, so VaR/CVaR **structurally excludes leveraged
  exposure**. The math is not the bug — the knockout floor is correctly
  implemented in `warrant_portfolio.warrant_pnl` and again in
  `exit_optimizer._compute_pnl_sek:302-339` (`max(exit_warrant_sek, 0)`) —
  correct code that production never invokes. Fix: wire a real caller, or make
  the manual path call it.

- `portfolio/risk_management.py:373-451` vs `:951-1002` vs `:454-568` — **P1** —
  **[ACCEPTED]** — three independent ATR-stop formulas diverge: only
  `compute_stop_levels` applies the 15%-of-price ATR cap and 3% minimum-distance
  floor; `check_atr_stop_proximity` and `compute_probabilistic_stops` use raw
  uncapped ATR. Same ticker, same moment, different stop distance depending on
  the caller. Fix: one shared helper.

- `portfolio/risk_management.py:215-240` — **P1** — **[ACCEPTED]** —
  `_compute_portfolio_value` substitutes `avg_cost_usd` when a live price is
  missing, so a holding silently reads as exactly breakeven. Drawdown and
  concentration therefore go quiet precisely when the feed is down — the moment
  risk limits matter most. Fix: exclude and flag stale rather than substitute.

- `portfolio/equity_curve.py:309-421` — **P2** — **[LATENT]** — FIFO round-trip
  pairing silently drops unmatched SELL shares (loop exits with
  `shares_to_match > 0`; SELL with no prior BUY hits `continue`). Reviewer
  explicitly could not reproduce: both state files have zero transactions, so the
  path is cold. Fix: warn instead of silently continuing.

**Coverage (reviewer's own statement):** read in full — `file_utils.py`,
`portfolio_mgr.py`, `risk_management.py`, `trade_guards.py`, `equity_curve.py`,
`warrant_portfolio.py`, `monte_carlo.py`, `monte_carlo_risk.py`, plus targeted
`exit_optimizer.py` / `scripts/pf.py`. **Not reviewed:** `price_targets.py`
(scope budget). `position_sizing*.py` / `warrants*.py` do not exist; sizing lives
in `kelly_sizing.py` / `kelly_metals.py`, unassigned this pass — **gap to close
in a future FGL**.

---

## 4. avanza-api — 4 findings (P0:2 P1:1 P2:1)

_pr-review-toolkit:silent-failure-hunter._ The BankID session core is
well-engineered (single-worker thread pinning, RLock discipline, mutation-timeout
journaling, fail-closed stop-loss reads). All four defects sit in the connective
tissue. Avanza intentionally dead today, so every finding is static-traced +
corroborated by config/journal history, **not live API replay** — reviewer stated
this limitation unprompted.

- `portfolio/avanza_session.py:514-546` (`_with_browser_recovery`) +
  `portfolio/avanza_resilient_page.py:166-182` — **P0** — **[ACCEPTED]** —
  **retry without idempotency key on mutating calls.** On any
  `is_browser_dead_error()` match the code relaunches Playwright and blindly
  re-executes the same call — `order/new`, `stoploss/new`, `order/delete`,
  `stoploss/{id}` DELETE — none carrying a client idempotency key. A browser
  death _after_ Avanza accepted the order but before the response is read
  (architecturally real: `TargetClosedError` is a CDP signal independent of
  whether the HTTP request landed) submits a **second identical order**. The
  caller sees only the last result and logs "FILLED" from it. Damning contrast:
  the sibling timeout path (`_record_mutation_timeout:114-140`) explicitly
  refuses to auto-retry a timed-out mutation for exactly this reason. Fix: on
  browser-death during POST/DELETE, re-query orders/positions/stops to determine
  whether the call landed instead of resubmitting.

- `portfolio/avanza_session.py:1047-1092` (`get_open_orders`) → consumed by
  `data/metals_loop.py:5578-5601` — **P0** — **[ACCEPTED]** — **silent empty on
  shape drift feeding a safety decision.** Line 1068 does
  `data.get("orders", []) if isinstance(data, dict) else data` with no shape
  assertion, so a 200 whose body lacks `orders` yields `[]` rather than raising.
  This endpoint's shape has **already drifted twice** (2026-03-24 cancel verb,
  2026-07-13 route change — both documented in the same file). Downstream:
  metals spike-rollback decides `spike_terminal = not any(o["orderId"] == …)`,
  so a drift-induced `[]` reads as "order gone", and the code restores the
  original full-volume stop-loss on top of a possibly-live spike sell — double
  encumbrance or a raced exit. The raise path is already fail-safe; only the
  no-exception-wrong-shape case is exposed. Fix: raise when a dict response lacks
  `orders`, mirroring `get_stop_losses_strict:1316-1334`. Secondary: the account
  filter's `"" in (aid, "")` wildcard over-includes orders missing both account
  fields.

- `portfolio/avanza_orders.py:32` → `avanza_control.py:36-45` →
  `avanza_client.py:40-62,285-324` — **P1** — **[VERIFIED]** — the
  Telegram-**CONFIRM** real-money execution path resolves `place_buy_order` /
  `place_sell_order` to the **TOTP client, never the working BankID session**.
  Orchestrator reproduced: `place_buy_order.__module__ == portfolio.avanza_client`
  and `avanza_client._load_credentials()` raises `KeyError` with the live
  (empty) `username`/`password`/`totp_secret`. So every CONFIRMed order fails —
  each looking like a one-off API hiccup rather than a structural break. The
  identical bug class was found and fixed at `dashboard/app.py:2296-2306`
  (2026-05-04) but never applied here. Compounding:
  `tests/test_avanza_orders.py:10-18` pre-seeds those names as MagicMocks at
  import time, so **the suite structurally cannot catch it**. Fix: import from
  `avanza_session` / the BankID `_no_page` wrappers.

- `data/metals_avanza_helpers.py:490-493` — **P2** — **[ACCEPTED]** —
  `delete_stop_loss` treats only `2xx` as success, missing the
  `or http_status == 404` idempotent-cancel exemption that all four sibling
  implementations have and explicitly comment. Reviewer traced every call site
  (`metals_swing_trader.py:963,1591,3267,3275`): return value is discarded or
  only logged, never gates a decision, and the money-critical cancel-before-sell
  path uses the correct 404-tolerant helpers — so impact is a misleading log, not
  a wrong action. Notably `scripts/fin_fish_monitor.py:141-149` documents this
  function as canonical 404-tolerant, which is **wrong**. Fix: add the exemption.

**Coverage:** read in full — `avanza_session.py`, `avanza_orders.py`,
`avanza_client.py`, `avanza_control.py`, `avanza_account_check.py`,
`avanza_order_lock.py`, `avanza_resilient_page.py`, `metals_avanza_helpers.py`,
`scripts/avanza_login.py`, `portfolio/avanza/trading.py`,
`tests/test_avanza_orders.py`, plus relevant sections of `metals_loop.py`,
`metals_swing_trader.py`, `dashboard/app.py`. Live: config shape only (secrets
never printed), session-file timestamps, 328 `avanza_*` journal rows (all trace
to already-fixed or expected causes — no new finding there).

---

_(Remaining subsystems appended as their reviewers report.)_

---

## 5. signals-modules — 8 findings (P0:1 P1:1 P2:3 P3:3)

*pr-review-toolkit:code-reviewer.* Individual plugins are mostly well-hardened
(BUG-NNN guard trails, engine-side `_validate_signal_result` enforcing each
signal's registered `max_confidence` regardless of plugin discipline). The
flagship finding is structural, not a plugin bug — **and it invalidates a fix the
orchestrator declared complete earlier the same day.**

- `portfolio/signal_engine.py:4021` (gate) + `:4177-4178` (fallthrough dispatch) +
  `:3986-3989` (`context_data`) + `:727,741-744` (`_DISABLED_SIGNAL_OVERRIDES`)
  vs `portfolio/signals/phi4_mini_reasoning.py:126-201` — **P0** —
  **[VERIFIED — supersedes the orchestrator's own 2026-07-26 asset_type fix]** —
  The rich LLM context (`_llm_ctx = dict(context_data); _llm_ctx.update(
  _build_llm_context(...))`, `signal_engine.py:4081-4085`) is built **only inside
  the shadow branch** guarded by `if _sig_globally_disabled(...) and not
  _promoted_override:` at line 4021. Its own comment confirms the scope: *"LLM
  shadows are all requires_context=True."* A signal promoted per-ticker via
  `_DISABLED_SIGNAL_OVERRIDES` — which is exactly what `phi4_mini` is on
  BTC-USD/ETH-USD/XAU-USD/XAG-USD since 2026-07-13 — makes
  `_sig_globally_disabled()` return False, never enters that branch, and instead
  falls through to the generic `if entry.get("requires_context"): compute_fn(df,
  context=context_data)` at 4177-4178. `context_data` is built once at 3986 as
  only `{ticker, config, macro, regime, seasonality_profile}` and never mutated.
  Orchestrator reproduced the resulting prompt directly by calling
  `_build_phi4_messages` with that bare dict:
  `Asset: XAG-USD (cryptocurrency)` / `Price: $0.00` / **9 fields `N/A`**.
  So on the promoted path — the only path on which phi4_mini actually votes —
  silver is still described as a cryptocurrency priced at zero with no
  indicators. **This means the orchestrator's same-day fixes (b5d2026b,
  597176b2) closed the shadow/qwen3/ministral path and the backtest path, but not
  this one**; the orchestrator's verification (0/221 backtest rows mentioning
  gold/crypto) had a blind spot, because the backtest exercises
  `scripts/llm_backtest.py`'s own builder while the live promoted path was never
  runnable with the loops stopped. Dormant only because `data/local_llm.disabled`
  (token-freeze flag, 2026-07-02) and the loop-stop both happen to be in effect —
  both documented as routine no-code-change operations. Concrete trigger:
  `rm data/local_llm.disabled` + loop restart → phi4_mini casts a real BUY/SELL
  vote at up to 0.7 confidence into XAG consensus from that prompt. Also
  `phi4_mini_reasoning.py:51-57` still claims the signal is *"NEVER included in
  the active voting consensus"* — false for those 4 tickers since 2026-07-13.
  Fix: enrich by signal identity (`sig_name in _SHADOW_LLM_SIGNALS`), not by
  which branch was entered — merge `_build_llm_context` before **every** dispatch
  of a `_SHADOW_LLM_SIGNALS` member, promoted or not; add a regression test
  asserting the promoted path never sees a context missing
  `asset_type`/`rsi`/`price_usd`; correct the docstring.

- `portfolio/signals/forecast.py:533-548,551-572` + `portfolio/forecast_accuracy.py:122-130`
  — **P1** — **[ACCEPTED]** — `data/forecast_predictions.jsonl` (35 rows) ends
  2026-07-18T20:12:48, *before* the same-day 22:13:59 fix (5a7c0750) that stopped
  labelling EMA-slope fallback votes as "forecast"; the loop has not run since.
  Every accuracy number these functions can return today is therefore 100%
  pre-fix contaminated with **zero** genuine post-fix samples in either the 14d or
  30d window, and the cutoff is a pure `now - days` with no fix-date floor. A
  promote/retire decision made on `--forecast-accuracy` today would use exactly
  the data the fix commit invalidated. Fix: floor the cutoff at the fix timestamp
  until post-fix samples exist; surface the sample count (currently 0).

- `portfolio/signal_engine.py:3116-3124` (`_LLM_ASSET_LABELS`) — **P2** —
  **[VERIFIED]** — the orchestrator's instrument-naming fix is a hardcoded
  `{ticker: label}` dict covering exactly today's 5 tickers, disconnected from the
  `CRYPTO_SYMBOLS`/`METALS_SYMBOLS` sets that already encode class membership.
  Adding a 6th ticker without a matching entry silently regresses to the ambiguous
  class-only label — the very bug fixed twice today. Fix: require a display name
  for every `ALL_TICKERS` entry, or assert coverage in a test so a new ticker
  fails loudly.

- `portfolio/signals/phi4_mini_reasoning.py:129,171` + `portfolio/qwen3_trader.py:113`
  — **P2** — **[VERIFIED]** — both prompt builders still default `asset_type` to
  `"cryptocurrency"`. The root-cause fix was applied at the shared context builder,
  not at these call sites — and `scripts/llm_backtest.py` needed its own separate
  fix in the same commit, proving not every caller uses the shared builder. Any
  future ad-hoc context (notebook, new harness, manual test) reintroduces the bug a
  third time. Fix: default to a neutral non-answer such as
  `"financial instrument"`, never to a specific asset class.

- `portfolio/signals/cot_positioning.py:362-365` — **P2** — **[ACCEPTED]** —
  `_fetch_cot_historical` is called with no `_cached()` wrapper, unlike every
  sibling external-data signal. `data/cot_history.jsonl` has 17 gold + 17 silver
  rows, both under the `< 20` threshold that triggers this branch, so whenever the
  loop runs COT hits the live CFTC endpoint synchronously (15s timeout) **every
  cycle** for both metals. Self-heals in ~3 weekly releases, permanent if the sole
  writer stalls. Fix: wrap in `_cached()` with a 24h TTL.

- `portfolio/signals/cot_positioning.py:375` — **P3** — **[ACCEPTED]** — a
  graduated `cot_conf` (0.4-0.7) is computed, captured, and never used; the
  composite falls back to plain vote-counting. Intended weighting apparently never
  wired. Fix: wire it or delete it.

- `portfolio/signals/finance_llama.py:99-113` — **P3** — **[ACCEPTED]** — the
  docstring lists `asset_type` as a context key the prompt never reads; the
  situation string carries only indicators plus a bare ticker. Low impact today
  (shadow-rotation only) but it shares `_SHADOW_LLM_SIGNALS` membership with
  phi4_mini, so a future promotion inherits **no** instrument-naming fix at all.
  Fix: thread the instrument name in now, before any promotion.

- `portfolio/component_registry.py:1-15` — **P3** — **[VERIFIED]** — module
  docstring claims *"Nothing imports this module yet"*; it is live-consumed by
  `instrument_profile.py:30,117` (unconditionally, **not** behind the
  `signals.use_registry` flag), `dashboard/app.py`, `dashboard/control.py`,
  `dashboard/system_status.py`, and `signal_engine`'s flag-gated helpers. Exactly
  the stale-comment class the protocol warns against. Fix: update it; consider CI
  failing when `registry_defaults.py` predates the last commit touching the
  constants it snapshots, since `instrument_profile.py` consumes it unconditionally.

**Coverage:** sampled `signal_utils.py`, `phi4_mini_reasoning.py`,
`qwen3_trader.py`, `finance_llama.py`, `forecast.py`, `forecast_accuracy.py`,
`crypto_macro.py`, `cot_positioning.py`, `drift_regime_gate.py`,
`amihud_illiquidity_regime.py`, `realized_skewness.py`, `bert_sentiment.py`,
`signal_registry.py` + decisive slices of the engine/registry.
**Not sampled (declared):** `momentum.py`, `mean_reversion.py`, `news_event.py`
body, `econ_calendar.py`, `statistical_jump_regime.py` body, the ~55
disabled/pending modules, `ministral_trader.py`, `meta_trader.py`,
`cryptotrader_lm.py` bodies, `ml_classifier` location.

---

## 2. orchestration — 4 findings (P0:0 P1:3 P2:1)

*pr-review-toolkit:silent-failure-hunter.* Notable **negative** result, worth as
much as a finding: the headline risk — a Layer-2 outage that looks like success —
is genuinely well defended now. Every failure mode (timeout, failed, auth_error,
incomplete, autonomous_failed) writes both an `invocations.jsonl` row and a
`layer2_journal.jsonl` stub, and the auth-log scan runs identically on the
completion and timeout-kill paths with a rotation-truncation guard. No P0. What
remains is a class of *observability* failures one layer up.

- `portfolio/health.py:33-35` — **P1** — **[VERIFIED]** — `update_health()` sets
  `last_invocation_ts = last_trigger_time` on every trigger regardless of whether
  Layer 2 ran, so `check_agent_silence()` and the dashboard's
  `agent_silent`/`agent_silence_seconds` measure **time-since-any-trigger**, not
  time-since-Claude-ran. Live proof: `health_state.json` has the two fields
  byte-identical (`2026-07-18T16:00:20.077969`) while `invocations.jsonl` shows
  **192/217 rows are `status:"autonomous"`** (Layer 2 never invoked) over the same
  window. The correct pattern already exists for `last_invocation_tier`
  (`agent_invocation.py:1491-1496`). Fix: derive from the last real
  invoked/success/auth_error/timeout/failed row.

- `scripts/health_check.py:279-281` — **P1** — **[ACCEPTED]** —
  `check_5_layer2_agent()` reads that same corrupted field and always returns
  `ok` with a reassuring "Last invocation Xm ago"; there is **no threshold branch
  on `inv_age` at all**, so the operator-facing Layer-2 silence check
  *structurally cannot fail*. Fix: honest field + a staleness threshold.

- `portfolio/file_utils.py:388-403` (the pytest guard) — **P1** —
  **[ACCEPTED, independently corroborates the orchestrator pass]** — the guard
  fixed the one pollution case it targets, but live evidence shows the same class
  reached writers it does not cover: 20 `portfolio_state_corrupt` rows in 4
  identical batches of 5 with byte-identical `corrupt-<hash>` suffixes
  referencing non-production `state.json`/`bold.json`; `invocations.jsonl`
  specialist rows reusing PIDs 11112/11113/11114 across 36+ minutes with paired
  success/auth_error rows 40-50ms apart at `duration_s: 0.0`; two
  `accuracy_degradation` contract rows 40ms apart with `cycle_id` 1 and 3. **No
  field distinguishes test artefact from real incident**, and 712 unresolved rows
  will keep tripping the mandatory startup gate indefinitely. Reviewer explicitly
  flagged that the producers of the corrupt/cycle_id rows were **not** code-traced
  (pattern inference from timestamps/format) — attribution unproven, existence
  proven. Fix: audit reachable writers; one-time resolution pass for identifiable
  pre-2026-07-19 artefacts so the unresolved count means something.

- `portfolio/claude_gate.py:164-177` (used 505-507, 561-573) — **P2** —
  **[ACCEPTED]** — `_load_config_layer2_enabled()` fails **OPEN** (returns True)
  on any config read exception, while `agent_invocation.py:1041-1054` was
  explicitly hardened to fail **CLOSED** for the identical failure mode on the
  identical kill switch, with a 2026-06-11 comment explaining why fail-open on a
  kill switch is a bug. Masked today by the hardcoded `CLAUDE_ENABLED = False`
  master switch, so no live impact — but lifting that switch makes a transient
  config read failure silently un-freeze Layer 2 through this path only. Fix:
  same fail-closed pattern.

**Explicitly dropped for lack of substantiation (reviewer's own discipline):**
`loop_contract.py:1505-1589` `_should_suppress_accuracy_degradation()` — the
`new_keys <= known_keys` subset test could theoretically swallow a real
degradation, but no concrete live before/after case could be constructed, so it
was **not** filed as a finding. Recorded here as a mechanism worth a second look.

**Coverage:** read in full — `main.py`, `agent_invocation.py`, `health.py`,
`claude_gate.py`, `loop_contract.py`, `trigger.py`, `autonomous.py`,
`escalation_gate.py`, `escalation_router.py`, `local_llm_gate.py`,
`market_timing.py`, plus targeted `file_utils.py` / `scripts/health_check.py`.
**Declared gap — no findings claimed either way:** `reporting.py`, `journal.py`,
`digest.py`, `crypto_scheduler.py`, `llm_batch.py`, `llama_server.py`.

---

## 9. web-control-registry — 4 findings (P0:0 P1:3 P2:1) — NEW subsystem this pass

*pr-review-toolkit:code-reviewer.* The write surface is soundly built: allowlist
checked before use, `hmac.compare_digest` throughout, the 6/min limiter confirmed
genuinely global (single `ThreadedWSGIServer` process, not per-worker), CSRF
Origin-host compare correctly anchored to `request.host`. Reviewer independently
re-verified **all ~20 fixes from the 2026-07-19 triage as present and correct**
with one exception (below), and correctly declined to re-file the two pre-existing
NOTE/SKIP items (blank-token fail-open, CF-Access-as-authZ) as new findings.

- `portfolio/component_registry.py:294` — **P1** — **[VERIFIED LIVE]** —
  `voter_state()` tests `meta["shadow_llm"]` **before** the per-ticker
  `DISABLED_SIGNAL_OVERRIDES` rescue, so a shadow-flagged signal reports SHADOW
  even when it is live-rescued for that ticker. Orchestrator reproduced against
  the running dashboard: `GET /api/control/registry?ticker=XAG-USD` returns
  phi4_mini with `enabled_default: true`, present in `applicable`, all 7 horizons
  true, yet `voter_state: {"state": "SHADOW", "reason": "shadow-tracked, not
  voting"}` — while `GET /api/system_status` **simultaneously** returns
  `voters.phi4_mini: {"state": "GATED_REMOTE_DOWN", "reason": "...would vote for:
  BTC-USD, ETH-USD, XAG-USD, XAU-USD"}`. `static/js/views/silver.js:258-268`
  renders both objects in the same "Component health" card, so the #silver page
  currently displays two contradictory truth-states for one component. This is
  **triage item 3/20 fixed incompletely**: the 2026-07-19 fix touched
  `system_status.py` and the JS but never the function `/api/control/registry`
  actually serves. Fix: make `voter_state()` ticker-aware (check the rescue before
  `shadow_llm`), or source both pills from `system_status.voters`.

- `dashboard/control.py:195-198,221-224,249-252` — **P1** — **[VERIFIED]** —
  CSRF-fail and rate-limit-fail returns in all three mutating routes happen
  **before any `_audit()` call**, directly contradicting the module docstring's
  promise that *"every action attempt — success or rejected — is appended to
  audit.jsonl"*. Orchestrator confirmed no `_audit` precedes those returns.
  Exceptions between validation and `_audit()` (e.g. `DISABLE_FLAG.touch()`,
  `atomic_write_json()`) also leave no row — nothing wraps write+audit in
  try/finally. Consequence on an internet-exposed write surface: **a rejected
  attempt leaves zero forensic trace**, and a CSRF rejection is exactly the
  signature of an attack probe. Fix: audit on every path via try/finally,
  including rejections.

- `portfolio/component_registry.py:127-131` vs `:164,174-176` — **P1** —
  **[LATENT — overlay file absent, confirmed]** — `is_enabled()` checks the
  `horizons` overlay key **before** `enabled`; `is_globally_disabled()` uses the
  reverse order, and its docstring at 164 falsely claims `enabled` *"takes
  precedence same as is_enabled"* — the two disagree.
  `scripts/tune_instrument.py:401-427` `merge_into_overlay()` only ever writes
  `horizons` and unconditionally overwrites `reason`. Net effect once an overlay
  exists: running `tune_instrument --write --yes` against a signal an operator
  previously killed with `{"enabled": false}` **silently re-enables it at that
  horizon** and stomps the operator's reason string. Same file/mechanism as the
  orchestrator's own `00-own-pass.md` overlay findings, different consequence
  (kill-switch silently undone vs legacy-reader divergence). Fix: check `enabled`
  first; make the merge refuse to touch `horizons` when `enabled: false` exists;
  append to rather than overwrite an operator-authored reason.

- `scripts/tune_instrument.py:90-115,137-160,183-190` — **P2** — **[VERIFIED]** —
  `data_span_days()` returns None on any exception or missing `signal_log.db`;
  `effective_windows()` then returns None; and the overlap gate
  `if windows is not None and windows < min_windows` **silently no-ops** when
  windows is None rather than blocking. So the min-independent-windows defence —
  the entire reason this script was written today — is bypassed by exactly the
  condition most likely to accompany statistically thin data. Only a printed
  "span UNKNOWN" marks it; nothing stops `--write`. Fix: treat
  `span_days is None` as a hard SKIP.

**Explicitly checked and found clean** (recorded so the absence is evidence, not
an omission): XSS sinks — all reviewed JS renders via `textContent`/DOM append,
`esc()` covers the single `innerHTML` site in `prophecy.html`; polling
generation-token guard and `storage.js`/`theme.js` try/catch wrappers correctly in
place; `fetch.js` per-URL error map and `silver-pipeline.js` `avanzaLevel` gating
both correctly block on error. Consistent with the 2026-07-19 fixes holding.

**Coverage:** read in full — `control.py`, `auth.py`, `cf_access.py`,
`system_status.py`, `component_registry.py`, `tune_instrument.py`,
`house_blueprint.py` manifest guard, `silver.js`, `silver-components.js`,
`silver-pipeline.js`, `fetch.js`, `polling.js`, `router.js`, `storage.js`,
`theme.js`, `freshness-banner.js`, `views/control.js`, `prophecy.html`, plus the
traced `app.py` sections and the full 2026-07-19 triage doc.
**Declared gap:** ~2300 of `app.py`'s 2689 lines outside traced sections; most of
the 62 JS files checked selectively for XSS rather than exhaustively.

---

## 8. metals-core — 15 findings (P0:4 P1:9 P2:1 P3:1)

*pr-review-toolkit:code-reviewer.* The real-money warrant stack is **three
loosely-coordinated protection systems** (legacy `POSITIONS` dict, `SwingTrader`,
`grid_fisher`) plus a fourth independent bot (`golddigger`), and every one has at
least one reachable path to an unprotected or stuck leveraged position. Two are
dead-code gates the developers already documented against a real 2026-04-15
incident; two are freshly-found EOD-flat gaps that mirror each other. Important
scoping from the reviewer: grid_fisher's 45k-entry decision log (2026-05-19→07-17)
shows it has **never had a buy filled in production**, so its findings are
reachable-but-unexercised paths, whereas the golddigger and legacy-POSITIONS
findings are corroborated by real trade/incident history.

- `portfolio/grid_fisher.py:2231` — **P0** — **[ACCEPTED]** — `eod_market_flat()`
  sets `eod_sell_order_id` after the first market-sell and every later tick skips
  that instrument while it is non-null. `reconcile_against_live` (995-1016) can
  mark the tier CANCELLED / partially-FILLED but **never resets
  `eod_sell_order_id` nor sets `stop_needs_rearm`**. Compounding: the stop is
  nulled unconditionally *before* the sell is attempted (2280-2288, no check on
  `cancel_stop_loss`'s status), so a silently-failed cancel leaves a live stop
  resting while the market-sell bounces off `short.sell.not.allowed`. Net: any EOD
  sell that doesn't clear in one shot leaves inventory naked — no stop, no resting
  sell — until next day's `roll_session_if_new_day`, i.e. after a full **overnight
  leveraged carry**. Fix: track EOD-sell resolution explicitly; re-trigger for any
  instrument with `inventory_units > 0` and no live resting sell; verify the stop
  cancel actually succeeded.

- `data/metals_loop.py:430,436` — **P0** — **[ACCEPTED]** —
  `STOP_ORDER_ENABLED=False` and `EMERGENCY_SELL_ENABLED=False` hardcoded with no
  override anywhere (grep-confirmed single assignment each); `emergency_sell()` is
  a permanent no-op. The code's own comment (1950-1954) states any position
  landing in the legacy `POSITIONS` dict without explicit `SwingTrader` migration
  gets **zero exit protection**, citing the incident it caused
  (`docs/PLAN-orphan-positions.md`: +5.78% → −1.27%, no exit). Any warrant ob_id
  outside the 3-entry `KNOWN_WARRANT_OB_IDS` allowlist — e.g. after Avanza
  reissues a series — still lands there. Fix: wire the flag or extend migration so
  nothing lands unmanaged; add a startup assertion that fails loudly.

- `portfolio/golddigger/runner.py:192-212` + `state.py:12-22` — **P0** —
  **[ACCEPTED]** — the hardware stop `sl_id` returned by `place_stop_loss` is
  logged but **never persisted** (no such field on `Position`) and **never
  cancelled** by any exit path — TAKE_PROFIT / SIGNAL_EXIT / SESSION_FLATTEN all
  sell via a plain order with zero cancel calls anywhere under
  `portfolio/golddigger/` (grep-confirmed). Failure: position A closes on
  take-profit, its stop keeps resting at trigger T1; position B opens later at a
  different price; price revisits T1 and the stale order fires against B's size at
  a price disconnected from B's risk parameters. Fix: persist the id; cancel it as
  the first step of every non-stop exit.

- `portfolio/golddigger/bot.py:88-135,187` — **P0** — **[ACCEPTED]** — `step()`
  returns early outside session hours, and the EOD flatten check is only reachable
  *inside* `step()` — so flattening a 20×-leveraged position gets one ~60s window
  (~12 tries at poll_seconds=5). If all fail (dead session, CSRF error, `bid<=0`
  early-return at 354-356, `_execute_order` exception) the position is never
  revisited that day and `_should_flatten` is False the next morning: **no forced
  retry, no overnight-carry check independent of the session gate**. Tests cover
  only the happy path. Fix: persist a flatten-pending/failed flag that keeps
  retrying regardless of the session window.

- `portfolio/grid_fisher.py:1682-1839,2199` — **P1** — **[ACCEPTED]** — ladder/stop
  state is mutated in memory right after a successful broker call but
  `self._persist()` only runs at end-of-tick. A crash in between orphans the
  broker order from on-disk state: the tier isn't in `existing_tiers` on restart
  so the same index can be re-placed (doubling notional invisibly, since the
  orphan never enters `planned_notional_sek()`), and a just-placed `stop_loss_id`
  is lost so the next rotation's cancel won't fire → two live stops on one
  position. Fix: persist immediately after each successful broker mutation.

- `portfolio/grid_fisher.py:916-1048` — **P1** — **[ACCEPTED]** —
  `reconcile_against_live` never calls `get_stop_losses()`/`get_stop_losses_strict()`
  (both exist and are used elsewhere in the same loop) to verify `stop_loss_id`
  matches a real resting stop. Combined with the above, a stale id is
  undetectable; the only re-arm trigger is the narrow `stop_needs_rearm` flag, not
  "our belief about the stop may be wrong."

- `data/metals_swing_trader.py:2797` (config `metals_swing_config.py:236-268`) —
  **P1** — **[ACCEPTED]** — the only *broker-resting* stop for a SwingTrader
  position is static and entry-anchored (~−30% warrant move on a 5× cert); the
  1.0-1.5% trailing stop and the momentum/reversal/time exits are in-memory only
  and never re-arm the broker order as they tighten. **If the loop dies, real
  protection reverts from ~1-1.5% to ~30%.** The only external detector
  (`scripts/loop_health_watchdog.py`) is Telegram-alert-only, throttled to 1/4h,
  and never inspects position state — and Telegram is currently muted.

- `data/metals_loop.py:590` — **P1** — **[ACCEPTED]** — `_load_positions` restores
  only the 3 hardcoded `POSITIONS_DEFAULTS` keys; dynamic keys written by
  `_handle_buy_fill` (`silver_queue`/`gold_queue`/`silver_q{N}`) are silently
  dropped on restart even though `_save_positions` wrote them. On rediscovery
  `detect_holdings` treats the position as new and recomputes
  `stop_price = entry*0.95`, **discarding any tighter trailing stop achieved
  pre-crash**.

- `data/metals_loop.py:1686` (writer `portfolio/reporting.py:957,1315`) — **P1** —
  **[ACCEPTED]** — `_vote_detail` is only ever written by the compact/tier2
  summary writers, never by the full-summary writer that `read_signal_data`
  prefers when present (confirmed live: `agent_summary.json`'s `extra` has
  `_votes`, no `_vote_detail`). So `metals_signal_tracker`'s per-signal accuracy
  has been a **permanent silent no-op in production** — `per_signal` always empty.

- `portfolio/golddigger/risk.py:87-139` — **P1** — **[VERIFIED, de-escalated to
  dry-run]** — `size_position()` rejects only `entry_ask <= 0`, with no floor on
  implausibly-small positive prices. Live: `data/golddigger_trades.jsonl` has
  **98 of 126 trades at `price_sek: 0.001`** (2026-03-23 → 2026-05-16), quantities
  up to **9,950,248 units** (~9,950 SEK notional). Orchestrator resolved the
  reviewer's stated condition: the systemd unit runs `--dry-run`, config has
  `trade_enabled: false`, and `--live` is opt-in — so **no real money was at
  risk**. The defect stands: the same bug under `--live` is a multi-million-unit
  order, and ~2 months of dry-run results are garbage. Fix: reject prices below a
  plausibility floor.

- `portfolio/golddigger/runner.py:193-201` — **P1** — **[ACCEPTED]** — when the
  stop trigger is within 3% of bid, hardware-stop placement is silently skipped
  (`logger.warning` only, **no Telegram alert**) unlike the `sl_ok is False`
  branch two lines below which does alert. The operator cannot distinguish "stop
  placed" from "stop skipped, position unprotected."

- `portfolio/instrument_profile.py:117` via `component_registry.py:257-270` —
  **P1** — **[ACCEPTED]** — `_derive_signal_lists`' "trusted" list comes from
  `applicable_signals()`, which checks `is_enabled()` only, **not** `shadow_llm`.
  phi4_mini is globally disabled but per-ticker rescued, so it reads as trusted
  even though `voter_state()` correctly classifies it SHADOW. Confirmed live:
  `format_profile_briefing("XAG-USD")` lists phi4_mini as trusted and
  `fin_fish.py:1207` **stars it in the human-read briefing before a manual
  trade** — a shadow signal presented as trusted. (Same root as web-control's
  finding #1: two methods answering the same question differently.)

- `portfolio/fin_fish.py:307-314` — **P1** — **[ACCEPTED]** — `fetch_fx_rate()`'s
  bare `except: return 10.0` is indistinguishable from a live rate downstream and
  feeds `warrant_price_at_fish`'s real order-price computation plus the printed
  "FX rate" a trader reads before placing a live order.

- `data/metals_loop.py:1035-1091` — **P2** — **[ACCEPTED, corrects the brief]** —
  the 10s fast-tick is single-threaded/serialised with the 60s loop, so **no
  race/re-entrancy** (the brief's hypothesis was wrong). But it has no aggregate
  per-iteration deadline — only per-call HTTP timeouts — so degraded network calls
  stretch the primary stop-check cadence via head-of-line blocking.

- `portfolio/golddigger/bot.py:100,189` — **P3** — **[ACCEPTED]** — logged
  `SESSION_FLATTEN` reason hardcodes "17:20 Stockholm" while the configured
  flatten time is 21:55 — misleads anyone reading the journal after the fact.

**Coverage:** full read of `grid_fisher.py`, `grid_fisher_config.py`, and all five
live state files (all confirm flat/empty). `metals_loop.py`,
`metals_swing_trader.py`, `golddigger/*`, `fin_fish.py`, `instrument_profile.py`
covered via two focused sub-reviews whose top claims the reviewer independently
spot-verified against source.

---

## 7. infrastructure — 8 findings (P0:1 P1:2 P2:4 P3:1)

*pr-review-toolkit:code-reviewer.* The load-bearing primitives are
**structurally sound** — same-lock discipline between `atomic_append_jsonl` and
`rotate_jsonl` genuinely closed the 2026-05-11 lost-append class. Residual risk
is not in the I/O mechanics but in two places code trusts a *process-level*
invariant it doesn't enforce.

**Answer to orchestrator Q1 — confirms the pytest-guard finding, escalates to P0,
and corrects the orchestrator's mechanism:**

- `portfolio/file_utils.py:396` — **P0** (escalated from the orchestrator's P1) —
  **[VERIFIED, mechanism corrected]** — confirmed: env-var check only, no
  `sys.modules` check, and **the guard's own comment already admits the
  inheritance** ("and inherited by Popen children"). Confirmed the naked-position
  alarm path and that the lock-key derivation matches (no mismatch there — that
  part is fine). **Correction to the orchestrator's framing:** "a shell where the
  variable leaked" is wrong — `os.environ` mutation inside pytest does not
  propagate upward to a parent shell. The real mechanism is **fork-time env
  copying**: any child spawned *while a test has the var set* keeps its own copy
  for life, so a detached/unjoined child doing later production work stays blind
  permanently. Reviewer audited every unmocked `Popen`/`run` in `tests/` — all are
  tiny inert `-c` scripts on `tmp_path`, all `.wait()`-joined — **but found a
  concrete proof-of-mechanism**: `tests/test_portfolio.py:993-999` spawns the
  **real** `portfolio/main.py --report` with `env={**os.environ, ...}`, i.e. a
  genuine production entrypoint inheriting the var. Bounded and disabled by
  default (`HERC2_SUBPROCESS_TESTS=1`), so not currently armed — but it proves the
  pattern exists in-repo, one missing `.wait()` from a permanent silent drop.
  Graded P0 on blast-radius/reversibility, the same standard the original P0-3
  this alarm exists to serve was graded on. Fix: require both conditions; log the
  drop at **WARNING**, not debug; add a conftest check failing any test that
  leaves an unjoined Popen past teardown.

**Answer to orchestrator Q2 — no, rotation cannot lose an append:**
traced `rotate_jsonl` (`log_rotation.py:349-507`): the entire
read→classify→archive→write-tmp→`os.replace` runs inside
`with jsonl_sidecar_lock(filepath):`, the same primitive and same lock-path
derivation `atomic_append_jsonl` takes. An append is **excluded by the same
mutex**, not coincidentally non-overlapping. The 2026-05-11 fix holds.

- `~/.config/systemd/user/pf-metalsloop.service` + `data/metals_loop.py:7968-8003`
  — **P1** — **[ACCEPTED]** — zero `signal.signal` calls in the file; cleanup
  (`_kill_claude()`, `stop_llm_thread()`, orchestrator stop, Playwright
  `page.close()`, `release_singleton_lock()`) lives in a `finally:` reachable only
  via `KeyboardInterrupt`. systemd's default `KillSignal=SIGTERM` **skips it** —
  orphaning Chromium and a Claude CLI child on every stop, worse under
  `Restart=always`. In-repo precedent exists: `pf-golddigger.service:11` already
  sets `KillSignal=SIGINT`. Fix: same, or port `crypto_loop.py:299-316`'s
  SIGTERM-handler pattern.
- `pf-dataloop.service` / `pf-mstrloop.service` — **P1** — **[ACCEPTED]** — same
  root cause, lower blast radius, same fix missing.
- `log_rotation.py:473-484` — **P2** — gz-archive merge truncates in place
  (`"wt"`), unlike the atomic tempfile+replace three lines below; a crash there
  loses archived history (not live appends). All 13 archives pass `gzip -t` today.
- `log_rotation.py` `ROTATION_POLICIES` + `main.py:408-409` — **P2** —
  **`critical_errors.jsonl` has no rotation policy** (confirmed unmanaged, 1505
  lines) and `rotate_all()`'s `"unmanaged"` warning key is discarded by `main.py`,
  so it never reaches any journal/dashboard. That file grows forever.
- `scripts/deck/with-herc.sh:51` — **P2** — **[VERIFIED — orchestrator's own
  script, written today]** — the busy-guard **fails open**: a failed or timed-out
  SSH query for `PF-LLMBacktest` is indistinguishable from "not running", so a
  post-WOL connectivity blip (repeatedly observed today) falls through to an
  unconditional `shutdown /s /f /t 0` on line 55 — the exact harm the guard
  exists to prevent, potentially killing another session's GPU campaign. Reviewer
  notes the orchestrator's *other* same-day script,
  `~/herc-idle-shutdown.sh`, gets this right ("GPU unreadable → treat as busy —
  fail safe, never shut down on missing evidence"). Two scripts written hours
  apart, opposite failure postures. Fix: copy the fail-safe pattern.
- `api_utils.py:21-36` — **P3** — mtime-cached config dict returned **by
  reference** to 10+ modules including an 8-worker pool; no current mutator, but
  in-place config mutation is already an idiom elsewhere (`iskbets.py:52,846`,
  `telegram_poller.py:467-468`) — one copy-paste from corruption.

**Dropped after investigation (recorded):** 33+21 identical-timestamp rows in
`critical_errors.jsonl` looked like a torn/duplicate-append bug; all 33
`resolves_ts` values proved distinct — a legitimate batch-resolution pass (the
orchestrator's own), not a defect.

**Coverage:** `file_utils.py`, `log_rotation.py`, `shared_state.py`,
`http_retry.py`, `api_utils.py`, `gpu_gate.py`, `local_llm_gate.py`,
`telegram_notifications.py`, `message_store.py`, `main.py`,
`mstr_loop/__main__.py`, `metals_loop.py`, `crypto_loop.py`, `oil_loop.py`,
`grid_fisher.py`, `claude_gate.py`, all `scripts/deck/*.sh` + `scripts/win/*.ps1`,
all 21 `pf-*` units, 3 test files. **Skipped:** `dashboard/app.py` (other
subsystem), signal-quality modules.
