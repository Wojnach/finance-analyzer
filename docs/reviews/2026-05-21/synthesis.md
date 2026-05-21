# /fgl Adversarial Review — Synthesis

Date: 2026-05-21
Baseline SHA: `604f0ef1` (PLAN commit on `main`)
Reviewers:
- Lead (`lead-review.md`) — independent fresh-read
- 8 subsystem subagents (`{signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure}.md`)

Total findings across all reviewers: 300+ (full counts in per-file documents).

---

## Top 14 P0 — fix-before-next-deploy

Findings flagged P0 by 2+ reviewers OR with concrete real-money / silent-failure path.

### 0. STOPS ARE BARRIER-BLIND across every metals code path (NEW — METALS-CORE LEAD FINDING)
**Flagged by:** metals-core (multiple P0 sites), portfolio-risk (warrant_portfolio no `financing_level`), lead (defense-in-depth at `place_stop_loss`).
**Where:**
- `portfolio/fin_snipe_manager.py:476` — `Position(financing_level=None)` passed even though `_summarize_market` already reads `barrierLevel`/`financingLevel` at 606-607.
- `portfolio/fin_snipe_manager.py:536` — `_compute_stop_plan` sets trigger at `position_avg * (1 − HARD_STOP_CERT_PCT)` with NO barrier reference. With 5x leverage a 5% warrant drop is only ~1% underlying — barrier rule structurally unreachable.
- `portfolio/grid_tiers.py:222` — `build_exit_levels` stops at `fill_price * (1 - 3.5%)`; knockout safety enforced only on BUY entries, never on stops.
- `data/metals_loop.py:2258` — reactivation hard-codes `pos["stop"] = entry * 0.95` ignoring barrier.
- `data/metals_loop.py:4902` — hardware stop only checks ≥3% below bid, not barrier-translated underlying distance.
- `data/metals_loop.py:2469` — stop-loss limit at trigger − 1% lands INSIDE the spread on VON BEAR GULD (2.2% spread); stop fires but never fills.
- `portfolio/exit_optimizer.py:644-646` — default fallback stop = `financing_level * 1.03`, EXACTLY the forbidden 3%-of-barrier band.
- `portfolio/iskbets.py:404` — Stage-1 hit moves stop to breakeven without barrier validation.
- `portfolio/fin_fish.py:732` — BEAR MINI knockout guard has `pass` instead of `continue` → knocked-out BEAR certs still ranked.
- `portfolio/fish_instrument_finder.py:151-174` — auto-discovery has no minimum-barrier-distance filter; picks closest-to-barrier MINIs by tightest spread.
- `portfolio/warrant_portfolio.py` — no `financing_level` field exists on holdings, so no consumer can be barrier-aware even if they wanted to.
**Causal chain:** User's hardcoded rule (`memory/feedback_mini_stoploss.md`): "NEVER place a stop-loss within 3% of current bid / near MINI warrant barriers." Multiple code paths place stops that look fine in cert-price space (5% below entry) but in underlying space are 0.7-1% from barrier — instant knockout risk. One barrier-proximity event in production = total position loss. This is the #1 finding of the entire review.
**Fix:** systemic — add a single `validate_stop_barrier_distance(stop_trigger_px, current_bid, barrier_level, leverage) -> bool` helper. Wire it into `place_stop_loss` itself so EVERY call site is gated. Persist `financing_level` in warrant_portfolio. Fix the `pass` typo. Add min-barrier-distance filter to fish_instrument_finder.

### 1. Layer 2 Bash + Edit/Write tools are an unrestricted exfil & corruption surface
**Flagged by:** lead, orchestration.
**Where:** `portfolio/agent_invocation.py:1015-1020` (`--allowedTools "Edit,Read,Bash,Write"` + `cwd=BASE_DIR`).
**Causal chain:** prompt-injection into a journal entry, news feed, or any file the agent `cat`s could `rm`, `git push --force`, exfiltrate `config.json` (symlink to API keys), or directly mutate `portfolio_state.json` bypassing atomic_write_json. The same surface let Mar-15 key leak happen via different vector.
**Fix:** drop `Edit,Write` from allow-list. Confine `Bash` to a whitelisted wrapper script (`scripts/agent_bash_safe.sh`) that only permits read-side operations + journal helpers.

### 2. "Incomplete" stub journal entry poisons accuracy stats with fake HOLDs
**Flagged by:** lead, orchestration.
**Where:** `portfolio/agent_invocation.py:1503-1518`.
**Causal chain:** Layer 2 subprocess timeout/crash → completion handler writes `{decisions: {patient: {action: "HOLD"}, bold: {action: "HOLD"}}}` into `layer2_journal.jsonl`. Analytics consume this as truth → Layer 2 HOLD rate inflates, accuracy_stats train on imaginary HOLD decisions, the agent loses any signal from its own failure mode. Orchestration also flags downstream `journal_written=True` poisoning fishing-context on next cycle.
**Fix:** action="NO_DECISION" (or omit decisions) so consumers distinguish "agent chose HOLD" from "agent crashed".

### 3. Layer 2 silent contract violations TODAY (root cause likely SKIP-path → loop_contract blind spot)
**Flagged by:** lead, orchestration, environment.
**Where:** `data/critical_errors.jsonl` shows 7 `contract_violation caller=layer2_journal_activity` entries on 2026-05-21 for XAU/XAG/ETH/BTC triggers. Lead identifies the structural cause: trigger fires → invoke_agent() returns False due to a gate (drawdown/trade_guards/no_position_skip/perception_gate/auth_cooldown/disabled) → no journal entry → loop_contract sees the gap and alarms.
**Fix:** thread SKIP outcomes back into loop_contract's "expected silence" set, OR emit a non-decision journal note for every gated-skip so the contract sees the trigger was acknowledged.

### 4. Bear-MINI long-bias in exit/risk computations
**Flagged by:** lead (P1), portfolio-risk (P0, "BEAR warrant knock-out distance computed without abs()").
**Where:** `portfolio/exit_optimizer.py:374, 431-451, 444, 568-598`; `portfolio/risk_management.py:374` (no barrier awareness); `portfolio/warrant_portfolio.py:1-266` (no `financing_level` persisted).
**Causal chain:** distance/knockout/stop math assumes financing_level < price. For Bear MINI (financing > price) the signed distance is negative, every evaluation force-exits or computes p_knockout on the wrong tail. Today's universe doesn't trade Bear MINIs, but the warrant catalog includes them; the moment one is added every Bear position becomes a force-exit landmine.
**Fix:** persist `financing_level_usd` in `record_warrant_transaction`. Add `side` field to Position. Branch distance/p_knockout on direction. Refuse stop within 3% of barrier at `place_stop_loss` boundary too (not just the exit optimizer).

### 5. FIFO round-trip P&L double-counts buy fees
**Flagged by:** portfolio-risk (P0).
**Where:** `portfolio/equity_curve.py:346-405`.
**Causal chain:** BUY `total_sek` per validator already embeds fee. Code derives `price_per_share = total_sek / shares` (fee-inflated) then ALSO subtracts `buy_fee_share` from pnl. Every round-trip understates by ~1× buy fee. profit_factor and Calmar are wrong; performance reporting biased pessimistic.
**Fix:** `buy_price_per_share = (total_sek - fee_sek) / shares` so cross-leg subtraction handles fees once.

### 6. 252 vs 365 trading-days annualization for crypto/metals
**Flagged by:** portfolio-risk (P0).
**Where:** `portfolio/monte_carlo.py:154,217-218`; `monte_carlo_risk.py`; `exit_optimizer.py:173`; `price_targets.py`. `equity_curve.py:23` already uses 365.
**Causal chain:** BTC/ETH/XAG/XAU trade 365 days/year. T = horizon_days/252 inflates T by 1.45×, vol-term by ~1.20×. EVERY MC stop-hit probability, exit fill_prob, VaR/CVaR, and price band for crypto/metals biased outward ~20%.
**Fix:** parameterize annualization per instrument class; reuse `equity_curve.ANNUALIZATION_DAYS=365` constant.

### 7. Signal core gate uses pre-persistence-filter counters → emits inconsistent raw vs weighted action
**Flagged by:** signals-core (P0 — multiple lines: 4011, 2562, 4116, 2602, 4239).
**Where:** `portfolio/signal_engine.py`.
**Causal chain:** Core gate at 4011 uses `active_voters` BEFORE persistence filter; `_raw_action` published at 4251 with that count is consumed by reporting/triggers, while weighted_action at 4239 uses post-persistence count. Reporting and triggers can fire on a raw consensus the weighted path force-HOLDs. Plus accuracy gate (2562) lets samples=0 + default 0.5 accuracy pass; per-ticker override (4116) leaks global directional sample counts; weighting (2602) uses 20 sample min while gating uses 30. Cumulatively: actively-wrong signals can vote at their failure rate.
**Fix:** unify the two paths (apply persistence filter before gate counters); force-HOLD on samples=0; drop directional fields from per-ticker override when ticker stats lack them; align min-sample thresholds.

### 8. Atomic-I/O contract violations — raw `open()` + `json.load()` on config.json
**Flagged by:** infrastructure (P0 — multiple paths), orchestration (P0 — claude_gate).
**Where:** `portfolio/api_utils.py:30-35`; `dashboard/auth.py:62`; `portfolio/claude_gate.py:148-161`. CLAUDE.md rule #4 violated.
**Causal chain:** `config.json` is a symlink to an external file with API keys; `telegram_poller.py /mode` handler atomically rewrites it. Mid-rename Windows read can hit ERROR_SHARING_VIOLATION or partial bytes (telegram_poller itself notes this). On read failure dashboard `_get_dashboard_token()` returns None → `require_auth` lets all requests through. Layer 2 fail-open returns True for `enabled` even though config might say false.
**Fix:** route ALL config reads through `portfolio.file_utils.load_json`. Make missing token a hard fail in dashboard, not pass-through.

### 9. Sustained-flip duration gate effectively neutered at 600s cadence
**Flagged by:** lead (P1), orchestration (P1).
**Where:** `portfolio/trigger.py:106` `SUSTAINED_DURATION_S = 120`.
**Causal chain:** loop cadence bumped from 60s to 600s but duration gate stayed 120s. As soon as cycle 2 sees a flip, mono_now-mono_start > 120s → fires regardless of count. The 3-cycle "sustained" guarantee collapses to "1 cycle confirmation". Combined with the lead's note that the OR rather than AND between count and duration also weakens the guarantee independently.
**Fix:** raise `SUSTAINED_DURATION_S` to 900s+ (1.5× cadence) AND change `count_ok or duration_ok` to require both.

### 10. Atomic-append vs prune race; rotate_text without sidecar lock
**Flagged by:** infrastructure (P0).
**Where:** `portfolio/file_utils.py:269-292, 379` (`prune_jsonl` skips sidecar lock); `portfolio/log_rotation.py:439` (`rotate_text` truncates with `"w"`).
**Causal chain:** appenders take the sidecar lock; prune_jsonl does NOT → an in-flight append-in-progress can be dropped on the prune's write-tmp+replace. rotate_text simply truncates a live log → concurrent appenders lose data. CLAUDE.md rule #4 (atomic I/O) violated where it matters most for journal integrity.
**Fix:** `prune_jsonl` acquires `jsonl_sidecar_lock` for the entire read-rewrite-replace. `rotate_text` reads-then-replace atomically, also under a sidecar lock.

### 11. Dashboard cookie = raw `dashboard_token` (master secret) + SameSite=Lax + 365d
**Flagged by:** infrastructure (P0).
**Where:** `dashboard/auth.py:98,154-159`; `dashboard/app.py:780-799` `/logout` unauth.
**Causal chain:** Cookie value is the master token used for Bearer/Query auth too. `?token=…` first-visit leaks the secret to browser history, referer, upstream proxy logs. SameSite=Lax + master token = CSRF logout primitive at minimum; ID compromise + 365d expiry = indefinite access.
**Fix:** server-side HMAC-signed cookie (`HMAC(server_secret, user_id || expiry)`) instead of echoing master. SameSite=Strict. Hard alert when `?token=` arrives (recommend cycle).

### 12. Outcome-tracker quantization conflicts with neutral threshold; missing change_pct → default 0 silently neutral
**Flagged by:** signals-core (P0 — `outcome_tracker.py:472, 458`; `accuracy_stats.py:225, 1336`).
**Causal chain:** `change_pct` rounded to 2 decimals, but `_MIN_CHANGE_PCT = 0.05` — a true 0.045% move rounds to 0.04, gets treated as neutral and SKIPPED. A 0.055% move rounds to 0.06 and counts. Borderline outcomes are randomly excluded from accuracy depending on rounding direction. Plus `outcome.get("change_pct", 0)` defaults missing data to 0 → silently treated as benign neutral. Plus cache_key `(ticker, int(target_ts // 3600))` shares price across different horizons in same hour. Cumulatively: training data biased toward larger moves, accuracy stats systematically undercounting marginal outcomes.
**Fix:** store unrounded change_pct (or 4-decimal); drop the default-0 (use `outcome.get("change_pct")` + None-guard); cache by minute-precision target_ts.

---

## High-impact P1 findings

### Orchestration / Layer 2
- **Stack-overflow auto-disable is permanent** (orch, lead). 5 crashes → Layer 2 dead forever, no decay. After a Claude CLI bad-release this leaves the system mute.
- **Auth-cooldown logic breaks on success-after-failure pattern** (orch P0). Walks last-50 backwards, breaks on first non-skip. A single recent `success` after a flurry of `auth_error` re-opens the spawn floodgate.
- **`count_jsonl_lines` heuristic accepts non-JSON appends** (orch P0). A panicking write of `"oops"\n` increments the count → `journal_written=True` even though content is garbage.
- **`_kill_overrun_agent` nulls `_agent_proc` even when kill fails** (lead, orch). Other watchers see "no work" while OS process is alive.
- **`_completion_lock` held across 15-25s subprocess wait()** (orch P1). Watchdog blocks during the exact kill sequence we need to observe.

### Signals
- **3d/5d/10d horizons collapse to 1d accuracy** (signals-core P1, explicit TODO at engine:4035). Long-horizon trades gated on 1d data.
- **MIN_VOTERS_METALS=2 vs `.claude/rules/signals.md` which states 3** (lead, signals-core). Inconsistent rule-of-record.
- **`_adx_cache` keyed on (len, first_close, last_close) → content collision** (signals-core P1). Two distinct frames same-boundary collide; stale ADX served.
- **MSTR reads BTC `_cross_ticker_consensus` from PREVIOUS cycle** (signals-core P1). BTC proxy systematically lags by one cycle.
- **`_topn_accuracy_key` falls back to 0.5 default → ties nondeterministic** (signals-core P1). Top-N signal selection unstable across runs.

### Portfolio / Risk
- **`update_state` lock bypassed by legacy `load_state`/`save_state`** (portfolio-risk P0). Two `update_state`s cannot race, but `update_state` + legacy `save_state` can; the latter wins, losing the mutation.
- **Validator: 1% share tolerance on close-out** (portfolio-risk P1). 0.1 BTC ≈ $8K vanishing silently.
- **`check_drawdown` recovers without hysteresis** (portfolio-risk P1). Whiplash between guarded and unguarded; consumers must dedupe.
- **`check_drawdown` agent_summary empty → cash-only value → false drawdown** (portfolio-risk P1). When the price feed stales while holdings are underwater, the breach reports an 80% drawdown that's actually a feed issue.
- **ATR stop never trails** (portfolio-risk P1). Position +30% then back to entry → full give-back.
- **t-copula df hardcoded to 4** (portfolio-risk P1). Never fit from data; stressed crypto needs df≈3.
- **252-trading-day mismatch** (already P0 #6).

### Avanza API
- **Singleton `_session_client` never re-verifies** (avanza-api P1). 401 mid-loop → flag stays True forever.
- **Session expiry off-by-one (`<=` vs `<`)** (avanza-api P1). 1-second early reauth.
- **`rearm_stop_losses_from_snapshot` doesn't re-validate account_id** (avanza-api P1). Corrupted snapshot could route stops to the (disallowed) pension account 2674244.
- **`place_buy_order` swallows `OrderLockBusyError`** (avanza-api P1). Indistinguishable from API failure.

### Infrastructure
- **`load_json` silent defaults on OSError** (infra P0). Antivirus lock → empty portfolio.
- **`gpu_gate._pid_alive` returns True when psutil missing** (infra P1). The 25h-outage failure mode.
- **`http_retry` "full jitter" is actually half jitter** (infra P1). Comment on commit fd64c7cd misleading.
- **`feature_normalizer._buffers` no lock under 8-worker concurrency** (infra P1).
- **Flask `static/` route bypasses `@require_auth`** (infra P1).
- **Telegram token redaction regex misses non-standard token shapes** (infra P1).
- **`message_throttle` non-atomic read+update can double-send** (infra P1).

---

## Cross-reviewer consensus (multi-reviewer hits)

These findings appeared in 2+ reviews and command extra weight.

| Finding | Reviewers |
|---------|-----------|
| Layer 2 `Edit,Write,Bash` allow-list too broad | lead, orchestration |
| Incomplete-journal stub HOLD pollutes accuracy | lead, orchestration |
| Bear-MINI long-bias in exit/risk math | lead, portfolio-risk |
| 252 vs 365 day annualization mismatch | portfolio-risk, lead (implicit) |
| Atomic-I/O contract violations on config.json | infrastructure, orchestration, claude_gate |
| `MIN_VOTERS_METALS` violates documented rule | lead, signals-core |
| `last_jsonl_entry` 4KB tail truncation | lead, infrastructure |
| Sustained duration gate weakened at 600s cadence | lead, orchestration |
| `_kill_overrun_agent` nulls `_agent_proc` on kill failure | lead, orchestration |
| ATR-based stop not barrier-aware for warrants | lead, portfolio-risk |
| Stack-overflow auto-disable permanent | lead, orchestration |

---

## Findings the lead pass missed but subagents caught

The fresh-context subagents found bugs the lead pass did not, particularly in deep code paths:

1. **`equity_curve.py:346-405` double-fee P&L** (portfolio-risk only). Lead read exit_optimizer.py but did not read equity_curve P&L matching.
2. **252 vs 365 day annualization** (portfolio-risk). Lead noted ATR stop math but missed the annualization constant.
3. **outcome_tracker quantization → neutral-threshold conflict** (signals-core). Lead did not read outcome_tracker.
4. **Persistence-filter consistency in core consensus gate** (signals-core). Lead surveyed signal_engine.py at module level but did not trace the raw vs weighted action publishing race.
5. **`count_jsonl_lines` accepts garbage appends as `journal_written`** (orchestration). Lead read the count-delta heuristic but missed that any newline-bearing write counts.
6. **`gpu_gate._pid_alive` True when psutil missing** (infrastructure). Lead skipped gpu_gate.
7. **Cookie = master token + SameSite=Lax** (infrastructure). Lead noted CSRF risk but did not trace the cookie value back to the master secret.
8. **t-copula df hardcoded** (portfolio-risk). Lead did not read monte_carlo_risk.
9. **Per-ticker accuracy override leaks global directional sample counts** (signals-core). Lead noted ticker accuracy as a class-of-bug but did not find this specific leak.
10. **Telegram offset advance before sender-auth check** (avanza-api). Lead did not read avanza_orders.py confirmation flow.

This is the value of the multi-reviewer pattern: depth-then-breadth.

## Findings the subagents missed but lead caught

1. **Trigger contract violations TODAY (root cause is SKIP-paths)** — lead surfaced this from the critical_errors journal context that subagents lacked. Orchestration noted the contract_violation pattern but did not connect to the SKIP-path explanation.
2. **`PF_HEADLESS_AGENT` recursion risk** — the headless agent dutifully logs the silent failures it itself is causing, never escalating. Lead-only.
3. **Multi-agent T2 30s blocking call inside main loop** — lead-only. Orchestration noted some Layer 2 timing but did not flag this as a main-loop stall risk.
4. **Trigger `_today_str` UTC vs CET** — first-of-day T3 promotion misaligned with user's local day. Lead-only.

## Severity counts

| Reviewer | P0 | P1 | P2 | Notes |
|----------|----|----|----|-------|
| Lead | 14 | ~16 | ~17 | Mixed-severity bands |
| signals-core | 19 | 51 | (P2/P3 filtered) | 70 total |
| orchestration | 9 | 27 | 26 | 63 total |
| portfolio-risk | 6+ | 27 | 18 | 51 total |
| metals-core | 9 | 21 | 18 | 48 total — BARRIER-BLIND STOPS dominant theme |
| data-external | 4 | 24 | 18 | 46 total |
| infrastructure | 7 | 12+ | rest | 40+ total |
| avanza-api | 5 (P1) | 10 (P2) | — | 15 total, smaller scope |
| **signals-modules** | _pending_ | | | |

**Total (so far): 73 P0 + 188 P1 + 100+ P2/P3 = ~360 findings**

---

## Backlog feed (route to `docs/IMPROVEMENT_BACKLOG.md`)

P0 + multi-reviewer P1 findings should become immediate backlog items. The full breakdown:

**Tier 1 (this week — fix-before-next-warrant-trade):**
- **F0: Barrier-aware stop-loss helper, gated at `place_stop_loss` (THE TOP PRIORITY — touches metals_loop, grid_tiers, fin_snipe_manager, exit_optimizer, iskbets, fin_fish, warrant_portfolio).**
- F1: Layer 2 tool allow-list hardening (Edit,Write,Bash → restricted).
- F2: Stub-incomplete journal entry → NO_DECISION.
- F3: Loop_contract SKIP-path bridging (resolves the 7 contract_violations TODAY).
- F4: Bear-MINI long-bias guards + financing_level persistence in warrant_portfolio.
- F5: Double-fee P&L fix in equity_curve.
- F6: 252→365 annualization mismatch fix for crypto/metals.
- F7: Config.json atomic-read normalization (api_utils, dashboard/auth, claude_gate).
- F8: Stack-overflow counter decay.
- F9: FOMC Jan 2026 dates off-by-one (data-external P0).
- F10: Alpha Vantage / NewsAPI / BGeometrics quota persistence across restarts.
- F11: yfinance vs Alpaca cache-key source-discrimination.

**Tier 2 (next 2 weeks):**
- Signal engine core-gate persistence-filter unification.
- outcome_tracker quantization fix + neutral-threshold normalization.
- `prune_jsonl` and `rotate_text` sidecar lock acquisition.
- Dashboard cookie/HMAC redesign.
- `update_state` lock unified across legacy callers.
- ATR trailing stop implementation.
- Drawdown circuit-breaker hysteresis.
- t-copula df fitting.
- Sustained-flip duration gate raise + AND logic.

**Tier 3 (backlog):**
- 100+ P2 findings from individual reviews (see per-subsystem docs).

---

## Reviewer bias / framing observations

- **Lead pass** emphasized cross-cutting integration risks (trigger contract gaps, Layer 2 silent-failure surface, SHORT-position future-proofing) — strengths of context awareness.
- **Subagent passes** found deep arithmetic / type / cache-race bugs the lead pass did not have time to drill into. They were sometimes too narrow to connect across subsystems.
- **Caveman:cavecrew-reviewer** (avanza-api) produced terser output and fewer findings (15 vs avg 50) — but every finding was tight and actionable. Recommend cavecrew for ≤8-file scopes; pr-review-toolkit:code-reviewer for larger.
- **No subagent found a fix that broke another subsystem** — partition was clean. The signals-modules subsystem is the only place where active vs disabled signals could cross-leak findings (still pending).

## Out-of-scope but spotted (composite)

- `data/metals_loop.py` (7880 LoC) — modularity hazard noted by lead and signals-core. Awaiting metals-core for detail.
- `data/critical_errors.jsonl` has 10 unresolved entries (1 avanza_account_mismatch, 2 accuracy_degradation, 7 contract_violation). PF-FixAgentDispatcher per-category cooldown may have given up.
- 31 raw `json.dump`/`json.load` paths outside file_utils — atomic-I/O audit in infrastructure review enumerates them.
- The `auth_error` cooldown logic interacts with `success` events oddly; consider rewriting around the auth_error itself, not "last non-skip status".

## Final verdict

The system has **multiple P0-class real-money risks dormant in the warrant flow** (Bear-MINI, double-fee P&L, annualization), **one active P0** (Layer 2 silent failures TODAY visible in critical_errors), and **two long-standing P0s in the Layer 2 surface** (tool allow-list, incomplete-stub poisoning) that haven't bitten yet but are landmine-shaped.

The signal engine has accumulated technical debt around accuracy gating semantics (default-0.5, sample-count mismatches, horizon collapse) that may explain the 12-signal accuracy degradation flagged in critical_errors on 2026-05-19.

Atomic-I/O contract is widely respected in file_utils but escapes through three load_config raw-open paths and a few legacy load_state/save_state calls. These should be the next compliance sweep.

**No code shipped in this review** — synthesis only. Per /fgl protocol, recommend converting Tier-1 backlog into per-finding PR workstreams next session.
