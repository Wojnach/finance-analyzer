# Adversarial Review 2026-05-13 — Synthesis

Second dual independent adversarial review of finance-analyzer. Codex (gpt-5.4, xhigh reasoning, read-only) and Claude (general-purpose subagents, sonnet-class) reviewed the same 8 disjoint subsystems in parallel. This doc cross-critiques both reviews and consolidates the highest-confidence findings.

Baseline: `main@59ad394e` (Merge fix/grid-fisher-buying-power). Worktree: `Q:/fa-adv-2026-05-13` on branch `adv-review-2026-05-13`. Eight orphan `adv-2026-05-13/baseline-N-<subsystem>` branches created with subsystem files removed.

## TL;DR

**~34 P0 blockers** and **~80 P1 incident-class defects** surfaced across 8 subsystems. Five patterns dominate:

1. **Silent-degradation is endemic.** Auth checks, accuracy gates, microstructure cache, journal readers, queue workers — multiple paths prefer "return default" / "return {}" over fail-closed-and-alert. Direct rule violation of the project's own incident-class learnings (Layer 2 outage Mar–Apr 2026; BUG-178; accuracy degradation).
2. **The Mar-3 stop-loss incident class is alive in 5+ sites.** `fin_snipe_manager._compute_stop_plan` entry-5% stop, `grid_fisher.rotate_on_buy_fill` no barrier check, `place_stop_loss_orders` doesn't verify cancel succeeded, `_cancel_stop_orders` falls back to generic cancel API on 404, `mstr_loop` live path mutates cash before fill confirmation. A shared `compute_safe_stop()` helper is overdue.
3. **Three Avanza trading stacks have diverged.** The clean `portfolio.avanza` package is missing every safety primitive (`avanza_order_lock`, account whitelist, 1000 SEK floor, fail-loud `get_buying_power`, tick rounding) that `avanza_session` grew organically. Same business capability, three idempotency models, three direction-cancel implementations.
4. **`claude_gate` is bypassed by 3+ live sites.** `analyze.py:282`, `analyze.py:746`, `multi_agent_layer2.py:171` all `subprocess.Popen(["claude", "-p", ...])` directly. Disables the kill-switch, rate-limit, auth-failure detection, and cost accounting the gate was built to provide.
5. **Project-rule drift between docs/code is widening.** `MIN_VOTERS_METALS = 2` in `signal_engine.py` violates the documented `MIN_VOTERS = 3` rule. `calendar_seasonal.py` + `vwap_zscore_mr.py` hardcode local quorum = 2. Forecast accuracy gate uses 55% / 10 samples / 14d snapshots vs documented 47% / 30 samples / 7d-recent weighting. `_BIAS_MIN_ACTIVE` reads `samples` (total inc HOLD) not active samples.

## Methodology

|                       | Codex                                                                                | Claude subagents                                                          |
| --------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Tool                  | `codex exec --sandbox read-only --output-last-message <file>` (gpt-5.4, xhigh)       | `general-purpose` agent, identical adversarial prompt                     |
| Workdir               | `Q:/fa-adv-2026-05-13`                                                               | `Q:/finance-analyzer`                                                     |
| Prompts               | `_prompts/<n>-<name>.txt`                                                            | identical content embedded in agent invocation                            |
| Final structured report | Emitted for **3 / 8** subsystems (avanza-api, signals-modules, infrastructure)     | Emitted for **8 / 8** subsystems                                          |
| Useful signal         | Prose narration extracted from stderr for the truncated 6 (mostly meta-narration; signals-modules partial-report survived) | Full P0/P1/P2/P3 + tests-missing + cross-cutting              |

**Meta-finding — codex truncation.** Five of the eight codex runs hit the OpenAI usage limit at ~290K-315K tokens per run before they could emit the final structured report. Limit reset is May 15. Aggregate consumption ~2M tokens. Three runs survived (avanza-api at 251K, signals-modules at 261K — written just before the limit cutoff, infrastructure at 294K) because there was just enough headroom on the account.

The previous review (2026-05-11) had a different failure mode: codex emitted transcripts instead of final reports because PowerShell shell commands kept getting rejected and `js_repl` tool calls consumed the turn budget. This time the turn-budget problem is solved (cleaner stderr streams, fewer aborted shells, `--output-last-message` captures the final message exclusively), but the per-account token budget is a new bottleneck. For the next FGL daily run, consider:

- Stagger codex runs serially with a sleep gate between them so the daily-rate-limit window can refresh between subsystems.
- Or reduce per-subsystem scope to ~10 files each so context per run stays under 200K tokens.
- Or run on a different OpenAI account for parallel coverage.

**Cross-critique direction.** For the 2 subsystems with complete codex reports, both findings sets are compared directly. For the 6 truncated runs, codex's prose was extracted from stderr (`_extract_prose.py`); most of it is process narration but `6-signals-modules-codex-prose.md` survived with a substantial partial-report that adds independent corroboration.

---

## Per-subsystem cross-critique

### 1 — signals-core

**Claude (3 P0, 7 P1, 17 P2, 13 P3, 9 tests-missing):**
- **P0** `signal_db.py:24-90` — `SignalDB._conn` shared across threads without `check_same_thread=False` or lock. Survives today only because callers create fresh instances; one refactor away from `ProgrammingError` storm. Silent fallback at `accuracy_stats.py:157` masks the symptom.
- **P0** `signal_engine.py:566-625` — Persistence filter effectively disabled for metals + crypto (`_PERSISTENCE_CYCLES_BY_ASSET = {"METALS": 1, "CRYPTO": 1, "STOCK": 2}`). Documented "2+ consecutive same-direction" contract is false for 4 of 5 Tier-1 tickers.
- **P0** `signal_engine.py:2593-2599` — `_BIAS_MIN_ACTIVE` reads `samples` (total votes inc HOLD) not active votes. Halves weight of rare informative directional signals.

**Codex prose hint** (truncated at file-load phase): only process narration survived. No corroborating signal but no contradictory either.

**Cross-critique.** Codex did not produce findings for this subsystem; Claude's set stands alone. The SignalDB thread-safety footgun is the most load-bearing because it depends on callers happening to use fresh instances — any future PR that hoists a module-level `SignalDB()` will detonate it. The persistence filter regression is documented but the rule-drift between code and CLAUDE.md doc needs a project-wide audit.

### 2 — orchestration

**Claude (5 P0, 14 P1, ~30 P2, ~25 P3):**
- **P0** `analyze.py:282`, `analyze.py:746`, `multi_agent_layer2.py:171` — three direct `subprocess.Popen(["claude", "-p", ...])` bypass `claude_gate`. Disables kill switch, rate-limit warning, in-process serialization (`_invoke_lock`), token+cost accounting.
- **P0** `agent_invocation.py:872` — Popens Claude without `creationflags=CREATE_NEW_PROCESS_GROUP`/`start_new_session=True`. On POSIX `_agent_proc.kill()` doesn't reach Node helpers; on Windows `taskkill /F /T` walks the tree only if PID hasn't been reused.
- **P0** `main.py:1188` + `trigger.py:170-199` — Startup grace defeated by `force_report=True`. Initial run synthesizes `reasons=["startup"]` → `classify_tier` returns T3 (first-trigger-of-day) → every loop restart fires a T3 (900s/40-turn) Layer 2.
- **P0** `agent_invocation.py:904-916` — `health_state.json` RMW without `_health_lock`. Races `heartbeat_keepalive` (60s tick). Last-writer-wins drops `last_invocation_tier`; downstream `loop_contract` tier-grace defaults to T3 (20m) for T1 calls — silent 12m alert delay.

**Codex prose hint:** truncated mid-load; only process narration survived.

**Cross-critique.** Claude's three `claude_gate` bypass sites are the exact incident class that motivated the gate (Mar–Apr 2026 outage). All three should be migrated this cycle. The startup-grace + force_report + first-of-day-T3 interaction is exactly the kind of trigger-loop-tier mismatch loop_contract is supposed to catch — but contract runs AFTER the T3 has already spawned. Codex emitted no independent corroboration; the orchestration findings are Claude-only.

### 3 — portfolio-risk

**Claude (6 P0, 15 P1):**
- **P0** `monte_carlo_risk.py:419`, `exit_optimizer.py:719`, `iskbets.py:743,798,875` — raw `agent_summary.get("fx_rate", ...)` bypasses the sanity-banded `_resolve_fx_rate` helper (P1-15 hardening). Stale `fx_rate=1.0` understates SEK risk 10×.
- **P0** `warrant_portfolio.warrant_pnl` — BULL-only, no financing-level clamp, no knockout, leverage drift unmodeled. BEAR positions compute nonsense P&L.
- **P0** `exit_optimizer._apply_risk_overrides` — BULL-only logic; would force-market-exit every BEAR position every cycle.
- **P0** `trade_risk_classifier` — silent unknown-regime → 0 score → LOW path. Type-error on None confidence.
- **P0** `iskbets.format_exit_alert/format_position_status` — mixes current vs entry fx_rate; breaks SEK P&L reporting.

**Confirmed fixed since 2026-05-11:** `trade_validation.py:32 min_order_sek=500.0` and `kelly_sizing.py:326 < 500` — both now 1000.

**Codex prose hint:** truncated mid-load; meta-narration only.

**Cross-critique.** This subsystem regressed since the 2026-05-11 review confirmed the 1000-SEK floor fix landed. The fx_rate bypass is a same-class regression — the same hardening function (`_resolve_fx_rate`) exists but five downstream call sites still use the raw `.get()` pattern. The MINI/BEAR warrant model gap is structural and unchanged from 2026-05-11: still "models MINI products as simple leverage multipliers" (codex prose hint then) → still BULL-only with no barrier (Claude now).

### 4 — metals-core

**Claude (6 P0, 12 P1):**
- **P0** `mstr_loop/config.py:19` — `PHASE = os.environ.get("MSTR_LOOP_PHASE") or "shadow"`. Single env var unlocks live orders. No approval file, no Telegram confirmation, no 90-day-shadow age check.
- **P0** `mstr_loop/execution.py:165-170` — Live path mutates `state.cash_sek` before order fill is confirmed. No reconciler.
- **P0** `grid_fisher.rotate_on_buy_fill` — Sell-side stop placement skips barrier check on existing inventory.
- **P0** `fin_snipe_manager._compute_stop_plan` — Entry-5% stop with zero barrier check.
- **P0** `place_stop_loss_orders` — Doesn't verify cancel succeeded before placing new stop. Overfill risk.
- **P0** `_cancel_stop_orders` — Falls back to regular-order cancel API on 404. Mar-3 inverse pattern.

**Codex prose hint:** "I've finished the grid-fisher state machine pass and already have a few concrete issues in hand. I'm switching to the signal/forecast math now…" — truncated before findings emitted. The 2026-05-11 codex prose flagged grid_fisher cancel fallback + MINI barrier guard only existing in pretrade helpers; that hint is **still relevant today** per Claude's findings.

**Cross-critique.** **5 independent stop-loss placement sites all skip barrier-distance checks.** The single dominant systemic finding. The 2026-05-11 review flagged 3 of those sites (grid_fisher cancel fallback, knockout/barrier check, mstr_loop env-var-only live phase). All 3 are still present 2 days later. A shared `compute_safe_stop(instrument, side, entry, current_bid)` helper is overdue and should be the next architectural commit.

### 5 — avanza-api

**Both codex and Claude produced full reports.** Strong agreement.

**P0 overlap (both flagged):**
- New `portfolio.avanza` package is missing every safety primitive of `avanza_session`: account whitelist (Claude P0), `avanza_order_lock` (Claude P0; codex P1 — both flag), 1000 SEK floor enforcement (codex P0), stop-loss policy gate for "within 3% of bid" / "near MINI barrier" (codex P0).
- `cancel_all_stop_losses_for()` defaults `account_id=None`, cancels stops cross-account (codex P0).

**Codex-only P0:**
- `avanza_control.py:395-404` — `delete_stop_loss_no_page()` ignores `_api_delete()`'s `ok/http_status`. 403/500 become `True`. Caller proceeds into sell while stop still live.
- `avanza_session.py:676-717`, `avanza_client.py:172-178` — BankID/session path returns positions across all accounts (whitelist filter exists only in TOTP fallback).

**Claude-only P0:**
- `_with_browser_recovery` retries non-idempotent POSTs after browser death → duplicate live orders.
- `avanza/account.get_buying_power` silently returns zeroes on account-not-found (replays 2026-04-09 incident).
- `place_stop_loss` doesn't validate `trigger_price > 0` for monetary non-trailing; `place_trailing_stop` accepts `trail_percent=0`.
- `stopLossTrigger.validUntil` vs `stopLossOrderEvent.validDays` desync risk.
- `cancel_all_stop_losses_for` doesn't clamp `poll_interval` → DOS-the-session footgun.

**Cross-critique.** Both reviews independently arrived at the same cross-cutting finding: three Avanza stacks (`avanza_session`, legacy `avanza_client`, `portfolio.avanza`) have drifted enough to create incident-class behavior differences. The fix is to consolidate trading mutations behind the hardened helpers and delete the duplicate public API in `portfolio.avanza.trading`. **Highest signal-density subsystem in this review.**

### 6 — signals-modules

**Claude (5 P0, 15 P1, 23 P2, 20 P3):**
- **P0** `crypto_evrp.py:42-43,195-201` — Direction inverted vs documented thesis ("high eVRP → bullish" but code returns SELL).
- **P0** `crypto_evrp.py:216` — `rv_hist` is dead code; sub-signal degrades to DVOL-only.
- **P0** `mahalanobis_turbulence.py:99` + `complexity_gap_regime.py:92` — `_cached(key, fn, ttl=...)` is wrong arg order → TypeError on enable.
- **P0** `cross_asset_tsmom.py:148-171` — Bond/equity sub-signals return same direction for both gold AND BTC when TLT up.

**Codex partial-report (truncated mid-write):**
- **P0** `forecast.py:415-424` — Chronos timeout ineffective; `with ThreadPoolExecutor()` waits for worker on exit, hung `forecast_chronos()` blocks indefinitely + ties up GPU gate.
- **P0** `forecast.py:541-546,557-570` — Accuracy gating fails open on accuracy-loader exception → raw forecast votes instead of force-HOLD. Disables safety gate exactly when accuracy backend is unhealthy.
- **P1** `forecast.py:484-496` — Policy constants drift (55% / 10 samples / 14d) vs CLAUDE.md (47% / 30 samples / 70-30 recency).
- **P1** `calendar_seasonal.py:463-486` + `vwap_zscore_mr.py:94-99` — Local quorum hardcoded to 2 active votes; violates `MIN_VOTERS = 3`.

**Cross-critique.** **Both reviewers found rule-drift independently** but in different files. Claude focused on direction-inversion bugs in the disabled-detector set + active `structure.py` breakout direction inversion (P1). Codex focused on `forecast.py` policy drift + fail-open safety gates. **Combined the set is most complete:** structure.py inverted (Claude P1, ACTIVE), fibonacci pivot timeframe wrong (Claude P1, ACTIVE), forecast accuracy gate fail-open (codex P0), forecast policy constant drift (codex P1), MIN_VOTERS=2 local quorum in two detectors (codex P1). All five are live production-affecting.

### 7 — data-external

**Claude (4 P0, 20 P1, 19 P2, 21 P3):**
- **P0** `http_retry.py:43-49` — Ignores HTTP `Retry-After` header. Only parses Telegram-style JSON `parameters.retry_after`. Binance/AV/NewsAPI/Frankfurter all send standard header → backoff ignored on 429 → bans.
- **P0** `funding_rate.py:42-49` — Thresholds asymmetric and wrong. SELL at >0.0003, BUY at <-0.0001. Structurally biased to SELL in any positive-funding regime.
- **P0** `price_source.fetch_klines:223-243` — Silent yfinance fallback when Binance fails returns 10-15min-stale bars with no source tag. Downstream mixes live microstructure with stale OHLCV — "BTC 12h BUY phantom" memory pattern.
- **P0** `_tool_cache` writers don't set `ttl` field. `data_collector._fetch_one_timeframe:309` writes without `ttl` → LRU eviction defaults to 3600s. 24h-TTL entries get evicted after 3h.

**Codex prose:** "stale fallback paths, cache semantics that treat fetch failures as fresh data, and a few places where hard-coded timing/consensus rules drift from the project contract" — truncated before specific citations.

**Cross-critique.** Codex's pre-truncation theme summary aligns with Claude's findings (stale-as-fresh cache + rule drift). The Retry-After bug is most actionable — fixing one function affects every external API path. The yfinance silent fallback is the most insidious because it manifests as "phantom signals" rather than visible errors.

### 8 — infrastructure

**Both codex and Claude produced full reports.** Strong agreement on dashboard auth defects.

**P0 overlap (both flagged):**
- **Dashboard auth fail-open** — Claude P0 (`dashboard/auth.py:132-135` Cloudflare header trust); codex P0 same lines + `dashboard/auth.py:60-67,81-83,121-123` (auth fails open when config.json is missing/unreadable). Codex catches an additional fail-open path.
- **`atomic_append_jsonl` + `prune_jsonl` race** — Claude P0 (TOCTOU on lock-file creation + on-OSError fallback proceeds unlocked); codex P1 (`prune_jsonl()` rewrites JSONL outside `jsonl_sidecar_lock()`). Same defect class, different framing.

**Codex-only P1:**
- `process_lock.py:39-44,63-67` — Windows lock acquisition on brand-new empty lock file unreliable; OSError treated as benign contention.
- `subprocess_utils.py:132,175,247-252` — `AssignProcessToJobObject()` return value not checked; fallback kill path omits `/T`.
- `shared_state.py:30-32` — `_cached_or_enqueue()` never expires stale `_loading_keys`; key can stay "loading" forever after crash.
- `house_blueprint.py:211-216,280,330,359-365` — Markdown/HTML rendered into authenticated dashboard origin without sanitization → same-origin XSS.

**Claude-only P0:**
- `atomic_write_*` silent OSError fallback (cross-volume / locked-target → silent skip).
- `atomic_append_jsonl` torn-lines on `>PIPE_BUF` writes + SIGTERM/ENOSPC mid-write → `prune_jsonl` erases corrupt line silently.

**Claude-only P1:**
- `process_lock` no stale-PID recovery on host crash.
- `subprocess_utils` uses `proc._handle` (private CPython attribute).
- `subprocess_utils.run_safe` does NOT scrub `CLAUDECODE` env var (34h-outage cause).
- `fix_agent_dispatcher` mutates `os.environ[RECURSION_ENV]` across siblings without lock.
- `backtester` in-sample (weights computed from same outcomes being scored).
- `message_store` has NO dedup despite brief saying so.

**Cross-critique.** **Strongest agreement of the 8 subsystems.** Both reviewers independently identified the dashboard auth fail-open as P0 and the JSONL prune/append race. Codex caught an additional auth fail-open path (config-read failure → no token → bypass) that Claude missed. Claude caught additional subprocess and atomic-write paths. **Combined the P0 set is now 6-8 distinct defects in this subsystem.** Dashboard auth is the most externally-reachable risk — fix immediately.

---

## Cross-cutting findings (across multiple subsystems)

### CC-1: Three Avanza stacks with three idempotency models
Confirmed in subsystem 5 (avanza-api). Echoes in subsystems 3 (portfolio-risk: warrant_portfolio model BULL-only) and 4 (metals-core: cascading-stop placement doesn't verify cancel). **Recommendation:** consolidate trading mutation behind `avanza_session._place_order` + `_place_stop_loss`; delete `portfolio.avanza.trading` public mutators.

### CC-2: Mar-3 stop-loss incident class — 5+ live sites
Found in subsystem 4 (metals-core) at 5 distinct stop-loss placement sites. Echoes in subsystem 5 (avanza-api): `place_stop_loss` skips "within 3% of bid" / "near MINI barrier" checks. **Recommendation:** Single `compute_safe_stop(instrument, side, entry, current_bid)` helper that all five sites must route through. Refuse to place if it returns "unsafe".

### CC-3: `claude_gate` bypass — 3 live sites
`analyze.py:282`, `analyze.py:746`, `multi_agent_layer2.py:171` plus `agent_invocation.py:872` (uses gate but without process group). **Recommendation:** add a lint rule / grep test that prohibits direct `subprocess.Popen(["claude"` outside `claude_gate.py`.

### CC-4: Silent fail-open is endemic
Concrete sites: dashboard auth fail-open on config read (codex P0), accuracy gate fail-open on loader error (codex P0), `get_buying_power` returns zero on account-not-found (Claude P0), `trade_risk_classifier` unknown-regime → LOW (Claude P0), `funding_rate` SELL-biased threshold (Claude P0), `crypto_evrp` direction inverted (Claude P0), `forecast` `_health_weighted_vote` fails open. **Recommendation:** sweep `except Exception` blocks across the codebase; require either re-raise OR explicit `critical_errors.jsonl` entry.

### CC-5: Rule drift between CLAUDE.md and code is widening
- `MIN_VOTERS = 3` documented; `MIN_VOTERS_METALS = 2` in `signal_engine.py`; local quorum = 2 hardcoded in `calendar_seasonal.py`, `vwap_zscore_mr.py`.
- Accuracy gate documented as 47% / 30+ samples / 70-30 recency; `forecast.py` uses 55% / 10 / 14d.
- Min order size documented as 1000 SEK; `portfolio.avanza.trading` doesn't enforce, `avanza_client._place_order` (legacy TOTP path) accepts `volume >= 1, price > 0` without floor.
- Persistence filter documented as "2+ consecutive same-direction"; actually disabled (`min_cycles=1`) for metals + crypto.

**Recommendation:** project-rule audit pass. Either fix code to match docs or fix docs to match code; current divergence is worst of both worlds (operators assume the documented contract holds).

### CC-6: Atomic I/O violations remain in 7+ files
`scripts/check_critical_errors.py:42-49`, `scripts/fix_agent_dispatcher.py:84-91`, `dashboard/app.py:898-905,910-917`, `dashboard/export_static.py:58,64-67`, `portfolio/vector_memory.py:267-281`, `portfolio/signal_decay_alert.py:34-39`, `portfolio/avanza_orders.py:264-266`. **Recommendation:** add a CI grep guard that flags `json.loads(open(...).read())` and `json.load(open(...))` outside `file_utils.py`.

### CC-7: Cross-asset detectors return same direction regardless of ticker class
Subsystem 6: `cross_asset_tsmom.py` (Claude P0), `vix_term_structure.py` (Claude P1), `treasury_risk_rotation.py` (P1 — direction may be inverted), `metals_cross_asset.py` shape gap. Pattern: aggregated cross-asset signals not gated by the target ticker's expected direction; gold and BTC receive identical BUY/SELL from TLT-up events. **Recommendation:** every cross-asset detector must return a `direction_basis: "safe_haven"|"risk_on"|"neutral"` and route through a per-ticker direction-mapping table.

---

## Top-priority fixes (next implementation cycle)

Ordered by blast radius × ease:

1. **Migrate 3 `claude_gate` bypass sites** — `analyze.py:282,746` + `multi_agent_layer2.py:171`. ~30 min. Eliminates a recurring incident class.
2. **fx_rate hardening at 5 call sites** — wrap with `_resolve_fx_rate` from `risk_management.py`. ~1h. Eliminates SEK risk understatement.
3. **`MIN_VOTERS_METALS = 2` audit** — either remove the override or document why metals is different; same for `calendar_seasonal` and `vwap_zscore_mr` local quorum = 2. ~2h.
4. **Dashboard auth fail-closed sweep** — `dashboard/auth.py:60-67,121-123,132-135` — fail closed on config error; verify Cloudflare JWT or refuse direct traffic. ~2h. **Highest external-reach risk.**
5. **`compute_safe_stop()` helper** + migrate 5+ call sites in metals-core to use it. Refuse to place if returns "unsafe". ~3h.
6. **Process-group fix for `agent_invocation.py:872`** — add `creationflags`/`start_new_session`, route through `_kill_process_tree` from `claude_gate`. ~30 min.
7. **MSTR phase live-confirmation gate** — sentinel file + shadow-age check + Telegram confirm. ~1h.
8. **`startup` reason → no Layer 2 invocation** — fix `force_report=True` + grace interaction. ~30 min. Eliminates restart-storm T3 burn.
9. **`HTTP Retry-After` parsing in `http_retry.py`** — fix the standard header path. ~30 min. Affects every external API.
10. **`compute_safe_stop` for new `portfolio.avanza` package** — same surface as fix #5, different stack. ~1h.

**Total estimated time:** ~12-15h of focused work.

## Top-priority fixes NOT to do this cycle

- Rewriting `signal_engine.py` (4,206 lines, complex regime-conditional gates). Carefully refactor in pieces; do NOT touch in a single PR. The signal-core findings are real but each fix should be independent.
- Consolidating the three Avanza stacks. High-value but disruptive; needs its own multi-PR design first.
- Adding fee/slippage to `backtester`. Important but not on the live trading path.

---

## Files

```
docs/adversarial-review-2026-05-13/
├── 00-PARTITION.md
├── 99-SYNTHESIS.md                       (this file)
├── 1-signals-core-claude.md              33 KB, 33 findings
├── 1-signals-core-codex-prose.md         truncated (process narration)
├── 2-orchestration-claude.md             44 KB, ~70 findings
├── 2-orchestration-codex-prose.md        truncated (process narration)
├── 3-portfolio-risk-claude.md            39 KB, 6P0+15P1
├── 3-portfolio-risk-codex-prose.md       truncated (process narration)
├── 4-metals-core-claude.md               36 KB, 6P0+12P1
├── 4-metals-core-codex-prose.md          truncated (process narration)
├── 5-avanza-api-claude.md                65 KB, 7P0+15P1
├── 5-avanza-api-codex.md                 9.5 KB, 4P0+6P1 (FULL REPORT)
├── 6-signals-modules-claude.md           45 KB, 5P0+15P1
├── 6-signals-modules-codex.md            11 KB, 2P0+8P1 (FULL REPORT)
├── 7-data-external-claude.md             64 findings
├── 7-data-external-codex-prose.md        truncated (process narration)
├── 8-infrastructure-claude.md            51 KB, 4P0+12P1
├── 8-infrastructure-codex.md             10 KB, 2P0+7P1 (FULL REPORT)
├── _prompts/                             (8 prompt files used)
├── _codex_out/                           (raw stdout/stderr per run)
├── _build_baselines.py
├── _extract_prose.py
├── _launch_codex.py                      (first attempt, --base mode)
└── _launch_codex_exec.py                 (second attempt, exec mode — used)
```

Baseline branches (8): `adv-2026-05-13/baseline-{signals-core,orchestration,portfolio-risk,metals-core,avanza-api,signals-modules,data-external,infrastructure}`. Cleanup after merge: `git branch -D adv-2026-05-13/baseline-*` (no remote push of these).
