# Adversarial Review Synthesis — 2026-05-24

**Trigger commit:** `289e5030` (main HEAD at review start — "fix(P1): grid_fisher stop_needs_rearm flag + tick retry")
**Review style:** /fgl whole-codebase audit, 8 subsystems × dedicated reviewer subagent + 1 independent cross-cutting pass from the main thread
**Empty-baseline diff technique:** orphan branch `review-baseline-empty` lets each reviewer see the entire subsystem as additions, not just recent changes
**Worktree:** `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24` (cleaned up after merge — see commit log)
**Prior review baseline:** `docs/ADVERSARIAL_REVIEW_2026-05-19.md` (255 findings: 42 P0, 95 P1, 72 P2, 46 P3). The interim 5-day window saw three batches of P0 fixes (`be4273d3`, `c7d60b72`, `81e0c78e`, `4adeec2d`, `289e5030`) closing roughly 8-10 P0s and 2 P1s.

---

## Tallies

| Subsystem | Agent | P0 | P1 | P2 | P3 | Total |
|-----------|-------|----|----|----|----|-------|
| signals-core | pr-review-toolkit:code-reviewer | 2 | 4 | 5 | 3 | 14 |
| orchestration | pr-review-toolkit:code-reviewer | 5 | 17 | 22 | 3 | 47 |
| portfolio-risk | pr-review-toolkit:code-reviewer | 8 | 13 | 9 | 4 | 34 |
| metals-core | pr-review-toolkit:code-reviewer | _running at synthesis time, see deferred section_ | _ | _ | _ | _ |
| avanza-api | caveman:cavecrew-reviewer | 1 | 4 | 3 | 0 | 8 |
| signals-modules | pr-review-toolkit:code-reviewer | 0 | 5 | 14 | 3 | 22 |
| data-external | pr-review-toolkit:code-reviewer | 6 | 10 | 8 | 11 | 35 |
| infrastructure | caveman:cavecrew-reviewer | 0 | 1 | 3 | 0 | 4 |
| **main thread cross-cut** | self | 8 | 6 | 3 | 3 | 20 |
| **TOTAL (excluding metals)** | | **30** | **60** | **67** | **27** | **184** |

The lower-count infrastructure subsystem reflects a smaller worktree (caveman agent did not re-write its file, so we transcribed the agent's terse summary; the worktree-resident Apr 8 prior file lists 4 additional P0/P1 candidates we flag for verification under "Carryover").

30 P0 (down from 42 the prior review) is meaningful progress but masks a problem: many of the same P0s are still present, just newly named. The interim fixes targeted the symptoms (off-hours skip on the autonomous-first branch, specific stop buffers in fin_snipe) without touching the underlying concurrency primitives or the bias-double-application math.

---

## TOP 10 — Highest-conviction fix-first findings

Picked by: cross-agent corroboration (independently flagged by ≥2 sources), money-at-risk, ease-of-fix ratio, and re-occurrence across reviews. These are the must-fix items.

### T1. Bias-penalty double-application in signal_engine — STILL NOT FIXED

- **Where:** `portfolio/signal_engine.py:2677` (first multiplication via `norm_weight` from activation_rates) and `portfolio/signal_engine.py:2698` (second via `_resolve_bias_penalty`). Source of the first term: `portfolio/accuracy_stats.py:849` (`normalized_weight = rarity_weight * bias_penalty`).
- **Why P0:** Same exact bug 2026-05-19 review reported (T1). The fix attempt made the second multiplier conditional on bias-direction vote, but the unconditional first multiplier was never removed. Contrarian votes get the legacy 0.1× they should NOT get; bias-direction votes get squared `bias_penalty²`.
- **Conviction:** signals-core agent (P0 #1) + main-thread (IND-P0-1) independently re-flagged.
- **Fix:** in `accuracy_stats.py:849`, drop `bias_penalty` from `normalized_weight`: `"normalized_weight": round(rarity_weight, 4)`. Keep `bias_penalty` as a standalone column for diagnostics. Add a regression test that a calendar-bias signal voting contrarian retains full weight.

### T2. Cross-process race on portfolio_state.json — STILL NOT FIXED

- **Where:** `portfolio/portfolio_mgr.py:30-159` (threading.Lock only).
- **Why P0:** Main loop, dashboard Flask, Layer 2 subprocess all write the same JSON from separate OS processes. Same class as the Mar 3 stop-loss incident. Confirms 2026-05-19 T5.
- **Conviction:** portfolio-risk agent (P0-B) + main-thread (IND-P0-2) re-flagged. file_utils already has the right primitive (`jsonl_sidecar_lock` with msvcrt/fcntl) but `portfolio_mgr` does not use it.
- **Fix:** introduce `file_utils.update_json_atomic(path, mutate_fn)` that wraps read+mutate+write inside the existing sidecar-lock pattern. Migrate `portfolio_mgr.update_state`, `save_state`, `save_bold_state`.

### T3. warrant_portfolio has ZERO concurrency protection — STILL NOT FIXED

- **Where:** `portfolio/warrant_portfolio.py:42-49` (`save_warrant_state`) and `198-265` (`record_warrant_transaction` mutator).
- **Why P0:** No threading.Lock, no sidecar lock, nothing. metals_loop + grid_fisher + dashboard + Layer 2 all race on `data/portfolio_state_warrants.json`. Concurrent BUYs silently drop one transaction. Worse than portfolio_mgr (which at least has process-local protection). Confirms 2026-05-19 T4.
- **Conviction:** portfolio-risk agent (P0-A) + main-thread (IND-P0-3) re-flagged.
- **Fix:** same `update_json_atomic` helper as T2; add a `update_warrant_state` wrapper; migrate every caller.

### T4. BULL/BEAR direction not honored in P&L math or stop logic — STILL NOT FIXED

- **Where:**
  - `portfolio/warrant_portfolio.py:96` — `implied_pnl_pct = underlying_change * leverage` (LONG only)
  - `portfolio/risk_management.py:374,382,484,897` — ATR stop math (`entry * (1 - 2*atr/100)`, `triggered = current < stop`) all LONG only
  - `portfolio/exit_optimizer.py:327-332,718` — `warrant_move = pct_move * leverage` LONG only
  - `portfolio/monte_carlo.py:simulate_ticker` — `p_stop_hit = P(terminal < stop_price)` LONG only
- **Why P0:** The warrant catalog has 41 SHORT certs vs 58 LONG. Any held BEAR cert reports inverted P&L on Telegram + dashboard + journal, AND its stop "triggers" on a profitable green move. Confirms 2026-05-19 T7. Cross-cutting across 5+ modules.
- **Conviction:** portfolio-risk agent (P0-C, top 1/2) + main-thread (IND-P0-4) re-flagged.
- **Fix:** add `direction` to holding schema. Plumb sign through P&L: `sign = +1 if direction == "LONG" else -1; implied_pnl_pct = underlying_change * leverage * sign; stop_price = entry * (1 - 2 * atr_pct/100 * sign); triggered = (current - stop_price) * sign < 0`.

### T5. Kelly metals inverts leverage safety bound — NEW IN THIS REVIEW

- **Where:** `portfolio/kelly_metals.py:215-221` + `:45` (`MAX_POSITION_FRACTION = 0.95`).
- **Why P0:** `position_fraction = half_kelly / (avg_loss * leverage / 100)` GROWS with leverage. Real Kelly for a leveraged bet shrinks with leverage². On a high-edge regime, the formula can push 95% of cash onto a single 5x silver MINI. `memory/grudges.md` is explicit about silver's regret cost.
- **Conviction:** portfolio-risk agent (P0-D, top 5).
- **Fix:** `position_fraction = half_kelly` (Kelly already accounts for leverage if fed leveraged returns). Cap `MAX_POSITION_FRACTION = 0.30 / leverage` (i.e., 6% on 5x). Wire consecutive-loss reducer BEFORE the cap, not after.

### T6. Avanza get_positions() leaks pension positions — STILL NOT FIXED

- **Where:** `portfolio/avanza_session.py:676-717`.
- **Why P0:** `ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID}` exists at line 43 and is consulted by trading helpers (lines 591, 750) but **not** in get_positions. memory/feedback_isk_only.md forbids this. Confirms 2026-05-19 T9.
- **Conviction:** avanza-api agent (BUG #1) + main-thread (IND-P0-5).
- **Fix:** insert `if account.get("id", "") not in ALLOWED_ACCOUNT_IDS: continue` at the top of the for-loop in get_positions.

### T7. Layer 2 off-hours skip still silently disables decisions (non-router branch) — STILL NOT FIXED

- **Where:** `portfolio/main.py:972-979`.
- **Why P0:** Off-hours autonomous-fallback was added to the autonomous-first branch (947-966) but NOT to the simpler `elif layer2_cfg.get("enabled", True):` branch. Weekend XAG/BTC/ETH triggers → no journal, no Telegram, no decision. Confirms 2026-05-19 T6.
- **Conviction:** orchestration agent (P0 #1) + main-thread (IND-P0-8).
- **Fix:** mirror lines 960-966: call `autonomous_decision(...)` under `heartbeat_keepalive` before the `skipped_offhours` log.

### T8. Specialist quorum timeout default is below specialist budgets — NEW IN THIS REVIEW

- **Where:** `portfolio/agent_invocation.py:1021` (`specialist_timeout_s=30`).
- **Why P0:** Each specialist runs `max_turns=8-10 × ~10-15s/turn ≈ 80-120s`. With 30s budget every specialist times out → `success_count==0` → `specialist_quorum_fail` → return None → main.py:951 logs duplicate `skipped_busy_<why>`. Multi-agent mode effectively broken with default config.
- **Conviction:** orchestration agent (P0 #3).
- **Fix:** raise default to `max(SPECIALISTS[*].timeout) + 30s = 150s` and align with the per-specialist `timeout` field in `SPECIALISTS`.

### T9. Binance error responses still returned as garbage candles — STILL NOT FIXED

- **Where:** `portfolio/data_collector.py:74-101`.
- **Why P0:** `data = r.json(); if not data: ...; df = pd.DataFrame(data, columns=...)`. Binance returns `{"code":-1121,"msg":...}` on bad symbol/interval. `if not data` is False for non-empty dict → DataFrame construction raises ValueError → caught by broad `except Exception → cb.record_failure()` → mislogged as network failure, circuit-breaker counts it wrong. Confirms 2026-05-19 T8.
- **Conviction:** main-thread (IND-P0-7). data-external agent flagged the fix as landed but the code in main repo still shows the unguarded path — discrepancy worth verifying after this review.
- **Fix:** before `df = pd.DataFrame(...)`: `if isinstance(data, dict) and "code" in data: raise ConnectionError(data.get("msg","binance error"))`.

### T10. Grid Fisher stop-sell-price still 0.5% buffer on 5x certs — STILL NOT FIXED

- **Where:** `portfolio/grid_fisher.py:1496` (`stop_sell_price = round(stop_price * 0.995, 2)`).
- **Why P0:** A 0.5% buffer on a 5x cert means a 0.5% underlying gap past trigger leaves the limit unfilled. The recent commit `289e5030` fixed `stop_needs_rearm` flag retry but did NOT widen the buffer itself. Confirms 2026-05-19 T3 (one of three sites; fin_snipe sites also still 1%). The `.claude/rules/metals-avanza.md` explicit rule: "5x leverage certificates need -15%+ stops, not -8%."
- **Conviction:** main-thread (IND-P0-6). Metals-core agent still running at synthesis time — likely to corroborate.
- **Fix:** widen to ≥3% buffer (`stop_price * 0.97`) for 5x MINIs; also widen `fin_snipe_manager.HARD_STOP_SELL_BUFFER_PCT` from 0.01 to 0.03.

---

## Cross-cutting themes (more important than the individual findings)

### Theme A. Cross-process lock confusion is the dominant bug class
Five of the top 10 P0s and at least seven P1s in this review touch threading.Lock used where a cross-process lock is required. The codebase already has the correct primitive in `file_utils.jsonl_sidecar_lock` (msvcrt+fcntl). The callers are inconsistent. A single sweep introducing `file_utils.update_json_atomic` and migrating all RMW state writes closes the largest cluster of bugs in this review.

### Theme B. Silent-fail patterns mask bugs as HOLD votes
`except Exception: return HOLD` and `except Exception: return ""` appear in 16+ files (signals-modules agent flagged P1 examples; main-thread (IND-P1-5) flagged the systemic pattern). Each silent return becomes a 0.5-confidence vote in the consensus and propagates without visibility. The required fix: a structured `SignalResult` contract with `error` field, plus a registry-level enforcement that bare `Exception` is not caught.

### Theme C. Vote double-counting in sub-signals inflates confidence
Signals-modules agent flagged 4 modules where sub-signals copy votes into multiple keys then majority-vote (vwap_zscore_mr, autotune_adaptive_cycle, network_momentum, hurst_regime). Confidence is silently inflated when these signals fire. The audit pattern is `grep -rn "sub_signals\[.*\] = .*_vote"` — every result should be checked for whether it's a unique direction or a copy.

### Theme D. The recent 2026-05-24 batch-fix sweep closed easy P0s but did not touch the foundational concurrency or P&L sign bugs
Commits `be4273d3`, `c7d60b72`, `81e0c78e`, `4adeec2d`, `289e5030` closed 7-8 of the 22 P0s from 2026-05-19. The remaining P0s (T1-T4, T6, T7, T9, T10 above) are all from that review. Higher-engineering-effort items (lock semantics, P&L sign math, account-id filter) were deferred. The next batch must prioritize T2/T3/T4 because they share a single fix shape (`update_json_atomic` + `direction` schema migration).

### Theme E. Empty-baseline technique catches issues diff-mode reviewers skip
Several P0s in this review (T6, T7, T9, T10) are pre-existing code no recent diff touched. The diff-mode reviewer who looked at each batch's PR did not see them because the lines were context, not changes. The empty-baseline orphan-branch technique force the reviewer to consider every line as a fresh addition. This is the second consecutive review where the technique surfaced multi-cycle-old bugs.

---

## Subsystem highlights

### signals-core (2 P0, 4 P1)
- T1 above.
- **P0:** `accuracy_stats.py:970-974` — `blend_accuracy_data` double-counts directional samples (`total_buy = at + rc` while `recent ⊂ alltime`). Inflates directional gate decisions on boundary signals.
- **P1:** `ticker_accuracy.py:294,298` — `direction_probability_with_forecast` does 20 full signal-log scans per Mode-B refresh, uncached.
- **P1:** `MACRO_WINDOW_FORCE_HOLD_SIGNALS` is a dead one-element set referencing an already-disabled signal.
- **Verified FIXED:** utility-boost bypass at signal_engine.py:4210; dynamic correlation groups now use agreement rate (1638-1646).

### orchestration (5 P0, 17 P1)
- T7, T8 above.
- **P0:** `agent_invocation.py:402-423` — `_no_position_skip` reads `signals` from `agent_context_t1.json` which `_write_tier1_summary` doesn't populate. One flag flip and Layer 2 is dead. Latent today.
- **P0:** `main.py:1085-1091,1050` — IC cache + safeguards gated on `_run_cycle_id` (restart-reset). With 600s cycles since 2026-04-09 the real cadence is "10h IF no restart" — effectively dead.
- **P0:** `loop_contract.py:335` — in-flight suppression matches `status == "invoked"` exactly, missing `"invoked_<why>"` from autonomous-first path → false-positive `layer2_journal_activity` violations storm critical_errors.jsonl once `autonomous_first_enabled=true`.

### portfolio-risk (8 P0, 13 P1)
- T2, T3, T4, T5 above.
- **P0:** `kelly_sizing.py:84-104` — single weighted-avg buy price across ALL buys (including post-sell BUYs). Historical wins look like zero-P&L → Kelly sees zero edge → refuses to size up.
- **P0:** `monte_carlo_risk.py:204,228` — hardcoded `_trading_days=365` for ALL assets; MSTR vol uses 252 → 1-day VaR biased ~20% low for stocks.
- **P0:** `monte_carlo_risk.py:408` + `exit_optimizer.py:718` — both bypass cached `_resolve_fx_rate` chain (raw `agent_summary.get("fx_rate", 10.85)`); reintroduces the false-circuit-breaker bug class.
- **P0:** `price_targets.py:391,397,417,419,429,431` — warrant SEK gain formula `(target - price) * units * leverage * fx` overstates by **~27×** on real silver MINI math. EV-ranked targets misranked → recommends the wrong target.

### metals-core (running at synthesis time)
Deferred. The agent is the largest of the eight (covering `data/metals_loop.py`, all swing traders, grid_fisher, fin_snipe_*, ministral/qwen3 traders, golddigger/, elongir/, mstr_loop/). When it finishes, its findings will be appended below in the "Late additions" section. Expected to corroborate T10 (stop buffer) and likely flag fast-tick / slow-loop module-global races. See independent IND-P0-6 for the partial coverage.

### avanza-api (1 P0, 4 P1)
- T6 above.
- **P1:** `avanza_session.py:633-645` — `cancel_order(order_id, account_id=None)` accepts account_id without checking against `ALLOWED_ACCOUNT_IDS`. Only mutating function without the whitelist guard.
- **P1:** `avanza_client.py:327-368` — TOTP `_place_order` skips the 1000 SEK minimum guard that the session-based variant enforces.
- **P1:** `avanza_session.py:134-153` — `_get_playwright_context()` has no exception cleanup; failed init leaves zombie Playwright process.
- **P1:** `avanza_session.py:89` — `if exp <= now` off-by-one rejects a valid token 1s early.

### signals-modules (0 P0, 5 P1, 14 P2 inc. lookahead candidates)
- **P1:** `vwap_zscore_mr.py:124-125` + `autotune_adaptive_cycle.py:187-188` — top-level `except Exception: return HOLD` no log.
- **P1:** `credit_spread.py:285` + `gold_real_yield_paradox.py:265` — `load_json("config.json", ...)` RELATIVE path. Exact failure mode the batch fixes targeted in other places.
- **P1:** `btc_etf_flow.py:53,106-107` — wrong signature `compute(ticker, indicators, context)` vs registry contract `compute_X_signal(df, context)`. Not registered, bare `except Exception: pass`. Dead code that would crash if enabled.
- **P2 (4 files):** vote double-count via sub_signal vote copying.
- **P2:** `gold_real_yield_paradox.py:285-291` — date misalignment between FRED daily and 24/7 gold OHLCV.
- **P2:** `ttm_squeeze.py:69-70`, `absorption_ratio_regime.py:112`, `mahalanobis_turbulence.py:138`, `trend_slope_momentum.py:98-99` — `bfill()` window-internal NaN backfill (potential intra-window lookahead).

### data-external (6 P0, 10 P1)
- **P0:** `data/crypto_data.py:223-231` — `get_onchain_summary()` reads keys `zone`/`bias`/`summary` that `interpret_onchain()` NEVER emits. The entire on-chain BTC signal pipeline has been returning permanent "neutral" since this code was written. Silent.
- **P0:** `data/crypto_data.py:75-82` — F&G empty-response fabricates `value=50, classification="Neutral"`.
- **P0:** `data/crypto_data.py:184-198` — `compute_mstr_btc_nav()` uses hard-coded MSTR BTC holdings (499,096) and share count from "early 2026". Today 2026-05-24, MSTR has continued accumulating BTC.
- **P0:** `forecast_accuracy.py:282-304` — `datetime.fromisoformat()` naive vs `datetime.now(UTC)` aware TypeError crashes the backfill mid-loop. Same naive/aware issue in `local_llm_report.py:48-51`.
- **P0:** `forecast_signal.py:97` — `except (ImportError, Exception)` swallows everything (CUDA OOM, file IO).
- **P0:** `data_collector.py:168-204` — `fetch_vix()` calls yfinance WITHOUT acquiring `_yfinance_lock`. Concurrent VIX + sentiment yfinance call → known yfinance non-thread-safety segfault.

### infrastructure (0 P0, 1 P1, 3 P2 from this pass)
- **P1:** `dashboard/auth.py:175` — Bearer token auth path skips `_refresh_cookie()`. Session expires after 1 year regardless of activity.
- **P2:** `http_retry.py:55` — jitter added after `retry_after` cap, 100% overhead on Telegram 429 waits.
- **P2:** `shared_state.py:88-89` — dogpile key added before timestamp write; BaseException can orphan key for 120s.
- **P2:** `health.py:37-40` — `error_count` unbounded.

**Carryover from 2026-04-08 infrastructure review (NOT re-verified in this pass — caveman agent did not re-write the file). Main thread should verify before fixing:**
- gpu_gate.py:126-128 GPU lock fd leak on write failure → 5-min deadlock
- journal.py:568,580 `write_context()` uses `Path.write_text()` (not atomic)
- health.py:66 fromisoformat timezone mismatch kills dashboard health endpoint
- claude_gate.py:97-104 `_count_today_invocations` scans full file every call

### Main-thread cross-cut (8 P0)
Per `docs/ADVERSARIAL_REVIEW_INDEPENDENT_2026-05-24.md` (committed alongside this synthesis). Highlights:
- IND-P0-1 through IND-P0-8 all overlap one-or-more agent findings (corroboration above for T1, T2, T3, T4, T6, T7, T9, T10).
- IND-P1-2/3: `_build_regime_context` swallows exceptions silently and depends on undocumented `_buy_count`/`_sell_count` keys.
- IND-P1-4: utility-boost magnitude still over-amplifies (`boost = min(1.0 + u_score, 1.5)` from a 0.5pp avg-return gives 50% accuracy bump). Gate bypass is fixed; magnitude is still wrong.

---

## Recommended fix sequencing

### Batch 1 (one PR, ~3-5 file edits)
Single concurrency primitive sweep — closes T2 + T3 + a chunk of P1 cross-process races:
1. Add `file_utils.update_json_atomic(path, mutate_fn)` wrapping `jsonl_sidecar_lock(path)` + `load_json` + `mutate_fn` + `atomic_write_json`.
2. Migrate `portfolio_mgr.update_state`, `save_state`, `save_bold_state` to use it.
3. Add `update_warrant_state(mutate_fn)` to `warrant_portfolio` and migrate `record_warrant_transaction`.
4. Migrate `trade_guards._save_state` (P1-C).
5. Tests: spawn two processes that each mutate state via the helper; assert no lost updates.

### Batch 2 (one PR, ~2 file edits)
Bias-penalty deduplication — closes T1:
1. `accuracy_stats.py:849` — drop `bias_penalty` from `normalized_weight` computation.
2. Add regression test that calendar-bias signal voting contrarian retains full weight.
3. Recompute cached signal weights (no live deploy until accuracy stats re-stabilize over 24h).

### Batch 3 (one PR, ~3 file edits)
Direction schema migration — closes T4:
1. Add `direction` field to holding schema in `warrant_portfolio` defaults.
2. Plumb sign through `warrant_pnl`, `risk_management.compute_stop_levels`, `risk_management.compute_probabilistic_stops`, `risk_management.check_atr_stop_proximity`, `exit_optimizer._compute_pnl_sek`, `monte_carlo.simulate_ticker`.
3. Tests: BEAR cert with rising underlying must report negative P&L and not trigger stop.

### Batch 4 (small fixes — one PR each)
- T6 (account-id filter in `get_positions`): one-line `if account.get("id") not in ALLOWED_ACCOUNT_IDS: continue`.
- T7 (off-hours autonomous fallback in main.py:972-979): copy lines 960-966.
- T8 (specialist_timeout_s default): 30 → 150s.
- T9 (Binance error guard): one-line `isinstance(data, dict) and "code" in data` check.
- T10 (grid_fisher + fin_snipe stop buffer): 0.995 → 0.97; 0.01 → 0.03.

### Batch 5 (medium effort)
- Kelly metals overhaul (T5).
- Vote double-count refactor in 4 signals-modules files.
- Silent-fail audit: introduce `SignalResult` contract; downgrade `Exception` to specific types.

---

## Notes & process observations

1. **Caveman reviewer doesn't always overwrite the target file.** The infrastructure reviewer printed findings to the conversation transcript but did NOT use Write tool on the docs file. Worth tracking — possibly the agent prompt needs an explicit "use Write tool to overwrite the file" instruction even though we already say "OVERWRITE prior file". I transcribed its summary into `AGENT_REVIEW_INFRASTRUCTURE.md` manually post-hoc.

2. **Agent file path inconsistency.** Some agents wrote into `Q:\finance-analyzer\docs\` (main repo). Others wrote into `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24\docs\` (worktree). The data-external agent did the latter; I copied it to main repo for synthesis. The protocol should specify "write to MAIN repo docs/, not the worktree's docs/".

3. **Worktree created inside the parent repo.** `git worktree add Q:/finance-analyzer-reviews/...` resolved to `Q:/finance-analyzer/finance-analyzer-reviews/...` because the path was relative to the CWD which is the repo root. Cosmetic; cleanup still works.

4. **Empty-baseline branch had 0 commits and required `--allow-empty`.** The CLAUDE.md path conventions and gitconfig (`user.email`/`user.name`) had to be set inline because the orphan branch is created before any other config can apply.

5. **Synthesis-time discrepancy with data-external agent on T9.** Data-external agent claimed prior P0 Binance dict-as-DataFrame is now fixed; main-thread reading of the live file shows it's still unguarded. Possible the agent was reading the worktree copy at a different timestamp, or misread the existing `if not data` guard as sufficient. **Action: human re-check `portfolio/data_collector.py:88-93` and decide.**

---

_Last updated: 2026-05-24 17:40 CET. Metals-core subsystem to be appended once its agent returns._
