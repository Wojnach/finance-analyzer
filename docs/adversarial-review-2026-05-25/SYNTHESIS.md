# Adversarial Review Synthesis — 2026-05-25

Eight subsystem reviewers (Claude code-reviewer agents) + one independent
self-review pass on the main thread. Total: ~9 hours of agent compute,
~50K LOC reviewed, ~100 findings catalogued.

## Reviewers

| # | Subsystem | P0 | P1 | P2 | P3 | Path |
|---|-----------|----|----|----|----|------|
| 0 | self-review (cross-cutting) | 1 | 6 | 3 | 1 | `00-self-review.md` |
| 1 | signals-core | 0 | 4 | 10 | 4 | `01-signals-core.md` |
| 2 | orchestration | 5 | 8 | — | — | `02-orchestration.md` |
| 3 | portfolio-risk | 3 | 10 | 12 | — | `03-portfolio-risk.md` |
| 4 | metals-core | 3 | 10 | 8 | 4 | `04-metals-core.md` |
| 5 | avanza-api | 4 | 8 | — | — | `05-avanza-api.md` |
| 6 | signals-modules | 4 | 14 | 16 | 4 | `06-signals-modules.md` |
| 7 | data-external | 3 | 8 | 7 | 2 | `07-data-external.md` |
| 8 | infrastructure | 0 | 11 | 9 | — | `08-infrastructure.md` |

## TL;DR

**Hot live-trading paths are hardened.** The signal voting engine, the
atomic JSON I/O primitive, dashboard auth, claude-subprocess hygiene
(stdin=DEVNULL, CLAUDECODE unset, PF_HEADLESS_AGENT, auth-marker scan),
stop-loss API routing, EOD/DST handling, drawdown circuit breaker —
all have layers of guards built up over ~25 prior reviews.

**Where bugs still live:** new typed `portfolio/avanza/` package
re-implements public surface without the safety guards that
`portfolio/avanza_session.py` carries; subprocess tree-kill missing
on Layer 2 + multi-agent specialist spawns; warrant portfolio
read-modify-write race unfixed across 6+ prior reviews; in-memory API
quota counters that reset on every loop restart; cross-process locks
where threading-only locks were assumed; and a smattering of
data-mislabeling / silent-fallback paths that mask broken upstream
APIs.

## P0 findings — convergent across multiple reviewers

### P0-A. Warrant portfolio cross-process write race
**Cited by**: self-review, portfolio-risk
**File**: `portfolio/warrant_portfolio.py:198-265`
**Severity**: real-money loss / state corruption
**Symptom**: `record_warrant_transaction()` does load-mutate-save of `portfolio_state_warrants.json` with no lock (neither in-process `threading.Lock` nor cross-process file lock). grid_fisher, fin_snipe, metals_loop, dashboard all read+write this file in different processes. Two concurrent fills silently lose one transaction.
**Recurrence**: flagged in 2026-05-08, 05-11, 05-13, 05-22, 05-23, 05-24 reviews. Unfixed.
**Fix**: mirror `portfolio_mgr.update_state()` — `_state_locks[path]` + `_rotate_backups` + sidecar `jsonl_sidecar_lock` for cross-process safety.

### P0-B. Typed avanza package bypasses ALL safety guards
**Cited by**: avanza-api (4 separate P0 findings)
**Files**: `portfolio/avanza/trading.py`, `portfolio/avanza/account.py`, `portfolio/avanza/client.py`
**Severity**: pension account hit, double orders, missing order-lock contention
**Symptom**:
- `place_order/modify_order/cancel_order/place_stop_loss/place_trailing_stop/delete_stop_loss` lack the `ALLOWED_ACCOUNT_IDS={"1625505"}` whitelist that `avanza_session.py:589-606` carries (H7/H8/BUG-211 guards) — pension account 2674244 can be hit.
- Same functions lack the `avanza_order_lock` wrap that `avanza_session.py:620` carries — cross-process order races.
- `client.py:65` reads `account_id` from config with no whitelist verification.
- `account.get_positions()` returns positions from EVERY account by default — violates `feedback_isk_only.md`.
**Fix**: every mutating function: top-of-function `if acct not in ALLOWED_ACCOUNT_IDS: raise ValueError`; wrap with `avanza_order_lock`; share `ALLOWED_ACCOUNT_IDS` constant via `portfolio/avanza/constants.py` (single source of truth).

### P0-C. Layer 2 + specialist subprocess spawns lack tree-kill
**Cited by**: orchestration (2 findings)
**Files**: `portfolio/agent_invocation.py:1163`, `portfolio/multi_agent_layer2.py:127-242`
**Severity**: orphan Node processes, RAM exhaustion over time, kill_orphaned_llama can't reach Claude CLI's Node children
**Symptom**: `subprocess.Popen` called without `creationflags=CREATE_NEW_PROCESS_GROUP` / `start_new_session=True`. `claude_gate._popen_kwargs_for_tree_kill()` exists for exactly this purpose but isn't used. On Windows, `taskkill /F /T /PID` usually works but invariants are fragile; on Unix, `proc.kill()` leaks grandchildren.
**Fix**: use `_popen_kwargs_for_tree_kill()` for every Claude subprocess spawn site.

### P0-D. Layer 2 kill wedge → infinite log loop
**Cited by**: orchestration
**File**: `portfolio/agent_invocation.py:629-737`
**Severity**: Layer 2 wedged forever, no alert, invocations.jsonl bloat
**Symptom**: When `taskkill` fails AND `wait(15)` times out, `_agent_proc` stays set ("block respawn"). Watchdog re-fires every 30s, re-runs taskkill+wait, all under `_completion_lock`. No critical_errors.jsonl entry on repeat. Main loop blocked from new invocations indefinitely.
**Fix**: (a) cooldown on repeat-pid kill attempts; (b) critical_errors.jsonl on first failure; (c) suppress duplicate timeout rows for same pid.

### P0-E. Auth-error masked as timeout defeats 30-min cooldown
**Cited by**: orchestration
**File**: `portfolio/agent_invocation.py:716-731`
**Severity**: auth-error storm resumes after a single timeout
**Symptom**: `_scan_agent_log_for_auth_failure` runs on timeout but `_log_trigger` writes `status="timeout"` (not `"auth_error"`). The cooldown gate at line 766-783 checks for `status == "auth_error"` only — timed-out auth failures bypass the cooldown. Storm of doomed Claude spawns can resume after a single timeout. Same class as the 2026-03/04 silent auth outage.
**Fix**: upgrade `status` to `"auth_error_timeout"` (or `"auth_error"`) when scan returns True; ensure cooldown matches either label.

### P0-F. Crash counter never resets pre-`run()` → silent alerts after 5 boots
**Cited by**: orchestration
**File**: `portfolio/main.py:1108-1130`
**Severity**: silent loop boot-loop, exact failure mode CLAUDE.md warns about
**Symptom**: `_consecutive_crashes` loaded from disk at module import, only reset after successful `run()`. Pre-run crashes (config validator throw, singleton lock fail, Telegram poller construction error) bump counter unbounded. After 5: alerts silenced.
**Fix**: reset counter on `_acquire_singleton_lock()` success (fresh-instance = healthy boot).

### P0-G. exit_optimizer BEAR-cert sign flip
**Cited by**: portfolio-risk
**File**: `portfolio/exit_optimizer.py:325-332`
**Severity**: every EV/PnL calculation for BEAR certs is sign-inverted
**Symptom**: warrant branch without `financing_level` computes `warrant_move = pct_move × position.leverage`, treating leverage as signed long. BEAR cert (5x) with underlying +1% → reports +5% warrant move when actual is −5%. `grid_fisher` actively trades BEAR certs. System will "take profit" on losers, "hold" winners.
**Fix**: add `direction` to `Position` (BULL/BEAR) and flip sign when computing `warrant_move` for BEAR.

### P0-H. portfolio_mgr update_state is process-local only
**Cited by**: portfolio-risk
**File**: `portfolio/portfolio_mgr.py:136-159`
**Severity**: cross-process write race on portfolio_state{,_bold}.json
**Symptom**: `update_state` uses in-process `threading.Lock`. Layer 2 subprocess, dashboard, mstr/crypto/oil loops all read+write. Layer 2 BUY can be clobbered by dashboard `/api/validate-portfolio` POST handler that loaded stale state.
**Fix**: add `jsonl_sidecar_lock` (file lock) around update_state's load-mutate-save window.

### P0-M. Daily-bar lookback constants applied to 15m bars across many signal modules
**Cited by**: signals-modules
**Files**: `portfolio/signals/gold_real_yield_paradox.py:282-291`, `cross_asset_tsmom.py:107-127,191`, `network_momentum.py:107-148,166`, `metals_vrp.py:125-153`, `cubic_trend_persistence.py:58-70,114-121`, `hash_ribbons.py:158-179`, `trend_slope_momentum.py:97-100,28-35`
**Severity**: silent garbage votes when these signals are re-enabled (today all DISABLED — latent P0)
**Symptom**: main.py:492-504 feeds `interval="15m", limit=100`. Modules use lookback constants sized in "trading days" (252, 60, 50, 30, 20) and apply them to 15m bars — silently shrinking lookback ~96×. `gold_30d_return` reads `close.iloc[-30]` (= 7.5h instead of 30d) and correlates it against 30 daily yield obs; `_compute_own_tsmom(close)` uses `iloc[-253]` (= 63h instead of 252d) and mixes with 252-day yfinance daily peers; `metals_vrp` annualises with `sqrt(252)` instead of `sqrt(252*96)`, underestimating realized vol ~9.8×.
**Fix**: each signal resamples df to daily inside (or fetches own daily series); document timeframe explicitly in indicator labels. Add a `_assert_timeframe()` helper that signals call when they need daily bars.

### P0-N. bocpd_regime_switch sub-signals violate {BUY,SELL,HOLD} contract
**Cited by**: signals-modules
**File**: `portfolio/signals/bocpd_regime_switch.py:227-232`
**Severity**: any downstream consumer using `votes.count("BUY")` over sub_signals gets garbage; today no consumer but landmine for future accuracy-by-sub-signal analysis
**Symptom**: emits "BREAK", "STABLE", "changepoint_mr", "trending"… as sub-signal values. `_validate_signal_result` only checks outer dict type.
**Fix**: extend validator to validate each sub-signal value against {BUY, SELL, HOLD} or coerce.

### P0-J. grid_fisher places duplicate orders on session-call timeout
**Cited by**: metals-core
**File**: `portfolio/grid_fisher.py:1392-1424` + `:997-1040`
**Severity**: 2× notional on a single tier; blows past per-instrument AND global caps
**Symptom**: `_safe_session_call` uses `future.result(timeout=30)`. If Avanza accepts order but response is slow, future times out, helper returns None, tier is NOT appended to `inst.buy_ladder`. Next tick reissues the same tier as a fresh order. Original is invisible because no `order_id` recorded.
**Fix**: persist a PENDING idempotency key before the call; require positive "order-not-found-on-Avanza" confirmation before retrying.

### P0-K. emergency_sell ships `price=0` SELL orders (latent — gated off)
**Cited by**: metals-core
**Files**: `data/metals_loop.py:2074-2075, 3736-3797`
**Severity**: dormant today (`EMERGENCY_SELL_ENABLED=False`) — but the moment the flag flips, malformed prices ship to broker
**Symptom**: `_eod_sell_fishing_positions` explicitly calls `emergency_sell(page, key, pos, 0)` when no bid available; `emergency_sell` constructs raw `page.evaluate` POST with `"price": 0` — no client-side guard (unlike `_place_order`).
**Fix**: refuse SELL payload when `bid <= 0`; for forced flat, use aggressive market-style limit (last seen × 0.95) like `grid_fisher.eod_market_flat`.

### P0-L. grid_fisher cross-account position confusion
**Cited by**: metals-core (and overlaps avanza-api findings)
**Files**: `portfolio/grid_fisher.py:666-675` + `portfolio/avanza_session.py:676-717`
**Severity**: phantom inventory from pension account → naked sells from ISK account
**Symptom**: `avanza_session.get_positions()` returns positions across ALL accounts. `_position_volume_for` returns first orderbook_id match — no account filter. If pension account 2674244 holds same ob_id as ISK 1625505, grid sees `live_vol > inventory_units`, records phantom fills at limit price, rotates to SELL the ISK account doesn't actually have. Avanza rejects with `short.sell.not.allowed`; state now claims phantom inventory.
**Fix**: `avanza_session.get_positions(account_id=...)` must accept and filter; grid_fisher must pass `self.account_id`.

### P0-I. Three external-API quota counters reset on every restart
**Cited by**: data-external
**Files**: `portfolio/shared_state.py:304-336` (NewsAPI), `portfolio/alpha_vantage.py:30-34` (Alpha Vantage), `portfolio/onchain_data.py` (BGeometrics — has none)
**Severity**: account suspension / billing overage
**Symptom**: In-memory module-level counters. Loop restart loop (crash → restart → fetch → crash) can burn 3× daily budget. AV: `earnings_calendar._fetch_earnings_alpha_vantage` bypasses the counter entirely.
**Fix**: persist `(count, last_reset_utc_date)` to `data/<api>_quota.json`; load on import; check BEFORE every quota'd call (no bypass paths).

## P1 findings — multi-reviewer

### P1-A. btc_proxy injected after dispatch loop bypasses DISABLED_SIGNALS guard
**Cited by**: signals-core (and confirmed by main thread)
**File**: `portfolio/signal_engine.py:3840`
**Severity**: disabled signal still pollutes unweighted consensus counts
**Symptom**: `tickers.py:DISABLED_SIGNALS` includes `btc_proxy` (44.6% 1d). Dispatch loop at line 3664 skips disabled signals, but lines 3834-3843 inject `btc_proxy` directly for MSTR after the loop with no guard. Downstream: `buy/sell` counts (line 4050-4051), `active_voters` (4063 — MIN_VOTERS quorum), `_buy_count/_sell_count` written to `extra_info` (4331-4332, consumed by unanimity penalty + entropy). Weighted consensus correctly gates via accuracy, but **unweighted action** and entropy calculation are warped.
**Fix**: `if "btc_proxy" not in DISABLED_SIGNALS: votes["btc_proxy"] = btc_action`.

### P1-B. ic_cache.json single-horizon clobber on multi-horizon refresh
**File**: `portfolio/main.py:1085-1091` + `portfolio/ic_computation.py:235-262`
**Symptom**: `compute_and_cache_ic(h)` overwrites entire cache with one horizon. main.py refreshes ("3h", "1d") sequentially → second wipes first. After process restart, only the last-written horizon is on disk. In-memory cache helps during runtime but cold-start cycles miss.
**Fix**: per-horizon nested dict `{"3h": {...}, "1d": {...}}` or separate files `ic_cache_<h>.json`.

### P1-C. accuracy_stats naive-vs-aware timestamp comparison silently swallowed
**File**: `portfolio/accuracy_stats.py:280-289, 327-332, 1641-1643`
**Symptom**: `signal_accuracy_ewma` catches `(ValueError, TypeError)` on subtraction → silently `continue`. Today all writers use `datetime.now(UTC).isoformat()` so it works, but any future writer dropping `+00:00` silently zeros EWMA weighting with no log.
**Fix**: explicit `if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=UTC)`.

### P1-D. Browser-recovery retry double-submits orders
**File**: `portfolio/avanza_session.py:212-232`
**Symptom**: `_with_browser_recovery` retries `op(ctx)` once on `TargetClosedError`. If POST reached Avanza but response was lost, retry re-issues same order. `avanza_order_lock` does not deduplicate same-caller retries.
**Fix**: classify ops mutate vs read; never retry mutations on browser death — close, raise, let caller inspect `get_open_orders()` before acting.

### P1-E. HTTP retry ignores standard Retry-After header
**File**: `portfolio/http_retry.py:48-55`
**Symptom**: only Telegram-body retry_after parsed; standard HTTP `Retry-After` header ignored. Binance escalates IP-ban duration when client keeps requesting during ban.
**Fix**: `resp.headers.get("Retry-After")` first, body fallback second, exponential third.

### P1-F. Kelly _compute_trade_stats double-counts fees
**File**: `portfolio/kelly_sizing.py:92, 99-103`
**Symptom**: `BUY.total_sek` includes fee, `SELL.total_sek` is net. `pnl_pct = (sell_net - buy_gross) / buy_gross` counts fee on both legs → win-rate biased pessimistic, Kelly under-sizes.
**Fix**: subtract fee from BUY.total_sek or use `shares × price_usd` directly.

### P1-G. compute_stop_levels unaware of MINI knockout barriers
**File**: `portfolio/risk_management.py:320-395`
**Symptom**: Computes flat ATR-based stop. Barrier check lives only in `exit_optimizer._apply_risk_overrides`. Any caller using `compute_stop_levels` output to PLACE an order can get a stop sitting below the knockout → guaranteed knockout fill before stop triggers. Memory rule violated.
**Fix**: pass instrument metadata (barrier_price for MINI) to `compute_stop_levels`; refuse to compute stop within 3% of barrier.

### P1-H. equity_curve wins-by-pct, profit-factor-by-sek inconsistency
**File**: `portfolio/equity_curve.py:494-528`
**Symptom**: A trade with `pnl_pct > 0` and `pnl_sek < 0` (fees > pct gain) counts as a win in `wins[]` AND reduces gross_profit in profit_factor. Contradictory truths from the same trade.
**Fix**: classify wins by `pnl_sek > 0`.

### P1-I. monte_carlo_risk fx_rate=1.0 silent fallback understates VaR 10×
**File**: `portfolio/monte_carlo_risk.py:408`
**Symptom**: bypasses `_resolve_fx_rate` (which exists to reject stale 1.0). If `agent_summary.fx_rate == 1.0`, SEK VaR/CVaR reads 10× understated. Layer 2 oversizes accordingly.
**Fix**: call `_resolve_fx_rate(agent_summary)`.

### P1-J. Atomic-append lock-acquire failure silently loses entries
**File**: `portfolio/file_utils.py:269-292` + sidecar lock at `:240-247`
**Symptom**: If sidecar create succeeds but `open("rb+")` fails (AV scan, sidecar deleted), contextmanager re-raises; append never happens; entry lost. Callers (`journal`, `message_store`, `telegram_poller`) let exception bubble.
**Fix**: on lock OSError, fall back to process-local `threading.Lock` so write still happens (durability > strict ordering for telemetry).

### P1-K. CF-Access success path sets pf_dashboard_token cookie to config secret
**File**: `dashboard/auth.py:152-153`
**Symptom**: After JWT-verified CF-Access request, cookie is set with literal config token. If CF-Access scope is later removed for the operator, the cookie still grants dashboard access for 365 days. Bypasses the access policy.
**Fix**: only set cookie on token-/bearer-auth success. CF-Access requests don't need the cookie.

### P1-L. _get_config caches {} for 60s on cold-start race → all-open auth window
**File**: `dashboard/auth.py:60-80`
**Symptom**: Transient read failure (AV scan + symlink to external file) returns `{}`. Cached for 60s. `_get_dashboard_token()` returns None → `require_auth` falls back to "no token = open access". 60-second open window on every dashboard cold start.
**Fix**: on cold-start failure, leave `_CFG_VALUE=None` (force re-read); fail closed if token unknown.

### P1-M. gpu_gate _pid_alive fails open without psutil
**File**: `portfolio/gpu_gate.py:73-83`
**Symptom**: Returns `False` (assumes dead) when psutil missing. Legitimate Chronos forecast at 200s gets lock reaped. Both processes load models → VRAM OOM. PID-reuse defeats stale-lock detection (25h wedge of 2026-05-02 can recur).
**Fix**: require psutil in `Q:/models/.venv-llm` requirements; fail CLOSED when missing; persist `executable_name` in lock metadata, validate via `Process(pid).name()` before honoring "alive".

### P1-S. credit_spread / gold_real_yield_paradox reintroduce relative-path config bug
**Files**: `portfolio/signals/credit_spread.py:283-288`, `gold_real_yield_paradox.py:262-269`
**Symptom**: fallback `_get_fred_key` reads `load_json("config.json", default={})` with no absolute-path resolution. PF-DataLoop runs under `C:\Windows\System32` on fresh Task Scheduler start → cwd-relative read fails silently → key returns empty → signal returns `empty` for the rest of the session. Exact bug cot_positioning fixed on 2026-05-02 (SM-P1-4).
**Fix**: mirror `_DATA_DIR = Path(__file__).resolve().parent.parent.parent` pattern.

### P1-T. hurst_regime double-counts vote
**File**: `portfolio/signals/hurst_regime.py:283-285,300-302`
**Symptom**: `sub_signals["hurst_regime"] = trend_vote` AND `sub_signals["trend_direction"] = trend_vote` (same value twice). `majority_vote` sees same vote twice, gives it 2/3 weight.
**Fix**: one canonical slot per distinct contribution.

### P1-U. crypto_evrp percentile mislabel + contradictory level/momentum theses
**File**: `portfolio/signals/crypto_evrp.py:204-243, 248-261`
**Symptom**: `evrp_percentile` sub-signal actually computes DVOL percentile (fallback path) — mislabeled. Separately, `_evrp_level_signal` and `_evrp_momentum_signal` carry contradictory semantic models (high eVRP→SELL vs rising DVOL→SELL) that fight each other in composite.
**Fix**: rename sub-signal `dvol_percentile`; reconcile level/momentum theses.

### P1-V. inversion / sub-signal divergence on safe-haven invert
**Files**: `copper_gold_ratio.py:251-264`, `treasury_risk_rotation.py:182-198`, `xtrend_equity_spillover.py:226-252`
**Symptom**: Headline `action` is inverted for metals/safe havens but per-sub-signal vote slots stay un-inverted. Per-sub-signal accuracy tracking on those tickers is meaningless.
**Fix**: invert sub_signals symmetrically OR store both raw and inverted with explicit names.

### P1-W. residual_pair_reversion index alignment → empty dataframe forever
**File**: `portfolio/signals/residual_pair_reversion.py:276-296`
**Symptom**: `df["close"]` carries RangeIndex (binance fetcher writes timestamp to `df["time"]`, not index). `target_close.index = pd.to_datetime(df.index)` converts integer positions → datetimes near 1970-01-01. Joined with daily DatetimeIndex driver → inner-join `.dropna()` is empty every call. Signal returns empty-hold result for process lifetime. Disabled today; will deliver zero votes silently when re-enabled.
**Fix**: use `pd.to_datetime(df["time"], unit="ms")` for the index.

### P1-O. fish_engine peak/trough state not persisted (Bug 3 silently regresses on every restart)
**File**: `data/fish_engine.py:1006-1048` (to_dict/from_dict)
**Symptom**: `underlying_peak_price/ts`, `underlying_trough_price/ts` (added 2026-04-13 to veto ORB continuation during pullbacks) are NOT serialized. After every process restart, veto disables and engine can fire LONG entries 4.7% below intraday peak — exact loss pattern Bug 3 was written to prevent.
**Fix**: add 4 fields to to_dict/from_dict.

### P1-P. minutes_until_eod returns inf on missing zoneinfo → silent overnight holds
**File**: `portfolio/grid_fisher.py:285-310`
**Symptom**: Returns `float("inf")` when zoneinfo or tzdata unavailable. Docstring frames as fail-safe; actually fail-DANGEROUS — EOD market-flat never fires, 5x leveraged metals/oil positions silently held overnight through gap risk.
**Fix**: log critical_errors.jsonl AND fall back to UTC-based cutoff (CET=UTC+1, CEST=UTC+2) when zoneinfo missing.

### P1-Q. fin_snipe_manager stop placement has no barrier-distance check
**File**: `portfolio/fin_snipe_manager.py:529-563`
**Symptom**: `trigger_price = position_avg * 0.95` (5% below entry). No check vs MINI knockout barrier. For 10-30x MINIs with barriers within 3-14% of underlying, a 5% trigger can place stop AT or PAST knockout → already-knocked-out cert → stop never fires → position decays to ~0. Memory `feedback_mini_stoploss.md` violated.
**Fix**: when `pos.api_type == "warrant"` and `barrier > 0`, compute implied underlying at trigger_price and refuse if within X% of barrier.

### P1-R. microstructure_state double-records OFI per persist cycle
**File**: `portfolio/microstructure_state.py:175-213`
**Symptom**: `get_microstructure_state` records OFI; `persist_state` (called every 5th cycle in metals_loop) ALSO calls `get_microstructure_state` → records OFI again. `_ofi_history` deque skewed; z-score compressed; orderbook_flow signal biased toward HOLD.
**Fix**: split read from mutation; `record_cycle_ofi` is the only mutator, called explicitly once per main cycle.

### P1-N. health.update_health vs dashboard read race → momentary RED
**File**: `portfolio/health.py:20-41, 338-368`
**Symptom**: `update_health` atomic-replaces full blob; dashboard `load_json` swallows transient OSError (Windows os.replace of open file) → returns `{}` → dashboard shows `cycle_count=0`, flips RED. 5s TTL cache reduces but doesn't eliminate.
**Fix**: retry read once on default-fallback.

## Cross-reviewer agreement matrix

| Finding | self | sig-core | orch | port | aval | data | infra |
|---------|------|----------|------|------|------|------|-------|
| warrant_portfolio race | ✓ | | | ✓ | | | |
| Layer 2 subprocess hygiene | ✓ (status quo) | | ✓ | | | | |
| Atomic I/O compliance | ✓ | | | | | | |
| God-file decomposition | ✓ | ✓ (signal_engine bloat) | | | | | |
| Auth-cooldown gate | | | ✓ | | | | |
| MIN_VOTERS doc drift | ✓ | | | | | | |
| Cross-process write races | ✓ | | | ✓ | ✓ | ✓ | |
| External API quota persistence | | | | | | ✓ | |

(Other findings are subsystem-specific; not amenable to cross-tabulation.)

## Contested / false-positive sweep

- **orchestration `_kill_overrun_agent` (P0-D)** — self-review did not independently surface this, but reading the code at agent_invocation.py:629-737 confirms: the watchdog re-fires every 30s holding `_completion_lock` during a 15s `wait()`, and a kill-failure leaves `_agent_proc` set. The conditions for the infinite loop are real. Not contested.
- **infrastructure P1-K (CF-Access cookie)** — design point worth confirming with the operator. The auth module's `_refresh_cookie` is called on the CF-Access path intentionally to make subsequent same-browser visits skip CF (latency). Pruning it changes UX. Flag as P1 to review intent, not P0 to fix immediately.
- **portfolio-risk P0-G (BEAR-cert)** — verify by checking which callers route BEAR certs through `_compute_pnl_sek`. If only `exit_optimizer.compute_exit_plan` is called and only for LONG positions, severity drops. Need callers audit.
- **`signal_weights.py` dead-code (signals-core P2)** — confirmed dead by `outcome_tracker.py:497-500` comment. Safe to delete.

## Cross-cutting themes

1. **Recurring P0 backlog.** Two of the convergent P0s (warrant race, typed-avanza guards) have been flagged in 4+ prior reviews. The auto-improve loop and adversarial-review cadence surface them but a *fix* batch hasn't landed. Recommend committing a "convergent-P0 backlog" doc and assigning each one to a named PR.

2. **Subprocess hygiene.** stdin=DEVNULL is fixed everywhere. tree-kill flags missing on Layer 2 + specialists. PF_HEADLESS_AGENT is set everywhere. CLAUDECODE unset everywhere. Auth-marker scan exists but doesn't propagate to `status="auth_error"` on timeout. The five elements of subprocess hygiene need a single audit point — recommend a `_safe_claude_popen()` helper that bundles all five guarantees.

3. **In-memory state that survives crashes is the dominant silent-failure class.** API quota counters (NewsAPI, AV, BGeometrics), F&G streak race, gold-BTC ratio history race, dashboard config cache, Layer 2 crash counter, `last_full_review_time` lost on restart. Pattern: any "state I refresh once per day/hour" that lives in module globals will burn quotas / lose state on restart. Recommend a small `persistent_counter.py` module that all daily-quota counters use.

4. **Twin code paths drift.** `portfolio/avanza_session.py` (legacy) has BUG-211 / H7 / H8 guards; `portfolio/avanza/` (typed) does not. `risk_management.compute_stop_levels` does ATR; `exit_optimizer._apply_risk_overrides` checks barriers. `kelly_sizing` uses % of cash; `check_concentration_risk` uses % of total_value. Each twin enforces a different invariant — neither is wrong but together they create gaps. Recommend explicit "delegate to canonical implementation" pattern when a refactor extracts a typed package.

5. **Doc drift.** `.claude/rules/signals.md` says MIN_VOTERS=3 universal but code has METALS=2 since 2026-05-11. `.claude/rules/infrastructure.md` says T1=150s but code has 180s. CLAUDE.md says 142 modules; actual ~406. Rules files should be generated, not maintained.

6. **God files.** signal_engine.py (4476), grid_fisher.py (1970), metals_loop.py (7880), dashboard/app.py (2353), agent_invocation.py (1724), main.py (1532). These accumulate complexity faster than tests can pin. The signal-engine reviewer noted "8+ orthogonal gate cascades with no integration test that exercises all in sequence" — a property-based test exercising the full gate cascade would have caught the btc_proxy injection bug (P1-A) earlier.

7. **What's *not* broken.** No raw `json.dump()` direct write outside file_utils. No subprocess `shell=True` taking user input. No `f"...{api_key}..."` log formatting. No unauthenticated `/api/*` endpoint. No bypass of stop-loss API to regular order API. No `claude` invocation that doesn't unset CLAUDECODE. EOD-DST via zoneinfo not magic UTC offsets. Empty-ticker guard live. The system's "you must not regress these" invariants are holding.

## Top-10 action backlog (recommend prioritising)

| Rank | Finding | Effort | Why |
|------|---------|--------|-----|
| 1 | P0-A warrant_portfolio race | small | Recurring P0 in 6+ reviews. Fix is mechanical (mirror portfolio_mgr pattern). |
| 2 | P0-B typed-avanza guards | medium | Concentrates risk in a new code path; cheap to add whitelist+lock; high downside (real money to wrong account). |
| 3 | P0-D Layer 2 kill wedge | small | One pid-cooldown + one critical_errors.jsonl write. |
| 4 | P0-E auth-error cooldown | trivial | Two-line label fix. Reopens defeated cooldown. |
| 5 | P0-I quota persistence | small | One persistent_counter.py module reused by 3 sites. |
| 6 | P1-A btc_proxy guard | trivial | One-line `if not in DISABLED_SIGNALS` guard. |
| 7 | P0-G BEAR-cert sign flip | small | Add `direction` to Position; flip in one branch. |
| 8 | P0-C tree-kill flags | small | Add `**_popen_kwargs_for_tree_kill()` to 2 spawn sites. |
| 9 | P0-F crash counter | small | Reset on singleton lock acquire. |
| 10 | P0-H portfolio_mgr cross-process lock | small | Add `jsonl_sidecar_lock` to update_state. |

## Top-10 backlog — UPDATED with metals+signals findings

| Rank | Finding | Effort | Why |
|------|---------|--------|-----|
| 1 | P0-A warrant_portfolio race | small | Recurring P0 in 6+ reviews. Mechanical mirror of portfolio_mgr pattern. |
| 2 | P0-B typed-avanza guards | medium | Concentrates risk in new code path; high downside (real money to wrong account). |
| 3 | P0-L grid_fisher cross-account confusion | medium | Roots fix in `avanza_session.get_positions(account_id=...)`; touches multiple callers. |
| 4 | P0-J grid_fisher dup orders on timeout | medium | PENDING marker before placement; reconcile via order-id discovery. |
| 5 | P0-I quota persistence (3 sites) | small | One persistent_counter.py reused by NewsAPI/AV/BGeometrics. |
| 6 | P0-D Layer 2 kill wedge cooldown | small | One pid-cooldown + one critical_errors.jsonl write. |
| 7 | P0-E auth-error cooldown label | trivial | Two-line label fix. Reopens defeated cooldown. |
| 8 | P0-G BEAR-cert sign flip in exit_optimizer | small | Add `direction` to Position; flip sign in one branch. |
| 9 | P0-F crash counter reset on singleton-acquire | small | Reset in one line after lock-acquire. |
| 10 | P0-C tree-kill flags on Layer 2 + specialist Popen | small | Add `**_popen_kwargs_for_tree_kill()` to 2 sites. |

**Honorable mentions** (P1 but trivial fixes — bundle into a single "drift cleanup" PR):
- P1-A `btc_proxy` guard (one-line `if not in DISABLED_SIGNALS`)
- P1-B `ic_cache.json` per-horizon nested dict
- P1-O `fish_engine` peak/trough persistence (4 fields to to_dict/from_dict)
- P1-S relative-path config fallback in `credit_spread` + `gold_real_yield_paradox` (mirror `_DATA_DIR` pattern)
- P1-T `hurst_regime` double-counted vote
- Doc drift: `.claude/rules/{signals,infrastructure}.md` vs code

## Recurring (≥3 reviews) — must-fix backlog

These findings have been flagged in *prior* adversarial reviews AND landed again in 2026-05-25:

1. **warrant_portfolio race** (P0-A) — flagged 2026-05-08, 05-11, 05-13, 05-22, 05-23, 05-24, 05-25
2. **typed avanza guards** (P0-B) — flagged in 2026-05-21, 05-23 reviews under "avanza module split"
3. **subprocess tree-kill flags** (P0-C) — flagged in 2026-04-16, 05-12 reviews under "claude_gate vs agent_invocation drift"
4. **API quota counter persistence** (P0-I) — flagged in 2026-05-13, 05-19 reviews

The auto-improve loop surfaces these consistently but no fix batch has landed. Recommend a dedicated 1-day "convergent-P0 sprint" with named PR per item.

## Worktrees + branches

8 worktrees created under `Q:/finance-analyzer-worktrees/review-<sub>` on branches `review/<sub>`, all on baseline commit `3d3830ba`. To be removed and branches deleted after this synthesis is committed to main.
