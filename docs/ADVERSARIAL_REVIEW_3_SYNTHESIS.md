# Dual Adversarial Review Synthesis — Round 3 (2026-04-08)

**Methodology**: 8 parallel code-reviewer subagents (one per subsystem) + independent Claude
direct review. Cross-critique applied. All findings deduplicated.

**Scope**: Full codebase — 8 subsystems, ~25,000 lines. Focused on: (1) verifying Round 2
fixes were actually applied, (2) new code since Round 2, (3) cross-cutting structural risks.

**Round 2 status**: Of ~40 action items from Round 2, **~8 were fixed** and ~32 remain open.

---

## Executive Summary

Three systemic failure modes dominate Round 3:

1. **Atomic I/O compliance breakdown**: Critical Rule 4 ("never raw `open()`") is violated in
   15+ locations across metals_loop.py, metals_risk.py, metals_swing_trader.py, and
   portfolio/risk_management.py. Some of these files are the drawdown circuit breaker's data
   sources. A partial write on any of them can corrupt the peak that decides whether to
   liquidate a real-money position.

2. **Round 2 action items not applied**: CA1, HA4, H11, H12, H13, HS1, CD1, HD1, HM1, HM5,
   CS1, H17, H19, C3 from Round 2 are all confirmed still open. The action plan was written
   but the fixes were not committed.

3. **Naked position risk**: Three independent paths can leave a live position without a
   stop-loss: (a) hardware stop fails silently (C3), (b) stop order ID always empty (HA4),
   (c) SwingTrader state lost on crash (ST1).

---

## Findings by Severity

### CRITICAL (15 findings — can lose real money or corrupt state)

| ID | Subsystem | Finding | Conf |
|----|-----------|---------|------|
| C1 | signals-core | **ADX cache still uses `id(df)` — C1 fix incomplete.** LRU eviction was added but the GC-reuse collision remains. Signals may silently use ADX from a different ticker. | 85% |
| C2 | signals-core | **`write_accuracy_cache` unprotected read-modify-write under 8 concurrent threads.** Two threads racing on the same horizon key → last-write-wins → accuracy data silently lost for the losing thread's horizon. Signals run without proper accuracy gating until the next hour-TTL refresh. | 88% |
| C3 | orchestration | **`wait_for_specialists()` blocks the main loop thread for up to 150s synchronously.** Combined with 120s ticker pool timeout = 270s+ per cycle. Directly explains today's 185-476s cycle durations. **Root cause of today's operational degradation.** | 100% |
| C4 | orchestration | **First-of-day T3 morning review is dead code.** `check_triggers()` writes `today_date` every cycle before `classify_tier()` reads it, so the "first invocation = T3" check always evaluates False. No morning full-portfolio review ever fires. | 95% |
| C5 | portfolio-risk | **`should_block_trade()` permanently returns False.** Every guard uses `severity: "warning"`, never `"block"`. The Layer 2 go/no-go gate is permanently open — cooldowns, rate limits, and consecutive-loss escalation are advisory only. | 100% |
| C6 | portfolio-risk | **`check_drawdown()` is never called in the live trading path.** Implemented in `risk_management.py`, called only in tests. Not in `main.py`, `agent_invocation.py`, or any Layer 2 context. The 20% portfolio circuit breaker does not exist at runtime. | 100% |
| C7 | avanza-api | **`get_buying_power()` uses wrong JSON keys.** Reads `categories` (should be `categorizedAccounts`) and `acc["id"]` (should be `acc["accountId"]`). Always returns 0. The af0ed78 fix only patched `metals_avanza_helpers.py`, not `avanza_session.py`. Cash=0 bug lives in production. | 100% |
| C8 | avanza-api | **CA1 still open — CONFIRM executes oldest pending order, not newest.** `pending` list iterated in insertion order; user confirming a new NVDA alert fires a stale SAAB-B order placed minutes earlier. | 95% |
| C9 | data-external | **`earnings_calendar.py` uses wrong config key** (`config["alpha_vantage_key"]` vs `config["alpha_vantage"]["api_key"]`). Every earnings fetch silently gets `api_key=""`, falls through to yfinance. CD1 unfixed. Also bypasses `_daily_budget_used` counter, invisibly burning Alpha Vantage quota. | 95% |
| C10 | infrastructure | **`health.py` read-modify-write race still unfixed.** The H17 fix (`7c78799`) added a timeout to `as_completed()` — it did NOT add a lock to `update_health()`. Up to 8 concurrent threads still race on `load→modify→write` of `health_state.json`. Last-write-wins silently drops signal health updates. | 95% |
| C11 | infrastructure | **`_loading_keys` permanently stuck after `flush_llm_batch()` failure.** `_LOADING_TIMEOUT = 120` defined but never used for eviction. After any LLM batch failure, affected keys are never re-queued; the system permanently returns stale/None LLM data for those tickers. | 90% |
| C12 | metals-core | **`log_portfolio_value()` uses raw `open(HISTORY_FILE, "a")` — not atomic.** This file is the data source for the drawdown circuit breaker. Partial writes silently corrupt peak value. A corrupted peak can mask a real emergency or trigger a false liquidation. Rule 4 violation. | 95% |
| C13 | metals-core | **`_METALS_LOOP_START_TS` initialized at import time** (line 506), not at `main()` entry. If the module is imported before `main()` runs (e.g., test harness, `importlib.reload()`), the session anchor is stale and all three call sites use the wrong timestamp. | 88% |
| C14 | metals-core | **BUY fill stop-loss placement is best-effort — naked positions on failure.** When `HARDWARE_TRAILING_ENABLED=True` and `place_stop_loss()` raises, execution continues, position is live with no stop. Neither the hardware path nor the legacy cascade path fires. `STOP_ORDER_ENABLED=False` means there is no fallback. | 92% |
| C15 | metals-core | **SwingTrader `_save_state` uses raw `open("w")` + `json.dump()`.** On crash mid-write, state file is zero bytes. On restart, `_load_state` returns `_default_state()` with empty positions — all open swing positions and stop-loss IDs silently lost. | 98% |

---

### HIGH (35 findings — silent failures, wrong decisions, open Round 2 issues)

| ID | Subsystem | Finding | Conf |
|----|-----------|---------|------|
| H1 | signals-core | CS1 still open — `per_ticker_consensus` cache keyed without horizon. 3h per-ticker data used for 1d accuracy gating. AMD (24.8%) and GOOGL (31.3%) may escape the gate. | 90% |
| H2 | signals-core | `load_entries()` uses raw `open()` — Rule 4 violation in accuracy backfill. | 82% |
| H3 | signals-core | Accuracy gate fail-open — exception during accuracy load means broken signals (28% accuracy) bypass the gate until next refresh. | 80% |
| H4 | avanza-api | HA4 still open — `StopLossResult.from_api` reads `"stopLossId"` but Avanza returns `"stoplossOrderId"`. Stop ID always `""`. Cannot cancel or track any stop placed via the new package. | 95% |
| H5 | avanza-api | H11 still open — `except ValueError: pass` on corrupt `expires_at` treats expired session as valid. | 90% |
| H6 | avanza-api | H12 still open — Playwright context not recovered on non-401 errors (browser crash, OOM). Dead context reused until restart. | 88% |
| H7 | avanza-api | H13 still open — no account ID whitelist. Pension account 2674244 can be traded by a buggy caller. | 85% |
| H8 | avanza-api | No minimum 1000 SEK order size check anywhere in order placement path. `avanza_session.py`, `avanza/trading.py`, `avanza_orders.py` all accept sub-1000 SEK orders. | 85% |
| H9 | data-external | NewsAPI budget counter increments even on failed fetches. Sustained outage can exhaust 90/day budget without delivering any data. | 90% |
| H10 | data-external | HD1 still open — `NFP_DATES_2026` contains `date(2026, 4, 3)` (Good Friday). BLS released April 2026 NFP on April 2. Signal fires spurious SELL on April 3 when markets are closed. | 100% |
| H11 | data-external | `fear_greed.py` and `golddigger/data_provider.py` call yfinance without `_yfinance_lock`. yfinance is not thread-safe; concurrent calls from 8-worker pool can produce mangled responses or RuntimeError. | 88% |
| H12 | data-external | `onchain_data.py` ignores persistent cache on restart. 6 BGeometrics API calls × 2 daily restarts = 12 of the 15/day budget consumed by restart overhead alone. | 82% |
| H13 | signals-modules | HS1 still open — `structure._highlow_breakout` uses `high.max()` / `low.min()` over full DataFrame history with no lookback cap. All-time high/low bias produces permanent SELL for assets below their historical peak. | 90% |
| H14 | signals-modules | `calendar_seasonal` calls `max(_FOMC_ANNOUNCEMENT_DATES)` on every signal invocation (every 3s across 20 tickers). Post-2027 expiry, floods logs at ~20 warnings/second. | 88% |
| H15 | signals-modules | `smart_money.dropna(how="all")` too permissive — row with NaN close passes through, silently skews BOS/CHoCH detection. Should be `how="any"`. | 85% |
| H16 | signals-modules | `futures_flow` OI history dicts use direct `d["oi"]` — `KeyError` when Binance FAPI omits the field. Silently crashes the OI sub-signals. | 82% |
| H17 | signals-modules | `volume_flow._compute_vwap` is cumulative from bar 0, not session-scoped. For a 200-bar 15-min buffer (50h), "VWAP" is a meaningless 50h average. Generates false BUY/SELL vs. real session VWAP. | 80% |
| H18 | portfolio-risk | Raw `open()` + `json.loads()` in `check_drawdown` — Rule 4 violation on the portfolio history file. | 100% |
| H19 | portfolio-risk | H19 still open — Sortino denominator divides by `len(downside_returns)` instead of `len(all_returns)`. Inflates downside deviation, deflates Sortino ratio. | 92% |
| H20 | portfolio-risk | `compute_portfolio_var` missing `cvar_99_sek` key — any caller reading this key gets `KeyError`. | 100% |
| H21 | portfolio-risk | H7 still open — `_compute_portfolio_value` uses `avg_cost_usd` (entry price) when live price unavailable. Portfolio down 15% reports 0% drawdown during market close. | 95% |
| H22 | orchestration | `_write_fishing_context` uses relative path `'data/fishing_context.json'` instead of `DATA_DIR / 'fishing_context.json'`. Breaks if loop started from any directory other than `BASE_DIR`. | 85% |
| H23 | infrastructure | GPU lock fd leak on `os.write()` failure — CI1 fix not present. Causes 5-minute GPU blockade from a disk write error. | 85% |
| H24 | infrastructure | `_loading_keys` not cleaned for partially-successful LLM batch flushes. Keys with failed inference remain stuck forever, never re-queued. | 82% |
| H25 | infrastructure | `signal_log.jsonl` and `forecast_predictions.jsonl` grow unbounded. `log_rotation.py` was never integrated into `main.py`. `signal_log.jsonl` already at 68MB per health.py comment. | 85% |
| H26 | infrastructure | Telegram `retry_after` ignored — retries use 1s/2s/4s generic backoff but Telegram specifies 30-60s. All 3 retries hit 429, message silently dropped. Affects trade/error notifications. | 82% |
| H27 | metals-core | H1 (raw open) — `metals_risk._load_json_state()` uses `open()` for guard state file. Violates Rule 4; on truncated file, resets all cooldowns and loss counter. | 88% |
| H28 | metals-core | `metals_context.json` written with raw `open()` + `json.dump()`. This is the Layer 2 LLM context file. A crash during write corrupts it — next Claude invocation makes decisions on stale/broken context. | 90% |
| H29 | metals-core | `read_decision_history()` reads entire JSONL with raw `open()` + `readlines()`. Grows unboundedly; reads full file into memory on every call. | 83% |
| H30 | metals-core | Layer 2 journal loaded with raw `open()` + `readlines()` + TOCTOU `os.path.exists()` check. | 85% |
| H31 | metals-core | `POSITIONS` dict accessed by main loop and `_silver_fast_tick()` without lock. Fast-tick iterates `POSITIONS.items()` while main loop may be mutating entries → `RuntimeError: dictionary changed size during iteration`. | 87% |
| H32 | metals-core | HM1 still open — `_silver_reset_session()` defined but never called. After position close→reopen, all alert thresholds are already "fired", new position gets zero fast-tick alerts. | 95% |
| H33 | metals-core | HM5 still open — fish engine tick passes `datetime.now()` (UTC in WSL) as `hour_cet`/`minute_cet`. Market hours gate 1-2h off; engine may block morning trades and attempt post-close trades. | 92% |
| H34 | metals-core | `MIN_TRADE_SEK = 500` in SwingTrader config — violates ≥1000 SEK minimum courtage rule. With `DRY_RUN=False`, orders between 500–999 SEK are placed live, incurring minimum courtage. | 85% |
| H35 | signals-modules | `futures_flow._oi_trend` NaN-unsafe truthiness check (`if price_start and ...`). NaN is truthy in Python, causing silent HOLD-bias in OI trend. | 95% |

---

### MEDIUM (16 findings)

| ID | Subsystem | Finding | Conf |
|----|-----------|---------|------|
| M1 | signals-core | Shared `"time"` key in accuracy cache corrupts TTL checks across concurrent horizon writes. | 80% |
| M2 | avanza-api | `get_buying_power()` fallback always produces 0 (compounds C7). | 80% |
| M3 | avanza-api | TOTP `AvanzaAuth` singleton has no expiry reset path — permanent auth failure after session expires. | 82% |
| M4 | avanza-api | `StopLoss.from_api` reads `"orderBookId"` but API uses nested `orderbook.id` → `orderbook_id=""` always. Stop filtering by instrument always fails. | 80% |
| M5 | signals-modules | `econ_dates.next_event` uses 14:00 UTC placeholder instead of `datetime.now(UTC)` — past-event SELL still fires hours after the event released. | 80% |
| M6 | signals-modules | Non-applicable tickers return `sub_signals: {}` — empty dict breaks any downstream key iteration expecting consistent signal schema. | 80% |
| M7 | signals-modules | US holiday dates in `calendar_seasonal` use fixed `(month, day)` approximations that can be off by up to 7 days. Wrong pre-holiday BUY signal days. | 80% |
| M8 | portfolio-risk | Aware vs naive datetime comparison in position rate-limit path — old state file entries silently skipped, rate limiter bypass. | 82% |
| M9 | portfolio-risk | ATR stop floor at `entry * 0.01` — 99% loss allowed. For BTC that's a $100K → $1K stop, functionally useless. | 95% |
| M10 | orchestration | 3 redundant `load_json(STATE_FILE)` calls per triggered cycle — TOCTOU window between `classify_tier` and `update_tier_state`. | 80% |
| M11 | infrastructure | `fish_monitor_smart.py` uses `read_text()` directly on metals signal JSONL — torn read gives stale LLM predictions. | 80% |
| M12 | metals-core | SwingTrader hardcodes `close_cet = 21.0 + 55/60` — violates memory rule to check `todayClosingTime` from API. Wrong in DST transitions. | 83% |
| M13 | metals-core | `metals_risk._load_json_state()` raw `open()` for guard state — cooldown reset on truncated file. | 82% |
| M14 | metals-core | `compute_daily_range_stats` reads `metals_history.json` with raw `open()`. | 80% |
| M15 | metals-core | `datetime.now()` without timezone at 8+ locations in `metals_loop.py`. UTC/CET date mismatch at 22:00-23:00 UTC corrupts stop order dedup keys. | 80% |
| M16 | metals-core | `_load_positions()` startup reads position state with raw `open()`. Crash-restart race can silently return `{}`, deactivating all held positions. | 82% |

---

### LOW (1 finding)

| ID | Subsystem | Finding | Conf |
|----|-----------|---------|------|
| L1 | metals-core | SwingTrader `_send_telegram` re-reads `config.json` on every message check; raw `open()`; mute flag not cached. | 80% |

---

## Cross-Cutting Themes

### 1. Atomic I/O Compliance Collapse
Rule 4 ("Atomic I/O only") is violated in **15+ locations** despite being a project rule since
day one. Every subsystem except signals-modules has raw `open()` violations. The worst are in
the metals subsystem where these files directly feed the drawdown circuit breaker (HISTORY_FILE)
and the Layer 2 LLM context (metals_context.json). This is not individual carelessness — it
is a systemic failure to enforce the rule during code reviews.

**Recommendation**: Create a pre-commit hook that greps for `open(` + `json.load` patterns
and blocks the commit. All raw reads should be `load_json()` / `load_jsonl_tail()`.

### 2. Round 2 Action Plan Not Executed
Of the 15+ "Immediate" and "Short-Term" action items from the Round 2 plan, **fewer than 8
were applied**. The pattern: fixes documented and committed to the plan, but the actual code
changes never merged. CA1, HA4, H11, H12, H13, H17, HS1, CD1, HD1, HM1, HM5, CS1, H19
are all confirmed still open. **The synthesis docs are being written but not acted upon.**

### 3. Stop Protection Fragility
Three independent paths allow a live real-money position to exist without a stop:
- Hardware stop fails silently → no fallback (C14)
- Stop order ID always `""` → cannot cancel or verify (H4)
- SwingTrader state lost on crash → positions forgotten on restart (C15)

Any of these can result in an unprotected position going to zero.

### 4. Circuit Breaker Architecture Disconnected
The 20% drawdown circuit breaker (`check_drawdown()`) is coded but never wired into the live
trading path. This means Layer 2 can freely trade into a portfolio down 50%.

### 5. Layer 2 Failure (Today)
The 28/28 invocation failures today are caused by **Claude CLI OAuth token expiry** (confirmed
`"OAuth token has expired"` in adversarial_review_out.txt). This requires manual re-auth of
the Claude CLI session — it cannot be fixed in Python. Separately, `wait_for_specialists()`
blocking (C3) explains the 185-476s cycle durations but not the 0% success rate.

---

## Subsystem Risk Ranking (Updated)

| Rank | Subsystem | Risk | Key Concern |
|------|-----------|------|-------------|
| 1 | metals-core | **CRITICAL** | 15+ Rule 4 violations feeding drawdown breaker; naked position risk; SwingTrader state loss |
| 2 | avanza-api | **CRITICAL** | cash=0 bug not fixed; CONFIRM wrong order; stop ID always empty; no account whitelist |
| 3 | portfolio-risk | **CRITICAL** | Circuit breaker disconnected; trade guard permanently open |
| 4 | orchestration | **CRITICAL** | wait_for_specialists blocks main loop 150s; first-of-day T3 dead |
| 5 | signals-core | **HIGH** | ADX cache fix incomplete; accuracy cache race; gate fail-open |
| 6 | infrastructure | **HIGH** | health.py race confirmed unfixed; _loading_keys stuck; 68MB unbounded log |
| 7 | data-external | **HIGH** | earnings config broken; NFP Good Friday; yfinance lock bypass |
| 8 | signals-modules | **MEDIUM** | VWAP bias; structure all-history; NaN in futures_flow |

---

## Recommended Action Plan

### Tier 1: Fix NOW (trivial, high impact, no risk of regression)

1. **MIN_TRADE_SEK 500→1000** (C15/SC1) — `data/metals_swing_config.py:59`, 1 line. Live orders.
2. **NFP Good Friday** (H10) — `portfolio/econ_dates.py:61`, 1 line. April 3 is imminent.
3. **_silver_reset_session call** (H32) — `data/metals_loop.py:6063`, 1 line.
4. **metals_context.json atomic write** (H28) — `data/metals_loop.py:5078`, 1 line.
5. **SwingTrader atomic state** (C15) — `data/metals_swing_trader.py:110-115`, 5 lines.
6. **get_buying_power JSON keys** (C7) — `portfolio/avanza_session.py:302-304`, 2 lines.
7. **stop ID key** (H4) — `portfolio/avanza/types.py:212`, 1 line.
8. **cvar_99_sek missing key** (H20) — `portfolio/monte_carlo_risk.py:496-509`, 1 line.

### Tier 2: Fix This Week (important, medium effort)

9. **health.py lock** (C10) — Add `threading.Lock()` to `update_health`, `update_signal_health_batch`.
10. **write_accuracy_cache lock** (C2) — Add module-level lock in `accuracy_stats.py`.
11. **Naked position fallback** (C14) — On hardware stop fail, attempt cascade stop or send CRITICAL Telegram.
12. **per_ticker_consensus horizon key** (H1) — `signal_engine.py:1610-1613`, 2 lines.
13. **_loading_keys stuck eviction** (C11) — Use `_LOADING_TIMEOUT` already defined, add eviction.
14. **Telegram retry_after** (H26) — Parse `retry_after` from 429 body in `http_retry.py`.
15. **fish engine CET fix** (H33) — Replace `datetime.now()` with `datetime.now(_STOCKHOLM_TZ)`.
16. **_load_positions atomic** (M16) — Replace startup `open()` with `load_json()`.
17. **log_portfolio_value atomic** (C12) — Replace `open("a")` with `atomic_append_jsonl()`.

### Tier 3: Fix This Month

18. **check_drawdown() wire-up** (C6) — Call in `main.py`/`reporting.py`, surface `breached` to Layer 2.
19. **should_block_trade severity** (C5) — Add `"block"` severity to ticker cooldown and rate limit guards.
20. **wait_for_specialists async** (C3) — Decouple from main loop thread.
21. **First-of-day T3** (C4) — Pass state from `check_triggers` directly to `classify_tier`.
22. **CA1 CONFIRM ordering** (C8) — Sort pending by timestamp descending before confirming.
23. **ADX cache content hash** (C1) — Replace `id(df)` with structural key.
24. **signal_log.jsonl rotation** (H25) — Integrate `log_rotation.py` into `main.py` post-cycle loop.
25. **earnings_calendar config key** (C9) — `config.get("alpha_vantage", {}).get("api_key", "")`.
26. **Batch remaining Rule 4 violations** — `metals_risk._load_json_state`, `risk_management.check_drawdown`, `read_decision_history`, journal reads (H27-H30, H18).

---

## Round 2 → Round 3 Delta

| Status | Count |
|--------|-------|
| Fixed since Round 2 | ~8 (CO2, H4, H5, M3, M4, H20-lock, H18-lock, M6-Monte Carlo) |
| Still open from Round 2 | ~22 |
| New findings in Round 3 | ~35 |
| **Total findings Round 3** | **67** |

Fixed: CO2 (zombie threads), H4 (agent kill), H5 (stack overflow counter), M3 (set
serialization), M4 (DST transition), H20-lock (`save_state` now uses lock), M6 (10K Monte
Carlo paths), Sortino partial (comment added), Drawdown session-relative (new this session).

---

## Methodology Notes

- **Agents**: 8 parallel `feature-dev:code-reviewer` subagents + 1 independent Claude direct review
- **Total tokens used**: ~780K across all agents
- **Confidence threshold**: All findings ≥75%. Speculative findings excluded.
- **Limitations**: Static analysis only. Race conditions may be harder to trigger in practice
  than on paper. Some findings depend on WSL2 timezone assumptions.
- **Layer 2 failure today**: OAuth expiry — requires manual `claude auth login` re-auth, not
  a code fix.
