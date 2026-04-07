# Adversarial Review #2 — Synthesis
**Date:** 2026-04-07
**Baseline HEAD:** `ad13eca` (main branch)
**Reviewers:** 8 parallel Claude Opus 4.6 agents + 1 independent Claude Opus 4.6 pass
**Method:** 8 subsystem-specific adversarial agents (background) + independent manual review
**Prior review:** `docs/ADVERSARIAL_REVIEW_SYNTHESIS.md` (2026-04-05, 85 findings)

---

## Delta Analysis: Prior Review C1-C10 Status

| ID | Finding | Status | Evidence |
|----|---------|--------|----------|
| C1 | Self-heal grants Edit+Bash+Write | **FIXED** | `allowed_tools="Read,Grep,Glob"` (read-only) |
| C2 | `place_order_no_page` fails open | **FIXED** | Strict validation added (line 323) |
| C3 | No position-size / stop volume invariant | **STILL OPEN** | No `free_volume` check in any order path |
| C4 | `record_trade()` has zero callers | **STILL OPEN** | Only golddigger's `record_trade_pnl()` exists |
| C5 | Singleton lock no-ops on non-Windows | **FIXED** | Now supports both msvcrt and fcntl |
| C6 | MWU signal weights written, never read | **STILL OPEN** | signal_engine.py never imports SignalWeightManager |
| C7 | `load_state` silently regenerates defaults | **FIXED** | Rolling backups via `_rotate_backups()` |
| C8 | Portfolio state no concurrency safety | **FIXED** | `update_state()` with per-file locks |
| C9 | Monte Carlo t-copula identity transform | **FIXED** | Uses `norm.ppf(U)` now |
| C10 | Regime-gated signals can't recover | **FIXED** | `raw_votes` captured pre-gate (line 1356) |

**Score:** 6/10 fixed in 2 days. C3, C4, C6 remain open — all with real financial impact.

---

## NEW FINDINGS — Independent Review (April 7)

### TIER 1: CRITICAL (Act Now)

#### N1. `record_trade()` STILL has zero callers — overtrading guards completely non-functional
**Source:** Confirmed re-verification of C4
**Files:** `portfolio/trade_guards.py:171`
**What:** grep for `record_trade(` across the entire portfolio directory returns only
the definition and the C4 warning. Not a single BUY/SELL execution path calls it.
**Impact:** The entire overtrading-guard subsystem (cooldowns, loss escalation, position
rate limits) is dead code. The system has NO protection against overtrading.
**Priority:** HIGHEST — this is a repeat finding from April 5 that was not fixed.

#### N2. `atomic_append_jsonl` is NOT atomic — concurrent appenders can corrupt JSONL
**Files:** `portfolio/file_utils.py:155-167`
**What:** `open(path, "a")` + `write()` + `fsync()` is not protected by any file lock.
When main loop and metals loop both append to the same JSONL file (e.g., signal_log.jsonl,
telegram_messages.jsonl), their writes can interleave, producing partial JSON lines.
**Impact:** Corrupted JSONL entries → accuracy tracking becomes unreliable, signal
postmortem analysis produces wrong results, telegram log analysis fails.
**Evidence:** `atomic_write_json` uses tempfile+replace (truly atomic). `atomic_append_jsonl`
does NOT — it's a misnomer.

#### N3. Two parallel signal systems (main loop vs metals loop) diverge silently
**Files:** `portfolio/signal_engine.py` vs `data/metals_loop.py`
**What:** The metals loop (5366 lines) has its OWN signal computation that doesn't use
signal_engine.py. All improvements made to signal_engine (accuracy gating, regime gating,
correlation groups, horizon-specific weights, crisis mode, per-ticker consensus gate) are
NOT applied to metals signals. The metals loop trades REAL warrants with REAL money.
**Impact:** Metals trades operate with an inferior, un-gated signal system while the main
loop benefits from 30+ bug fixes and tuning iterations.

#### N4. No account ID validation on Avanza order placement
**Files:** `portfolio/avanza_session.py:371-379`
**What:** `_place_order` defaults to ISK account 1625505 but accepts any `account_id`
parameter without validation. No allowlist prevents orders on pension account 2674244.
**Impact:** A parameter error could route a live order to the pension account.

#### N22. `check_drawdown()` is NEVER called — drawdown circuit breaker is dead code
**Source:** Agent review (portfolio-risk), verified against codebase
**Files:** `portfolio/risk_management.py:53`
**What:** `check_drawdown()` is the max drawdown circuit breaker (20% threshold). Grep
reveals zero production callers — only test files call it. The main loop, agent_invocation,
and autonomous.py never check drawdown before allowing trades.
**Impact:** A catastrophic loss sequence will never be stopped. Combined with N1 (no
overtrading guards), the system has ZERO automated risk protection in production.

#### N23. Sortino ratio uses wrong denominator — underestimated by ~1.58x
**Source:** Agent review (portfolio-risk)
**Files:** `portfolio/equity_curve.py:246`
**What:** Sortino downside deviation divides by `len(downside_returns)` instead of
`len(all_returns)`. The standard formula includes zero contributions from positive days
in the total count. With 60% positive days, error factor is sqrt(1/0.4) = 1.58x.
**Impact:** Risk-adjusted performance metrics are systematically wrong.

#### N24. Circuit breaker HALF_OPEN can get permanently stuck
**Source:** Agent review (portfolio-risk)
**Files:** `portfolio/circuit_breaker.py:83-93`
**What:** If a HALF_OPEN probe crashes before recording success/failure, the
`_half_open_probe_sent` flag stays True forever. All subsequent requests are blocked.
**Impact:** A data source (Binance, Alpaca) could become permanently blocked until restart.

### TIER 2: HIGH (Fix This Sprint)

#### N5. `check_drawdown` scans full history file every call — O(n) grows unbounded
**Files:** `portfolio/risk_management.py:97-110`
**What:** Reads `portfolio_value_history.jsonl` line by line. After 1 year: 500K+ entries.
Uses raw `open()` + `json.loads()`, not `load_jsonl_tail()`.
**Impact:** Increasingly slow drawdown checks; eventually causes cycle timeout.

#### N6. `_compute_portfolio_value` falls back to entry price when live price unavailable
**Files:** `portfolio/risk_management.py:46`
**What:** Uses `avg_cost_usd` when ticker not in agent_summary. For a stock down 50%,
this overstates portfolio value by 2x.
**Impact:** Masks drawdown, prevents circuit breaker from triggering.

#### N7. ~~Accuracy data mutation may corrupt shared cache~~ — DOWNGRADED to LOW
**Files:** `portfolio/signal_engine.py:1518`
**What:** Per-ticker accuracy overrides write directly into `accuracy_data`. HOWEVER,
`blend_accuracy_data()` returns a new dict (verified: line 551 creates fresh dict),
and `generate_signal()` is called once per ticker with its own copy. The regime
accuracy overlay (line 1505) assigns shared references but utility boost (line 1555)
creates copies via `{**accuracy_data[sig_name], ...}`.
**Assessment:** Not a real bug in current code. Each ticker gets its own accuracy_data.
Keeping as LOW for documentation: future refactoring that shares accuracy_data across
tickers would introduce this bug.

#### N8. `_adx_cache` keyed by `id(df)` — stale values after GC
**Files:** `portfolio/signal_engine.py:25`
**What:** `id(df)` returns memory address. After DataFrame GC, new DataFrame may get
same id(), serving stale ADX from a different ticker.
**Impact:** Incorrect ADX values → wrong regime detection → wrong signal gating.

#### N9. metals_loop.py imports Playwright at module level
**Files:** `data/metals_loop.py:54`
**What:** Top-level `from playwright.sync_api import sync_playwright` — if Playwright
is broken, the entire metals loop fails to start.
**Impact:** Complete metals loop failure on Playwright issues.

#### N10. `health.py` read-modify-write has no locking
**Files:** `portfolio/health.py:16-36`
**What:** `update_health()` and `update_signal_health_batch()` both do load→modify→write
without any locking. Concurrent calls clobber each other.
**Impact:** Health data loss, inaccurate monitoring.

#### N11. Avanza `check_pending_orders` confirms FIFO, not by ID
**Files:** `portfolio/avanza_orders.py:127-132`
**What:** CONFIRM matches first pending order, not the intended one. Two pending orders
means CONFIRM could execute the wrong one.
**Impact:** Wrong order confirmed with real money.

#### N34. Emergency sell uses stale bid — position rides to knockout
**Source:** Agent review (metals-core), finding RISK-1 — MOST DANGEROUS BUG IN REVIEW
**Files:** `data/metals_loop.py:2636-2680`
**What:** `emergency_sell()` receives `bid` price from calling code, fetched seconds earlier.
By the time the sell executes (CSRF fetch + API call = 2-5 seconds), a crashing market has
already moved. The LIMIT sell at stale bid won't fill because current bid is now below. The
position remains open while crashing toward the knockout barrier.
**Impact:** Total position loss (knockout) in a scenario specifically designed to prevent it.
**Fix:** Re-fetch bid immediately before placing sell, OR use market order for emergencies.

#### N35. Chronos drift annualized 16x too high — fishing decisions dominated by error
**Source:** Agent review (metals-core), finding MATH-2
**Files:** `portfolio/price_targets.py:511` vs `portfolio/fin_fish.py:367`
**What:** `price_targets.py` uses `sqrt(252)` (~15.87x) to annualize daily drift.
`fin_fish.py` uses `252` (linear). Drift should be linear; sqrt is for volatility.
One is wrong by a factor of 16x.
**Impact:** Fishing system directional bias completely dominated by this 16x error.

#### N36. No Binance staleness tracking — all decisions on frozen prices
**Source:** Agent review (metals-core), finding REL-3
**Files:** `data/metals_loop.py:1102-1148`
**What:** Binance FAPI is the sole source for XAG/XAU prices. If Binance goes down,
`_underlying_prices` retains stale values with no age check. All trailing stops,
emergency sells, and fish engine decisions operate on frozen data.
**Impact:** System blind during Binance outages — exactly when volatility spikes.

#### N28. `metals_avanza_helpers.place_order()` has ZERO input validation
**Source:** Agent review (avanza-api), finding F1
**Files:** `data/metals_avanza_helpers.py:86-133`
**What:** Zero validation on price, volume, or side. Accepts any values and sends directly
to Avanza's live order API. The metals loop uses this unvalidated function via
`avanza_control.place_order()`. Compare with `avanza_session._place_order()` which
validates `volume >= 1` and `price > 0`.
**Impact:** Malformed order sent to live brokerage. Worst case: Avanza interprets
unexpectedly, creating unintended positions with real money.

#### N29. `place_stop_loss()` has no 3% bid guard at API layer
**Source:** Agent review (avanza-api), finding F2
**Files:** `data/metals_avanza_helpers.py:136-194`
**What:** The CLAUDE.md rule "NEVER place stop within 3% of bid" is only enforced in
golddigger. The underlying `place_stop_loss()` function has no guard. Any caller
(fin_snipe_manager, metals_loop) can place stop-losses that immediately trigger on
normal market noise.
**Impact:** Unnecessary forced sale at loss on volatile silver warrants.

#### N30. Playwright context not thread-safe — CSRF corruption risk
**Source:** Agent review (avanza-api), finding F5
**Files:** `portfolio/avanza_session.py:33-37, 180-256`
**What:** `_pw_lock` only guards Playwright context creation, not API calls. `api_get()`,
`api_post()`, `api_delete()` all use shared context without locking. Playwright contexts
are explicitly not thread-safe per documentation.
**Impact:** Concurrent API calls can corrupt CSRF tokens, mix up responses, or cause one
request to see another's response. Buy order response could be interpreted as stop-loss
response.

#### N31. `/mode` Telegram command destroys config.json symlink — TICKING TIME BOMB
**Source:** Agent review (infrastructure), finding 8.2
**Files:** `portfolio/telegram_poller.py:150-160`, `portfolio/file_utils.py:27`
**What:** The `/mode` command reads config.json, modifies it, writes it back via
`atomic_write_json`. But config.json is a symlink. `os.replace(tmp, str(path))` on a
symlink replaces the SYMLINK ITSELF, not the target file. After one `/mode` command,
config.json becomes a regular file and the external config at
`C:\Users\Herc2\.config\finance-analyzer\config.json` is orphaned.
**Impact:** One Telegram command permanently breaks the config setup. The external config
(which contains API keys and is the canonical source) is disconnected.
**Fix:** Resolve symlink before writing: `path = Path(path).resolve()`.

#### N32. Telegram bot token leaked in retry log messages
**Source:** Agent review (infrastructure), finding 3.1
**Files:** `portfolio/http_retry.py:43-44, 47-48, 56-57, 60-61`
**What:** `fetch_with_retry` logs the full URL on retry/failure. When called from
`telegram_notifications.py`, the URL includes `bot<TOKEN>`. Logged at WARNING level.
**Impact:** Anyone with log access can impersonate the Telegram bot.
**Fix:** Redact URLs containing `api.telegram.org/bot` before logging.

#### N33. Health monitor reports "healthy" when system is broken
**Source:** Agent review (infrastructure), finding 7.1
**Files:** `portfolio/health.py:215-245`
**What:** `status` is "healthy" if heartbeat is fresh (<300s). But heartbeat updates at
end of every cycle REGARDLESS of signal success. System can report healthy while all 32
signals fail, Telegram is down, and agent is silent for days.
**Impact:** Dashboard shows green while the system is effectively dead. User trusts status
indicator and doesn't investigate.

#### N25. SQLite accuracy queries skip neutral outcome filter — accuracy inflation
**Source:** Agent review (signals-core), finding B6
**Files:** `portfolio/signal_db.py:271,302,330,370`
**What:** All accuracy SQL queries in SignalDB count outcomes where `change_pct` is tiny
(e.g., 0.01%) as correct for BUY signals (since 0.01 > 0). The JSONL-based accuracy in
`accuracy_stats.py` applies `_MIN_CHANGE_PCT = 0.05` to filter neutral outcomes. Since
SQLite is the preferred data source (`load_entries()` line 27-35), ALL accuracy numbers
used for signal weighting may be systematically inflated.
**Impact:** Overconfident signals → larger position sizes → increased financial risk.

#### N26. No `change_pct` outlier validation — single corrupt price permanently skews accuracy
**Source:** Agent review (signals-core), finding D1
**Files:** `portfolio/outcome_tracker.py:374-378`
**What:** No validation that `change_pct` is within reasonable bounds. A Binance flash-crash
or yfinance split-adjusted price would produce extreme values (e.g., +10000%). These outliers
are permanently stored and used in all accuracy calculations.
**Impact:** One corrupt price event permanently classifies a signal as "correct" or "incorrect",
distorting all downstream trading decisions.

#### N27. NaN propagation in weighted consensus — NaN confidence in trade pipeline
**Source:** Agent review (signals-core), finding E2
**Files:** `portfolio/signal_engine.py:637-692`
**What:** If accuracy_data returns 0/0=NaN for a signal, the weight becomes NaN. All
subsequent `buy_weight += NaN` produces NaN. Final confidence is NaN, which propagates
through penalty cascade into trade decisions.
**Impact:** Entire consensus computation corrupted by one signal's NaN accuracy.
**Fix:** Add `if not np.isfinite(weight): continue` after weight computation.

#### N21. `_loading_keys` leak in `_cached_or_enqueue` — LLM signals permanently stale
**Files:** `portfolio/shared_state.py:108-130`, `portfolio/llm_batch.py:65-100`
**What:** `_cached_or_enqueue` adds keys to `_loading_keys` (line 123) to prevent
re-enqueue. Keys are only removed by `_update_cache()` (line 136). If `flush_llm_batch()`
fails for a key (LLM inference error, import failure), that key stays in `_loading_keys`
permanently. On every subsequent cycle, the key is never re-enqueued because
`key not in _loading_keys` is False. The stale LLM result is served until it exceeds
`_MAX_STALE_FACTOR` (3x TTL = 45 minutes), then None is returned forever.
**Impact:** A single LLM inference failure permanently disables that ticker's Ministral/Qwen3
signal for the session. Only a loop restart recovers it.
**Fix:** Clean `_loading_keys` in `flush_llm_batch()` after processing, OR add a timeout
to `_loading_keys` entries.

### TIER 3: MEDIUM (Fix Next Sprint)

#### N12. Post-cycle tasks run synchronously — cycle time bloat
**Files:** `portfolio/main.py:262-371`

#### N13. `econ_calendar.py` uses hardcoded FOMC dates (2026-2027 only)
**Files:** `portfolio/signals/econ_calendar.py`

#### N14. Trade guards use `severity: "warning"` — never "block"
**Files:** `portfolio/trade_guards.py`

#### N15. `sentiment.py` silent degradation on NewsAPI exhaustion
**Files:** `portfolio/sentiment.py`

#### N16. `fx_rates.py` minimal error handling for critical FX data
**Files:** `portfolio/fx_rates.py`

#### N17. Rate limiters use `time.sleep()` — blocks ThreadPoolExecutor workers
**Files:** `portfolio/shared_state.py:167-174`

#### N18. `_GROUP_LEADER_GATE_THRESHOLD` hardcoded inside function body
**Files:** `portfolio/signal_engine.py:596`

#### N19. Signal module interface inconsistency
**Files:** `portfolio/signals/*.py` (23 modules, 3 calling conventions)

#### N20. `prune_jsonl` reads entire file into memory
**Files:** `portfolio/file_utils.py:232-276`

---

## Cross-Cutting Architectural Concerns

### A1. Two Signal Systems (CRITICAL)
The main loop and metals loop have independent signal computation. This is the single
biggest architectural risk — every signal engine improvement since February only benefits
the main loop. The metals loop, which trades REAL money, operates with an older, untuned
signal system.

### A2. ThreadPoolExecutor(8) + Module-Level Mutable State
Multiple modules use module-level dicts protected by individual locks. The overall
concurrency model is hard to reason about. Risk: deadlock between any two locks
freezes the main loop forever.

### A3. JSONL Append-Only Logs Without File Locking
10+ JSONL files use `atomic_append_jsonl` which is neither truly atomic nor locked.
Multiple processes (main loop, metals loop, outcome tracker) write to overlapping files.

### A4. Config Symlink Risk
`config.json` is a symlink to an external file. Mid-edit partial JSON could crash the loop.

---

## Recommendations (Priority Order)

1. **Wire `record_trade()` into all execution paths** (C4/N1 — day 1)
2. **Add file locking to `atomic_append_jsonl`** (N2 — day 1)
3. **Validate account_id against allowlist** (N4 — day 1, 5 lines of code)
4. **Refactor metals_loop to use signal_engine.py** (N3/A1 — week-long project)
5. **Track peak_value in health_state.json** (N5 — day 1)
6. **Add pre-trade volume invariant** (C3 — day 2)
7. **Delete or wire SignalWeightManager** (C6 — day 1)
8. **Deep-copy accuracy_data before mutation** (N7 — day 1)
9. **Add locking to health.py read-modify-write** (N10 — day 1)
10. **Fix Avanza order confirmation to use order ID** (N11 — day 1)

---

## Agent Review Results

8 parallel adversarial review agents were launched (one per subsystem). Results are
appended as they complete.

### Agent 3: portfolio-risk (COMPLETED — 20 findings)

**NEW CRITICAL finding not in independent review:**
- **`check_drawdown()` is NEVER called in production** — the 20% max drawdown circuit
  breaker exists as code but is dead. No production path invokes it. The system will
  trade through a 50%+ drawdown with no protection. (risk_management.py:53)

**NEW HIGH findings:**
- **Sortino ratio denominator bug** — uses `len(downside_returns)` instead of
  `len(all_returns)`, systematically underestimating Sortino by ~1.58x. (equity_curve.py:246)
- **Circuit breaker HALF_OPEN stuck state** — if a probe request crashes before recording
  its outcome, the breaker stays in HALF_OPEN forever, permanently blocking the data
  source. (circuit_breaker.py:83-93)
- **ATR proximity check uses magic string "CHECK"** — works by accident but a refactor
  could silently break held-position monitoring. (risk_management.py:769)
- **Monte Carlo volatility estimation assumes fixed candle frequency** — if ATR is from
  4h candles, volatility is overestimated by 2x. (monte_carlo.py:39-57)

**Converged with independent review (confirmed):**
- record_trade() zero callers (N1/C4)
- check_drawdown O(n) scan (N5)
- Fallback to entry price (N6)
- Concentration check cash-based (N12 in independent)
- Trade guards no file locking (N10 partial)

### Agent 1: signals-core (COMPLETED — 22 findings)

**NEW findings not in independent review:**
- **SQLite accuracy skips neutral filter** — `signal_db.py` queries don't apply
  `_MIN_CHANGE_PCT=0.05`, inflating accuracy vs JSONL computation. Since SQLite is
  preferred data source, ALL accuracy numbers may be systematically wrong. (B6)
- **No `change_pct` validation** — a single corrupt Binance flash-crash price permanently
  skews accuracy for that signal forever. No outlier detection. (D1)
- **NaN propagation in weighted consensus** — if accuracy is 0/0=NaN, weight becomes NaN,
  all subsequent additions produce NaN, confidence becomes NaN in trade pipeline. (E2)
- **`generate_signal` crashes on missing indicator keys** — direct `ind["rsi"]` access
  without `.get()` defaults. Any data quality issue → total signal failure for ticker. (B2)
- **`signal_history.py` race condition** — read-modify-write without locking in
  ThreadPoolExecutor. History entries silently lost. (R1)
- **Utility boost can inflate accuracy by 1.5x** — signals catching a few lucky outliers
  get dramatically overweighted. (C2)

**Converged with independent review:**
- ADX cache keyed by id(df) (B1 = N8)
- Accuracy→gating feedback loop (D3 ≈ C10)
- Signal weight optimizer dead code (A3 ≈ C6)

### Agent 5: avanza-api (COMPLETED — 20 findings, 2 CRITICAL)

**NEW CRITICAL findings — highest financial risk in the review:**
- **`metals_avanza_helpers.place_order()` has ZERO validation** — price, volume, side
  all unvalidated before hitting Avanza's live API. Contrast with `avanza_session._place_order()`
  which validates both. The metals loop uses this unvalidated path. (F1)
- **`place_stop_loss()` has no 3% bid guard at the API layer** — the CLAUDE.md rule
  "NEVER place stop within 3% of bid" is only enforced in golddigger, not in the
  underlying helper function. Any other caller (fin_snipe_manager, metals_loop) can
  violate the rule. (F2)

**NEW HIGH findings:**
- **Playwright context not thread-safe** — `_pw_lock` only guards creation, not API calls.
  Concurrent requests corrupt CSRF tokens or mix up responses. (F5)
- **`get_buying_power()` fallback mixes pension + ISK data** — when account not found,
  uses first category's total (could be pension) minus ISK positions. (F4)
- **`get_positions()` returns ALL accounts** — no ISK filter by default, pension
  positions included in portfolio calculations. (F3)
- **No trading hours validation on ANY order function** — orders outside hours silently
  proceed to API, may queue and fill at gap prices. (F7)
- **`EXPIRY_BUFFER_MINUTES = 30` defined but never enforced** — session can expire
  mid-order-placement. (F6)
- **`get_account_id()` returns first ISK found** — may be wrong if multiple ISK accounts. (F8)

**Converged with independent review:**
- No account ID validation (N4 = F3 partial)
- FIFO order confirmation (N11 = F11)
- api_delete treats 404 as success (noted in independent review)

### Agent 8: infrastructure (COMPLETED — 22 findings, 0 CRITICAL but 3 HIGH)

**NEW findings not in independent review:**
- **`/mode` Telegram command destroys config.json symlink** — `atomic_write_json` on a
  symlink replaces the symlink itself via `os.replace`, not the target file. One `/mode`
  command permanently breaks external config. TICKING TIME BOMB. (8.2)
- **Telegram bot token logged in retry warnings** — `fetch_with_retry` logs full URL
  including `bot<TOKEN>` at WARNING level. Anyone with log access gets the token. (3.1)
- **Health reports "healthy" when system is broken** — heartbeat updates regardless of
  signal success. Dashboard shows green while all 32 signals fail, Telegram is down,
  and agent is silent for days. (7.1)
- **`weekly_digest.py` reads entire 68MB signal_log.jsonl** — OOM risk on constrained
  systems. Only file that reads the full log for weekly data. (5.3)
- **`claude_gate._clean_env()` copies all env vars to subprocess** — API keys in
  environment accessible to Claude subprocess with Edit+Bash+Write tools. (3.2)

**Converged with independent review:**
- atomic_append_jsonl not truly atomic (1.1 = N2)
- health.py read-modify-write race (2.3 = N10)
- _loading_keys leak (2.2 = N21)
- prune_jsonl reads entire file (8.3 = N20)

### Agent 4: metals-core (COMPLETED — 26 findings, 1 CRITICAL)

**The most financially dangerous bug in the entire review:**
- **Emergency sell uses stale bid price** — during a crash, `emergency_sell()` places a
  LIMIT sell at a bid fetched seconds earlier. In a fast crash toward knockout, the bid
  has already moved below. The order WON'T FILL and the position rides to total loss
  (knockout). Should use market order or re-fetch bid immediately before placing. (RISK-1)

**NEW HIGH/MEDIUM findings:**
- **Chronos drift annualized 16x too high in fin_fish.py** — `price_targets.py` uses
  `sqrt(252)` while `fin_fish.py` uses `252`. One is wrong by 16x, dominating all
  fishing decisions. Drift should be linear (252), volatility uses sqrt. (MATH-2)
- **No Binance staleness tracking** — if Binance FAPI goes down, prices silently freeze.
  All decisions (stops, emergency sells) operate on stale data with no age check. (REL-3)
- **Warrant P&L uses linear approximation** — ignores non-linear leverage near financing
  level, underestimating losses by 3-6% at the worst possible moment. (MATH-4)
- **ORB predictor hardcodes UTC winter time** — morning window wrong during CEST summer.
  ORB predictions systematically wrong half the year. (STALE-2)
- **Peak bids lost on restart** — trailing stop jumps to lower level after crash/restart,
  giving back significant profits. (STATE-1)
- **`page.evaluate` no timeout on emergency sell** — Avanza server hang = total position
  loss as price crashes through knockout barrier. (EDGE-3)
- **Stop distance 3% too close for 5x warrants** — uses `< 3.0` not `<= 3.0`, and 3%
  underlying = 15% warrant move, within normal silver noise. (RISK-2)
- **Snipe manager HARD_STOP_CERT_PCT = 5%** — at 5x leverage, 5% cert = 1% underlying.
  Will trigger constantly on normal fluctuations. (RISK-3)

**Converged with independent review:**
- God module (5366 lines) — N/A in independent but noted
- Non-atomic I/O in metals_loop — partial convergence with N2

**Agent subsystem assignments (remaining in progress):**
2. `review-orchestration` — main.py + 8 supporting files
6. `review-signals-modules` — 23 signal module files
7. `review-data-external` — data_collector.py + 13 supporting files

---

## Verification Methodology

Each finding was verified by:
1. Reading the source file at the specific line number
2. Grepping for callers/consumers of the identified function
3. Cross-referencing against the prior April 5 review
4. Checking git log for any fixes applied since April 5
5. Assessing financial impact based on the system's stated purpose (real money trading)
