# Adversarial Review Synthesis — Finance Analyzer

**Date**: 2026-04-10
**Methodology**: Dual review — independent manual analysis + 8 parallel agent reviews
**Scope**: Full codebase (135 files, ~60,377 lines) across 8 subsystems
**Commit baseline**: 9804a55 (main)

---

## Executive Summary

This adversarial review examined the entire finance-analyzer codebase across 8 subsystems:
signals-core (5.6K lines), orchestration (6.4K), portfolio-risk (4.3K), metals-core (19K),
avanza-api (2.3K), signals-modules (10.9K), data-external (6.1K), infrastructure (5.7K).

**Overall assessment**: The codebase shows strong defensive engineering — 28 thread locks,
atomic I/O patterns, fail-closed accuracy gating, circuit breakers for APIs, and thorough
crash recovery. The system has been significantly hardened through iterative bug fixes (BUG-85
through BUG-181). The main areas of concern are:

1. **[P0] Playwright context thread-unsafe** — Concurrent API calls corrupt trade responses [NEW from avanza-api agent]
2. **[P0] TOTP path has no account whitelist** — Pension account 2674244 can receive orders [NEW from avanza-api agent]
3. **[P1] Directional accuracy gate silently disabled** — Per-ticker accuracy override drops buy/sell fields [from signals-core agent]
4. **[P1] Subprocess governance** — 3 modules bypass the centralized claude_gate
5. **[P1] Stop-loss safety** — Hardware trailing stop failure has no automatic fallback
6. **[P1] Portfolio drawdown blind spot** — Fallback to stale prices masks real drawdowns
7. **[P2] Timezone consistency** — 8 naive datetime.now() calls in metals_loop
8. **[P2] Windows-specific atomicity** — JSONL append lacks true atomicity on NTFS

**CRITICAL UPDATE**: 2 P0 (critical/money-losing) findings identified by the avanza-api
agent review. The Playwright context thread-safety issue can cause corrupt trade responses
when concurrent API calls are made. The TOTP account whitelist gap can route orders to the
pension account. Both require IMMEDIATE remediation.

**Agent review update**: 2 of 8 agent reviews complete (signals-core: 10 findings,
avanza-api: 13 findings). The avanza-api review is the most alarming — it found the only
P0 issues in the entire codebase.

---

## Subsystem Health Scorecard

| Subsystem | Lines | Findings | P1 | P2 | P3 | Health |
|-----------|-------|----------|----|----|----|----|
| signals-core | 5,640 | 17 | 4 | 9 | 4 | Fair — directional gate bypassed |
| orchestration | 6,412 | 16 | 3 | 9 | 4 | Fair — TimeoutError bug, stale config |
| portfolio-risk | 4,281 | 4 | 1 | 2 | 1 | Fair — drawdown blind spot |
| metals-core | 19,014 | 6 | 1 | 3 | 2 | Fair — God file, timezone issues |
| avanza-api | 2,298 | 16 | 2 | 10 | 4 | **CRITICAL** — 2 P0 findings |
| signals-modules | 10,949 | 3 | 0 | 1 | 2 | Good — consistent pattern |
| data-external | 6,062 | 12 | 5 | 7 | 0 | Fair — budget drain, yfinance compat |
| infrastructure | 5,721 | 8 | 2 | 3 | 3 | Fair — atomicity + gate bypass |
| **Total** | **60,377** | **82+** | **18+** | **44+** | **20+** | |

---

## Top Priority Findings (Recommended Fixes)

### EMERGENCY: Fix Playwright context thread safety [P0, avanza_session.py]

**File**: `portfolio/avanza_session.py:183-291`

**Problem**: `_pw_lock` is only held during Playwright context *creation*. All API calls
(`api_get`, `api_post`, `api_delete`) use `ctx.request.*` WITHOUT holding the lock.
Playwright's sync_api is NOT thread-safe. The metals loop's 10s fast-tick thread and
the main loop's 8-worker ThreadPoolExecutor can make concurrent requests, corrupting
internal HTTP state. A BUY confirmation could be consumed by a different request.

**Impact**: Corrupt trade responses. Order failures, double-executions, wrong-asset trades.

**Fix**: Wrap the entire body of each `api_get/api_post/api_delete` function with
`with _pw_lock:`. This serializes all Playwright API calls.

**Effort**: 15 minutes. CRITICAL — must be done before next trading session.

---

### EMERGENCY: Add account whitelist to TOTP path [P0, avanza_client.py]

**File**: `portfolio/avanza_client.py:220-320`

**Problem**: `get_account_id()` discovers the ISK account by scanning for "ISK" in
accountType. There is NO hardcoded allowlist against "1625505". If Avanza re-orders
accounts in the API response, pension account 2674244 could receive real-money trades.
The `avanza_session.py` path has `ALLOWED_ACCOUNT_IDS`, but the TOTP fallback path does not.

**Impact**: Trades on pension account. Regulatory risk, tax implications, wrong risk profile.

**Fix**: Add `assert account_id == "1625505"` in `get_account_id()` before caching.
Also filter `get_positions()` and `get_portfolio_value()` to account "1625505" only.

**Effort**: 15 minutes. CRITICAL — must be done before next trading session.

---

### Priority 0a: Fix directional accuracy gate disabled by per-ticker override [P1]

**File**: `portfolio/signal_engine.py:1840-1849`

**Problem**: When per-ticker accuracy data overrides global accuracy, the constructed dict
only includes `accuracy`, `total`, `correct`, `pct`. The keys `buy_accuracy`, `sell_accuracy`,
`total_buy`, `total_sell` are DROPPED. The directional gate in `_weighted_consensus()` (lines
829-837) uses `stats.get("buy_accuracy", acc)` which falls back to overall accuracy.

**Impact**: Signals with extreme directional bias are NOT gated. Example: qwen3 has BUY=30%
accuracy (should be gated below 35% threshold) but SELL=74%. With per-ticker data, `dir_acc`
falls back to ~50% overall, so the 35% gate never fires. The signal votes BUY freely.

**Fix**: Either:
(a) Extend `accuracy_by_ticker_signal` in `accuracy_stats.py` to compute per-direction
accuracy per-ticker, OR
(b) In the override block at line 1844, preserve directional fields from the global data:
```python
global_stats = accuracy_data.get(sig_name, {})
accuracy_data[sig_name] = {
    "accuracy": t_stats["accuracy"],
    "total": t_stats["total"],
    "correct": t_stats.get("correct", 0),
    "pct": t_stats.get("pct", ...),
    # Preserve directional accuracy from global data
    "buy_accuracy": global_stats.get("buy_accuracy", t_stats["accuracy"]),
    "sell_accuracy": global_stats.get("sell_accuracy", t_stats["accuracy"]),
    "total_buy": global_stats.get("total_buy", 0),
    "total_sell": global_stats.get("total_sell", 0),
}
```

**Effort**: 30 minutes. Low risk — additive fix.

---

### Priority 1: Fix subprocess governance [P1, 3 files]

**Files**: `portfolio/bigbet.py:170`, `portfolio/iskbets.py:318`, `portfolio/agent_invocation.py:302`

**Problem**: Three modules call `subprocess.run/Popen(["claude", "-p", ...])` directly,
bypassing `claude_gate.py`'s kill switch, rate limiter, invocation tracking, and env cleanup.

**Impact**: Kill switch (`CLAUDE_ENABLED = False`) is ineffective for 3/5 Claude callers.
During runaway invocations, only 2/5 paths can be stopped. The `CLAUDECODE` env var stripping
is missing, risking "nested session" errors.

**Fix**: Replace direct subprocess calls with `from portfolio.claude_gate import invoke_claude`.
For agent_invocation.py's async Popen path, at minimum call `_clean_env()` and `_log_invocation()`.

**Effort**: 1-2 hours. Low risk — mechanical replacement.

---

### Priority 2: Hardware trailing stop failure needs fallback [P1, metals_loop.py]

**File**: `data/metals_loop.py:4088-4134`

**Problem**: When hardware trailing stop placement fails (API error, auth issue), the new
position is created but has NO broker-level protection. A Telegram alert fires, but there's
no automatic fallback to the legacy cascade stop-loss system.

**Impact**: During Avanza API issues, new positions sit unprotected. A sharp price drop
could cause knockout without any stop-loss to limit damage.

**Fix**: On hardware trailing stop failure, automatically fall through to the legacy cascade
stop-loss block (lines 4124-4134) as a safety net. Add: `if not hw_stop_placed: HARDWARE_TRAILING_ENABLED_temp = False; [run legacy block]`.

**Effort**: 30 minutes. Medium risk — testing stop-loss behavior requires care.

---

### Priority 3: Fix timezone consistency in metals_loop [P2, 8 instances]

**File**: `data/metals_loop.py` lines 889, 1883, 3119, 3564, 4183, 4583, 6430, 6575

**Problem**: All 8 instances use `datetime.datetime.now().strftime(...)` (naive local time)
instead of `datetime.datetime.now(datetime.UTC)`. This causes date-boundary issues near
midnight CET, particularly for stop-loss deduplication ("already placed today" check).

**Impact**: Duplicate stop-loss orders after midnight CET boundary, potentially exceeding
position volume (sell + stop > units).

**Fix**: Replace all 8 with `datetime.datetime.now(datetime.UTC).strftime(...)`.

**Effort**: 30 minutes. Low risk — straightforward replacement.

---

### Priority 4: Portfolio drawdown fallback masks real crashes [P1, risk_management.py]

**File**: `portfolio/risk_management.py:43-47`

**Problem**: When live prices are unavailable (API outage), `_compute_portfolio_value` falls
back to `avg_cost_usd` from holdings. During a crash, true market price could be far below
cost basis, but the drawdown circuit breaker sees 0% drawdown.

**Impact**: Circuit breaker doesn't trigger during the worst possible scenario — a crash
combined with API outage. The system continues generating trade signals when it should be
in emergency mode.

**Fix**: When falling back to stale prices, apply a staleness penalty (assume -10%) or
refuse to compute drawdown and default to `breached = True`. Conservative is correct here.

**Effort**: 1 hour. Low risk — affects only the fallback path.

---

### Priority 5: JSONL append atomicity on Windows [P1, file_utils.py]

**File**: `portfolio/file_utils.py:155-167`

**Problem**: `atomic_append_jsonl` opens in append mode and writes. On Windows/NTFS, there
is no kernel guarantee that concurrent append-mode writes from different threads are atomic.
Multiple ticker threads could interleave bytes, producing corrupt JSONL lines.

**Impact**: Data loss in signal_log.jsonl, layer2_journal.jsonl, and other JSONL files.
The system's parsers skip malformed lines, so the effect is silent data loss rather than
crashes. The SQLite signal_db mitigates this for signal logging, but other JSONL files
(journal, invocations, telegram_messages) have no SQLite backup.

**Fix**: Add a per-file threading.Lock to `atomic_append_jsonl`. Use a module-level dict
of `{path: Lock}` to serialize appends to the same file.

**Effort**: 1 hour. Low risk — adds serialization without changing the API.

---

## Positive Patterns Found

The review also identified several excellent defensive patterns worth preserving:

1. **Fail-closed accuracy gate** (`signal_engine.py:1807-1810`): When accuracy stats loading
   fails, ALL signals are gated with 0% accuracy. Prevents trading on blind data.

2. **28 thread locks**: Comprehensive locking across all shared state. No nested lock
   patterns detected (no deadlock risk).

3. **BUG tracking**: 181+ named bugs tracked inline with code comments, creating an audit
   trail. Each fix references a specific bug number.

4. **Dogpile prevention** (`shared_state.py:23-89`): Cache-through helper prevents
   thundering herd on cache misses. Loading keys tracked with timeout eviction.

5. **Kelly criterion guards** (`kelly_sizing.py:38-51`): All edge cases properly handled
   (win_prob ≤0 or ≥1, avg_win/loss ≤0). Division by zero impossible.

6. **Circuit breaker for APIs** (`circuit_breaker.py`): Thread-safe state machine with
   CLOSED/OPEN/HALF_OPEN states. Prevents hammering failing APIs.

7. **Regime-aware signal gating**: Signals that produce negative alpha in specific market
   regimes are automatically silenced, with per-ticker exemptions for signals that work
   despite the regime.

---

## Deferred Items (Not Urgent)

1. **metals_loop.py God file** (6,963 lines): Should be split into 4-5 modules. Large
   refactor — defer to a dedicated session.

2. **Signal result schema**: No TypedDict for signal results. Add `SignalResult` dataclass.
   Low urgency — `_validate_signal_result` handles normalization.

3. **Econ calendar date staleness**: Hardcoded FOMC/CPI/NFP dates expire after 2027.
   Add a staleness warning check.

4. **journal.py raw open()**: Uses `open()` instead of `file_utils.load_jsonl()`. Low
   risk since it's read-only, but violates the project's own rules.

5. **main.py re-exports**: 50+ re-exports for backward compatibility. Gradually deprecate.

---

## Agent Review Summary

*Eight parallel adversarial review agents were launched, one per subsystem. Their findings
are being cross-referenced against the independent review above.*

### Agent: review-signals-core — COMPLETE (10 findings: 4 P1, 4 P2, 2 P3)
See `AGENT_REVIEW_SIGNALS_CORE.md` for full details. Key findings:
- **A-SC-1 [P1]**: Per-ticker accuracy override strips `buy_accuracy`/`sell_accuracy` → directional gate disabled
- **A-SC-2 [P1]**: Regime accuracy cache shared timestamp → cross-horizon contamination
- **A-SC-3 [P1]**: Ministral applicable count says crypto-only but code runs for all tickers
- **A-SC-5 [P2]**: `blend_accuracy_data` uses `max()` for sample count → inflated gate threshold
- **A-SC-6 [P2]**: `signal_history.py` read-modify-write race under ThreadPoolExecutor

### Agent: review-orchestration — COMPLETE (12 findings: 3 P1, 7 P2, 2 P3)
See `AGENT_REVIEW_ORCHESTRATION.md` for full details. Key findings:
- **A-OR-1 [P1]**: Wrong `TimeoutError` type caught — ticker hangs crash the loop on Python <3.11
- **A-OR-2 [P1]**: classify_tier/update_tier_state TOCTOU — three independent reads of trigger_state.json
- **A-OR-3 [P1]**: Multi-agent mode blocks main loop 30s synchronously
- **A-OR-4 [P2]**: `_maybe_send_digest` not wrapped in `_track` — crash aborts all post-cycle tasks
- **A-OR-5 [P2]**: Stale config in post-cycle — config changes require restart
- **A-OR-7 [P2]**: BUY↔SELL flips poison trigger consensus — rapid crypto oscillations miss signals

### Agent: review-portfolio-risk
*(Pending — will be updated when agent completes)*

### Agent: review-metals-core
*(Pending — will be updated when agent completes)*

### Agent: review-avanza-api — COMPLETE (13 findings: 2 P0, 7 P1, 3 P2, 1 P3)
See `AGENT_REVIEW_AVANZA_API.md` for full details. **MOST CRITICAL SUBSYSTEM.**
- **A-AV-1 [P0]**: Playwright context used outside lock — concurrent requests corrupt trades
- **A-AV-2 [P0]**: TOTP path has no account whitelist — pension account can receive orders
- **A-AV-3 [P1]**: get_positions/get_portfolio_value include pension account data
- **A-AV-4 [P1]**: Single CONFIRM matches wrong (most-recent) order
- **A-AV-5 [P1]**: Confirmed orders use TOTP path, not BankID session
- **A-AV-6 [P1]**: get_quote() hardcodes "stock" type — wrong price for warrants
- **A-AV-9 [P1]**: Pending orders TOCTOU race — potential double execution

### Agent: review-signals-modules
*(Pending — will be updated when agent completes)*

### Agent: review-data-external — COMPLETE (10 findings: 5 P1, 5 P2)
See `AGENT_REVIEW_DATA_EXTERNAL.md` for full details. Key findings:
- **A-DE-1 [P1]**: Alpha Vantage earnings calls bypass the 25/day budget counter
- **A-DE-4 [P1]**: fear_greed.py missing yfinance MultiIndex flatten — stock F&G signal silently dead
- **A-DE-5 [P1]**: onchain_data.py cache seeding crashes on old-format cache (ISO string as epoch)
- **A-DE-6 [P2]**: sentiment.py subprocess fallback uses wrong Python venv (.venv instead of .venv-llm)
- **A-DE-8 [P2]**: llama_server.py lock PID check uses substring match — PID "123" matches "1234"

### Agent: review-infrastructure
*(Pending — will be updated when agent completes)*

---

## Methodology Notes

- **Independent review**: Direct reading of all key files (signal_engine.py 2058 lines,
  main.py 1148 lines, metals_loop.py 6963 lines, file_utils.py 276 lines,
  risk_management.py 801 lines, avanza_orders/session/control/client.py 2298 lines,
  plus targeted scans of all other subsystems). Total ~10K lines read in detail.

- **Agent reviews**: 8 feature-dev:code-reviewer agents, each given the complete file list
  for their subsystem with specific instructions on what to look for (financial logic errors,
  thread safety, silent failures, data corruption, etc.).

- **Cross-referencing**: Both independent and agent reviews were conducted without knowledge
  of each other's findings. Agreement increases confidence; disagreement triggers investigation.

- **Vulnerability scans**: Grep-based scans for `eval()`, `exec()`, `shell=True`,
  `json.loads(open())`, bare `except: pass`, and `subprocess.run/Popen` bypasses.

---

*Review complete. Recommended action: Fix Priority 1-5 in order. Commit fixes in batches
of 3-5 files. Run full test suite after each batch.*
