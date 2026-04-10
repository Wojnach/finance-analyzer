# Agent Adversarial Review: data-external

**Agent**: feature-dev:code-reviewer
**Subsystem**: data-external (6,062 lines, 21 files)
**Duration**: ~196 seconds
**Findings**: 10 (5 P1, 5 P2)

---

## P1 Findings

### A-DE-1: Alpha Vantage Earnings Calls Bypass Budget Counter [P1]
- **File**: `portfolio/earnings_calendar.py:49-52`
- **Description**: `_fetch_earnings_alpha_vantage()` directly calls the AV EARNINGS endpoint without incrementing `alpha_vantage._daily_budget_used`. Comment in code explicitly acknowledges this: "earnings calls bypass alpha_vantage.py's _daily_budget_used counter."
- **Impact**: Invisible API budget drain. With 12 stock tickers (pre-Apr 9), 12/25 daily calls consumed invisibly.
- **Fix**: Export `increment_budget()` from `alpha_vantage.py`, call after successful earnings fetch.

### A-DE-2: get_open_interest() Docstring Promises oi_usdt Key That Doesn't Exist [P1]
- **File**: `portfolio/futures_data.py:36,49-53`
- **Description**: Docstring says returns `{oi, oi_usdt, symbol, time}`. Actual return dict is `{oi, symbol, time}` — `oi_usdt` is absent.
- **Impact**: Any caller accessing `["oi_usdt"]` gets KeyError, causing futures signal to fall back to HOLD.
- **Fix**: Remove `oi_usdt` from docstring and verify no callers depend on it.

### A-DE-3: local_llm_report.py Naive vs Aware Datetime Comparison [P1]
- **File**: `portfolio/local_llm_report.py:34,48,51`
- **Description**: `cutoff` is aware (UTC), but `fromisoformat()` on old entries may return naive datetime. TypeError is caught and entry silently discarded.
- **Impact**: Old forecast entries silently dropped from accuracy reports.
- **Fix**: Normalize: `if ts_raw.tzinfo is None: ts_raw = ts_raw.replace(tzinfo=UTC)`.

### A-DE-4: fear_greed.py Missing yfinance MultiIndex Flatten [P1]
- **File**: `portfolio/fear_greed.py:118`
- **Description**: `get_stock_fear_greed()` accesses `h["Close"]` without flattening potential MultiIndex columns from newer yfinance versions. `data_collector.py` already has this fix.
- **Impact**: Stock F&G signal (MSTR) silently returns None every cycle — no F&G vote for stocks.
- **Fix**: Add `if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.get_level_values(0)`.

### A-DE-5: onchain_data.py Cache Seeding Crashes on Old-Format Cache [P1]
- **File**: `portfolio/onchain_data.py:211-212`
- **Description**: `cache_ts = persistent.get("ts", 0) or persistent.get("_fetched_at", 0)` — when `_fetched_at` is an ISO string, `time.time() - cache_ts` raises TypeError.
- **Impact**: On restart with old cache file, on-chain data loading fails entirely. Burns a BGeometrics API call.
- **Fix**: Validate `cache_ts` is numeric: `cache_ts = float(persistent.get("ts") or 0)`.

---

## P2 Findings

### A-DE-6: sentiment.py Subprocess Fallback Uses Wrong Python Venv [P2]
- **File**: `portfolio/sentiment.py:34`
- **Description**: `MODELS_PYTHON` points to `.venv/Scripts/python.exe` (main venv) instead of `.venv-llm` (ML venv with torch). Subprocess fallback fails with ModuleNotFoundError.
- **Impact**: Double-failure: if in-process BERT breaks AND subprocess fallback breaks, sentiment returns HOLD with no alert.
- **Fix**: Change to `r"Q:\models\.venv-llm\Scripts\python.exe"`.

### A-DE-7: crypto_macro_data.py Gold/BTC Ratio Uses Stale Prices [P2]
- **File**: `portfolio/crypto_macro_data.py:208-219`
- **Description**: Reads prices from `agent_summary_compact.json` (previous cycle) instead of live API. Violates "live prices first" rule. Currently disabled via DISABLED_SIGNALS.
- **Impact**: If re-enabled, ratio trend could be wrong during fast markets.

### A-DE-8: llama_server.py Lock PID Check Uses Substring Match [P2]
- **File**: `portfolio/llama_server.py:393-413`
- **Description**: `str(lock_pid) not in result.stdout` — PID "123" matches output for PID "1234", creating false positive "alive" determination.
- **Impact**: Stale lock appears alive, blocking all LLM inference for 5 minutes.
- **Fix**: Check for "No tasks are running" in output instead of substring PID match.

### A-DE-9: social_sentiment.py Uses print() Instead of Logger [P2]
- **File**: `portfolio/social_sentiment.py:110,122`
- **Description**: Reddit errors reported via `print()` not `logger.debug()`. Invisible to structured logging.

### A-DE-10: forecast_signal.py Chronos Column Access May Break [P2]
- **File**: `portfolio/forecast_signal.py:219-220`
- **Description**: Accesses `row["0.5"]` as string key, but Chronos-2 may return float-typed column labels. KeyError would silently kill all forecast signals.
- **Fix**: Normalize: `pred_df.columns = [str(c) for c in pred_df.columns]`.
