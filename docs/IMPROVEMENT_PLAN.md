# Improvement Plan — Auto-Session #4 (2026-03-01)

## Status: COMPLETE

All planned items implemented, tested, and committed. 4 commits on `main`.

## Priority: Critical Bugs > Architecture > Tests > Features > Polish

Previous sessions fixed BUG-13 through BUG-28, ARCH-1/2/3/4, REF-5/6/7/10.
This session continues from BUG-30 onward.

---

## 1. Bugs & Problems Found

### BUG-30: dashboard signal heatmap missing 3 signals
- **File:** `dashboard/app.py:286-291`
- **Severity:** MEDIUM (dashboard shows incomplete data)
- **Issue:** `/api/signal-heatmap` hardcodes 16 enhanced signal names. Missing: `forecast` (#28), `claude_fundamental` (#29), `futures_flow` (#30). Also, the core_signals list includes disabled `ml` and `funding` which always show HOLD.
- **Fix:** Add the 3 missing enhanced signals. Keep disabled signals visible (they serve as documentation).
- **Impact:** Dashboard only. No functional change to trading.

### BUG-31: digest.py reads wrong key from signal_log entries
- **File:** `portfolio/digest.py:98`
- **Severity:** MEDIUM (consensus breakdown in digest is always 0/0/0)
- **Issue:** `e.get("signals", {}).values()` looks for a top-level `signals` key, but `signal_log.jsonl` entries use `e["tickers"][ticker_name]` as the structure (written by `outcome_tracker.py:132-136`). The digest never finds any signal data.
- **Fix:** Change to iterate `e.get("tickers", {}).values()`, then access `ticker_data.get("consensus", "HOLD")` for each ticker's consensus.
- **Impact:** `digest.py` only. Fixes the 4h digest consensus breakdown.

### BUG-32: http_retry returns response on final retryable failure
- **File:** `portfolio/http_retry.py:49`
- **Severity:** LOW-MEDIUM (callers may parse error responses as data)
- **Issue:** When all retries are exhausted on a retryable status code (429, 503), the function returns the error response object. Connection-error path returns `None`. Callers checking `if resp is None` will incorrectly process a 429 response body.
- **Fix:** Return `None` after final retryable failure, consistent with exception branch.
- **Impact:** `http_retry.py` and all callers. Callers that already check `resp.ok` are unaffected. Risk: callers that relied on getting the response object back (unlikely since the behavior was undocumented).

### BUG-33: message_store SEND_CATEGORIES includes invocation
- **File:** `portfolio/message_store.py:34`
- **Severity:** LOW-MEDIUM (noisy Telegram notifications)
- **Issue:** `SEND_CATEGORIES` includes `"invocation"` and `"analysis"`, but the module docstring (lines 9-14) says these are save-only. This sends Layer 2 invocation messages and analysis messages to Telegram — the invocation messages are the brief "Layer 2 Tx invoked: reason" lines that outnumber full analyses because many Layer 2 sessions fail before completing.
- **Fix:** Remove `"invocation"` from `SEND_CATEGORIES`. Keep `"analysis"` since Layer 2 is the SOLE Telegram sender per CLAUDE.md — analysis messages from Layer 2 should be sent. Update docstring.
- **Impact:** Reduces Telegram noise. Layer 2 trade/analysis messages still sent.

### BUG-34: journal_index XAG price buckets capped at $35
- **File:** `portfolio/journal_index.py:114`
- **Severity:** LOW-MEDIUM (BM25 journal retrieval loses price context for silver)
- **Issue:** `_PRICE_BUCKETS["XAG-USD"] = [20, 25, 30, 35]`. Silver is currently trading ~$89+, prophecy target $120. All prices above $35 map to `"XAG-USD_above_35"` — no price-level discrimination for current/future silver prices.
- **Fix:** Expand to `[25, 30, 35, 50, 75, 100, 120]` covering prophecy target range.
- **Impact:** `journal_index.py` only. Improves journal retrieval relevance for silver analysis.

### BUG-35: alpha_vantage imports from portfolio_mgr instead of file_utils
- **File:** `portfolio/alpha_vantage.py:55`
- **Severity:** LOW (fragile import chain, not a crash)
- **Issue:** `from portfolio.portfolio_mgr import _atomic_write_json` — uses a private re-export from portfolio_mgr. Every other module imports from the canonical source `portfolio.file_utils`. If portfolio_mgr removes the re-export, alpha_vantage breaks.
- **Fix:** Change to `from portfolio.file_utils import atomic_write_json`.
- **Impact:** `alpha_vantage.py` only.

### BUG-36: _get_held_tickers() reads disk 6x per reporting cycle
- **File:** `portfolio/reporting.py:506-517, 539, 737, 789`
- **Severity:** LOW (unnecessary I/O, no incorrect behavior)
- **Issue:** `_get_held_tickers()` reads both portfolio JSON files from disk on every call. Called 3x per triggered cycle (compact summary, T1 summary, T2 summary) = 6 disk reads of portfolio files.
- **Fix:** Add a per-cycle cache parameter — compute once in `write_agent_summary()` and pass to callees.
- **Impact:** `reporting.py` only. Saves 4 disk reads per triggered cycle.

---

## 2. Architecture Improvements

### ARCH-6: Centralize Bold portfolio loader
- **Files:** `portfolio/portfolio_mgr.py`, `portfolio/digest.py`, `portfolio/reporting.py`, `portfolio/trigger.py`
- **Why:** Bold portfolio (`portfolio_state_bold.json`) is read directly via `json.loads()` in 4+ modules without going through a centralized loader like Patient has. No validation, no fallback handling, inconsistent error handling.
- **Change:** Add `load_bold_state()` and `save_bold_state()` to `portfolio_mgr.py`. Replace all direct reads.
- **Impact:** All modules that read Bold state. Safer, consistent.

### ARCH-7: Deduplicate Telegram Markdown fallback
- **Files:** `portfolio/telegram_notifications.py`, `portfolio/message_store.py`
- **Why:** Both implement identical Markdown fallback logic (try Markdown, retry with plain text on 400). DRY violation with divergence risk.
- **Change:** Extract shared `_send_telegram_message(token, chat_id, text)` to `telegram_notifications.py`. Have `message_store.py` call it instead of duplicating.
- **Impact:** Both files. Reduces code duplication by ~15 lines.

---

## 3. Test Coverage Improvements

### TEST-1: volume_flow.py dedicated tests
- **File:** `tests/test_signals_volume_flow.py` (new)
- **Why:** Only smoke-tested via `test_enhanced_signals.py`. 6 sub-indicators (OBV, VWAP, A/D Line, CMF, MFI, Volume RSI) have no dedicated edge-case tests.

### TEST-2: oscillators.py dedicated tests
- **File:** `tests/test_signals_oscillators.py` (new)
- **Why:** Only smoke-tested. 8 sub-indicators (Awesome, Aroon, Vortex, Chande, KST, Schaff, TRIX, Coppock) untested individually.

### TEST-3: smart_money.py dedicated tests
- **File:** `tests/test_signals_smart_money.py` (new)
- **Why:** Only smoke-tested. 5 sub-indicators (BOS, CHoCH, FVG, Liquidity Sweeps, Supply/Demand) have complex logic with no direct tests.

### TEST-4: heikin_ashi.py dedicated tests
- **File:** `tests/test_signals_heikin_ashi.py` (new)
- **Why:** Only smoke-tested. 7 sub-indicators untested individually. The `_majority_vote` wrapper is also unnecessary.

---

## 4. Refactoring

### REF-11: Remove heikin_ashi _majority_vote wrapper
- **File:** `portfolio/signals/heikin_ashi.py`
- **Why:** `_majority_vote(votes)` is a one-line passthrough to `signal_utils.majority_vote(votes)`. Unnecessary indirection.
- **Fix:** Replace all `_majority_vote()` calls with direct `majority_vote()` calls. Remove the wrapper.
- **Impact:** `heikin_ashi.py` only.

---

## 5. Items NOT Planned (Justified)

1. **`econ_dates.py` event time inaccuracy** — CPI/NFP actually release at 8:30 AM ET, not 14:00 UTC. The 30-60min difference doesn't change the 4h proximity window meaningfully. Would need DST-aware logic for marginal benefit.
2. **`forecast_signal.py` Prophet cache** — `_prophet_cache` is unused and Prophet refits every call. However, Prophet is only called when forecast signal runs (cached 5min), and the forecast signal is gated by config. Low frequency + low impact.
3. **`vector_memory.py` MD5 collision** — Two journal entries with same timestamp could collide in ChromaDB. Module is disabled by default and rare edge case.
4. **`agent.log` rotation** — No size limit. Adding RotatingFileHandler would require changes to agent_invocation.py subprocess stdout/stderr redirection. Low priority since logs are rarely inspected manually.
5. **`load_jsonl()` full file read** — Loads entire file into memory before applying limit. For signal_log.jsonl (growing unbounded), this is a concern. But the primary read path is now SQLite. JSONL reads are fallback only.
6. **`reflection.py` PnL mismatch vs equity_curve.py** — Both compute PnL differently (avg-cost vs FIFO). Unifying on FIFO is the right call but touches two modules with different consumers. Document for manual review.

---

## 6. Dependency/Ordering — Implementation Batches

### Batch 1: Bug fixes (5 files, BUG-30 through BUG-35)
**Files:** `dashboard/app.py`, `portfolio/digest.py`, `portfolio/http_retry.py`, `portfolio/message_store.py`, `portfolio/journal_index.py`, `portfolio/alpha_vantage.py`
**Changes:** BUG-30, BUG-31, BUG-32, BUG-33, BUG-34, BUG-35
**Tests needed:** Verify existing tests pass. Add regression tests for BUG-31 (digest key), BUG-32 (http_retry None).

### Batch 2: Architecture + reporting optimization (3 files, ARCH-6, ARCH-7, BUG-36)
**Files:** `portfolio/portfolio_mgr.py`, `portfolio/reporting.py`, `portfolio/telegram_notifications.py`, `portfolio/message_store.py`
**Changes:** ARCH-6 (Bold loader), ARCH-7 (Telegram dedup), BUG-36 (held tickers cache)
**Tests needed:** Verify portfolio_mgr tests pass. Add tests for bold state load/save.

### Batch 3: Signal module tests (4 new test files)
**Files:** `tests/test_signals_volume_flow.py`, `tests/test_signals_oscillators.py`, `tests/test_signals_smart_money.py`, `tests/test_signals_heikin_ashi.py`
**Changes:** TEST-1 through TEST-4
**Dependencies:** None — purely additive.

### Batch 4: Refactoring (1 file, REF-11)
**Files:** `portfolio/signals/heikin_ashi.py`
**Changes:** REF-11
**Dependencies:** Batch 3 (heikin_ashi tests must exist first).
