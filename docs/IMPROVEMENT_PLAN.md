# Improvement Plan — Auto-Session 2026-06-06

Generated from deep exploration of all major subsystems by 4 parallel agents.

## Prioritization Key

- **P0 (Critical)**: Correctness bugs that affect trading decisions or data integrity
- **P1 (High)**: Performance, reliability, or silent-failure issues
- **P2 (Medium)**: Code quality, maintainability, dead code
- **P3 (Low)**: Style, documentation, minor inconsistencies

---

## 1. Bugs & Problems Found

### B1 [P0] `_cached()` TTL field not stored — eviction defaults wrong
**File:** `portfolio/shared_state.py:99`
**Issue:** `_cached()` stores cache entries as `{"data": ..., "time": ...}` without a `ttl` field. The eviction logic at line 56 uses `v.get("ttl", 3600)` — so entries with TTL=300s (Fear & Greed) are evicted at 3600×3=10800s instead of 300×3=900s. Entries from `_update_cache` and `_cached_or_enqueue` correctly include `ttl`.
**Impact:** Stale data served from cache beyond intended TTL during eviction pressure. Low-frequency because eviction only triggers at `_CACHE_MAX_SIZE`, but semantically wrong.
**Fix:** Add `"ttl": ttl` to the cache entry dict in `_cached()`.

### B2 [P0] `risk_management.check_drawdown` bypasses `portfolio_mgr` backup recovery
**File:** `portfolio/risk_management.py`
**Issue:** `check_drawdown` calls `load_json(portfolio_path)` directly instead of `portfolio_mgr.load_state()`. If `portfolio_state.json` is corrupt, `load_json` returns `{}`, which means `cash_sek = 0`, `holdings = {}` → no drawdown detected → trading continues when it should halt.
**Impact:** Circuit breaker fails to trip on corrupt state. The backup recovery and quarantine logic in `portfolio_mgr` exists precisely for this case.
**Fix:** Import and use `portfolio_mgr.load_state()` / `load_bold_state()`.

### B3 [P1] `claude_gate.invoke_claude_text` docstring says 3-tuple, returns 4-tuple
**File:** `portfolio/claude_gate.py`
**Issue:** Docstring says returns `tuple[str, bool, int]` but actual return is `(text, status == "invoked", exit_code, status)` — 4 elements. Any caller destructuring to 3 will crash.
**Impact:** Runtime error for callers that trust the docstring. Need to verify call sites.
**Fix:** Update docstring to match actual 4-tuple return.

### B4 [P1] `last_jsonl_entry` does not use recovery decoder
**File:** `portfolio/file_utils.py:373-411`
**Issue:** Uses `json.loads(line)` directly. Concatenated-object lines (from legacy append-race corruption) raise `JSONDecodeError`, causing the entry to be skipped. The recovery decoder `_decode_jsonl_line` exists for this exact case but is not used here.
**Impact:** Returns older entry instead of latest on corrupt lines. Affects `check_agent_completion`, health monitoring, and any module calling `last_jsonl_entry`.
**Fix:** Use `_decode_jsonl_line(line)` and take the last decoded object.

### B5 [P1] `metals_cross_asset` signal listed as active but disabled per-ticker
**File:** `portfolio/signal_engine.py:948-956`, `portfolio/tickers.py`
**Issue:** `metals_cross_asset` is listed in CLAUDE.md as active signal #11 but exists in `_TICKER_DISABLED_SIGNALS["XAU-USD"]` and `_TICKER_DISABLED_SIGNALS["XAG-USD"]` — the only tickers it applies to. Computes every cycle (yfinance + FRED calls) but vote is force-HOLD'd.
**Impact:** Wasted I/O (2 API calls/cycle), misleading signal count in docs, zero accuracy data accumulated.
**Fix:** Add to `DISABLED_SIGNALS` properly (saves the API calls) and update CLAUDE.md signal count.

### B6 [P1] `_SHADOW_SAFE_SIGNALS` contains active signals (dead code)
**File:** `portfolio/signal_engine.py`
**Issue:** `_SHADOW_SAFE_SIGNALS` includes `drift_regime_gate`, `vol_ratio_regime`, `bocpd_regime_switch`, `amihud_illiquidity_regime` — all currently active (not in `DISABLED_SIGNALS`). The shadow path only executes for disabled signals, so these entries are dead code.
**Impact:** Maintenance confusion. Changing status of these signals requires auditing two structures.
**Fix:** Remove active signals from `_SHADOW_SAFE_SIGNALS`.

### B7 [P1] `trigger.py` stores non-serializable `set` in state dict
**File:** `portfolio/trigger.py:233`
**Issue:** `state["_current_tickers"] = set(signals.keys())` — Python `set` is not JSON-serializable. Works only because `_save_state` pops this key before writing. If the pop is ever missed (e.g., early return on error), the state file write crashes.
**Impact:** Fragile — one missed pop causes `TypeError` on state save, breaking trigger detection.
**Fix:** Use `list(signals.keys())` instead of `set(...)`, or move the set to a local variable instead of state dict.

### B8 [P2] `autonomous.py` writes orphaned `layer2_decisions.jsonl`
**File:** `portfolio/autonomous.py:187`
**Issue:** Writes to `DECISIONS_FILE` (`layer2_decisions.jsonl`) on every call. Nothing in the codebase reads this file. It grows unbounded with no pruning.
**Impact:** Disk space leak, confusing artifact.
**Fix:** Remove the write, or wire up a consumer. Given nothing reads it, remove.

### B9 [P2] `silver_monitor.py` raw JSON I/O violations
**File:** `data/silver_monitor.py:106,606,608`
**Issue:** Uses `json.load(open(...))` and `with open(..., 'a') as f: f.write(json.dumps(...))` bypassing `file_utils`. Violates CLAUDE.md rule 4.
**Impact:** No atomic writes, no corruption recovery, no sidecar locking on JSONL appends.
**Fix:** Replace with `load_json` / `atomic_append_jsonl`.

### B10 [P2] `agent_invocation.py` auth cooldown loads entire `invocations.jsonl`
**File:** `portfolio/agent_invocation.py:818-838`
**Issue:** Calls `load_jsonl(INVOCATIONS_FILE)` then slices `[-50:]`. On a long-running system with thousands of entries, this is O(n) per invocation check. `load_jsonl_tail` is available and used elsewhere.
**Impact:** Unnecessary I/O on every Layer 2 trigger.
**Fix:** Replace with `load_jsonl_tail(INVOCATIONS_FILE, max_entries=50)`.

---

## 2. Architecture Improvements

### A1 [P2] `atomic_write_json` `ensure_ascii` inconsistency
**File:** `portfolio/file_utils.py:53`
**Issue:** `atomic_write_json` uses `ensure_ascii=True` while `atomic_append_jsonl` uses `ensure_ascii=False`. Raw file inspection of JSON files shows escaped Unicode.
**Fix:** Change `atomic_write_json` default to `ensure_ascii=False` to match.

### A2 [P3] `classify_tier` double-reads trigger state
**File:** `portfolio/trigger.py:609`, `portfolio/main.py`
**Issue:** `classify_tier` loads state from disk when `state=None`, but `main.py` just loaded it in `check_triggers`. The `state` parameter exists but isn't passed.
**Fix:** Pass state from `check_triggers` return to `classify_tier`.

### A3 [P3] `MARKET_OPEN_HOUR = 7` exported as constant
**File:** `portfolio/market_timing.py:13`
**Issue:** Labelled "backward compat, kept at 7 (summer value)" but exported. Wrong by 1h in winter.
**Fix:** Deprecate or remove; callers should use `_eu_market_open_hour_utc(dt)`.

---

## 3. Implementation Results

### Batch 1: Critical data integrity fixes [B2, B4] ✅
- B1 SKIPPED: already fixed (line 99 includes `"ttl": ttl`)
- B2 DONE: `check_drawdown` now uses `portfolio_mgr._load_state_from()` with backup recovery
- B4 DONE: `last_jsonl_entry` now uses `_decode_jsonl_line()` recovery decoder + 8 new tests
- 329 signal engine tests + 142 file_utils tests pass

### Batch 2: Signal system cleanup [B5, B6] ✅
- B5 DONE: `metals_cross_asset` added to `DISABLED_SIGNALS` (saves yfinance+FRED API calls)
- B6 DONE: removed `drift_regime_gate` and `amihud_illiquidity_regime` from `_SHADOW_SAFE_SIGNALS` (active signals, shadow path unreachable)
- 329 signal engine tests pass

### Batch 3: Reliability fixes [B3, B7, B10] ✅
- B3 DONE: `invoke_claude_text` type annotation fixed to 4-tuple
- B7 DONE: `trigger.py` `_current_tickers` changed from `set()` to `list()`
- B10 DONE: auth cooldown replaced `load_jsonl` with `load_jsonl_tail(max_entries=50)`
- 349 trigger/agent tests pass (1 pre-existing failure: test_bug38_empty_set_prunes_all)

### Batch 4: I/O cleanup [A1] ✅
- B8 SKIPPED: `decision_outcome_tracker.py` reads `layer2_decisions.jsonl` (exploration agent incorrectly reported no consumers)
- B9 SKIPPED: flagged lines 606/608 are inside a Claude prompt template string, not real code
- A1 DONE: `atomic_write_json` default `ensure_ascii=False` (was True, inconsistent with `atomic_append_jsonl`)
- 142 file_utils tests pass

### Batch 5: Minor cleanup [A3] ✅
- A2 SKIPPED: requires changing `check_triggers` return signature — too invasive for P3
- A3 DONE: removed dead `MARKET_OPEN_HOUR` re-export from `main.py`, added deprecation comment
- 223 market_timing tests pass

---

## 4. Impact Assessment

All changes are backward-compatible. No API changes, no config changes, no new dependencies.

Risk areas:
- B2 (check_drawdown): changes the load path for circuit breaker. If `portfolio_mgr.load_state()` has different error handling than `load_json`, the drawdown check could behave differently on edge cases. Mitigation: the backup recovery in `portfolio_mgr` is strictly more robust.
- B4 (last_jsonl_entry): changing the parser could surface previously-hidden corrupt lines. This is desired behavior but may cause different entries to be returned for the same file. Mitigation: `_decode_jsonl_line` is already used in `load_jsonl` and is well-tested.
- A1 (ensure_ascii): existing files with `\uXXXX` escapes will still be readable. New writes will use raw UTF-8. No backward compat issue.

---

## 5. Out of Scope (Deferred)

- **signal_engine.py decomposition** (4698 lines → multiple modules): High-impact but high-risk refactor. Needs dedicated session with careful test coverage.
- **metals_loop.py decomposition** (7904 lines): Same — too large for this session.
- **Circuit breaker per-ticker isolation**: Architectural change to data_collector. Needs design doc.
- **IC-based signal weighting**: P1 research priority, already has a plan in `quant_research_priorities.md`.
- **Backtester look-ahead bias** (P1.6): Needs walk-forward infrastructure.
