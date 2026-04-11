# Session Progress — Fix-Queue 2026-04-11 (Adversarial-Review P0 Batch)

## Status: SHIPPING

Autonomous fix-queue session driven by `/fgl` protocol. Worked through the
P0/P1 findings from the 2026-04-10 8-agent adversarial review
(`docs/adversarial-review/SYNTHESIS.md`) plus the auto-session 2026-04-11
improvement plan.

Branch: `fix/queue-2026-04-11` (worktree at `/mnt/q/finance-analyzer-fixq`)

## What shipped (11 fixes + test/plan housekeeping = 23 commits)

### Batch 1 — Defensive additive fixes (low risk, P0)

1. **`7ad33cf`** fix(avanza): A-AV-1 wrap api_get/post/delete in `_pw_lock`
   - Upgraded `_pw_lock` from `Lock` to `RLock` (reentrant)
   - Wrapped all 5 Playwright touch points in `with _pw_lock:`
   - 4 new thread-safety tests prove max-concurrent = 1 across api_get/post/delete

2. **`0daa2ef`** fix(avanza): A-AV-2 hardcode account whitelist in TOTP path
   - New `ALLOWED_ACCOUNT_IDS = {"1625505"}` constant in `avanza_client.py`
   - `get_account_id()` enforces whitelist before caching → pension account 2674244 cannot be reached
   - `get_positions()` and `get_portfolio_value()` filter to whitelist
   - 7 new tests including the actual pension-ID rejection scenario

3. **`f07469c`** fix(validator): A-PR-3 use `file_utils.load_json` instead of raw `open()`
   - Replaces TOCTOU-prone raw json.load with atomic-aware loader
   - New `tests/test_portfolio_validator.py` with AST-based regression guard

4. **`122658b`** fix(claude_gate): A-IN-2 kill subprocess **tree** on TimeoutExpired
   - Refactored to `Popen` + `_run_with_tree_kill` helper with platform-specific tree-kill
   - Windows: `taskkill /T /F /PID`; Unix: `os.killpg(SIGKILL)` on `start_new_session=True`
   - 8 new tests including a verified **grandchild kill on Windows**

### Batch 2 — Data correctness (low risk, P0)

5. **`7597650`** fix(fin_snipe): A-MC-2 use `fetch_usd_sek()` instead of `usdsek=1.0`
   - Was actually in `fin_snipe_manager.py:420`, not metals_loop
   - Live FX rate via `portfolio.fx_rates.fetch_usd_sek` (15-min cached, 10.85 fallback)
   - Exit-optimizer SEK rewards previously wrong by ~10x
   - AST regression guard

6. **`174eca0`** fix(risk): A-PR-2 stream full history for drawdown peak
   - Previously `load_jsonl_tail(max_entries=2000)` lost peaks older than ~33h
   - New `_streaming_max(history_path, value_key, floor)` walks the entire JSONL line-by-line
   - Regression test seeds 999K peak as oldest entry with 2500 lower entries on top

7. **`1a8381f`** fix(fear_greed): A-DE-4 flatten yfinance MultiIndex columns for VIX
   - Newer yfinance returns MultiIndex even on single Ticker.history() — silently killed the stock F&G signal
   - Defensive flatten + 4 new tests covering MultiIndex shape

8. **`aecec90`** fix(onchain): A-DE-5 coerce ISO-string ts in onchain cache to epoch
   - New `_coerce_epoch()` helper handles int/float/numeric-string/ISO-string/garbage
   - Fixes silent on-chain BTC voter death after restart with old cache
   - 8 new tests including end-to-end ISO-cache reload

### Batch 3 — Logic bugs (medium risk, P0/P1)

9. **`c595ab0`** fix(signals): A-SM-1 explicit guard against gap-fill firing on widening gap
   - **A-SM-1 was a false positive** — investigation showed existing `fill_pct < 0.3` already handled the case (math: positive/negative = negative < 0.3 → HOLD)
   - Added explicit `if fill_pct < 0: HOLD` for clarity + 3 regression tests covering all 4 (gap_dir × day_dir) quadrants

10. **`a75e8f7`** fix(volatility): A-SM-2 GARCH in `_empty_result` schema
    - Empty path was missing `garch` sub_signal + `garch_vol`/`realized_vol`/`garch_ratio` indicators
    - Cross-path schema regression test

11. **`ceab91b`** fix(fin_snipe): A-MC-4 persist real `entry_ts` so HOLD_TIME_EXTENDED works
    - Was `entry_ts=now()` every cycle → `hold_hours ≈ 0` → flag never fired
    - Now persisted in `instrument_state` on first non-zero observation, cleared on close
    - Bootstrap: existing positions get entry_ts = first cycle after fix (acceptable)

### Batch 4 — Concurrency

12. **`e0a4605`** fix(claude_gate): A-IN-3 in-process invocation lock
    - Module-level `_invoke_lock` (threading.Lock) wraps the actual subprocess call inside both invoke_claude / invoke_claude_text
    - 8-worker ticker pool can no longer spawn 5+ concurrent expensive Claude processes
    - 5-thread serialization test verifies max-concurrent = 1

### Batch 5 — Signal-system tuning

13. **`6c7e289`** fix(signals): raise `ACCURACY_GATE_THRESHOLD` 0.45 → 0.47
    - Gates the 4 signals sitting in the 45-47% coin-flip-adjacent band per the 2026-04-10 audit
    - Companion test fixes in `d55f8fe`

## Tasks dropped per prior-session findings (4)

- **BUG-184** trade_guards lock — Layer 2 runs as subprocess, not thread. atomic_write_json adequate.
- **BUG-183** autonomous per-ticker throttle — BUY/SELL signals already bypass the global throttle.
- **fear_greed gate verification** — prior session verified blended = 0.586, correctly ungated.
- **per-ticker signal blacklist** — per-ticker accuracy gate already catches `ministral × XAG` (18.9%).

## Test status

- **All ~617 tests in changed areas pass** — 100% green for everything I touched.
- **29 pre-existing failures** in unrelated areas (test_strategy.py freqtrade ModuleNotFoundError ×7, test_meta_learner signal-count mismatch, test_forecast_circuit_breaker integration, test_metals_llm_orphan JobObject, test_signal_improvements vote-count + low-sample neutral-weight from BUG-182). All confirmed pre-existing on `main` HEAD before this branch — verified by checking out parent commits and re-running.

## Next session priorities

1. Update the ~7 pre-existing test failures (count drift after recent signal additions, BUG-182 behavior change in `test_low_sample_uses_neutral_weight`)
2. Codex adversarial review on this branch (`/codex:adversarial-review --wait --scope branch --effort xhigh`)
3. After merge: restart `PF-DataLoop` so the new `_invoke_lock` and accuracy gate take effect
4. Monitor 48h for: drawdown circuit breaker firing rate, on-chain voter restart resilience, HOLD_TIME_EXTENDED metals exits
5. Cross-process file lock on `claude_gate` (deferred from A-IN-3 — only in-process is shipped)
