# Improvement Plan — Auto-Session 2026-05-19

Created: 2026-05-19
Branch: `improve/auto-session-2026-05-19`

## Exploration Summary

Inherited 9 unresolved P1s from the 2026-05-17 adversarial review (P1.4–P1.12).
The 2026-05-18 session fixed P1.1 (trailing stop TypeError), P1.2 (trigger baseline
wipe), and P1.3 (multi-agent gate bypass). All remaining P1s verified against current
`main` at commit `17df25d3`.

---

## 1. Bugs to Fix (all verified open)

### P1.5 [CRITICAL] Telegram bot token leaked to retry logs
**File:** `portfolio/http_retry.py:50,54,63,67`
**Bug:** `logger.warning("HTTP %s from %s, ...")` logs full URL. Telegram callers
construct `https://api.telegram.org/bot{token}/sendMessage` — every 429/500/timeout
dumps the token into `agent.log`.
**Fix:** Add `_redact_url()` helper that masks `/bot[0-9]+:[A-Za-z0-9_-]+/`.
**Risk:** Very low. Pure string masking before logging. No behavioral change.
**Tests:** Unit test for `_redact_url` covering Telegram URLs, non-Telegram URLs,
and edge cases.

### P1.4 [HIGH] atomic_write_json silently destroys symlinks
**File:** `portfolio/file_utils.py:59`
**Bug:** `os.replace(tmp, str(path))` replaces symlinks themselves, not their targets.
`config.json` is a symlink. `telegram_poller.py:361` writes config via atomic_write_json.
**Fix:** In `atomic_write_json` and `atomic_write_text`, resolve symlinks before replace:
`path = Path(os.path.realpath(path))`.
**Risk:** Low. `os.path.realpath()` is a no-op for non-symlinks. All callers benefit.
**Tests:** Unit test that creates a symlink, writes via atomic_write_json, and verifies
the symlink still exists and target was updated.

### P1.7 [HIGH] taskkill in agent_invocation has no timeout
**File:** `portfolio/agent_invocation.py:623`
**Bug:** `subprocess.run(["taskkill", ...], capture_output=True)` — no `timeout=`.
If target hangs, blocks indefinitely while holding `_completion_lock`.
**Fix:** Add `timeout=10`. On `TimeoutExpired`, log critical and return False.
**Risk:** Very low. Adds safety net to existing kill path.
**Tests:** Existing kill tests cover this; add one for timeout scenario.

### P1.8 [HIGH] Sentiment-triggered Layer 2 misses ticker context
**File:** `portfolio/main.py:244`
**Bug:** `_TICKER_PAT` regex only matches `consensus|moved|flipped`. Sentiment reversal
reasons fall through, so Layer 2 gets no triggered ticker.
**Fix:** Extend regex to `consensus|moved|flipped|sentiment`.
**Risk:** Very low. Only adds one more match keyword.
**Tests:** Unit test for sentiment reason string extraction.

### P1.9 [HIGH] Empty yfinance DataFrame returned, crashes downstream
**File:** `portfolio/price_source.py:152-153`
**Bug:** `if df.empty: return df` — returns empty DF instead of raising
`SourceUnavailableError`. Downstream `.iloc[-1]` crashes.
**Fix:** `if df.empty: raise SourceUnavailableError(...)`.
**Risk:** Low. The outer `fetch_klines` try/except already handles this error type.
**Tests:** Unit test verifying SourceUnavailableError raised on empty result.

### P1.11 [MEDIUM] Hidden 50% accuracy gate in Mode B probability
**File:** `portfolio/ticker_accuracy.py:130`
**Bug:** Hardcoded `if accuracy < 0.50: continue` drops signals at 47-49.9% from
probability computation. Should use `ACCURACY_GATE_THRESHOLD` (0.47).
**Fix:** Import and use `ACCURACY_GATE_THRESHOLD` from signal_engine.
**Risk:** Low. May slightly change Mode B probability values (more signals included).
**Tests:** Unit test verifying a signal at 48% is included.

### P1.10 [MEDIUM] News event signal does shared-file disk I/O from worker threads
**File:** `portfolio/signals/news_event.py:96`
**Bug:** `atomic_write_json(_HEADLINES_PATH, payload)` called from 8 worker threads.
Last-write-wins race causes cross-ticker pollution.
**Fix:** Add a threading.Lock around the persist call, or better: accumulate
headlines in-memory and let a single post-cycle consumer write them.
**Risk:** Medium. Lock approach is simpler but adds synchronization to signal path.
Will use lock approach (simpler, proven pattern in this codebase).
**Tests:** Verify lock prevents concurrent writes.

### P1.6 [MEDIUM] Backtester look-ahead bias
**File:** `portfolio/backtester.py:93`
**Bug:** `accuracy_data = _build_accuracy_data(horizon)` built ONCE from full signal_log
including future outcomes. Every scored entry has look-ahead.
**Risk assessment:** This is a structural fix that changes backtester behavior. All
existing comparisons would need re-running. High complexity, moderate risk.
**Decision:** SKIP for auto-session. Document with TODO comment. Needs dedicated
backtester overhaul session.

### P1.12 [LOW-MEDIUM] Horizon mismatch: 3d/5d/10d uses 1d accuracy
**File:** `portfolio/signal_engine.py:4026`
**Bug:** `acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"` collapses
multi-day horizons to 1d accuracy stats.
**Risk assessment:** Fixing requires multi-day accuracy data to exist in the cache.
If it doesn't, gate falls back to 1d anyway. The horizon-disabled list at L875
already manually handles the worst cases. Partial fix possible.
**Decision:** SKIP for auto-session. Complex data dependency. Document with TODO.

---

## 2. False Positives / Deferred

- **P1.6** (backtester look-ahead): Too structural for auto-session. Needs walk-forward redesign.
- **P1.12** (horizon mismatch): Requires multi-day accuracy cache infrastructure.
- **ARCH-18** (metals_loop monolith): 7800+ line file, needs dedicated session.
- **ARCH-17** (main.py re-exports): Breaks 10+ test files, too risky.

---

## 3. Implementation Batches

### Batch 1: Security & Logging (2 files)
1. `portfolio/http_retry.py` — P1.5: URL redaction before logging
2. `portfolio/file_utils.py` — P1.4: symlink-safe atomic writes

### Batch 2: Reliability (2 files)
1. `portfolio/agent_invocation.py` — P1.7: taskkill timeout
2. `portfolio/price_source.py` — P1.9: raise on empty yfinance

### Batch 3: Signal Accuracy (2 files)
1. `portfolio/main.py` — P1.8: extend trigger ticker regex
2. `portfolio/ticker_accuracy.py` — P1.11: use ACCURACY_GATE_THRESHOLD

### Batch 4: Thread Safety (1 file)
1. `portfolio/signals/news_event.py` — P1.10: lock around headline persist

### Batch 5: Documentation
1. `portfolio/backtester.py` — P1.6: TODO comment for look-ahead bias
2. `portfolio/signal_engine.py` — P1.12: TODO comment for horizon mismatch

---

## 4. Success Criteria

- [ ] All batches implemented with passing tests
- [ ] Full test suite green (`pytest tests/ -n auto`)
- [ ] No new test failures introduced
- [ ] Adversarial review passes
- [ ] Merged to main and pushed
