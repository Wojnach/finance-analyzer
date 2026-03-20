# Improvement Plan

Updated: 2026-03-20
Branch: improve/auto-session-2026-03-20

Previous sessions: 2026-03-05 through 2026-03-19.

## Session Plan (2026-03-20)

### Theme: Thread Safety, NaN Resilience, Agent Reliability

Previous sessions completed Python modernization (REF-16/17), IO safety hardening (BUG-47
through BUG-74), silent exception elimination (BUG-56 through BUG-83), and lint cleanup
(BUG-80 through BUG-84). This session addresses three systemic gaps found by deep
code audit:

1. **Thread safety** — `generate_signal` runs in 8-thread pool but several globals lack locks
2. **NaN resilience** — NaN values from data sources propagate silently through indicators,
   signals, and into JSON output (invalid JSON per RFC 8259)
3. **Agent lifecycle** — timed-out agents lose completion logs, zombie processes possible

### 1) Bugs & Problems Found

#### BUG-85 (P1): Thread-unsafe `_prev_sentiment` access + per-ticker serialization data loss

- **File**: `portfolio/signal_engine.py:43-81`
- **Issue**: `_prev_sentiment` dict and `_prev_sentiment_loaded` flag are module globals
  accessed from `generate_signal()` which runs in an 8-thread `ThreadPoolExecutor`.
  No lock protects reads, writes, or disk serialization. Race conditions:
  1. Two threads call `_load_prev_sentiments()` simultaneously, both see `_loaded=False`
  2. `_set_prev_sentiment()` calls `atomic_write_json()` per-ticker (12-14× per cycle),
     and the last writer wins — intermediate entries from other threads are lost on disk
  3. On restart, lost entries cause incorrect hysteresis (0.40 threshold instead of 0.55)
- **Fix**: Add `threading.Lock()`. Batch disk write to once per cycle instead of per-ticker.
- **Impact**: HIGH — Wrong sentiment hysteresis after restart; potential RuntimeError on dict
  mutation during concurrent iteration.

#### BUG-86 (P2): Thread-unsafe `_adx_cache` dictionary

- **File**: `portfolio/signal_engine.py:24, 304-339`
- **Issue**: `_adx_cache` dict accessed from multiple threads without lock. `.clear()` on
  line 334 can race with reads/writes from other threads. `id(df)` keys can theoretically
  collide if DataFrames are GC'd and reallocated at the same address.
- **Fix**: Add `threading.Lock()` around cache access.
- **Impact**: MEDIUM — Possible RuntimeError on concurrent clear + read; unlikely wrong ADX.

#### BUG-87 (P1): NaN propagation from `compute_indicators` into JSON and signal decisions

- **File**: `portfolio/indicators.py:13-84`
- **Issue**: `compute_indicators()` checks `len(df) < 26` but never validates that the close
  series contains actual numeric (non-NaN) data. When data sources return gaps:
  1. BB upper/lower become NaN → `float(NaN)` propagates to agent_summary.json
  2. `json.dumps(NaN)` produces invalid JSON per RFC 8259 (Python allows it by default)
  3. JavaScript consumers (dashboard) fail to parse the file
  4. Signal comparisons like `close <= NaN` always return False, silently suppressing signals
  5. RSI with flat-price data propagates NaN through EWM if close has any NaN values
- **Fix**: Add NaN guard: forward-fill NaN close values before computation, or return None
  if the last close is NaN. Also add `allow_nan=False` to `json.dumps` in `atomic_write_json`
  to fail-fast instead of producing invalid JSON.
- **Impact**: HIGH — Silent signal suppression + invalid JSON corrupts files read by Layer 2
  and dashboard.

#### BUG-88 (P1): Tier 1 votes string always shows 0 HOLD count

- **File**: `portfolio/reporting.py:1003`
- **Issue**: Vote string uses `_voters` (which equals `buy + sell`, active voters only) to
  compute HOLD count: `_voters - _buy_count - _sell_count = 0` always. Should use
  `_total_applicable`.
- **Fix**: Replace `extra.get('_voters', 0)` with `extra.get('_total_applicable', 0)`.
- **Impact**: HIGH — Tier 1 invocations (~70% of all) show misleading vote breakdown that
  hides abstention counts, making consensus appear unanimous to Layer 2.

#### BUG-89 (P1): `update_module_failures` can crash entire summary write

- **File**: `portfolio/reporting.py:651-655`
- **Issue**: `update_module_failures()` does disk I/O (writes health_state.json). If that
  write fails (file locked, permission error), the exception propagates BEFORE
  `agent_summary.json` and `agent_summary_compact.json` are written. The entire cycle
  produces no output files.
- **Fix**: Wrap in `try/except Exception` with logger.warning.
- **Impact**: HIGH — Transient file lock on health_state.json prevents ALL summary output.

#### BUG-90 (P2): Confidence penalty cascade amplifies above 1.0 before final clamp

- **File**: `portfolio/signal_engine.py:360-440`
- **Issue**: Stage 1 (trending regime) multiplies confidence by 1.10, then Stage 2 (high
  volume) multiplies by 1.15. If conf enters at 0.85: 0.85 × 1.10 × 1.15 = 1.075.
  This inflated intermediate value bypasses Stage 2's volume_adx_gate (0.65 threshold):
  a signal with true confidence 0.62 gets boosted to 0.682, bypassing the gate.
- **Fix**: Clamp `conf = min(1.0, conf)` after each penalty stage, or apply gates before
  multipliers.
- **Impact**: MEDIUM — Marginal signals pass through gates that should stop them.

#### BUG-91 (P1): Timed-out agent completion never logged; state/metadata lost

- **File**: `portfolio/agent_invocation.py:137-161`
- **Issue**: When a timed-out agent is killed, the code falls through to spawn a new agent.
  But the timed-out invocation is never recorded to `invocations.jsonl` (no duration, tier,
  or failure status logged). `_agent_proc` is immediately overwritten, so
  `check_agent_completion()` never sees the old invocation.
- **Fix**: Log the timeout to invocations.jsonl before spawning new agent. Reset all globals.
- **Impact**: HIGH — Timeout events silently lost from logs; completion rate stats unreliable.

#### BUG-92 (P1): `taskkill` failure not checked; potential concurrent agents

- **File**: `portfolio/agent_invocation.py:148-151`
- **Issue**: `taskkill /F /T` return code is never inspected (`capture_output=True` swallows
  errors). If kill fails (access denied, PID already exited), `_agent_proc.wait(timeout=10)`
  catches `TimeoutExpired` silently, then code falls through to spawn a new agent while
  the old one may still be running. Two concurrent agents = double Claude API consumption.
- **Fix**: Check taskkill return code. If kill fails, log error and don't spawn new agent.
- **Impact**: HIGH — Potential concurrent agents consuming API quota.

#### BUG-93 (P2): Circuit breaker HALF_OPEN allows unlimited concurrent requests

- **File**: `portfolio/circuit_breaker.py:85-86`
- **Issue**: In HALF_OPEN state, `allow_request()` returns True for every caller. With
  `ThreadPoolExecutor` (up to 7 concurrent threads per ticker in `collect_timeframes`),
  all threads fire simultaneously against a recovering API. The probe should allow exactly
  one request.
- **Fix**: Add an atomic `_half_open_probe_sent` flag; only first thread gets through.
- **Impact**: MEDIUM — Floods recovering API with 7 requests instead of 1 probe.

#### BUG-94 (P2): `atomic_append_jsonl` is not atomic for concurrent writers

- **File**: `portfolio/file_utils.py:74-86`
- **Issue**: Despite the name, no file locking is used. Multiple processes (main loop,
  Layer 2 agent, outcome checker) write to the same JSONL files. On Windows, `open("a")`
  does not guarantee atomicity for writes >4096 bytes. Long JSON entries can produce
  interleaved partial lines, corrupting the JSONL file.
- **Fix**: Add `msvcrt.locking` (Windows) file lock around the write.
- **Impact**: MEDIUM — Rare in practice (most entries <4096 bytes), but possible corruption.

#### BUG-95 (P2): Stack overflow counter not reset on non-overflow failures

- **File**: `portfolio/agent_invocation.py:351-376`
- **Issue**: `_consecutive_stack_overflows` resets only on `status == "success"`. A normal
  failure (exit code 1) doesn't reset it. If pattern is: 3 overflows → 1 normal fail →
  2 overflows = counter reaches 5 → auto-disable Layer 2. The consecutive chain was broken.
- **Fix**: Also reset counter on any non-stack-overflow completion.
- **Impact**: MEDIUM — False positive auto-disable of Layer 2.

#### BUG-96 (P3): Price trigger baseline stale after long quiet periods

- **File**: `portfolio/trigger.py:245-260`
- **Issue**: `state["last"]["prices"]` only updates when a trigger fires. During long quiet
  periods, the baseline drifts arbitrarily far from current prices. A gradual 2.1% move
  over hours triggers on the next tiny change, even though no sudden event occurred.
- **Fix**: Update price baseline periodically (e.g., every 30 minutes) even without trigger.
- **Impact**: LOW — Spurious Layer 2 invocations after quiet periods; not harmful, just noisy.

#### BUG-97 (P2): Exception in `check_agent_completion` leaves agent state dirty

- **File**: `portfolio/agent_invocation.py:278-286`
- **Issue**: `_last_jsonl_ts()` can raise OSError (file locked by antivirus on Windows).
  If exception propagates, the cleanup block never runs — `_agent_proc` points to a
  completed process indefinitely. Subsequent cycles repeatedly fail in
  `check_agent_completion` and no new agents can be spawned.
- **Fix**: Wrap `_last_jsonl_ts` calls in try/except; on failure, use `None` as timestamp.
- **Impact**: MEDIUM — Agent invocation permanently blocked until I/O issue resolves.

#### BUG-98 (P3): Ghost tickers in timeframes/fear_greed never pruned

- **File**: `portfolio/reporting.py:627-645`
- **Issue**: Stale pruning logic (24h) only applies to the `"signals"` section. For
  `"timeframes"` and `"fear_greed"`, previous entries for removed tickers carry forward
  indefinitely. After the Mar 1 cleanup (12 instruments removed), their data persists.
- **Fix**: Apply same stale pruning to timeframes and fear_greed sections.
- **Impact**: LOW — Slow data accumulation; misleading to Layer 2 but not harmful.

#### BUG-99 (P3): ZeroDivisionError if `initial_value_sek` is 0

- **File**: `portfolio/reporting.py:72-73`
- **Issue**: `pnl_pct = ((total - initial) / initial) * 100` with no zero guard. If
  portfolio state has `"initial_value_sek": 0`, crashes `write_agent_summary`.
- **Fix**: Add `if initial else 0` guard (matching `_portfolio_snapshot` on line 904).
- **Impact**: LOW — Unlikely under normal operation; catastrophic if triggered.

#### BUG-100 (P2): Binance empty response recorded as circuit breaker success

- **File**: `portfolio/data_collector.py:77-95`
- **Issue**: When Binance returns 200 OK with empty list `[]`, `cb.record_success()` is
  called even though the response has no useful data. Keeps circuit breaker in CLOSED state.
- **Fix**: Check `if not data: cb.record_failure(); return pd.DataFrame()`.
- **Impact**: MEDIUM — Circuit breaker fails to activate on useless-but-successful responses.

#### BUG-101 (P3): `detect_regime` cache invalidation not thread-safe

- **File**: `portfolio/indicators.py:88-103`
- **Issue**: `_regime_cache` and `_regime_cache_cycle` on shared_state accessed without lock.
  Thread A and B can both see stale cycle ID, both clear cache, losing results written
  between the two clears.
- **Fix**: Add lock around cache check-and-clear. CPython GIL mitigates most dict ops but
  the multi-step check-clear-write pattern is a TOCTOU race.
- **Impact**: LOW — In practice, `_run_cycle_id` rarely changes during a single collection.

### 2) Architecture Improvements

None planned. The codebase architecture is solid after 15 sessions. All findings are concrete
bugs, not design issues.

### 3) What We're NOT Doing

- **Not fixing BUG-96** (price baseline drift): The behavior is arguably correct — Layer 2
  should know about 2%+ moves even if gradual. Changing trigger semantics mid-flight is risky.
- **Not fixing BUG-94** (JSONL locking): File locking on Windows is notoriously tricky
  (`msvcrt.locking` requires the file to be opened in binary mode). The current approach
  works because most entries are <4096 bytes. Marking as TODO: MANUAL REVIEW.
- **Not fixing BUG-101** (regime cache race): GIL provides sufficient protection for dict
  operations on CPython. The TOCTOU race requires exact timing that's vanishingly rare.
- **Not fixing BUG-98** (ghost ticker pruning): Low impact, and changing pruning logic could
  accidentally remove data that Layer 2 relies on for cross-asset analysis.

### 4) Dependency/Ordering

**Batch 1** (Thread Safety & NaN Guards — BUG-85, BUG-86, BUG-87):
- Files: `signal_engine.py`, `indicators.py`, `file_utils.py`
- Tests first: thread-safety tests, NaN propagation tests
- Risk: MEDIUM — threading changes require careful testing
- Commit: `fix: thread-safe sentiment state + NaN guards in indicators`

**Batch 2** (Reporting & Agent Fixes — BUG-88, BUG-89, BUG-91, BUG-92):
- Files: `reporting.py`, `agent_invocation.py`
- Tests first: Tier 1 vote string tests, timeout logging tests
- Risk: LOW — isolated fixes in separate functions
- Commit: `fix: Tier 1 vote display, summary write resilience, agent timeout logging`

**Batch 3** (Infrastructure — BUG-90, BUG-93, BUG-100):
- Files: `signal_engine.py`, `circuit_breaker.py`, `data_collector.py`
- Tests first: confidence clamping tests, half-open probe tests
- Risk: LOW — each fix is localized
- Commit: `fix: confidence clamping, circuit breaker probe, empty response handling`

**Batch 4** (Remaining Medium — BUG-95, BUG-97, BUG-99):
- Files: `agent_invocation.py`, `reporting.py`
- Risk: LOW — defensive additions only
- Commit: `fix: stack overflow counter reset, completion check resilience, zero-division guard`
