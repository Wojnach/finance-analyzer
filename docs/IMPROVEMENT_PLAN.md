# Improvement Plan

Updated: 2026-03-06
Branch: worktree-auto-improve-0306

Previous session (2026-03-05) fixed: dashboard JSONL hardening, accuracy stats resilience,
static export parity/auth. Those items are marked DONE below.

## 1) Bugs & Problems Found

### BUG-1 (P1): CircuitBreaker not thread-safe

- **File**: `portfolio/circuit_breaker.py`
- **Evidence**: `_state`, `_failure_count`, `_last_failure_time` are plain instance attributes with no synchronization. The `data/metals_llm.py` LLM thread and Chronos thread run concurrently and share circuit breaker state via `data_collector` imports.
- **Failure mode**: Under concurrent access, state transitions can be inconsistent — e.g., failure count increment lost, or HALF_OPEN → CLOSED while another thread records a failure.
- **Impact**: Data collection can retry failing APIs or block working ones.

### BUG-2 (P1): `autonomous.py` caches consensus accuracy forever

- **File**: `portfolio/autonomous.py:38-62`
- **Evidence**: `_consensus_acc_cache` is set once on first call and never expires. Since the process is long-lived, accuracy data becomes stale. Worse, `_consensus_acc_cache = None` (line 61) caches the "not found" state forever.
- **Failure mode**: Autonomous decisions use outdated or missing accuracy data for the entire loop lifetime.
- **Impact**: Medium — degrades autonomous decision quality over time.

### BUG-3 (P2): `_held_tickers_cache` in reporting.py not restart-safe

- **File**: `portfolio/reporting.py:575`
- **Evidence**: Cache uses `cycle_id: -1` but `_run_cycle_id` starts at 0 on restart. If the previous session reached cycle_id=100, the inequality check fails.
- **Failure mode**: Stale held-tickers on first few cycles after restart.
- **Impact**: Low — self-corrects after a few cycles.

### BUG-4 (P2): `_classify_tickers` T3 sell_count is always 0

- **File**: `portfolio/autonomous.py:226-229`
- **Evidence**: T3 branch adds all SELL tickers to `actionable` on line 222. The subsequent loop (line 226-229) searching for SELLs _not_ in actionable is dead code — they were all already added.
- **Failure mode**: Telegram summary line `_+N hold · M sell_` never shows sells for T3 tier.
- **Impact**: Low — display-only bug.

### BUG-5 (P3): `prune_jsonl` races with `atomic_append_jsonl`

- **File**: `portfolio/file_utils.py:84-121`
- **Evidence**: `prune_jsonl` reads all lines, then does `os.replace(tmp, path)`. If `atomic_append_jsonl` appends between read and replace, that line is lost.
- **Failure mode**: Occasional loss of a single JSONL entry during pruning. Window is small since both run in the same thread, but external processes (metals_loop) could also append.
- **Impact**: Low — pruning is infrequent and the data is telemetry.

### ~~P1 — Dashboard JSONL schema assumptions~~ (DONE, 2026-03-05)
### ~~P1 — Accuracy endpoint malformed JSONL~~ (DONE, 2026-03-05)
### ~~P2 — Static export parity/auth~~ (DONE, 2026-03-05)

## 2) Architecture Improvements

### ARCH-1: Thread-safe CircuitBreaker

- Add `threading.Lock` to `CircuitBreaker` and wrap all state mutations.
- Low risk, purely additive. No behavioral change for single-threaded usage.

### ARCH-2: TTL-based autonomous accuracy cache

- Replace `_consensus_acc_cache` with a time-bounded cache (e.g., 5 minute TTL).
- Low risk, improves decision quality.

### ARCH-3: Restart-safe held-tickers cache

- Reset `_held_tickers_cache` when `cycle_id == 0` (process restart signal).
- Trivial fix.

### ARCH-4: Fix T3 sell_count computation

- Compute sell_count from `actionable` dict instead of dead second loop.
- Low risk, fixes display bug.

### ARCH-5: Config validation for all CLI modes

- Run `validate_config_file()` before all `main.py` CLI commands, not just `--loop`.
- Low risk, better developer experience. Guard with try/except to not break legacy commands.

## 3) Implementation Batches

### Batch 1: CircuitBreaker thread safety + health exposure (BUG-1, ARCH-1)

Files:
- `portfolio/circuit_breaker.py` — add Lock
- `portfolio/health.py` — expose CB status
- `tests/test_circuit_breaker.py` — new, comprehensive tests

### Batch 2: Autonomous fixes (BUG-2, BUG-4, ARCH-2, ARCH-4)

Files:
- `portfolio/autonomous.py` — fix accuracy cache + sell_count
- `tests/test_autonomous.py` — new tests for these specific behaviors

### Batch 3: Cache + startup fixes (BUG-3, ARCH-3, ARCH-5)

Files:
- `portfolio/reporting.py` — fix held_tickers_cache
- `portfolio/main.py` — add config validation before CLI commands

### Batch 4: File safety (BUG-5)

Files:
- `portfolio/file_utils.py` — add a note about the race; use a simple lock flag
- Document as known limitation if cross-platform locking is too complex
