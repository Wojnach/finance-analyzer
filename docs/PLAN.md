# Loop Contract — Runtime Invariant Verification

## Date: 2026-04-04

## Problem

The main loop wraps every operation in try/except that silently swallows errors.
Changes to the codebase can break parts of the loop without any visible failure.
Unit tests verify individual functions but nothing verifies the loop actually
executed all expected steps each iteration. The system can be half-broken and
look "fine."

## Solution

A **Loop Contract** — a set of runtime invariants checked after every cycle.
Violations are categorized (CRITICAL/WARNING), tracked for escalation, logged,
alerted via Telegram, and optionally trigger a self-healing Claude Code session.

## Design

### New Module: `portfolio/loop_contract.py`

#### Data Structures

```python
@dataclass
class CycleReport:
    """Populated during run() to track what actually happened."""
    cycle_id: int
    active_tickers: set[str]
    signals_ok: int = 0
    signals_failed: int = 0
    signals: dict                    # ticker -> signal result
    cycle_start: float = 0.0        # monotonic time
    cycle_end: float = 0.0
    llm_batch_flushed: bool = False
    health_updated: bool = False
    heartbeat_updated: bool = False
    summary_written: bool = False
    post_cycle_results: dict         # task_name -> bool
    errors: list                     # (phase, exception_str)

@dataclass
class Violation:
    invariant: str        # e.g. "all_tickers_processed"
    severity: str         # "CRITICAL" or "WARNING"
    message: str          # human-readable description
    details: dict         # structured data for diagnostics
```

#### Contract Invariants (10 checks)

| # | Name | Severity | Check |
|---|------|----------|-------|
| 1 | all_tickers_processed | CRITICAL | signals_ok + signals_failed == len(active_tickers) |
| 2 | min_success_rate | CRITICAL | signals_ok / len(active_tickers) >= 0.5 |
| 3 | cycle_duration | WARNING | cycle_duration_s < 180 |
| 4 | llm_batch_flushed | WARNING | llm_batch_flushed == True |
| 5 | valid_signals | CRITICAL | each signal has action + confidence |
| 6 | health_updated | WARNING | health_updated == True |
| 7 | summary_written | WARNING | summary_written == True |
| 8 | signal_count_stable | WARNING | per-ticker active voter count hasn't dropped >30% vs previous |
| 9 | heartbeat_updated | WARNING | heartbeat_updated == True |
| 10 | post_cycle_complete | WARNING | all post_cycle_results are True |

#### Escalation

`ViolationTracker` persists to `data/contract_state.json`:
- Tracks consecutive violation count per invariant
- 3 consecutive warnings for the same invariant → escalate to CRITICAL
- Resets count when invariant passes

#### Self-Healing

On CRITICAL violation:
- Cooldown: max 1 session per 30 minutes
- Uses `invoke_claude()` from `claude_gate.py`
- Model: sonnet, max_turns: 15, timeout: 180s
- Prompt includes: violation details, recent errors, relevant file paths
- Allowed tools: Read, Edit, Bash, Write (full fix capability)

#### Logging

All violations → `data/contract_violations.jsonl` via `atomic_append_jsonl()`
CRITICAL + escalated → Telegram via `send_or_store(config, category="error")`

### Changes to `portfolio/main.py`

Minimal integration — the contract is a post-cycle check, not a restructuring:

1. Create `CycleReport(cycle_id, active_tickers)` at top of `run()`
2. After LLM batch flush: `report.llm_batch_flushed = True/False`
3. After health update: `report.health_updated = True/False`
4. After summary write: `report.summary_written = True/False`
5. Return report from `run()` so the loop can pass it to post-cycle
6. Pass report to `_run_post_cycle()` to track task results
7. After `_run_post_cycle()`: call `verify_and_act(report, config)`
8. After heartbeat: `report.heartbeat_updated = True/False`

The existing try/except blocks remain — we don't change error handling,
we add a verification layer ON TOP of it.

### New Test File: `tests/test_loop_contract.py`

- Test each invariant catches its violation
- Test severity classification
- Test escalation (3 consecutive → CRITICAL)
- Test self-healing cooldown
- Test violation tracking persistence
- Test CycleReport construction
- Smoke test: valid report passes all checks

## Execution Order

### Batch 1: Core module
- `portfolio/loop_contract.py` — all contract logic

### Batch 2: Main loop integration
- `portfolio/main.py` — CycleReport wiring + verify_and_act() call

### Batch 3: Tests
- `tests/test_loop_contract.py` — comprehensive test coverage

### Batch 4: Verify & ship
- Run full test suite, fix any failures
- Merge, push, clean up

## What Could Break

- Existing tests that mock `run()` or `_run_post_cycle()` — may need updated signatures
- Performance: contract check adds ~1ms per cycle (negligible vs 60s cycle)
- Self-healing sessions could be noisy if misconfigured — cooldown prevents this

## What This Does NOT Change

- No signal logic changes
- No trading behavior changes
- No config format changes
- Existing error handling remains intact
- Loop resilience strategy unchanged (keep running despite failures)
