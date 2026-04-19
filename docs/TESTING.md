# Testing Guide

## Quick Reference

```bash
# Full suite (sequential) — ~16 minutes
.venv\Scripts\python.exe -m pytest --tb=short -q

# Full suite (parallel, all CPU cores) — ~4-5 minutes
.venv\Scripts\python.exe -m pytest -n auto --tb=short -q

# Single module (fast)
.venv\Scripts\python.exe -m pytest tests/test_monte_carlo.py -v

# Run by keyword
.venv\Scripts\python.exe -m pytest -k "monte_carlo" -v
```

## Test Suite Stats (as of 2026-04-19)

| Metric | Value |
|--------|-------|
| Total tests | ~7,730 |
| Passing | 7,730 |
| Pre-existing failures | 24 (infra deps: freqtrade 15 + Ministral 9) |
| Sequential runtime | ~16 min |
| Parallel runtime (`-n auto`) | **~8 min** (8 workers) |
| Test files | ~242 |

## Pre-existing Failures & xdist Hygiene

### Integration skip (always-ignored)

| Count | Test File | Cause |
|-------|-----------|-------|
| 15 | `tests/integration/test_strategy.py` | Missing `ta_base_strategy` (Freqtrade). Always skipped via `--ignore=tests/integration`. |

### xdist isolation flakes (2026-04-17 diagnosis)

Under `pytest -n auto`, the FULL suite on main reports 5-10 failures **per
run**, with a **different set** each time. They all pass when run in
isolation or in small subsets. Root cause: module-level state leakage
across test files under xdist's worker sharding — `_agent_proc`,
`_chronos_proc`, GPU-gate caches, signal-registry imports, etc. Tests
that don't reset shared globals get hit by whichever earlier test the
scheduler happened to run first on that worker.

Known-affected clusters (incomplete — set rotates):

| Cluster | Typical tests | Shared state leak |
|---------|---------------|-------------------|
| `tests/test_consensus.py` | `TestStockConsensus::*`, vote-count tests | signal_engine cache + `_cached_or_enqueue` |
| `tests/test_4h_digest.py` | `TestGetLastDigestTime::*` | `4h_digest_state.json` path constant |
| `tests/test_forecast_circuit_breaker.py` | `TestForecastFullPathEnabled::*` | `_FORECAST_MODELS_DISABLED` + ticker-accuracy cache |
| `tests/test_metals_loop_pre_sell_cancel.py` | server-exception flow | metals-loop `_loop_page` + snapshot fn |
| `tests/test_seasonality_updater.py` | fetch-failure | `_fetch_hourly_klines` module patch |

### Global state reset fixture (2026-04-19, auto-session)

A global `autouse` fixture in `conftest.py` (`_reset_module_state`) now
resets all HIGH-risk module state before and after every test:

- `agent_invocation`: `_agent_proc`, `_agent_log`, `_agent_start`, etc.
- `signal_engine`: `_adx_cache`, `_last_signal_per_ticker`, `_prev_sentiment`, etc.
- `shared_state`: `_tool_cache`, `_regime_cache`, `_run_cycle_id`, etc.

Reset helpers live in `tests/_state_reset.py`. The module also provides
`reset_all()` for MEDIUM/LOW-risk modules (forecast, logging_config,
api_utils, trigger) — use these in test files that interact with those
modules.

This eliminated 5+ random xdist flakes per run.

### Prior mitigations (2026-04-17, merge 86572817)

Four tests hardened with explicit state resets — see
`docs/plans/2026-04-17-pre-existing-tests.md`:

- `test_consensus.py::TestStockSignalVoteCounts::test_stock_total_applicable`
  — stale assertion 26→27 (signals grew to 43).
- `test_metals_llm_orphan.py::TestJobObjectIntegration::test_start_chronos_uses_popen_in_job`
  — autouse fixture resetting `_chronos_proc` / `_chronos_job` +
  `get_vram_usage` mock (GPU-state leak).
- `test_perception_gate.py::TestAgentInvocationIntegration::test_gate_skips_invocation`
  — inline reset of `_agent_proc` / `_agent_start` / `_agent_timeout` /
  `_agent_log` before the test.

### How to triage a new xdist flake

1. Run in isolation: `pytest <test> -v`. If passes → flake; proceed.
2. Identify the module-level state the test reads (grep the production
   module for `global` declarations or module-scope mutable state).
3. Add an autouse fixture (or inline reset at the top of the test)
   that sets each such variable back to its default.
4. Worst case: the test reads real OS state (GPU VRAM, subprocess
   output, file timestamps). Mock it with `patch("<dotted>.<fn>",
   return_value=<safe>)`.
5. Verify with `pytest -n auto -q` in a clean worktree.

### Future work (tracked in `docs/IMPROVEMENT_BACKLOG.md`)

- ~~**Comprehensive xdist-hygiene pass.**~~ **DONE** (2026-04-19).
  Global autouse fixture + `tests/_state_reset.py` covers all HIGH-risk
  modules. MEDIUM-risk modules have per-function reset helpers.
- **`tests/test_llama_server_job_object.py`.** This file sits
  untracked in the repo root since at least 2026-04-17. It ships
  regression tests for Windows Job Object lifecycle features
  (`popen_in_job`, `close_job`, `kill_orphaned_llama_server`,
  `_kill_orphaned_by_name`) that are NOT implemented in
  `portfolio/llama_server.py` / `portfolio/subprocess_utils.py`.
  Either commit the production feature it tests, or delete the
  file. See `docs/plans/2026-04-17-pre-existing-tests.md` for the
  full triage.

## Parallel Execution

`pytest-xdist` is installed. Use `-n auto` to run across all CPU cores.

**Important**: Tests that write to shared files (e.g., `_PREDICTIONS_FILE`,
`signal_log.jsonl`, `trigger_state.json`) must use `tmp_path` fixture for
isolation. Tests using module-level state (like `trigger.STATE_FILE`) should
patch those paths to `tmp_path` via an `autouse` fixture. See
`test_trigger_edge_cases.py` for the pattern:

```python
@pytest.fixture(autouse=True)
def _isolate_state(tmp_path):
    with mock.patch("portfolio.trigger.STATE_FILE", tmp_path / "state.json"):
        yield
```

```bash
# Auto-detect CPU cores
.venv\Scripts\python.exe -m pytest -n auto --tb=short -q

# Explicit core count
.venv\Scripts\python.exe -m pytest -n 8 --tb=short -q

# Parallel with verbose output
.venv\Scripts\python.exe -m pytest -n auto -v --tb=short
```

## Test Organization

| Directory/Pattern | Coverage |
|-------------------|----------|
| `tests/test_monte_carlo.py` | GBM engine, antithetic variates, price bands (39 tests) |
| `tests/test_monte_carlo_risk.py` | t-copula VaR/CVaR, correlated crash (32 tests) |
| `tests/test_signal_*.py` | Individual signal modules |
| `tests/test_indicators*.py` | Technical indicator calculations |
| `tests/test_portfolio*.py` | Portfolio state, trading logic |
| `tests/test_trigger*.py` | Trigger system |
| `tests/test_dashboard.py` | Dashboard API endpoints (48 tests) |
| `tests/test_http_retry.py` | HTTP retry logic (60 tests) |
| `tests/integration/` | End-to-end (mostly broken — Freqtrade deps) |

## Performance Benchmarks

Monte Carlo module benchmarks (from test suite):
- Single ticker, 10K paths: **< 1 second**
- 5-ticker batch, 10K paths each: **< 5 seconds**
- Portfolio VaR (3 positions, 10K paths): **< 2 seconds**
