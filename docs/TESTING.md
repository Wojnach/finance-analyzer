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

## Test Suite Stats (as of 2026-03-01)

| Metric | Value |
|--------|-------|
| Total tests | ~3,168 |
| Passing | 3,142 |
| Pre-existing failures | 26 |
| Sequential runtime | **16 min 12 sec** |
| Parallel runtime (`-n auto`) | **~4-5 min** (estimated) |
| Test files | ~85 |

## Pre-existing Failures (26 — not regressions)

These failures existed before the current session and are not caused by recent changes:

| Count | Test File | Cause |
|-------|-----------|-------|
| 15 | `tests/integration/test_strategy.py` | Missing `ta_base_strategy` module (Freqtrade) |
| 4 | `tests/test_consensus.py` | MIN_VOTERS threshold mismatch |
| 2 | `tests/test_forecast_config.py` + `test_forecast_accuracy_gating.py` | Kronos config state |
| 3 | `tests/test_portfolio.py` | Signal/trigger test mismatches |
| 1 | `tests/test_portfolio.py::test_full_report` | Subprocess integration |
| 1 | `tests/test_forecast_accuracy_gating.py` | Kronos disabled default |

## Parallel Execution

`pytest-xdist` is installed. Use `-n auto` to run across all CPU cores.

**Important**: Tests that write to shared files (e.g., `_PREDICTIONS_FILE`, `signal_log.jsonl`)
must use `tmp_path` fixture for isolation. Most tests already do this.

```bash
# Auto-detect CPU cores
.venv\Scripts\python.exe -m pytest -n auto --tb=short -q

# Explicit core count
.venv\Scripts\python.exe -m pytest -n 8 --tb=short -q

# Parallel with verbose output
.venv\Scripts\python.exe -m pytest -n auto -v --tb=short
```

**Known issue**: On Windows with Git Bash, `pytest -n auto` may fail with
"bringing up nodes... no tests ran" if the working directory has path issues.
Run from cmd.exe or PowerShell if this happens:

```cmd
cd Q:\finance-analyzer
.venv\Scripts\python.exe -m pytest -n auto --tb=short -q
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
