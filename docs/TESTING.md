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
- `signal_engine`: `_adx_cache`, `_last_signal_per_ticker`, `_prev_sentiment`,
  and (since 2026-05-02) `_ic_data_cache` + `_macro_window_cache`.
- `shared_state`: `_tool_cache`, `_regime_cache`, `_run_cycle_id`, etc.
- (2026-05-02) `data_collector` circuit breakers: `alpaca_cb`,
  `binance_spot_cb`, `binance_fapi_cb` — reset to CLOSED via the new
  `CircuitBreaker.reset()` method so a test that fails 5+ Alpaca calls
  doesn't leave the breaker OPEN for the next test on the same worker.

### Residual flakes that DO NOT respond to module-state resets (2026-05-02)

Some `test_consensus.py` tests using `_NO_PENALTIES` config (notably
`test_stock_buy_with_3_voters`, `test_stock_sell_with_3_voters`,
`test_all_stock_tickers_use_3_voter_threshold`,
`test_crypto_buy_with_3_voters`,
`test_flip_direction_above_threshold_votes`,
`TestStockSignalVoteCounts::test_stock_total_applicable`,
`TestStockSignalVoteCounts::test_crypto_total_applicable`) fail
when run together OR after other consensus tests, even with all
state-reset fixtures wired correctly.

Root cause: signals get force-HOLD'd by the per-ticker accuracy gate
read from `data/accuracy_cache.json` — a real production file in the
repo, not module state. The contents of that file change as the
production loop writes accuracy snapshots. Whether RSI/MACD/BB are
considered "above 45% accuracy on stocks" depends on what the file
currently says, not on the test's setup. Until the consensus tests
mock `data/accuracy_cache.json` themselves (or the production gate
takes a config override), they will be data-coupled flakes.

Recipe to skip them in CI / local runs:

```bash
.venv/Scripts/python.exe -m pytest tests/ -n auto \
    --deselect 'tests/test_consensus.py::TestStockConsensus::test_stock_buy_with_3_voters' \
    --deselect 'tests/test_consensus.py::TestStockConsensus::test_stock_sell_with_3_voters' \
    --deselect 'tests/test_consensus.py::TestStockConsensus::test_all_stock_tickers_use_3_voter_threshold' \
    --deselect 'tests/test_consensus.py::TestCryptoConsensus::test_crypto_buy_with_3_voters' \
    --deselect 'tests/test_consensus.py::TestSentimentHysteresis::test_flip_direction_above_threshold_votes' \
    --deselect 'tests/test_consensus.py::TestStockSignalVoteCounts::test_stock_total_applicable' \
    --deselect 'tests/test_consensus.py::TestStockSignalVoteCounts::test_crypto_total_applicable'
```

Proper fix (deferred): rewrite the consensus tests to mock
`data/accuracy_cache.json` via tmp_path and patch the loader, the same
way `metals_swing_trader` tests already isolate state files.

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
| `tests/test_dashboard_legacy_route.py` | /legacy fallback route during the 2026-05-03 mobile rollout (5 tests) |
| `tests/test_dashboard_static_assets.py` | New mobile-dashboard CSS/JS/PWA asset paths (20 tests) |
| `tests/test_dashboard_skeleton.py` | New `index.html` skeleton integrity — viewport-fit, manifest, Chart.js order, bottom-nav routes, bottom-sheet shell (15 tests) |
| `tests/test_dashboard_frontend.py` | Frozen content-string asserts on the legacy file (`index_legacy.html`) — delete after the /legacy rollout window closes |
| `tests/test_http_retry.py` | HTTP retry logic (60 tests) |
| `tests/integration/` | End-to-end (mostly broken — Freqtrade deps) |

## Performance Benchmarks

Monte Carlo module benchmarks (from test suite):
- Single ticker, 10K paths: **< 1 second**
- 5-ticker batch, 10K paths each: **< 5 seconds**
- Portfolio VaR (3 positions, 10K paths): **< 2 seconds**

## Manual phone smoke test (mobile dashboard, 2026-05-03)

The redesigned dashboard at `/` is built mobile-first; existing automated
tests verify routing + asset presence + skeleton integrity, but visual
behaviour on a real phone needs eyeballs. Run this checklist before
merging any mobile-affecting PR:

1. Open the dashboard in Chrome devtools mobile-emulator (390×844 iPhone).
2. **Bottom-nav** appears with 4 items: Home / Decisions / Signals / More.
3. **Home** renders P&L card, positions strip (horizontal scroll snaps
   per card), consensus chips, latest decision, system pulse dots.
4. Tap a position card → navigates to /#signals/<ticker> (heatmap with
   that ticker pre-selected).
5. Tap the latest decision card → /#decisions list view.
6. Tap a decision card → /#decisions/<ts> detail view; tap ← Decisions
   to go back.
7. **Signals** view: sub-tab bar (Heatmap / Accuracy / History). Heatmap
   shows transposed grid with sticky leftmost column. Long-press a cell
   → bottom-sheet drill opens. Tap the backdrop → sheet closes.
8. **More** menu lists Health, Messages, Metals, GoldDigger, Equity,
   Settings. Each navigates to its full view; bottom-nav still
   highlights "More".
9. **Settings**: theme toggle flips light↔dark. Pause toggles polling.
   "Refresh now" forces re-fetch. Legacy view link opens /legacy
   (existing single-file dashboard, fully functional).
10. **PWA**: open Chrome → ⋮ → "Install Portfolio". App icon appears on
    home screen. Open it standalone (no browser chrome). First launch
    redirects to CF Access SSO (PWA cookie jar isolated from Safari).
11. **Service worker**: in devtools Application tab, SW shows
    `pi-shell-v1-2026-05-03` controlling the page. Disconnect network
    → reload → cached shell renders with offline badge for /api/*.
12. **Visibility-aware polling**: switch to another tab for 30s, switch
    back. Network panel shows polling pauses while hidden and
    re-fires once on return.

Failures during this checklist are not test failures — log them as
follow-up issues and assess whether to roll back or patch.

## Mutation Testing (mutmut)

Claude writes both the production code AND the tests in this repo. A bug
in the code can be papered over by an equally-confused test that asserts
the wrong thing — and CI stays green. Mutation testing closes that loop:
mutmut modifies the code under test (flips a `<`, deletes a line, swaps a
sign) and re-runs the suite. If a mutated version still passes, the test
is weak — it claims to verify the behaviour but doesn't actually catch
the bug the mutation introduced. A surviving mutant is unambiguous
evidence of a missing assertion.

Run a single-module mutation pass (scoped, slow — 10–30 min per module):

```bash
.venv/Scripts/python.exe scripts/run_mutation_test.py --module portfolio/signal_engine.py
```

Inspect any survivors after the run finishes:

```bash
mutmut results                  # list bucket counts + IDs
mutmut show <mutant_id>         # show the diff mutmut applied
mutmut apply <mutant_id>        # apply the diff to your worktree to study it
                                # (revert with: git checkout -- <file>)
```

Pilot scope: `signal_engine.py`, `risk_management.py`, `portfolio_mgr.py`
(see `[tool.mutmut]` in `pyproject.toml`).

Threshold policy:

- **Today (2026-05-10):** 50% kill-rate gate — a survivor budget,
  enforced by `scripts/run_mutation_test.py` exit code.
- **Target by 2026-06-01:** 80% kill rate. Tighten the threshold flag in
  CI as test coverage hardens.

Full-suite mutmut runs are slow (10–30 min per module, single-threaded
inside each pytest invocation). Run scoped during PR review of high-risk
changes; CI runs the full pilot scope nightly only.

## Property-Based Tests (Hypothesis)

Properties are universal truths about a function — invariants that must
hold for every valid input ("after a buy then a sell of the same size,
cash returns to its original value modulo fees", "VaR is monotone in
confidence level"). Hypothesis generates thousands of random inputs,
including pathological edge cases (empty lists, NaN, INT_MAX, unicode),
and shrinks any failing input to a minimal reproducer. Where unit tests
check 3 examples you thought of, properties check 10,000 the library
thought of.

Run the property suite with statistics:

```bash
.venv/Scripts/python.exe -m pytest tests/test_property_invariants.py -v --hypothesis-show-statistics
```

Adding a new property is one decorator + one assertion:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1))
def test_portfolio_value_non_negative(prices):
    # Invariant: a long-only portfolio with positive prices has positive value.
    assert portfolio_value(prices) > 0
```

A failing property is either a real bug OR a wrong property. Investigate
the shrunk counterexample before "fixing" by relaxing the property — the
whole point of Hypothesis is that it finds inputs you wouldn't have
written by hand. Loosening the invariant defeats the test.

## Type Checking (mypy --strict)

```bash
.venv/Scripts/python.exe -m mypy --config-file mypy.ini \
    portfolio/signal_engine.py \
    portfolio/portfolio_mgr.py \
    portfolio/risk_management.py \
    portfolio/loop_contract.py
```

Pilot scope: the four modules above. Expand the gate by adding a new
`[mypy-portfolio.<module>]` section to `mypy.ini` per module — keep the
rest of the codebase unchecked until each module is hardened.

Suppression policy: every `# type: ignore[…]` comment must include a
date and rationale, e.g.

```python
result = legacy_thing()  # type: ignore[no-untyped-call]  # 2026-05-10: third-party stub missing, see issue #NNN
```

Bare `# type: ignore` (no error code, no comment) is forbidden — it
silences future errors too and rots the gate.
