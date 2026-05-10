# PLAN — Test-Trust Hardening (mypy + Hypothesis + mutmut + invariants)

**Date:** 2026-05-10
**Branch:** `test-trust-2026-05-10`
**Worktree:** `.worktrees/test-trust-2026-05-10`

## Why

User concern: Claude writes Python code AND its tests. Bad code can pass bad tests
written by the same agent. Switching to Rust would catch ~20% of bug classes
(import/typo/dead-code) while costing 22 missing signals + 5994 lost tests.

This plan attacks the trust problem at its root with four mechanisms that work
in Python today and don't depend on Claude's tests being correct:

1. **mypy --strict** on hot files — catch the 20% Rust would catch.
2. **Hypothesis property tests** — assert universal invariants Claude can't fake.
3. **mutmut config** — kill weak tests (mutated code → tests must fail).
4. **Runtime invariants in `loop_contract.py`** — production assertions that
   fire regardless of what the test suite did or didn't check.

## What

### Batch 1 — mypy --strict pilot

| File | Action |
|------|--------|
| `mypy.ini` (NEW) | strict pilot scoped to 4 modules below |
| `portfolio/signal_engine.py` | type fixes / `# type: ignore[…]` with TODO refs |
| `portfolio/portfolio_mgr.py` | type fixes |
| `portfolio/risk_management.py` | type fixes |
| `portfolio/loop_contract.py` | type fixes |
| `requirements-dev.txt` (NEW) | `mypy`, `hypothesis`, `mutmut` |
| `docs/TESTING.md` (UPDATE) | section: how to run mypy/hypothesis/mutmut |

**Goal:** mypy --strict passes on these 4 modules. Document ignores with
inline rationale comments.

### Batch 2 — Hypothesis property tests

| File | Properties |
|------|-----------|
| `tests/test_property_invariants.py` (NEW) | <ul><li>`portfolio_total_equals_cash_plus_holdings_value`</li><li>`atomic_jsonl_round_trip`</li><li>`atomic_write_no_tmp_residue`</li><li>`signal_voting_deterministic`</li><li>`risk_position_size_bounds`</li></ul> |

These are PROPERTIES — universal truths. Claude cannot satisfy them with a
wrong implementation without the property catching the wrongness on a random
counterexample.

### Batch 3 — mutmut config

| File | Action |
|------|--------|
| `pyproject.toml` (UPDATE or NEW) | `[tool.mutmut]` section: paths-to-mutate = signal_engine.py, risk_management.py, portfolio_mgr.py, loop_contract.py |
| `scripts/run_mutation_test.py` (NEW) | runs mutmut, parses surviving mutants, exits 1 if kill rate < 70% |
| `docs/TESTING.md` (UPDATE) | how to run + interpret mutation tests |

NOT running full mutation in this batch (slow). Wire the gate. A small scoped
trial run validates the script.

### Batch 4 — Expand `loop_contract.py` runtime invariants

| File | New invariants |
|------|---------------|
| `portfolio/loop_contract.py` (UPDATE) | <ul><li>`check_portfolio_arithmetic()` — cash ≥ 0; total = cash + Σ(qty × last_price)</li><li>`check_atomic_write_residue()` — orphaned `.tmp` files older than 5 min → WARNING</li><li>`check_journal_uniqueness()` — last 50 layer2 entries: no duplicate trigger_id</li></ul> |
| `tests/test_loop_contract_invariants.py` (NEW) | one test per new invariant: passing + failing case |

Wire new invariants into `verify_contract()` so they run every cycle.

## Risks

- **mypy fixes** could change call signatures. Mitigation: only annotate, don't refactor.
- **Property tests** that catch real bugs in production code. Treat as feature.
- **Loop_contract additions** could add latency. Cap atomic_write_residue scan at
  100 files / 200ms timeout.
- **mutmut threshold** — first run will likely fail. Set initial threshold
  permissively (50%), document target trajectory.

## Execution order

1. Batch 1 (mypy) — independent
2. Batch 2 (hypothesis) — independent
3. Batch 3 (mutmut) — independent
4. Batch 4 (loop_contract invariants) — depends on Batch 1 type fixes
5. Tests + Codex review + merge + push

## Verify

After each batch: targeted pytest. Final: full suite + mypy + property tests.
