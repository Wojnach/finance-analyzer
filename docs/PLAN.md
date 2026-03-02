# Plan: Dashboard Auto-Improvement Session

## Problems Found

### Bugs (Critical)
1. **BUG-1: Signal count labels stale** — Heatmap tab says "25-Signal" but system has 30 signals. Enhanced group header says "(12-25)" should be "(12-30)".
2. **BUG-2: Core signals list incomplete** — `app.py` `core_signals` list missing `custom_lora`. Has 10 items, should have 11.
3. **BUG-3: Disabled signals shown as dots** — Overview cards show ML, LoRA, Funding as signal dots even though they are DISABLED (always HOLD). Adds visual noise.

### Missing Features (High Value)
4. **FEAT-1: No warrant portfolio display** — Warrant holdings (`portfolio_state_warrants.json`) not visible. User holds leveraged products (MINI-SILVER 5x) — critical P&L visibility gap.
5. **FEAT-2: No Monte Carlo risk display** — Price bands, stop-loss probability, VaR/CVaR computed but not surfaced.
6. **FEAT-3: No weighted confidence shown** — `weighted_confidence` (accuracy-adjusted) not displayed on cards. Only raw vote counts visible.
7. **FEAT-4: No regime badge on cards** — `regime` field available per ticker but not shown.
8. **FEAT-5: Module failures not in Health tab** — `get_health_summary()` returns `module_failures` but Health tab ignores it.
9. **FEAT-6: No holdings in header** — Header shows Patient value + cash but not what's held at a glance.

## Changes (4 batches, 2 files + tests)

### Batch 1: Fix bugs in app.py + index.html (BUG-1,2,3) + FEAT-3,4
**Files:** `dashboard/app.py`, `dashboard/static/index.html`

1. `app.py` line 290: Add `custom_lora` to `core_signals` list
2. `app.py` line 279 comment: Fix "25-signal" → "30-signal"
3. `index.html` line 581: Fix "25-Signal Heatmap" → "30-Signal Heatmap"
4. `index.html` line 1160: Fix "(1-11)" core signals label
5. `index.html` line 1173: Fix "(12-25)" → "(12-30)" for enhanced signals
6. `index.html` votes() function: Skip disabled signals (ml, funding, custom_lora)
7. `index.html` signal cards: Add weighted_confidence display + regime badge

### Batch 2: Add warrant API + display + header holdings (FEAT-1, FEAT-6)
**Files:** `dashboard/app.py`, `dashboard/static/index.html`

1. `app.py`: Add `/api/warrants` endpoint reading `portfolio_state_warrants.json`
2. `index.html`: Add warrant holdings panel in Overview tab
3. `index.html`: Show current holdings in header next to portfolio values

### Batch 3: Add risk display + health module failures (FEAT-2, FEAT-5)
**Files:** `dashboard/app.py`, `dashboard/static/index.html`

1. `app.py`: Add `/api/risk` endpoint reading Monte Carlo + VaR data from `agent_summary_compact.json`
2. `index.html`: Add risk panel to Overview tab (price bands, stop probability, VaR)
3. `index.html`: Show module_failures in Health tab

### Batch 4: Tests + verify
**Files:** `tests/test_dashboard.py`

1. Add tests for `/api/warrants` endpoint
2. Add tests for `/api/risk` endpoint
3. Run full test suite, merge, push, restart

## What could break

- All existing endpoints unchanged — pure additions
- Warrant/risk displays handle missing data gracefully (show "no data")
- Adding `custom_lora` to core signals list won't break heatmap (already in `_votes` dict)

## Execution order

1. Batch 1: Bug fixes + weighted confidence + regime
2. Batch 2: Warrants + header holdings
3. Batch 3: Risk + health module failures
4. Batch 4: Tests + merge + push + restart
