# Adversarial Review: portfolio-risk (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08
Confidence scores included per finding.

---

## CRITICAL

### CR1. Trade guards cooldown: timezone-aware vs naive datetime comparison [95% confidence]
**File**: `portfolio/trade_guards.py:89`

`record_trade()` stores timestamps via `datetime.now(UTC).isoformat()` (timezone-aware:
`"2026-04-08T10:00:00+00:00"`). `check_overtrading_guards()` at line 89 calls
`datetime.fromisoformat(last_trade_str)` without ensuring timezone-awareness.

- **Python 3.10 and earlier**: `fromisoformat` can't parse `+00:00` suffix → `ValueError`
  caught by `except (ValueError, TypeError)` at line 110 → **cooldown silently bypassed**.
- **Python 3.11+**: Works correctly, but behavior is version-dependent and fragile.

Same pattern at lines 146 and 216.

**Fix**: After `fromisoformat`, normalize:
```python
if last_trade.tzinfo is None:
    last_trade = last_trade.replace(tzinfo=UTC)
```

### CR2. All trade guards are severity="warning" — `should_block_trade()` always returns False [100% confidence]
**File**: `portfolio/trade_guards.py:65, 278-290`

Every `warnings.append(...)` in the file uses `"severity": "warning"`. The `"block"` value
documented in the API contract is never emitted. `should_block_trade()` (line 290) returns
`True` only when severity is `"block"` — meaning it **unconditionally returns False**.

The trade guards are advisory only. No code in the entire codebase ever sets `severity: "block"`.
If Layer 2 relies on `should_block_trade()` as a go/no-go gate, the gate is permanently open.

### CR3. Drawdown check cash-only fallback when agent_summary empty [85% confidence]
**File**: `portfolio/risk_management.py:81-89`

When portfolio has holdings but `agent_summary.json` is empty/stale:
- `load_json` returns `{}` (falsy) → fallback to `cash_sek`
- Portfolio with 400K in holdings + 100K cash → reports 100K value → 80% drawdown
- OR: stale high prices → understated drawdown → circuit breaker doesn't fire

No warning logged on this fallback path.

---

## HIGH

### H-A1. `load_state()`/`save_state()` bypass the `update_state()` lock [88% confidence]
**File**: `portfolio/portfolio_mgr.py:151-157`

`update_state()` holds a per-file lock for read-modify-write. But the public `load_state()`
and `save_state()` are lock-unaware. `main.py` at lines 393 and 599 uses the pattern:
`state = load_state()` → 200 lines of processing → `save_state(state)`.
Two concurrent callers can overwrite each other's changes.

### H-A2. Kelly sizing uses cash_sek only, not total portfolio value [87% confidence]
**File**: `portfolio/kelly_sizing.py:241-243, 286-287`

`max_alloc = cash_sek * alloc_frac`. Position sizes are based on cash only, ignoring
mark-to-market value of open positions. A portfolio with 800K total but only 200K cash
sizes new positions against the 200K base, ignoring that total exposure is already 600K.
Concentration limits are cash-relative, not portfolio-relative.

### H-A3. Sortino ratio uses biased downside deviation denominator [85% confidence]
**File**: `portfolio/equity_curve.py:244-250`

Standard Sortino: `downside_dev = sqrt(sum(r^2) / n_total)` (all periods in denominator).
Implementation: `downside_dev = sqrt(sum(r^2) / n_downside)` (only downside periods).

For a 60% win rate, this inflates downside deviation by ~26%, making the strategy look
worse than it is. BUG-177 was supposed to fix this but only addressed the input variable.

### H-A4. Concentration check uses hypothetical max allocation, not actual trade size [82% confidence]
**File**: `portfolio/risk_management.py:584-585`

`proposed_alloc = min(total_value * alloc_pct, cash)` is a hypothetical max, not the actual
proposed trade size. Also doesn't prevent cumulative buys that each pass individually but
collectively exceed the threshold.

### H-A5. `PortfolioRiskSimulator` vs `compute_portfolio_var` filter inconsistency [81% confidence]
**File**: `portfolio/monte_carlo_risk.py:211 vs 434`

`PortfolioRiskSimulator.__init__` filters `shares != 0` (retains negatives).
`compute_portfolio_var` filters `shares <= 0` (skips negatives).
If a holding ever has negative shares (rounding artifact), the simulator includes it
while the standalone function doesn't — producing inconsistent VaR estimates.

---

## MEDIUM

### M-A1. Kelly metals: all-wins or all-losses scenario loses real signal data [80% confidence]
**File**: `portfolio/kelly_metals.py:115-116`

When all trades are wins, `avg_loss = 0.0`. The guard at line 198 correctly falls through
to defaults, but discards the valid win data. Uses hardcoded defaults even when there IS
relevant outcome data.

### M-A2. Drawdown check reads history with raw `open()` — violates atomic I/O rule [80% confidence]
**File**: `portfolio/risk_management.py:95`

CLAUDE.md rule #4: "Never raw `json.loads(open(...).read())`". Lines 99-108 use raw
`open()` + per-line `json.loads()`. Concurrent append by `atomic_append_jsonl` can
produce a partial-line read.

### M-A3. Non-functional guard detection logic is inverted [80% confidence]
**File**: `portfolio/trade_guards.py:262-270`

The C4 detection fires when BOTH no recorded trades AND no warnings. Can false-positive
when old timestamps are pruned after 24h.

---

## LOW

### L-A1. `total_sek` field semantics ambiguous — possible fee double-counting [80% confidence]
**File**: `portfolio/equity_curve.py:339`

`price_per_share = total_sek / shares` — if `total_sek` includes fees, P&L is overstated.
`fee_sek` is tracked separately and added back, potentially double-counting.

### L-A2. Kelly ATR default reads from wrong dict level [80% confidence]
**File**: `portfolio/kelly_sizing.py:266-274`

Reads `ticker_data.get("atr_pct", 1.5)` from top-level. CLAUDE.md documents `atr_pct`
under `extra`. Falls back to 1.5% for all tickers regardless of actual volatility.
For BTC (~3% ATR), this understates avg_win by 2x, producing a Kelly fraction ~2x too high.

---

## Cross-Critique: Claude Direct vs Agent

### Issues agent found that Claude missed:
1. **CR1**: Timezone-aware/naive datetime comparison in trade_guards — complete miss
2. **CR2**: `should_block_trade()` always returns False — complete miss (critical design gap)
3. **H-A2**: Kelly cash-only denominator — missed
4. **H-A3**: Sortino ratio biased denominator — missed
5. **L-A2**: Kelly ATR wrong dict level — missed

### Issues Claude found that agent confirmed/expanded:
1. **H6/CR3**: Drawdown bypass on corrupt file — both found, agent added more detail
2. **M11/H-A1**: Portfolio lock bypass — Claude rated MEDIUM, agent correctly escalated to HIGH
3. **H7**: Stale price fallback in _compute_portfolio_value — confirmed

### Issues Claude found that agent didn't cover:
1. **M5**: ATR stop floor at 1% — not mentioned by agent
2. **M6**: Monte Carlo 2000 paths insufficient — not mentioned

### Net assessment:
The agent review of portfolio-risk is **stronger** than my independent review for this
subsystem. The trade_guards findings (CR1, CR2) are especially valuable — they represent
a complete failure of the overtrading prevention system that I entirely missed.
