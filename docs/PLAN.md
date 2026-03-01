# Monte Carlo Simulation Integration Plan

**Date:** Mar 1, 2026
**Branch:** `feat/monte-carlo`

## Motivation

The system has strong accuracy-driven probability foundations (30 signals, 43K+ outcomes,
`direction_probability()` engine) but lacks **stochastic simulation**. From the quant desk
article, these techniques fill specific gaps in the current system:

| Current capability | Gap MC fills |
|---|---|
| P(up) = 72% point estimate | **Price quantile bands**: 95% CI = $87.2-$93.5 |
| ATR stop at 2x ATR distance | **Stop-loss hit probability**: 15% chance of hit in 24h |
| Kelly fraction from point accuracy | **Expected trade P&L distribution**: full histogram |
| Per-position risk flags | **Portfolio VaR/CVaR**: joint correlated loss at 95% |

## What We Skip (Low Value for This System)

| Technique | Why skip |
|---|---|
| Particle filter (article Part IV) | System recalculates every minute — same effect |
| Agent-based simulation (Part VII) | We make directional bets, not market-making |
| Vine copulas | Simple t-copula is enough for 19 instruments |
| Importance sampling (Part III) | Standard MC with 10K paths is fine (>1% events) |
| GARCH stochastic volatility | ATR already captures recent vol; marginal gain |

## Implementation Batches

### Batch 1: Core GBM Engine + Tests (test-first)
**Files:** `tests/test_monte_carlo.py`, `portfolio/monte_carlo.py`

Tests first:
- GBM path shape and statistics
- Quantile extraction matches analytical solution
- Antithetic variates reduce variance vs. crude MC
- Stop-loss probability computation
- Drift from directional probability

Then implement `MonteCarloEngine`:
- `simulate_paths()` — GBM with antithetic variates
- `price_quantiles()` — percentile bands at horizon
- `probability_below(threshold)` / `probability_above(threshold)`
- `expected_return()` — mean, std, skew of return distribution
- `simulate_ticker(ticker, agent_summary)` — convenience function

Key formulas:
```
S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
mu = (p_up - 0.5) * sigma * sqrt(252)   # drift from directional probability
sigma = atr_pct / 100 * sqrt(252/14)    # annualize 14-period ATR
```

Antithetic variates: for each Z, also simulate -Z (free 50-75% variance reduction).
N_paths = 10,000 (5K pairs). Horizons: 1d, 3d.

### Batch 2: Portfolio VaR with t-Copula + Tests (test-first)
**Files:** `tests/test_monte_carlo_risk.py`, `portfolio/monte_carlo_risk.py`

Tests first:
- t-copula produces correlated returns
- VaR/CVaR computation correct for known distribution
- Single vs multi-position portfolios
- Correlated crash probability > independent crash probability

Then implement `PortfolioRiskSimulator`:
- `simulate_correlated_returns()` — t-copula (v=4) with correlation matrix
- `portfolio_pnl()` — aggregate position-level P&L
- `var(confidence)` / `cvar(confidence)` — Value-at-Risk / Expected Shortfall
- `drawdown_probability(threshold_pct)`
- `compute_portfolio_var(portfolio_state, agent_summary)` — convenience function

Correlation from existing `CORRELATED_PAIRS` in risk_management.py + ATR volatilities.

### Batch 3: Reporting Integration
**Files:** `portfolio/reporting.py`, `config.json`

- Add `monte_carlo` section to compact + tier2 summaries
- Config flag: `monte_carlo.enabled` (default true)
- Config: `monte_carlo.n_paths` (default 10000), `monte_carlo.horizons` ([1, 3])
- Only compute for held positions + focus tickers (not all 19)
- Graceful degradation on failure

### Batch 4: Edge Case Tests + Performance
**Files:** `tests/test_monte_carlo.py` (additions), `tests/test_monte_carlo_risk.py` (additions)

- Zero volatility, single position, no positions, extreme prices
- Config disabled → no MC in summary
- Reporting integration test
- Performance: MC completes in <5s for all tickers

### Batch 5: Docs + Cleanup
**Files:** `docs/SYSTEM_OVERVIEW.md`, `memory/todo.md`

- Document MC module
- Final test suite run
