# Market Health Integration Plan

**Date:** 2026-03-31
**Branch:** `feature/market-health-integration`
**Goal:** Add market-level awareness (breadth, distribution days, FTD, earnings, exposure coaching) to the signal pipeline without increasing Layer 2 Claude Code invocations.

## Problem Statement

The finance-analyzer has 30 signals across 20 instruments, but they're all **instrument-level**. The system answers "what should I do with NVDA?" but never asks "is this a good market to be buying anything in?" This is the "fishing in the right pond" gap.

Additionally:
- Economic calendar dates are hardcoded for 2026-2027 (stale risk)
- No earnings date awareness for 16 US stocks (blind spot)
- No portfolio-level exposure management (fixed 15-30% regardless of market health)

## Critical Constraint: No Extra Claude Code Invocations

The user's Max subscription has usage limits. These new features must NOT add new trigger reasons to `trigger.py`. Instead:

1. **Enrich context** — Add market health data to `agent_summary.json` (Layer 2 reads it when already triggered)
2. **Gate signals** — Earnings proximity gates prevent false BUY consensus → actually REDUCES invocations
3. **Confidence penalties** — Poor market health → lower BUY confidence → fewer consensus triggers → fewer invocations

**Net effect: invocations should DECREASE, not increase.**

## Architecture

```
                    ┌─────────────────────────────┐
                    │   market_health.py (NEW)     │  Runs hourly in post-cycle
                    │  distribution_days, FTD,     │  Cached 1h via shared_state
                    │  breadth_score (0-100)       │  Data: yfinance SPY/QQQ
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   exposure_coach.py (NEW)    │  Synthesizes health + regime
                    │  exposure_ceiling (0.0-1.0)  │  Advisory, not enforcement
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────▼─────────────────────────┐
         │              signal_engine.py (MODIFY)             │
         │  + market health confidence penalty on BUY signals │
         │  + earnings proximity HOLD gate (stocks only)      │
         └─────────────────────────┬─────────────────────────┘
                                   │
         ┌─────────────────────────▼─────────────────────────┐
         │              reporting.py (MODIFY)                 │
         │  + market_health section in agent_summary.json     │
         │  + earnings_proximity per ticker                   │
         │  + exposure_recommendation section                 │
         └───────────────────────────────────────────────────┘
```

## Data Sources (No New API Keys Required)

| Source | Data | Cache TTL | Already Available |
|--------|------|-----------|-------------------|
| yfinance SPY | Daily OHLCV for distribution days, FTD, SMA | 1 hour | Yes (yfinance is a dependency) |
| yfinance QQQ | Daily OHLCV for dual-index tracking | 1 hour | Yes |
| yfinance .calendar | Earnings dates per stock ticker | 24 hours | Yes |
| Existing regime data | From indicators.py detect_regime() | Per-cycle | Yes |
| Existing portfolio state | From portfolio_mgr.py | Per-cycle | Yes |

## New Modules

### 1. `portfolio/market_health.py` — Market Health Score (0-100)

**Distribution Day Tracking (O'Neil method):**
- Fetch SPY and QQQ daily bars from yfinance (30 days)
- Distribution day = price down ≥0.2% on volume ≥ previous day's volume
- Count in rolling 25 trading days
- Stalling day = closes in upper 25% of range on higher volume (churning)
- 4-5 distribution days = caution, 6+ = danger

**Follow-Through Day (FTD) Detection:**
- State machine: `correcting` → `rally_attempt` → `ftd_confirmed` → `confirmed_uptrend`
- Correction = index drops ≥5% from recent high
- Rally attempt starts on first up day after correction
- FTD = day 4+ of rally with ≥1.25% gain on higher volume than previous day
- Failure = index undercuts rally low within 10 days of FTD

**Breadth Score Computation (0-100):**
- Component 1 (25 pts): Distribution day severity (0 days=25, 3=15, 5=5, 7+=0)
- Component 2 (25 pts): FTD state (confirmed_uptrend=25, ftd_confirmed=20, rally_attempt=10, correcting=0)
- Component 3 (20 pts): SPY vs 200-SMA (above=20, below=0)
- Component 4 (15 pts): SPY vs 50-SMA (above=15, below=0)
- Component 5 (15 pts): 10-day return direction (>2%=15, 0-2%=10, -2-0%=5, <-2%=0)

**API:**
```python
def compute_market_health(force=False) -> dict:
    """Returns cached market health snapshot. Refreshes hourly."""
    return {
        "score": 72,                    # 0-100 composite
        "zone": "healthy",              # healthy/caution/danger
        "distribution_days_spy": 2,
        "distribution_days_qqq": 3,
        "ftd_state": "confirmed_uptrend",
        "spy_above_200sma": True,
        "spy_above_50sma": True,
        "spy_return_10d_pct": 1.5,
        "components": {...},
        "updated_at": "2026-03-31T14:00:00Z",
    }
```

### 2. `portfolio/earnings_calendar.py` — Earnings Proximity Gate

**Fetching:**
- Use yfinance `Ticker.calendar` for each stock ticker
- Cache 24h per ticker (earnings dates don't change intraday)
- Only fetch for STOCK_SYMBOLS (16 tickers), not crypto/metals

**Gate Logic:**
- If ticker has earnings within 2 calendar days → force HOLD
- Returns proximity info for agent_summary enrichment

**API:**
```python
def get_earnings_proximity(ticker: str) -> dict | None:
    """Returns earnings proximity info, or None if no upcoming earnings."""
    return {
        "earnings_date": "2026-04-15",
        "days_until": 1,
        "gate_active": True,           # True = force HOLD
        "timing": "after_close",       # before_open / after_close / unknown
    }

def should_gate_earnings(ticker: str) -> bool:
    """True if ticker has earnings within gate window (2 days)."""
```

### 3. `portfolio/exposure_coach.py` — Portfolio Exposure Recommendation

**Synthesis:**
- Reads market_health score + current regime + portfolio concentration
- Outputs exposure_ceiling multiplier (0.0 to 1.0)

**Logic:**
```
if market_health.zone == "danger":
    ceiling = 0.3  # max 30% of normal allocation
elif market_health.zone == "caution":
    ceiling = 0.6
else:  # healthy
    ceiling = 1.0

# Regime adjustment
if regime == "trending-down":
    ceiling *= 0.7
elif regime == "high-vol":
    ceiling *= 0.8

# Floor
ceiling = max(ceiling, 0.2)
```

**API:**
```python
def get_exposure_recommendation() -> dict:
    return {
        "exposure_ceiling": 0.6,
        "rationale": "Market caution (3 distribution days) + trending-down regime",
        "market_health_zone": "caution",
        "regime": "trending-down",
        "new_entries_allowed": True,    # False if danger + trending-down
        "bias": "defensive",           # growth / neutral / defensive
    }
```

## Modifications to Existing Files

### 4. `portfolio/signal_engine.py` — Confidence Penalty + Earnings Gate

**Market health confidence penalty (in `generate_signal()`):**
```python
# After weighted consensus computation, before returning:
health = get_market_health_cached()
if health and action == "BUY":
    score = health.get("score", 50)
    if score < 30:
        confidence *= 0.6   # harsh penalty
    elif score < 50:
        confidence *= 0.8   # moderate penalty
    # score >= 50: no penalty
```

**Earnings gate (in `generate_signal()`):**
```python
# For stock tickers only:
if ticker in STOCK_SYMBOLS and should_gate_earnings(ticker):
    action = "HOLD"
    # Add earnings_gate flag to extra_info
```

**Effect on invocations:** When market health is poor, BUY confidence drops → fewer tickers reach consensus threshold → fewer triggers. When earnings are near, tickers are gated to HOLD → no false BUY trigger. NET REDUCTION in invocations.

### 5. `portfolio/reporting.py` — Agent Summary Enrichment

Add three new sections to `agent_summary.json`:
1. `market_health`: Full health snapshot (after macro section)
2. Per-ticker `earnings_proximity`: In signal extra data
3. `exposure_recommendation`: Exposure coach output

These are **context-only** — Layer 2 reads them when already invoked. No trigger impact.

### 6. `portfolio/main.py` — Post-Cycle Hook

Add to `_run_post_cycle()`:
```python
# Market health refresh (hourly, self-checking)
try:
    from portfolio.market_health import maybe_refresh_market_health
    maybe_refresh_market_health()
except Exception as e_mh:
    logger.warning("Market health refresh failed: %s", e_mh)
```

### 7. `dashboard/app.py` — New API Endpoint

Add `/api/market-health` endpoint returning the cached market health data.

## Implementation Batches

### Batch 1: `market_health.py` + `tests/test_market_health.py`
**Files:** 2 new
- Core distribution day computation
- FTD state machine
- Breadth score computation
- Hourly caching via shared_state
- Tests: distribution day counting, FTD transitions, score computation, caching

### Batch 2: `earnings_calendar.py` + `tests/test_earnings_calendar.py`
**Files:** 2 new
- yfinance calendar fetching with 24h cache
- Proximity gate logic
- Tests: gate window, cache behavior, ticker filtering

### Batch 3: `exposure_coach.py` + `tests/test_exposure_coach.py`
**Files:** 2 new
- Synthesis of market health + regime
- Exposure ceiling computation
- Tests: all zone/regime combinations, floor enforcement

### Batch 4: Integration (signal_engine.py, reporting.py, main.py)
**Files:** 3 modified
- signal_engine.py: confidence penalty + earnings gate
- reporting.py: market_health + earnings + exposure sections
- main.py: post-cycle hook
- Run existing tests to verify no regressions

### Batch 5: Dashboard + cleanup
**Files:** 1 modified (dashboard/app.py)
- /api/market-health endpoint
- Run full test suite
- Commit session progress

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| yfinance rate limits | Hourly cache (not every 60s), graceful degradation on failure |
| False distribution day counts | Use standard O'Neil thresholds, well-tested in literature |
| Over-aggressive gating | Earnings gate only 2 days, market health penalty caps at 0.6x |
| Breaking existing signals | Penalties are multiplicative on final confidence, don't change individual signal logic |
| Extra API calls | yfinance SPY/QQQ = 2 calls/hour, earnings = 16 calls/day — negligible |

## What We're NOT Doing (Deferred)

- **institutional-flow-tracker**: Requires 13F data source, complex to automate
- **walk-forward-validation**: ML retraining is a separate initiative
- **signal-postmortem**: Interesting but orthogonal to this feature set
- **Dynamic econ calendar API**: FMP requires API key we don't have; hardcoded dates are fine for 2026-2027
- **TraderMonty CSV**: Would be nice but yfinance-based breadth is sufficient

## Success Criteria

1. Market health score computes correctly from SPY/QQQ data
2. Distribution days and FTD state machine track accurately
3. Earnings gate prevents BUY signals within 2 days of earnings
4. Market health confidence penalty reduces false triggers in bearish markets
5. All new code has tests, all existing tests still pass
6. Layer 2 invocation count does NOT increase (should decrease)
7. Agent summary JSON is enriched with market-level context
