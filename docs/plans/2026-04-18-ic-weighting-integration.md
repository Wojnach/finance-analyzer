# IC-Based Signal Weighting Integration Plan

**Author**: After-Hours Research Agent, 2026-04-18  
**Priority**: HIGH (identified as #1 quant research improvement)  
**Estimated effort**: 3-5 hours implementation + testing  
**Data**: `data/ic_analysis_2026-04-18.json`, `data/regime_analysis_2026-04-18.json`

## Problem Statement

The current weighting scheme in `_weighted_consensus()` uses **directional accuracy** (binary correct/wrong) as the primary signal weight. This misses a critical dimension: **return magnitude prediction**.

A signal can be 55% accurate but IC=0.02 (catches only noise-level moves), or 52% accurate but IC=0.15 (catches big moves). The system currently treats both equally.

### Evidence

Live IC computation (2026-04-18 00:11 CET) reveals three categories of misweighted signals:

**Phantom Performers** — high accuracy, zero IC:
| Signal | Accuracy | IC (1d) | Samples | Issue |
|--------|----------|---------|---------|-------|
| calendar | 58.9% | 0.0000 | 7,470 | Always BUY. Accuracy = market drift. |
| econ_calendar | 62.6% | 0.0000 | 5,298 | Always BUY. Same phantom pattern. |
| fear_greed | 58.6% | 0.0000 | 10,410 | Zero IC despite high accuracy. |
| claude_fundamental | 61.9% | -0.020 | 10,398 | Near-zero IC. Directionally right but no magnitude info. |

**Genuinely Good** — high accuracy AND high IC:
| Signal | Accuracy | IC (1d) | ICIR | Samples |
|--------|----------|---------|------|---------|
| ministral | 58.6% | 0.094 | 0.079 | 6,068 |
| credit_spread_risk | 55.0% | 0.201 | 0.521 | 795 |
| mean_reversion | 52.7% | 0.068 | 0.296 | 23,264 |
| momentum | 53.6% | 0.063 | 0.333 | 8,400 |
| candlestick | 50.4% | 0.089 | 0.220 | 12,199 |

**Suppressed Contrarians** — below accuracy gate, high |IC|:
| Signal | Accuracy | IC (1d) | ICIR | Samples |
|--------|----------|---------|------|---------|
| ml | 41.7% | -0.321 | 0.169 | 1,729 |
| futures_flow | 35.4% | -0.227 | -0.499 | 1,483 |

## Architecture

### Integration Point

File: `portfolio/signal_engine.py`  
Function: `_weighted_consensus()` (line ~1241)  
Insert after: line 1519 (accuracy weight computed) or line 1524 (after horizon adjustment)

### Proposed Weight Chain

Current:
```
weight = accuracy × regime_mult × horizon_mult × crisis_adj × norm_weight × activity_cap × correlation_penalty × bias_penalty
```

Proposed:
```
weight = accuracy × ic_mult × regime_mult × horizon_mult × crisis_adj × norm_weight × activity_cap × correlation_penalty × bias_penalty
```

Where `ic_mult` is a multiplicative adjustment based on the signal's Information Coefficient.

### IC Multiplier Formula

```python
# IC-based weight multiplier
# IC > 0: signal predicts return magnitude correctly → boost
# IC ≈ 0: signal has no magnitude predictive power → slight penalty
# IC < 0: signal is contrarian → handled by accuracy gate, no additional action
_IC_ALPHA = 2.0  # IC sensitivity parameter
_IC_MULT_FLOOR = 0.6  # minimum IC multiplier (don't zero out)
_IC_MULT_CAP = 1.5  # maximum IC multiplier (don't let IC dominate)
_IC_MIN_SAMPLES = 100  # need enough data for IC to be meaningful
_IC_STABILITY_MIN = 0.10  # minimum |ICIR| to trust IC estimate

def _compute_ic_mult(ic: float, icir: float, samples: int) -> float:
    """Compute IC-based weight multiplier."""
    if samples < _IC_MIN_SAMPLES:
        return 1.0  # not enough data
    if abs(icir) < _IC_STABILITY_MIN:
        return 1.0  # IC estimate is unstable
    raw = 1.0 + _IC_ALPHA * ic  # IC=0.10 → 1.20, IC=-0.10 → 0.80
    return max(_IC_MULT_FLOOR, min(_IC_MULT_CAP, raw))
```

### Data Source

```python
from portfolio.ic_computation import load_cached_ic, compute_and_cache_ic

def _get_ic_data(horizon: str) -> dict:
    """Load IC data, computing if cache is stale."""
    cache = load_cached_ic(horizon)
    if cache is None:
        cache = compute_and_cache_ic(horizon)
    return cache.get("global", {}) if cache else {}
```

IC cache TTL is already 3600s. The `_load_entries()` call in IC computation reuses the same signal_log.jsonl that accuracy_stats uses, so no new I/O.

### Per-Ticker IC (Phase 2)

The `compute_signal_ic_per_ticker()` function already exists. Phase 2 would:
1. Load `cache.get("per_ticker", {})` 
2. Look up `per_ticker.get(ticker, {}).get(signal_name, {})`
3. Use per-ticker IC when `samples >= 100`, fall back to global IC

This is higher impact but higher risk (per-ticker IC can be noisy with small samples).

### Impact Analysis

**Phantom performers** (calendar, econ_calendar, fear_greed):
- IC=0.0 → ic_mult = 1.0 (no change from IC alone)
- But ICIR < 0.10 → ic_mult = 1.0 (stability filter catches them)
- These are already partially handled by bias_penalty. IC adds no new correction here.
- To penalize zero-IC signals, change formula to: `raw = 1.0 + _IC_ALPHA * max(ic, 0) + _IC_ZERO_PENALTY * (1 if abs(ic) < 0.01 and samples > 500 else 0)`
- With `_IC_ZERO_PENALTY = -0.15`: zero-IC signals get 0.85x multiplier

**Genuinely good signals** (ministral, momentum, mean_reversion):
- ministral IC=0.094 → ic_mult = 1.188 → ~19% boost
- momentum IC=0.063 → ic_mult = 1.126 → ~13% boost  
- mean_reversion IC=0.068 → ic_mult = 1.136 → ~14% boost
- credit_spread_risk IC=0.201, ICIR=0.521 → ic_mult = 1.402 → 40% boost (if re-enabled)

**Contrarian signals** (ml, futures_flow):
- These are already accuracy-gated (force-HOLD). IC multiplier doesn't apply to gated signals.
- The architectural tension remains: should high-|IC| contrarian signals bypass the accuracy gate?
- Recommendation: NOT in this phase. Defer to after observing IC-weighted system performance.

## Implementation Steps

### Batch 1: Core IC Multiplier
1. Add `_get_ic_data()` helper (lazy, cached)
2. Add `_compute_ic_mult()` function
3. In `_weighted_consensus()`, after accuracy weight computation (line 1519), apply IC multiplier
4. Log IC multiplier in extra_info for observability

### Batch 2: Tests
1. Test IC multiplier math (positive IC → boost, zero IC → no change, negative IC → penalty)
2. Test stability filter (low ICIR → skip)
3. Test sample minimum (insufficient data → skip)
4. Test integration: mock IC cache, verify weight chain includes IC

### Batch 3: Per-Ticker IC (optional, Phase 2)
1. Add per-ticker IC lookup
2. Fall back to global when per-ticker samples < 100
3. Additional tests for fallback logic

## Risks

1. **IC computation cost**: `compute_signal_ic()` loads all signal_log entries and computes Spearman correlation. Currently O(N*S) where N=entries, S=signals. With 6575 entries and 36 signals, this is ~240K correlations. Cached with 1h TTL.

2. **Stale IC data**: IC cache could become stale during volatile periods. 1h TTL is probably fine since IC is a long-term metric (30+ sample minimum).

3. **Overfitting**: IC computed on all historical data may not represent future IC. Rolling IC (ICIR) partially addresses this. Could add recency weighting to IC computation (70/30 blend like accuracy).

4. **Regime interaction**: IC may differ by regime (e.g., BB has positive IC in uptrends, negative in downtrends). The regime-conditioned weighting plan (`data/regime_analysis_2026-04-18.json`) should be implemented alongside or after IC weighting.

## Deferred Decisions

- [ ] Should high-|IC| contrarian signals (ml, futures_flow) bypass accuracy gate?
- [ ] Should IC get recency weighting (70/30 blend)?
- [ ] Should per-ticker IC override global IC or blend with it?
- [ ] Should IC multiplier interact with or replace existing horizon_mults?
- [ ] What alpha value optimizes risk-adjusted returns? (needs backtest)
