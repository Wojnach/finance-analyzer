# After-Hours Research Plan — 2026-03-29

## Context
Sunday session. All instruments ranging. No trades today (weekend). System uptime 63h+,
zero signal failures. Key finding: massive horizon-specific accuracy divergence discovered
in signal audit. Regime gating is not horizon-aware — suppressing signals that work well
on shorter horizons.

## Bugs & Problems Found

### BUG-149: Regime gating is not horizon-aware (P1)
- **File:** `portfolio/signal_engine.py` line 162
- **Root cause:** `REGIME_GATED_SIGNALS` applies the same gating regardless of prediction
  horizon. In ranging regime, `trend` is gated — but it has 61.6% recent accuracy on 3h.
  We're suppressing a profitable signal on short horizons.
- **Data:** trend 1d_recent=40.7% (correctly gated), trend 3h_recent=61.6% (incorrectly gated)
- **Fix:** Make `REGIME_GATED_SIGNALS` horizon-specific. Only gate on horizons where the
  signal genuinely fails.

### BUG-150: HORIZON_SIGNAL_WEIGHTS stale (P2)
- **File:** `portfolio/signal_engine.py` line 171
- **Root cause:** Static weights from March 27 audit. Accuracy has shifted since then.
  Multiple high-performing signals on 3h (smart_money 63.2%, volatility_sig 60.2%,
  momentum_factors 60.1%) have no boost. Multiple poor performers on 1d (ema 40.8%,
  heikin_ashi 42.0%) have no penalty.
- **Fix:** Update weights with March 29 accuracy data.

### BUG-151: EMA has no 1d penalty despite 40.8% recent accuracy (P2)
- **File:** `portfolio/signal_engine.py` HORIZON_SIGNAL_WEIGHTS
- **Root cause:** ema at 40.8% on 1d_recent is near gating threshold. Has 1.3x boost
  on 3h (correct — 62.9%) but no penalty on 1d.
- **Fix:** Add ema 0.6x penalty for 1d horizon.

## Improvements Prioritized (impact × ease)

### Tier 1: Implement NOW

#### 1. Horizon-aware regime gating
- **Impact:** HIGH — unlocks 61.6% trend signal on 3h in ranging regime
- **Effort:** Easy — small dict restructure
- **Files:** `portfolio/signal_engine.py`
- **Approach:** Change `REGIME_GATED_SIGNALS` to `{regime: {horizon: frozenset(signals)}}`
  format. Gate trend on 1d in ranging but NOT on 3h/4h. Gate mean_reversion on 3h
  in trending but NOT on 1d.

#### 2. Update HORIZON_SIGNAL_WEIGHTS with fresh data
- **Impact:** MEDIUM — better weights = better consensus
- **Effort:** Easy — update dict values
- **Files:** `portfolio/signal_engine.py`
- **New 3h boosts:** smart_money 1.2, volatility_sig 1.2, momentum_factors 1.2
- **New 3h penalties:** bb 0.6, mean_reversion 0.7
- **New 1d penalties:** ema 0.6, heikin_ashi 0.7
- **New 1d boosts:** macd 1.2

#### 3. Dynamic horizon weight computation (replace static with computed)
- **Impact:** HIGH — weights auto-update as accuracy changes
- **Effort:** Medium — new function + caching
- **Files:** `portfolio/signal_engine.py`
- **Approach:** Compute multipliers from accuracy_cache per-horizon data. Each signal
  gets a multiplier = this_horizon_acc / cross_horizon_mean_acc, clamped [0.5, 1.5].
  This automatically captures horizon-specific performance without manual updates.
  Keep static dict as default fallback.

### Tier 2: Low-hanging fruit

#### 4. Add `sentiment` to CORRELATION_GROUPS or regime gating
- **Impact:** MEDIUM — 33.8% on 3h, 46.8% on 1d with large samples
- **Effort:** Easy
- **Files:** `portfolio/signal_engine.py`
- **Approach:** Sentiment is consistently bad. Add to "macro_external" correlation group
  so it gets penalized when voting with other external signals.

#### 5. Add `forecast` to regime gating for ranging markets
- **Impact:** MEDIUM — <40% on both horizons recently
- **Effort:** Easy
- **Files:** `portfolio/signal_engine.py`

### Deferred
- Per-ticker accuracy for all signals (not just LLMs) — requires significant refactor
- Walk-forward PPO/RL for signal combination — too complex for overnight session
- Multi-agent debate architecture — requires new infrastructure
- Dynamic accuracy gate threshold per ticker — needs more data

## Execution Order

### Batch 1: signal_engine.py improvements (BUG-149, BUG-150, BUG-151)
1. Make REGIME_GATED_SIGNALS horizon-aware
2. Update HORIZON_SIGNAL_WEIGHTS with fresh March 29 data
3. Add new correlation group for macro-external signals
4. Add forecast to ranging regime gating
5. Run targeted tests

### Batch 2: Test updates
1. Add tests for horizon-aware regime gating
2. Add tests for updated weights
3. Run full suite

### Batch 3: Dynamic horizon weight computation
1. Implement `_compute_horizon_weights()` function
2. Add caching with 1h TTL
3. Integrate into `_weighted_consensus()`
4. Add tests
5. Run full suite

## Risk Assessment
- All changes in weighting/gating layer, NOT signal computation
- Signals still compute — we just change combination weights
- Trivially reversible (revert to static dicts)
- No config.json changes, no API key exposure
- Horizon-aware gating is strictly better — it unlocks signals that work
