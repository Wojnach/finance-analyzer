# PLAN — Tighten directional-bias penalty so direction-balanced voters dominate consensus

Date: 2026-05-19
Branch: `fix-tighten-bias-penalty-threshold`
Worktree: `.worktrees/rebalance-bias-threshold`

## Problem

Investigation in prior session (`docs/SESSION_PROGRESS.md` 2026-05-18)
confirmed the 60-70% "accuracy" the user remembered came from a 1-2
week rally tailwind (5/06-5/12). 5 BUY-biased signals (sentiment,
crypto_macro, econ_calendar, structure, macro_regime) showed
60-92% during the rally then collapsed to 33-47% after the 5/11-5/13
regime flip. Net consensus accuracy 90d/all-tickers = 47-53% — a small
BTC edge (~3pp), everything else inside noise.

Per-signal honest hit rates by ALL-TIME bias:

| Signal              | bias | active_rate | active BUY:SELL | 14d hit |
|---------------------|------|-------------|-----------------|---------|
| rsi                 | 0.11 | 0.36        | 44:56           | 63.1%   |
| bb                  | 0.15 | 0.09        | 42:58           | 55.0%   |
| mean_reversion      | 0.20 | 0.30        | 40:60           | 52.9%   |
| momentum            | 0.04 | 0.36        | 49:51           | 52.5%   |
| sentiment           | 0.81 | 0.44        | 91:9            | 51.4%   |
| crypto_macro        | 0.91 | 0.28        | 95:5            | 53.9%   |
| econ_calendar       | 0.48 | 0.13        | 26:74           | 53.2%   |
| structure           | 0.37 | 0.19        | 68:32           | 50.2%   |
| macro_regime        | 0.26 | 0.34        | 37:63           | 49.6%   |

Direction-balanced (bias <0.3): rsi, bb, momentum, mean_reversion,
statistical_jump_regime, macro_regime. These survived the regime flip
(rsi 56→69→66 across rally/flip/now).

Direction-biased (bias >0.65): sentiment, crypto_macro,
claude_fundamental, williams_vix_fix, crypto_evrp. These looked
amazing in rally, terrible after.

## Existing infrastructure

`portfolio/signal_engine.py` already has a directional-bias penalty at
lines 500-504 + 2620-2630:

```python
_BIAS_THRESHOLD = 0.85  # >85% bias triggers penalty
_BIAS_PENALTY = 0.5     # 0.5x weight for high-bias (85-95%)
_BIAS_EXTREME_THRESHOLD = 0.95  # >95% extreme
_BIAS_EXTREME_PENALTY = 0.2     # 0.2x weight for extreme bias
_BIAS_MIN_ACTIVE = 30   # need 30+ active votes
```

Applied only when the signal votes IN its bias direction — contrarian
votes keep full weight (an informative signal).

Current penalty assignments (with current thresholds):

| Signal              | bias | bucket | mult |
|---------------------|------|--------|------|
| sentiment           | 0.81 | none   | 1.0  |
| crypto_macro        | 0.91 | high   | 0.5  |
| claude_fundamental* | 0.81 | none   | 1.0  |
| williams_vix_fix*   | 0.83 | none   | 1.0  |
| crypto_evrp*        | 0.67 | none   | 1.0  |
| fear_greed          | 1.00 | extr.  | 0.2  |
| calendar            | 0.95 | extr.  | 0.2  |
| funding             | 1.00 | extr.  | 0.2  |
| news_event          | 0.99 | extr.  | 0.2  |

(* = disabled or pending validation; not in active 17)

## Goal

Add a moderate-bias tier (0.65 < bias ≤ 0.85) at 0.7x and promote
crypto_macro from "high" (0.5x) to "extreme" (0.2x) by lowering the
extreme threshold 0.95 → 0.90.

Net effect on weighted_consensus:
- **sentiment** 1.0x → 0.7x on BUY votes (its bias direction)
- **crypto_macro** 0.5x → 0.2x on BUY votes
- claude_fundamental, williams_vix_fix, crypto_evrp also catch the new
  moderate tier (but disabled/pending — no immediate live effect)
- Direction-balanced signals (rsi, bb, momentum, mean_reversion,
  statistical_jump_regime, macro_regime, structure) — UNCHANGED at 1.0x
- Contrarian votes (e.g., sentiment SELL, crypto_macro SELL) — UNCHANGED
  at 1.0x (informative; keep them)

This is a config-only change. No new code path. No new signals.

## Design

`portfolio/signal_engine.py`:

```python
# Was:
_BIAS_THRESHOLD = 0.85
_BIAS_PENALTY = 0.5
_BIAS_EXTREME_THRESHOLD = 0.95
_BIAS_EXTREME_PENALTY = 0.2

# After:
_BIAS_MODERATE_THRESHOLD = 0.65  # NEW: catches sentiment (0.81)
_BIAS_MODERATE_PENALTY = 0.7     # NEW: 0.7x weight (lighter)
_BIAS_THRESHOLD = 0.85           # unchanged trigger (now "high")
_BIAS_PENALTY = 0.5              # unchanged
_BIAS_EXTREME_THRESHOLD = 0.90   # lowered 0.95 → 0.90 (catches crypto_macro 0.91)
_BIAS_EXTREME_PENALTY = 0.2      # unchanged
```

Update the cascade in `apply_confidence_penalties` / weighted_consensus
loop to use the three-tier check, choosing the lowest applicable
multiplier (extreme > high > moderate > none).

## Files Touched

1. `portfolio/signal_engine.py` — 4 constant changes + 3-tier branch
2. `tests/test_signal_engine_core.py` or `tests/test_weighted_consensus.py`
   — add tests for moderate-tier bias penalty + extreme-threshold drop
3. `docs/PLAN.md` — this file (committed before implementation)
4. `docs/SESSION_PROGRESS.md` — end-of-session note (after merge)

## Files NOT Touched (and why)

- `portfolio/accuracy_stats.py` — bias is already computed correctly there
- `portfolio/accuracy_degradation.py` — the prior session's window change
  is independent and stays
- Trade gates (ACCURACY_GATE_THRESHOLD, GRID_MIN_SIGNAL_CONFIDENCE) —
  intentionally untouched; this is about WEIGHTING not GATING
- `data/activation_cache.json` — auto-recomputes; no manual write

## Risks

1. **Sentiment goes near-mute.** sentiment activates 44% of cycles
   (highest active-rate signal we have). 0.7x of its BUY votes lowers
   its consensus contribution. If sentiment was the swing vote on a
   real BUY, we miss it. Mitigation: 0.7x is mild; rsi+bb+momentum
   should dominate if they agree.

2. **Crypto_macro near-silent on BUY.** 0.2x effectively zero. But its
   91% BUY-bias means it's saying nothing new vs the bull regime
   itself. Honest cost: in clean rallies it stops boosting confidence.

3. **Layer 2 prompt changes weighted_confidence outputs.** Layer 2
   reads `agent_summary.json` which contains weighted_confidence per
   ticker. Lower numbers could change Layer 2's decision style —
   especially under "Patient" mandate where high confidence is
   required. Mitigation: trade gates unchanged, so the decision
   ENVELOPE doesn't shift, just where inside it we sit.

4. **Backtest math will look worse retroactively.** Backtester replays
   historical votes through current consensus weights. Past performance
   numbers in dashboard `/api/accuracy-history` will subtly change.
   Acceptable — that's the honest math, not regression.

5. **Tests asserting specific weight values may break.** Need to find
   and update them.

## Execution Order

1. Plan + commit (this step)
2. Premortem via fresh general-purpose agent
3. Update plan with premortem findings + commit
4. Implement constant changes + 3-tier branch (1 file)
5. Run targeted tests; fix any value-asserting tests
6. Add new tests for moderate-tier + extreme-threshold-drop
7. Run full suite
8. Adversarial review (`caveman:cavecrew-reviewer`)
9. Address P1/P2 findings
10. Commit + merge + push
11. Restart PF-DataLoop

## Premortem

(To be filled by fresh agent — see step 2.)
