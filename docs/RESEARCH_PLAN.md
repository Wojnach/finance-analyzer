# Research Session Plan — 2026-04-29

## Context

After-hours research session. Markets closed. FOMC held rates at 3.5-3.75% with
historic 8-4 dissent. Oil at $118 Brent (Iran blockade). GDP/PCE tomorrow.

## Key Findings

1. **Signal engine is already sophisticated** — IC-based weighting, correlation dedup,
   directional rescue, macro-window gating, crisis mode, persistence filter all implemented.
   Previous plan's "IC-based weighting" improvement is already shipped.

2. **Tier 1 consensus accuracy is coin-flip**: BTC 52.5%, ETH 49.8%, XAG 49.4%, XAU 49.4%,
   MSTR 47.8% at 1d. Slightly better at 3h: XAU 54.3%, XAG 53.5%, BTC 50.7%.

3. **Fibonacci consistently bad** — 43.6% at 1d (17K samples), 43.3% at 3h. Already
   accuracy-gated but still computed every cycle. Should be formally disabled.

4. **Pending validation signals**: 17 signals pending. Only 5 shadow-tracked (hurst,
   shannon_entropy, statistical_jump, realized_skewness, oscillators). The other 12
   have zero accuracy data because they're not shadow-safe.

5. **Realized skewness should be killed** — 33.3% at 1d with 90 samples. Definitively bad.

6. **MSTR is a BTC proxy** — 0.58 correlation. Our signals treat it as a stock, ignoring
   that BTC price is the primary driver. Cross-asset BTC signal for MSTR would help.

7. **Statistical jump regime is marginal** — 52.7% at 1d with 110 samples. Above gate
   threshold but small sample. Worth enabling for further validation.

## Bugs & Problems Found

1. **Fibonacci computed but always gated** — wastes CPU on every 60s cycle for a signal
   that can never pass the 47% gate with 17K samples. Formal disable saves ~50ms/cycle.

2. **12 pending signals have no shadow tracking** — can never be evaluated. Need to add
   non-network-heavy signals to _SHADOW_SAFE_SIGNALS.

3. **MSTR treated as generic stock** — accuracy at 47.8% because signal system ignores
   the BTC treasury dependency. Cross-asset signal needed.

## Improvements Prioritized

### Batch 1: Signal Cleanup (low risk, saves CPU, enables evaluation)
- **Files**: `portfolio/tickers.py`, `portfolio/signal_engine.py`
- **Changes**:
  1. Add `fibonacci` to DISABLED_SIGNALS (43.6% at 1d, 17K+ sam)
  2. Add non-network signals to `_SHADOW_SAFE_SIGNALS` to accumulate accuracy data:
     `complexity_gap_regime`, `mahalanobis_turbulence`, `crypto_evrp`, `hash_ribbons`
  3. Enable `statistical_jump_regime` (52.7% at 110 sam — above gate, worth live validation)
  4. Update `realized_skewness` comment to "killed" (33.3% at 90 sam, below gate anyway)
  5. Remove `fibonacci` from HORIZON_SIGNAL_WEIGHTS (no longer needed if disabled)

### Batch 2: MSTR Cross-Asset BTC Signal (medium risk, targets worst ticker)
- **Files**: `portfolio/signal_engine.py`
- **Changes**: When computing consensus for MSTR, also include the most recent BTC-USD
  consensus action as a synthetic signal with weight proportional to BTC's per-ticker
  consensus accuracy. Implementation: in the vote-building section, if ticker==MSTR and
  BTC consensus is cached, inject a `btc_proxy` vote.

### Batch 3: Prophecy Review (no code risk)
- **Files**: `data/prophecy.json`
- **Changes**: Review and update macro beliefs with today's data:
  - Silver at $72.81 (checkpoints through $80 triggered, $120 pending)
  - BTC at $77.5K ($75K triggered, $85K pending)
  - ETH at $2,330 (all checkpoints pending, $2.5K next)

## What to Defer

- Sentiment model calibration (CryptoBERT bullish bias) → needs separate session
- Walk-forward optimization → already partially handled by dynamic horizon weights
- Multi-agent bull/bear debate (TradingAgents) → too complex, low ROI
- XGBoost feature importance for signal selection → needs backtesting infrastructure

## Execution Order

1. Create worktree `research/daily-2026-04-29`
2. Batch 1: Signal cleanup + tests
3. Batch 2: MSTR BTC cross-asset + tests
4. Batch 3: Prophecy review
5. Run full test suite
6. Write signal audit deliverable
7. Merge, push, restart loops
8. Morning briefing + Telegram
