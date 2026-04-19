# Agent Review: signals-modules

## P1 Findings
1. **news_event.py "cut" keyword** sends negative headlines to positive bucket (news_event.py:255-263). Confidence: 95
2. **copper_gold_ratio metals inversion** doesn't update sub_signals dict (copper_gold_ratio.py:251-275). Confidence: 90
3. **volume_flow NaN defaults to BUY** — price_up=True when price_change is NaN (volume_flow.py:289). Confidence: 92

## P2 Findings
1. **hurst_regime duplicate vote** — single EMA calculation counted twice (hurst_regime.py:280-298)
2. **statistical_jump_regime O(n^2)** rolling apply performance hazard
3. **volatility sqrt(365)** for all assets including stocks (should be sqrt(252))
4. **cross_asset_tsmom "GLD" not in tickers** — indicator always None
5. **news_event asymmetric severity gate** — moderate positive never votes

## P3 Findings
1. shannon_entropy momentum window overlap (80% shared bars, near-constant)
2. calendar_seasonal January triple-counted by 3 overlapping sub-signals
