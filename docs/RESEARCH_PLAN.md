# After-Hours Research Plan — 2026-05-27

## Findings Summary

### Phase 0: System State
- System healthy: 0 loop errors, 87 cycles, all 55 signal modules green
- XAG bought at $74.39 (RSI 21), oscillated to $74.84 by EOD
- All L2 decisions HOLD/DORMANT (correct — range-bound regime)
- 3 enabled signals badly degraded: qwen3 36%, econ_calendar 41.5%, crypto_macro 38.2%
  - All auto-gated by accuracy gate (blended < 47%) — not actively harmful
- news_event surged to 69.6% recent — star performer

### Phase 1: Market Events
- Iran 14-point peace deal draft — geopolitical premium unwinding in metals
- Tomorrow MASSIVE: GDP 2nd estimate, Core PCE, jobless claims (14:30 CET)
- BTC whale accumulation 270K BTC in 30 days vs 8-day ETF outflow ($2B+)
- Central bank gold buying +35% QoQ (243t Q1 2026)
- Silver in 5th year supply deficit, solar demand 230M oz projected

### Phase 2: Quant Research
- TrustTrade selective consensus — weight by inter-signal agreement + temporal stability
- Fractional Kelly + vol-targeting — 75% growth of full Kelly, <50% max drawdown
- Adaptive ATR trailing stops — 1.5x ATR low-vol, 3.0x ATR high-vol

### Phase 3: Signal Audit
- Accuracy gate correctly handling degraded signals
- 17 shadow signals at 0 samples — outcome tracking gap
- Several disabled signals with strong recent accuracy worth investigation

## Implementation Plan

### Batch 1: Signal temporal consistency filter (HIGH IMPACT)
Discard signals that flip direction within 2 checks — known noise pattern.

Files:
- `portfolio/signal_engine.py` — add temporal consistency check
- `tests/test_signal_consistency.py` — new test file

### Batch 2: Updated morning briefing + Telegram
Tomorrow has 11 macro events. User needs morning briefing.

Files:
- `scripts/write_morning_briefing.py` — rewrite for today
- `data/morning_briefing.json` — output

### Batch 3: Research deliverables + signal audit
Files:
- `data/daily_research_signal_audit.json`
- `data/daily_research_ticker_deep_dive.json`

### Deferred to backlog:
- Walk-forward weight loop (2d)
- Fractional Kelly sizing (3d)
- BTC on-chain disaggregation (3d)
- Bull/bear adversarial sub-agents for L2 (4d)
