# Agent Review: metals-core (2026-04-20)

## P1 Critical
1. **seed=42 in Monte Carlo VaR (STILL PRESENT)** — `metals_risk.py:185`. Deterministic. Risk metrics are static theater. Also `fin_snipe_manager.py:60`.
2. **ATR annualization underestimates vol** — Uses stock-market 252*6.5h. Correct for 24/7 metals is 365*24. Two MC engines now disagree.
3. **No barrier proximity guard in stop-loss placement** — `metals_swing_trader.py:2516`. 12.5% warrant stop can be below knockout barrier when barrier distance is 10%.

## P2 High
1. DST-hardcoded close time (21:55 CET) — wrong during transition weeks
2. Silver fast-tick reads/writes globals without synchronization
3. `_load_json_state` uses raw `open()` (violates atomic I/O rule)
4. `fin_snipe_manager.py` exit plans are deterministic (seed=42) — front-runnable

## P3 Medium
1. Avanza session expiry mid-trade leaves position without stop protection
2. ORB predictor hardcodes UTC winter offsets (wrong in summer)
3. Raw `open()` to read last trade reads entire file into memory
4. Swing trader `_cet_hour()` fallback always UTC+1 (wrong in summer)
5. Barrier distance defaults to 999 for certificates with barrier=None (correct but implicit)

## Prior Finding Status
- MC seed=42: **STILL PRESENT**
- ATR annualization: **PARTIALLY ADDRESSED** (metals_loop.py fixed, metals_risk.py not)
