# Fishing Tactics — ORB, Gold-Leads-Silver, VWAP, Time Gating

## What to Add (from research)

### Tactic 1: Gold-Leads-Silver (5-min edge)
Gold moves 0.85 correlation with silver, 5-min lead time.
If gold breaks +0.5% in 5 min and silver hasn't moved → buy BULL.
If gold drops -0.5% and silver flat → buy BEAR.
Infrastructure: Binance FAPI for both XAUUSDT and XAGUSDT.
Integration: add to fish_monitor_live.py as additional entry trigger.

### Tactic 2: Opening Range Breakout (ORB)
First 15 min range after 08:15 CET. Trade breakout direction.
TP at 50% of range width, SL at 60%.
Research: 411% return on gold futures, fully automatable.
Infrastructure: portfolio/orb_predictor.py already exists for XAGUSDT.
Integration: compute morning range, add as entry trigger in monitor.

### Tactic 3: VWAP as Mean Reversion Target
Replace SMA20 with VWAP in straddle mode exits.
VWAP already computed in agent_summary price_levels.
Better target because volume-weighted.

### Tactic 4: Time-of-Day Gating
14:00-17:00 CET = highest volume = best entries.
10:00-14:00 CET = dead zone = avoid or require stronger signal.
Integration: boost entry confidence during US hours, penalize EU dead zone.

## Batch Plan

### Batch 1: Gold-leads-silver detection
- Add gold price tracking to fish_monitor_live.py
- Every 60s: fetch XAUUSDT from Binance FAPI alongside XAGUSDT
- Track gold 5-min rolling change
- If gold_5min_change > 0.5% AND silver_5min_change < 0.2% → BULL entry
- If gold_5min_change < -0.5% AND silver_5min_change > -0.2% → BEAR entry
- This is an ADDITIONAL entry trigger alongside momentum/straddle

### Batch 2: ORB integration
- On startup: call orb_predictor to compute today's morning range
- If range is formed (need 08:00-10:00 UTC data):
  - Track if price breaks above morning high → BULL signal
  - Track if price breaks below morning low → BEAR signal
  - TP at 50% of range, SL at 60%
- If starting after 10:00 UTC, use the already-formed range

### Batch 3: VWAP exit + time gating
- Read VWAP from agent_summary price_levels
- In straddle mode, use VWAP as exit target instead of SMA20
- Add time-of-day multiplier: 14:00-17:00 CET = 1.0 (full), 10-14 = 0.5 (weak)
- Apply to entry threshold: in dead zone, require stronger MC (>80%/<20%)
