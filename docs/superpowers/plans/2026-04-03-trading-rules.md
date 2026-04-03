# Trading Rules — 9 Rules from Apr 1-2 Losses

## Evidence
Apr 2: 11 round-trips, avg -37.27 SEK per round-trip = -409 SEK total.
Root cause: spread friction on cheap certs + entering on weak signals.

## The 9 Rules

1. Orders > 1,000 SEK (courtage-free AVA Markets threshold)
2. Only AVA products (isAza=true, no SG/BNP/VON courtage)
3. Voting: 2+ tactics must agree (no single-tactic entries)
4. 5 min cooldown between trades (prevent churn)
5. Prefer higher-priced certs (lower proportional spread)
6. Minimum expected gain > 50 SEK per trade
7. Track every trade with exact amounts in persistent log
8. Mode detection (straddle on chop days — already done)
9. Spread-aware: skip instruments with >1% round-trip spread

## Implementation — single batch in fish_monitor_live.py

### buy_position() changes:
- Enforce min order 1,000 SEK: `if budget < 1000: skip`
- Use higher-priced cert: BULL_OB = '1650161' (AVA 4 at ~7 SEK, not AVA 3 at ~1 SEK)
- Check spread before buying: `spread = (ask-bid)/bid`, skip if >1%
- Log trade to data/fish_trades.jsonl with exact amounts

### Voting changes:
- Remove single-tactic half-size entry (the `len(longs)==1 and len(shorts)==0` block)
- Require 2+ tactics to agree, otherwise NO TRADE

### Cooldown change:
- All cooldowns to 300s (5 min) minimum
- High-conviction exits: 120s (was 5s — too aggressive)

### Trade logging:
- Every buy/sell appends to data/fish_trades.jsonl
- Format: {ts, action, instrument, units, price, amount, tactic, session_pnl}
