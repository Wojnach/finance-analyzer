# Fish Engine — Consolidate fishing into metals loop architecture

## Problem
The fish_monitor_live.py is a standalone primitive script that reimplements
(worse) many features the metals loop already has: Monte Carlo, trade guards,
velocity detection, exit optimization, signal accuracy tracking. It runs as
a separate process with its own data fetching, adding latency and complexity.

## Architecture

### Current (wrong):
```
Main Loop → writes signals → agent_summary.json
Metals Loop → writes signals → metals_signal_log.jsonl
Fish Monitor → reads both files → makes decisions → trades on Avanza
(3 separate processes, duplicated computation, no shared state)
```

### Target:
```
Main Loop → writes signals → agent_summary.json
Metals Loop → reads signals + runs fish_engine each cycle:
  - fish_engine.evaluate() returns: entry/exit/hold decision
  - Uses metals loop's existing: Monte Carlo, trade guards,
    velocity detection, exit optimizer, signal accuracy
  - Executes on Avanza via existing avanza_orders infrastructure
  - Layer 2 (Claude) invoked for strategic decisions only:
    overnight hold/sell, mode selection, event playbook
```

### New module: `data/fish_engine.py`

Called by metals loop every 60s cycle (alongside existing signal computation).
Pure function: takes state in, returns decision out.

```python
class FishEngine:
    """Intraday fishing decision engine — runs inside metals loop."""
    
    def __init__(self, config):
        self.config = config
        self.mode = 'momentum'  # or 'straddle'
        self.active_position = None
        self.session_pnl = 0
        self.trade_count = 0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        self.votes = {}
        
    def evaluate(self, state: dict) -> dict:
        """Main decision function. Called every 60s by metals loop.
        
        Args:
            state: {
                'silver_price': float,
                'gold_price': float,
                'gold_5min_change': float,
                'signals': dict (from agent_summary),
                'metals_signals': dict (from metals loop),
                'mc': dict (Monte Carlo from main loop),
                'exit_optimizer': dict (from exit_optimizer.py),
                'trade_guard': dict (from metals_risk.py),
                'velocity': dict (from silver fast-tick),
                'orb_range': dict,
                'temporal_pattern': dict,
                'news': dict,
            }
            
        Returns:
            {'action': 'BUY'/'SELL'/'HOLD',
             'direction': 'LONG'/'SHORT',
             'instrument': str,
             'volume': int,
             'reason': str,
             'tactics_agreed': list}
        """
```

## Batch Plan

### Batch 1: fish_engine.py — core decision engine
- Create data/fish_engine.py with FishEngine class
- Move all 8 tactics + voting + 9 rules from fish_monitor_live.py
- Uses metals loop state dict as input (no own data fetching)
- Returns decision dict (no own Avanza execution)

### Batch 2: metals loop integration
- Add FishEngine instantiation at metals loop startup
- Each 60s cycle: build state dict from existing data → call engine.evaluate()
- Execute decisions via existing avanza_orders infrastructure
- Use existing trade guards + drawdown circuit breaker
- Use existing exit_optimizer for probabilistic exits

### Batch 3: Layer 2 strategic decisions
- On mode changes: invoke Layer 2 T1 (quick, 2 min) for confirmation
- On overnight hold/sell: invoke Layer 2 T2 (6 min is fine at 21:45)
- On event day detection: invoke Layer 2 for playbook selection
- These are RARE (1-2x per day), not per-cycle

### Batch 4: Remove standalone fish_monitor_live.py
- Keep as fallback only (if metals loop is down)
- Primary fishing now runs inside metals loop
- Update /fin-fish skill to use the integrated engine

## What Changes vs Current

| Feature | fish_monitor_live.py | fish_engine.py |
|---------|---------------------|----------------|
| Runs as | Standalone process | Inside metals loop |
| Price fetch | Own Binance FAPI calls | Uses metals loop's existing prices |
| Gold tracking | Own FAPI call + in-memory history | Uses metals loop's gold history |
| Monte Carlo | None | Uses metals loop's existing MC |
| Trade guards | Fixed 5 min cooldown | Loss escalation (30min→4h) |
| Exit logic | Static RSI+MC thresholds | Exit optimizer (probabilistic EV) |
| Velocity | None (60s checks) | Silver fast-tick (10s) |
| Signal accuracy | None | Existing accuracy tracker |
| Claude reasoning | Never | Strategic decisions (1-2x/day) |

## Risk
- Metals loop is production code (4,300+ lines, runs 24/7)
- Minimal changes to metals loop itself — just add one function call
- All new logic in fish_engine.py (separate, testable, removable)
- Config flag: `FISHING_ENABLED = True/False` to disable without code changes
