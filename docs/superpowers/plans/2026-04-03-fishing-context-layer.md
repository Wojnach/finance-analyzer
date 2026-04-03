# Fishing Context Layer — Claude Sets Strategy, Engine Executes Tactics

## The Insight
Layer 2 (Claude) produces the best analysis in the system — adversarial bull/bear
debate, news interpretation, event awareness, macro reasoning — but the fish engine
ignores ALL of it. Meanwhile the engine churns on noisy signals because it has no
context for WHEN to trade vs WHEN to sit out.

## The Design

### Layer 2 produces a "fishing context" (2-3x per day on triggers)
When Layer 2 is invoked for XAG-USD (trigger detected by main loop), it also writes
a fishing-specific context file:

```json
// data/fishing_context.json
{
  "timestamp": "2026-04-02T14:30:00Z",
  "valid_until": "2026-04-02T21:45:00Z",
  "ticker": "XAG-USD",
  
  // STRATEGY (set by Claude, used as bias by engine)
  "direction_bias": "bearish",         // bullish/bearish/neutral
  "bias_confidence": 0.7,
  "bias_reasoning": "Liberation Day crash, -6.5% today, post-crash chop likely",
  
  // RULES (set by Claude, enforced by engine)  
  "allow_long": false,                 // block LONG entries today
  "allow_short": true,
  "max_trades_today": 3,               // Claude says limit trading
  "max_hold_minutes": 60,              // shorter holds on event day
  "position_size_multiplier": 0.5,     // half size due to high vol
  "allow_overnight": false,            // no overnight on event day
  
  // CONTEXT (from Layer 2 reasoning)
  "event_context": "Liberation Day anniversary, Trump tariff speech",
  "news_summary": "Oil +6%, metals crash, DXY strengthening",
  "bull_case": "RSI 29 oversold, 67% bounce probability next day",
  "bear_case": "Crash days close near low (22% recovery), Friday risk",
  "journal_action": "SELL",            // Layer 2's last journal decision
  "journal_confidence": 0.7,
  
  // LAYER 2 REASONING AS 9TH TACTIC VOTE
  "tactic_vote": "SHORT",             // direction for voting system
  "tactic_weight": 2.0,               // counts as 2 votes (highest trust)
}
```

### Fish engine reads this context every cycle
```python
def _vote_layer2_context(self, state):
    """Tactic 9: Layer 2 reasoning as a weighted vote."""
    ctx = self._load_fishing_context()
    if not ctx or self._is_stale(ctx, max_age_hours=4):
        return None
    
    # Enforce Claude's rules
    if not ctx.get('allow_long') and direction == 'LONG':
        return 'BLOCKED'
    if not ctx.get('allow_short') and direction == 'SHORT':
        return 'BLOCKED'
    
    # Add Layer 2's vote (counts as 2 in the voting system)
    return ctx.get('tactic_vote')  # 'LONG', 'SHORT', or None
```

### The voting becomes:
```
Before: 8 tactics, need 2+ to agree
After:  9 tactics, Layer 2 vote counts as 2:
  - If Layer 2 says SHORT + 1 other tactic agrees → 3 votes → ENTER
  - If Layer 2 says SHORT but 3 tactics say LONG → 3L vs 2S → CONFLICT
  - If Layer 2 is stale (>4h) → falls back to 8-tactic system
```

### How fishing_context.json gets written

**Option A: Add to existing Layer 2 prompt (cheapest)**
When Layer 2 is invoked for XAG-USD, add to the prompt:
"Also write a fishing context to data/fishing_context.json with: direction_bias,
allow_long, allow_short, max_trades_today, position_size_multiplier, event_context,
and your reasoning."

Layer 2 already reads the playbook and makes decisions — this just asks it to also
output a structured JSON for the fish engine.

**Option B: Separate T1 invocation for fishing context**
Invoke a quick T1 (Haiku, 2 min) specifically for fishing context:
"Given today's signals, news, and events, what trading rules should the fish engine
follow today? Output fishing_context.json."

This runs in parallel, doesn't slow down the main T2/T3 invocation.

**Option C: Perception gate triggers fishing-specific invocation**
Add fishing-specific triggers to the gate:
- "All 8 tactics agree" → invoke T1 for confirmation
- "Event day detected" → invoke T2 for strategy
- "Session ending with position" → invoke T2 for overnight decision

### What else the engine should read (currently wasted)

1. **layer2_journal.jsonl** — last XAG-USD entry → direction + confidence
2. **headlines_latest.json** — top headline title + severity → display in monitor
3. **prophecy.json** — long-term belief → bias the overnight hold decision
4. **accuracy_cache.json** — which signals are reliable → weight the consensus
5. **Monte Carlo bands** — probabilistic price targets → smarter TP/SL levels
6. **Chronos forecast % move** — short-term direction prediction

## Implementation Plan

### Batch 1: fishing_context.json writer (in agent_invocation.py)
- When Layer 2 runs for XAG-USD, also write fishing_context.json
- Add to T2/T3 prompt: "output fishing strategy as JSON"
- OR: add post-processing that extracts strategy from journal entry

### Batch 2: fish_engine reads fishing_context
- Add _vote_layer2_context() as 9th tactic
- Weight = 2 (counts as 2 votes)
- Enforce direction_bias, allow_long/short, max_trades, position_size_multiplier
- Display in monitor log: "[L2: SHORT 70% 'Liberation Day crash']"

### Batch 3: fish_engine reads other wasted signals
- Read layer2_journal.jsonl for latest XAG-USD decision
- Read Monte Carlo bands for dynamic TP/SL
- Read Chronos forecast for short-term bias
- Read prophecy for overnight decision

### Batch 4: Perception gate for fishing triggers
- Add fishing-specific triggers to main loop
- "Fish engine entered a position" → log for Layer 2 awareness
- "Event day + fish engine active" → invoke T1 for strategy check
- "Session ending with position" → invoke T2 for overnight

## What This Changes

| Before | After |
|--------|-------|
| Fish engine ignores Layer 2 | Layer 2 journal is the strongest tactic vote |
| No trading rules from Claude | Claude sets daily strategy (direction, limits) |
| Churn on event days | Claude says "don't trade LONG today, max 3 trades" |
| Monitor shows raw numbers | Monitor shows "L2: SHORT 70% 'crash day chop'" |
| Overnight decision is RSI/MC | Overnight decision includes Claude's reasoning |
| News is just BUY/SELL flag | News headlines visible in monitor, severity weighted |

## Risk
- Layer 2 might not run on a given day (no trigger) → engine falls back to 8 tactics
- fishing_context.json could be stale → 4h max age, then ignore
- Layer 2 could be wrong → it's one vote (weight 2), not a veto. 
  If 4 other tactics disagree, they still win (4 vs 2).
