# Plan: Drastically Reduce Claude Code Invocation Frequency

## Problem

Claude Code Max subscription usage is being consumed very fast. The metals loop invokes
`claude -p` (Layer 2) far too often — on Mar 2, **12 Tier 3 Opus invocations in 68 minutes**
during a volatile session. The 5-minute global cooldown is the only throttle, and broad
trigger conditions (L1 at 8%, trailing at 3%) fire continuously.

**Goal**: Reduce invocations by 80%+ while keeping all local processing (Chronos, Ministral,
price fetching, stop-loss monitoring, risk calculations) running at full speed.

## Root Causes

1. **L1 WARNING (8% threshold) fires constantly** — warrants spend hours inside this zone
   during normal volatility. Each L1 fires a T3 Opus invocation. ~60% of all invocations.
2. **"Dropped from peak" (3%) co-fires with L1** — `len(reasons) >= 2` escalates to T2+,
   and then the L1 pattern escalates to T3. Every volatile check = T3 Opus.
3. **MIN_INVOKE_INTERVAL = 300s (5 min)** — far too short. Produces 12 invocations/hour.
4. **classify_tier() over-escalates** — "stop-loss" pattern in reasons matches L1 WARNING
   (which contains "from stop-loss"), forcing everything to T3.
5. **HEARTBEAT_CHECKS = 20 (~30 min)** — a periodic invocation even when nothing changed.

## What We'll Do

### Changes to `data/metals_loop.py` only (single file)

**A. Raise global cooldown: 5 min → 30 min**
```python
MIN_INVOKE_INTERVAL = 1800   # 30 min minimum between invocations
```

**B. Stop L1 WARNING from invoking Claude**
L1 is a warning — log it, flag it in context, but do NOT add it to `reasons` list.
Only L2 and L3 should trigger Claude invocation. L1 stays as a log-only event.

**C. Raise trailing trigger: 3% → 8%**
A 3% warrant drop from peak is 0.6% underlying — noise. Raise to 8% (1.6% underlying on 5x).
```python
TRIGGER_TRAILING = 8.0
```

**D. Raise price move trigger: 2% → 5%**
2% warrant moves happen every check during active sessions. 5% = 1% underlying.
```python
TRIGGER_PRICE_MOVE = 5.0
```

**E. Fix classify_tier() over-escalation**
The pattern "stop-loss" matches L1 WARNING reason strings ("from stop-loss"). Fix:
- Only match "L2 ALERT" and "L3 EMERGENCY" as critical, not "stop-loss"
- Remove `len(reasons) >= 2 → T2` rule — multiple weak reasons shouldn't escalate
- "Dropped from peak" alone = T1

**F. Raise heartbeat: 20 checks → 80 checks (~2 hours)**
```python
HEARTBEAT_CHECKS = 80
```

**G. Add emergency-specific debounce**
Once an EMERGENCY drawdown or L3 fires, suppress repeat invocations for 15 min
(the loop's own emergency_sell handles the actual sell — Claude can't help faster).

## What Could Break

1. **Missed trade opportunities** — Claude sees fewer signals, may miss fast-moving setups.
   Mitigated: Chronos/Ministral/signals still run every 90s. Claude sees accumulated data
   when it does fire. Trade queue still works.
2. **Delayed reaction to danger** — L1 no longer invokes Claude, so the first Claude call
   in a decline is at L2 (5% from stop). Mitigated: L3 emergency auto-sell is pure Python
   (no Claude needed). Hardware stops on Avanza are independent.
3. **Stale Telegram updates** — user gets fewer Telegram messages from Layer 2.
   Mitigated: the loop's own status logging and the digest system still run.

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Invocations during volatile hour | 12 (all T3 Opus) | 1-2 (mostly T1 Haiku) |
| MIN_INVOKE_INTERVAL | 5 min | 30 min |
| L1 WARNING triggers Claude | Yes | No (log only) |
| "Dropped from peak" fires at | 3% | 8% |
| Price move trigger | 2% | 5% |
| Heartbeat | Every 30 min | Every 2 hours |
| Estimated daily invocations | 50-100 | 5-15 |

## Execution Order

### Batch 1: Config constants (lines 87-142)
- Raise `MIN_INVOKE_INTERVAL` to 1800
- Raise `TRIGGER_PRICE_MOVE` to 5.0
- Raise `TRIGGER_TRAILING` to 8.0
- Raise `HEARTBEAT_CHECKS` to 80

### Batch 2: check_triggers() — stop L1 from triggering invocations
- L1 WARNING: log only, do NOT append to reasons
- Keep L2 and L3 as invocation triggers

### Batch 3: classify_tier() — fix over-escalation
- Remove "stop-loss" from critical_patterns (matches L1 WARNING text)
- Remove `len(reasons) >= 2 → T2` escalation
- Only T3 for: L2 ALERT, L3 EMERGENCY, EMERGENCY drawdown, end_of_day
- Default everything else to T1 (Haiku)

### Batch 4: Verify syntax, commit, push, restart loop
