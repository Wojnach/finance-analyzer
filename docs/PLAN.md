# Plan: Metals Intraday Trader with Claude Decision-Making

**Date:** Mar 2, 2026 (updated)

## Goal

Build a Layer 1/Layer 2 system for intraday metals warrant trading on Avanza.
Layer 1 (Python) collects price data every 90 seconds. When conditions warrant,
it invokes Claude Code (Layer 2) which analyzes the full context and decides
whether to buy/sell/hold, then executes trades via the Avanza API.

## Architecture

```
metals_loop.py (Layer 1)           Claude Code (Layer 2)
+---------------------+           +----------------------+
| Every 90s:          |  trigger  | Reads metals_context  |
| - Fetch 3 prices    |---------->| + metals_decisions    |
| - Track peaks/lows  |           | Analyzes:             |
| - Read main signals |           |  - P&L from ENTRY     |
| - Detect triggers   |  tier 1-3 |  - Signal consensus   |
| - Write context JSON|---------->|  - Previous decisions  |
| - Classify tier     |           |  - ATH thesis          |
|                     |<----------| Decides: BUY/SELL/HOLD|
|                     |  results  | Logs decision history |
+---------------------+           | Sends Telegram        |
                                  +----------------------+
```

### Token Cost Management (Claude Max subscription)

Tiered model selection to stay within daily budget (~75K tokens/day):

| Tier | Model  | Timeout | Max Turns | Use Case                              | Frequency   |
|------|--------|---------|-----------|---------------------------------------| ------------|
| T1   | Haiku  | 60s     | 8         | Workhorse: price moves, trails, beats | ~25/day     |
| T2   | Sonnet | 180s    | 15        | Multi-trigger: 2+ triggers at once    | ~4/day      |
| T3   | Opus   | 300s    | 20        | Critical: stop proximity, profit, EOD | ~1-2/day    |

Haiku is the cheapest and handles the bulk of invocations. Sonnet only fires when
multiple triggers coincide (requiring deeper analysis). Opus reserved for decisions
that affect real money (stop zones, profit targets, end-of-day).

Guards:
- **Never invoke if already running** (poll check)
- **5 min minimum cooldown** between invocations
- **Startup grace period** — first check establishes baseline without triggering
- **Signal flip requires sustained change** (not single-check noise)

### Trigger Conditions (invoke Claude)
1. Price moved >2% from last invocation → T1 (haiku)
2. Trailing stop: bid dropped 3%+ from session peak → T1 (haiku)
3. Profit target: any position +4%+ from entry → T3 (opus)
4. Signal consensus flip (XAG or XAU changed) → T1 (haiku)
5. Periodic heartbeat (every ~30 min) → T1 (haiku)
6. End-of-day (17:00 CET) → T3 (opus)
7. Hard stop proximity (within 5% of stop-loss) → T3 (opus)
8. Multiple triggers simultaneously → T2 (sonnet)

### Decision History: `data/metals_decisions.jsonl`
Every Claude invocation appends a JSON line with:
- Current prices and P&L
- Action taken per position (HOLD/SELL)
- Reflection on previous decision accuracy
- Prediction (direction, confidence, horizon) for future accuracy tracking

The next invocation reads the last 5 entries to inform its decision.

### Strategic Thesis
Silver bull 2026: target $120/oz ATH. Bias toward HOLD. Only sell on structure break.

### Context File: `data/metals_context.json`
Written by Layer 1, read by Layer 2 Claude. Contains all data needed for decision,
including recent_decisions (last 5 from metals_decisions.jsonl).

## Files

| File | Status | Purpose |
|------|--------|---------|
| `data/metals_loop.py` | DONE | Layer 1: data collection, triggers, tiered invocation |
| `data/metals_agent_prompt.txt` | DONE | Layer 2: decision prompt with history + ATH thesis |
| `data/metals_context.json` | Auto-generated | Written by L1, read by L2 |
| `data/metals_decisions.jsonl` | Auto-generated | Decision history + accuracy tracking |
| `data/metals_agent.log` | Auto-generated | Claude subprocess output log |
| `data/metals_trades.jsonl` | Auto-generated | Trade execution log |
| `data/metals_monitor_v2.py` | Running | Passive price alerter (independent) |
| `docs/PLAN.md` | This file | Architecture plan |

## What NOT to modify
- metals_monitor_v2.py (keep running as passive monitor)
- portfolio/main.py or any core portfolio modules
- config.json

## Risks & Mitigations
- **Nested session**: strip CLAUDECODE env var (same fix as agent_invocation.py) ✅
- **Session expiry**: BankID valid ~24h. Monitor for 401s.
- **Token budget**: Tiered models + 5min cooldown + heartbeat at 30min ✅
- **Double invoke**: Poll check + cooldown guard ✅
- **Double trading**: This system only touches warrants, main system only simulated
- **Decision drift**: Decision history + reflection loop ✅

## Next Steps
1. ✅ Dry-run test (prices, session, claude CLI all verified)
2. Launch metals_loop.py as background process
3. Monitor first few invocations via agent.log
4. Build accuracy review script (compare predictions vs outcomes)
