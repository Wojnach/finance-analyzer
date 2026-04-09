# Round 4 Dual Adversarial Review — 2026-04-09

## Context
- Round 3 (2026-04-08): 67 findings (15 CRITICAL, 35 HIGH, 16 MEDIUM, 1 LOW)
- ~22 Round 2 findings still open in Round 3
- Since Round 3: ~2,056 lines changed across 16 files
- Key changes: metals_swing_trader overhaul (+510), fingpt daemon (+246), sentiment.py rewrite (+260)

## Goals
1. Verify which Round 3 findings were fixed by recent commits
2. Review all new code since Round 3 (16 changed files)
3. Find new bugs introduced by the changes
4. Deep cross-cutting analysis of persistent patterns
5. Produce actionable synthesis with severity rankings

## 8 Subsystem Partition

| # | Subsystem | ~Lines | Focus for Round 4 |
|---|-----------|--------|-------------------|
| 1 | signals-core | 5,892 | C1 ADX cache, C2 accuracy race, H1 horizon key, H3 fail-open |
| 2 | orchestration | 4,981 | C3 wait_for_specialists, C4 T3 dead, main.py changes, trigger.py changes |
| 3 | portfolio-risk | 4,410 | C5 guard always open, C6 drawdown disconnected, H19 Sortino, H20 CVaR |
| 4 | metals-core | 15,588 | **HEAVY**: swing_trader overhaul, C12-C15, H27-H34, 15+ Rule 4 violations |
| 5 | avanza-api | 4,487 | C7 buying power, C8 CONFIRM order, H4 stop ID, new fetch_positions |
| 6 | signals-modules | 10,597 | H13 structure, H14 calendar, H15 smart_money, H17 VWAP, H35 futures_flow |
| 7 | data-external | 3,683 | C9 earnings, H10 NFP, H11 yfinance lock, H12 onchain budget, sentiment.py rewrite |
| 8 | infrastructure | 6,136 | C10 health race, C11 loading_keys, H23 GPU lock, H25 log rotation, H26 Telegram |

## Methodology
- 8 parallel code-reviewer agents (one per subsystem)
- 1 independent manual review (cross-cutting focus)
- Cross-critique in both directions
- Synthesis with severity classification and action plan

## Execution
1. Write plan ✓
2. Commit plan
3. Launch 8 agents in parallel (background)
4. Read key changed files + write independent review
5. Collect agent results
6. Cross-critique
7. Write synthesis doc
8. Commit all, merge, push, clean up
