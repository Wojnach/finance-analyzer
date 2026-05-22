# Improvement Plan — Auto-Session 2026-05-22

Created: 2026-05-22
Branch: `improve/auto-session-2026-05-22`
Prior sessions: 6 auto-sessions (05-04 through 05-21), adversarial review (05-21).

## Context

Based on 5 parallel exploration agents + adversarial review synthesis (300+ findings, 73+ P0s).

## Bugs & Fixes

### Batch 1: Data Quality (affects trading decisions)

| ID | File(s) | Bug | Fix |
|----|---------|-----|-----|
| B1 | monte_carlo.py:154, monte_carlo_risk.py:227, exit_optimizer.py:173+184, price_targets.py:40-41+511 | 252 trading-days used for crypto/metals (should be 365). Biases MC probs ~20%. | Parameterize per asset class. equity_curve.py already uses 365. |
| B2 | agent_invocation.py:1511 | Incomplete stub journal writes "HOLD" — poisons accuracy stats with fake decisions. | Change to "NO_DECISION". Ensure consumers skip it. |
| B3 | outcome_tracker.py:471-472,458 | change_pct rounded to 2 decimals (excludes borderline outcomes). cache_key hour-bucket shares price across horizons. | Round to 4 decimals. Cache by minute precision. |
| B4 | file_utils.py:89 | OSError logged at DEBUG. Antivirus lock → empty portfolio → false drawdown. | Elevate to WARNING. |

### Batch 2: Concurrency / Safety

| ID | File(s) | Bug | Fix |
|----|---------|-----|-----|
| B5 | file_utils.py:379-419 | prune_jsonl skips sidecar lock. Concurrent append lost during prune. | Wrap read-rewrite-replace in jsonl_sidecar_lock. |
| B6 | api_utils.py:30-31 | Raw open()+json.load() on config.json symlink. CLAUDE.md rule #4. | Use file_utils.load_json(). |
| B7 | fin_fish.py:735 | `pass` instead of `continue`. Knocked-out BEAR certs still ranked. | Replace pass with continue. |

### Batch 3: Trigger Gate

| ID | File(s) | Bug | Fix |
|----|---------|-----|-----|
| B8 | trigger.py:106 | SUSTAINED_DURATION_S=120 at 600s cadence = 1-cycle gate. | Raise to 900s. |

## Skipped (Too Risky)

- P0 #0: Barrier-blind stops — 10+ files, real-money paths. TODO: MANUAL REVIEW.
- P0 #1: Layer 2 Edit/Write/Bash tools — security design session needed.
- P0 #7: Signal core gate pre/post persistence — 4400-line file, high risk.
- P0 #11: Dashboard cookie = master token — could lock user out.
