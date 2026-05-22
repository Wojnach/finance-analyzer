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

### Batch 4: Security / Safety

| ID | File(s) | Bug | Fix |
|----|---------|-----|-----|
| B9 | gpu_gate.py:82 | `_pid_alive` returns True when psutil missing → stale GPU locks never reaped (25h outage). | Return False (assume dead). |
| B10 | fish_instrument_finder.py:169 | No min barrier distance filter → auto-discovery picks closest-to-barrier MINIs. | Filter <5% barrier distance before ranking. |
| B11 | dashboard/auth.py:76 | Config read failure → {} → dashboard_token=None → all access allowed. | Keep previous cached config on read failure. |

### Batch 5: Stability / Correctness

| ID | File(s) | Bug | Fix |
|----|---------|-----|-----|
| B12 | agent_invocation.py:677 | _kill_overrun_agent nulls _agent_proc even when kill fails → duplicate spawn. | Only null on successful kill. |
| B13 | signal_engine.py:2346 | _topn_accuracy_key 0.5 default → nondeterministic top-N selection. | Add signal name as tiebreak key. |
| B14 | signal_engine.py:2842 | _adx_cache key (len, first, last) → content collision across tickers. | Expand to 6-field key (len, first, mid, last, high_max, low_min). |

## Skipped / Deferred

- B6: api_utils.py raw config read — false positive (mtime cache + threading lock + must-raise semantics correct).
- P0 #0: Barrier-blind stops — 10+ files, real-money paths. TODO: MANUAL REVIEW.
- P0 #1: Layer 2 Edit/Write/Bash tools — security design session needed.
- P0 #7: Signal core gate pre/post persistence — 4400-line file, high risk.
- P0 #11: Dashboard cookie = master token — could lock user out.
- P1: Auth-cooldown success-after-failure — needs window-based check, not single-entry.
- P1: 3d/5d/10d horizon → 1d accuracy collapse — needs per-horizon accuracy infrastructure.
