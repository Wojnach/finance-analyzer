# FGL Adversarial Review — 2026-05-30

Full-codebase adversarial review per the `/fgl` protocol. 8 subsystems, 8 fresh
review subagents in parallel + an independent orchestrator pass, cross-critiqued
against source and live runtime data.

## Read in this order
1. **[02-SYNTHESIS.md](02-SYNTHESIS.md)** — start here. Executive summary, the P0
   master list, the live-data correction of the `contract_violation` root cause,
   the 5 cross-cutting meta-themes, and the remediation roadmap.
2. **[01-subsystem-findings.md](01-subsystem-findings.md)** — all ~76 subagent
   findings verbatim, per subsystem, `path:line — Pn — problem. fix.`
3. **[00-own-pass.md](00-own-pass.md)** — orchestrator's independent foundation
   review (atomic I/O, locking, shared state, Layer-2 path) + cross-critique seeds.

## Headline results
- **5 P0s:** silent Avanza session-expiry outage; stop-loss reachable via the wrong
  endpoint; negative warrant value (no knockout floor); `price_source` silent
  stale-yfinance fallback; Layer-2 `failed` writes no journal stub (real silent crash).
- **Flagship correction:** the ~20×/week `contract_violation` is **72% `success`-lag**
  (contract window/timestamp defect), not the `skipped_busy` clobber two independent
  static analyses agreed on (~3%). Only live-journal validation caught it.
- **5 meta-themes** (each = one structural fix closing a class of bugs): atomic-RMW
  bypass · silent stale-data fallback · producer/consumer contract drift · EOD-flat
  reachability for leveraged inventory · constant-price history reconstruction.

## Scope
- Baseline: `main @ 1730651f`. Review surface: ~120K LOC across `portfolio/` (167
  top-level + 74 signals + 5 subpackages) and `data/` (35 modules).
- Review is **documentation only** — no code changed. Fixes are scheduled in the
  §4 roadmap; none applied here (live-config/threshold changes need human approval
  per the playbook).

## Reproduce
Empty-baseline branch `fgl/empty-baseline` (orphan, empty tree) lets a diff
reviewer see a whole subsystem as additions:
`git diff fgl/empty-baseline -- portfolio/signal_engine.py`.
(The branch + the `Q:/fa-fgl-review` worktree were cleaned up after the review;
recreate with `git commit-tree $(git hash-object -t tree /dev/null) -m base`.)
