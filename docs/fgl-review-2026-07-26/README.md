# FGL Adversarial Review — 2026-07-26

Full-codebase adversarial review per the `/fgl` protocol. 9 subsystems, 9 fresh
review subagents in parallel + an independent orchestrator pass, cross-critiqued
against source and live runtime artefacts.

## Read in this order

1. **[02-SYNTHESIS.md](02-SYNTHESIS.md)** — start here. Executive summary, the P0
   master list, 5 cross-cutting meta-themes, the notable non-findings, and a
   remediation roadmap ordered by _"arms on next restart"_ rather than by severity
   label (several P0s are dormant only because the stack is intentionally down).
2. **[01-subsystem-findings.md](01-subsystem-findings.md)** — all subsystem
   findings, `path:line — Pn — problem. Fix: …`, each stamped
   VERIFIED / ACCEPTED / LATENT / REJECTED by the orchestrator.
3. **[00-own-pass.md](00-own-pass.md)** — orchestrator's independent pass
   (this session's own changes + cross-boundary contracts), including one
   orchestrator finding **rejected** by its own verification.

## Headline results

- **8 P0s.** Five are _wiring_ defects — correct code that production never
  reaches: `bar_ts` forwarded for staleness and never read; `update_state` and
  `record_warrant_transaction` with zero callers; the knockout floor implemented
  twice and never invoked; `_build_llm_context` skipped on the promoted path.
- **Flagship correction:** the same-day `asset_type` fixes (b5d2026b, 597176b2)
  are **incomplete**. phi4_mini, promoted on BTC/ETH/XAU/XAG, bypasses the branch
  that builds the rich LLM context and would vote from a prompt reading
  `Asset: XAG-USD (cryptocurrency)` / `Price: $0.00` / 9× `N/A`. Reproduced
  directly. Dormant only behind `data/local_llm.disabled` + loops-off.
- **Live data-honesty defects, visible right now:** Bold equity reports a −100%
  wipeout across 343/343 rows for a strategy whose state file doesn't exist; the
  #silver page renders `SHADOW` and `GATED_REMOTE_DOWN` for the same component in
  one card, from two disagreeing endpoints.
- **Dominant meta-theme:** _every_ safety mechanism added in the preceding 8 days
  has a hole on the one path that couldn't be executed when it was written. The
  fixes were verified on runnable paths; the defects live in the unrunnable ones.
- **Verified-clean list matters too:** warrant knockout floor, Layer-2
  silent-outage defence, confidence-cap ordering, metals quorum threading,
  registry promotion/blacklist separation, `price_source` fail-closed,
  `registry_defaults` freshness. Prior P0s that are now stale are marked as such.

## Scope

- Baseline: `main @ 597176b2` (2026-07-26). Surface: 440 Python + 62 JS files
  across `portfolio/` (incl. ~79 signal modules), `data/`, `dashboard/`,
  `scripts/`.
- Partition: the 8 canonical subsystems from the 2026-05-30 FGL plus a 9th,
  **web-control-registry**, for code that did not exist then (rebuilt dashboard,
  component registry, control write API, `tune_instrument.py`).
- Review is **documentation only** — no code changed, no services restarted, no
  control state mutated.

## Honesty notes

- **metals-core and infrastructure did not report**, so `data/metals_loop.py`
  (8011 lines) and `grid_fisher.py` are covered only indirectly. All other
  declared coverage gaps are listed in §6 of the synthesis. Absence of findings in
  an unreviewed file is _not_ evidence of cleanliness.
- Loops are intentionally stopped and Avanza intentionally down, so avanza-api and
  metals findings are static-traced rather than runtime-observed; reviewers stated
  this limitation themselves.
- Reviewers disproved eight of their own candidate findings against live data, and
  the orchestrator rejected one of its own. Those are recorded rather than deleted
  — a review that hides its false positives can't be calibrated.
- **Security:** during the review a reviewer printed the live BGeometrics
  `api_token` into its tool-call transcript and self-reported it. Low blast radius
  (free-tier, read-only, dead endpoint) but the credential should be considered
  exposed. Surfaced to the user; not actioned in this review.
