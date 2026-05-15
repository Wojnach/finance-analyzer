# Dual Adversarial Review — Plan (2026-05-14)

## Goal

Run a full dual adversarial review of the finance-analyzer codebase. For each of
8 subsystems, produce two independent adversarial reviews (Claude + Codex), then
have each side cross-critique the other, then synthesise a single prioritized
findings list.

## Why dual?

Single-reviewer reviews drift toward the reviewer's blind spots. Two reviewers
working independently surface a wider class of bugs; the cross-critique step
forces each to defend its own findings and challenge the other's. Where they
agree, confidence in the finding goes up. Where they disagree, the disagreement
itself is a signal worth investigating.

## Delta vs prior run (2026-05-04)

The partition is preserved verbatim — coupling and failure-mode coherence
haven't changed. File set updated for additions since 2026-05-04:

- **signals-core +1**: `signal_state_since.py`
- **metals-core +9**: `crypto_loop.py`, `crypto_monitor.py`, `crypto_swing_*.py`,
  `crypto_warrant_refresh.py`, `oil_loop.py`, `oil_swing_*.py`, `oil_warrant_refresh.py`,
  `orb_backtest.py`, `grid_fisher.py`, `grid_fisher_config.py`, `grid_tiers.py`,
  `oil_grid_signal.py`
- **avanza-api +1**: `avanza_account_check.py`
- **signals-modules +7 / -1**: added `breakeven_inflation_momentum`,
  `cubic_trend_persistence`, `gold_overnight_bias`, `intraday_seasonality`,
  `metals_vrp`, `treasury_risk_rotation`, `vwap_zscore_mr`; removed
  `crypto_cross_asset.py` (renamed/dropped)
- **data-external +2**: `onchain_data.py`, `oil_precompute.py`
- **infrastructure +1**: `llm_prewarmer.py`

Subsystem partition table unchanged:

| Subsystem | Files | Critical path |
|-----------|-------|---------------|
| 1. signals-core | signal engine + registry + accuracy + outcome tracking | Layer 1 voting math |
| 2. orchestration | main loop, agent invocation, trigger, journal, digest | The 60s cycle + Layer 2 spawn |
| 3. portfolio-risk | portfolio_mgr, trade_guards, risk_management, equity, MC | Trade decisions + sizing |
| 4. metals-core | metals_loop + swing trader + grid fisher + crypto/oil sibling loops + fish/orb engines | Avanza warrant trading + sister asset loops |
| 5. avanza-api | portfolio/avanza/* + session + orders + tracker | All Avanza HTTP |
| 6. signals-modules | portfolio/signals/* (50 modules) | Per-signal logic |
| 7. data-external | data_collector, fear/greed, sentiment, FAPI, on-chain, news | Market data ingestion |
| 8. infrastructure | file_utils, http_retry, health, gates, telegram, logging, LLM plumbing | Cross-cutting plumbing |

Exact file lists are in `subsystems.txt`.

## Execution protocol

1. **Worktree** — `git worktree add Q:/fa-review review/baseline-2026-05-14` from
   an empty initial commit (orphan). The main repo is untouched while review runs.
2. **8 review branches** — each branch is `baseline` + one commit containing
   only that subsystem's files. The single commit's diff is the entire subsystem,
   so `codex review --commit <SHA>` reviews it as if it were a fresh PR.
3. **Codex in background** — launch 8 `codex review --commit <SHA>` runs in
   parallel via background bash. Each writes stdout/stderr to a per-subsystem
   log file in `codex-raw/`. Do NOT block on them.
4. **Claude independent reviews** — while Codex runs, write 8 reviews to
   `docs/dual-review-2026-05-14/claude-<n>-<subsystem>.md`. Do NOT peek at any
   Codex log until all 8 Claude reviews are written. (Independence is the whole
   point.)
5. **Collect Codex** — once all 8 codex runs complete, save outputs to
   `docs/dual-review-2026-05-14/codex-<n>-<subsystem>.md`.
6. **Cross-critique** — for each subsystem, write
   `docs/dual-review-2026-05-14/cross-<n>-<subsystem>.md` answering:
   - Which Codex findings did Claude miss? Why?
   - Which Claude findings did Codex miss? Why?
   - Where do they disagree? Who is right?
   - What did *both* miss?
7. **Synthesis** — write `docs/dual-review-2026-05-14/SYNTHESIS.md` with a
   single prioritized punch-list (P0/P1/P2) drawn from the 24 review docs.
8. **Commit + push + cleanup** — commit the docs to main, push via Windows git,
   `git worktree remove Q:/fa-review`, delete review branches.

## What could break

- **Codex parallelism may exhaust API quota or be rate-limited.** Mitigation:
  if the first 4 finish cleanly, launch the next 4. If quota hits, fall back to
  serial execution.
- **Empty-baseline branch + selective checkout might miss imports.** Codex sees
  files in isolation; missing imports look like errors. We accept this — codex
  is reading code, not running it. Our prompt tells it the modules import from
  `portfolio/` and `data/` packages, which won't appear in the diff.
- **`codex review --commit` may not handle the largest diff (signals-modules,
  50 files).** Mitigation: if it errors, split signals-modules into 2 by alpha.

## Out of scope

- Implementing fixes — this is review-only. Findings go in SYNTHESIS.md as a
  punch list. Implementation is a follow-up session.
- Running tests after — no code changes are made, so no test run needed. The
  only files added to main are the 24 review docs + SYNTHESIS.md + this plan.

## Success criteria

- 8 codex reviews complete (or fail with documented reason).
- 8 Claude reviews written before any Codex output is read.
- 8 cross-critique docs written.
- SYNTHESIS.md has a prioritized P0/P1/P2 list with file:line references.
- Worktree cleaned up. Branches deleted. Main has 1 commit adding only docs.
