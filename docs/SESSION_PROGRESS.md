# Session Progress — After-Hours Research (2026-05-02 → 2026-05-03)

**Session start:** 2026-05-02 ~23:00 CET (21:00 UTC)
**Status:** COMPLETE
**Branch:** `research/daily-20260502` (worktree, merged + removed)

---

## 2026-05-03 evening — utility_overlay perf regression fix

**Trigger:** during `/fin-status` we observed cycles taking 12 min and the
`utility_overlay` phase consistently clocking 110-114 s/ticker (matched
April's BUG-178 magnitude even though the fix had shipped).

**Root cause:** `_compute_signal_utility` at `accuracy_stats.py:609` raised
`TypeError: bad operand type for abs(): 'NoneType'` whenever the entries
included a `change_pct=None` outcome. The 2026-04-22 None-guard fix had
been applied to `_vote_correct` (line 112) and one other site (line 1636)
but **missed this function**. The exception was silently swallowed by the
broad `except` at `signal_engine.py:3486`, so the cache populate-on-success
line never ran. Every call paid cold compute (~2.5 s sequential, ~49 s
under 4-thread contention). The intermittency we saw — sometimes 3.6 s,
sometimes 110 s — depended on which horizons had unbackfilled None
outcomes at any given cycle.

**Fix:** one-line `change_pct is None` guard mirroring the established
pattern. Profile confirms warm-cache 0 ms, cold sequential 2.5 s, cold
parallel-4 49 s wall.

**Commits:**
- `b2bd9dce` fix(BUG-178): None-guard in _compute_signal_utility
- `dede91ec` fix(review): address adversarial review findings on b2bd9dce

**Tests:** 24/24 in test_signal_utility.py (incl. new TestNoneChangePct
covering no-crash + skip-like-neutral via the explicit `entries=` bypass
that avoids cache leakage between tests). Full suite: 7650 passed, 43
failed — all 43 confirmed pre-existing by re-running the suspect tests
on parent commit `ef486cb4`.

**Adversarial review:** fresh code-reviewer subagent flagged 2 P2s
(test cache-leakage, profile harness wrote to disk despite "read-only"
docstring) — both fixed in `dede91ec`.

**New artifact:** `scripts/perf/profile_utility_overlay.py` — pure-observer
profile harness for the `utility_overlay` phase. Useful next time this
class of regression appears.

---

## What was done

### Phase 0-4: Research (8-phase protocol)
- **Daily Review**: 20+ Layer 2 invocations on May 2, ALL HOLD. System correctly restrained.
- **Market Research**: Weekend session, markets closed. BTC ~$78K, Silver ~$76. MSTR earnings May 5.
- **Quant Research**: IC-based weighting plan confirmed. Bayesian CPD for signal health identified.
- **Signal Audit**: CRITICAL finding — 7 signals are 93-100% BUY-only, inflating consensus by +5 net BUY votes. Regime accuracy inverted: 20-22% in trending-down (worse than random).

### Phase 5: Plan
- Wrote `docs/RESEARCH_PLAN.md` with 3 batches:
  - Batch 1: Per-ticker blacklist expansion (HIGH IMPACT, EASY)
  - Batch 2: Directional bias penalty (HIGH IMPACT, MEDIUM)
  - Batch 3: Decision feedback loop for Layer 2 (MEDIUM IMPACT, EASY)

### Phase 6-7: Implementation

**Batch 1 — Per-ticker blacklist expansion** (commit `f8fe3a77`):
- XAG-USD: added `sentiment` to `_default` (33.3% accuracy, 285 samples)
- MSTR `_default`: added `statistical_jump_regime` (27.0%, 74 sam), `realized_skewness` (36.0%, 50 sam)
- MSTR `1d`: added `macro_regime` (40.3%, 1475 sam) — moved from `_default` to preserve good 3h performance (81.4%)
- 5 new tests in `TestMay2BlacklistExpansion`

**Batch 2 — Direction-aware bias penalty** (commit `dd3fe799`):
- Changed bias penalty from direction-agnostic (penalizes ALL votes) to direction-aware (only penalizes in-bias votes)
- BUY-biased signals (calendar, crypto_evrp, funding, onchain, etc.) get 0.5x when voting BUY, but keep full weight on rare contrarian SELL
- Uses runtime `buy_rate`/`sell_rate` from activation data to determine bias direction
- 6 tests: in-bias penalized, contrarian preserved, SELL-biased contrarian, below threshold, few samples
- Key insight: rare contrarian signals from biased sources carry more Shannon information

**Batch 3 — Decision feedback loop** (commit `ef486cb4`):
- Added `_build_decision_feedback(ticker, max_entries=5)` to `agent_invocation.py`
- Scans `layer2_journal.jsonl` most-recent-first for entries mentioning trigger ticker
- Formats last 5 decisions with actions and prices into prompt context
- Injected after drawdown/guard context, wrapped in try/except (never fatal)
- 6 tests: empty journal, no match, formatting, max entries, missing price

**Merge + Push**:
- Fast-forward merged into main, pushed to origin
- 303 tests passed across both changed files (107 signal_engine + 196 agent_invocation)

### Phase 8: Morning Briefing
- Wrote `data/morning_briefing.json` (May 3 briefing)

## Key Research Findings

1. **Directional bias is massive**: 7 perma-BUY signals (+5 phantom votes), 2 perma-SELL signals (-2). Net +5 BUY bias in every consensus regardless of market conditions.
2. **Regime accuracy inversion**: 20-22% in trending-down (predicts wrong 80% of time). "Unknown" regime has BEST accuracy (57-63%).
3. **Per-ticker divergence**: credit_spread_risk 67% BTC vs 17% XAG. sentiment 33% on XAG. macro_regime 40% on MSTR at 1d but 81% at 3h.
4. **Correlation clusters**: ema+trend+macro_regime+structure: 95-100% agreement (6 signals = 1 vote). Meta-cluster dedup (yesterday) partially addresses this.

## What's next
- **Live validation**: Monitor bias penalty + blacklist impact on consensus quality 24-48h.
- **IC-based signal weighting**: Plan ready in `memory/quant_research_priorities.md`. Highest-priority deferred item.
- **Bayesian Online CPD**: Auto-detect signal accuracy collapses (would have caught claude_fundamental collapse weeks earlier).
- **fear_greed investigation**: 74.3% ranging accuracy but HOLD-only for 30 days — is the threshold too tight?
- **MSTR earnings May 5**: Potential trigger for BTC+MSTR positions.
- **Regime model**: Current binary detection may add noise. "Unknown" outperforms all known regimes.

### 2026-05-03 18:28 UTC | main
b2bd9dce fix(BUG-178): None-guard in _compute_signal_utility — cache had been silently broken since 2026-04-22
portfolio/accuracy_stats.py
scripts/perf/profile_utility_overlay.py
tests/test_signal_utility.py

### 2026-05-03 18:34 UTC | main
dede91ec fix(review): address adversarial review findings on b2bd9dce
scripts/perf/profile_utility_overlay.py
tests/test_signal_utility.py
