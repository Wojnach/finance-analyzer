# Session Progress

## Mobile Dashboard Redesign (2026-05-03 evening)

**Session start:** 2026-05-03 ~20:30 UTC
**Status:** MERGED — branch `feat/mobile-dashboard-redesign-2026-05-03`
removed after merge.
**Goal:** replace desktop-first single-file dashboard with a mobile-first
ES-module + PWA dashboard, preserving every endpoint and every
functional surface.

### Phases

1. Research (Tracks 1-6, 4 parallel agents + me) → 7 deliverables under
   `docs/research/2026-05-03-mobile-dashboard/`.
2. Spec at `docs/superpowers/specs/2026-05-03-mobile-dashboard-redesign-design.md`.
3. PLAN.md committed before any code changes (per `/fgl`).
4. Nine implementation batches (each committed independently):
   - Batch 1: skeleton + /legacy fallback + CSS tokens
   - Batch 2: state/fetch/router/polling/format/theme/main JS modules
   - Batch 3: 10 reusable UI components
   - Batch 4: Home view + Chart.js wrapper + sparkline
   - Batch 5: Decisions list + drill-down detail
   - Batch 6: Signals heatmap + accuracy + history
   - Batch 7: More + Health + Messages + Settings views
   - Batch 8: Metals + GoldDigger + Equity views + chart configs
   - Batch 9: PWA manifest + service worker + icons + skeleton tests + docs.
5. Codex adversarial review (`codex review --base main`) — 5 findings
   (3 P1, 2 P2, 1 P3): loop_health rollup unwrap, signal-heatmap shape,
   equity field names, GoldDigger normalized fields, /logout server
   endpoint for HttpOnly cookie, pulse navigation. All fixed in
   commit `ed67c288`.
6. Tests: 161 dashboard tests pass. Pre-existing full-suite failures
   verified on main HEAD before merge.

### Manual phone smoke test
Documented in `docs/TESTING.md`; required before each mobile-affecting PR.

---

## After-Hours Research (2026-05-02 → 2026-05-03)

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

### 2026-05-03 21:55 UTC | feat/gpu-gate-sweeper-2026-05-03
b258b37f plan: gpu_gate background sweeper for stale locks
docs/plans/2026-05-03-gpu-gate-sweeper.md

### 2026-05-03 ~21:30-23:40 CEST | fix/fingpt-batch-observability-2026-05-03 → main aa804a7f
8642e243 docs(plan): fingpt batch observability fix
b3b5c687 fix(llm_batch): make fingpt batch outcome legible from logs
aa804a7f fix(llm_batch): address codex P2+P3 — empty-text guard, log unit consistency
docs/PLAN_fingpt_observability.md
portfolio/llm_batch.py
tests/test_llm_batch.py

**What this was:** /fin-status caught what looked like a fingpt silent
failure in `data/loop_out.txt`: `"LLM batch: 0 results in 10.4s
(M:0 Q:0 F:6)"`. After ~30 min of wrong-direction probing (including a
`/v1/chat/completions` test that hit Qwen3 thinking-mode and returned
empty `message.content`), traced to the misleading log line at
`portfolio/llm_batch.py:258` — `results` only counted Phase 1+2
(Ministral/Qwen3); Phase 3 (fingpt) stashes via
`sentiment._stash_fingpt_result` and never appears in `results`. So a
fingpt-only cycle (every 3rd LLM cycle in the rotation) ALWAYS logged
"0 results" — whether fingpt produced 6 valid sentiments or silently
failed. Confirmed by grepping `data/sentiment_ab_log.jsonl` and finding
4 valid fingpt entries timestamped to the exact cycle that "reported"
0 results.

**What changed:**
- `_flush_fingpt_phase` now returns a metrics dict on every code path:
  `{queries, received, parsed, stashed_groups, exception}`. Implicit
  `None` return is gone.
- Summary log replaced with `"LLM batch: M=%d/%d Q=%d/%d F=%d/%d in
  %.1fs"` (parsed/queued for each phase). F=0/N now flags real silent
  failures.
- Per-failure-mode warnings inside `_flush_fingpt_phase`:
  - `"fingpt: server returned None for all N prompts"`
    (server/swap broke)
  - `"fingpt: parser returned None for K/N completions (>50%)"` (parser
    regression — same fingerprint as the 2026-04-09
    parser-defaulting-neutral incident)
  - top-level `except` now logs `repr(e)` for one-line operator triage
- `_parse_fingpt_completion` now treats empty/whitespace text as parse
  failure (codex P2). Production `fingpt_infer._parse_sentiment` falls
  back to "neutral" for unparseable input AND `llama_server._query_http`
  returns `""` (not None) for HTTP 200 with empty body. Without this
  guard, empty cycles silently scored as neutral parses.
- Phase-start log renamed `"%d fingpt queries"` → `"%d fingpt groups"`
  (codex P3) — `len(f_batch)` counts groups, not prompts.
- 10 new tests in `tests/test_llm_batch.py` (TestFingptPhaseMetrics +
  TestFlushLlmBatchSummaryLog), 36 total in the file. Existing 26 tests
  still pass.

**Codex adversarial review:** codex-rescue at effort xhigh returned
1×P2 + 2×P3. P2 was the empty-text false-success path. P3a was the
unit drift between phase-start and summary log denominators. P3b was
test-coverage gaps for realistic degradation paths (empty text, mixed
`[None, "", garbage, clean]`, import failure). All three addressed in
commit `aa804a7f` with 3 additional tests.

**Verification (LIVE in production at 23:38-23:40 CEST):**
```
23:38:25  LLM batch start: rotation_slot=warmup counter=0 queues M=4 Q=4 F=4
23:38:25  LLM batch: 4 Ministral queries
23:39:16  LLM batch: 4 Qwen3 queries
23:39:57  LLM batch: 4 fingpt groups          ← new label
23:40:04  LLM batch: M=4/4 Q=4/4 F=43/45 in 98.4s   ← new format
```
F=43/45 is the first real production data point — 2 fingpt prompts
produced empty/None content. The OLD format would have logged
"8 results" and hidden those two; the NEW format makes them visible.
Exactly what the fix is for.

**No live trading behavior changed.** Only loop logging + observability.

**Loop restart was bumpy:** `schtasks /end /tn PF-DataLoop` did not
kill the old loop process — singleton lock from PID 16396 persisted
(mtime 20:35) for ~80 min. The bat wrapper's auto-restart eventually
won at 23:37:52 when the old process finally died. Lesson: on Windows
the singleton lock can outlast the process if the OS hasn't reaped
file handles, and `schtasks /end` doesn't force-kill — for fast
restart use `taskkill /F /PID <loop-pid>`.

**Saved memory:** `reference_worktree_symlinks.md` — git worktrees
don't replicate the `config.json` symlink, causing 30-50 false test
failures in worktree pytest runs. Targeted tests still work; full
suite passes after merge.

## What's next (optional follow-ups, NOT in this PR)

- **Persist fingpt health to data/fingpt_health.json + contract alert:**
  metrics now exist per-cycle but aren't aggregated. If
  `parsed/queries < 0.5` for K consecutive cycles, the contract
  dispatcher should fire. Would have caught the 2026-04-09
  parser-defaulting-neutral regression weeks earlier.
- **Distinguish empty-text from server-None in warnings:** currently
  both flow through the parser-majority warning. A third category for
  "all responses non-None but empty content" would identify model
  truncation / Qwen3-thinking-mode-style failures separately from
  server connectivity.
