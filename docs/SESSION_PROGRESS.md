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

## 2026-05-03 night — Cold-start performance follow-ups

After the utility_overlay fix landed, cycles 1 of subsequent restarts still
paid two distinct cold-start costs that became dominant once
`utility_overlay` was no longer the long pole:

### Issue A — GPU contention from Kronos subprocess
Forecast phase took ~210 s on cold-start. Kronos (subprocess that holds
the GPU file-lock during model-load + inference) ran first across 4
ticker threads in parallel. Chronos (in-process, ~50 ms warm, ~1.7 s
cold-load) timed out behind it on a 120 s gate.

**Fix** (`789cc91c`): swap the order in `portfolio/signals/forecast.py`
so Chronos runs first. Chronos pipelines through GPU in seconds for all
4 tickers; Kronos shadow then runs and threads that can't grab the gate
within 90 s skip silently — fine because Kronos is in shadow mode and
its sub-signals are filtered from live consensus.

Block-move only, no logic change. Adversarial review (fresh
code-reviewer subagent): clean, no findings.

### Issue B — Per-restart cold-compute on `signal_utility`
Even after the None-guard fix, every process restart paid ~49 s under
4-thread contention because `_signal_utility_cache` is in-memory only —
empty on every fresh process.

**Fix** (`7416a6fd` + `5c476cbc`): persist `signal_utility` results to
disk as an L2 cache at `data/signal_utility_cache.json`, mirroring the
`regime_accuracy_cache.json` pattern: single global "time" key gates
TTL (3600 s, matches `ACCURACY_CACHE_TTL`), per-horizon data persists
via load-merge-write. Lookup order: L1 in-memory → L2 disk → compute.
Both layers populate on successful compute.

Adversarial review (fresh code-reviewer subagent) flagged 2 P2 findings,
both fixed in `5c476cbc`:
1. **Multi-horizon write race**: lockless read-merge-write would lose
   3 of 4 horizons on a 4-thread cold cycle (each thread overwrites
   the others' merges). Added `_signal_utility_disk_lock` separate from
   the L1 lock so disk IO doesn't block L1 reads. New regression test
   `test_l2_concurrent_different_horizons_all_persist` spawns 4
   ThreadPoolExecutor writers and asserts all 4 horizons survive.
2. **Cross-process invalidation scope**: `invalidate_signal_utility_cache`
   deletes the shared L2 file. Verified by grep that satellite loops
   (crypto/oil/metals) don't call it; only `outcome_tracker.py` does
   (which runs as the daily PF-OutcomeCheck task — exactly the right
   caller). Updated docstring with explicit cross-process scope.

### Live verification (cold-start cycle 1, all fixes deployed)

| Stage | Cycle wall | utility_overlay phase |
|---|---|---|
| Pre-everything | 387.7 s | ~110 s × 4 |
| + None-guard fix | 332.1 s | 9–17 s × 2 |
| + Chronos-first swap | 140.4 s | 58–62 s × 4 |
| + L2 disk cache | **131.1 s** | **0.0 s × 4** |

The remaining 131 s is genuinely-required cold-start work (BERT, Chronos
model load, llama-server warmup, LLM batch). None of my fixes target it
because all of those are first-call costs that warm-cycles already avoid.

### Operational lesson learned

`schtasks /end` terminates the scheduled-task wrapper, not the
worker python.exe. The worker holds the singleton lock at
`data/main_loop.singleton.lock`. Subsequent `schtasks /run` instances
detect the lock and exit with code 11. **You must `taskkill /pid <pid> /f`
the worker python.exe directly before running `schtasks /run`**, or the
new code is never loaded — even though `schtasks /query` shows "Running"
and a Get-CimInstance for python.exe shows fresh PIDs (those are stale
launchers, not the active worker).

This is documented at `docs/GUIDELINES.md` step 9 but easy to skip. Cost
this evening: ~3 hours of confused observations before realizing the live
loop was running pre-Chronos-swap, pre-L2 code despite three "successful"
restart cycles.

### Commits this session

| SHA | Subject |
|---|---|
| `b2bd9dce` | fix(BUG-178): None-guard in _compute_signal_utility |
| `dede91ec` | fix(review): test cache-leakage + profile harness write |
| `2a6da0fa` | docs(session): record utility_overlay perf fix |
| `789cc91c` | perf(forecast): run Chronos before Kronos |
| `7416a6fd` | perf(accuracy_stats): persist signal_utility cache to disk |
| `5c476cbc` | fix(review): multi-horizon write race + cross-process docstring |

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

### 2026-05-03 22:04 UTC | main
bcd919e0 feat(loop): pre-warm dashboard accuracy cache once per hour
portfolio/accuracy_stats.py
portfolio/main.py
tests/test_accuracy_compute_lock.py

### 2026-05-03 22:23 UTC | main
99115711 fix(dashboard): dual-stack IPv4+IPv6 bind — eliminates Windows localhost 2s delay
dashboard/app.py

### 2026-05-03 22:26 UTC | feat/loop-infra-cleanup-2026-05-04
21fbec8f plan: loop-infra cleanup (2026-05-04)
docs/plans/2026-05-04-loop-infra-cleanup.md

### 2026-05-03 22:27 UTC | feat/loop-infra-cleanup-2026-05-04
e9d5e0d1 refactor(loops): migrate crypto/oil/mstr write_heartbeat shims to shared helper
data/crypto_loop.py
data/oil_loop.py
portfolio/mstr_loop/loop.py
tests/test_loop_health_write_heartbeat.py

### 2026-05-03 22:30 UTC | feat/loop-infra-cleanup-2026-05-04
9d5e5328 feat(accuracy): persist dashboard prewarm timestamp across loop restarts
.gitignore
portfolio/accuracy_stats.py
tests/test_accuracy_compute_lock.py

### 2026-05-03 22:30 UTC | feat/loop-infra-cleanup-2026-05-04
55692d86 docs(claude.md): replace stale dashboard endpoint list with reconciled 32
CLAUDE.md

### 2026-05-03 22:31 UTC | feat/loop-infra-cleanup-2026-05-04
3b0a3d78 test(prewarm): bypass disk lazy-load in TestDashboardAccuracyPrewarm reset
tests/test_accuracy_compute_lock.py

### 2026-05-03 22:32 UTC | main
8558fb5a docs(session): cold-start perf follow-ups + ops lesson
docs/SESSION_PROGRESS.md

### 2026-05-03 23:00 UTC | fix/bert-meta-tensor-2026-05-04
88c2a827 docs(plan): bert_sentiment meta-tensor defensive load
docs/PLAN_bert_meta_fix.md

### 2026-05-03 23:01 UTC | main
faaa32e6 fix(dashboard): _read_jsonl seeks from end of file — 139x speedup on /api/golddigger
dashboard/app.py
tests/test_dashboard.py

### 2026-05-03 23:03 UTC | fix/bert-meta-tensor-2026-05-04
a03a5f14 fix(bert_sentiment): defensive meta-tensor detection at load time
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

### 2026-05-03 23:10 UTC | feat/dashboard-avanza-view-2026-05-04
c6ccb642 feat(dashboard): Avanza account view + click-feedback on refresh buttons
dashboard/app.py
dashboard/static/js/main.js
dashboard/static/js/views/avanza.js
dashboard/static/js/views/more.js
dashboard/static/js/views/settings.js
tests/test_dashboard_avanza_account.py

### 2026-05-03 23:11 UTC | fix/bert-meta-tensor-2026-05-04
2c646026 fix(bert_sentiment): also walk buffers for meta-tensor check
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

### 2026-05-03 23:11 UTC | 
c6581c5f docs(plan): bert_sentiment meta-tensor defensive load
docs/PLAN_bert_meta_fix.md

### 2026-05-03 23:11 UTC | 
b46553db fix(bert_sentiment): defensive meta-tensor detection at load time
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

### 2026-05-03 23:11 UTC | 
f1a406b4 fix(bert_sentiment): also walk buffers for meta-tensor check
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

### 2026-05-03 23:13 UTC | main
f77de36a fix(conftest): redirect SIGNAL_UTILITY_CACHE_FILE to session tmpdir
tests/conftest.py

### 2026-05-03 23:13 UTC | fix/codex-review-followups-2026-05-04
4fcb4104 plan: codex review followups (2026-05-04)
docs/plans/2026-05-04-codex-review-followups.md

### 2026-05-03 23:23 UTC | fix/codex-review-followups-2026-05-04
c2bdfd18 fix(codex-review): 4 findings from adversarial review of 8558fb5a..faaa32e6
.gitignore
dashboard/app.py
data/crypto_loop.py
data/oil_loop.py
portfolio/accuracy_stats.py
portfolio/file_utils.py
tests/test_accuracy_compute_lock.py
tests/test_dashboard.py
tests/test_loop_health_write_heartbeat.py

### 2026-05-03 23:24 UTC | feat/dashboard-avanza-view-2026-05-04
5c7bdde7 fix(dashboard): codex P1+P2 fixes for avanza view + Live prices + Assets + history hint
dashboard/app.py
dashboard/static/js/charts/accuracy-chart.js
dashboard/static/js/main.js
dashboard/static/js/views/assets.js
dashboard/static/js/views/avanza.js
dashboard/static/js/views/more.js
dashboard/static/js/views/prices.js
dashboard/static/js/views/signals.js
tests/test_dashboard_avanza_account.py

### 2026-05-03 23:32 UTC | main
b0048e1d fix(dashboard): switch /api/avanza_account to portfolio.avanza_session
dashboard/app.py
tests/test_dashboard_avanza_account.py

### 2026-05-04 ~01:00-01:13 CEST | fix/bert-meta-tensor-2026-05-04 → main f1a406b4
c6581c5f docs(plan): bert_sentiment meta-tensor defensive load (post-rebase)
b46553db fix(bert_sentiment): defensive meta-tensor detection at load time (post-rebase)
2c646026 fix(bert_sentiment): also walk buffers for meta-tensor check
docs/PLAN_bert_meta_fix.md
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

**What this was:** /fin-status caught a BERT FinBERT meta-device warning
in `data/loop_out.txt` at 00:27:34: `BERT FinBERT batched predict failed:
Tensor on device meta is not on the expected device cpu!`. ~20-30 such
warnings per cycle since the loop restart at 23:38:02 — every FinBERT
prediction silently failed and wrote a zero-confidence neutral
placeholder to `data/sentiment_ab_log.jsonl`.

**Root cause:** race between Chronos's CUDA load and concurrent BERT
loads via `main.py`'s ThreadPoolExecutor. Triggering commit was
`789cc91c` (perf/forecast Chronos-before-Kronos) at 21:08 UTC on
2026-05-03 — that commit moved Chronos's load into the parallel
ticker phase concurrent with BERT loads from `sentiment.py`.
`accelerate`'s lazy init can leave some FinBERT weights on the `meta`
device when CUDA init runs on another thread. Standalone repro of
FinBERT alone works fine; needs the loop's specific concurrent timing.

**Why FinBERT only:** loaded from a snapshot path
(`Q:\models\finbert\models--ProsusAI--finbert\snapshots\<hash>`)
without `cache_dir`/`local_files_only` kwargs. The snapshot dir
contains `pytorch_model.bin` + `flax_model.msgpack` + `tf_model.h5`,
which routes transformers into a path more sensitive to accelerate's
lazy init. CryptoBERT and Trading-Hero-LLM use the standard
`cache_dir + hf_name` pattern and don't hit it.

**What changed:**
- New `_has_meta_tensor(model)` walks both `parameters()` and
  `buffers()` (LayerNorm running mean/var live as buffers, not
  parameters; would slip past a parameters-only check).
- New `_model_load_kwargs(name, config, cache_dir)` extracts the
  FinBERT-vs-others dispatch so the same path resolution can be
  reused by the retry without duplicating the branching.
- After `from_pretrained(...)`, run `_has_meta_tensor`. If True:
  log a WARNING naming the model + race hypothesis, retry once with
  `torch_dtype=torch.float32, low_cpu_mem_usage=False`. If retry
  still has meta tensors, raise `RuntimeError(...)` with accelerate
  version + load_path for diagnostic correlation.
- `_get_model()` doesn't catch the RuntimeError, so the bad model is
  NOT cached; subsequent predict calls retry from scratch instead of
  compounding corruption.
- New `_accelerate_version()` helper used in the error message.
- 6 new tests in `TestMetaTensorRecovery` + smoke test for
  `_accelerate_version`. 21/21 bert_sentiment tests pass.

**Codex adversarial review:** spawned codex-rescue at effort xhigh
with 8 questions. Hit usage limit before emitting findings — the
streaming log shows it completed Phase 1 (code/diff/library
inspection) but didn't reach the final report. Self-reviewed my own
8 questions and found one valid concern (parameters() doesn't
include buffers); commit `2c646026` extends `_has_meta_tensor` to
walk both. Other 7 questions checked clean.

**Verification (LIVE in production at 01:12 CEST):**
```
01:12:18  GPU gate ACQUIRED by chronos
01:12:22  Loading BERT model Trading-Hero-LLM
01:12:23  Loading Chronos-2 model amazon/chronos-2 on cuda...   ← race window
01:12:24  Loading BERT model FinBERT from snapshot              ← race window
01:12:26  Chronos-2 model loaded
01:12:49  LLM batch: 4 Ministral queries                        ← cycle running
```
The race window was active (BERT/Chronos loads overlapped — same
fingerprint as the broken 23:38 load) but **zero `predict failed`
lines**, **zero `loaded with meta tensors` warnings**. Either the
race didn't trigger meta corruption this time, or it did and the
defensive check silently retried — either way, the symptom is gone.

**Loop restart procedure:** `taskkill /F /PID <loop-pid>` (per the
ops lesson from earlier tonight — `schtasks /end` can leave the
singleton lock held by a zombie file handle for 80+ min). Loop's
bat wrapper auto-restarted in <30s.

## What's next (optional follow-ups)

- **Serialize Chronos vs BERT loads** in `main.py` to eliminate the
  underlying race entirely. Cleaner than defensive detection but
  adds startup latency. Not urgent — defensive check handles it.
- **Watch for the WARN line in production:** `tail -F data/loop_out.txt
  | grep -aE "BERT.*meta tensors"`. If the race ever triggers in a
  way that needs the retry, you'll see one warning + clean recovery
  instead of 30 silent prediction failures per cycle.
- **Audit other `from_pretrained` call sites** for the same lazy-meta
  vulnerability if they run concurrently with CUDA-loading models.
  Quick grep: `grep -rn "from_pretrained" portfolio/ | grep -v test`.
