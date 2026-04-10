# Session Progress — Dual Adversarial Review 2026-04-10

## Status: SHIPPED (independent review), AGENTS PENDING

Full dual adversarial review of the entire finance-analyzer codebase (135 files, ~60K lines) across 8 subsystems. Independent manual analysis found 37 findings (7 P1, 20 P2, 10 P3). No P0 critical/money-losing issues.

### What shipped (1 commit, pushed)
- `4a0f344` docs(review): adversarial review — independent analysis + synthesis
  - `docs/adversarial-review/INDEPENDENT_REVIEW.md` — 37 findings across 8 subsystems
  - `docs/adversarial-review/SYNTHESIS.md` — prioritized fix list, health scorecard, methodology
  - `docs/adversarial-review/CROSS_CRITIQUE.md` — framework for dual-review cross-referencing

### Top 5 findings
1. **Subprocess governance** — bigbet.py, iskbets.py, agent_invocation.py bypass claude_gate.py
2. **Hardware trailing stop failure** — no fallback to legacy cascade stops in metals_loop
3. **Timezone consistency** — 8 naive datetime.now() calls in metals_loop
4. **Portfolio drawdown blind spot** — fallback to avg_cost_usd masks crashes
5. **JSONL append atomicity** — not truly atomic on Windows/NTFS

### 8 parallel agent reviews
- **signals-core**: COMPLETE — 10 findings (4 P1, 4 P2, 2 P3). Cross-critiqued and integrated.
  - Top finding: per-ticker accuracy override strips directional fields, disabling directional gate
- **orchestration**: Still running
- **portfolio-risk**: Still running
- **metals-core**: Still running (19K lines — largest subsystem)
- **avanza-api**: Still running
- **signals-modules**: Still running
- **data-external**: Still running
- **infrastructure**: Still running

Agent findings from signals-core integrated in commit 21dca8b.

### Positive patterns found
- Fail-closed accuracy gate (signal_engine.py:1807)
- 28 thread locks across the codebase
- Dogpile prevention in shared_state.py
- Circuit breaker pattern for API calls
- Kelly criterion properly guarded against edge cases

### Next priorities
- Fix the 5 findings above
- Integrate agent review results when they complete
- Cross-critique: identify agreement/disagreement between independent and agent reviews

---

# Session Progress — perf/llama-swap-reduction 2026-04-10

## Status: SHIPPED + MEASURED (merged d21ec5f, 17 rotation cycles confirmed working, ~62% reduction in mean LLM batch phase time)

User asked "we want all LLMs on GPU" and after I initially proposed BERT GPU migration, pushed back with "how many seconds does BERT take on the cpu?" and "how about looking into if we cana reduce the llama-server swap overhead?" Investigation showed BERT is ~0.5 s/cycle on CPU (negligible) and corrected my initial wrong estimate of llama-server swap overhead (70-120s/cycle): the real swap time is 15-18 s/cycle, with Ministral *inference* (40 s/cycle) being the actual bottleneck.

### What shipped (3 commits merged as d21ec5f)
1. **`7264889` perf(llama_server): active VRAM poll + KV cache reuse** — Replaces `time.sleep(4)` in `_start_server` with `_wait_for_vram_reclaim()` that polls `nvidia-smi` every 100 ms until ≥5.5 GB free, with 4s ceiling fallback. Adds `cache_prompt: True` to every `/completion` request for KV cache reuse across same-prefix batches. 11 new tests.
2. **`3062c01` perf(llm_batch): rotation scheduling for llama-server LLMs** — Rotates ministral → qwen3 → fingpt across successive batch flushes. Cold-start warmup runs all 3 once, then rotation kicks in. Uses new `shared_state._full_llm_cycle_count` counter + extended `_cached_or_enqueue(should_enqueue_fn=, max_stale_factor=)` signature. `max_stale_factor=5` at call sites gives 75 min stale tolerance covering the 45-60 min rotation period with slack. 14 new tests.
3. **`a168952` perf: address review findings N1+N2** — Treat cached `data is None` as "stale not available" so failed-flush cooldown cache entries auto-retry on next cycle regardless of rotation gate. Defensive test cleanup.

### Code review
- `pr-review-toolkit:code-reviewer` subagent: **NO BLOCKERS**, 6 minor nits (2 fixed: N1 None-cache handling, N2 test queue cleanup; 4 cosmetic left).
- 25 new tests, 281 adjacent tests pass, full suite: 6437 passed / 7 pre-existing failures (verified unrelated).

### Live measurement (4 hours, 17 rotation cycles)
- Old `PF-DataLoop` process PID 5956 was running since 2026-04-09 23:12 and ignored `schtasks /end`. Had to `powershell Stop-Process -Id 5956 -Force` to actually kill it. New process started at 10:34:22.
- **Warmup cycle (counter=0)** at 10:36:09: all 3 LLMs ran as expected. Total LLM batch 369.5 s, Signal loop done 475.6 s. Slow because of pre-existing Chronos+llama-server VRAM contention (Chronos resident at ~3.5 GB + llama-server at ~5 GB on a 10 GB card). Same pattern as the PRE-RESTART slow cycles (10:03=510 s, 10:15=489 s, 10:20=489 s) — NOT a regression from my changes.
- **17 rotation cycles confirmed (11:03 → 14:34)**: counter=1 ministral → 2 qwen3 → 3 fingpt → 4 ministral → … all the way to counter=17 qwen3. Every cycle ran exactly ONE LLM as designed.
- **Mean LLM batch phase: ~32 s** (vs 85 s baseline = **62% reduction**)
  - ministral-only cycles: 42-58 s (mean ~50 s)
  - qwen3-only cycles: 25-29 s (mean ~27 s)
  - fingpt-only cycles: 19-21 s (mean ~20 s)
- **Mean Signal loop done: ~85 s** on full-LLM cycles (vs 150 s baseline = **43% reduction**)
- **Lever 1 (active VRAM poll)**: First swap 4 s (vs 5-6 s baseline). Smaller win than expected because Chronos VRAM pressure caps the gain.
- **Stability**: zero "fingpt phase failed" warnings, zero "In-process BERT failed" warnings, all 17 rotation cycles successful. 2 partial-failure cycles (1 OK / 4 failed) at 11:56 and 14:06 were ticker fetch failures unrelated to LLM changes.

### Result: shipped, measured, rotation working as designed
- Memory updated: `project_llama_swap_reduction.md`
- Worktree cleaned up, branch deleted, main at d21ec5f

### Root cause of VRAM pressure (deferred)
Chronos-2 (~3.5 GB) stays resident in main.py's process between forecast calls. llama-server (~5 GB) needs to swap 8B models in/out. With PyTorch overhead, total is ~10+ GB on a 10.24 GB card. This predates my changes (the 10:03/10:15/10:20 slow cycles pre-ship). Possible fixes: (a) move Chronos to CPU (slow), (b) retire Chronos, (c) add an eager unload/reload gate around Chronos calls, (d) run Chronos in a separate process with its own VRAM budget.

---

# Session Progress — Auto-Improve Session 2026-04-10 (earlier)

## Status: COMPLETE (7 commits merged + pushed)

### What shipped
1. **CLAUDE.md refresh** — Updated signal counts (30→32 active, 34→36 total), ticker list (12→5 instruments after Apr 9 reduction), applicable counts (crypto=31, stocks=26, metals=28), added funding rate re-enable + on-chain BTC + credit_spread_risk entries.
2. **SYSTEM_OVERVIEW.md refresh** — Signal system (36 signals, 32 active, 4 disabled), test count (159→242 files), ticker table (MSTR only for stocks), signal inventory.
3. **Import sorting** — Ruff I001 auto-fix in 3 files (bert_sentiment, llm_batch, signal_engine).
4. **Metals lint cleanup** — 4× contextlib.suppress(), 2× any() builtin, 5× collapsed nested if across metals_loop.py + metals_swing_trader.py. Zero behavioral change.
5. **Test fix** — Crypto applicable count 29→31 in test_consensus.py (was failing on main).

### Test results
- Branch: 29 failures (all pre-existing) / 6512 passed
- Main before merge: 37 failures / 6531 passed
- Net: reduced failures by fixing consensus assertion, no new regressions

### Next priorities
- COT positioning re-enable if CFTC data quality improves
- HMM-based regime detection research
- Walk-forward Bayesian optimization for REGIME_WEIGHTS
- Remaining ruff violations: 87 (mostly E402 lazy imports, SIM117 cosmetic — intentional)

---

# Previous Session — After-Hours Research Agent 2026-04-09→10

## Status: COMPLETE (4 commits merged + pushed)

### What shipped (overnight research session)
1. **Funding Rate 3h-only** — removed from DISABLED_SIGNALS, horizon-gated via REGIME_GATED_SIGNALS. 74.2% at 3h (535 sam), blocked at 1d (29.9%). Crypto-only.
2. **On-Chain BTC Signal (#35)** — MVRV Z-Score, SOPR, NUPL, exchange netflow promoted to voting signal. Sub-metric majority vote. BTC-only, uses existing 12h cache.
3. **G/S Ratio Velocity** — 6th sub-indicator added to metals_cross_asset. 5d rate of change of gold/silver ratio. Captures momentum alongside mean-reversion zscore.
4. **Test count fixes** — Updated 5 test files for new signal counts (36 total, 31 crypto, 28 metals, 26 stocks). 15 new tests added.

### What to monitor
- Onchain signal accuracy over first 24h (new, zero history)
- Funding signal activations at 3h horizon in live loop
- G/S velocity threshold (±2%) — may need tuning
- CPI release today — high volatility expected

### Next priorities
- Consider COT positioning re-enable if CFTC data quality improves
- HMM-based regime detection research
- Walk-forward Bayesian optimization for REGIME_WEIGHTS

---

# Previous Session — Overnight log migration + fish None-guards 2026-04-09 late night

## Status: SHIPPED (14 commits on main, 7 codex review rounds addressed)

### Overnight summary

Autonomous `/fgl` session launched by user going to bed. Tackled the
two concrete Fleet v2 follow-ups from `docs/LOG_MIGRATION_AUDIT_20260409.md`:

1. **Fish script None-guards** — scripts/fish_straddle.py + scripts/fish_monitor_live.py
   were calling `get_buying_power().get('buying_power', 0)` without a None
   guard. After Fleet v2 Agent A's contract change (6a20c7d) the function
   returns dict|None on failure. Both sites now fail loud with explicit
   diagnostic + early return.

2. **Log migration stages 1-3** — the first three stages of Agent D's
   6-stage plan:
   - Stage 1: shim infrastructure (`log()`/`_log()` delegate to logger.info)
   - Stage 2: swing_trader ERROR sites → logger.exception/warning
   - Stage 3-A: metals_loop emergency + stops ERROR sites (~12 sites)
   - Stage 3-B: metals_loop catch-all ERROR sites (~14 sites)

### Commits (all on main, merged via FF after rebase onto BERT+fingpt work)

| # | Commit | Description |
|---|---|---|
| 1 | `4894a59` | plan(overnight): Fleet v2 follow-ups + log migration stages 1-3 |
| 2 | `7dc6cf9` | fix(fish): None-guard get_buying_power() call sites |
| 3 | `8245bb5` | feat(metals): log migration Stage 1 — shim log()/_log() |
| 4 | `0572497` | feat(metals): log migration Stage 2 — swing_trader ERROR sites |
| 5 | `bc06e0b` | feat(metals): log migration Stage 3-A — emergency + stops ERROR |
| 6 | `becb342` | feat(metals): log migration Stage 3-B — 14 catch-all ERROR |
| 7 | `dcb04bd` | fix(metals): capsys-safe lazy stdout handler + update shim test |
| 8 | `fa782c3` | fix(metals): codex v1 — scoped loggers, no root mutation |
| 9 | `5c3df15` | fix(metals): codex v2 — library-discipline logging setup |
| 10 | `02e9c7a` | fix(metals): codex v3 — metals_swing_trader as child logger |
| 11 | `f2be45b` | fix(metals): codex v4 — log volume + library fallback + unicode |
| 12 | `a5a5df7` | fix(metals): codex v5 — hasHandlers gate + [SWING] provenance |
| 13 | `2aff9e1` | fix(metals): codex v6 — walk ancestors + propagate=False |
| 14 | `4699448` | fix(metals): codex v7 — duplicate _has_ancestor_emitter locally |

Rebased onto the BERT in-process + fingpt parser work from parallel
sessions (43d2417). Final merged commit on main: `9317849`.

### Logging architecture after Stage 1

**Library discipline** — modules are handler-free at import time:
- `metals_loop.py`: `logger = getLogger("metals_loop")`, no handler setup
  at import. `_install_stage1_logging()` runs only under `if __name__
  == "__main__":`.
- `metals_swing_trader.py`: `logger = getLogger("metals_loop.swing_trader")`
  — a CHILD of metals_loop in the dotted hierarchy. Inherits level AND
  propagates records to the parent's Stage 1 handler.

**`_install_stage1_logging()` does** (only under `__main__`):
- `sys.stdout/stderr.reconfigure(utf-8, replace)` (Unicode safety)
- Attaches a `_LazyStdoutHandler` to the `metals_loop` logger only
- Sets `logger.setLevel(INFO)` and `propagate=False` (single output owner,
  no double-emission with embedding processes that have root handlers)
- Idempotent via `_metals_loop_stage1` marker attribute

**`_LazyStdoutHandler`** — StreamHandler subclass with:
- Re-resolves `sys.stdout` on every emit (pytest capsys compat)
- Catches `UnicodeEncodeError` and retries with ASCII-sanitized message
  (old `_safe_print` safety net integrated into logging path)

**`log()` and `_log()` shims** — delegate to `logger.info()` when a
non-NullHandler emitter exists in the ancestor chain (via
`_has_ancestor_emitter()` walk that skips NullHandlers + checks level).
Otherwise fall back to direct stdout print via `_safe_print` (metals_loop)
or plain `print()` (metals_swing_trader).

Truth table:
| Caller context | Branch taken |
|---|---|
| Production `__main__` (Stage 1 installed) | logger path (handler fires) |
| pytest `caplog.at_level(logger='metals_loop')` | logger path (caplog handler) |
| Library import, no setup | fallback → stdout |
| External NullHandler on root | fallback → stdout |
| External root ERROR-level handler | fallback → stdout |

### Codex adversarial review findings addressed (16 total)

All HIGH/MEDIUM findings from 7 rounds of `/codex:adversarial-review`:

1. [HIGH v1] Root logger handlers clobbered at import → scoped setup
2. [MED v1] metals_swing_trader silent standalone → child logger
3. [HIGH v2] `propagate=False` blocks parent telemetry → walk-ancestors
4. [HIGH v3] Sibling logger doesn't inherit level → dotted child name
5. [HIGH v4] Hot-loop tracebacks could evict `[LLM]` heartbeat lines →
   5 sites demoted to single-line warnings (no exc_info)
6. [MED v4] `log()` drops records when imported without handler →
   library fallback to `_safe_print`
7. [MED v4] `_LazyStdoutHandler` loses Unicode fallback on `reconfigure`
   failure → integrated UnicodeEncodeError retry in emit()
8. [HIGH v5] `isEnabledFor(INFO)` insufficient gate → `hasHandlers() && isEnabledFor()`
9. [MED v5] Migrated swing_trader errors lost `[SWING]` tag → 10 prefixes added
10. [MED v6] `hasHandlers()` passes for NullHandler / ERROR-level handlers →
    `_has_ancestor_emitter()` walk with level + NullHandler awareness
11. [MED v6] Stage 1 install left `propagate=True` → duplicate records
    with embedding root handler → `propagate=False` after install
12. [HIGH v7] `_log()` lazy-imported metals_loop → Playwright + cwd mutation
    as side effect → duplicate `_has_ancestor_emitter` locally

Plus 4 internal fixes found during inline test iterations (capsys-stale
references, direct stream write for UnicodeEncodeError catch, etc.).

### Live state at handoff

- Main HEAD: `9317849` (FF-merge from rebased overnight branch)
- PF-MetalsLoop: restarted, healthy — `[23:13:32] [INFO] [SWING] Cash
  synced: 4834 SEK` + `SwingTrader init: cash=4834 SEK, cash_sync_ok=True,
  catalog=115 warrants, DRY_RUN=False` in new format
- PF-DataLoop: restarted, cycling normally (signals + GPU gate + forecast)
- Test suite: 6376/6379 passing (3 pre-existing flakes: test_meta_learner,
  test_signal_improvements, test_metals_llm_orphan — all state-isolation
  known failures per TESTING.md, unrelated to this branch)
- Worktree: `/mnt/q/finance-analyzer-overnight` cleaned up
- Branch: `fgl/overnight-log-migration-20260409` deleted (local)

### Deferred follow-ups (stages 4-6 of Agent D's migration plan)

Out of scope for this session but ready to pick up:
- **Stage 4**: DEBUG sites (~20) → `logger.debug()` with env-gated level
- **Stage 5**: WARNING sites (~50) → `logger.warning(...)`
- **Stage 6**: Retire `log()`/`_log()` shims via bulk replace after
  stages 1-5 prove stable

Also deferred:
- `metals_llm.py` `[LLM]` print migration (requires `scripts/health_check.py`
  substring-match update in the same session)
- ARCH-18 `metals_loop.py` decomposition (multi-session)
- Batch 5 remaining ~35 low-value catch-all sites (network transients, etc.)

### Lessons worth carrying forward

- **Codex adversarial review is aggressive**: 7 rounds surfaced 16
  findings on a single infrastructure change. Each round is worth
  running as long as it keeps finding real issues. Stopping rule: when
  findings become purely theoretical or scope-creep beyond the task.
- **Library discipline is non-negotiable**: modules that are imported
  into test harnesses and embedding processes should NEVER call
  `basicConfig`, modify root handlers, or disable propagation at
  import time. Handler configuration belongs in `__main__` only.
- **Dotted logger names form a real hierarchy**: using
  `metals_loop.swing_trader` instead of `metals_swing_trader` gives
  you inheritance for free. Much cleaner than manual handler setup on
  both modules.
- **`logger.hasHandlers()` is not the same as "a record will be emitted"**.
  Walk the ancestor chain AND check handler levels AND skip NullHandlers
  if you need that guarantee.

---

# Session Progress — Fingpt parser + BERT in-process migration 2026-04-09 late evening

## Status: SHIPPED (2 back-to-back migrations + 2 hotfixes)

### What shipped tonight
1. **fingpt parser fix** (`fde9cf8` + `28aa5d0`)
   - Root cause: Llama-3 chat template fed to `wiroai-finance-llama-8b` which
     is a completion model, not chat-tuned. Model was echoing the user
     question verbatim and `_parse_sentiment` was falling through to
     `neutral 0.7` for every headline in `sentiment_ab_log.jsonl`.
   - Fix: rewrote `PROMPT_TEMPLATES["finance-llama-8b"]` + `CUMULATIVE_PROMPT`
     in `/mnt/q/models/fingpt_infer.py` to few-shot plain-text (4/4 correct
     in probe, 8/8 + 3/3 in end-to-end smoke test). Updated stop tokens in
     `portfolio/llm_batch.py::_flush_fingpt_phase` to `["\n\n"]`.
   - Post-restart A/B log shows varied fingpt sentiments (positive/negative/
     neutral) with realistic confidences, not the constant 0.7 signature.

2. **BERT sentiment in-process cache** (`aa393b5` + `e5e226e` + `fb584fb` + `ddc68fd`)
   - Moved CryptoBERT / Trading-Hero-LLM / FinBERT from subprocess-per-call
     to an in-process lazy-loaded batched cache in `portfolio/bert_sentiment.py`.
   - Hotfix 1 (`fb584fb`): forced BERT to CPU by default after the GPU path
     caused VRAM contention with llama-server. 10 GB budget = BERT (1.5) +
     Chronos (3.5) + llama-server (5) + overhead (0.5) ≈ 10.5 GB, no margin.
     llama-server swaps were taking 200+ s waiting for VRAM. CPU is fine
     because subprocess removal is the main win; GPU inference was only
     ~2-3 s/cycle on top.
   - Hotfix 2 (`ddc68fd`): batched tokenize + forward pass. Initial version
     looped per-text which was actually slower than subprocess for larger
     batches. Legacy subprocess scripts always batched; my port regressed
     that by accident. Fixed in `_predict_batched` helper.

### Measured impact (per-ticker sig time)
| Cycle | Version | BTC sig | XAU sig | Total |
|---|---|---|---|---|
| 21:12 | pre-migration (subprocess) | 61.4 s | 70.7 s | **164.9 s** |
| 21:48 | GPU + VRAM contention (BAD) | 96.9 s | 102.7 s | 429.6 s |
| 21:54 | CPU per-text (not great) | 92.6 s | 93.9 s | 214.8 s |
| 22:05 | CPU batched (cold) | 33.4 s | 41.9 s | **149.9 s** |
| 22:16 | CPU batched (cached) | 34.4 s | 40.0 s | 42.0 s |

Per-ticker savings: ~29 s off the ticker phase per cycle. LLM batch phase
unchanged (~90-110 s). Net cycle savings: ~15 s (pre-fingpt baseline) to
much more when the old fingpt daemon was still in the mix earlier today.

### Known state at handoff
- PF-DataLoop running new code, measured healthy at 22:16 UTC (local 00:16)
- BERT on CPU by default. Set `BERT_SENTIMENT_USE_GPU=1` if VRAM pressure
  eases (e.g. if Chronos is retired) to opt back in to GPU.
- Fingpt shadow is producing varied sentiments in `data/sentiment_ab_log.jsonl`
- All 3 legacy subprocess scripts (`cryptobert_infer.py`,
  `trading_hero_infer.py`, `finbert_infer.py`) UNCHANGED — they remain as
  fallback + CLI tools. Can retire once in-process path proves stable.
- Tests: 15 in `tests/test_bert_sentiment.py`, all pass. 85 regression
  tests across `test_llm_batch` + `test_loop_contract` still pass.

### Follow-ups NOT shipped (deferred)
- Retire the 3 legacy subprocess scripts after a week of stability.
- GPU BERT path opt-in documentation (env var works but undocumented outside
  the code comment).
- If more GPU headroom opens up, consider cross-model batching (CryptoBERT +
  FinBERT in parallel).

---

# Session Progress — Metals Bug Fleet v2 2026-04-09 evening

## Status: COMPLETE (3 agent fixes shipped + log migration audit)

### Fleet v2 headline

Three parallel agents (A/B/C) each shipped one fix on branch
`fix/metals-fleet-v2-apr09`, merged to main as commits 6a20c7d, 6b42431,
6c72628. Agent D produced a read-only audit with no code changes. All
fixes live across PF-MetalsLoop + PF-DataLoop + PF-GoldDigger (all three
restarted after merge).

**Commits**:
1. `6a20c7d` fix(avanza): port multi-shape fallback to get_buying_power (bug C7)
2. `6b42431` fix(golddigger): drop stale cache + log-once for yfinance proxy failures
3. `6c72628` refactor(metals): consolidate duplicate get_cet_time (dedup + DST fix)

### Fleet v2 details

**Agent A — `portfolio/avanza_session.py:get_buying_power`** (+305/-26):
Ports the 3-path + multi-field-ID pattern from `metals_avanza_helpers.py`
(shipped 873b2df this afternoon). Handles legacy categorizedAccounts, new
flat `data.accounts`, new `data.categories`. Tries four ID fields
(accountId/id/accountNumber/number) and balance-field alternates. Added
10-test `TestGetBuyingPower` suite (53/53 tests pass). **Contract change**:
return type is now `dict | None`. Two scripts (`fish_straddle.py:174`,
`fish_monitor_live.py:431`) need None-guards as a follow-up — they'll
crash loudly instead of sizing trades off 0 SEK (safer failure mode).

**Agent B — `portfolio/golddigger/data_provider.py:_fetch_yfinance_proxy`**
(+47/-7): Returns None on stale bars instead of masking outage with cached
stale value. This unblocks the existing `fetch_us10y_context →
fetch_us10y(DGS10)` FRED fallback path, which was previously unreachable.
DXY gets same benefit for free (falls through to `read_macro_context()`).
Log-once per cache_key via `_PROXY_STALE_WARNED` dict. Happy path (fresh
`^TNX` bar) byte-identical. All 107 golddigger tests pass. Affects
**both** PF-DataLoop (via `macro_context.py`) and PF-GoldDigger (direct
import) — both restarted.

**Agent C — `data/metals_loop.py` + `data/metals_shared.py`** (-60/+30 net):
Removes the duplicate `get_cet_time()` copy in metals_loop.py (along with
`_WARNED_TIMEAPI_METALS_LOOP` and the comment block from f557f9f),
imports canonical version from metals_shared instead. **Found real
behavior difference**: metals_loop.py had DST-aware tz labeling (CEST in
summer), metals_shared.py had hardcoded "CET". Ported DST logic into
metals_shared — canonical version is now strictly better. Telegram + log
output will correctly show "CEST" during DST months. Hoisted DATA_DIR
sys.path insert to top of metals_loop.py so the hard import works.
Smoke-tested: returns `(19.42, '19:25 CEST', 'zoneinfo')`.

**Agent D — Migration audit** (`docs/LOG_MIGRATION_AUDIT_20260409.md`):
Read-only audit of 290 `log()` sites in metals_loop.py and 48 `_log()`
sites in metals_swing_trader.py. Key finding: **no machine parser depends
on the custom format** (only `health_check.py` reads the file, and it
only substring-matches `[LLM] Chronos`/`[LLM] Ministral` emitted by
`metals_llm.py`, not in-scope files). Recommends 6-stage partial
migration (~6-8 hours). One real gotcha: `_safe_print` Windows Unicode
wrapper must be preserved. **Decision: DEFER** — out of scope for this
session. User can act on the audit report in a future dedicated session.

### Live state at handoff

- PF-MetalsLoop: running, `cash_sync_ok=True`, `buying_power=1515.77 SEK`,
  catalog=115 warrants, `DRY_RUN=False`. XAG + XAU BUY 100% consensus.
- PF-DataLoop: running, cycle ~57s, signals healthy, GPU gate cycling
  Chronos/Kronos cleanly.
- PF-GoldDigger: running in SIGNAL-ONLY mode, no open position, "Entry
  blocked: low volume" on each poll (legit market state). No TNX/proxy
  spam since restart — Agent B's log-once working.
- 3 new commits on main: 6a20c7d, 6b42431, 6c72628.

### Deferred follow-ups (from Fleet v2 work)

- **`fish_straddle.py:174` + `fish_monitor_live.py:431` None-guards** —
  these two scripts will now crash on broken `get_buying_power()` calls
  instead of silently trading at 0 cash. Safer, but should be fixed.
- **Log migration stage 1** (infrastructure + shim) per
  `docs/LOG_MIGRATION_AUDIT_20260409.md` — 1-hour zero-risk task, could
  be tackled next session.
- **XAU SHORT canary activation** — still user-gated, awaiting
  observation window.
- **ARCH-18 metals_loop.py decomposition** — multi-session effort.
- **Duplicate python process mystery** — two global Python PIDs keep
  spawning alongside venv ones for each loop. Killed manually each
  restart, source unknown. Not investigated.

---

# Session Progress — Metals Bug Fleet 2026-04-09 afternoon

## Status: COMPLETE (cash sync recovered, noise suppressed, logging audit)

### What we fixed
7-commit fleet targeting bugs uncovered during the afternoon log audit
of the metals loop (following the morning's reliability merge).
Branch `fix/metals-bug-fleet-apr09` merged to main.

**Commits** (in order):
1. `8b59acb` fix(crypto): guard Data slice + log-once news errors
2. `77cf5ac` chore(metals): log-once timeapi.io failures
3. `03cdfbb` feat(metals): instrument avanza helpers with diagnostic logging
4. `fec1dde` chore(metals): bare-except audit + log-once unknown ob_id
5. `65f0998` fix(metals): cash sync — handle new Avanza response shape (Fix 4b)
6. `873b2df` fix(metals): cash sync — multi-field ID + balance fallback (Fix 4c)
7. `f557f9f` chore(metals): log-once for timeapi.io in metals_loop.py (dup function)

### Headline result: **cash sync recovered**
Swing trader was dormant for 3h 40m due to `fetch_account_cash` returning
None. Root cause identified in ~1 diagnostic cycle after deploying Fix 4a
instrumentation: **Avanza changed the account API response shape** during
the day. Old shape was `data.categorizedAccounts[].accounts[].accountId`.
New shape is `data.categories[].accounts[].id` or `data.accounts[].id`
(both coexist). Fix 4b added top-level shape fallbacks; Fix 4c added
per-account ID + balance field fallbacks. Swing trader is now:
`cash_sync_ok=True`, `buying_power=1515.77 SEK`, catalog=115 warrants,
DRY_RUN=False — ready to trade on next qualifying signal.

### Noise reduction
- `timeapi.io failed`: 379 per 3.5h (~800/hour peak) → 1 per boot
- `unknown ob_id 257734/1042976` (NEE, VRT): every 60s → once per id per run
- `CryptoCompare news error: slice(None, 15, None)`: every cycle → caught
  at source (Fix 1 isinstance guard) + log-once on exception path

### Logging audit (Fix 5, Agent δ)
Added Python `logging` module alongside existing `log()`/`_log()` helpers
in metals_loop.py and metals_swing_trader.py. Converted 43 previously-
silent bare-except sites to `logger.warning` (SILENT-FAIL category with
`exc_info=True`) or `logger.debug` (EXPECTED-FALLBACK). Other files in
scope (`metals_warrant_refresh.py`, `portfolio/avanza_control.py`) were
already fully instrumented and needed no changes. Zero behavior changes,
pure observability additions.

### Diagnostic instrumentation pattern that unlocked everything
The cash-sync bug was invisible for 3.5h because
`data/metals_avanza_helpers.py` had ZERO logging and bare
`except Exception: return None` at every site. Fix 4a added a
module-level logger + diagnostic JS that returns `{_error: ...}`
objects instead of bare nulls. The very first post-deploy sync cycle
wrote this one log line:
```
fetch_account_cash: diagnostic failure account_id=1625505
result={'_error': 'no_account_match', ...,
        'top_level_keys': ['categories', 'accounts', 'loans']}
```
That single line pinpointed the exact bug with zero guessing. Fix 4b
and 4c followed from evidence, not hypothesis.

### Followups (deferred)
See `memory/project_metals_deferred_20260409.md`:
- Golddigger `^TNX` intraday proxy staleness
- `portfolio/avanza_session.py:get_buying_power()` C7 bug (same endpoint,
  needs the same multi-shape fix)
- XAU SHORT canary activation (awaiting observation window)
- `metals_loop.py` decomposition (ARCH-18)
- Consolidate duplicate `get_cet_time()` (metals_shared.py + metals_loop.py)

### What's next — Bug Fleet v2 (4-agent parallel dispatch)

User decision: tackle all deferred items in parallel with multiple
subagents. This section is the handoff — after a `/compact`, the next
session should read this and dispatch 4 agents exactly as described.

**Branch name**: `fix/metals-fleet-v2-apr09` (create fresh worktree at
`/mnt/q/finance-analyzer-fleet-v2`).

**Disjoint-file discipline** — each agent owns exclusive write access
to the files listed; no overlap allowed.

| Agent | Task | Files | Risk |
|---|---|---|---|
| **A** | Port Fix 4c multi-shape pattern to `portfolio/avanza_session.py:get_buying_power()` (lines ~296-326). Same Avanza response shape change, same endpoint, same solution — copy the JS + Python logic from `data/metals_avanza_helpers.py:fetch_account_cash`. Add a test to `tests/test_avanza_session.py` if it exists, otherwise note test-coverage gap. | `portfolio/avanza_session.py`, `tests/test_avanza_session.py` (if exists) | LOW |
| **B** | Fix golddigger intraday `^TNX` proxy staleness. At `portfolio/golddigger/data_provider.py:_fetch_intraday_proxy` (around line 134), swap the stale yfinance `^TNX` path to use the same FRED DGS10 fallback we shipped this morning in `macro_context.py:_fred_10y_fallback`. Reuse the existing `fetch_us10y(fred_key, series_id="DGS10")` helper from `portfolio/golddigger/data_provider` (the function is already defined in the same file — it's just not used by the intraday proxy path). Log the fallback transition once per recovery. | `portfolio/golddigger/data_provider.py` | LOW-MED (touches golddigger which IS actively trading — preserve success path) |
| **C** | Consolidate duplicate `get_cet_time()`. `data/metals_shared.py:62` and `data/metals_loop.py:940` are near-identical copies. Keep the `metals_shared.py` version as canonical; replace the `metals_loop.py` copy with `from data.metals_shared import get_cet_time` (or just `get_cet_time` if already imported). Delete the metals_loop.py copy, its `_WARNED_TIMEAPI_METALS_LOOP` global, and the module-level comment block added in `f557f9f`. Verify the `tz_label = datetime.datetime.now(_STOCKHOLM_TZ).tzname() or "CET"` logic is still applied — either port it into `metals_shared.get_cet_time` or wrap the call. Test via `python -c "from data.metals_loop import get_cet_time; print(get_cet_time())"`. | `data/metals_loop.py`, `data/metals_shared.py` | MEDIUM (cross-module change, verify import order) |
| **D** | Audit `log()` / `print()` → `logging` migration candidates. This is the BIGGER refactor — DO NOT apply it in one pass. Instead, produce a **report only** of every `log()` call site in `data/metals_loop.py` and `_log()` in `data/metals_swing_trader.py`, grouped by log level (info/warning/error), with estimated line counts and per-section migration risk. NO CODE CHANGES from this agent — the report is the deliverable. Main session decides whether to apply, split, or defer based on the report. | READ-ONLY — no writes | N/A (audit only) |

**NOT in the fleet**:
- **XAU SHORT canary activation**: requires user go-ahead + observation
  window + catalog query. User-gated, do NOT delegate to an agent.
- **ARCH-18 metals_loop.py decomposition**: multi-session effort, don't
  touch in this fleet.

**Execution order (after /compact)**:
1. Create worktree: `git worktree add /mnt/q/finance-analyzer-fleet-v2 -b fix/metals-fleet-v2-apr09 main`
2. Fire Agents A, B, C, D in parallel (single message, 4 tool calls)
3. Wait for all 4 returns
4. Review diffs (expect ~200 lines total across A+B+C)
5. Run ruff + targeted tests
6. Commit in 3 logical commits (A, B, C; D is a report so no commit)
7. Push + merge + restart loops (metals + main — Fix B touches data_provider
   which is shared, so PF-DataLoop needs a restart too)
8. Observe: cash sync via avanza_session path (if any caller uses it),
   TNX staleness resolved in metals_loop logs, no new noise from the
   consolidated get_cet_time
9. Based on Agent D's report, decide whether to tackle the log()
   migration in this session or defer

**Live loop state at handoff**:
- Metals loop: running, healthy, cash_sync_ok=True, buying_power=1515.77 SEK
- XAG + XAU BUY 100% consensus all afternoon — swing trader now able to
  act on it; watch for first entry attempt in the 1h post-restart window
- Noise reduction: ~99.6% (from ~800 warnings/hour to 3 per boot)
- All 7 commits from this session merged to main

**Lessons worth carrying forward**:
- The 3-layer silent-failure pattern (logger + bare-except conversion +
  targeted JS/function diagnostic) from Agent γ's Fix 4a is a reusable
  template whenever you encounter a dark subsystem. Next use case: the
  same pattern applied to `portfolio/avanza_session.py` (Fleet v2 Agent A)
- Avanza API response shapes drift unpredictably. Always use multi-field
  fallback (`acc.accountId || acc.id || ...`) rather than a single exact
  match. Log `sample_account_keys` in diagnostic responses so the next
  drift is trivially diagnosed.
- When the same helper exists in two modules (e.g. `get_cet_time`), patches
  must touch both. Consolidation is cheaper than remembering.

---

# Session Progress — Round 4 Adversarial Review 2026-04-09

## Status: COMPLETE (merged + pushed)

### What we did
Full dual adversarial review of the finance-analyzer codebase (Round 4):
- Partitioned into 8 subsystems (~55,774 lines)
- Launched 8 parallel code-reviewer agents + 1 independent manual review
- Cross-critique between independent review and orchestration agent
- Synthesis document with action plan

### Key Results

**Round 3 Fix Rate: 70%+ of CRITICAL findings addressed** (19 of 67 confirmed fixed)

**3 new CRITICAL findings:**
1. OR-R4-1: `loop_contract.py` MAX_CYCLE_DURATION_S=180 not updated for 600s cadence
   → self-heal sessions burning Claude budget every 30 min
2. IC-R4-1: `metals_execution_engine.py` MIN_TRADE_SEK=500 fallback bypasses the fix
3. IC-R4-2: `trigger.py` SUSTAINED_DURATION_S=120 negates sustained checks at 600s cadence

**Theme: cadence change ripple effects** — 60s→600s change has 5+ cascading effects
on hardcoded thresholds (loop contract, trigger duration, safeguard checks, etc.)

### Deliverables
- `docs/ADVERSARIAL_REVIEW_4_SYNTHESIS.md` — Full synthesis with 29 active findings
- `docs/INDEPENDENT_ADVERSARIAL_REVIEW_4.md` — Independent review (11 new findings)
- `docs/PLAN.md` — Review plan

### What's Next
1. **IMMEDIATE**: Fix OR-R4-1 (MAX_CYCLE_DURATION_S → 650) to stop burning Claude budget
2. Fix IC-R4-2 (SUSTAINED_DURATION_S → 700) to stop noise triggers
3. Fix IC-R4-1 (metals_execution_engine MIN_TRADE_SEK → 1000)
4. Integrate remaining 7 agent results when they complete
5. Wire C6 (check_drawdown) into the live trading path

### 2026-04-09 16:37 UTC | fix/metals-bug-fleet-apr09
8b59acb fix(crypto): guard Data slice + log-once news errors
data/crypto_data.py

### 2026-04-09 16:37 UTC | fix/metals-bug-fleet-apr09
77cf5ac chore(metals): log-once timeapi.io failures
data/metals_shared.py

### 2026-04-09 16:37 UTC | fix/metals-bug-fleet-apr09
03cdfbb feat(metals): instrument avanza helpers with diagnostic logging
data/metals_avanza_helpers.py

### 2026-04-09 16:38 UTC | fix/metals-bug-fleet-apr09
fec1dde chore(metals): bare-except audit + log-once unknown ob_id
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 16:44 UTC | fix/metals-bug-fleet-apr09
65f0998 fix(metals): cash sync — handle new Avanza response shape
data/metals_avanza_helpers.py

### 2026-04-09 16:46 UTC | feat/fingpt-in-llmbatch
51f47ca feat(sentiment): move fingpt into llm_batch rotation, retire bespoke daemon
portfolio/llama_server.py
portfolio/llm_batch.py
portfolio/main.py
portfolio/sentiment.py
scripts/fingpt_daemon.py
tests/test_fingpt_daemon.py
tests/test_llm_batch.py

### 2026-04-09 16:47 UTC | 
3736673 feat(sentiment): move fingpt into llm_batch rotation, retire bespoke daemon
portfolio/llama_server.py
portfolio/llm_batch.py
portfolio/main.py
portfolio/sentiment.py
scripts/fingpt_daemon.py
tests/test_fingpt_daemon.py
tests/test_llm_batch.py

### 2026-04-09 16:48 UTC | fix/metals-bug-fleet-apr09
c2d2fa9 fix(metals): cash sync — multi-field ID + balance fallback
data/metals_avanza_helpers.py

### 2026-04-09 16:48 UTC | 
873b2df fix(metals): cash sync — multi-field ID + balance fallback
data/metals_avanza_helpers.py

### 2026-04-09 16:54 UTC | fix/metals-bug-fleet-apr09
dd3d5bd chore(metals): log-once for timeapi.io in metals_loop.py
data/metals_loop.py

### 2026-04-09 16:54 UTC | 
f557f9f chore(metals): log-once for timeapi.io in metals_loop.py
data/metals_loop.py

### 2026-04-09 16:55 UTC | hotfix/post-llmbatch-timeouts
3ac7b81 hotfix(main): _TICKER_POOL_TIMEOUT 500 → 180 after fingpt daemon retirement
portfolio/main.py

### 2026-04-09 17:10 UTC | main
5de675c docs(session): metals bug fleet 2026-04-09 afternoon + v2 handoff
docs/SESSION_PROGRESS.md

### 2026-04-09 17:26 UTC | fix/metals-fleet-v2-apr09
6a20c7d fix(avanza): port multi-shape fallback to get_buying_power (bug C7)
portfolio/avanza_session.py
tests/test_avanza_session.py

### 2026-04-09 17:26 UTC | fix/metals-fleet-v2-apr09
6b42431 fix(golddigger): drop stale cache + log-once for yfinance proxy failures
portfolio/golddigger/data_provider.py

### 2026-04-09 17:27 UTC | fix/metals-fleet-v2-apr09
6c72628 refactor(metals): consolidate duplicate get_cet_time (dedup + DST fix)
data/metals_loop.py
data/metals_shared.py

### 2026-04-09 17:30 UTC | fix/fingpt-parser-prompt
b680c7b fix(fingpt): rewrite prompts for wiroai-finance-llama-8b base model
docs/PLAN.md
portfolio/llm_batch.py

### 2026-04-09 17:31 UTC | 
fde9cf8 fix(fingpt): rewrite prompts for wiroai-finance-llama-8b base model
docs/PLAN.md
portfolio/llm_batch.py

### 2026-04-09 17:34 UTC | main
d9d7e72 docs(session): Fleet v2 results + log migration audit
docs/LOG_MIGRATION_AUDIT_20260409.md
docs/SESSION_PROGRESS.md

### 2026-04-09 19:24 UTC | fgl/overnight-log-migration-20260409
4894a59 plan(overnight): Fleet v2 follow-ups + log migration stages 1-3
docs/PLAN-OVERNIGHT-20260409.md

### 2026-04-09 19:25 UTC | fix/bert-inproc-gpu
aa393b5 feat(sentiment): move BERT models from subprocess to in-process GPU cache
portfolio/bert_sentiment.py
portfolio/sentiment.py
tests/test_bert_sentiment.py

### 2026-04-09 19:25 UTC | fgl/overnight-log-migration-20260409
7dc6cf9 fix(fish): None-guard get_buying_power() call sites (Fleet v2 followup)
scripts/fish_monitor_live.py
scripts/fish_straddle.py

### 2026-04-09 19:26 UTC | fgl/overnight-log-migration-20260409
8245bb5 feat(metals): log migration Stage 1 — shim log()/_log() to logger.info
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 19:29 UTC | fgl/overnight-log-migration-20260409
0572497 feat(metals): log migration Stage 2 — swing_trader ERROR sites → logger.exception
data/metals_swing_trader.py

### 2026-04-09 19:32 UTC | fgl/overnight-log-migration-20260409
bc06e0b feat(metals): log migration Stage 3-A — emergency + stops ERROR sites
data/metals_loop.py

### 2026-04-09 19:36 UTC | fgl/overnight-log-migration-20260409
becb342 feat(metals): log migration Stage 3-B — 14 catch-all ERROR sites
data/metals_loop.py

### 2026-04-09 19:49 UTC | fgl/overnight-log-migration-20260409
dcb04bd fix(metals): capsys-safe lazy stdout handler + update shim test
data/metals_loop.py
tests/test_unified_loop.py

### 2026-04-09 19:50 UTC | main
fb584fb hotfix(bert): default BERT to CPU, avoid VRAM contention with llama-server
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

### 2026-04-09 19:57 UTC | fgl/overnight-log-migration-20260409
fa782c3 fix(metals): codex review findings — scoped loggers, no root mutation
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 20:02 UTC | main
ddc68fd hotfix(bert): batch tokenize+forward pass (was N sequential calls)
portfolio/bert_sentiment.py
tests/test_bert_sentiment.py

### 2026-04-09 20:08 UTC | fgl/overnight-log-migration-20260409
5c3df15 fix(metals): codex review v2 — library-discipline logging setup
data/metals_loop.py
data/metals_swing_trader.py
tests/test_metals_loop_functions.py
tests/test_unified_loop.py

### 2026-04-09 20:16 UTC | fgl/overnight-log-migration-20260409
02e9c7a fix(metals): codex review v3 — metals_swing_trader as child logger
data/metals_swing_trader.py

### 2026-04-09 20:28 UTC | main
43d2417 docs(session): record fingpt parser + BERT in-process migration 2026-04-09 late evening
docs/SESSION_PROGRESS.md

### 2026-04-09 20:29 UTC | fgl/overnight-log-migration-20260409
f2be45b fix(metals): codex review v4 — log volume + library fallback + unicode
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 20:43 UTC | fgl/overnight-log-migration-20260409
a5a5df7 fix(metals): codex review v5 — hasHandlers gate + [SWING] provenance
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 20:55 UTC | fgl/overnight-log-migration-20260409
2aff9e1 fix(metals): codex review v6 — walk ancestors + propagate=False
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 21:04 UTC | fgl/overnight-log-migration-20260409
4699448 fix(metals): codex review v7 — duplicate _has_ancestor_emitter locally
data/metals_swing_trader.py

### 2026-04-09 21:11 UTC | 
070fdb1 plan(overnight): Fleet v2 follow-ups + log migration stages 1-3
docs/PLAN-OVERNIGHT-20260409.md

### 2026-04-09 21:11 UTC | 
6bc66ad fix(fish): None-guard get_buying_power() call sites (Fleet v2 followup)
scripts/fish_monitor_live.py
scripts/fish_straddle.py

### 2026-04-09 21:11 UTC | 
b222384 feat(metals): log migration Stage 1 — shim log()/_log() to logger.info
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 21:11 UTC | 
f4ced52 feat(metals): log migration Stage 2 — swing_trader ERROR sites → logger.exception
data/metals_swing_trader.py

### 2026-04-09 21:11 UTC | 
f7c9741 feat(metals): log migration Stage 3-A — emergency + stops ERROR sites
data/metals_loop.py

### 2026-04-09 21:11 UTC | 
903b05c feat(metals): log migration Stage 3-B — 14 catch-all ERROR sites
data/metals_loop.py

### 2026-04-09 21:11 UTC | 
de82b30 fix(metals): capsys-safe lazy stdout handler + update shim test
data/metals_loop.py
tests/test_unified_loop.py

### 2026-04-09 21:11 UTC | 
91de651 fix(metals): codex review findings — scoped loggers, no root mutation
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 21:11 UTC | 
9eb0d36 fix(metals): codex review v2 — library-discipline logging setup
data/metals_loop.py
data/metals_swing_trader.py
tests/test_metals_loop_functions.py
tests/test_unified_loop.py

### 2026-04-09 21:12 UTC | 
eb1d15e fix(metals): codex review v3 — metals_swing_trader as child logger
data/metals_swing_trader.py

### 2026-04-09 21:12 UTC | 
ccbc270 fix(metals): codex review v4 — log volume + library fallback + unicode
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 21:12 UTC | 
3496447 fix(metals): codex review v5 — hasHandlers gate + [SWING] provenance
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 21:12 UTC | 
2e5a6d7 fix(metals): codex review v6 — walk ancestors + propagate=False
data/metals_loop.py
data/metals_swing_trader.py

### 2026-04-09 21:12 UTC | 
9317849 fix(metals): codex review v7 — duplicate _has_ancestor_emitter locally
data/metals_swing_trader.py

### 2026-04-09 21:15 UTC | main
2866fe8 docs(session): overnight log migration + fish None-guards handoff
docs/SESSION_PROGRESS.md

### 2026-04-10 07:49 UTC | fix/metals-catalog-refresh-sync-playwright
fdd788e fix(metals): route warrant refresh through loop page to avoid sync_playwright conflict
data/metals_swing_trader.py
data/metals_warrant_refresh.py
tests/test_metals_warrant_refresh.py

### 2026-04-10 07:55 UTC | perf/llama-swap-reduction
7264889 perf(llama_server): active VRAM poll + KV cache reuse
portfolio/llama_server.py
tests/test_llama_server.py

### 2026-04-10 08:04 UTC | perf/llama-swap-reduction
3062c01 perf(llm_batch): rotation scheduling for llama-server LLMs
portfolio/llm_batch.py
portfolio/sentiment.py
portfolio/shared_state.py
portfolio/signal_engine.py
tests/test_llm_batch.py

### 2026-04-10 08:13 UTC | fix/metals-cleanup-apr10
add91cb fix(tests): add gs_ratio_velocity key to metals cross_asset mock
tests/test_new_signals_integration.py

### 2026-04-10 08:15 UTC | fix/metals-cleanup-apr10
d1fcb40 fix(metals): detect_holdings recognizes swing-managed warrants
data/metals_loop.py

### 2026-04-10 08:16 UTC | fix/metals-cleanup-apr10
1040d7d fix(metals): cycle status line shows swing trader positions
data/metals_loop.py

### 2026-04-10 08:30 UTC | perf/llama-swap-reduction
a168952 perf(llm_batch): address review findings N1+N2 on rotation gate
portfolio/shared_state.py
tests/test_llm_batch.py

### 2026-04-10 12:46 UTC | main
91b731d docs(session): perf/llama-swap-reduction shipped + measured (rotation working)
docs/SESSION_PROGRESS.md

### 2026-04-10 13:27 UTC | fix/metals-swing-sizing-and-time-limit
2a65d21 fix(metals): Kelly-based sizing + EOD-only exit for swing trader
data/metals_swing_config.py
data/metals_swing_trader.py
tests/test_metals_swing_sizing.py

### 2026-04-10 13:54 UTC | fix/metals-adversarial-review
3844ace fix(metals): adversarial review follow-ups — 3 bugs + 4 silent failures + logging
data/metals_swing_config.py
data/metals_swing_trader.py
data/test_metals_swing_trader.py
tests/test_metals_swing_sizing.py

### 2026-04-10 14:17 UTC | fix/bug178-disabled-signals
8d5b412 fix(signal_engine): respect DISABLED_SIGNALS in dispatch loop (BUG-178 root cause)
docs/PLAN.md
portfolio/main.py
portfolio/signal_engine.py
tests/test_signal_engine.py

### 2026-04-10 14:23 UTC | bug178 fix shipped + measured
- Merge `d3712f5` → main → push
- Pre-fix: 49 BUG-178 events since 2026-04-09 (45 yesterday, 4 today by 14:18 UTC)
- Root cause: signal_engine dispatch loop ignored DISABLED_SIGNALS, so
  crypto_macro / cot_positioning / credit_spread_risk were doing network
  I/O every cycle. CLAUDE.md said they were "force-HOLD pending validation"
  but the compute path ran them anyway.
- Fix: skip DISABLED_SIGNALS in dispatch loop (mirror skip_gpu pattern).
- Diagnostic: per-ticker last-signal tracker; BUG-178 handler now logs
  which signal each stuck ticker was running (so any future hang names
  the culprit instead of just listing the stuck tickers).
- Measurement: first post-restart cycle 16:23:34 UTC = 5 OK / 0 failed
  in 208 s (41.6 s/ticker avg, warmup phase). Per-ticker times:
  - MSTR: 41.3 s sig (was: 169 s)
  - XAU-USD: 57.7 s sig
  - XAG-USD: 69.5 s sig
  - BTC-USD: ~70 s sig
  - ETH-USD: 71.7 s sig
- Tests: 4 new in test_signal_engine.py + 321 adjacent tests pass.
- Investigation pivoted from Chronos VRAM contention (premise rejected,
  see project_chronos_vram_contention.md memory) to BUG-178 silent hangs
  after measuring Chronos GPU latency at ~50 ms p50. Closed task #16,
  opened + closed task #17.
- Schtasks restart caveat: `schtasks /run /tn PF-DataLoop` did NOT actually
  restart the loop because the singleton lock was held by old PID 28604
  from 10:34 AM start. Required manual `Stop-Process -Id 28604 -Force`
  before `schtasks /run` worked. Worth knowing for future restarts.

### 2026-04-10 14:25 UTC | main
dc7683d docs(session): bug178 fix shipped + measured (49 events / day → 0)
docs/SESSION_PROGRESS.md

### 2026-04-10 14:57 UTC | fix/metals-finish-review
9804a55 fix(metals): adversarial review round 2 — L3 + S3 + 4 test gaps
data/metals_swing_trader.py
data/test_metals_swing_trader.py
tests/test_metals_swing_sizing.py
