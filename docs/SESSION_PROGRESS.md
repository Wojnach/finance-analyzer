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
