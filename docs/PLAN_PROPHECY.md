# PLAN — Prophecy daily price-prediction system

Spec: `docs/superpowers/specs/2026-06-06-prophecy-design.md`. Worktree
`Q:/finance-analyzer-prophecy`, branch `prophecy/build-2026-06-06`.
**Tests deferred. Merge/push/task-install gated on user sign-off.**

## Reuse (no reinvention)
- `portfolio.file_utils`: `atomic_write_json`, `load_json`, `atomic_append_jsonl`,
  `load_jsonl_tail`, `last_jsonl_entry`, `count_jsonl_lines`.
- `portfolio.outcome_tracker._fetch_current_price(ticker)` (binance fapi/spot + YF/alpaca),
  `_fetch_historical_price(ticker, target_ts)` for outcome backfill.
- `portfolio.prophecy.get_context_for_layer2()` — read-only macro-belief context.
- `data/agent_summary.json`, `accuracy_cache.json`, `morning_briefing.json`,
  `daily_research_*.json` — stored signals/equations/research.
- Dashboard: `DATA_DIR`, `_read_json`, `require_auth`, `send_from_directory("static", …)`.

## Files (batched)

### Batch 1 — core (no internal deps)
1. `prophecy/__init__.py` — package marker + version.
2. `prophecy/schema.py` — `HORIZONS` (10), record validators, `Coverage` block,
   `validate_record()` (repairs/quarantines bad horizons, never upgrades
   `needs_work` true→false), `grade_sufficiency(found, required)`.
3. `prophecy/strategies.py` — `PLAYBOOKS: dict[str, Playbook]`; per-instrument
   `signal_emphasis / equations / web_questions / forum_sources / special_factors /
   price_model`. 13 instruments. `playbook_for(instrument)`.
4. `prophecy/config.py` — load `data/prophecy/config.json` (default: all enabled,
   model=claude-opus-4-8, horizons, `budget_usd_soft_cap=null`); `enabled_instruments()`.
5. `data/prophecy/config.json` — default config (all 13 enabled).

### Batch 2 — pipeline (dep: batch 1)
6. `prophecy/prep.py` — zero-token. Per enabled instrument: live price (reuse
   `_fetch_current_price`; None→coverage downgrade), gather stored signals from
   agent_summary[ticker] + accuracy_cache + macro beliefs + recent research, attach
   playbook, seed deterministic `coverage`. Write `data/prophecy/context_<date>.json`.
7. `prophecy/publish.py` — zero-token. Read `raw_<date>.json`, `validate_record` each,
   stamp `spot_at_prediction` + `run_id`, atomic-append `prediction_journal.jsonl`,
   write `latest.json` (+ `coverage_summary`, `cost_summary`). Quarantine malformed.
8. `prophecy/cost.py` — zero-token. Parse `claude -p --output-format json` stdout
   (`total_cost_usd`, `usage`); fallback stream-json scan; append `cost_log.jsonl`,
   roll into `latest.json.cost_summary`.
9. `prophecy/outcomes.py` — zero-token. For matured horizons compute dir-hit, target
   MAE, Brier(prob_up); append `accuracy.jsonl`, roll `accuracy.json`.

### Batch 3 — orchestration + prompt
10. `docs/prophecy-prompt.md` — daily prompt: `ultracode`; per enabled instrument run
    `/deep-research` + forum sentiment; fuse stored signals/equations/research; emit
    STRICT JSON to `data/prophecy/raw_<date>.json` incl. `coverage`; self-flag needs_work.
11. `scripts/prophecy-daily.bat` — mirror `after-hours-research.bat`: detach env,
    prep → `claude -p --model claude-opus-4-8 --verbose --output-format json` → publish
    → outcomes → cost. Progress JSON + JSONL log + exit handling.
12. `scripts/win/install-prophecy-task.ps1` — register `PF-Prophecy` daily 10:00 CET.

### Batch 4 — dashboard (additive only)
13. `dashboard/app.py` — add `@app.route("/api/prophecy")` + `@app.route("/prophecy")`
    → `static/prophecy.html`. No edits to existing routes.
14. `dashboard/static/prophecy.html` — standalone page: per-instrument horizon grid,
    **⚠ needs-work column** (tooltip=missing_inputs), running cost, accuracy.

## Non-goals (deferred)
- Tests; Phase-2 loop integration; token budget enforcement.

## Verify (no pytest; static only until greenlit)
- imports clean; `python -m prophecy.prep` hits live prices; publish on fixture →
  valid latest.json; dashboard route 200 + shape.

## Premortem (fresh agent a404df49 + my decisions)

**DESIGN CHANGE — adopted:** physical data dir renamed `data/prophecy/` →
**`data/prophecy_runs/`** to eliminate any collision class with the existing
`data/prophecy.json` macro-beliefs FILE. User-facing name + `/api/prophecy` route
unchanged.

1. **[a] prophecy.json clobber (HIGH×HIGH).** New dir at same stem as load-bearing
   `data/prophecy.json` (read by main/reporting/news_event/crypto_scheduler/...). A glob
   cleanup or path typo could nuke it → silent macro-context loss everywhere.
   → **FIX:** dir renamed to `data/prophecy_runs/` (collision impossible). prep.py also
   `assert os.path.isfile("data/prophecy.json")` after its own dir create (tripwire).

2. **[f] COST 10× during freeze (CRITICAL×CRITICAL).** One `claude -p` fans /deep-research
   ×13 + `ultracode` multi-agent → token multiplication; cost.py is post-hoc (too late).
   → **FIX:** `--max-turns` on the claude call + `ExecutionTimeLimit` on PF-Prophecy +
   pre-run warn (read yesterday's cost_log; if > soft-cap, log critical + still run since
   "unhinged" was authorized). cost.py writes a critical_errors alert when a run exceeds
   `budget_usd_soft_cap`. Real-time mid-run cap impossible via `claude -p`; turns+wallclock
   are the bounds.

3. **[c] freeze bypass (HIGH×CRITICAL).** `.bat` bypasses claude_gate (like the other
   research .bat); only guard is task-disabled state. New task ships ENABLED would spend
   despite the freeze.
   → **FIX:** `.bat` checks `data/prophecy_runs/SYSTEM_DISABLED` sentinel → exit 0 before
   any claude call. install-prophecy-task.ps1 registers PF-Prophecy in **/DISABLE** and
   creates the sentinel. Going live = explicit user action (remove sentinel + enable task).
   Documented in the .bat header.

4. **[d] silent empty/stale (HIGH×MED).** Agent crashes mid-fan-out / hits turn limit /
   prints prose → raw_<date>.json missing/torn → publish quarantines all → latest.json
   keeps yesterday's calls dated today → outcomes poisons accuracy. Exit 0 throughout.
   → **FIX:** publish.py asserts raw mtime ≥ run start AND record_count > 0; on failure
   writes `data/critical_errors.jsonl` + sets `latest.json.stale=true` (never silently
   overwrites with stale, never appends phantom journal rows). Also parses
   `run_<date>.json` for `is_error`/`subtype`.

5. **[b] races (MED×MED).** (i) publish reading raw mid-write → strictly sequential .bat
   (publish only after claude exits) + defensive load_json+size check. (ii) prep's 13×
   `_fetch_current_price` FAPI burst collides with 60s main-loop FAPI → 429.
   → **FIX:** prep throttles ~250ms/instrument; logs source; tolerates None.

6. **[e] worktree/prod drift (MED×MED).** Worktree lacks config.json symlink → stock
   prices (MSTR/SAAB-B/SEB-C/INVE-B via Alpaca) silently yfinance-fallback or None; relative
   paths assume cwd=repo root but .bat could launch elsewhere.
   → **FIX:** prep logs price source per instrument + flags None→coverage needs_work (no
   crash); .bat `cd /d Q:\finance-analyzer` + asserts config.json exists before claude;
   verify step `python -m prophecy.prep --dry-run` (no pytest).

ACCEPT: no real-time token kill mid-`claude -p` (not exposed); bounded by turns+wallclock
+ ship-disabled. Acceptable given explicit "unhinged to begin with" + merge gate.
