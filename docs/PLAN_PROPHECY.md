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

## Premortem
_(populated by fresh agent below)_
