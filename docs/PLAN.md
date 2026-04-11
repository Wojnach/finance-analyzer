# PLAN — Outstanding Issue Fix Queue (2026-04-11)

## Context

Continuation of after-hours signal audit (2026-04-10) + adversarial review
(8 agents, 148 findings in `docs/adversarial-review/SYNTHESIS.md`).
Driven by ralph-loop autonomous iteration started 2026-04-11.

Branch: `fix/queue-2026-04-11`
Worktree: `/mnt/q/finance-analyzer-fixq`

## Already Verified Done (do not repeat)

- [x] Per-ticker directional accuracy override propagates BUY/SELL fields
      (`portfolio/signal_engine.py:1895-1898`, commit `6ec4be9`)
- [x] Directional gate raised 0.35 → 0.40 (commit `6ec4be9`)
- [x] BUG-182: directional accuracy as consensus weight (commit `11aaf27`)
- [x] BUG-178 slow-cycle hunt (commits `5a13ed6`, `5d661cf`, `68d22ba`, `8d5b412`)

## Open Tasks (ordered: safest+highest-impact first)

### Batch 1 — Defensive additive fixes (low risk, P0)

- [x] **A-AV-1** Wrap `api_get`/`api_post`/`api_delete` bodies in `_pw_lock`
      (`portfolio/avanza_session.py:184-291`). Prevents Playwright corruption
      when metals 10s fast-tick + main 8-worker pool race on the context.
      → commit `7ad33cf` (RLock + 4 thread-safety tests, 57/57 pass)
- [x] **A-AV-2** Add hardcoded account whitelist in `get_account_id()`
      (`portfolio/avanza_client.py:223-244`). Reject anything != "1625505".
      Mirror the `ALLOWED_ACCOUNT_IDS` pattern from `avanza_session.py`.
      → commit `0daa2ef` (whitelist + filters in get_positions/get_portfolio_value, 7 new tests, 231/231 pass)
- [x] **A-PR-3** Replace raw `json.load()` with `file_utils.load_json()` in
      `portfolio/portfolio_validator.py`. TOCTOU race with concurrent saves.
      → commit `f07469c` (new tests/test_portfolio_validator.py with AST
      regression guard, 7/7 pass)
- [x] **A-IN-2** Kill subprocess **tree** on `TimeoutExpired` in `portfolio/claude_gate.py`
      (zombie cleanup including grandchildren — Node helpers, MCP servers,
      local-LLM processes). Refactored to Popen + tree-kill helper.
      → commit `122658b` (8 tests, grandchild kill verified on Windows)

### Batch 2 — Data correctness (low risk, P0)

- [x] **A-MC-2** Replace hardcoded `usdsek=1.0` with `fetch_usd_sek()`
      in `portfolio/fin_snipe_manager.py:420` (was actually here, not metals_loop).
      All exit_optimizer SEK rewards previously wrong by ~10x.
      → commit `7597650` (live FX rate + AST regression guard, 48/48 pass)
- [x] **A-PR-2** Fix drawdown peak scan in `portfolio/risk_management.py` to
      walk full equity history, not just the last 2000 entries (~33h window).
      → commit `174eca0` (new `_streaming_max` helper, 47/47 pass)
- [x] **A-DE-4** Add yfinance MultiIndex flatten in `portfolio/fear_greed.py`
      stock path. Stock F&G signal is silently dead.
      → commit `1a8381f` (defensive flatten + 4 new tests)
- [x] **A-DE-5** Fix `portfolio/onchain_data.py` cache crash on old-format
      ISO-string-as-epoch entries. Add format-detection fallback.
      → commit `aecec90` (`_coerce_epoch` helper, 8 tests, end-to-end coverage)

### Batch 3 — Logic bugs (medium risk, P0/P1, requires tests)

- [x] **A-SM-1** Investigated: synthesis was a false positive (existing
      `fill_pct < 0.3` already handled the case, verified with concrete trace).
      Added explicit `if fill_pct < 0: HOLD` guard for clarity + 3 regression
      tests covering all 4 (gap_dir × day_dir) quadrants.
      → commit `c595ab0`
- [x] **A-SM-2** Add GARCH key to volatility `_empty_result` schema in
      `portfolio/signals/volatility.py`. Inconsistent sub_signals.
      → commit `a75e8f7` (also added garch_vol/realized_vol/garch_ratio
      to indicators + cross-path schema regression test)
- [x] **A-MC-4** Use real `entry_ts` from instrument_state instead of `now()`
      in fin_snipe_manager. Persists on first non-zero observation, clears
      on position close. HOLD_TIME_EXTENDED now actually fires.
      → commit `ceab91b`

### Batch 4 — Threading and concurrency (medium risk, P1)

- [~] **BUG-184** Trade guards lock — DROPPED. Per prior session
      `docs/SESSION_PROGRESS.md`: "Layer 2 runs as subprocess, not thread.
      threading.Lock wouldn't help; atomic_write_json is adequate."
- [~] **BUG-183** autonomous per-ticker throttle — DROPPED. Per prior session:
      "BUY/SELL signals always bypass the global throttle. The throttle only
      suppresses pure-noise HOLD messages."
- [x] **A-IN-3** Added in-process `_invoke_lock` (threading.Lock) wrapping
      both invoke_claude / invoke_claude_text. Max in-process concurrency
      now 1. Cross-process file lock deferred (separate concern).
      → commit `e0a4605` (5-thread serialization test passes)

### Batch 5 — Signal-system tuning (medium risk, P0)

- [~] **Verify fear_greed gating** — DROPPED. Prior session verified blended
      accuracy = 0.586 (above 0.45 gate). Cache absence is likely a refresh
      lag, not a gating bug.
- [~] **Per-ticker signal blacklist** — DROPPED. Per-ticker accuracy gate
      already catches `ministral × XAG-USD` (18.9% < 0.45). No additional
      blacklist needed.
- [x] **Raise accuracy gate 45 → 47** — single constant in `signal_engine.py`,
      then re-run audit to count newly-gated signals.
      → commit `6c7e289` (gates ~4 additional 45-47% signals; 37 tests pass)

## DEFERRED (too risky for autonomous overnight)

- A-MC-1 `HARD_STOP_CERT_PCT` change (live trading-execution behavior)
- A-MC-3 ORB window CET/CEST hardcoded (need DST table verification)
- A-PR-1 `record_trade()` hookup (need careful call-site analysis)
- HMM regime blending, IC weighting (Tier 2/3 from RESEARCH_PLAN)

## Iteration Protocol

For each ralph-loop iteration:

1. `cd /mnt/q/finance-analyzer-fixq`
2. Read `docs/PLAN.md`, find FIRST unchecked `[ ]` task in earliest open batch
3. Implement that ONE task only (5-10 files max)
4. Write or extend tests in `tests/` for the change
5. Run targeted tests:
   `.venv/Scripts/python.exe -m pytest tests/test_<area>.py -n auto --timeout=60`
6. Fix failures before committing
7. Commit with conventional message + bug ID (e.g. `fix(avanza): A-AV-1 ...`)
8. Update PLAN.md: change `[ ]` to `[x]` with the commit SHA appended
9. Commit the PLAN.md update separately

When ALL tasks `[x]` checked:

1. Run full test suite: `.venv/Scripts/python.exe -m pytest tests/ -n auto`
2. Fix or document failures per `docs/TESTING.md` known-failure list
3. Merge worktree to main:
   `cd /mnt/q/finance-analyzer && git merge fix/queue-2026-04-11 --no-ff`
4. Push: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
5. Cleanup:
   `git worktree remove /mnt/q/finance-analyzer-fixq && git branch -d fix/queue-2026-04-11`
6. Update `docs/SESSION_PROGRESS.md` with what shipped
7. Output exactly: `<promise>ALL_TASKS_COMPLETE</promise>`

## Constraints

- Worktrees only — never modify main directly
- Never modify `config.json` (external symlink with API keys)
- Never skip git hooks (--no-verify forbidden unless explicit user request)
- Use `file_utils.atomic_write_json` / `load_json` for all JSON I/O
- Per `.claude/rules/signals.md`: don't introduce ticker="" callers, don't invert sub-50% signals
- Per `.claude/rules/metals-avanza.md`: stop-loss API only, ≥1000 SEK orders
- Be decisive — no approval-asking
- Save SESSION_PROGRESS.md every 3 batches
