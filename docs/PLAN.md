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
- [ ] **A-AV-2** Add hardcoded account whitelist in `get_account_id()`
      (`portfolio/avanza_client.py:223-244`). Reject anything != "1625505".
      Mirror the `ALLOWED_ACCOUNT_IDS` pattern from `avanza_session.py`.
- [ ] **A-PR-3** Replace raw `json.load()` with `file_utils.load_json()` in
      `portfolio/portfolio_validator.py`. TOCTOU race with concurrent saves.
- [ ] **A-IN-2** Kill subprocess on `TimeoutExpired` in `portfolio/claude_gate.py`
      (zombie cleanup). Add `proc.kill(); proc.wait()` in the timeout handler.

### Batch 2 — Data correctness (low risk, P0)

- [ ] **A-MC-2** Replace hardcoded `usdsek=1.0` with `fx_rates.usd_to_sek()`
      in `data/metals_loop.py` (find via grep). All SEK P&L currently wrong by ~10x.
- [ ] **A-PR-2** Fix drawdown peak scan in `portfolio/risk_management.py` to
      walk full equity history, not just the last 2000 entries (~33h window).
- [ ] **A-DE-4** Add yfinance MultiIndex flatten in `portfolio/fear_greed.py`
      stock path. Stock F&G signal is silently dead.
- [ ] **A-DE-5** Fix `portfolio/onchain_data.py` cache crash on old-format
      ISO-string-as-epoch entries. Add format-detection fallback.

### Batch 3 — Logic bugs (medium risk, P0/P1, requires tests)

- [ ] **A-SM-1** Fix gap-fill BUY direction during continuing gap-down
      (find module via grep on "gap_fill"). Currently buys into crashes.
- [ ] **A-SM-2** Add GARCH key to volatility `_empty_result` schema in
      `portfolio/signals/volatility.py`. Inconsistent sub_signals.
- [ ] **A-MC-4** Use real `entry_ts` from holdings instead of `now()` in
      metals_loop HOLD_TIME_EXTENDED check.

### Batch 4 — Threading and concurrency (medium risk, P1)

- [ ] **BUG-184** Add `threading.Lock` to `portfolio/trade_guards.py` state
      read/write. Concurrent ThreadPoolExecutor threads bypass cooldowns.
- [ ] **BUG-183** Replace global throttle in `portfolio/autonomous.py` with
      per-ticker dict. Currently one HOLD message throttles ALL tickers 30min.
- [ ] **A-IN-3** Add concurrency lock to `portfolio/claude_gate.py` (file lock
      for cross-process, threading.Lock for in-process).

### Batch 5 — Signal-system tuning (medium risk, P0)

- [ ] **Verify fear_greed gating** — currently MISSING from `data/accuracy_cache.json`.
      Investigate why backfill skipped it. Force a refresh via `--accuracy --force`.
      Confirm the 45% gate fires (blended is 0.357).
- [ ] **Per-ticker signal blacklist** — add `_TICKER_SIGNAL_BLACKLIST` constant
      in `portfolio/signal_engine.py` for known-bad pairs (start with
      `("ministral", "XAG-USD")` at 20.4% accuracy).
- [ ] **Raise accuracy gate 45 → 47** — single constant in `signal_engine.py`,
      then re-run audit to count newly-gated signals.

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
