# FGL Review — portfolio-risk

Reviewed (full) in worktree `Q:/finance-analyzer/.worktrees/fgl/portfolio-risk`:
`portfolio_mgr.py`, `risk_management.py`, `trade_guards.py`, `equity_curve.py`,
`exit_optimizer.py`, `monte_carlo.py`, `monte_carlo_risk.py`. Cross-referenced
`portfolio/file_utils.py`, `portfolio/main.py`, `portfolio/agent_invocation.py`,
`dashboard/app.py`, `docs/TRADING_PLAYBOOK.md`, the live `data/critical_errors.jsonl`,
and the corruption tests in `tests/`.

---

## 0. Corruption root-cause analysis (the #1 target)

### TL;DR — the "recurring live corruption" is a TEST-ISOLATION LEAK, not a portfolio loss.

All six `portfolio_state_corrupt` entries logged "today 15:11–15:19" point at
pytest tmp dirs, not the live files. Evidence from `data/critical_errors.jsonl`:

```
context.path = C:\Users\Herc2\AppData\Local\Temp\pytest-of-herc2\pytest-2872\test_load_all_corrupt_returns_0\portfolio_state.json   (bytes: 7)
...\pytest-2873\popen-gw3\test_corrupt_json_returns_defa0\state.json   (bytes: 15)
...\popen-gw3\test_null_json_returns_default0\state.json   (bytes: 4)
...\popen-gw3\test_corrupt_json_returns_defa1\bold.json   (bytes: 12)
...\popen-gw3\test_returns_defaults_when_all0\portfolio_state.json   (bytes: 8)
```

`popen-gw3` = xdist worker; the byte counts (4/7/8/12/15) are the deliberately
corrupt fixtures (`null`, `{invalid json!!`, `[1,2,3]`, etc.). The LIVE
`data/portfolio_state.json` is 26 KB, mtime 16:29, intact; `bold.json` 11 KB,
untouched since May 19. **No real portfolio was wiped.** The startup
`check_critical_errors.py` + the `PF-FixAgentDispatcher` then read these
test-generated criticals out of the live journal and reported "recurring
corruption with no backup recovered" — but there was never anything to recover
because the corruption only ever existed inside `tmp_path`.

### Why the leak happens — causal chain (the actual defect introduced by this PR)

The merged resilience fix added a NEW side effect to the read path:
`_load_state_from` → corrupt branch → `_quarantine_corrupt_state`
(`portfolio_mgr.py:175`) → `atomic_append_jsonl(CRITICAL_ERRORS_LOG, …)`
(`portfolio_mgr.py:115`). `CRITICAL_ERRORS_LOG` is a module-level `Path`
constant (`portfolio_mgr.py:24`) pointing at the real `data/critical_errors.jsonl`.

The pre-existing corruption tests patch only `STATE_FILE` to `tmp_path`, NOT
`CRITICAL_ERRORS_LOG`:
- `tests/test_portfolio_mgr_core.py:279` `test_corrupt_json_returns_default`
- `tests/test_portfolio_mgr_core.py:295` `test_null_json_returns_default`
- `tests/test_portfolio_mgr_core.py` `TestLoadStateBackupRecovery::test_returns_defaults_when_all_corrupt` (~:483)
- the same-named tests in `tests/test_portfolio_mgr.py` and `tests/test_io_safety_sweep.py`

So when those tests write corrupt JSON and call `load_state()`, the new
quarantine fires and appends a real `critical` entry to the live journal. This
is a direct violation of the repo rule "Tests using module-level file paths must
patch to `tmp_path` for xdist safety" (`CLAUDE.md`, `.claude/rules/testing.md`).
The new dedicated test `tests/test_portfolio_mgr_corrupt_quarantine.py` DOES
patch `pm.CRITICAL_ERRORS_LOG` correctly (line 29) — the older tests were simply
never updated to anticipate the new write. See P0-1.

### Was the live portfolio ever corruptible? Where would real corruption come from?

Yes, the failure mode the PR is defending against is real — but the writer that
produces it is OUTSIDE this subsystem and is NOT protected by any of the atomic
machinery here:

- `docs/TRADING_PLAYBOOK.md:127`: *"Edit `data/portfolio_state.json` (patient) or
  `data/portfolio_state_bold.json` (bold)."*
- `portfolio/agent_invocation.py:1159`: Layer 2 Claude subprocess is launched with
  `--allowedTools "Edit,Read,Bash,Write"`.

The live mutation path for both portfolios is the Layer 2 LLM hand-editing the
JSON via the `Edit`/`Write` tool. A botched edit (dangling `]`, trailing comma)
produces exactly the shape the quarantine test models
(`tests/test_portfolio_mgr_corrupt_quarantine.py:22`:
`b'{"cash_sek": 467803.17,\n  ],\n  {"orphan": true}\n'`). None of
`portfolio_mgr.atomic_write_json` / backup rotation guards this path, because the
LLM writer never calls `save_state()`. This is the genuine corruption source
(P1-1). The dashboard is NOT a writer — `/api/validate-portfolio`
(`dashboard/app.py:1064`) is read-only validation and all portfolio access there
is `_read_json`.

### (b) Why "no backup recovered" can happen for real

Backup CREATION is `_rotate_backups` (`portfolio_mgr.py:50`), called
unconditionally inside `_save_state_to` (`:187`) and `update_state` (`:232`)
*before* the atomic write. It `shutil.copy2` the CURRENT on-disk primary into
`.bak`. So the `.bak` is only ever as good as the last file that was on disk at
save time. If the LLM corrupts the primary out-of-band (an Edit, not a
`save_state` call), and *then* something calls `save_state`/`update_state`, the
rotation copies the already-corrupt primary into `.bak` and `.bak` → `.bak2`,
laundering corruption through the entire ring. There is no
"backup-only-after-successful-load" gate — backups are taken at write time off
whatever bytes are present, never validated. That is exactly how a real incident
ends up with "corrupt primary AND corrupt/missing .bak." See P1-2.

### (c) Can a save still clobber the quarantined/corrupt file?

Partially mitigated for the CURRENT wiring; the door is not closed in code.

- The hot loop's only `portfolio_mgr` write is `portfolio/main.py:796`
  `save_state(state)`, guarded by `if not STATE_FILE.exists():`. A corrupt file
  still EXISTS, so the loop does NOT overwrite it with defaults. Good — credit
  where due: the loop will not silently re-baseline a corrupt-but-present file.
- BUT `save_state` / `save_bold_state` / `update_state` themselves have NO
  exists/quarantine guard (`portfolio_mgr.py:196-234`). Any caller that does
  read-modify-write after a corrupt load will (i) get fresh in-memory defaults
  from `_load_state_from`, (ii) `_rotate_backups` (corrupt primary → `.bak`),
  (iii) `_atomic_write_json` defaults over the primary — permanent loss of the
  real holdings, with only the `.corrupt-<sha>` quarantine copy surviving. The
  quarantine docstring itself (`:88-91`) warns of this `update_state` clobber,
  yet the write functions don't enforce it. Today no hot-loop caller does this,
  so it's latent (P2-1), but it means the PR's own stated guarantee ("the wipe is
  never silent / recoverable") rests on caller discipline, not on the module.

**Verdict on the merged fix:** the quarantine + idempotent journaling + fail-loud
read path is sound and correctly built (content-addressed once-only quarantine,
best-effort never-raises, corrupt bytes captured before the recovery loop). It
preserves evidence the old `logger.critical`-only path destroyed. It does NOT,
however, (1) stop the test leak that is currently generating the false
"corruption" alarms (P0-1), (2) protect the real write path the LLM uses (P1-1),
or (3) validate backups so a recoverable `.bak` actually exists (P1-2).

---

## P0 — silent state/portfolio loss, money loss, or loop crash

- **[P0-1] tests/test_portfolio_mgr_core.py:279,295,~483 (+ test_portfolio_mgr.py, test_io_safety_sweep.py) — corruption tests leak `critical` entries into the LIVE journal.**
  Chain: these tests patch only `STATE_FILE` to `tmp_path`, call `load_state()`
  on corrupt bytes → `_quarantine_corrupt_state` → `atomic_append_jsonl(CRITICAL_ERRORS_LOG, …)`
  where `portfolio_mgr.CRITICAL_ERRORS_LOG` (`portfolio_mgr.py:24`) is still the
  real `data/critical_errors.jsonl`. The six entries dated 2026-06-01 15:11–15:19
  in the live journal are these tests (paths under `pytest-of-herc2`,
  `popen-gw3`). Effect: the CLAUDE.md startup check and `PF-FixAgentDispatcher`
  treat them as unresolved live criticals, manufacturing the entire
  "recurring portfolio corruption" incident and burning fix-agent spawns. This
  is the bug behind the reported symptom and a direct violation of the
  "patch module-level paths to tmp_path" rule.
  **Fix:** in every test that drives `_load_state_from` / `load_state` /
  `load_bold_state` down the corrupt branch, add
  `monkeypatch.setattr(portfolio_mgr, "CRITICAL_ERRORS_LOG", tmp_path / "critical_errors.jsonl")`
  (the pattern already used in `tests/test_portfolio_mgr_corrupt_quarantine.py:29`).
  Then append a `resolution` line to `data/critical_errors.jsonl` for each of the
  six leaked `ts` values so the startup check and dispatcher clear. Consider a
  session-scoped autouse fixture in `tests/conftest.py` that redirects
  `portfolio_mgr.CRITICAL_ERRORS_LOG` (and `claude_gate.CRITICAL_ERRORS_LOG`) to
  tmp for the whole suite, to prevent the next module that grows a journal write
  from re-leaking.

## P1 — wrong persisted state / wrong risk number under realistic inputs

- **[P1-1] portfolio_mgr.py (whole module) vs docs/TRADING_PLAYBOOK.md:127 + agent_invocation.py:1159 — the real portfolio writer bypasses all atomic/backup protection.**
  Chain: Layer 2 mutates `portfolio_state*.json` by LLM `Edit`/`Write`
  (`--allowedTools "Edit,Read,Bash,Write"`), per the playbook, NOT via
  `save_state()`. A malformed Edit writes invalid JSON directly to the live file
  with no tmp+rename, no fsync, no pre-write backup — exactly the corruption the
  quarantine docstring (`portfolio_mgr.py:88`) attributes to "a hand-edit." The
  atomic-I/O invariant is satisfied by this subsystem but defeated end-to-end.
  **Fix:** stop having the LLM hand-edit state. Provide a tiny Bash-callable
  mutator (e.g. `python -m portfolio.apply_trade --strategy patient --json '…'`)
  that validates against `portfolio_validator.validate_portfolio` and writes via
  `portfolio_mgr.update_state` (atomic + rotated backup). Update the playbook to
  call that instead of "Edit …", and drop `Edit`/`Write` on the state files from
  the allow-list. At minimum, have Layer 2 write a successful `.bak` *before*
  editing.

- **[P1-2] portfolio_mgr.py:50-68 / 187 / 232 — backups are taken off unvalidated current bytes, so corruption launders into every `.bak`.**
  Chain: `_rotate_backups` runs before the atomic write and `shutil.copy2`s the
  current primary into `.bak` without parsing it. If the primary was corrupted
  out-of-band (P1-1) and any code then calls `save_state`/`update_state`, the
  corrupt primary becomes `.bak`, the old good `.bak` shifts to `.bak2`, etc. —
  across ≥3 saves all backups are corrupt. This is the mechanical reason a real
  incident reaches "unparseable and no backup recovered."
  **Fix:** validate before rotating. In `_rotate_backups` (or in `_save_state_to`
  before calling it), `load_json(path)` the current primary and SKIP rotating it
  into `.bak` if it does not parse to a dict — i.e. only ever promote a
  known-good file into the backup ring. Optionally also write the freshly-saved
  good state to `.bak` *after* a successful `_atomic_write_json` so a good
  baseline always exists.

## P2 — edge cases / races / cross-cutting inconsistencies

- **[P2-1] portfolio_mgr.py:196-234 — `save_state`/`save_bold_state`/`update_state` will overwrite a corrupt-but-present primary with fresh defaults.**
  Chain: after a corrupt load returns `_DEFAULT_STATE`, an unconditional
  `update_state(mutate_fn)` mutates those defaults and atomically writes them over
  the primary (rotating the corrupt file into `.bak` per P1-2). The hot loop
  avoids this via `main.py:795 if not STATE_FILE.exists()`, but the module
  guarantees nothing. The quarantine docstring (`:88-91`) explicitly calls this
  out as the danger yet the write functions don't enforce it.
  **Fix:** gate `_save_state_to`/`update_state`: if a `*.corrupt-*` sibling exists
  for `path` and the in-memory state equals `_DEFAULT_STATE` (i.e. we're about to
  persist defaults right after a quarantine), refuse the write and re-journal
  rather than clobber. Or carry a "loaded_from_corrupt" sentinel out of
  `_load_state_from` and have `update_state` raise instead of persisting defaults.

- **[P2-2] risk_management.py:376 vs :469/:912 and monte_carlo.py:305 — three different ATR stop levels for the same position.**
  Chain: `compute_stop_levels` applies a 3% floor and a 15% ATR cap
  (`stop_distance_pct = max(2*atr_pct, 3.0)`, `atr_pct = min(atr_pct, 15.0)`),
  but `compute_probabilistic_stops` (`:469`), `check_atr_stop_proximity` (`:912`)
  and `monte_carlo.simulate_ticker` (`:305`) use plain
  `entry_price*(1 - 2*atr_pct/100)` with no floor/cap. So the stop-hit
  probability and the "danger zone < 1.0x ATR" flag are computed against a
  DIFFERENT, tighter/looser stop than the one `compute_stop_levels` reports to the
  agent — the probabilities don't correspond to the displayed stop. For a
  low-ATR instrument the reported stop is 3% but the modeled stop is e.g. 1.2%,
  understating `stop_hit_prob`.
  **Fix:** extract a single `_atr_stop_price(entry, atr_pct)` helper (floor 3%,
  cap 15% ATR) and call it from all four sites so the displayed stop and every
  probability/flag use the same level.

- **[P2-3] monte_carlo_risk.py:408 — `compute_portfolio_var` trusts `agent_summary["fx_rate"]` raw, re-introducing the P1-15 anti-pattern.**
  Chain: `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)`. The whole
  point of `risk_management._resolve_fx_rate` (`:121`) was that a stale/erroneous
  `fx_rate` of `1.0` understates SEK ~10x and once produced a false drawdown
  breach. The fallback-when-absent here is safe, but a PRESENT `1.0` (or any
  out-of-band value) passes straight through into `var_*_sek`/`cvar_*_sek`,
  reporting SEK tail risk 10x too small.
  **Fix:** route through `risk_management._resolve_fx_rate(agent_summary)` (or
  duplicate its `[FX_RATE_MIN, FX_RATE_MAX]` sanity band) instead of the raw
  `.get`. Same raw-`.get` pattern in `exit_optimizer.compute_exit_plan_from_summary:718`
  (`fx_rate = agent_summary.get("fx_rate", 10.85)`) — feeds warrant SEK P&L.

- **[P2-4] equity_curve.py:361-421 — FIFO round-trip matcher silently drops unmatched SELL shares, undercounting realized P&L.**
  Chain: `_pair_round_trips` builds all BUY queues first, then walks SELLs in
  list order. If a SELL's shares exceed the FIFO BUY shares available for that
  ticker (out-of-order timestamps, a partial-state import, or a SELL with no
  preceding BUY in the slice), `while shares_to_match > 0 and buy_queues[ticker]`
  just exits and the excess sold shares produce no round-trip — so
  `total_pnl_sek`, `profit_factor`, expectancy and Calmar all silently omit that
  realized P&L. No warning is logged.
  **Fix:** when `shares_to_match > 1e-10` remains after the queue drains, log a
  WARNING (ticker, residual shares) and/or record a synthetic round-trip so the
  realized SEK isn't dropped. Also sort transactions by timestamp before matching
  to make FIFO order-independent.

- **[P2-5] equity_curve.py:489-499 vs 463-467 — win/loss classification uses gross `pnl_pct` while profit_factor uses net `pnl_sek`, so they can disagree.**
  Chain: `wins`/`losses` split on `t["pnl_pct"] > 0` (gross price move, fees
  excluded — see the comment at `:391-397`), but `gross_profit`/`gross_loss` for
  `profit_factor` sum `t["pnl_sek"]` (net of fees). A trade with +0.1% gross that
  is net-negative after fees counts as a "win" for `win_rate`/streaks but a loss
  for `profit_factor` — internally inconsistent metrics reported side by side.
  **Fix:** pick one basis per metric family and document it, or classify
  win/loss on `pnl_sek` for consistency with `profit_factor`/expectancy. (The
  in-code comment acknowledges the split is intentional; if so, at least surface
  both `win_rate_gross` and `win_rate_net` rather than one ambiguous number.)

- **[P2-6] trade_guards.py:32,126,264 — `record_trade` cooldown state is only thread-safe, not process-safe.**
  Chain: `_state_lock` is an in-process `threading.Lock`. The loop, the dashboard,
  and concurrent Layer 2 subprocesses are separate processes; two of them doing
  `_load_state` → mutate → `atomic_write_json(STATE_FILE)` will last-writer-win,
  dropping one process's cooldown/loss-streak update. Effect: an overtrading guard
  can be silently reset, letting a re-trade through inside the cooldown window.
  **Fix:** wrap the read-modify-write in `file_utils.jsonl_sidecar_lock`-style
  cross-process locking (a sidecar lock on `trade_guard_state.json`), the same
  primitive `atomic_append_jsonl` already uses.

- **[P2-7] exit_optimizer.py:614-620 — hold-to-close EV is the mean of 5 terminal percentiles, not the true expected P&L; not comparable to the limit candidates' EV.**
  Chain: `hold_ev = mean(pnl at terminal percentiles [10,25,50,75,90])`. Warrant
  P&L is clipped at 0 on knock-out (`_compute_pnl_sek:322,330`), so the terminal
  P&L distribution is asymmetric and a 5-point percentile mean is a biased
  estimator of E[P&L]. Meanwhile limit candidates use
  `fill_prob*pnl + (1-fill_prob)*fallback_pnl` where `fallback_pnl` is P&L at the
  MEDIAN terminal (`:560-561`), another non-expectation. `recommended` is chosen by
  `candidates.sort(key=ev_sek)` (`:636`) across these inconsistently-defined EVs,
  so the ranking can pick the wrong action near knock-out.
  **Fix:** compute true expectations from the full path arrays:
  `hold_ev = mean([_compute_pnl_sek(pos, p, …) for p in terminal])` (vectorize or
  sample), and use the same all-paths conditional mean for the limit fallback
  (`mean(pnl over paths that did NOT hit target)`), so all EVs share one basis.

## P3 — maintainability / minor

- **[P3-1] monte_carlo_risk.py / monte_carlo.py / exit_optimizer.py — default `seed=None` makes VaR/CVaR/exit EV non-deterministic run-to-run.**
  At 10K paths VaR jitters a few percent between cycles; dashboards/journals show
  numbers that don't reproduce. Acceptable for production randomness but worth a
  fixed seed for the reported (vs. internal) figure, or a note that the value is a
  noisy estimate. `exit_optimizer.compute_exit_plan_from_summary` also never
  forwards a `seed` to `compute_exit_plan` (`:751`) — it can't be made
  reproducible by callers at all.

- **[P3-2] risk_management.py:252-270 — agent_summary-empty fallback values held positions at cash-only, making drawdown look tiny.**
  Already self-documented with a WARNING (good), but the circuit breaker reads
  optimistically when the price feed is stale while positions are underwater. The
  P0-style fail-safe is only wired for non-finite values (`:291`), not for
  feed-stale. Consider treating "holdings present but summary empty" as
  inconclusive (skip the breach decision) rather than reporting a small drawdown.

- **[P3-3] monte_carlo.py:91 `drift_from_probability` clamps `p_up` to [0.01,0.99]`** — fine, but combined with `_get_directional_probability` scaling BUY→[0.5,0.8]
  (`:360`) the drift can only ever express a bounded edge; documented here only so
  a future reader doesn't mistake MC drift for a free parameter.

---

## Summary

The headline "recurring live portfolio corruption" is **not a portfolio loss** —
it is a **test-isolation leak (P0-1)**: corruption tests in
`test_portfolio_mgr_core.py` / `test_portfolio_mgr.py` / `test_io_safety_sweep.py`
fail to patch `portfolio_mgr.CRITICAL_ERRORS_LOG`, so the PR's new
quarantine-journal side-effect writes real `critical` entries into the live
`data/critical_errors.jsonl`; the startup check and fix-agent dispatcher then read
those test artifacts as live incidents. The live portfolio files are intact
(26 KB / 11 KB, untouched). Fix the tests and append `resolution` lines for the
six leaked timestamps.

The merged quarantine/fail-loud read path itself is **well-engineered and
adequate** for what it does (content-addressed once-only quarantine, bytes
captured pre-recovery, never raises) — but it defends only the READ side. The
real corruption SOURCE is the Layer 2 LLM hand-editing the JSON via `Edit`/`Write`
(`TRADING_PLAYBOOK.md:127`, `agent_invocation.py:1159`), entirely bypassing
`portfolio_mgr`'s atomic write + backup machinery (P1-1). And backups are copied
off unvalidated bytes (P1-2), so real corruption can launder through the whole
`.bak` ring — which is precisely how an incident reaches "no backup recovered."
Route LLM trade writes through an atomic, validated mutator and only ever rotate
known-good files into backups.

Risk-math findings are secondary but real: three divergent ATR stop levels feeding
mismatched stop probabilities (P2-2), raw `fx_rate` trust in VaR/exit SEK
conversions re-opening the P1-15 10x bug (P2-3), FIFO dropping unmatched sell
P&L (P2-4), inconsistent gross-vs-net win classification (P2-5), process-unsafe
trade-guard state (P2-6), and a biased/inconsistent hold-to-close EV that drives
the exit recommendation (P2-7). The Monte Carlo GBM and t-copula cores
(`monte_carlo.py`, `monte_carlo_risk.py`) are correct — the C9 Gaussian-marginal
fix, antithetic variates, Cholesky PSD fallback, and `_nearest_psd` are all sound;
only the `seed=None` non-determinism (P3-1) and the raw-fx input (P2-3) detract.
