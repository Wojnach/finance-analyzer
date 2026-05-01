# PLAN — midfinance follow-ups (2026-05-02)

**Date:** 2026-05-02
**Branch:** `feat/midfinance-followups-2026-05-02`
**Goal:** Resolve all remaining code-side items from the 2026-05-01 midfinance merge punch list. Three batches:
1. Back-port the warrant-sizing P1 fix from oil → crypto.
2. Add a stale-heartbeat watchdog so the heartbeats actually do something.
3. Stabilise the 3 documented xdist test flakes (4th is an external module, correctly --ignore'd).

---

## Context

After the 2026-05-01 multi-asset merge (commit `0c00ac8c`), 3 code-side items
remained (see prior session summary):

- **Crypto sizing latent bug** — `data/crypto_swing_trader.py:491` still has
  `warrant.get("ask") or warrant.get("last") or 1.0`. Codex caught this in
  oil; same pattern exists in crypto, just predates the codex review.
  Crypto also ships DRY_RUN so behavioural blast-radius is zero today, but
  the latent bug would fire the moment DRY_RUN is flipped.
- **Heartbeats are written, never read** — `data/{crypto,oil}_loop.heartbeat`
  are JSON-formatted but no watchdog reads them. The dashboard `/api/oil`
  surfaces it as a field but nothing alerts when it goes stale.
- **3 documented xdist flakes** in `docs/TESTING.md`:
  - `test_consensus.py::TestStockConsensus::test_stock_buy_with_3_voters`
    — passes in isolation, fails under `-n auto` (signal_engine cache leak).
  - `test_fg_regime_gating.py::TestFGGatedTrendingUp::test_extreme_fear_gated_trending_up`
    — needs investigation.
  - `test_backtester.py::TestRunBacktest::test_days_filter_applied` — needs
    investigation.
  - `tests/integration/test_strategy.py::test_strategy_loads` — missing
    `ta_base_strategy` (Freqtrade), correctly ignored via `--ignore=tests/integration`.
    Out of scope here.

---

## What this PR does

### Batch A — Crypto sizing back-port

| File | Change |
|---|---|
| `data/crypto_swing_trader.py` | Refuse to size when `warrant.get("ask")` and `warrant.get("last")` are both missing/zero. Mirror the oil fix verbatim — same reason text. |
| `data/crypto_warrant_refresh.py` | Persist `bid`, `ask`, `last` into each catalog entry (parity with `oil_warrant_refresh.py` Batch C). |
| `tests/test_crypto_swing_trader.py` | Add a regression test asserting BUY refused when warrant has no live quote. |

### Batch B — Loop health rollup + watchdog

| File | Change |
|---|---|
| `portfolio/loop_health.py` | New module. `read_loop_health()` returns a dict keyed by loop_name → {is_alive, age_seconds, is_fresh, payload}. Reads all `data/*_loop.heartbeat` files (currently crypto + oil; metals + main loops can be added later when they grow heartbeats). |
| `dashboard/app.py` | New `/api/loop_health` endpoint returning the rollup. |
| `scripts/loop_health_watchdog.py` | One-shot script that reads heartbeats, sends a telegram alert if any loop is stale (>5 min) or missing. Designed for a periodic scheduled task (suggest every 30 min). Has a per-loop cooldown so a dead loop doesn't spam telegram every 30 min. |
| `scripts/win/install-loop-health-watchdog-task.ps1` | Registers `PF-LoopHealthWatchdog` (every 30 min, AtLogOn). User runs to enable. |
| `tests/test_loop_health.py` | Cover: fresh, stale, missing, malformed JSON, file-IO failure paths. |
| `tests/test_loop_health_watchdog.py` | Cover: alert-fires-on-stale, no-alert-when-fresh, cooldown gate. |

### Batch C — Test flake stabilisation

| File | Change |
|---|---|
| `tests/conftest.py` | Extend `_reset_module_state` autouse fixture to reset more signal_engine caches (the consensus-test leak source per `docs/TESTING.md:52`). |
| `tests/test_consensus.py` | Add explicit `_state_reset.reset_all()` call to TestStockConsensus class setUp. |
| `tests/test_fg_regime_gating.py` | Investigate + fix root cause. Likely shared signal-engine state. |
| `tests/test_backtester.py` | Investigate + fix root cause. May be tmp_path patch missing. |

If a flake's root cause is genuinely module-level shared state we can't reset (e.g. C-extension cache), document it explicitly in `docs/TESTING.md` with a `pytest -n auto --ignore=...` recipe, then close out.

---

## What this PR does NOT do

- **Does not flip DRY_RUN to False anywhere.**
- **Does not auto-register any scheduled tasks.** Provides install scripts.
- **Does not touch oil/MSTR loops** (those are stable from the 2026-05-01 merge).
- **Does not implement Brent expansion** — the catalog already labels OLJA warrants as `OIL-USD` underlying despite tracking Brent under the hood. Cleanest fix is to reclassify, but that requires a live Avanza probe to confirm which warrants track WTI vs Brent. Defer until the user runs that probe (per OIL_LOOP_NOTES.md).
- **Does not add new oil signal modules** (`oil_cross_asset` etc.) — `oil_precompute.py` already produces the deep context the swing trader needs; adding voters risks over-fitting on a 30-day window.
- **Does not touch `tests/integration/test_strategy.py`** — `ta_base_strategy` (Freqtrade) is an external module dependency, correctly --ignore'd per `docs/TESTING.md:36`.

---

## Risks

1. **conftest.py global fixture changes** can affect every test in the repo. Mitigation: keep additions strictly additive (only add new resets, never remove existing ones); pytest -n auto regression run before merge.
2. **Loop health watchdog spamming telegram** if heartbeats are persistently stale. Mitigation: cooldown logic in `scripts/loop_health_watchdog.py` (default 4h between alerts per-loop).
3. **Crypto warrant catalog migration** — entries written before this PR don't have bid/ask/last. The trader gracefully refuses to size (Batch A fix), so existing catalog files won't crash; they'll just block BUYs until refresh runs. This is the same behaviour that already shipped for oil and is the correct conservative default.

---

## Execution order

| Batch | Files | LOC | Tests | Commit prefix |
|---|---|---|---|---|
| **A** | crypto_swing_trader + crypto_warrant_refresh + 1 test | ~80 | yes | `fix(crypto): back-port warrant-sizing safety (Batch A)` |
| **B** | loop_health + dashboard + watchdog + install.ps1 + 2 test files | ~700 | yes | `feat(ops): loop health rollup + watchdog (Batch B)` |
| **C** | conftest.py + 3 test fixes + TESTING.md | ~200 | yes | `fix(tests): stabilise xdist flakes (Batch C)` |

After Batch C:
1. **Codex review** on the worktree branch.
2. Address P1/P2 findings.
3. **Full pytest** with `-n auto`. The benchmark is "≤4 known flakes from before this PR"; aim for 0 with these fixes.
4. **Merge** to main.
5. **Push** via Windows git.
6. **Clean up** worktree + branch.

---

## Why this design

1. **Back-port the proven fix** — codex already validated the oil fix; crypto gets the exact same treatment, no design debate.
2. **Heartbeats become useful** — already write them, may as well alert on staleness. Single new module + small dashboard endpoint.
3. **Stabilise tests, don't redesign them** — add state resets to existing fixtures rather than rewriting test files. Leave the 4th flake (Freqtrade external dep) alone since it's correctly classified.
4. **No new features, no new flips** — pure operational hardening. Same risk profile as the 2026-05-01 merge.
