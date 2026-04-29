# Plan — Crypto + MSTR Swing Subsystem (mirrors metals subsystem)

**Date:** 2026-04-30
**Branch:** `feat/crypto-mstr-swing`
**Owner:** Sydney + Claude

## Goal

Build a "metals-equivalent" autonomous trading subsystem for **BTC**, **ETH**, and **MSTR**.

Gold (XAU) is already covered by the metals subsystem (64 XAU warrants in
`data/metals_warrant_catalog.json`, full support in `data/metals_loop.py`,
`data/metals_swing_trader.py`, `data/fin_fish_config.py`). No work needed for gold.

## Scope

### Already exists
- `portfolio/mstr_loop/` — full MSTR loop with strategies, risk, execution, session, state, telegram. Complete.
- `data/crypto_data.py` — Fear & Greed, news, MSTR price, on-chain summary (TTL caching).
- `portfolio/signals/crypto_macro.py`, `signals/funding_rate.py`, `signals/network_momentum.py`,
  `signals/hash_ribbons.py`, `signals/crypto_evrp.py` — crypto signals.

### Gaps (this PR fills)

#### Crypto (BTC + ETH)
1. `data/crypto_swing_config.py` — thresholds, sizing, exits (mirror `metals_swing_config.py`).
2. `data/crypto_warrant_catalog.json` — initial warrant catalog scaffold.
3. `data/crypto_warrant_refresh.py` — Avanza warrant discovery for BTC/ETH instruments.
4. `data/crypto_swing_trader.py` — autonomous BUY/SELL logic (parameterized per instrument).
5. `data/crypto_loop.py` — 60s cycle with 10s fast-tick monitor (BTC + ETH together, since both 24/7).
6. `portfolio/crypto_precompute.py` — emits `data/crypto_deep_context.json` every 4h (funding, fear/greed, on-chain BTC, ETH staking yield).
7. `portfolio/signals/crypto_cross_asset.py` — aggregated crypto cross-asset signal.
8. State files (created on first run): `data/crypto_swing_state.json`,
   `data/crypto_swing_decisions.jsonl`, `data/crypto_swing_trades.jsonl`,
   `data/crypto_value_history.jsonl`, `data/crypto_signal_log.jsonl`,
   `data/crypto_signal_outcomes.jsonl`, `data/crypto_risk.json`.

#### MSTR
9. `portfolio/mstr_precompute.py` — emits `data/mstr_deep_context.json` (NAV premium, BTC correlation, IV skew).

#### Dashboard
10. `dashboard/app.py` — add `/api/btc`, `/api/eth`, `/api/mstr`, `/api/crypto` endpoints.

#### Tests
11-15. Tests for each new module.

## Out-of-scope (deferred)

- `crypto_loop.py` end-to-end live integration to Avanza order placement (DRY_RUN=True default).
- Per-position trailing-stop hardware orders for crypto warrants (depends on probe of XBT-TRACKER metadata).
- ORB predictor for crypto (24/7 markets need different design).
- ISKBETS / fishing planners — generic `price_targets.py` + `exit_optimizer.py` already work.

## Execution Order

| Batch | Files | LOC budget |
|---|---|---|
| 1 | swing_config + warrant_catalog.json + warrant_refresh + signals/crypto_cross_asset | ~600 |
| 2 | crypto_precompute + mstr_precompute | ~500 |
| 3 | crypto_swing_trader + crypto_loop | ~1500 |
| 4 | dashboard/app.py edits | ~120 |
| 5 | 5 test files | ~700 |
| 6 | verify, merge, push | n/a |

## Risk

- **Live trading risk: zero.** `DRY_RUN=True` in `crypto_swing_config.py`. No real orders.
- **No live loop side effects.** New modules are inert until `crypto_loop.py` is wired into a scheduled task — explicitly deferred.
- **Worktree:** `feat/crypto-mstr-swing`. Merged into main and cleaned up after green tests.

## Why this design

1. **Unified `crypto_*` namespace** for BTC+ETH (same venue, same execution pattern).
2. **MSTR stays in its existing `mstr_loop` package** — only adds precompute + dashboard endpoint.
3. **Dry-run-by-default** parallels how metals_loop shipped originally.
