# Improvement Plan — Auto-Session 2026-05-10

## Date: 2026-05-10 (Sunday)
## Branch: improve/auto-session-2026-05-10

## 1. Bugs & Problems Found

### CRITICAL

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| C1 | `send_telegram(msg)` missing required `config` — raises silent `TypeError` | `main.py:936,948` | Safeguard alerts never reach Telegram |
| C2 | CF header bypass: LAN clients spoof `Cf-Access-Authenticated-User-Email` | `dashboard/auth.py:131` | Unauthorized dashboard access |
| C3 | COT positioning reads wrong deep context path (dead 27 days) | `signals/cot_positioning.py:347` | Signal produces 0 samples |
| C4 | `prune_jsonl` uses predictable `.tmp` path (not `mkstemp`) | `file_utils.py:320` | Race on concurrent prune |
| C5 | `crypto_loop._pid_alive` missing `subprocess` import on POSIX | `data/crypto_loop.py` | `NameError` crash |
| C6 | `forecast_accuracy.py` raw `read_text()` instead of `load_jsonl` | `forecast_accuracy.py:75,92,380` | Torn-line data integrity |

### HIGH

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| H1 | `reporting.py` imports private `_atomic_write_json` from portfolio_mgr | `reporting.py:11` | Fragile alias coupling |
| H2 | Dead `CORRELATED_PAIRS` for removed tickers | `risk_management.py` | Dead code |
| H3 | Stale `HORIZON_SIGNAL_WEIGHTS` for disabled signals | `signal_engine.py` | Dead weight |
| H4 | Swedish holidays missing Whit Monday | `market_timing.py:216` | Warrant trading on closed day |
| H5 | `claude_invocations.jsonl` never pruned | `claude_gate.py` | Unbounded growth |
| H6 | `crypto_cross_asset.py` orphan — never registered | `signals/crypto_cross_asset.py` | Dead file |
| H7 | `_CORE_SIGNALS` + `@register_signal` unused | `signal_registry.py` | Dead API |
| H8 | `_loading_timestamps` not cleaned on error path | `shared_state.py:113` | Spurious eviction logs |
| H9 | `_sell_in_may` SELL bias — root cause of calendar 29.3% | `calendar_seasonal.py:160` | Blocks signal re-enable |
| H10 | 9 unresolved calendar critical_errors from 2026-05-09 | `data/critical_errors.jsonl` | Alert noise |

### MEDIUM

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| M1 | ORB predictor DST bug | `orb_predictor.py:32` | Wrong morning window in summer |
| M2 | `distance_to_stop_pct` wrong denominator | `risk_management.py:373` | Overstated safety margin |
| M3 | `_write_lock` dead code | `gpu_gate.py:98` | Dead function |
| M4 | Dead `CORRELATION_PRIORS` for removed tickers | `monte_carlo_risk.py` | Dead data |

## 2. Implementation Batches

### Batch 1: Critical bugs (6 files)
1. `portfolio/main.py` — Fix `send_telegram(msg)` → `send_telegram(msg, config)`
2. `dashboard/auth.py` — Only trust CF header when `dashboard_token` is configured
3. `portfolio/file_utils.py` — Fix `prune_jsonl` to use `tempfile.mkstemp`
4. `data/crypto_loop.py` — Add `subprocess` import at module level
5. `portfolio/forecast_accuracy.py` — Use `file_utils.load_jsonl` for all reads
6. `portfolio/signals/cot_positioning.py` — Fix deep context path

### Batch 2: Signal cleanup (4 files, 1 deletion)
1. `portfolio/signals/calendar_seasonal.py` — `_sell_in_may` SELL→HOLD
2. `portfolio/signal_engine.py` — Clean `HORIZON_SIGNAL_WEIGHTS`
3. `portfolio/signal_registry.py` — Remove `_CORE_SIGNALS` dead code
4. DELETE `portfolio/signals/crypto_cross_asset.py`

### Batch 3: Infrastructure cleanup (7 files)
1. `portfolio/reporting.py` — Fix import to `file_utils.atomic_write_json`
2. `portfolio/risk_management.py` — Remove dead `CORRELATED_PAIRS` + fix stop denominator
3. `portfolio/monte_carlo_risk.py` — Remove dead `CORRELATION_PRIORS`
4. `portfolio/market_timing.py` — Add Whit Monday to Swedish holidays
5. `portfolio/shared_state.py` — Fix `_loading_timestamps` error-path cleanup
6. `portfolio/gpu_gate.py` — Remove dead `_write_lock`
7. `portfolio/main.py` — Add `claude_invocations.jsonl` to pruning list

### Batch 4: Final fixes (3 files)
1. `portfolio/orb_predictor.py` — DST-aware morning window
2. `portfolio/tickers.py` — Update calendar comment (root cause fixed)
3. `data/critical_errors.jsonl` — Resolve 9 calendar entries
