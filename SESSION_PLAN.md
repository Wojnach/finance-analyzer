# Parallel Session Plan — Phase 2 Implementation

> Created 2026-02-10. Two Claude Code sessions working in parallel.

## Session A — Strategy Code (ACTIVE)

Owner: Claude session in terminal 1
Touches: `user_data/strategies/ta_base_strategy.py`, `tests/`

### Tasks

1. **Multi-timeframe 1h filter**
   - Add `informative_pairs()` returning 1h candles for each pair
   - Merge 1h EMA(50) trend into 5m dataframe
   - Add ADX(14) on 1h — only enter when ADX > 25 (trending market)
   - Gate `enter_long`: require 1h EMA uptrend + ADX > 25

2. **Top 15 candlestick patterns**
   - Add TA-Lib pattern functions: CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, CDLEVENINGSTAR, CDL3WHITESOLDIERS, CDL3BLACKCROWS, CDLSHOOTINGSTAR, CDLHANGINGMAN, CDLDOJI, CDLHARAMI, CDLPIERCING, CDLDARKCLOUDCOVER, CDLINVERTEDHAMMER, CDLMARUBOZU, CDLKICKING
   - Bullish patterns boost entry confidence, bearish patterns reduce it
   - Add hyperoptable weight parameter for pattern contribution

3. **Risk management in strategy**
   - ATR-based position sizing via `custom_stake_amount()`
   - Max daily loss tracking via `custom_exit()` — stop entering after X% daily drawdown
   - Drawdown kill switch — if total drawdown hits threshold, block new entries + Telegram alert

4. **Unit tests**
   - Tests for 1h informative data merging
   - Tests for candlestick pattern scoring
   - Tests for risk management logic (daily loss, drawdown kill)

### Files modified

- `user_data/strategies/ta_base_strategy.py`
- `tests/unit/test_indicators.py`
- `tests/unit/test_signal_logic.py`
- Possibly new: `tests/unit/test_risk_management.py`

---

## Session B — Infrastructure + Validation

Owner: Claude session in terminal 2
Touches: `scripts/`, `docs/`, `TODO.md`, systemd configs

### Tasks

1. **Walk-forward validation script**
   - New script: `scripts/ft-walkforward.sh` (or Python wrapper)
   - Split 2yr data into rolling windows: 6-month train / 2-month test
   - Run hyperopt on train window, backtest on test window, roll forward
   - Output summary table: profit factor, trades, drawdown per window
   - This validates whether the strategy generalizes or is overfit

2. **Update TODO.md**
   - Sync with actual project state — many Phase 1 items are marked undone but were completed
   - Update backtest results section with latest 2yr numbers
   - Mark Phase 2 items that Session A is implementing

3. **Systemd user service for dry-run**
   - Create `~/.config/systemd/user/ft-dry-run.service`
   - Runs `podman start ft-dry-run` or equivalent
   - Survives reboot, auto-restarts on failure
   - Enable with `systemctl --user enable ft-dry-run`

4. **Health check script**
   - New script: `scripts/ft-health.sh`
   - Checks: is container running? Last log timestamp? Any errors?
   - Optional: send Telegram alert if unhealthy
   - Could be run by a systemd timer

### Files modified

- `scripts/ft-walkforward.sh` (new)
- `scripts/ft-health.sh` (new)
- `TODO.md`
- Systemd unit files in `~/.config/systemd/user/`

### Files NOT to touch (Session A owns these)

- `user_data/strategies/ta_base_strategy.py`
- `tests/unit/*`
- `tests/integration/*`

---

## Coordination Rules

- **No overlapping files.** Session A owns strategy + tests. Session B owns scripts + docs + infra.
- **Commit independently.** No need to wait for the other session.
- **After both done:** Session A re-runs hyperopt + backtest with new indicators, Session B runs walk-forward validation on the updated strategy.
- **Communication:** Update this file with status. Mark tasks DONE as you complete them.

## Status

| Task                      | Session | Status |
| ------------------------- | ------- | ------ |
| Multi-timeframe 1h filter | A       | DONE   |
| Candlestick patterns      | A       | DONE   |
| Risk management           | A       | DONE   |
| Integration tests (16/16) | A       | DONE   |
| Walk-forward script       | B       | DONE   |
| Update TODO.md            | B       | DONE   |
| Systemd dry-run service   | B       | DONE   |
| Health check script       | B       | DONE   |

## Session A Notes

- Strategy loads and backtests cleanly with all new features
- Backtest with OLD hyperopt params: 123 trades, -12.92%, profit factor 0.55
- This is EXPECTED — old params don't account for trend filter, patterns, or risk controls
- **Must re-hyperopt** after Session B builds the walk-forward framework
- New hyperoptable params: pattern_weight, adx_threshold, max_daily_loss_pct, max_drawdown_pct
- Hyperopt spaces needed: `--spaces buy sell` (covers all new params)
- The old `ta_base_strategy.json` should be deleted before re-hyperopt (stale params)
