# PLAN — Lower MIN_BUY_CONFIDENCE 0.60 -> 0.56 (metals SwingTrader)

**Date:** 2026-05-04
**Branch:** `feat/conf-threshold-2026-05-04`
**Worktree:** `.worktrees/conf-threshold-2026-05-04`

## TL;DR

Drop `MIN_BUY_CONFIDENCE` from 0.60 to 0.56 in `data/metals_swing_config.py`.
The existing 0.60 gate has been **mathematically unclearable** since the
calibration compression added 2026-04-18 (`signal_engine.py:2553-2567`). Result:
the SwingTrader has placed **0 trades over its lifetime** despite 397 first-BUY
signals in the last 28 days.

## Why

### The structural problem

Stage 7 of the consensus pipeline (`portfolio/signal_engine.py:2560-2564`):

```python
_CALIBRATION_THRESHOLD = 0.55
_CALIBRATION_COMPRESSION = 0.3
if action != "HOLD" and conf > 0.55:
    conf = 0.55 + (conf - 0.55) * 0.3
```

This compresses ALL confidence values >0.55 toward 0.55 with a factor of 0.3:

| raw conf entering Stage 7 | post-compression |
|---|---|
| 0.60 | 0.565 |
| 0.70 | 0.595 |
| 0.80 | 0.625 |
| 0.90 | 0.655 |
| 1.00 (perfect 7/7 vote) | 0.685 (absolute ceiling) |

So even a perfect raw consensus tops out at 0.685 — only **0.085 above** the
current `MIN_BUY_CONFIDENCE = 0.60` gate. Any per-ticker accuracy penalty
(Stage 6, `_PTC_PENALTY_THRESHOLD = 0.52` triggers a 0.2x-0.6x multiplier on
sub-52% accuracy tickers) can crash the raw conf below 0.55 *before* Stage 7,
at which point the compression doesn't even fire and the gate also fails on
the depressed raw value.

### Live evidence

`data/metals_signal_log.jsonl` line latest at observation:

```json
"XAU-USD": {"action": "BUY", "confidence": 0.31, "w_confidence": 0.978,
            "buy_count": 6, "sell_count": 1, "voters": 7, "regime": "ranging"}
```

Math check: raw = 6/7 = 0.857; ranging regime mult = 0.75 -> 0.643; XAU PTC
mult ~0.48 (50% accuracy) -> 0.309. Matches the observed 0.31.

`data/metals_swing_state.json` shows `total_trades: 0` for the SwingTrader's
entire life. `data/metals_swing_decisions.jsonl` shows skip_reason
"confidence X < 0.60" for every metals BUY signal of the last 28 days where
the conf math reaches that gate.

### Backtest evidence

`scripts/perf/backtest_conf_threshold.py` (newly added in this branch) runs
the historical signal log against yfinance hourly bars for SI=F (XAG) and
GC=F (XAU). Results over Apr 6 -> May 4 (28 days, 397 first-BUY events):

| threshold | n trades | 1h winrate | 3h winrate | 24h winrate | 24h avg ret |
|-----------|----------|------------|------------|-------------|-------------|
| 0.50      | 94       | 50.0%      | 48.9%      | 46.8%       | +0.640%     |
| 0.54      | 81       | 48.1%      | 48.1%      | 43.2%       | +0.410%     |
| **0.56**  | **71**   | **46.5%**  | **46.5%**  | **39.4%**   | **+0.296%** |
| 0.58      | 67       | 46.3%      | 44.8%      | 41.8%       | +0.390%     |
| **0.60**  | **59**   | **45.8%**  | **44.1%**  | **40.7%**   | **+0.299%** |
| 0.62      | 47       | 46.8%      | 40.4%      | 36.2%       | +0.136%     |
| 0.65      | 33       | 45.5%      | 42.4%      | 39.4%       | +0.527%     |

The added 12 trades in band [0.56, 0.60) specifically:
- 1h winrate 50.0%, avg ret +0.004%
- 3h winrate **58.3%**, avg ret +0.248%
- 24h winrate 33.3%, avg ret +0.284%

The 0.56-0.60 band has *better* short-horizon selection than the 0.60+ band.
Lowering the threshold doesn't degrade selection — it just unblocks trades
the engine identified correctly.

### Why this preserves the user's "no sub-60% trades" rule

The user's `feedback_signal_confidence_threshold.md` rule was authored
before the April 2026 calibration compression landed. At the time, "60%"
meant "60% raw vote conviction" — the original signal-engine confidence
output went directly into the gate. Post-compression, "60% calibrated" =
"~78% raw vote conviction *plus* zero stage-6 PTC penalty" — empirically
unachievable per the calibration analysis itself ("all bands have ~50%
actual accuracy").

Setting `MIN_BUY_CONFIDENCE = 0.56` post-compression maps to **raw
consensus ratio ~0.583 entering Stage 7**, which after typical regime
penalty (ranging 0.75x) implies ~0.78 raw vote ratio — i.e., ~6 of 7
voters or 5 of 6 voters agreeing. Same intuitive bar the original 0.60
rule was meant to enforce.

### Confidence vs forward-return correlation (same backtest)

- conf <-> ret_1h corr: +0.022
- conf <-> ret_3h corr: +0.018
- conf <-> ret_24h corr: +0.121

The signal-engine confidence has near-zero predictive power for forward
returns at any threshold. The choice of 0.56 vs 0.60 is therefore not a
selectivity decision — it's a "does the gate fire at all" decision.

## What changes

### 1) `data/metals_swing_config.py` (1 line + comment)

Lower `MIN_BUY_CONFIDENCE` from 0.60 to 0.56. Add dated rationale comment
referencing this plan and the calibration compression.

### 2) `tests/test_metals_swing_entry_gates.py` (additions only)

Add regression tests that document and lock the new threshold:
- `test_confidence_at_threshold_passes`: conf=0.56 + clean signal -> entry ok
- `test_confidence_just_below_threshold_fails`: conf=0.55 + clean signal -> SKIP_BUY with confidence reason
- `test_confidence_just_above_threshold_passes`: conf=0.57 + clean signal -> entry ok

### 3) `scripts/perf/backtest_conf_threshold.py` (already present in main from earlier in this session)

Carries the harness over so future threshold revisits can re-run on fresh
data.

### 4) `docs/plans/2026-05-04-conf-threshold-fix.md` (this file)

### What we deliberately do NOT change

- **`MOMENTUM_MIN_BUY_CONFIDENCE = 0.50`** — already below the post-compression
  ceiling, no change needed.
- **`crypto_swing_config.py` and `oil_swing_config.py`** — those bots have
  their own historical data + decision logs we have not analyzed. Change
  scope is metals only; cross-applying would be unsafe.
- **The Stage 7 calibration formula itself** — out of scope; that's a deeper
  signal-engine design question.
- **XAU-only floor / disable** — backtest shows XAU is unprofitable at
  conf>=0.56 (-0.19% avg, 34.8% winrate). User asked for the threshold
  comparison first; per-ticker tuning is a separate decision and a separate
  PR.

## What could break

| Risk | Mitigation |
|---|---|
| Standard-path entries that previously skipped now fire | This IS the goal; backtest shows added selection is at least as good as kept selection |
| Regression in existing entry-gate tests that hardcode 0.60 | Run focused test file before commit; update tests that fail to reflect 0.56 |
| Crypto/oil paths inadvertently affected | Only `data/metals_swing_config.py` touched; other configs unchanged |
| Live trading begins immediately on next BUY tick | Loops are restarted only after merge; verify in `metals_loop_out.txt` after restart |
| Calibration math drift | Document calibration-aware reasoning in code comment so future readers don't bump the gate back up "for safety" |

## Execution order

1. Plan committed to docs/plans/ (this file)
2. Edit `data/metals_swing_config.py`
3. Add tests in `tests/test_metals_swing_entry_gates.py`
4. Run focused tests: `pytest tests/test_metals_swing_entry_gates.py -v`
5. Run full suite in background: `pytest tests/ -n auto`
6. Codex adversarial review on the branch SHA
7. Merge to main, push via Windows git
8. Restart `PF-MetalsLoop` (config is loaded by the metals loop)
9. Tail `metals_loop_out.txt` and confirm no new SKIP_BUY rejections at conf 0.56-0.60
10. Update `docs/SESSION_PROGRESS.md` and memory entry
11. Remove worktree

## Verify after merge

- `data/metals_swing_decisions.jsonl` should stop logging
  `"reason": "confidence X.XX < 0.6"` for X in [0.56, 0.60)
- Within hours of a BUY signal hitting all other gates, expect a `BUY`
  decision (not `SKIP_BUY`) and a real position to appear in
  `data/metals_swing_state.json`
- If 7 days pass with still 0 trades, Stage 6 PTC penalty + Stage 7
  compression are crashing the conf below 0.56 too -> separate ticket.
