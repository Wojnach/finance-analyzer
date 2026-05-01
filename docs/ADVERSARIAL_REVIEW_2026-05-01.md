# Adversarial Review Synthesis — 2026-05-01

**Methodology**: Dual adversarial review. 8 independent specialist agents (one per subsystem)
reviewed the codebase in parallel. A separate independent review was conducted without seeing
agent results. This document cross-critiques in both directions, deduplicates, and synthesizes
a prioritized action plan.

**Subsystems**: signals-core, orchestration, portfolio-risk, metals-core, avanza-api,
signals-modules, data-external, infrastructure.

---

## P0 — Critical (Financial Loss or System Down)

### P0-1: `ticker_accuracy.py` — No neutral-outcome filter inflates accuracy
**Source**: Agent (signals-core, Finding 1)
**Cross-validation**: Confirmed. `accuracy_stats.py:97` has `_MIN_CHANGE_PCT=0.05` filter;
`ticker_accuracy.py` has no equivalent. The ticker-specific accuracy fed to Mode B Telegram
notifications and directional probability calculations is systematically wrong.
**Impact**: Overstated accuracy → oversized positions via Kelly → excess risk.
**Fix**: Apply `_vote_correct()` helper from accuracy_stats.

### P0-2: `ic_computation.py:19` — Relative `Path("data")` silently disables IC weights
**Source**: Agent (signals-core, Finding 2)
**Cross-validation**: Confirmed. Every other module uses `Path(__file__).resolve().parent.parent / "data"`.
When the scheduled task CWD differs from repo root, IC cache reads/writes go to a phantom directory.
**Impact**: IC-based weight multipliers silently fall to 1.0 for all signals every cycle.
**Fix**: `DATA_DIR = Path(__file__).resolve().parent.parent / "data"`

### P0-3: `signal_history.py` — No lock on concurrent read-modify-write
**Source**: Agent (signals-core, Finding 3)
**Cross-validation**: Confirmed. 5 ThreadPoolExecutor workers call `update_history()`
simultaneously. Last-writer-wins causes history entry loss.
**Impact**: Signal persistence scores, streaks, and history data are corrupted over time.
**Fix**: Add `threading.Lock` around the read-modify-write in `update_history()`.

### P0-4: `avanza_client.py` — TOTP order path bypasses `avanza_order_lock`
**Source**: Agent (avanza-api, Finding 2)
**Cross-validation**: Confirmed. `avanza_orders.py` imports `place_buy_order`/`place_sell_order`
from `avanza_control.py` which re-exports from `avanza_client.py`. The TOTP path has zero
integration with the order lock architecture.
**Impact**: Concurrent orders from metals loop + human CONFIRM can overdraw the ISK account.
**Fix**: Wrap `_place_order` in avanza_client.py with `avanza_order_lock`.

### P0-5: Drawdown circuit breaker caught by bare `except` — silently passes
**Source**: Agent (portfolio-risk, Finding 1)
**Cross-validation**: Confirmed. The entire block in `agent_invocation.py:402` is wrapped in
`try/except Exception` that logs WARNING and continues. Any failure in `check_drawdown`
waves the trade through.
**Impact**: Portfolio in 50%+ drawdown can continue trading.
**Fix**: Default to blocking on check failure, not passing.

### P0-6: `equity_curve.py:384` — P&L excludes fees; Kelly oversizes
**Source**: Agent (portfolio-risk, Finding 2)
**Cross-validation**: Confirmed. `pnl_sek = (sell_price - buy_price) * matched` doesn't
subtract computed `buy_fee_share + sell_fee_share`. All downstream metrics are gross, not net.
**Impact**: Win rate and expectancy overstated → Kelly recommends larger positions.
**Fix**: `pnl_sek -= buy_fee_share + sell_fee_share`

---

## P1 — Important (Incorrect Behavior)

### P1-1: `signal_engine.py` — `_rescued` variable not reset per loop iteration
**Source**: Agent (signals-core, Finding 10)
**Cross-validation**: Confirmed. When a signal passes cleanly (neither gated nor rescued),
`_rescued` retains value from previous iteration. A legitimately-passing signal gets 0.70x
weight if the prior signal was rescued.
**Impact**: Non-deterministic weight distortion depending on signal processing order.
**Fix**: `_rescued = False` at the top of the loop body.

### P1-2: Persistence filter cold-start seeds `cycles=MIN_CYCLES`
**Source**: Both (independent P1-A, agent Finding 6)
**Cross-validation**: Both reviews flagged this independently. On restart, all signals
immediately appear "persistent" without ever being validated across cycles.
**Impact**: First 1-2 cycles after restart trust noisy signals.
**Fix**: Seed with `cycles=1`.

### P1-3: Layer 2 bypasses `claude_gate.py` kill switch and `_invoke_lock`
**Source**: Agent (orchestration, P1-1)
**Cross-validation**: Confirmed. `invoke_agent()` calls `subprocess.Popen` directly.
`claude_gate.invoke_claude()` has rate limiter, kill switch, and serialization lock.
**Impact**: Concurrent Claude sessions possible; quota burn; concurrent file writes.
**Fix**: Route through `claude_gate` or at minimum acquire `_invoke_lock`.

### P1-4: `volume_flow.py:63` — VWAP is lifetime cumulative, not session-based
**Source**: Agent (signals-modules, Finding 5)
**Cross-validation**: Confirmed. Standard VWAP resets per session. This implementation
runs cumulative over the entire DataFrame.
**Impact**: Systematic BUY bias in uptrends, SELL bias in downtrends.
**Fix**: Detect session boundaries and reset cumsum per session.

### P1-5: `heikin_ashi.py:317` — Alligator shift makes `.iloc[-1]` read 3-8 bars stale
**Source**: Agent (signals-modules, Finding 7)
**Cross-validation**: Confirmed. Forward-shift for charting applied to signal computation.
**Impact**: 3-8 bar structural lag in fast markets.
**Fix**: Use unshifted values for signal computation.

### P1-6: `mean_reversion.py:462` — Seasonality detrend compounds against modified prices
**Source**: Agent (signals-modules, Finding 9)
**Cross-validation**: Confirmed. Loop reads `df[close].iloc[i-1]` which is already modified.
**Impact**: Cumulative error in detrended price series.
**Fix**: Copy original close column before loop; read from copy.

### P1-7: `macro_regime.py:203` — Yield threshold 1.5 is ~10x too high; permanently HOLD
**Source**: Agent (signals-modules, Finding 10)
**Cross-validation**: Confirmed. 150bps in 5 days is historically extreme (3-4 times in 40y).
**Impact**: Dead sub-signal consuming a voter slot.
**Fix**: Lower to 0.15 (15bps).

### P1-8: `metals_swing_trader.py` — Hard stop at 2% underlying / 10% cert is too tight
**Source**: Agent (metals-core, Finding 1)
**Cross-validation**: Confirmed. Memory rules state "5x certificates need -15%+ stops".
**Impact**: Stops fire repeatedly on normal silver intraday wicks.
**Fix**: Raise `HARD_STOP_UNDERLYING_PCT` from 2.0 to 3.0-4.0.

### P1-9: `metals_swing_trader.py:2823` — Hardcoded USD/SEK=10.85 in exit optimizer
**Source**: Agent (metals-core, Finding 4)
**Cross-validation**: Confirmed. Live FX available via `portfolio/fx_rates.py` but not used.
**Impact**: 5% FX error → EV thresholds crossed at wrong levels → wrong exit decisions.
**Fix**: Fetch live USD/SEK from `fx_rates`.

### P1-10: `avanza_orders.py` — CONFIRM races to match wrong pending order
**Source**: Agent (avanza-api, Finding 1)
**Cross-validation**: Confirmed. Any "CONFIRM" text fires against the newest pending order.
**Impact**: User confirms wrong order; wrong trade executes.
**Fix**: Require CONFIRM + order UUID or restrict to one pending at a time.

### P1-11: `kelly_sizing.py:139` — Fallback uses system-wide accuracy, not per-ticker
**Source**: Agent (portfolio-risk, Finding 4)
**Cross-validation**: Confirmed. BTC at 35% uses same win_prob as XAG at 55%.
**Impact**: Systematic oversizing for low-accuracy tickers.
**Fix**: Use per-ticker accuracy as primary path.

### P1-12: `trade_guards.py:373` — `should_block_trade()` never called anywhere
**Source**: Agent (portfolio-risk, Finding 5)
**Cross-validation**: Confirmed. Block-severity guards exist but are only advisory text.
**Impact**: Cooldown and position-rate limits are unenforceable soft prompts.
**Fix**: Call `should_block_trade()` from `agent_invocation.py`.

### P1-13: `fear_greed.py:100` — Crashes on API response without `"data"` key
**Source**: Agent (data-external, Finding 1)
**Cross-validation**: Confirmed. `body["data"][0]` unguarded.
**Fix**: `data_list = body.get("data", []); if not data_list: return None`

### P1-14: `onchain_data.py:101` — `_load_onchain_cache()` doesn't use `_coerce_epoch()`
**Source**: Agent (data-external, Finding 2)
**Cross-validation**: Confirmed. Can raise TypeError if `ts` is ISO string.
**Fix**: `ts = _coerce_epoch(data.get("ts", 0))`

### P1-15: FX rate fallback to 1.0 triggers false circuit breaker
**Source**: Independent review (P1-O)
**Cross-validation**: Confirmed in `risk_management.py:99`.
**Impact**: Portfolio appears 10.5x smaller; circuit breaker fires incorrectly.
**Fix**: Cache last known good FX rate; only use 1.0 as absolute last resort.

---

## P2 — Degraded (Performance, Reliability, Minor Risk)

| ID | Subsystem | Issue |
|----|-----------|-------|
| P2-1 | infrastructure | `rotate_jsonl` uses non-unique .tmp, no fsync, no lock coordination |
| P2-2 | infrastructure | `message_throttle` TOCTOU race causes duplicate Telegram sends |
| P2-3 | data-external | `_cached()` None-result not cached — infinite retry on persistent failure |
| P2-4 | orchestration | Auth failure (exit 0) resets stack overflow counter |
| P2-5 | orchestration | Self-heal sessions lack `NODE_OPTIONS` stack-size fix |
| P2-6 | metals-core | `_check_exits` doesn't guard `fill_verified==False` positions |
| P2-7 | metals-core | `pos_id` collision on rapid-fire buys (same-second resolution) |
| P2-8 | avanza-api | `fin_fish.py` hardcodes 21:55 close; violates DST rule |
| P2-9 | avanza-api | `fin_snipe_manager` extracts stop-loss ID with wrong key |
| P2-10 | signals-core | Ticker accuracy cache uses single shared timestamp for all horizons |
| P2-11 | signals-modules | `calendar_seasonal` January double-counted in two sub-signals |
| P2-12 | data-external | `crypto_macro_data.py` reads stale `agent_summary_compact.json` |
| P2-13 | data-external | `http_retry` doesn't distinguish 401/403 from retryable errors |
| P2-14 | portfolio-risk | `_streaming_max` cache rotation detection fails on Windows text-mode |
| P2-15 | portfolio-risk | `cost_model.total_cost_pct()` bps conversion is ambiguous |

---

## Cross-Critique: Agent Findings Rejected

These agent findings were reviewed and determined to be false positives or lower risk
than claimed:

1. **Agent metals-core Finding 3** (direction-blind SHORT stops): Correctly noted that
   `SHORT_ENABLED=False` means this code is dead. Not a production risk until enabled.
   Downgrade from P0 to "future risk noted."

2. **Agent signals-modules Finding 2** (BB breakout "inversion"): Withdrawn by the agent
   itself — the volatility module correctly uses breakout logic, not mean-reversion.

3. **Independent P0-A** (ADX cache keyed by `id(df)`): On closer inspection, the cache
   has `_ADX_CACHE_MAX=200` and is cleared when entries exceed that bound. The bounded size
   plus the fact that DataFrames survive within a single loop cycle (not GC'd mid-cycle)
   makes this low probability. Downgrade to P3.

4. **Independent P1-D** (monotonic timestamps persisted in trigger state): The code
   handles this correctly — duration gate starts fresh on restart, and `_mono_start`
   values are only compared within the same process lifetime. The `_update_sustained`
   correctly uses `time.monotonic()` internally and resets on direction change.

---

## Cross-Critique: Independent Findings Confirmed by Agents

| Independent ID | Agent Confirmation |
|----------------|-------------------|
| P1-A (persistence cold-start) | signals-core Finding 6 (same issue, same fix) |
| P1-N (None-result infinite retry) | data-external Finding 8 (wider: auth errors too) |
| P0-B (pending order race) | avanza-api Finding 1 (deeper: CONFIRM matches wrong order) |
| P1-O (FX fallback to 1.0) | portfolio-risk Finding 3 (same category; agent found peak-cache issue) |

---

## Priority Action Plan

### Immediate (P0 — fix before next deploy)
1. `ic_computation.py:19` — fix relative path (1 line)
2. `signal_engine.py` — initialize `_rescued = False` at loop top (1 line)
3. `equity_curve.py:384` — subtract fees from pnl_sek (1 line)
4. `agent_invocation.py:402` — change except to block on failure (3 lines)
5. `ticker_accuracy.py` — add neutral-outcome filter (5 lines)
6. `signal_history.py` — add threading.Lock (3 lines)

### Short-term (P1 — fix within 1 week)
7. `signal_engine.py:282` — seed persistence with `cycles=1`
8. `volume_flow.py:63` — implement session-anchored VWAP
9. `heikin_ashi.py:317` — use unshifted values for signal
10. `macro_regime.py:203` — lower yield threshold to 0.15
11. `metals_swing_config.py` — raise `HARD_STOP_UNDERLYING_PCT` to 3.0+
12. `metals_swing_trader.py:2823` — fetch live FX rate
13. `avanza_orders.py` — require CONFIRM+UUID
14. `fear_greed.py:100` — guard `body["data"]` access
15. `onchain_data.py:101` — use `_coerce_epoch()`

### Medium-term (P2 — fix within 2 weeks)
16-30. See P2 table above.

---

## Metrics

- **Total unique findings**: 52 (across all reviewers, deduplicated)
- **P0 (critical)**: 6
- **P1 (important)**: 15
- **P2 (degraded)**: 15
- **P3 (code smell)**: 8
- **False positives rejected**: 4
- **Cross-confirmed (independent ∩ agent)**: 4
- **Novel agent findings (not in independent)**: 31
- **Novel independent findings (not in agents)**: 5

---

## Conclusion

The system has significant risk in three areas:

1. **Accuracy data integrity** — Multiple paths compute accuracy differently
   (ticker_accuracy vs accuracy_stats), with different filters and different
   staleness characteristics. This feeds position sizing (Kelly) and signal
   gating (the accuracy gate that determines which signals vote).

2. **Concurrency** — The ThreadPoolExecutor processing 5 tickers creates races
   in signal_history, sentiment state, and any module that does read-modify-write
   without locks. Most modules were fixed (BUG-85, BUG-86) but signal_history was missed.

3. **Order safety** — The TOTP path bypasses the lock architecture, pending order
   confirmation can match the wrong order, and stop-loss thresholds are too tight
   for 5x warrants.

The 6 P0 fixes are all 1-5 line changes. Recommend fixing them immediately.
