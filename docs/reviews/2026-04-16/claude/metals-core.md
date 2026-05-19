# Adversarial Review — metals-core (Claude-independent)

## Summary
Metals-core trades real XAG/XAU warrants via Avanza. 60s main + 10s silver fast-tick. Focus: barrier-proximity, order-API confusion, account ID mixing, market hours, LLM VRAM, races.

## P0 (Catastrophic)

### 1. Barrier-distance calculation for SHORT positions — score-penalty inverted
**File:** `data/metals_swing_trader.py:2040-2055`
SHORT's barrier is a ceiling (knockout if underlying rises above), LONG's a floor. Code computes distance correctly but feeds identical `barrier_dist` to `barrier_score` for both sides. SHORT with numeric 10% safe has ~4x different actual risk. Scoring unfairly penalises SHORT candidates; separately, no directional test `if SHORT and price > barrier * 0.98: reject`.
**Fix:** store direction; apply 1.5× penalty for SHORT in score; explicit reject for SHORT near-ceiling.

### 2. MIN_BARRIER_DISTANCE_PCT lowered to 10% — violates user's 3% grudge envelope
**File:** `data/metals_swing_config.py:58`
Comment: "was 15, but excluded high-lev AVA MINIs". Code permits 10% selection; `_set_stop_loss` at line 2246 places stop at trigger price without re-checking barrier. 0.5% intraday drift → 9.5% → still above 3% floor but dangerously close for volatile silver.
**Fix:** revert to 15, OR add re-check in `_set_stop_loss`: `if barrier_dist < 12.0: reject stop placement`.

### 3. Silver fast-tick starves main loop on entry
**File:** `data/metals_loop.py:943-955`
10s fast-tick alerts fire during the 60s sleep of the main loop. User can manually sell mid-sleep, but SwingTrader state sync doesn't happen until next cycle; `_migrate_orphans` may adopt stale position at stale price.
**Fix:** block new BUYs 30s after a SILVER_FAST_TICK alert; sync swing_state from Avanza positions BEFORE entry check, not just at adoption.

## P1 (High)

### 4. Warrant refresh race — mid-trade reload invalidates ob_id
**File:** `data/metals_warrant_refresh.py:52`, `data/metals_loop.py:6575`
`KNOWN_WARRANT_OB_IDS` loaded at startup; refresh JSON every 6h but swing_trader never reloads. If a MINI is knocked out and replaced, swing_trader posts to dead ob_id → 404.
**Fix:** reload `metals_warrant_catalog.json` at start of each BUY candidate selection.

### 5. Account whitelist bypassed in legacy position adoption
**File:** `data/metals_swing_trader.py:1007-1020`
`ingest_position()` iterates `data["withOrderbook"]` without checking `account_id == "1625505"`. Can adopt pension account (2674244) orphans → tax penalties.
**Fix:** `if str(account_id) != ACCOUNT_ID: return None`.

### 6. LLM concurrent load — no VRAM gate for Ministral
**File:** `data/metals_llm.py:205-249`
Chronos launch gated at `vram.free_mb < 1000`. Ministral (8B, ~4GB) has no gate. Concurrent launch → OOM → zombie processes holding GPU memory → deadlock/stale outputs.
**Fix:** VRAM gate to `_launch_ministral_server`: `if vram.free_mb < 5000: return None`.

### 7. pos_id timestamp collision on rapid BUYs
**File:** `data/metals_swing_trader.py:1020`
`pos_id = f"pos_{int(time.time())}_{ob_id_str}"` — two same-second BUYs on different tickers collide → second overwrites first in `state["positions"]` → overwritten position never exits.
**Fix:** `pos_id = f"pos_{uuid.uuid4().hex[:8]}"`.

### 8. Stop-loss placement without prior cancellation
**File:** `data/metals_swing_trader.py:2265-2275`
Places stop via `place_stop_loss()` without first querying/cancelling existing stops. Stacked stops on same position → double-fill on trigger.
**Fix:** `get_stop_losses_strict()` + `api_delete()` before placing new.

## P2 (Moderate)

### 9. EOD exit disabled — 32.5% accuracy SELL with nothing replacing it
**File:** `data/metals_swing_config.py:113`
`EOD_EXIT_MINUTES_BEFORE = 0`. Replacement (trailing 1.5%, hard stop -2% underlying / -10% warrant) doesn't fire on overnight gaps; 32.5% signal can't be trusted.
**Fix:** restore time-based exit (skip SELL signal check), OR add `MAX_OVERNIGHT_DRAWDOWN = -7%` emergency sell.

### 10. Market hours hardcoded 21:55 CET — DST gap weeks miss close by 45min
**File:** `data/metals_swing_trader.py:1156`
During Mar 8-29 and Oct 25-Nov 1, US DST ≠ EU DST, real close is 21:00 CET. Code thinks 21:55 → EOD at 21:30 → 45 minutes after market closed → orders fail.
**Fix:** fetch `todayClosingTime` from marketplace_info API; never hardcode.

### 11. price_targets 3% TAKE_PROFIT is 0.68 ATR for silver — noisy
**File:** `portfolio/exit_optimizer.py` / `portfolio/price_targets.py`
With XAG ATR ~4.4%, 3% target = 0.68 ATR (breached 40+ times intraday). Fires on noise.
**Fix:** dynamic `take_profit_underlying = 2.0 * current_atr_pct`.

### 12. microstructure_state not persisted atomically
**File:** `portfolio/microstructure_state.py:50-70`
In-memory buffers accumulated, no `atomic_write_json` visible in `record_ofi`/`accumulate_snapshot`. Process crash loses 60-snapshot history; OFI restarts from zero, yields unstable signal on first cycle after restart.
**Fix:** `atomic_write_json` every 30s inside `record_ofi`.

### 13. ORB predictor undefined for 24/7 metals
**File:** `portfolio/orb_predictor.py`
Opening Range is for 09:30 EST market open. Metals trade 24/7 on Binance FAPI. Applying ORB logic creates phantom "breakout" daily.
**Fix:** `if ticker in {"XAG-USD", "XAU-USD"}: skip ORB`.

## P3 (Minor)

### 14. Fishing/snipe min-quantity not verified post-placement
`fin_snipe.py` / `iskbets.py` — Avanza silently rejects <1000 SEK orders; code logs dry-run but never verifies order exists via GET `/_api/trading/rest/orders`.

### 15. Legacy POSITIONS dict no lock — rare TOCTOU
`data/metals_loop.py:6450` + `data/metals_swing_trader.py:1122`. Concurrent mutation can skip updates on emergency_sell iteration.

### 16. metals_llm prediction log append after restart — duplicate inflation
`data/metals_llm.py:583-586` appends without crash-time offset. Accuracy window includes pre-crash predictions whose outcomes never logged.

### 17. Swing trader Kelly uses stale signal data
60s signal age acceptable; during slow LLM threads (Chronos still running from prev cycle) may exceed. Add age check: `if signal_ts < now - 30s: fallback sizing`.

## Looked OK
- **metals_avanza_helpers.py** — CSRF extraction, price fetch, position fetch clean.
- **place_stoploss_once.py** — safety checks at 92-108 solid; uses `/_api/trading/stoploss/new` correctly.
- **metals_risk.py** — MC VaR, trade guards, circuit breaker math sound.
- **metals_signal_tracker.py** — JSONL append-only is race-safe.
- **avanza_session.py (from this lens)** — RLock for concurrent Playwright access is correct; browser dead recovery robust.

## Reviewer confidence
0.82 for P0/P1, lower for P3 (fin_snipe/iskbets not deeply traced)
