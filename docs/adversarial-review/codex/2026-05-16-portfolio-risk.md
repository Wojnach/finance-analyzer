## P1 Corrupt portfolio state becomes fake cash
**File:** portfolio/portfolio_mgr.py:105
**Bug:** If the state file exists but is corrupt and all backups fail, the loader returns a fresh default portfolio.
**Why it matters:** A real account with open holdings can be treated as `500_000` SEK cash and no positions, causing duplicate buys or missed exits.
**Fix:** Fail closed on existing corrupt state; require manual recovery. Only create defaults for explicit first-run initialization.

## P1 Portfolio updates are not process-safe
**File:** portfolio/portfolio_mgr.py:29
**Bug:** State locks are `threading.Lock`, so `update_state()` is protected only inside one Python process.
**Why it matters:** Layer 1 and a Claude subprocess can both read the same cash/holdings, write different mutations, and lose one transaction despite atomic writes.
**Fix:** Use an OS file lock around the full load-mutate-backup-write sequence.

## P1 Trade guard check/record race
**File:** portfolio/trade_guards.py:126
**Bug:** `check_overtrading_guards()` reads state and releases the lock before the trade is placed; `record_trade()` runs later.
**Why it matters:** Two workers can both pass the same cooldown/rate-limit check and place duplicate BUYs before either records.
**Fix:** Add an atomic “check and reserve” operation persisted before order placement.

## P1 Cooldown can block stop-loss exits
**File:** portfolio/trade_guards.py:147
**Bug:** Per-ticker cooldown blocks any action, including `SELL`.
**Why it matters:** A BUY followed by a stop-loss SELL within 30 minutes can be blocked, forcing the system to hold a losing position.
**Fix:** Never apply re-entry cooldowns to risk-reducing sells or liquidation orders.

## P1 NaN trades validate as safe
**File:** portfolio/trade_validation.py:52
**Bug:** The validator checks `<= 0` but never checks `math.isfinite()`.
**Why it matters:** `price=float("nan")` makes all comparisons false, reaches `ValidationResult(True)`, and can crash or poison order placement.
**Fix:** Reject non-finite `price`, `volume`, `cash_available`, bid/ask, last price, and derived `order_value`.

## P1 Cash check ignores fees
**File:** portfolio/trade_validation.py:69
**Bug:** BUY sufficiency compares only `price * volume` against cash.
**Why it matters:** An order that exactly consumes cash can still require broker fees/courtage, causing rejection or negative internal cash.
**Fix:** Include estimated fees and a slippage buffer in required cash.

## P1 Peak-cache lock does not protect the scan
**File:** portfolio/risk_management.py:70
**Bug:** `_peak_cache_lock` is released before scanning the file and reacquired only for the final write.
**Why it matters:** Concurrent drawdown checks can return or write an older/lower peak after another thread saw a newer high, weakening the circuit breaker.
**Fix:** Hold the per-key lock across stat, scan, and cache update, or use versioned compare-and-swap.

## P1 Raw JSONL read can hide peak values forever
**File:** portfolio/risk_management.py:85
**Bug:** `_streaming_max()` uses raw `open()`/`json.loads()`, skips decode errors, then stores EOF offset.
**Why it matters:** A partial/corrupt peak line is skipped once and then never reread, so drawdown is measured from a too-low peak.
**Fix:** Use a locked JSONL reader and never advance cached offset past an undecodable trailing line.

## P1 Missing quote is valued at entry cost
**File:** portfolio/risk_management.py:211
**Bug:** `_compute_portfolio_value()` falls back to `avg_cost_usd` when live price is missing.
**Why it matters:** A holding bought at 100k and now worth 70k is still valued at 100k if the quote disappears, hiding drawdown.
**Fix:** Use last fresh marked price with TTL or fail closed/block trading when valuation is stale.

## P1 Stop-level calculation crashes on zero price
**File:** portfolio/risk_management.py:378
**Bug:** `current_price` is used as a divisor before positive/finite validation.
**Why it matters:** `price_usd=0` or `None` from one stale signal can crash risk reporting instead of flagging unavailable stop data.
**Fix:** Validate finite positive `current_price` and `atr_pct` before computing stop distance.

## P1 VaR drops stale positions
**File:** portfolio/monte_carlo_risk.py:432
**Bug:** Positions with missing or zero live price are skipped.
**Why it matters:** If all held quotes are stale, VaR returns `n_positions=0`, zero exposure, and zero drawdown probability.
**Fix:** Use validated last-known prices with staleness flags or fail closed when held positions cannot be priced.

## P1 VaR SEK conversion trusts invalid FX
**File:** portfolio/monte_carlo_risk.py:419
**Bug:** `fx_rate` is read directly from `agent_summary` without finite/range validation.
**Why it matters:** `fx_rate=1.0` underreports SEK exposure by about 10x; `None` crashes result construction.
**Fix:** Reuse the validated FX fallback/cache logic used by drawdown valuation.

## P2 Stop-hit probability is terminal-only
**File:** portfolio/monte_carlo.py:328
**Bug:** `p_stop_hit_*` calls `probability_below()`, which checks only terminal price below the stop.
**Why it matters:** Paths that breach the stop intraday and recover are counted as safe, understating stop risk.
**Fix:** Simulate pathwise barrier hits or rename the metric to terminal-below probability.

## P2 Held HOLD positions are omitted from MC risk
**File:** portfolio/monte_carlo.py:408
**Bug:** `_interesting_tickers()` claims held+focus+signaling, but code includes only focus tickers and active BUY/SELL signals.
**Why it matters:** A real held position with a HOLD signal gets no Monte Carlo risk unless it happens to be a focus ticker.
**Fix:** Include current portfolio holdings in ticker selection.

## P2 Realized P&L double-counts fees
**File:** portfolio/equity_curve.py:405
**Bug:** P&L subtracts buy/sell fees even though transaction `total_sek` already represents fee-inclusive buy cash outflow and net sell proceeds.
**Why it matters:** A trade with actual net +100 SEK and 38 SEK total fees is reported as +62 SEK.
**Fix:** Compute from gross prices plus fees once, or from net cash flows without subtracting fees again.

## P2 Net losing trades can count as wins
**File:** portfolio/equity_curve.py:494
**Bug:** Win/loss stats use `pnl_pct` instead of net `pnl_sek`.
**Why it matters:** A +0.1% price move with 0.3% fees is counted as a win, inflating win rate, streaks, and expectancy.
**Fix:** Classify performance metrics from net realized P&L.

## P2 Kelly stats use future buys and ignore fees
**File:** portfolio/kelly_sizing.py:95
**Bug:** Every sell is compared to a lifetime weighted average of all buys for that ticker.
**Why it matters:** Buy 100@100, sell 100@110, later buy 100@200 makes the first sell look like a loss versus avg 150.
**Fix:** Reuse FIFO round-trip matching with net fees.

## P2 Leveraged metals sizing falls back silently on DB failure
**File:** portfolio/kelly_metals.py:94
**Bug:** `_get_outcome_stats()` catches all exceptions and returns `None` without logging.
**Why it matters:** A locked or broken `signal_log.db` silently falls back to hard-coded 52% win rate and still sizes a leveraged trade.
**Fix:** Log the failure and fail closed or cap size to zero when outcome stats are unavailable due to error.

## P2 Validator accepts missing positive holdings
**File:** portfolio/portfolio_validator.py:130
**Bug:** If a holding is removed, a share mismatch under 1% of total bought is accepted.
**Why it matters:** A 0.99% remainder of a large position can be worth thousands of SEK while validation passes.
**Fix:** Use an absolute dust threshold in SEK/share precision; do not accept positive missing holdings by percentage.

## P2 Invalid risk inputs can score LOW
**File:** portfolio/trade_risk_classifier.py:69
**Bug:** Numeric inputs are not checked for finite/range validity, and unknown regimes default to zero risk.
**Why it matters:** `NaN` position/confidence/consensus values bypass comparisons; bad upstream data can produce LOW risk.
**Fix:** Validate finite ranges and known regimes; treat invalid or unknown inputs as HIGH/block.

## SUMMARY
P1=12 P2=9 P3=0