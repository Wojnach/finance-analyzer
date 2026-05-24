# Adversarial Review — Portfolio Risk Subsystem

**Date:** 2026-05-24
**Branch:** `review/fgl-2026-05-24` (worktree: `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24`)
**Scope:** `portfolio_mgr.py`, `portfolio_validator.py`, `warrant_portfolio.py`,
`risk_management.py`, `monte_carlo.py`, `monte_carlo_risk.py`,
`equity_curve.py`, `kelly_sizing.py`, `kelly_metals.py`, `trade_guards.py`,
`trade_validation.py`, `trade_risk_classifier.py`, `circuit_breaker.py`,
`cost_model.py`, `exit_optimizer.py`, `price_targets.py`,
`cumulative_tracker.py`, `decision_outcome_tracker.py`, `outcome_tracker.py`.

**Empty-baseline approach.** Every file treated as new. Three prior P0s from
2026-05-19 re-checked and confirmed still present.

---

## TOP 5

1. **`warrant_portfolio.py` ignores BULL/BEAR direction in P&L math.**
   `warrant_pnl()` line 96 hard-codes `implied_pnl_pct = underlying_change * leverage`,
   correct only for LONG. The catalog (`data/metals_warrant_catalog.json`) contains
   **41 SHORT warrants alongside 58 LONG**. Any held SHORT cert reports inverted P&L,
   feeds inverted P&L into Telegram, the dashboard, the journal, and into
   `risk_management.compute_stop_levels` → stops trip backwards. Confirms 2026-05-19 P0 #3.

2. **`risk_management.compute_stop_levels()` and `compute_probabilistic_stops()` assume LONG only.**
   `risk_management.py:374` computes `stop_price = entry * (1 - 2*atr_pct/100)` and
   line 382 checks `triggered = current_price < stop_price`. For a SHORT warrant or
   BEAR cert, the stop should sit ABOVE entry and trigger on `current > stop`. The
   Monte Carlo stop in `compute_probabilistic_stops()` uses `direction="below"` at
   line 484 — same LONG-only blind spot. Wrong P&L = wrong stop = forced exit on a
   profitable BEAR move.

3. **`portfolio_mgr.save_state` / `save_bold_state` use threading.Lock only — no cross-process protection.**
   `portfolio_mgr.py:108-113` and the `update_state` mutator wrapper at line 136 hold
   a `threading.Lock` keyed by `str(path)`. Layer 2 subprocess, main loop, dashboard,
   metals_loop, and Avanza CLI tools all write to `data/portfolio_state.json` and
   `portfolio_state_bold.json` from **separate processes** — the lock is invisible
   between them. `atomic_write_json` (file_utils.py:53) is per-write atomic but two
   concurrent read-modify-write cycles still last-writer-wins, dropping transactions.
   Confirms 2026-05-19 P0 #1. The repo already has a working cross-process pattern
   in `file_utils.jsonl_sidecar_lock` (line 210) — `save_state`/`update_state` need
   to adopt it.

4. **`warrant_portfolio.record_warrant_transaction` has zero concurrency protection.**
   Line 198-265: pure `load_warrant_state()` → mutate dict → `save_warrant_state()`.
   No threading lock, no cross-process lock. metals_loop and grid_fisher both write
   here from separate processes. A concurrent BUY/SELL race silently drops one
   transaction and corrupts both `holdings[key]["units"]` and the avg-cost ledger.
   Confirms 2026-05-19 P0 #2.

5. **`kelly_metals.recommended_metals_size` can recommend 95 % of buying power on a 5x warrant.**
   Line 215-221: `position_fraction = half_kelly / cert_loss_frac` where
   `cert_loss_frac = avg_loss_underlying% × leverage / 100`. With avg_loss=2.43 %
   (XAG default), leverage=5, half_kelly=0.05 → position_fraction = 0.05 / 0.1215 = 0.41
   (41 % of cash on one 5x leveraged trade). The cap `MAX_POSITION_FRACTION = 0.95`
   (line 45) lets a high-edge regime push this to 95 % of buying power on a single
   knock-out-able leveraged silver bet. Kelly's whole point is to bound position by
   edge × loss; dividing by the leveraged loss inverts that bound — higher leverage
   yields larger size for the same edge. The formula needs a sanity ceiling of
   `<= 0.5 / leverage` or equivalent, not 0.95.

---

## Critical (severity 90-100)

### P0-A `warrant_portfolio.py:42-49` | concurrency | **lost writes on warrant book**

`save_warrant_state(state)` is a one-line `atomic_write_json` call with no lock of
any kind. `record_warrant_transaction()` (line 198-265) calls
`load_warrant_state()` → mutates `holdings` and `transactions` → calls
`save_warrant_state()`. metals_loop, grid_fisher, fin_snipe, iskbets, and
Layer 2 all run in different processes and all call this function. Two
concurrent BUYs silently drop one. Two concurrent SELLs silently drop one
or leave orphan units in `holdings`.

**Fix.** Wrap the read-modify-write under
`portfolio.file_utils.jsonl_sidecar_lock(WARRANT_STATE_FILE)` (msvcrt/fcntl
backed, already used in 20+ writers). Add an `update_warrant_state(mutate_fn)`
helper that mirrors `portfolio_mgr.update_state()`. Every existing caller of
`save_warrant_state` must migrate.

### P0-B `portfolio_mgr.py:35-159` | concurrency | **cross-process race on Patient/Bold state**

`_state_locks` is a process-local `dict[str, threading.Lock]`. Layer 2 is a
separate Python process spawned by `agent_invocation.py`; the dashboard is
another process; the main loop is a third. Each holds its own dict.
`update_state()` reads, mutates, and writes under a lock that only blocks
threads within its OWN process. Mixed-process collisions:

* Layer 2 commits a BUY → dashboard's `/api/validate-portfolio` POST reads
  the stale Patient state during Layer 2's write window → false reconciliation
  failure surfaced to the user.
* Two Layer 2 invocations spawned 1 second apart (T1 retry on T2 timeout)
  both load → both mutate cash_sek → both write → second write loses first
  trade. The transactions list looks correct (both BUY records present) but
  `cash_sek` is computed from one and stamped over the other, breaking the
  validator's cash-reconciliation check.

**Fix.** Replace `_get_lock` with `file_utils.jsonl_sidecar_lock` for `STATE_FILE`,
`BOLD_STATE_FILE`. Hold the sidecar lock across the entire read-mutate-rotate-write
cycle inside `update_state` and `_save_state_to`.

### P0-C `risk_management.py:374, 382, 484, 897` + `warrant_portfolio.py:92-101` | correctness | **stops & P&L hard-coded LONG**

The catalog has 41 SHORT certs / 58 LONG. Code paths:

* `risk_management.compute_stop_levels`: `stop = entry * (1 - 2*atr/100)`,
  `triggered = current < stop`.
* `risk_management.compute_probabilistic_stops`: simulates one-sided stop
  with `direction="below"`.
* `risk_management.check_atr_stop_proximity`: `stop = entry * (1 - 2*atr/100)`,
  distance `(current - stop)`.
* `warrant_portfolio.warrant_pnl`: `implied_pnl_pct = underlying_change * leverage`.
* `exit_optimizer._compute_pnl_sek` (line 327-332): `warrant_move = pct_move *
  leverage`, `exit_warrant_sek = entry * (1 + warrant_move)`.
* `exit_optimizer._first_hit_times` only flips on string compare to "above"/"below";
  the caller in `compute_exit_plan` always passes `"above"` for sell targets.
* `monte_carlo.simulate_ticker`: `stop_price = price * (1 - 2 * atr_pct / 100)`,
  `p_stop_hit = P(terminal < stop_price)`.

Net effect on a BEAR cert: a profitable underlying down-move shows as a
loss, the stop "triggers" on green, and the agent sells into strength.

**Fix.** Add `direction` to the holding schema (LONG=+1, SHORT=-1). Plumb it
through every P&L and stop-level computation:

```python
sign = +1 if direction == "LONG" else -1
implied_pnl_pct = underlying_change * leverage * sign
stop_price = entry * (1 - 2 * atr_pct / 100 * sign)
triggered = (current - stop_price) * sign < 0
```

For `exit_optimizer.simulate_intraday_paths` callers, generate session_max for
LONG sells, session_min for SHORT sells. For the probabilistic stop, pass
`direction="above"` for SHORT.

### P0-D `kelly_metals.py:215-221, 45` | correctness | **leveraged Kelly inverts safety bound**

`position_fraction = half_kelly / (avg_loss * leverage / 100)`. As leverage
increases, position_fraction GROWS. The conventional Kelly for a leveraged
bet is `f* = edge / variance_of_leveraged_return`, where variance scales
with leverage² — so size should SHRINK with leverage, not grow. The code's
"convert underlying-Kelly to cert-Kelly by dividing by leveraged loss" is
backwards. Combined with `MAX_POSITION_FRACTION = 0.95`, the system can
recommend nearly all of buying power on one 5x silver warrant. `memory/grudges.md`
is explicit about silver's volatility regret cost.

**Fix.** Replace with `position_fraction = half_kelly` (Kelly already
operates on the leveraged return distribution if you feed it
`avg_win_underlying% × leverage` and `avg_loss_underlying% × leverage`).
Cap at `MAX_POSITION_FRACTION = 0.30 / leverage` (per-trade 6 % at 5x).
Wire the consecutive-loss reducer in BEFORE the cap, not after, so high-edge
streaks can't undo the loss-streak protection.

### P0-E `kelly_sizing.py:84-104` | correctness | **avg buy price mixes pre- and post-sell BUYs**

`_compute_trade_stats` collects ALL BUY transactions for a ticker into one
weighted average, then evaluates EVERY SELL against that average. If you
buy 100 @ $50, sell 100 @ $60, buy 100 @ $70 — avg = ($5000+$7000)/200 = $60,
and the historical $60 sell looks like 0 % P&L when it was actually a clean
+20 % win. Kelly then sees zero edge and refuses to size up.

Also line 110: `losses = [abs(p) for p in pnl_list if p <= 0]` buckets
break-even sells (p == 0) as losses, biasing win_rate down.

**Fix.** Use the FIFO matcher already in `equity_curve._pair_round_trips`
(line 314-426). It's correct and well-tested. Replace the in-house averaging
with a call into that function, then compute `pnl_pct` per round-trip. Move
`p == 0` cases into a separate `breakevens` bucket so they don't pollute
win_rate.

### P0-F `monte_carlo_risk.py:204, 228` | correctness | **trading_days hardcoded 365 for stocks**

`PortfolioRiskSimulator._trading_days = 365` for ALL positions, including
MSTR which `monte_carlo.trading_days_for_ticker()` correctly returns 252
for. `volatility_from_atr` uses td=252 for MSTR (line 428 in
compute_portfolio_var), but `simulate_correlated_returns` then scales time
`T = horizon_days / 365`, mismatching the vol's annualization base. For a
1-day horizon, the sim uses sigma*sqrt(1/365) instead of sigma*sqrt(1/252),
systematically undersizing 1-day VaR for stocks by ~20%.

**Fix.** Make `_trading_days` per-asset, or rescale T per-ticker inside
the loop at line 267. Simpler: convert all vols to a single calendar-day
basis (multiply by sqrt(252/365) for stocks) before passing to the simulator.

### P0-G `monte_carlo_risk.py:408` + `exit_optimizer.py:718` | correctness | **fx_rate bypasses cached fallback chain**

Both bypass `risk_management._resolve_fx_rate` and use raw
`agent_summary.get("fx_rate", FX_RATE_FALLBACK)` / `get("fx_rate", 10.85)`.
The P1-15 fix in risk_management.py (line 121-186) was added because
`agent_summary` can carry `fx_rate=1.0` during early-cycle / rotation
windows, producing 10x-off SEK valuations and a false 95 % drawdown. These
two callers reintroduce the same bug for VaR reporting and exit P&L.

**Fix.** Move `_resolve_fx_rate` to `fx_rates.py` (no risk_management
dependency), import it in monte_carlo_risk.py and exit_optimizer.py.

### P0-H `price_targets.py:391, 397, 417, 419, 429, 431` | correctness | **warrant SEK gain formula is wrong by ~27x**

```python
gain_if_filled = (target_price - price_usd) * position_units * warrant_leverage * fx_rate
```

This treats `position_units` (warrant cert units) as if they tracked
underlying USD-per-unit. For a 5x silver mini at SEK 12 entry, 100 units,
underlying $30, target $30.30 (+1 %):

* True warrant SEK gain ≈ 100 units × (12 SEK × 0.01 × 5) = 60 SEK.
* Formula yields (0.30 × 100 × 5 × 10.85) = 1627.5 SEK — **27× overstated**.

EV-ranked targets are then misranked (higher absolute gains rank wrong
relative to fill_prob). The recommended target picks the wrong side of
the EV frontier.

**Fix.** Compute warrant SEK gain via the same formula `exit_optimizer._compute_pnl_sek`
already has:

```python
pct_move = (target - price_usd) / price_usd
warrant_move = pct_move * warrant_leverage * direction_sign
gain_per_unit_sek = entry_price_sek * warrant_move
gain_sek = max(gain_per_unit_sek * position_units, -entry_price_sek * position_units)
```

This requires plumbing `entry_price_sek` and `direction` through the
`compute_targets` signature.

---

## Important (severity 80-89)

### P1-A `risk_management.py:217-269` | risk-management | drawdown circuit breaker can be optimistic for hours

When `agent_summary` is empty (rotation window, fetch failure), the breaker
falls back to "cash only" value while warning. Has been on the books since
2026-04-17. The warning hits Telegram but Layer 2 still gets a non-breached
verdict. Suggestion: when feed is stale AND positions exist, return
`{"breached": False, "stale": True}` and gate all BUY decisions on `not stale`.

### P1-B `equity_curve.py:494-495` | correctness | round-trip break-even bucketed as loss

`losses = [t for t in trips if t["pnl_pct"] <= 0]`. A SELL at exactly the
buy price (rare but happens for limit-order fills) is bucketed as a loss,
underestimating win_rate and inflating loss_count. Bug-for-bug match with
`kelly_sizing.py:110`. Use `< 0` for losses, separate `== 0` bucket for breakevens.

### P1-C `trade_guards.py:32, 126` | concurrency | `_state_lock` is per-process only

`trade_guard_state.json` is written by `_save_state` from main loop, Layer 2
subprocess, AND the bigbet/iskbets subprocesses. Threading lock doesn't span
those — same class as P0-B. Cooldowns can be reset/ignored on race. Replace
with `jsonl_sidecar_lock(STATE_FILE)`.

### P1-D `exit_optimizer.py:54, 718` | risk-management | hardcoded fx default 10.85

`MarketSnapshot.usdsek: float = 10.85`. The default ships into every test
and every callsite that forgets to override. CET FX has been in the 9.6-11.5
band in 2026; embedding a default at one end of the range silently biases
P&L. Either make it required (no default) or pull from `_resolve_fx_rate`
on construction.

### P1-E `monte_carlo_risk.py:225-281` | correctness | drift time-scaling unverified for stocks

The drift_term `(mu - 0.5*sigma^2)*T` and vol_term `sigma*sqrt(T)` both use
the same `T = horizon/_trading_days`. For stocks this T is wrong (see P0-F),
so both drift AND vol scaling are biased. Fix in lockstep with P0-F.

### P1-F `price_targets.py:104-107` | correctness | mirrored-buy fill prob has algebraic ambiguity

`fill_probability_buy` calls `fill_probability(price, price**2/target, vol,
-drift, ...)`. The reflection trick is valid for symmetric Brownian motion
without drift, but drift breaks it: the reflected process has drift -mu, not
the SDE's true geometry. For meaningful drift (|p_up - 0.5| > 0.1) the
buy-side fill probability is biased. Use a direct first-passage formula
for the running-min (negate the log returns, run the same formula).

### P1-G `cost_model.py:64-69` | risk-management | WARRANT_COSTS spread underestimates illiquid silver MINIs

40 bps half-spread + 10 bps slippage = 0.50 % one-way. Real Avanza silver MINI
spreads observed in `data/metals_warrant_catalog.json` (`spread_pct`) are
typically 0.04-0.20 % full-spread for liquid certs but 1.5-3 % for the
illiquid high-leverage ones. The flat 40 bps half-spread will over-estimate
cost on liquid certs and under-estimate on illiquid. Make spread dynamic
(read `spread_pct` from the warrant catalog).

### P1-H `kelly_sizing.py:321-323` | risk-management | recommended_sek doesn't subtract round-trip cost

`rec_sek = min(half_kelly * cash_sek * exposure_ceiling, max_alloc)` doesn't
deduct estimated round-trip cost. For Avanza stocks (courtage 6.9 bps + 5 bps
spread) the cost is negligible, but for warrants (40 bps half-spread × 2) a
half-Kelly position with a 2% edge sinks 0.8 % into costs. After fee, the
realized Kelly should be `(p*b' - q)/b'` where `b' = (avg_win - fee) /
(avg_loss + fee)`. Currently fees never enter the sizing decision.

### P1-I `risk_management.py:284-286` | correctness | new peak set from optimistic-stale current_value

After the cash-only fallback (line 270), `peak_value` is updated from
`current_value` at line 285. A subsequent valid feed read could legitimately
exceed the cash-only fallback but be below the true peak — the breaker
records a downward-revised peak permanently because `_streaming_max` reads
disk history, not memory. Mitigation: don't update peak from current_value
when `summary` was empty.

### P1-J `monte_carlo.py:111-180` | correctness | tail-quantile precision overstated

Antithetic on `Z` for terminal price is correct for the mean estimator, but
quantiles (which `price_quantiles` returns) are NOT variance-reduced by
antithetic variates. Tail percentiles (5, 95) are noisy. For p_stop_hit
estimates near the tails the standard error is ~1/sqrt(n_paths/2) at the
true rate. Either bump to 50K paths for tail probabilities or use a
deterministic Latin-hypercube sample for the tails. Not a correctness bug,
but the 0.001 rounding in `result["p_stop_hit_{h}"]` advertises false
precision.

### P1-K `trade_guards.py:73-100` | correctness | bit-shift halving truncates non-power-of-2 multipliers

`base = max(1, base >> halvings)`. 8x → 4 → 2 → 1 after 24h, 48h, 72h is
fine. But 2x after one halving = 2 >> 1 = 1, which is base case, not the
intended "halved" 1.5x or float-decay. Use float arithmetic: `base = max(1.0,
base / (2 ** halvings))`. Current code is correct only for power-of-2
multipliers — works today because LOSS_ESCALATION maxes at 8, but fragile
when the table is tuned.

### P1-L `equity_curve.py:404-406` | correctness | partial-sell fee allocation correct, but rounding-loss not tracked

The proportional fee split uses `matched / original_shares` for buy fee
and `matched / sell_shares` for sell fee. Correct. But `pnl_sek` is rounded
to 2 decimals before summing into `total_pnl_sek`. Over 1000 trades the
cumulative rounding error can exceed 100 SEK. Sum the unrounded values
inside `compute_trade_metrics`, round at the END.

### P1-M `portfolio_validator.py:144-145` | correctness | "no transactions" branch warns on manually-seeded holdings

`shares > 0 and ticker not in all_tx_tickers` flags an error, but during
the FIRST loop tick after a manually-seeded holding (used for live recovery
or staging an Avanza-only position) the transactions list is empty. The
validator surfaces "Holdings contains X with N shares but no matching
transactions" — a misleading error. Consider a `"manual_seed": true` flag
on the holding or a `seed_transactions` list separate from `transactions`.

### P1-O `kelly_metals.py:170-180` | correctness | DB win_rate uses underlying %, not leveraged warrant %

The query selects `ts.consensus` (BUY/SELL) and checks if the price moved
the right direction over `horizon`. But the consensus is an action taken at
ts; the realized warrant return is `change_pct * leverage`, not `change_pct`.
For Kelly avg_win / avg_loss to match the variance of the actual leveraged
trade, multiply by leverage AT THIS STEP, not at the position_fraction
conversion step. The current code computes Kelly on underlying %, then
attempts a leverage adjustment via the broken division at P0-D.

---

## P2 / P3 (severity 60-79)

### P2-A `outcome_tracker.py:380-403` | concurrency | head-streaming relies on byte-equality across rewrite

The signal_log rewrite copies `head_end_offset` bytes verbatim from the
ORIGINAL file inside the lock (line 533). If a concurrent appender extends
the file between phase 1 and phase 3, those bytes land past `snapshot_size`
and the copy at line 549 handles them. Correct, but the `head_count`
boundary was computed under phase-1 lock — if an external tool rotates the
log between phase 1 and phase 3, the head bytes copied from the new file
are meaningless. Mitigation: re-open and verify file size at line 533 still
== `snapshot_size`.

### P2-B `monte_carlo_risk.py:251-258` | correctness | t-copula → Gaussian marginal transformation correct

The C9 fix comment (line 260) explains the bug correctly. The fix is
correct. Suggestion: add a unit test that confirms terminal returns'
empirical std matches `sigma * sqrt(T)` to within 2%.

### P2-C `cost_model.py:43` | correctness | total_cost_pct() excludes min_fee floor

`courtage = max(trade_value_sek * courtage_bps / 10_000, min_fee_sek)`. The
min fee is over courtage only, not over total cost. For Avanza stocks at
0.069 % with 1 SEK min — a 100 SEK trade has courtage = max(0.0069, 1) = 1
SEK total = 1 % cost. The `total_cost_pct()` method (line 49) gets a value
that excludes min_fee, hiding the 1-SEK floor on small orders. Add
`total_cost_pct(trade_value=...)` that uses actual total over the trade value.

### P2-D `risk_management.py:756-787` | correctness | concentration uses total_value but cash-capped allocation

Verified path: if total_value >> cash (fully invested), proposed_alloc = cash,
new_position_value = existing_value + cash, concentration_pct computed
against total_value. Logic is correct. False alarm.

### P2-E `equity_curve.py:534-560` | correctness | Calmar excludes mark-to-market drawdown while holding

`compute_trade_metrics` builds a mini equity curve from initial_value +
cumulative round-trip PnL. This excludes mark-to-market drawdown WHILE
holding (which can be larger than the realized round-trip drawdown).
Calmar therefore understates risk for buy-and-hold legs. Acceptable for
the trade-metrics view; document the limitation.

### P3-A `trade_validation.py:30, 31, 33` | correctness | hardcoded defaults too tight for warrants

`max_spread_pct=2.0`, `max_cash_pct=50.0`, `min_order_sek=1000.0`. Reasonable
for stocks; warrants routinely violate `max_spread_pct=2.0` (silver MINIs
hit 2-3 %). Plumb instrument-type into the signature.

### P3-B `cumulative_tracker.py:67-79` | correctness | last-snapshot tail read uses 2KB window

A pathological multi-ticker snapshot can exceed 2KB (60+ tickers × 30 bytes).
Then `lines[-1]` is mid-line junk and `json.loads` returns the cached
fallback. Bump to 8KB or read backwards in chunks until a newline boundary.

### P3-C `cumulative_tracker.py:217` | correctness | cache key joins tickers without sorting

`",".join(tickers)` is order-sensitive; identical sets in different orders
miss the cache. Use `",".join(sorted(tickers))`.

### P3-D `price_targets.py:125` | correctness | hardcoded seed=42 in running_extremes

Every call to `running_extremes` uses the same seed. Path distribution is
deterministic. Good for unit tests, bad for Monte Carlo — the same MC error
is baked into every cycle's recommendation. Either accept a seed parameter
or use `np.random.default_rng()` without seed.

### P3-E `decision_outcome_tracker.py:75` | correctness | bare-except on _fetch_historical_price

Silently swallows network/rate-limit errors. The outcome is dropped, the
decision never gets backfilled, accuracy stats stay biased toward the
samples that DID succeed. Log at WARNING, increment a fail counter visible
on the health endpoint.

### P3-F `outcome_tracker.py:464` | correctness | bare-except on price-fetch in backfill loop

Same pattern. Same fix.

### P3-G `kelly_sizing.py:296-310` | correctness | ATR-based default 1.5:1 R:R biases Kelly up

The 1.5:1 reward:risk assumes positive selection skill. Without trade
history this is a free assumption that biases Kelly upward. Use 1:1 until
empirical data exists.

---

## Cross-cutting concerns

* **Direction (LONG/SHORT) is not a first-class field in any risk module.**
  Adding it to `Position`, `holdings[ticker]` dict, and the warrant catalog
  helper is a 1-week refactor that touches ~12 files. Highest-impact
  single fix in this review.
* **threading.Lock vs cross-process locks.** Three separate state files
  (`portfolio_state.json`, `portfolio_state_bold.json`,
  `portfolio_state_warrants.json`, `trade_guard_state.json`) use threading
  locks. All four are written by multiple processes. Adopt
  `jsonl_sidecar_lock` system-wide.
* **fx_rate resolution.** The robust `_resolve_fx_rate` lives in
  risk_management.py and is bypassed in two critical callsites
  (monte_carlo_risk, exit_optimizer). Promote to fx_rates.py as the canonical
  source.

---

## Verified non-issues

* `monte_carlo_risk._nearest_psd` (line 88-106) — correct Higham-style
  projection with diagonal-1 rescale.
* `circuit_breaker.py` — single-process is fine here, the breaker tracks
  per-process API health. `reset()` resets `_last_failure_time` correctly
  (line 132).
* `portfolio_validator.py:69-90` — cash reconciliation invariant is
  correct (BUY = full alloc with fee, SELL = net proceeds).
* `outcome_tracker.backfill_outcomes` (line 342-575) — the sidecar-lock
  three-phase rewrite is sound; verbatim byte-copy preserves concurrent
  appends.
* `equity_curve._pair_round_trips` — FIFO matching is correct, fee
  allocation uses original_shares/sell_shares correctly (PR-P0-6).
* `trade_validation.py` — straightforward validation; defaults could be
  per-instrument (P3-A) but logic is sound.

---

## Suggested fix order

1. **P0-C direction plumbing** (correctness ripple — affects 6 modules).
2. **P0-A warrant write lock** (1-hour fix, lifesaver).
3. **P0-B portfolio_mgr cross-process lock** (mirrors P0-A approach).
4. **P0-H price_targets gain formula** (currently overstating warrant EV by 25×).
5. **P0-D kelly_metals leverage division** (currently sizes UP with leverage).
6. **P0-E kelly_sizing round-trip pairing** (low-frequency bug, but biases the only data feeding Kelly).
7. **P0-G fx_rate fallback in mc_risk + exit_optimizer.**
8. **P0-F mc_risk trading_days per-asset.**

Items 1-3 are pure safety. Items 4-8 are correctness — the system is
running today with these biases and the Layer 2 agent is sizing trades
on the wrong numbers.

— end of report —
