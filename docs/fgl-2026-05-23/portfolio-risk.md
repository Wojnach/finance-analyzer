# Adversarial Review — Portfolio / Risk subsystem

Scope: portfolio/{portfolio_mgr,portfolio_validator,warrant_portfolio,risk_management,trade_guards,trade_validation,trade_risk_classifier,kelly_sizing,kelly_metals,monte_carlo,monte_carlo_risk,exit_optimizer,exposure_coach,circuit_breaker,equity_curve,cost_model}.py
Date: 2026-05-23
Reviewer: empty-baseline pass against current main worktree
(`Q:\finance-analyzer-fgl-2026-05-23`).

---

## P0 — Critical (correctness, money-losing)

### P0-1 — `kelly_metals.recommended_metals_size` leverage rescaling is wrong by factor `100/avg_loss_pct`; cap is the only thing keeping it from going all-in every trade
File: `portfolio/kelly_metals.py` lines 207-221

Construction:

```
full_kelly       = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct)   # unitless
half_kelly       = full_kelly / 2.0
cert_loss_frac   = avg_loss * leverage / 100.0          # fractional cert loss per losing trade
position_fraction = half_kelly / cert_loss_frac          # <-- this is the bug
position_fraction = min(position_fraction, MAX_POSITION_FRACTION)  # 0.95 ceiling
```

`kelly_fraction` returns the all-or-nothing Kelly fraction `(p*b - q)/b`
where `b = avg_win_pct / avg_loss_pct`. That is the *fraction of bankroll
to bet* in a binary outcome that pays `b:1` and loses 100% of the stake.
For real per-trade losses (avg_loss% ≪ 100%), the correct trade-percentage
Kelly is `f* = (p*win - q*loss)/(win*loss)`, and the correct leveraged
position fraction is then `f*_underlying / leverage`.

The code instead divides `half_kelly` by `(avg_loss * leverage / 100)`.
That introduces an extra factor of `100/avg_loss_pct`. With the module's
own defaults `_DEFAULT_AVG_WIN = 3.09`, `_DEFAULT_AVG_LOSS = 2.43`,
`_DEFAULT_WIN_RATE = 0.52`, `leverage = 5`:

- `b = 1.272`, `q = 0.48`, `full_kelly ≈ 0.143`, `half_kelly ≈ 0.071`
- `cert_loss_frac = 2.43 * 5 / 100 = 0.1215`
- `position_fraction = 0.071 / 0.1215 ≈ 0.585`

Correct trade-percentage Kelly at `(p=0.52, win=3.09, loss=2.43)` is
`f* ≈ 0.0152`. Levered Kelly is `0.0076`. The code recommends ~58%, the
correct value is ~0.8%. With a slightly stronger edge (win_rate=0.55) the
code instantly saturates at `MAX_POSITION_FRACTION = 0.95` — meaning it
will *always* recommend 95% of buying power on any non-degenerate trade.
The 0.95 cap is the only thing standing between the system and a
fully-leveraged all-in on every metals BUY.

Severity rationale: Kelly is supposed to be the *guardrail* for warrant
sizing in a system the user already calls "5x because 10x has too much
volatility drag". Instead it's been silently telling the bot to bet the
account. Consecutive-loss reduction (`LOSS_REDUCTION_STEP = 0.25`) is
applied multiplicatively after the cap, so it takes 4 *consecutive*
losses before sizing drops to 0 — by which point the account is gone.

Fix:
- Replace the formula with the trade-percentage Kelly:
  `f_underlying = (p*win - q*loss) / (win*loss)`, then divide by
  `leverage` to get the levered position fraction.
- Reconsider whether `kelly_sizing.kelly_fraction` should be reused at
  all for warrants — its all-or-nothing form is wrong here.
- Add a regression test that pins position_fraction at typical inputs
  (win_rate=0.52, avg_win=3.09, avg_loss=2.43, leverage=5) to a value in
  the [0.005, 0.020] band, NOT the current ~0.58.

### P0-2 — Warrant P&L ignores knock-out barrier entirely; `warrant_portfolio.warrant_pnl` cannot model MINI cert death
File: `portfolio/warrant_portfolio.py` lines 52-113

```
underlying_change = (current_underlying_usd - underlying_entry) / underlying_entry
implied_pnl_pct  = underlying_change * leverage
current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
```

MINI BULL/BEAR certificates are *barriered* products: once the
underlying crosses the financing/stop-loss level, the cert is closed at
~0 (a "knockout") and the value never recovers. `warrant_portfolio` has
no `financing_level` / `barrier` / `knockout` fields anywhere — grep
confirms zero matches in this module. The model says a 5x BULL SILVER
cert with underlying down 25% is worth `entry × (1 - 1.25) = -25%` of
entry → it floors `current_implied_sek` at nothing (just multiplies),
producing a NEGATIVE implied value.

Consequences:
- Portfolio reporting will show *negative position value* and a
  symmetric (paper) recovery if XAG rallies back through the barrier,
  which never happens in reality.
- `risk_management.check_drawdown` uses these portfolio values to drive
  the 20% drawdown circuit breaker — a barrier knockout that should
  trip the breaker may instead show a healthy "underlying recovered"
  reading on the next cycle.
- The "stop-loss within 3% of current bid" rule in user memory cannot
  be enforced against a barrier this module doesn't know about.

The richer model in `exit_optimizer.Position` *does* have
`financing_level` and `_compute_pnl_sek` floors warrant value at 0, but
that path is only used by the exit optimizer — not by the portfolio
value path that feeds the drawdown breaker, the Telegram digest, or the
dashboard.

Fix: extend the holding schema with `financing_level_usd` and
`knockout_observed_ts`. In `warrant_pnl`, floor `current_implied_sek` at
0, and if any historical price has crossed the financing level mark the
position dead permanently. Mirror `exit_optimizer._compute_pnl_sek`'s
`max(exit_warrant_sek, 0)` clamp at minimum.

### P0-3 — Drawdown circuit breaker compares current-fx-rate SEK against historical-fx-rate SEK peaks
File: `portfolio/risk_management.py` lines 270-317 (`check_drawdown`) and
561-624 (`log_portfolio_value`)

`log_portfolio_value` writes `patient_value_sek` to
`portfolio_value_history.jsonl` using `_resolve_fx_rate(summary)` — the
fx_rate of the *moment of the write*. `check_drawdown` later compares
*today's* SEK value (computed with today's fx_rate) against the
*historical max* of those SEK entries from `_streaming_max`.

If USD/SEK has moved 8% (well within annual normal — FX_RATE_MIN/MAX
band is 7.0-15.0 i.e. ±20%), an unchanged USD portfolio reads 8%
different in SEK terms. That alone can trigger or hide a 20% drawdown
breach. Worse, the cache returns the peak from a stale-fx era; a SEK
weakening run inflates peaks for the rest of the file's life because
`_streaming_max` is monotonic-up.

Real-world example to construct the bug: BTC = 100K USD throughout.
USD/SEK 11.0 → 9.5. Recorded peak `1_100_000 SEK`; current value
`950_000 SEK`. Drawdown reads `(1.1M - 0.95M)/1.1M = 13.6%` from peak —
all of which is FX, none is portfolio P&L.

Fix options:
1. Record USD values alongside SEK in `portfolio_value_history.jsonl`,
   compute drawdown on USD. (Best — matches what the bot can actually
   control.)
2. Or, store the fx_rate per row and recompute historical SEK using
   today's fx on the fly in `_streaming_max`.
3. Or, drawdown vs. `initial_value_sek` only (already done as fallback),
   and emit two breakers — one in SEK, one in USD — and trip if either
   crosses.

### P0-4 — Stop-loss is static at entry, not trailing — module docstring claims trailing
File: `portfolio/risk_management.py` lines 1-9, 320-395 (`compute_stop_levels`)

Module docstring says: "ATR-based **trailing** stop-loss tracking".
`compute_stop_levels` computes `stop_price = entry_price * (1 - 2 *
atr_pct / 100)` — purely from entry. No ratchet, no peak tracking, no
"once price moved up X%, raise the stop". `Position.trailing_peak_usd`
is defined in `exit_optimizer.Position` and never read/written.

For a position that doubled on the underlying, the stop sits at `entry
- 2 × ATR`, well below current price — meaning the bot has no
mechanism to lock in unrealized gains, and `check_atr_stop_proximity`
flags will look healthy right up until a 50% retracement triggers them.

This is *not* a bug in computation, it's a missing feature falsely
advertised. Either build the trailing component or correct the
docstring, because the surrounding code (and other modules: see
`fin_snipe`, `iskbets`) reasons about "stop-loss" as if it could trail.

Fix: track `pos.peak_underlying_usd` updated on every cycle a position
is held; recompute `stop_price = max(pos.peak_underlying_usd * (1 - 2 *
atr_pct/100), entry_price * (1 - 2 * atr_pct/100))`.

---

## P1 — Important (sizing/state correctness)

### P1-1 — `update_state` lock does not span the `_load_state_from → mutate → save` cycle when callers reach for `load_state()` / `save_state()` directly
File: `portfolio/portfolio_mgr.py` lines 116-159

`update_state` correctly holds `_get_lock(path)` for the whole
read-modify-write. But `load_state()` and `save_state()` are still
public and are called *separately* in many places (grep confirms 30+
call sites). Any code path that does
`state = load_state(); ...mutate...; save_state(state)` re-introduces
the C8 race the lock was meant to fix. Two threads racing on
patient-state can each:

1. T1 `load_state()` → cash=500K
2. T2 `load_state()` → cash=500K
3. T1 mutates: cash -= 100K BUY; `save_state(state)` → 400K
4. T2 mutates: cash -= 100K BUY; `save_state(state)` → 400K (loses
   T1's BUY entirely)

The new `_state_locks` are only acquired inside
`save_state → _save_state_to`. So the per-file lock serializes
*writes*, not the *read-modify-write*, when the caller doesn't go
through `update_state`.

Fix: deprecate `save_state(state)` for any caller that mutated; have a
hooks-style audit that flags `save_state` not preceded by
`update_state`. Alternatively, document the contract in the module
docstring and convert all external mutators to `update_state`.

Verification suggestion: `grep -rn "save_state\|save_bold_state"
portfolio/` and check each site does load-from-disk in the same
critical section.

### P1-2 — `warrant_portfolio.record_warrant_transaction` is NOT atomic and does NOT acquire any lock; concurrent BUY+SELL from metals loop + manual snipe will lose updates
File: `portfolio/warrant_portfolio.py` lines 182-266

`record_warrant_transaction` does:
```
state = load_warrant_state()
state["transactions"].append(txn)
... mutates state["holdings"] ...
save_warrant_state(state)  # atomic write of full state
```

`save_warrant_state` is atomic *at the file-system level*
(`atomic_write_json`), but there is no in-process lock around the
`load → mutate → save`. The metals loop (`metals_loop.py`) and
`fin_snipe.py` both record warrant transactions, and the metals fast
tick path runs at 10s cadence. Two near-simultaneous BUYs:

1. T1 loads → transactions list has N items
2. T2 loads → transactions list has N items
3. T1 appends BUY-A → N+1, writes
4. T2 appends BUY-B → N+1, writes (overwrites; BUY-A is silently lost)

This is exactly the C8 bug class that `portfolio_mgr.update_state`
fixes. `warrant_portfolio` has no equivalent — and warrant trades are
the only path that touches the real Avanza account.

Fix: mirror `portfolio_mgr._get_lock(path)` and provide
`update_warrant_state(mutate_fn)` for all callers; add a sidecar lock
file for cross-process safety (the metals loop and the dashboard live
in different processes).

### P1-3 — Kelly trade-stats wins/losses skew by counting `pnl_pct == 0` as a loss
File: `portfolio/kelly_sizing.py` lines 109-110

```
wins = [p for p in pnl_list if p > 0]
losses = [abs(p) for p in pnl_list if p <= 0]
```

A trade that exited exactly at break-even (`p == 0`) is counted as a
loss with `abs(loss) = 0`. Then `avg_loss = sum(losses) / len(losses)`
is dragged toward 0 by zeros, inflating `b = avg_win / avg_loss` and
thus inflating Kelly. With a single break-even and one real loss,
avg_loss is halved → b doubles → Kelly nearly doubles.

`trade_guards.record_trade` line 281 uses the consistent (and correct
for that context) `pnl_pct < 0` to count consecutive losses; the
mismatch in `kelly_sizing` is the bug. Note also `equity_curve`'s
`_pair_round_trips` uses `pnl_pct <= 0` for `losses`, same problem in
the trade metrics path (`profit_factor`, `expectancy_pct`,
`max_consecutive_losses`).

Fix: in `kelly_sizing._compute_trade_stats`, `losses = [abs(p) for p in
pnl_list if p < 0]`; treat `p == 0` as neither (or as a tiny win)
consistently across kelly_sizing and equity_curve.

### P1-4 — `t_dist.cdf(T_samples, df) → norm.ppf(U)` round-trips lose tail dependence at the very horizon you want to measure tails
File: `portfolio/monte_carlo_risk.py` lines 257-273

The C9 fix comment correctly identifies the issue with the *previous*
identity-transform, but the *current* implementation has a subtler
defect: by going `t-samples → t.cdf → uniform → norm.ppf → standard
normal marginals`, the dependence structure copied across is the
t-copula's *rank* correlation, but the *marginal* tails are Gaussian.
That's a valid t-copula + Gaussian-marginal construction — but it
means:

- VaR/CVaR at 99% is *understated* compared to a true Student-t
  marginal at the same df.
- The "tail dependence λ ~ 0.18 at df=4" claim in the module docstring
  applies to the COPULA, not to the marginals. The pair joint
  probability of simultaneous large losses IS captured; the per-asset
  tail risk is NOT.

For the 5-instrument portfolio at df=4, the standard normal marginals
have kurtosis 3 vs. t(4)'s kurtosis ∞ (well, technically undefined,
but ~9 at finite samples). 99% one-asset losses can be 30-50%
larger under t marginals.

This isn't strictly wrong — both choices appear in the literature — but
the docstring + comments oversell what the module captures. Decide
whether the spec wants t-copula+normal-marginal (current) or
t-copula+t-marginal (one extra `t.ppf(U, df)` call instead of
`norm.ppf`). Document and pin in tests.

### P1-5 — `check_atr_stop_proximity` uses `atr_pct` capped *implicitly* differently in `compute_stop_levels` (cap=15%) vs. proximity check (uncapped)
File: `portfolio/risk_management.py` line 373 vs. 897

`compute_stop_levels`:
```
atr_pct = min(atr_pct, 15.0)
stop_price = entry_price * (1 - 2 * atr_pct / 100)
```

`check_atr_stop_proximity`:
```
stop_price = entry_price * (1 - 2 * atr_pct / 100)   # uncapped
```

For an instrument with 18% ATR (silver during a fast move can hit
this), `compute_stop_levels` says stop is `entry * 0.70`, but
`check_atr_stop_proximity` thinks the stop is `entry * 0.64`. The two
disagree on *where the stop is*, and the proximity check will say
"safe, far from stop" while the actual stop (from `compute_stop_levels`,
which is what's surfaced to operators) is much closer.

This is dangerous because it's the exact path that issues
`severity: "warning"` flags during the high-vol regime where the user
most needs accurate stop-distance info.

Fix: extract one helper `_compute_atr_stop(entry, atr_pct)` that does
the cap and is used by all three sites (`compute_stop_levels`,
`compute_probabilistic_stops`, `check_atr_stop_proximity`).

### P1-6 — `check_drawdown` peak-cache returns stale peaks after fx-rate band rejection rewrites history; cache is not invalidated by `log_portfolio_value` overrides
File: `portfolio/risk_management.py` lines 28-110

`_peak_cache` is keyed by `(history_path, value_key)` and is *process-
scoped* but never invalidated except on file shrink. If the loop
process restarts after a rolled-back fx_rate change (operator
manually edits a row to fix a bad fx), the peak from the bad-fx era
sits in cache forever in the running process.

Less hypothetical: if `_resolve_fx_rate` is rejected by the sanity
band (e.g., agent_summary briefly had `fx_rate: 1.0` from a stale
write) and the disk cache also has 1.0 from before the bounds were
added, the entry written to history is bogus. The cache will record
the bogus peak and serve it for the life of the process.

Fix: stamp `_peak_cache` entries with the fx_rate they were derived
from; invalidate on FX_RATE_MIN/MAX changes; or simpler, expire the
cache every N minutes regardless.

### P1-7 — `trade_guards.check_overtrading_guards` cooldown can be bypassed by trading the SAME ticker under a different strategy
File: `portfolio/trade_guards.py` lines 137-167

Key is `key = f"{strategy}:{ticker}"`. Patient and Bold each get their
own cooldown for the same ticker. So if Patient just bought BTC-USD and
30 minutes hasn't passed, Bold can still buy BTC-USD freely. CLAUDE.md
treats them as two independent portfolios so this is "by design", BUT:
the position rate limit (Guard 3) is also per-strategy. Concurrent
Patient+Bold trades on the same trigger event = 2 positions on the
same instrument with no coordinated cap. Combined with no global
"max correlated exposure" check, a single high-conviction signal can
trigger 4 buys in 4 hours across both strategies.

Fix: add a portfolio-wide cooldown layer that triggers when *any*
strategy has touched the ticker recently (e.g., 5-10 min hard floor),
OR add a global rate limit `cfg.global_position_window_h`.

### P1-8 — `portfolio_value` (mgr) does not respect the same fx_rate sanity band as `risk_management`; legacy fx=1.0 still produces bad valuations through this path
File: `portfolio/portfolio_mgr.py` lines 162-180

```
def portfolio_value(state, prices_usd, fx_rate):
    if not isinstance(fx_rate, (int, float)) or not math.isfinite(fx_rate) or fx_rate <= 0:
        ...
    total = state.get("cash_sek", 0)
    for ticker, h in state.get("holdings", {}).items():
        ...
        total += shares * price * fx_rate
```

The guard rejects 0/negative/non-finite, but accepts `fx_rate=1.0`,
which (as P1-15 in the project's own history reminds us) understates
SEK valuations 10x. The risk_management module has `_resolve_fx_rate`
with FX_RATE_MIN/MAX band; the mgr module does not. Anything that
calls `portfolio_value()` directly (the dashboard's `/api/portfolio`
endpoint, for instance) skips the sanity band.

Fix: route both modules through the same `fx_rates.resolve()` helper,
or duplicate the band check here. Either way: hardcode FX_RATE_MIN=7.0
as the floor in `portfolio_mgr.portfolio_value`.

---

## P2 — Notable (quality, edge cases)

### P2-1 — `_compute_pnl_sek` in exit_optimizer does not subtract a buy-side cost for the original entry, only the exit cost
`portfolio/exit_optimizer.py` line 338: `cost = costs.total_cost_sek(exit_value)`. The round-trip cost is `2 × total_cost_pct()`, but here only one side is charged. Result: EV ranking will systematically over-estimate P&L by one half-spread + one slippage on every candidate, biasing toward limit orders that would otherwise be marginal. Either subtract entry cost at position creation (preferred — already paid) or charge a 2x cost here as a worst-case (acceptable).

### P2-2 — `_compute_trade_stats` (kelly_sizing) computes weighted average buy price using `total_sek` divided by shares; this implicitly INCLUDES the buy-side fee in the buy price, then compares against `sell_total_sek / sell_shares` which has fee DEDUCTED (it's net proceeds per the portfolio_validator contract)
`portfolio/kelly_sizing.py` lines 91-103. Net effect: per-trade P&L is understated by 2x the per-share fee. Across many small trades this compounds and drags `avg_win_pct` down / `avg_loss_pct` up, lowering Kelly. Cross-check with `equity_curve._pair_round_trips` which correctly stores `price_per_share = total_sek / shares` for BOTH legs but explicitly subtracts `buy_fee_share + sell_fee_share` from `pnl_sek` afterwards. Kelly does not. Inconsistent between modules.

### P2-3 — `_estimate_volatility` and `volatility_from_atr` both have a hard 5% annualized floor; for stocks 5% annual vol is below the 1st percentile (typical 15-30%) but for some currencies/dollar-pegged stables would be way too high
`portfolio/exit_optimizer.py` line 174 and `portfolio/monte_carlo.py` line 32. This is an asymmetric floor — for crypto/metals it's fine, for stable-leg pairs it inflates VaR/CVaR. Low priority — the system doesn't currently trade stable-leg things — but worth being explicit about the assumption.

### P2-4 — `concentration_pct` in `check_concentration_risk` only checks the BUY allocation as % of total but never CALL it at SELL time — selling INTO a concentrated position (i.e., selling other things) increases the held % and would be missed
`portfolio/risk_management.py` lines 742-788. Edge case but real: a portfolio with BTC at 35% and a SELL on MSTR raises BTC to ~42% and would not be flagged because the SELL flow returns `None` immediately at line 742.

### P2-5 — Monte Carlo seed handling in `simulate_all` increments by `i` for independence, but `_get_directional_probability` is sensitive to whether `extra._weighted_action` is missing
`portfolio/monte_carlo.py` lines 339-364. If `extra` is missing or has `_weighted_action: None`, it falls through to `ticker_data.get("action", "HOLD")` — but the conf will then come from `ticker_data.get("weighted_confidence", 0.5)` which may not match the action. Result: drift sign and confidence get desynchronized. Add an explicit `if action == "HOLD" or conf == 0.5: return 0.5` guard at the top.

### P2-6 — `equity_curve.compute_metrics` computes the variance twice (once for `daily_vol`, once for `daily_std_dec`) using two different scales (% vs. decimal); these should be equal up to scaling but the second uses `mean_dec` from a fresh sum, not the already-computed `mean_ret/100`. Minor precision / consistency issue.
`portfolio/equity_curve.py` lines 230-247. Both compute the same std but the second pass is unnecessary now that BUG-225 extracted the mean.

### P2-7 — `_resolve_fx_rate` writes to disk on every successful resolution; high-frequency callers (Layer 1 60s cycle × 8 workers) cause unnecessary disk writes and an `atomic_write_json` per call
`portfolio/risk_management.py` lines 153-162. Throttle to ~1/min or only on change. Minor IO; not load-bearing.

### P2-8 — `trade_validation.validate_trade` price-deviation check uses `last_known_price` blindly without recency information — a 5%-deviation gate against a 3-day-old price is meaningless
`portfolio/trade_validation.py` lines 95-104. Add a `last_known_age_sec` parameter and bypass the check if age > 60s; the current behavior either gates valid trades during normal moves OR lets through bad trades on stale references.

### P2-9 — `CircuitBreaker.recovery_timeout` mutates an instance attribute that's also read outside the lock for status reporting (`get_status`). Race on Windows where Python ints over 256 are reboxed
`portfolio/circuit_breaker.py` lines 65-66 (write), 110-116 (`get_status` does hold the lock, OK), but several callers read `cb.state` (line 39-41, a property) which holds the lock — also OK. Looks fine on closer inspection. Withdrawing this concern.

### P2-10 — `exposure_coach` regime multiplier table has typo / inconsistency: "range-bound" key but `regime_mismatch` checks use "ranging"
`portfolio/exposure_coach.py` line 30 has `"range-bound": 0.9` while `risk_management.check_regime_mismatch` line 804 uses `"ranging"`, and `_REGIME_SCORES` in `trade_risk_classifier.py` line 22 uses `"ranging"`. Whichever the signal engine actually emits, one of these tables is dead code. Audit one canonical regime vocabulary.

---

## P3 — Minor / Style

- `portfolio_mgr.py` line 19: `INITIAL_CASH_SEK = 500_000` is in two places (also `risk_management.INITIAL_VALUE_DEFAULT`); extract to a single constants module.
- `monte_carlo.py` line 31: `DEFAULT_HORIZONS = [1, 3]` but the docstring says "3h, 1d, 3d horizons" — 3h is not in the default.
- `kelly_metals.py` line 65-95 (`_get_outcome_stats`): bare `except Exception:` swallows sqlite errors silently. Log at WARNING with reason.
- `trade_guards.py` line 96: `base >> halvings` on a Python int is safe but unidiomatic; `base // (2**halvings)` is clearer and matches the docstring "halve repeatedly".
- `risk_management.py` line 548: `first_buy_ts.replace(tzinfo=datetime.UTC)` mutates a *naive* datetime by hand; consistent with the codebase but `datetime.UTC` is preferred over `datetime.timezone.utc` aliasing — pick one.
- `equity_curve.py` line 495: `losses = [t for t in trips if t["pnl_pct"] <= 0]` — same `<= 0` vs `< 0` inconsistency as P1-3.
- `monte_carlo_risk.py` line 408: `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` accepts any number including 1.0 — should use `_resolve_fx_rate` from risk_management for consistency with the other portfolio paths.
- `warrant_portfolio.py` line 196: `from datetime import datetime` is a re-import inside the function — top-level already imports `datetime` from line 8 (different style); pick one place.
- `cost_model.py` `total_cost_sek` does not distinguish maker vs. taker fees, market vs. limit slippage. The exit_optimizer assigns all candidates the same cost, which underestimates limit-order economics (they pay maker fees, no slippage on the way in). Could justify a `for_action: str` parameter.

---

## Stop-loss API compliance

The CLAUDE.md rule "stop-loss MUST use `/_api/trading/stoploss/new`" lives
in `portfolio/avanza_session.py` / `avanza_orders.py` — none of the
portfolio/risk modules I reviewed place orders. They compute stop
*prices* and emit *flags*. The execution path is out of scope for this
file but the rule is implicit when `risk_management.compute_stop_levels`
output is consumed by snipe / metals_loop code. Confirmed clean: no
direct stop-loss order placement in this subsystem.

## Atomic state I/O compliance

| File | Atomic? | Locked? | Notes |
| --- | --- | --- | --- |
| `portfolio_state.json` | yes (via `_atomic_write_json`) | yes (`_state_locks`) when via `update_state` | direct save_state callers bypass lock — P1-1 |
| `portfolio_state_bold.json` | yes | yes (same as above) | same caveat |
| `portfolio_state_warrants.json` | yes (write) | **NO** (no lock, no `update_state` equivalent) | **P1-2 race** |
| `portfolio_value_history.jsonl` | yes (`atomic_append_jsonl` with sidecar lock) | yes (sidecar) | clean |
| `trade_guard_state.json` | yes | yes (`_state_lock`) | clean |
| `fx_rate_cache.json` | yes (`atomic_write_json`) | no | benign, single writer |

---

## Summary (5 lines)

1. P0-1: kelly_metals is off by ~50x and recommends ~95% of buying power on every trade — only the hard cap prevents account-level catastrophe.
2. P0-2: warrant_portfolio has no concept of MINI knockout barriers; portfolio P&L can go negative on paper, breaking the drawdown breaker.
3. P0-3: drawdown breaker compares current-fx SEK to historical-fx SEK peaks — an 8% USD/SEK move alone can trip or hide the 20% gate.
4. P0-4: `compute_stop_levels` claims trailing but is static-from-entry; `Position.trailing_peak_usd` exists but is never used.
5. P1-2 + P1-1: warrant state mutations are unlocked across processes; non-update_state callers on patient/bold portfolios skip the read-modify-write lock and can lose appends.
