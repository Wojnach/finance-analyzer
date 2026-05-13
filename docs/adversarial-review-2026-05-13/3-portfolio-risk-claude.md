# Claude adversarial review: portfolio-risk

## Summary

Scope: 18 files (portfolio_mgr, portfolio_validator, trade_guards, trade_validation,
trade_risk_classifier, risk_management, monte_carlo, monte_carlo_risk, equity_curve,
exit_optimizer, kelly_sizing, kelly_metals, exposure_coach, warrant_portfolio,
cost_model, instrument_profile, stats, iskbets) plus
`portfolio/strategies/{base,orchestrator,elongir_strategy,golddigger_strategy}.py`.
Total ~6,871 LOC.

Headline findings:

1. **fx_rate fallback regression** persists in `monte_carlo_risk.compute_portfolio_var`,
   `exit_optimizer.compute_exit_plan_from_summary`, and three call sites in
   `iskbets.py`. The P1-15 hardening in `risk_management._resolve_fx_rate` (sanity
   band 7-15 SEK, on-disk cache, hardcoded fallback) is **not** reused by these
   downstream consumers — they all still do `agent_summary.get("fx_rate", 1.0|10.85|10.5)`.
   A stale `fx_rate=1.0` in `agent_summary.json` silently understates SEK VaR / exit
   P&L by ~10×.
2. **MINI warrant model is BULL-only and ignores knock-out**: `warrant_portfolio.warrant_pnl`
   treats every warrant as a pure `leverage × underlying_change` multiplier with no
   financing-level enforcement, no direction (BULL vs BEAR), and no leverage drift over
   time. `exit_optimizer._compute_pnl_sek` and `_apply_risk_overrides` similarly assume
   BULL (long financing < spot). Selling a held BEAR cert will compute negative
   "warrant_value" without flipping the formula, producing nonsense P&L.
3. **Currency mixing in iskbets P&L attribution**: `format_exit_alert` /
   `format_position_status` recompute `shares = amount_sek / (entry_price * fx_rate)`
   using the **current** (caller-supplied) fx_rate as if it were the entry fx — direct
   contradiction of the entry fx stored on the position. Round-trip SEK P&L is wrong
   any time SEK/USD has moved.
4. **Risk classifier silent fallback to 0**: `trade_risk_classifier._REGIME_SCORES.get(regime, 0)`
   maps every unknown regime (missing field, typo, new value) to LOW risk, masking
   up to 3 score points. Combined with the `confidence < 0.50` line that will
   `TypeError` if confidence is None, the classifier fails open in two distinct ways.
5. **Kelly sizing computes "win/loss" gross of fees** (`kelly_sizing._compute_trade_stats`,
   line 103) and **ignores knock-out tail risk for MINI warrants** in
   `kelly_metals.recommended_metals_size` — the formula `cert_loss_frac = avg_loss * leverage / 100`
   assumes losses are bounded at `avg_loss * leverage %`, but MINIs can lose 100%
   in a single barrier touch.
6. **Concentration check is portfolio-proportional but the `proposed_alloc` cap on
   `cash` lets a near-zero-cash portfolio bypass the 40% threshold**, because
   `concentration_pct = (existing + min(total_value*frac, cash)) / total_value` collapses
   to `existing/total_value` when `cash≈0`. A fully-invested patient portfolio with one
   45% position will not flag concentration on a BUY for the same ticker.
7. **Warrant portfolio state file has no lock, no backup rotation, and isn't
   covered by `validate_portfolio_file`** (the validator only checks patient + bold).
   Corruption is undetected and unrecoverable.

The trade_validation min-order floor (1000 SEK) and the kelly_sizing 1000 SEK
rejection are now in place — those P0s from prior reviews are fixed.

---

## P0 — Blockers

### P0-1 — monte_carlo_risk.py:419 — raw `agent_summary.get("fx_rate", FX_RATE_FALLBACK)` bypasses sanity band, lets `fx_rate=1.0` corrupt SEK VaR
`Q:\finance-analyzer\portfolio\monte_carlo_risk.py:419`
```python
fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)
```
The defended P1-15 path (`risk_management._resolve_fx_rate`) explicitly rejects any
rate outside `[FX_RATE_MIN=7, FX_RATE_MAX=15]` and falls back to a cached or
hard-coded 10.50. Here the value is taken at face. Any stale agent_summary that
still has `"fx_rate": 1.0` (the legacy default in many code paths) produces
`var_95_sek = var_95_usd × 1.0` — the SEK breach number is 10× smaller than reality
and the dashboard / Layer 2 see "fine" when the real loss is 10× that. Drawdown
circuit breaker reads from `_compute_portfolio_value` (which IS fixed); VaR reads
from here, so the two disagree.
Fix: import and use `risk_management._resolve_fx_rate(agent_summary)` (or hoist it to
`fx_rates.resolve_safe_rate`).

### P0-2 — exit_optimizer.py:719 + iskbets.py:743,798,875 — same raw `.get("fx_rate", 10.85|10.5)` anti-pattern
`Q:\finance-analyzer\portfolio\exit_optimizer.py:719`
```python
fx_rate = agent_summary.get("fx_rate", 10.85)
```
`Q:\finance-analyzer\portfolio\iskbets.py:743,798,875` — three more sites with
fallback `10.5` (off by 0.50 from the canonical `FX_RATE_FALLBACK=10.50` — actually
10.5 ≠ 10.50, but consistent at 10.50; still: not sanity-banded). Any caller that
forgets to populate fx_rate or hands in a corrupt agent_summary gets:
- Wrong exit P&L SEK (exit_optimizer)
- Wrong P&L SEK in entry alert / exit alert / status (iskbets) — all of these are
  the *only* P&L the user sees on Telegram for ISKBETS trades.

Fix: shared resolver. Reject `fx_rate ≤ 0` and `> 15` everywhere.

### P0-3 — warrant_portfolio.warrant_pnl ignores financing level and direction
`Q:\finance-analyzer\portfolio\warrant_portfolio.py:80-113`
```python
underlying_change = (current_underlying_usd - underlying_entry) / underlying_entry
implied_pnl_pct = underlying_change * leverage
current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
```
Problems:
1. **No barrier check.** For MINI BULL silver with financing $20, if underlying
   drops below $20 (knockout), `current_implied_sek` would go negative if leverage
   * underlying_change < -1. Code does not clamp. Test: leverage=5, change=-25%
   → implied_pnl_pct = -125% → current_implied_sek negative. `total_value_sek` is
   then negative — portfolio reporting shows a non-zero negative position value
   instead of the real "knocked out, position is zero" state.
2. **No direction.** BEAR certs invert the formula (`pnl_pct = -underlying_change *
   leverage`). The function doesn't check holding["direction"]. Holding a BEAR
   cert with underlying rising → reported as gain instead of loss.
3. **Pure-multiplier leverage drift unmodeled.** MINIs daily-rebalance, so 5×
   over 1 day ≠ 5× over 30 days (compounding). The `leverage` field is treated
   as constant. Over a multi-day hold, computed value drifts from actual.

`exit_optimizer._compute_pnl_sek` lines 322-323 at least clamps to 0
(`max(exit_warrant_sek, 0)`), but is itself BULL-only.

Fix: pass direction; clamp at 0 on knockout; consider that for valuation purposes
the MINI loses all value once `(price - financing_level) * direction_sign ≤ 0`;
re-fetch live `entry_price_sek` from Avanza periodically for long holds.

### P0-4 — exit_optimizer._apply_risk_overrides only handles BULL knock-out
`Q:\finance-analyzer\portfolio\exit_optimizer.py:432-436`
```python
distance_pct = (market.price - position.financing_level) / market.price * 100
if distance_pct < 3:
    return market_exit
```
For a BEAR MINI position, `financing_level > market.price` always (financing is
above spot); `distance_pct` is **always negative**, so the override always fires
and force-market-exits every BEAR position on every cycle. Conversely, the
BULL-formula riske flags (`_compute_risk_flags` lines 372-378) are also
BULL-only — for a BEAR position approaching its (upper) barrier, no warning ever
fires.
Fix: branch on `position.direction` (BULL vs BEAR) and use
`abs(market.price - financing_level) / market.price` with direction-aware sign.

### P0-5 — trade_risk_classifier silent fallback to LOW on unknown regime + None confidence crash
`Q:\finance-analyzer\portfolio\trade_risk_classifier.py:81`
```python
regime_score = _REGIME_SCORES.get(regime_lower, 0)
```
Unknown regime (missing, typo, new entry like "transitional") → 0 score = LOW. A
proposed trade in an unrecognized regime appears safe regardless of conditions.
Combined with `_REGIME_SCORES["trending-up"] = 0` this is exactly the "no regime
→ 0 risk" pattern called out.
And at line 101:
```python
if confidence < 0.50:
```
If a caller passes `confidence=None` (which `agent_summary.get("weighted_confidence")`
will return when missing), TypeError, classification aborts. No try/except wrapper.
Fix: raise ValueError on unknown regime (or treat as MEDIUM); explicit None checks
on confidence/consensus_ratio/position_pct/existing_exposure_pct.

### P0-6 — iskbets.format_exit_alert / format_position_status — current fx_rate masquerades as entry fx for share count
`Q:\finance-analyzer\portfolio\iskbets.py:512-513`
```python
shares = amount_sek / (entry_price * fx_rate)
pnl_sek = shares * (price - entry_price) * fx_rate
```
The caller (`check_iskbets` line 639) passes the loop's **current** `fx_rate`, not
the entry fx that's stored on the position (`pos["fx_rate"]`). When SEK/USD has
moved since entry — which it always has on a multi-hour ISKBETS hold — the
recomputed `shares` is wrong, and the reported `pnl_sek` is internally inconsistent
(it should be `shares_at_entry × (current_price − entry_price) × fx_at_exit`, but
ends up as `(amount_sek / (entry_price × fx_now)) × (price − entry_price) × fx_now`
which is equivalent to `amount_sek × ((price/entry_price) − 1)` — i.e., SEK gain
ignoring fx entirely). Telegram exit alerts and the status command show
fx-collapsed P&L.
`format_position_status` lines 581-583: same bug.
Same defect path in `_handle_sold` line 810-811 uses `pos.get("fx_rate", 10.5)` —
that one IS the entry fx, so handle_sold is correct in isolation but the user-facing
alerts called from the loop are not.
Fix: store `shares` on position at entry time (`_handle_bought` already does this
at line 766) and use `pos["shares"]` and `pos["fx_rate"]` everywhere; don't
recompute.

---

## P1 — High

### P1-1 — warrant_portfolio has no backup rotation, no lock, and isn't validated
`Q:\finance-analyzer\portfolio\warrant_portfolio.py:42-49` (`save_warrant_state`):
direct call to `atomic_write_json` with no `_rotate_backups`, no per-file lock
(unlike `portfolio_mgr._save_state_to`). A second writer racing on the same file
last-writer-wins.
`Q:\finance-analyzer\portfolio\portfolio_validator.py:281-285` (`validate_all`):
hardcoded patient + bold paths only; warrants portfolio is never validated. Schema
drift in `portfolio_state_warrants.json` is invisible. Fix: hoist
`_save_state_to`/`_get_lock`/`_rotate_backups` to a shared helper in
`portfolio_mgr` and import from `warrant_portfolio`; add a third call to
`validate_portfolio_file` in `validate_all`.

### P1-2 — risk_management.check_concentration_risk collapses to existing/total when cash≈0
`Q:\finance-analyzer\portfolio\risk_management.py:770`
```python
proposed_alloc = min(total_value * alloc_pct, cash)
```
When `cash → 0`, `proposed_alloc → 0`, so `concentration_pct ≈ existing_value / total_value`.
A fully-invested portfolio that already holds 40% in BTC will NOT flag concentration
on a (currently-impossible-to-fund) BUY for BTC — the check passes because the
*proposed* add is zero, but if the BUY then succeeds (e.g., from a SELL of
something else freeing cash mid-cycle), the gate has approved it. The check
needs to consider the *intended* alloc, not just the cash-constrained alloc.

### P1-3 — monte_carlo_risk uses Gaussian copula remap that destroys tail dependence on small samples
`Q:\finance-analyzer\portfolio\monte_carlo_risk.py:266-289`. The C9 fix correctly
applies `t_dist.cdf → norm.ppf` to extract a Gaussian marginal from the t-copula.
However the dependence structure is still the **Gaussian-correlated** `W = Z @ L.T`
with t-scaled marginals — which is what *Gaussian* copula does, NOT t-copula. A
true t-copula uses `T = mvt_rng × scale` and PIT through `t_dist.cdf` with **the same df**
for marginals (or normal marginals, but the joint must be multivariate t, not
Gaussian × t-scaling). At df=4 with this construction, the tail dependence λ
ends up near zero (vs the 0.18 claimed in the module docstring). The CVaR_99 is
biased OPTIMISTIC by 20-30% in stress scenarios — exactly when you need it most.
Fix: sample `(W, S) → T = W * sqrt(df/S)`, run `t_dist.cdf(T, df)` (keeping df for
both copula and marginal), then `norm.ppf(U)` for normal marginals OR `t.ppf(U, df)`
for t marginals. The current code only does the first half then immediately discards
the t structure.

### P1-4 — kelly_sizing._compute_trade_stats is gross of fees
`Q:\finance-analyzer\portfolio\kelly_sizing.py:95-103`
```python
avg_buy_price = total_cost / total_shares_bought
sell_price_per_share = sell_total / sell_shares
pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
```
`total_cost` here is `sum(tx.total_sek)` over BUYs — for BUY transactions in this
codebase, `total_sek` is the **allocation including fee** (per
`portfolio_validator` line 70: "BUY total_sek = full allocation (including fee)").
Sell `total_sek` is **net of fee**. So `avg_buy_price` is buy-price-plus-fee per
share, and `sell_price_per_share` is sell-price-minus-fee per share. The fee
asymmetry partially cancels (favoring conservativism), but ATR-based fallback at
line 308-309 is straight gross. The downstream Kelly fraction over-allocates by
~2-4× courtage_bps fraction. Worse, classifying `pnl_pct <= 0` as a loss at
line 110 sweeps break-evens into losses, deflating avg_loss with zero-magnitude
entries → unreliable b ratio.

Fix: use `equity_curve._pair_round_trips` (which DOES net buy + sell fees) for
the Kelly inputs.

### P1-5 — kelly_metals doesn't model knock-out tail; max loss assumed bounded at `avg_loss × leverage`
`Q:\finance-analyzer\portfolio\kelly_metals.py:215-217`
```python
cert_loss_frac = avg_loss * leverage / 100.0
if cert_loss_frac > 0:
    position_fraction = half_kelly / cert_loss_frac
```
For MINI 5× certs, the empirical avg_loss on underlying is ~2.4% (per
`_DEFAULT_AVG_LOSS["XAG-USD"]`) → cert_loss_frac = 12%. But a 20% adverse
underlying move → 100% knock-out, not 100% cert_loss. Kelly assumes bounded
loss-per-trial; with knock-out, the distribution has a left-tail mass that
classic Kelly ignores. Risk of ruin is materially higher than the formula reports.
Fix: cap `cert_loss_frac = min(avg_loss * leverage / 100, knockout_probability * 1.0 + (1-p_ko) * avg_loss_cert)`, or use the safer formulation in
`docs/quant_research_priorities.md` (drawdown-bounded Kelly).

Also: line 235 `units = int(position_sek / ask_price_sek)` floors; line 236-237
only zeros when `units <= 0`. If `ask=99 SEK` and `position_sek=1000`, `units=10`,
`actual_sek = 990 SEK < MIN_TRADE_SEK = 1000`. Round-down silently pushes the
order below Avanza minimum. Fix: post-check `units * ask_price_sek ≥ MIN_TRADE_SEK`.

### P1-6 — exit_optimizer assumes financing_level is below spot (BULL-only) for risk flags
`Q:\finance-analyzer\portfolio\exit_optimizer.py:373-378` (matches P0-4 but at a
weaker severity layer — flagging vs override). Same symptom: for BEAR positions
the distance_pct calc produces a negative number always; `KNOCKOUT_DANGER` flag
is hard-coded "on" for every BEAR cert. The probability-based knockout at line
396-400 has the same bug: `session_min <= position.financing_level * 1.03` — for
BEAR the threat is `session_max >= financing_level * 0.97`.

### P1-7 — exposure_coach defaults to "healthy" zone on missing data, allows entries in danger+ranging
`Q:\finance-analyzer\portfolio\exposure_coach.py:71`
```python
zone = market_health.get("zone", "healthy")
```
Permissive default — if the market_health dict exists but is malformed, exposure
ceiling = 1.0 and entries allowed. Should default conservatively (e.g., "caution").

Lines 88-89:
```python
new_entries = not (zone == "danger" and regime in ("trending-down", "high-vol"))
```
A `zone="danger"` + `regime="ranging"` portfolio gets new entries allowed. The
zone name is "danger" — entries shouldn't be allowed just because the regime
isn't downtrending. Fix: any danger zone blocks new entries.

### P1-8 — risk_management.check_regime_mismatch silently accepts BUY in trending-down when volume data missing
`Q:\finance-analyzer\portfolio\risk_management.py:818-820`
```python
if volume_ratio is not None and volume_ratio < 1.5:
    mismatch = True
```
If `volume_ratio` is None (signal didn't compute, or pre-market for stocks) →
mismatch = False; no flag fired. So a BUY signal in trending-down regime with
missing volume data slides past the regime audit. The "unknown is fine" assumption
is the opposite of what a risk audit should do. Fix: treat missing volume as a
mismatch (or at minimum a separate `regime_data_missing` flag).

### P1-9 — equity_curve._pair_round_trips uses input-list pre-grouping, ordering relies on input being chronological
`Q:\finance-analyzer\portfolio\equity_curve.py:339-353`. FIFO matching depends on
the `transactions` list arriving in chronological order. The Sort happens nowhere
in this module. If `portfolio_state_*.json` ever has out-of-order transactions
(possible after manual edits, post-recovery from backup, or rare cross-thread
write races) the FIFO pairing is incorrect → wrong `pnl_pct`, wrong `hold_hours`,
wrong Calmar.

Fix: `sorted(transactions, key=lambda t: t.get("timestamp", ""))` at the top of
`_pair_round_trips`.

### P1-10 — Sharpe ratio computes daily_vol twice with two formulas; sample-vs-population mix
`Q:\finance-analyzer\portfolio\equity_curve.py:230-247`. Two std calculations:
- line 230: `variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)`
  on percent returns
- line 244: `daily_std_dec` recomputed on decimal returns

The sortino at line 254 uses **population denominator** (`/ len(daily_rets_dec)`)
while sharpe uses **sample** (`/ (n-1)`). Comment at line 252 says "standard
formula" — accurate, but the *combination* makes Sharpe and Sortino non-comparable.
Fix: pick one (sample n-1 is standard for both).

### P1-11 — _streaming_max cache write under lock, but `peak` value can be reused stale if file shrinks mid-call
`Q:\finance-analyzer\portfolio\risk_management.py:70-110`. After reading
`cached["peak"]` under lock and dropping the lock, a different thread can do a
full rescan that finds a smaller peak (file rotation). When the first thread
reaches line 108 it overwrites with its **older + offset-based** peak. Race window
is small but observable — drawdown reading may resurrect a peak the rotation
intentionally cleared. Fix: re-check `_peak_cache.get(cache_key)` under the write
lock and merge with `max(...)`.

### P1-12 — monte_carlo.simulate_ticker drift calibrated for 1-day horizon, applied uniformly across horizons
`Q:\finance-analyzer\portfolio\monte_carlo.py:286-307`. `drift_from_probability`
returns annualized drift that, at horizon=1d, makes `P(S_T > S_0) = p_up`. At
horizon=3d, the same drift gives `P(S_3d > S_0) = N(z * sqrt(3))` — i.e., probability
is amplified. So 3d simulations don't reflect the input `p_up`. Either calibrate
per-horizon or document explicitly. The downstream consumers of `expected_return_3d`
(reporting) will misattribute "model edge" to longer horizons.

### P1-13 — cost_model.WARRANT_COSTS has min_fee_sek=0, masking 1 SEK Avanza minimum on partial fills
`Q:\finance-analyzer\portfolio\cost_model.py:64-70`. Avanza's standard-class
courtage on warrants/certificates is courtage-free above thresholds **but with
a 1 SEK floor** in the Mini class for partial fills under 10K SEK. Setting
`min_fee_sek=0.0` underestimates true cost on small orders, which is exactly
where the grid_fisher and metals_loop operate. Effect on `exit_optimizer`: EV
ranking favors small-quantile candidates that wouldn't actually be profitable
after the SEK floor. Magnitude is small per trade, but compounds over the
6500-SEK-cap grid laddering scenario.

### P1-14 — trade_guards.LOSS_DECAY uses `int(elapsed_hours // LOSS_DECAY_HOURS)` for bit-shift halvings — large gaps over-halve
`Q:\finance-analyzer\portfolio\trade_guards.py:94-96`. If the loop is down for
3 weeks (~504h) and `LOSS_DECAY_HOURS=24`, `halvings=21`, `base >> 21 = 0` →
`max(1, 0) = 1`. So a sufficient gap **always** resets all consecutive-loss
escalation to baseline — the system "forgets" a 4-loss streak after just 8 days
of downtime. Trading sessions returning from a multi-week outage start at 1× with
no escalation; the cooldown protection is gone. Wall-clock-only timing; no
trading-day-aware decay.
Fix: cap halvings at log2(max_multiplier), use trading-time elapsed (skip outages),
and explicitly persist `escalation_level` rather than recomputing.

### P1-15 — iskbets._save_state writes on **every** loop cycle even when nothing changed
`Q:\finance-analyzer\portfolio\iskbets.py:653-656,695-696`
```python
else:
    changed = True   # ← "Just update state with highest_price etc."
...
if changed:
    _save_state(state)
```
The else-branch (no exit triggered) always sets `changed=True` so every 60s cycle
writes `iskbets_state.json` even if the price hasn't moved. Causes:
1. Excess disk wear (Tens of writes/day per active position).
2. Real race window with `_handle_bought`/`_handle_sold` from the Telegram poller
   thread — no lock on `_load_state`/`_save_state` (compare to `portfolio_mgr._get_lock`).
3. If two Avanza/Telegram commands fire concurrently, last-writer-wins
   on the JSON file → lost `trade_history` entry.

Fix: only mark changed when `highest_price` actually updated; add a per-file lock
identical to `portfolio_mgr._get_lock`.

---

## P2 — Medium

### P2-1 — portfolio_mgr.update_state is recursion-unsafe — mutate_fn calling save_state will deadlock
`Q:\finance-analyzer\portfolio\portfolio_mgr.py:151-159`. `_state_lock` is a
`threading.Lock()` (not `RLock`). If a `mutate_fn` calls `save_state` (or any
other path that re-enters `_get_lock(path).acquire()`) the same thread blocks
forever. Not currently triggered but documented contract should require
non-reentrant mutate functions. Or change to `RLock`.

### P2-2 — portfolio_validator Check 8 uses `total_sek` for buy weighted average — includes fees in numerator
`Q:\finance-analyzer\portfolio\portfolio_validator.py:228-234`
```python
tx_price = tx.get("price_usd", 0) or 0
total_cost += tx_shares * tx_price
```
Wait — uses `price_usd`, not `total_sek`. Good in isolation. But `avg_cost_usd`
stored on the position should be USD-per-share net of fees too. The validator
compares `expected_avg = total_cost / total_bought` (gross of fees) to the
stored `avg_cost_usd`. If portfolio_mgr stores `avg_cost_usd` net of fees (it
should, per the cost-accounting convention), the validator generates false
positives at the 1% threshold. Audit the writer-vs-reader convention and pick
one.

### P2-3 — portfolio_mgr backup rotation has no fsync; power loss mid-rotation can lose ALL backups
`Q:\finance-analyzer\portfolio\portfolio_mgr.py:44-62`. `shutil.copy2` does
not fsync the destination; `os.rename` is atomic but `copy2` is not. If power
fails after the rename `path.bak2 → path.bak3` succeeds but `path.bak → path.bak2`
hasn't flushed, both backups can be corrupt. The recovery logic
`_load_state_from` will then return defaults.
Fix: explicit `fd.flush()`/`os.fsync()` on each rotation step, or use
`atomic_write_json` semantics (write+fsync+rename) for backup files too.

### P2-4 — risk_management.check_atr_stop_proximity caps stop at 0 but doesn't check MINI financing
`Q:\finance-analyzer\portfolio\risk_management.py:902-924`. The stop is `entry_price *
(1 - 2 * atr_pct / 100)`. For a MINI position with financing level at 90% of entry
price, the ATR stop at 88% is BELOW financing level — i.e., on knockout the
position is already gone before the stop fires. The proximity flag fires on the
*wrong* level. Fix: clamp `stop_price = max(stop_price, financing_level * 1.03)`
for MINI products (passes through holdings metadata).

### P2-5 — monte_carlo.simulate_paths antithetic logic loses one path on odd n_paths
`Q:\finance-analyzer\portfolio\monte_carlo.py:159-179`. For `n_paths=10001`:
`n_half = 5000`, generates 10000 paths from antithetic pair, then 1 extra.
But the extra path uses a fresh draw, breaking the antithetic variance property
for that 1 path. Negligible numerically. Worse: `n_paths=1` produces n_half=0,
empty arrays, then 1 path. Edge-case only.

### P2-6 — trade_validation max_cash_pct=50% silent: position can be 50% of cash even if `cash > 50% of total_value` (already saturated)
`Q:\finance-analyzer\portfolio\trade_validation.py:75-81`. The check is
`order_value / cash_available > 50%`. It doesn't see the rest of the portfolio.
A trader with 95% of net worth in BTC already and 5% cash can drop 50% of *cash*
on another instrument — passes validation, concentration risk silently spikes.
The portfolio-wide concentration check is supposed to catch this in `risk_management`,
but `trade_validation` is the gate before order placement. Fix: caller passes
`total_value`, not just `cash_available`; check `order_value / total_value < 30%`
as a secondary cap.

### P2-7 — trade_guards.get_all_guard_warnings — record_trade wiring sanity check is silent for first hour after deploy
`Q:\finance-analyzer\portfolio\trade_guards.py:360-368`. Warning only fires when
`all_warnings == []` AND `_portfolios_have_transactions()`. Right after a deploy
clears `trade_guard_state.json` but portfolios are pre-populated, the first
N cycles with no current BUY/SELL signals will fire the warning. After signals
fire, the empty-warnings branch is taken — so the check isn't quite "is wiring
broken", it's "was wiring broken AND there's currently no actionable trade". The
spurious warning during normal pre-trade idle is noise. Mild.

### P2-8 — cost_model rounding: total_cost_sek uses floats without fee floor enforcement
`Q:\finance-analyzer\portfolio\cost_model.py:44`. `max(trade_value_sek * bps / 10000, min_fee_sek)`
— if `bps=0` and `min_fee_sek=0` (warrant), returns 0 regardless. Spread+slippage
are pure bps. For trades below 1000 SEK that bypass the validator (e.g., grid
fisher's stop-loss legs that are computed against partial position), cost is
trivial. Mild.

### P2-9 — equity_curve daily_returns groups by `dt.date()` — UTC date, mixes Stockholm/NY market days
`Q:\finance-analyzer\portfolio\equity_curve.py:82-93`. ts is parsed UTC,
date_key uses UTC date. A trade at 22:00 UTC on a Tuesday (Wednesday 00:00 Stockholm)
is grouped as Tuesday for daily P&L. Reasonable convention but inconsistent with
"market session" reporting that uses local time elsewhere. Document or align.

### P2-10 — instrument_profile signal-trust tiers are hardcoded with 2026-04 accuracies — drift undetected
`Q:\finance-analyzer\portfolio\instrument_profile.py:26-57`. `_SILVER_TRUSTED`,
`_SILVER_IGNORED`, etc. contain hand-tuned accuracy numbers in comments (e.g.,
"94.7%", "4.5%"). These are static; signal accuracy drifts daily. The
`format_profile_briefing` does pull live accuracy when `signal_data` is passed,
but the trust/ignore lists themselves are not re-sorted. A signal that has
regressed below the ignore threshold (e.g., `mean_reversion` from 72.9% to 45%)
will still be on the trusted list, fed into Kelly weighting elsewhere.
Fix: drive the trust/ignore split from `accuracy_cache.json` at module import or
each `get_profile()` call.

### P2-11 — strategies/orchestrator: halted strategies never auto-resume
`Q:\finance-analyzer\portfolio\strategies\orchestrator.py:104-114`. Once a
strategy hits 10 consecutive errors it's added to `_halted` and never removed.
No reset path — even when the underlying error (transient network) clears.
Operator must restart the process. Fix: time-decay similar to LOSS_DECAY, or
explicit `resume(name)` API.

### P2-12 — strategies/orchestrator busy-loops at 0.5s; multiple strategies stall on slow tick
`Q:\finance-analyzer\portfolio\strategies\orchestrator.py:116`. `time.sleep(0.5)`
inside the for-loop. If Elongir's kline fetch takes 5s (3 separate Binance
calls per tick at line 60-62), GoldDigger waits. With poll intervals of 60s+
this is fine, but if either drops to 5s the other suffers head-of-line blocking.
Fix: spawn one thread per strategy with independent timing.

### P2-13 — exit_optimizer MarketSnapshot.usdsek default is 10.85 (off-by-cent from FX_RATE_FALLBACK=10.50)
`Q:\finance-analyzer\portfolio\exit_optimizer.py:54`. The dataclass default
`usdsek: float = 10.85` differs from `fx_rates.FX_RATE_FALLBACK = 10.50` and
from iskbets' `10.5`. Inconsistent fallbacks across modules → analyses produce
different SEK numbers given the same USD inputs when fx_rate isn't supplied.
Fix: import FX_RATE_FALLBACK; no module-local constants.

### P2-14 — kelly_sizing — `pnl_pct <= 0` classifies break-even as loss
`Q:\finance-analyzer\portfolio\kelly_sizing.py:110`
```python
losses = [abs(p) for p in pnl_list if p <= 0]
```
A round-trip with `pnl_pct == 0` adds `0` to losses, deflating `avg_loss`. With
one zero-loss and one real -3% loss, `avg_loss = (0 + 3) / 2 = 1.5%` — half of
the actual average. Kelly fraction inflates.

### P2-15 — risk_management.compute_probabilistic_stops swallows ImportError silently
`Q:\finance-analyzer\portfolio\risk_management.py:418-423`. Returns `{}` on
`from portfolio.exit_optimizer import ...` failure. Layer 2 sees no
`probabilistic_stops` block and assumes there are no positions to flag —
fails open. Should log critical at minimum.

### P2-16 — monte_carlo_risk.compute_portfolio_var:436 ATR fallback 2.0% for unknown ticker
`Q:\finance-analyzer\portfolio\monte_carlo_risk.py:436`. For MSTR (~5% daily ATR
typical) this underestimates VaR by ~2.5×. Combined with the FX bug (P0-1) the
total SEK VaR error can be 25×. Fix: import `_atr_default_for_ticker` from
monte_carlo.

---

## P3 — Low

### P3-1 — instrument_profile.get_regime_behavior — default to {} on missing regime, not None
`Q:\finance-analyzer\portfolio\instrument_profile.py:236-240`. Returns hardcoded
default dict — callers must guard against None. Minor API friction.

### P3-2 — kelly_metals._get_outcome_stats uses bare `except Exception`
`Q:\finance-analyzer\portfolio\kelly_metals.py:94-95`. Catches everything
including KeyboardInterrupt? Actually `except Exception` is fine, KeyboardInterrupt
is BaseException. But silent return None on sqlite errors masks real problems
(corrupt DB, missing schema). Fix: log warning.

### P3-3 — exposure_coach format strings could overflow with very long regime names
Cosmetic. Not a real issue but the rationale builder doesn't truncate.

### P3-4 — trade_guards write timestamps with `datetime.now(UTC).isoformat()` then parse without Z handling in some places
`Q:\finance-analyzer\portfolio\trade_guards.py:88-89,142,206,303`. Inconsistent
Z-stripping. Python 3.11+ handles both, so non-issue on current runtime.

### P3-5 — portfolio_validator Check 5 — "inconsistent fee tracking" if some tx have fee_sek and some don't
`Q:\finance-analyzer\portfolio\portfolio_validator.py:159-163`. This triggers as
a validation error even during migration windows where the writer was just
updated to include fee_sek. False positive during upgrades.

### P3-6 — stats.py invocation_stats — no timezone normalization
`Q:\finance-analyzer\portfolio\stats.py:21,57`. `datetime.fromisoformat(e["ts"])`
without UTC normalization. If invocation logs contain mixed-tz timestamps the
day grouping is per-author-timezone. Mild.

### P3-7 — equity_curve.compute_trade_metrics calmar_ratio uses raw `years` without trading-day adjustment
`Q:\finance-analyzer\portfolio\equity_curve.py:556`. For round-trips that span
only weekends, `years` is tiny, annualized return explodes, Calmar absurd. Add
floor `years ≥ 1/52` (one week).

### P3-8 — monte_carlo logs `exc_info=True` on every simulate_all per-ticker exception
`Q:\finance-analyzer\portfolio\monte_carlo.py:403`. Full traceback for every
ticker failure — log noise when one ticker is consistently broken. Reduce to
`exc_info=False` or rate-limit.

### P3-9 — warrant_portfolio.record_warrant_transaction has no concurrency lock
Same class as P1-1. If Telegram poller thread and metals loop thread both call
this, last-writer-wins. Mild because typically single-writer.

### P3-10 — iskbets._get_current_price logs "Bet parse failed" — misleading wording
`Q:\finance-analyzer\portfolio\iskbets.py:915`. Should be "Failed to fetch price
for %s" — the message is from copy-paste. Cosmetic.

### P3-11 — exit_optimizer.compute_exit_plan computes hold_to_close EV from 5 percentile midpoints
`Q:\finance-analyzer\portfolio\exit_optimizer.py:617-621`
```python
terminal_pnls = np.array([
    _compute_pnl_sek(position, float(p), market, costs)
    for p in np.percentile(terminal, [10, 25, 50, 75, 90])
])
hold_ev = float(np.mean(terminal_pnls))
```
Mean of 5 quantile-midpoints is a coarse, biased estimator of the true mean.
Should be `_compute_pnl_sek` applied to every terminal sample then `np.mean`, OR
analytical mean of GBM. The 5-point quadrature can be off by 5-10% vs sample mean
for fat-tailed distributions.

---

## Tests missing

1. **Per-instrument MINI BEAR P&L**: no test covers `warrant_pnl` or
   `_compute_pnl_sek` with a BEAR cert (direction flip). Knockout-while-held
   is also untested.
2. **fx_rate fallback chain** in monte_carlo_risk + exit_optimizer +
   iskbets — tests should pass `agent_summary["fx_rate"] = 1.0` and assert the
   resolved rate is NOT 1.0 (sanity band rejection).
3. **Concentration with zero cash**: a portfolio at 0 cash + existing 45% BTC
   position should still flag concentration on additional BTC BUY (P1-2).
4. **Risk classifier with None inputs** and unknown regime — should not silently
   degrade to LOW.
5. **Backup rotation with shrunk file**: `_streaming_max` cache invalidation
   when `portfolio_value_history.jsonl` is truncated/rotated mid-stream.
6. **Out-of-order transactions**: `_pair_round_trips` with manually swapped
   timestamps — verifies the sort prerequisite is enforced.
7. **Loss-decay over multi-week outage**: after 21 days down, escalation should
   NOT silently reset to 1× (P1-14).
8. **iskbets fx mixing**: enter at fx=10.5, exit at fx=10.8 — exit alert SEK
   P&L must match `pos["shares"] * (exit_price - entry_price) * fx_at_exit`,
   not the collapsed formula.
9. **warrant_portfolio validator coverage** — explicit test that
   `validate_portfolio_file("portfolio_state_warrants.json")` exists and runs.
10. **kelly_metals units floor**: ask_price_sek=99, position_sek=1000 →
    units=10 → actual_sek=990 → should round position_sek to 0 OR bump to 11
    units (1089 SEK).
11. **Exit_optimizer BEAR override**: BEAR cert with `financing_level=110`,
    `market.price=100` — should NOT force market exit on every cycle.
12. **monte_carlo_risk t-copula tail dependence**: assert empirical λ_lower
    at df=4 ≥ 0.15 for ρ=0.85.
13. **portfolio_mgr update_state recursion**: confirm RuntimeError or document
    that mutate_fn must not call save_state.
14. **strategies/orchestrator halted-resume**: after error subsides, strategy
    should eventually resume.

## Cross-cutting observations

1. **FX rate handling has 4+ implementations** with different fallbacks and
   different sanity bands. `risk_management._resolve_fx_rate` is the gold
   standard (cache + band); `monte_carlo_risk`, `exit_optimizer`, `iskbets`,
   `kelly_sizing` (via consumers), and `warrant_portfolio` all roll their own.
   This is a refactor candidate: `portfolio.fx_rates.resolve_safe_rate(agent_summary)`.

2. **MINI warrant model is incomplete across the entire stack.** Knockout
   detection, BEAR-vs-BULL direction, financing level proximity, and leverage
   drift over time are either missing or only partially handled. `warrant_portfolio`
   is the worst offender; `exit_optimizer` is best (clamps to 0) but still
   BULL-only. With grid_fisher placing live orders on BULL+BEAR certs (per
   CLAUDE.md "with-signal direction of BULL/BEAR certs"), this is a live
   correctness issue, not a theoretical one.

3. **Atomic I/O coverage is partial**: `portfolio_mgr` has lock + rotation;
   `warrant_portfolio`, `iskbets`, `trade_guards` have only atomic_write_json
   (no per-file lock, no backup rotation). The "atomic I/O only" rule in
   CLAUDE.md is followed in spirit but not uniformly applied to all state
   files.

4. **Three different MIN_TRADE_SEK constants**: `trade_validation.min_order_sek = 1000.0`,
   `kelly_metals.MIN_TRADE_SEK = 1000.0`, `kelly_sizing` hardcoded `1000` at line
   326. Project-rule of 1000 SEK is correctly applied across all three NOW (the
   prior 500 SEK defaults have been fixed) — but they're three separate sources of
   truth. One module-level constant in `portfolio.cost_model` or `portfolio.constants`
   would prevent drift.

5. **Wall-clock dependence**: `trade_guards`, `risk_management.get_position_ages`,
   `equity_curve._pair_round_trips`, `iskbets._past_time_exit/_before_cutoff` all
   rely on `datetime.now(UTC)`. None use `time.monotonic()` for elapsed
   measurements; clock skew or NTP corrections during loop runtime can affect
   cooldown/decay calculations. Low-frequency events (cooldown expiry, EOD
   exit) are most exposed.

6. **`accuracy_stats` per-ticker block is preferred path, but the system-wide
   fallback in `kelly_sizing._get_ticker_signal_accuracy` (line 215-219) silently
   drops signals with `<5 samples on BOTH axes`** — a signal that's brand-new on
   a ticker gets zero weight regardless of its system-wide reliability. Compare
   to the documented "70% recent + 30% all-time" pattern elsewhere; here it's
   "5+ samples or nothing".

7. **Risk audit (`compute_all_risk_flags`) emits flags only — no severity
   threshold to translate flags into a go/no-go decision.** `should_block_trade`
   in `trade_guards` returns boolean but the risk_management equivalent doesn't
   exist. Layer 2 sees the flags list and decides. If Layer 2 is broken or down,
   nothing blocks.

8. **Compounding fee accounting drift**: `kelly_sizing._compute_trade_stats`
   uses gross P&L. `equity_curve._pair_round_trips` uses net P&L (P0-6 in the
   2026-05-02 review). These two modules produce different "win rates" for the
   same transaction list. Whichever one Layer 2 cites depends on which API is
   reached first.

9. **Strategy orchestrator integrates only Elongir + GoldDigger** — neither
   is using the `SharedData.trade_queue_lock` for non-queue state (e.g., cert
   prices). Patient and Bold strategies live OUTSIDE the orchestrator and have
   their own thread management. The "Patient vs Bold parameter drift" risk from
   the prompt isn't visible here because the two never share code — the
   parameter divergence happens in `agent_invocation`/`autonomous` modules out
   of scope for this review.

10. **No bear/short modeling anywhere.** All P&L formulas assume long-only.
    The warrant subsystem nominally trades BULL+BEAR certs, but treats both as
    "leveraged underlying" — a 5× BEAR cert is modeled identically to a 5×
    BULL cert with the same `leverage=5` field. The "direction" attribute on
    a holding is, as far as I can tell from the in-scope code, either absent
    or unused.
