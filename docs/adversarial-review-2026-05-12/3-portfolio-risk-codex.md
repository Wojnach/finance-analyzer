# Codex adversarial review: portfolio-risk
## Summary
Three blockers and eight additional concrete issues. The worst paths are broken MINI warrant valuation/KO handling, live-price violations inside risk controls, and non-atomic warrant state updates.

## P0 — Blockers
- portfolio/warrant_portfolio.py:52-103, 249-255 — Why it bites: `warrant_pnl()` ignores `fx_rate`, prices warrants as `underlying_change * leverage`, never clamps KO at zero, and holdings never store a financing/barrier field, so barrier knockouts cannot be detected at all. A KO’d MINI can be materially misvalued. Fix: persist financing/barrier metadata in holdings and price MINI warrants from barrier math, e.g. `max((underlying - financing_level) * fx * ratio, 0)`, failing closed if barrier data is missing.
- portfolio/risk_management.py:206-213, 249-251, 762-775 — Why it bites: drawdown and concentration math silently fall back to `avg_cost_usd` when a live price is missing. That violates the “live prices first” rule and can keep the circuit breaker open or understate concentration while a held asset is actually collapsing. Fix: fail closed on missing live prices for held names, surface a hard risk flag, and do not substitute cost basis in risk math.
- portfolio/warrant_portfolio.py:198-265 — Why it bites: warrant state updates are a bare load-mutate-save sequence with no lock around the read-modify-write pair. Parallel BUY/SELL calls can drop transactions or overwrite unit counts even though the final write is atomic. Fix: add a per-file lock or an `update_state`-style atomic mutation helper for the warrants state file.

## P1 — High
- portfolio/trade_guards.py:126-127, 189-231, 264-312 — Why it bites: guard evaluation and guard recording are separate critical sections. Two parallel orders can both pass cooldown/rate-limit checks against the same snapshot and only record after both have executed, bypassing the guard. Fix: provide an atomic `check_and_record` reservation path for order admission.
- portfolio/kelly_sizing.py:90-103, 291-299 — Why it bites: `_compute_trade_stats()` compares every sell against the weighted average of all historical buys for that ticker, without FIFO or share depletion. Scale-ins and partial exits therefore contaminate `avg_win_pct`/`avg_loss_pct`, and `recommended_size()` consumes those distorted Kelly inputs. Fix: match sells against remaining buy lots FIFO/LIFO consistently and compute realized trade stats from matched lots only.
- portfolio/trade_validation.py:32, 57-64; portfolio/kelly_sizing.py:325-327; portfolio/kelly_metals.py:44, 228-238 — Why it bites: the minimum order floor is still `500 SEK`, not `1000 SEK`, and `recommended_metals_size()` checks the floor before unit rounding, so it can recommend one cert whose actual notional is below the minimum. Fix: centralize `MIN_ORDER_SEK = 1000` and enforce it on the post-rounding actual order value.
- portfolio/monte_carlo.py:50-65; portfolio/risk_management.py:461-463; portfolio/exit_optimizer.py:181-184 — Why it bites: ATR is described as hourly, but annualization uses `sqrt(252/14)` as if ATR were 14 daily bars. That understates intraday volatility and therefore understates stop-hit / knockout probabilities. Fix: annualize from the real bar frequency or pass a precomputed annual vol from upstream and use it consistently.
- portfolio/portfolio_validator.py:70-72; portfolio/equity_curve.py:343-405 — Why it bites: the validator defines `BUY total_sek` as fee-inclusive and `SELL total_sek` as fee-net, but `_pair_round_trips()` derives per-share prices from those totals and then subtracts buy and sell fees again. Realized P&L is therefore too low, especially on partial fills. Fix: either use gross proceeds/costs for per-share prices or stop subtracting fees a second time when `total_sek` already embeds them.
- portfolio/exit_optimizer.py:335-340; portfolio/monte_carlo_risk.py:419 — Why it bites: stock/crypto exit P&L revalues the entry leg at the current FX rate, which erases SEK FX P&L, and VaR-in-SEK trusts raw `agent_summary["fx_rate"]` without the sanity-band fallback used elsewhere. SEK risk can therefore be mis-ranked by exit EV and mis-scaled in VaR. Fix: carry entry SEK cost or entry FX explicitly, and route all SEK conversions through the validated FX resolver.

## P2 — Medium
- portfolio/trade_risk_classifier.py:80-84 — Why it bites: unknown or renamed regimes get `0` regime-risk points via `_REGIME_SCORES.get(..., 0)`, which is a silent permissive fallback exactly when upstream data is degraded. Fix: treat missing/unknown regime as explicit uncertainty, not as risk-free.
- portfolio/monte_carlo.py:205-219, 309-328 — Why it bites: `p_stop_hit_*` is computed as terminal price below the stop, not first-passage/hit probability. Any path that breaches the stop intrahorizon and recovers by the close is missed, so stop risk is understated. Fix: compute hit probability from path minima / first-hit times, not terminal-only outcomes.

## P3 — Low
- None.

## Tests missing
- A warrant valuation test where the underlying crosses the MINI barrier and the marked value clamps to zero, including USD/SEK movement.
- A concurrent warrant-state mutation test with simultaneous BUY and SELL proving no transaction or unit count is lost.
- A guard admission race test showing two parallel BUYs cannot both pass cooldown / position-window checks.
- A Kelly sizing regression with scale-in plus partial exit, verifying realized trade stats use matched lots and not all historical buys.
- A minimum-order test suite that enforces `1000 SEK` everywhere, including after cert unit rounding.
- A risk-control test where one held ticker is missing from live prices and drawdown/concentration fail closed instead of using `avg_cost_usd`.
- An equity-curve test that confirms `total_sek` fee semantics do not double-charge round-trip P&L.
- A Monte Carlo stop test that distinguishes terminal-below-stop from intrahorizon stop-hit probability.