# Codex Review — 3-portfolio-risk

## Summary

Several core portfolio/risk outputs are materially wrong, including stop-hit probabilities, multi-day Monte Carlo calibration, and leveraged warrant valuation. The patch also has state/guard bookkeeping bugs that can misreport rolling changes or block valid trades, so it should not be considered correct.

Full review comments:

- [P1] Estimate stop hits from the path, not the terminal price — Q:\fa-review\portfolio\monte_carlo.py:328-328
  For any volatile ticker or multi-hour/day horizon where price can dip through the stop and then recover, this understates `p_stop_hit_*` because it only measures `P(S_T < stop_price)` from terminal prices. The field name/docstring describe a stop *hit* probability over the whole horizon, so downstream risk/exits will miss intrahorizon stop-outs unless the simulation tracks path minima or uses discretized paths.

- [P1] Floor leveraged warrant marks at zero before computing P&L — Q:\fa-review\portfolio\warrant_portfolio.py:100-103
  When the underlying drops by more than `100 / leverage` percent (for example, a 5x product on a >20% drop), `current_implied_sek` becomes negative here and the returned loss exceeds -100%. That impossible negative mark then flows into `total_value_sek` and portfolio summaries, so a normal adverse move can produce nonsensical negative position values unless the implied warrant price is clamped at zero.

- [P2] Recompute drift per horizon instead of reusing the 1d inversion — Q:\fa-review\portfolio\monte_carlo.py:304-307
  This `drift` is derived once from `p_up` via `drift_from_probability()`, but that helper inverts the GBM CDF specifically for a 1-day horizon. Reusing the same annualized drift for the default 3-day run makes the longer-horizon simulation systematically too bullish/bearish (for example, `p_up=0.60` implies about 0.67 probability of finishing up at 3d), so the multi-day bands and expected returns are miscalibrated.

- [P2] Count only genuinely new positions toward the BUY rate limit — Q:\fa-review\portfolio\trade_guards.py:286-291
  `new_position_timestamps` is supposed to enforce a “max new positions per window” guard, but this appends an entry for every BUY. Once a strategy scales into an existing holding, those adds still consume the quota, and `check_overtrading_guards()` can block the next unrelated entry as if a fresh position had been opened.

- [P2] Anchor rolling windows to the latest snapshot timestamp — Q:\fa-review\portfolio\cumulative_tracker.py:129-130
  If snapshot logging stalls or a caller passes historical snapshots, this compares against wall-clock `now` instead of the timestamp of `snapshots[-1]`. In practice a file that is one day stale can report a 1d change of `0.0` or `None` even though the last two samples are 24 hours apart, so the 1d/3d/7d summaries become wrong whenever the newest sample is not “right now”.

- [P2] Deep-copy the default portfolio state before returning it — Q:\fa-review\portfolio\portfolio_mgr.py:68-69
  When the state file is missing or corrupt, these branches only shallow-copy `_DEFAULT_STATE`. The returned `holdings` and `transactions` objects are still shared with the module-level default, so mutating one freshly initialized state contaminates later `load_state()` calls with ghost holdings/transactions in the same process.

- [P3] Initialize `total_fees_sek` in the default state shape — Q:\fa-review\portfolio\portfolio_mgr.py:21-26
  On a brand-new or recovered portfolio, `load_state()` returns this default structure, but `validate_portfolio()` in the same patch treats a missing `total_fees_sek` as an error. Since the state manager never seeds that field, every fresh portfolio starts out invalid and fee reporting has to special-case around an avoidable missing key.
Several core portfolio/risk outputs are materially wrong, including stop-hit probabilities, multi-day Monte Carlo calibration, and leveraged warrant valuation. The patch also has state/guard bookkeeping bugs that can misreport rolling changes or block valid trades, so it should not be considered correct.

## Full review comments

- [P1] Estimate stop hits from the path, not the terminal price — Q:\fa-review\portfolio\monte_carlo.py:328-328
  For any volatile ticker or multi-hour/day horizon where price can dip through the stop and then recover, this understates `p_stop_hit_*` because it only measures `P(S_T < stop_price)` from terminal prices. The field name/docstring describe a stop *hit* probability over the whole horizon, so downstream risk/exits will miss intrahorizon stop-outs unless the simulation tracks path minima or uses discretized paths.

- [P1] Floor leveraged warrant marks at zero before computing P&L — Q:\fa-review\portfolio\warrant_portfolio.py:100-103
  When the underlying drops by more than `100 / leverage` percent (for example, a 5x product on a >20% drop), `current_implied_sek` becomes negative here and the returned loss exceeds -100%. That impossible negative mark then flows into `total_value_sek` and portfolio summaries, so a normal adverse move can produce nonsensical negative position values unless the implied warrant price is clamped at zero.

- [P2] Recompute drift per horizon instead of reusing the 1d inversion — Q:\fa-review\portfolio\monte_carlo.py:304-307
  This `drift` is derived once from `p_up` via `drift_from_probability()`, but that helper inverts the GBM CDF specifically for a 1-day horizon. Reusing the same annualized drift for the default 3-day run makes the longer-horizon simulation systematically too bullish/bearish (for example, `p_up=0.60` implies about 0.67 probability of finishing up at 3d), so the multi-day bands and expected returns are miscalibrated.

- [P2] Count only genuinely new positions toward the BUY rate limit — Q:\fa-review\portfolio\trade_guards.py:286-291
  `new_position_timestamps` is supposed to enforce a “max new positions per window” guard, but this appends an entry for every BUY. Once a strategy scales into an existing holding, those adds still consume the quota, and `check_overtrading_guards()` can block the next unrelated entry as if a fresh position had been opened.

- [P2] Anchor rolling windows to the latest snapshot timestamp — Q:\fa-review\portfolio\cumulative_tracker.py:129-130
  If snapshot logging stalls or a caller passes historical snapshots, this compares against wall-clock `now` instead of the timestamp of `snapshots[-1]`. In practice a file that is one day stale can report a 1d change of `0.0` or `None` even though the last two samples are 24 hours apart, so the 1d/3d/7d summaries become wrong whenever the newest sample is not “right now”.

- [P2] Deep-copy the default portfolio state before returning it — Q:\fa-review\portfolio\portfolio_mgr.py:68-69
  When the state file is missing or corrupt, these branches only shallow-copy `_DEFAULT_STATE`. The returned `holdings` and `transactions` objects are still shared with the module-level default, so mutating one freshly initialized state contaminates later `load_state()` calls with ghost holdings/transactions in the same process.

- [P3] Initialize `total_fees_sek` in the default state shape — Q:\fa-review\portfolio\portfolio_mgr.py:21-26
  On a brand-new or recovered portfolio, `load_state()` returns this default structure, but `validate_portfolio()` in the same patch treats a missing `total_fees_sek` as an error. Since the state manager never seeds that field, every fresh portfolio starts out invalid and fee reporting has to special-case around an avoidable missing key.
