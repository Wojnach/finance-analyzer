# Adversarial Review: portfolio-risk subsystem (2026-05-08)

[P1] portfolio/risk_management.py:760-770
**Concentration check silently zeros existing_value when avg_cost_usd=0.**
Problem: Missing live price falls back to avg_cost_usd; if avg_cost is 0 (corrupt or
freshly-zero-init holding), existing_value=0 and concentration check passes for every
new buy. No price>0 validation.
Fix: Reject the check or escalate with warning if neither live price nor a positive
avg_cost is available.

[P1] portfolio/risk_management.py:206-211
**Holdings value falls back to stale `avg_cost_usd` if live price missing.**
Problem: If `avg_cost_usd` is 0 (corruption or freshly initialised), drawdown circuit
breaker reports false safety. No critical-level log surfaces the silent fallback.
Fix: Treat avg_cost==0 as missing, raise a critical log entry, and refuse to compute
drawdown until a live price is recovered.

[P1] portfolio/warrant_portfolio.py:88
**`underlying_entry_price_usd=0` (API timeout) silently omits PnL.**
Problem: VWAP averaging skips when new price is zero, leaving a stale reference. After
an avg-in into a position that fetched zero, the recorded reference is wrong and all
subsequent P&L is off.
Fix: Refuse to record the avg-in if price<=0; raise to caller for retry.

[P1] portfolio/equity_curve.py:104-108
**Commit `08e0f378` adds `returns.append(0.0)` on `prev_val=0`.**
Problem: Treats "no data" as "flat day" — biases Sharpe upward by lowering volatility
during cold-start. Sharpe overstated 1–2% on new portfolios.
Fix: Skip the bar entirely when prev_val<=0 (no return measurable), don't synthesise a
0.0 return.

[P2] portfolio/kelly_sizing.py:280-289
**Per-ticker fallback hides data source after fallback fires.**
Problem: When per-ticker block has zero votes the code falls back to consensus weights
but the source string still claims `"per-ticker"`. Misleads downstream confidence
reporting.
Fix: Set source to "fallback:consensus" whenever the fallback path fires.

[P2] portfolio/risk_management.py:776-778
**Concentration false-negative when cash=0 + NaN fx_rate.**
Problem: `proposed_alloc=0` short-circuits, but if the underlying fx_rate is NaN,
`existing_value` becomes 0 and the check silently passes for a fully-invested portfolio.
Fix: Validate fx_rate is finite before computing existing_value.

[P2] portfolio/risk_management.py:372-374
**ATR stop hard-capped at 15% hides tight stops on volatile warrants.**
Problem: 20% ATR day collapses to 15% capped without warning, violating the
"never stop near MINI barriers" memory rule when barrier is <15% away.
Fix: Either lift the cap when barrier is the binding constraint, or log and refuse
to place a stop that would be inside the barrier corridor.

[P2] portfolio/kelly_sizing.py:326
**`recommended_sek=0` silently disables all trading when `kelly_pct=0`.**
Problem: Broken signal silently disables sizing with no error or critical log; loop
continues to think trades are being recommended.
Fix: Raise critical alert when kelly_pct lands at 0 due to non-trivial reasons (signal
gap vs deliberate skip).

[P2] portfolio/equity_curve.py:378
**`distance_to_stop_pct=inf` if `stop_price<=0`.**
Problem: Infinity not clamped; confuses stop-proximity dashboards and any consumer that
sorts on the field.
Fix: Return None/sentinel (or 0) when stop_price is invalid.

[P2] portfolio/circuit_breaker.py:98-99,106
**`_half_open_probe_sent` flag never checked in production code.**
Problem: State machine relies solely on `_state`. Latent bug: if `allow_request()`
returns True but the probe is never actually sent, half-open state never closes back.
Fix: Either remove the unused flag or wire it into `allow_request()`.

[P2] portfolio/cost_model.py:55
**`round_trip_pct()` assumes symmetric slippage on BUY and SELL.**
Problem: Warrants have asymmetric bid/ask; crossing ask vs bid differs 2–5 bps. Round-
trip cost undercounts the sell side.
Fix: Take separate `slippage_bps_buy` / `slippage_bps_sell` params, or apply a known
warrant-side multiplier.

[P3] portfolio/equity_curve.py:413
**Shares always rounded to 8 decimals in reporting.**
Problem: Warrants trade in whole units; "1.00000000" is misleading and amplifies
rounding noise in profit_factor.
Fix: Round to instrument-class precision (0 for warrants, 8 for crypto, 4 for stocks).

[P3] portfolio/risk_management.py:741-800
**`check_concentration_risk` only checks BUY actions.**
Problem: SELL-side concentration risk (reducing diversification by trimming the
diversifier) is never enforced. One-directional check.
Fix: Apply concentration logic to SELL-direction trades that reduce a non-overweight
position toward zero.

## Summary

13 findings. Top theme: silent zero/NaN fallbacks that report safety when state is
actually corrupt (avg_cost=0, fx_rate=NaN, stale price). Equity curve fix from
2026-05-08 introduced a new bias (treating no-data as flat day). Kelly + cost model
both have asymmetric or ambiguous semantics that misreport confidence.
