# Claude critique of codex review — portfolio-risk

Codex caught a real bag of P1/P2 valuation/sizing bugs (FX bypass at monte_carlo_risk.py:431, terminal-only stop probability, drop-on-zero-price warrants, fee-double-count claim) but it (a) missed the Avanza min-order-size policy violation (validation default 500 vs project rule 1000), (b) missed the explicit knockout-barrier blindness in ATR stop placement, (c) missed the cash-only Kelly base that conflicts with concentration check, and (d) missed the negative-warrant-value bug. Several of codex's "P2" findings are minor (shared default mutation, tx field type-checks) while real P0/P1s went uncalled.

## Codex finding verdicts

[CONFIRM] portfolio/portfolio_mgr.py:68 — `{**_DEFAULT_STATE, ...}` reuses shared `holdings`/`transactions` containers | The shallow `{**_DEFAULT_STATE}` spread does NOT clone nested `holdings={}` / `transactions=[]` — the literal module-level dict/list are reused. portfolio_mgr.py:21-26 + 68 confirms; same pattern in _validated_state at line 69. PARTIAL impact in practice (the `if not isinstance(...)` guards at 71-74 force-replace before mutation in the validator path) but the `_load_state_from` at 105 still returns a dict whose holdings/transactions ARE the module defaults.

[CONFIRM] portfolio/portfolio_validator.py:61 — `validate_portfolio()` assumes `holdings` is dict of dicts | portfolio_validator.py:61 `for ticker, pos in holdings.items()` raises AttributeError if `holdings=[]`. No type guard at line 41. Reasonable bug; severity P2 fair.

[CONFIRM] portfolio/portfolio_validator.py:225 — avg-cost averages every historical BUY | portfolio_validator.py:226-238 sums ALL BUY tx for the ticker without subtracting SELL-consumed lots. Reentry after full close → false validation failure. Confirmed.

[CONFIRM] portfolio/risk_management.py:205 — Missing live prices fall back to `avg_cost_usd` | risk_management.py:204-207 `holdings_value += shares * avg_cost * fx_rate` when ticker absent from signals. Stale entry-cost valuation suppresses drawdown correctly identified.

[CONFIRM] portfolio/risk_management.py:247 — Empty agent_summary → cash-only valuation | risk_management.py:246-265 explicitly logs the "NOT truly conservative" comment and falls back to cash. The warning is logged but the breaker still returns a non-breached result. This is in the code as a known blind spot — codex correctly flags it as a real bug.

[CONFIRM] portfolio/risk_management.py:763 — `check_concentration_risk()` hardcodes 15%/30% buy size | risk_management.py:762-764 `alloc_pct = 0.30 if strategy == "bold" else 0.15; proposed_alloc = min(total_value * alloc_pct, cash)` — uses the strategy's policy ceiling instead of the actual proposed trade. Confirmed; the function never receives a real trade-size parameter.

[CONFIRM] portfolio/equity_curve.py:405 — Subtracts `sell_fee_sek` from PnL even though SELL.total_sek is net | equity_curve.py:405 `pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share`. validate_portfolio at portfolio_validator.py:71 confirms "SELL total_sek = net proceeds (after fee deducted)". So sell_price_per_share = sell_total/sell_shares is already net-of-fee, and then sell_fee_share is subtracted again. Genuine double-count, P2 understated — this directly distorts profit_factor and total_pnl_sek which the comment at 398-402 explicitly claims are "net of fees".

[CONFIRM] portfolio/monte_carlo.py:328 — `p_stop_hit_*` uses terminal price below stop | monte_carlo.py:328 `result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)` — `probability_below` at 217-219 uses terminal `_terminal_prices`, not path minima. Confirmed; the engine only stores terminal prices (line 173). Note: `compute_probabilistic_stops` in risk_management.py:475 DOES use path minima correctly via `np.min(paths[:, 1:], axis=1)`, but `simulate_ticker` is the codepath used by reporting/MC summary.

[PARTIAL] portfolio/monte_carlo.py:95 — `drift_from_probability()` hardcodes 1-day scaling | The math at line 95 (`mu = sigma * z * sqrt(252) + 0.5 * sigma^2`) embeds T=1/252 from inverting the 1-day P(S_T>S0) relation. `simulate_paths` at line 154 uses `T = self.horizon_days/252.0` — drift is annualised, so it scales correctly with T inside simulate_paths. So multi-day bands DO scale appropriately; what stays fixed is the IMPLIED p_up at the 1-day horizon (matches the source signal). Codex's "doesn't match input p_up at 3d" framing is partially valid — but this is by design: the input p_up is a 1-day directional probability, not a 3-day one. Severity overstated.

[CONFIRM] portfolio/monte_carlo_risk.py:431 — SEK VaR trusts raw `agent_summary["fx_rate"]` | monte_carlo_risk.py:431 `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` skips `_resolve_fx_rate`. risk_management.py:121-180 has the validated chain that bounds rate to [FX_RATE_MIN, FX_RATE_MAX]. If a stale agent_summary embeds an out-of-band 1.0, monte_carlo_risk.py uses it directly. Strong confirm; this is a regression of the P1-15 fix.

[CONFIRM] portfolio/monte_carlo_risk.py:444 — Missing/non-positive live prices silently dropped | monte_carlo_risk.py:443-445 `price = ticker_data.get("price_usd", 0); if price <= 0: continue` — position is omitted entirely, exposure disappears from VaR. Genuine P0.

[CONFIRM] portfolio/monte_carlo_risk.py:399 — `drawdown_probability()` thresholds against invested exposure only | monte_carlo_risk.py:393-400 `total_value = sum(shares × price)` (cash NOT included), then `loss_threshold = -total_value × threshold/100`. Confirmed; the metric is exposure drawdown, not portfolio drawdown.

[CONFIRM] portfolio/kelly_sizing.py:95 — Weighted avg buy price uses every historical BUY | kelly_sizing.py:91-95 `total_shares_bought = sum(b.get("shares", 0) for b in buys); total_cost = sum(b.get("total_sek", 0) for b in buys); avg_buy_price = total_cost/total_shares_bought` — non-FIFO, no SELL-consumption tracking. P3 understates this — Kelly inputs to the actual sizing routine are downstream affected.

[CONFIRM] portfolio/trade_guards.py:291 — `record_trade()` counts every BUY as a new position | trade_guards.py:291-296 unconditionally appends `now_str` to `new_position_timestamps[strategy]` for every BUY action regardless of whether shares were 0 prior. Averaging-in burns rate-limit budget. Confirmed.

[PARTIAL] portfolio/trade_guards.py:126 — Guard admission not atomic with recording | trade_guards.py:126-128 acquires `_state_lock`, reads state, and releases the lock before returning. trade_guards.py:264 `record_trade` re-acquires the lock for write. Between check and write there IS a TOCTOU window, but record_trade is called ONLY by Layer 2 sequential decision flow per `agent_invocation`, not concurrently across BUYs for the same strategy/ticker. Race exists in theory; in practice the loop's serialized invocation gates it. Severity P1 overstated.

[CONFIRM] portfolio/trade_validation.py:22 — `validate_trade()` only referenced from tests | grep confirms `from portfolio.trade_validation` only appears in tests/test_trade_validation.py and nowhere in production code paths. Dead code. Confirmed.

[CONFIRM] portfolio/trade_risk_classifier.py:29 — `classify_trade_risk()` only referenced from tests | grep confirms `from portfolio.trade_risk_classifier` only in tests/test_trade_risk_classifier.py and a planning doc. Dead code. Confirmed.

[CONFIRM] portfolio/cost_model.py:116 — Unknown instrument types fall back to STOCK_COSTS | cost_model.py:116 `return _COST_MODELS.get(instrument_type, STOCK_COSTS)` — unknown type silently uses stock cost (1 SEK floor, 5bps spread) instead of warrant or crypto. Real risk on typo. Confirmed.

[CONFIRM] portfolio/warrant_portfolio.py:100 — Leveraged losses drive `current_implied_sek` below zero | warrant_portfolio.py:100 `current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)` — for 5x silver and -25% underlying, multiplier is -0.25; warrant value goes negative. No knockout flag. Confirmed P1.

[CONFIRM] portfolio/warrant_portfolio.py:154 — `get_warrant_summary()` drops position when underlying price missing | warrant_portfolio.py:153-155 `current_price = prices_usd.get(underlying); if not current_price: continue` — warrant disappears from the dashboard summary. Confirmed P0; also note the `not current_price` falsy check would drop a legitimate-but-very-small price (silver < 1.0). The whole position is excluded silently.

## MISSED BY CODEX

The following P0/P1 items from claude-portfolio-risk.md were NOT in codex's report — verifying each independently:

[CONFIRM-MISSED] portfolio/trade_validation.py:32 — `min_order_sek: float = 500.0` contradicts 1000 SEK Avanza minimum | trade_validation.py:32 default is 500.0; portfolio-risk.md (project rule injected as system reminder) explicitly states "Min order size: 1000 SEK per leg (Avanza minimum courtage threshold)". P0 policy violation; codex missed.

[CONFIRM-MISSED] portfolio/kelly_sizing.py:326 — `if rec_sek < 500: rec_sek = 0` uses 500 not 1000 | kelly_sizing.py:326 hardcodes 500 SEK floor. Same project-rule violation as above; Kelly will recommend 500-999 SEK trades that fail Avanza minimum. P0 policy violation; codex missed.

[CONFIRM-MISSED] portfolio/risk_management.py:367-369 — ATR stop placement has zero awareness of warrant knockout barriers | risk_management.py:367-369 `atr_pct = min(atr_pct, 15.0); stop_price = entry_price * (1 - 2 * atr_pct/100)`. No barrier lookup, no MEMORY-mandated 3% safety zone. CLAUDE.md memory/feedback_mini_stoploss.md flags this as a critical user-feedback rule. P0 violation; codex missed entirely.

[CONFIRM-MISSED] portfolio/risk_management.py:233 — Corrupt portfolio file silently bypasses circuit breaker | risk_management.py:233 `portfolio = load_json(portfolio_path, default={})` returns empty dict on corruption → `holdings = {}` → cash-only path returns clean drawdown. portfolio_mgr.py:90 has the CRITICAL log + backup recovery via `_load_state_from`, but check_drawdown imports `load_json` raw. P0 — codex missed; the Layer 2 circuit breaker becomes blind to corrupt-file scenarios that the portfolio_mgr path explicitly handles.

[CONFIRM-MISSED] portfolio/equity_curve.py:495 — Flat (`pnl_pct == 0.0`) round-trip classification inconsistent | equity_curve.py:494-495 `wins = [t for t in trips if t["pnl_pct"] > 0]; losses = [t for t in trips if t["pnl_pct"] <= 0]`. Streak counter at line 511-519 uses `t["pnl_pct"] > 0` for wins, ELSE = loss. A pnl_pct=0 round trip counts as a loss in both streaks AND in the win_loss_ratio losses. Consistent at 0 actually — both treat 0 as loss. CodeX missed but on re-read claude's claim about "excluded from losses for win_loss_ratio" is wrong; both branches use `<= 0`. PARTIAL — bug exists in spirit (zero should be neutral), but the inconsistency claude cited isn't there.

[CONFIRM-MISSED] portfolio/monte_carlo.py:62-65 — `volatility_from_atr` hardcodes daily-scale annualization | monte_carlo.py:62-65 `annual_factor = math.sqrt(252.0/period)` assumes ATR period units are days. If ATR comes from hourly candles (system primary timeframe per CLAUDE.md), volatility is understated by sqrt(24)≈4.9x. monte_carlo_risk.py:449 calls this directly. Codex missed P1.

[CONFIRM-MISSED] portfolio/cost_model.py:73-78 — `STOCK_COSTS.min_fee_sek=1.0` understates Avanza floor | cost_model.py:73-78 sets stock min fee to 1 SEK; Avanza Mini courtage is ~39 SEK minimum on most tiers. Real-world cost model is wildly off. Codex missed P1.

[CONFIRM-MISSED] portfolio/warrant_portfolio.py:182-214 — `record_warrant_transaction` has no `reason` field | warrant_portfolio.py:200-208 builds the txn dict with no reason, violating the project rule (portfolio-risk.md: "Log every trade with a reason in the transaction record"). Codex missed P0 policy violation.

[CONFIRM-MISSED] portfolio/warrant_portfolio.py:215 — `record_warrant_transaction` not under lock | warrant_portfolio.py:198-265 reads `state = load_warrant_state()` and writes via `save_warrant_state(state)` with no lock around the read-modify-write. portfolio_mgr.py uses `_get_lock(path)` + `update_state` for the patient/bold portfolios; warrants don't. Concurrent metals-loop fast-tick + main loop could lose units. Codex missed P0.

[CONFIRM-MISSED] portfolio/kelly_sizing.py:269-323 — Kelly uses `cash_sek` instead of total portfolio value | kelly_sizing.py:269-270 `alloc_frac = 0.30 if strategy == "bold" else 0.15; max_alloc = cash_sek * alloc_frac`. risk_management.check_concentration_risk:751-757 correctly uses total_value (cash + holdings). The two subsystems disagree on the sizing base. Codex missed P0.

[CONFIRM-MISSED] portfolio/risk_management.py:43-110 — `_streaming_max` byte-offset cache vulnerable to file rewrite | risk_management.py:73 `if file_size >= cached["offset"]` — file truncated and rewritten to similar size passes this check, seek skips the new content, peak stuck. mtime/hash check missing. Codex missed P1.

End: CONFIRM=15 DISPUTE=0 PARTIAL=4 UNVERIFIED=0 MISSED=11
