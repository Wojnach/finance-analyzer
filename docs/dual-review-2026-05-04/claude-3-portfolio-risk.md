# Claude Review — portfolio-risk

## P0 (money-losing or data-corrupting)

- `portfolio/monte_carlo.py:63` — `volatility_from_atr` hardcodes 252-day annualization regardless of candle period
  ```python
  def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
      atr_frac = atr_pct / 100.0
      annual_factor = math.sqrt(252.0 / period)   # 252 hardcoded
      vol = atr_frac * annual_factor
  ```
  `atr_pct` flowing into `simulate_ticker()` comes from whichever timeframe's DataFrame is primary. For 1h candles ("12h" label), correct annual_factor is `sqrt(8760/14) ≈ 25.0`, not `sqrt(252/14) ≈ 4.24`. **Underestimates annualized vol by ~5.9x.** All MC stop-hit probabilities (`p_stop_hit_1d`, `p_stop_hit_3d`) and `compute_portfolio_var` come out far too tight — system systematically understates tail risk. Fix: pass candle period through `simulate_ticker`. Confidence 88.

- `portfolio/monte_carlo_risk.py:426` — `compute_portfolio_var` uses raw `agent_summary.get("fx_rate", 10.0)` instead of `_resolve_fx_rate()`
  ```python
  fx_rate = agent_summary.get("fx_rate", 10.0)
  ```
  `risk_management.py` was previously fixed to use `_resolve_fx_rate()`; `monte_carlo_risk.py` still uses bare `.get()`. Fallback 10.0 is close to typical USD/SEK but if `agent_summary` is empty, VaR in SEK off by ~8% (10.0 vs 10.85). Confidence 92.

- `portfolio/warrant_portfolio.py:97` — `pnl_pct` returned as percentage (25.0) but `pnl_sek` calculation uses fraction (0.25) — easy for downstream to misuse
  ```python
  implied_pnl_pct = underlying_change * leverage      # fraction, e.g. 0.25
  implied_pnl_pct_rounded = round(implied_pnl_pct * 100, 2)  # e.g. 25.0
  return {"pnl_pct": implied_pnl_pct_rounded, "pnl_sek": pnl_sek, ...}
  ```
  Internally consistent but a downstream that feeds `pnl_pct` into Kelly as a fraction is 100x off. Trace any pipeline using `pnl["pnl_pct"]` for sizing math. Confidence 85.

## P1 (high-confidence bugs)

- `portfolio/kelly_sizing.py:91-95` — `_compute_trade_stats` averages ALL buy prices globally (not FIFO), giving wrong P&L for multi-leg positions
  ```python
  total_shares_bought = sum(b.get("shares", 0) for b in buys)
  total_cost = sum(b.get("total_sek", 0) for b in buys)
  avg_buy_price = total_cost / total_shares_bought
  ```
  Buy 80 → sell → buy 100 → sell at 95: second sell looks like loss vs blended avg of 90 instead of true loss vs 100. Distorts win_rate, avg_win_pct, avg_loss_pct → wrong Kelly. `equity_curve._pair_round_trips()` has correct FIFO; Kelly should call it. Confidence 83.

- `portfolio/risk_management.py:374` — ATR stop level anchored to `entry_price`, not current price
  ```python
  stop_price = entry_price * (1 - 2 * atr_pct / 100)
  ```
  ATR is current; entry is potentially old. Position +20% from entry with 3% ATR → stop at `entry*0.94` is far below where current-price-based stop would be. Gives up 26% of unrealized gains. Same bug in `check_atr_stop_proximity()` line 909. Confidence 81.

- `portfolio/equity_curve.py:492-493` — `profit_factor` uses net SEK (post-fee) but `win_rate`/`expectancy` use pct (gross, pre-fee) — inconsistent basis
  ```python
  wins = [t for t in trips if t["pnl_pct"] > 0]      # gross
  losses = [t for t in trips if t["pnl_pct"] <= 0]
  gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)  # net of fees
  gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))
  ```
  Trade with `pnl_pct > 0` but `pnl_sek < 0` (fees ate gain) counts as win for win_rate, loss for profit_factor. `expectancy_pct` overstates net edge; Kelly inputs derived from gross overstate the fraction. For 1% round-trip warrant fees this is material. Confidence 80.

- `portfolio/trade_validation.py:76` — TOCTOU on cash check between concurrent Layer 2 invocations
  ```python
  cash_pct = (order_value / cash_available) * 100
  if cash_pct > max_cash_pct:
      return ValidationResult(False, ...)
  ```
  Two L2 subprocesses (T1 + T2 can overlap) both read same `cash_sek` at time T, both pass 50% check, both commit — total 100% of cash. `update_state()` lock prevents save race but `validate_trade` runs before the atomic update with stale snapshot. No reserved-cash debit pattern. Confidence 80.

## P2 (concerns / smells)

- `portfolio/kelly_metals.py:215-217` — `cert_loss_frac = 0` when `avg_loss_pct == 0` → silent zero position with no warning
  ```python
  cert_loss_frac = avg_loss * leverage / 100.0
  if cert_loss_frac > 0:
      position_fraction = half_kelly / cert_loss_frac
  else:
      position_fraction = 0.0   # silent
  ```
  When ticker has no losses in history (or all tied at exactly 0), zero recommended size. Caller gets `position_sek = 0` with no explanation. Log a WARNING.

- `portfolio/portfolio_validator.py:236-243` — `avg_cost_usd` validation sums ALL historical buys, including already-sold shares
  ```python
  for tx in transactions:
      if tx.get("ticker") != ticker or tx.get("action") != "BUY":
          continue
      total_cost += tx_shares * tx_price
      total_bought += tx_shares
  expected_avg = total_cost / total_bought
  ```
  After partial sell-and-rebuy, expected_avg reflects all-time blended avg; actual holdings track only current open lot. Legitimate divergence > 1% causes false validation failures.

- `portfolio/risk_management.py:461-462` — same `sqrt(252/14)` annualization bug, inlined here
  ```python
  vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)
  ```
  For 15m or 1h ATRs, stop-hit probability estimates from `simulate_intraday_paths` too low.

- `portfolio/warrant_portfolio.py:259-261` — SELL with `units > current holding` silently clamps to 0 (deletes position) without logging
  ```python
  remaining = existing.get("units", 0) - units
  if remaining <= 0:
      del holdings[config_key]    # no warning
  ```
  State drift, manual intervention, or partial fill not recorded → silent 0 with phantom-units sold. Reconciliation gap. Log WARNING and clamp to actual.

- `portfolio/equity_curve.py:550` — Calmar uses `max_dd` as fraction (0.07) and `annualized` as fraction (0.12); mathematically correct but `compute_metrics()` returns both in pct space → trap for consumers.

## Did NOT find

1. Raw `json.dump`/`open(...).read()` calls — all use atomic_write_json/load_json.
2. Sign errors on long P&L — `portfolio_value` and validator cash recon are correct.
3. Stop-loss API misuse — these files compute levels analytically, no direct stop-loss API calls.
4. Stop-loss within 3% of bid — `compute_stop_levels` uses 2x ATR (ATR cap 15% → min 30% below entry).
5. Patient/Bold portfolio_state race — per-file lock with file path as key.
6. Round-trip P&L double-count — FIFO in `_pair_round_trips` matches each share once.
7. NaN/Inf bypassing drawdown CB — `check_drawdown` has `math.isfinite` guard, fail-safe to 100%.
8. Kelly formula error — `kelly_fraction()` correctly implements `f* = (p*b - q) / b`.
