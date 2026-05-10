# Claude critique of codex findings — portfolio-risk

## Verdicts

- [P1] Match sells to their own entry lots in Kelly trade stats — portfolio/kelly_sizing.py:90-104
  Verdict: CONFIRMED
  Reason: Lines 90-95 compute single weighted average buy price across all BUYs; lines 97-104 apply that same basis to every SELL, mismatching realized P&L for multi-round-trip histories.

- [P1] Compute stop-hit probability from path crossings — portfolio/monte_carlo.py:205-219, 328
  Verdict: CONFIRMED
  Reason: `probability_below()` uses only terminal prices from `_terminal_prices`, not barrier crossing events. Stop-hit probability at line 328 misses any path that crosses the barrier and rebounds.

- [P1] Record only opening buys in the position-rate guard — portfolio/trade_guards.py:290-296, 189-229
  Verdict: CONFIRMED
  Reason: Line 296 appends every BUY to `new_position_timestamps` without checking ticker already held; Guard 3 (lines 189-229) counts all entries as "new positions", allowing scale-ins to block further adds.

- [P1] Floor leveraged warrant value at zero — portfolio/warrant_portfolio.py:100-103
  Verdict: CONFIRMED
  Reason: Line 100 computes `current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)` with no floor; leveraged drops worse than -100%/leverage produce negative values (e.g., 5x product down 30% yields -5% P&L, making value negative).

- [P2] Score concentration from post-trade exposure, not current exposure — portfolio/trade_risk_classifier.py:105-111
  Verdict: CONFIRMED
  Reason: Lines 105-111 score concentration based on `existing_exposure_pct` only; a 10% BUY on 24% existing (post-trade 34%) scores zero, while SELLs reducing exposure get penalized.

- [P2] Use total portfolio value for drawdown probabilities — portfolio/monte_carlo_risk.py:368-388, 395-419
  Verdict: CONFIRMED
  Reason: Lines 381-382 compute `total_value = sum(shares*price_usd)` from invested positions only; line 387 thresholds drawdown against this subset, ignoring cash_sek from portfolio_state entirely. 20% invested portfolio with 5% holdings loss incorrectly reports 5% drawdown, not 1%.

## New findings (mine)

- [P1] Warrant P&L P&L scaling sign error — portfolio/warrant_portfolio.py:96
  Line 96 computes `implied_pnl_pct = underlying_change * leverage`, where `underlying_change` is already a ratio (line 92). This is correct. No new issue.

- [P2] Kelly sizing falls back to all-ticker average when per-ticker data insufficient — portfolio/kelly_sizing.py:291-295
  If `_compute_trade_stats(transactions, ticker=ticker)` returns None (line 292), the code retries with `ticker=None` (line 294), blending all-ticker wins/losses with single-ticker signal accuracy. For a ticker with poor cross-ticker average (e.g. XAG 55% signal, but all tickers average 45% win/loss), this produces undersized Kelly bets. Not a bug per se, but undocumented fallback path that may diverge from stated "ticker-specific Kelly."

## Summary
- Confirmed: 6
- Partial: 0
- False-positive: 0
- New: 1 (low severity, fallback behavior)

All 6 P-tier findings match the code. Additional finding is a behavioral quirk (kelly sizing fallback to all-ticker stats when per-ticker insufficient), not a bug, but worth documenting.
