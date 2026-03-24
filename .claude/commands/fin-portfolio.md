Show the current state of all portfolios — Patient, Bold, and Warrants. Read-only inspection.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`

2. **Read portfolio states** — Read all three files in parallel:
   - `data/portfolio_state.json` — Patient strategy
   - `data/portfolio_state_bold.json` — Bold strategy
   - `data/portfolio_state_warrants.json` — Warrant holdings

3. **Read latest prices** — Read `data/agent_summary_compact.json` for current prices of held instruments to calculate live P&L. Use the `price_usd` and `fx_rate` fields.

4. **Read equity curve** — Read last 10 lines of `data/layer2_journal.jsonl` for recent portfolio value snapshots and decisions.

5. **Calculate for each portfolio:**
   - Cash position (SEK)
   - Per-holding: ticker, shares, avg_cost, current_price, unrealized P&L (SEK + %)
   - Total portfolio value (cash + holdings at current prices)
   - Total return from 500K starting capital (SEK + %)
   - Total fees paid (cumulative drag)
   - Number of trades executed

6. **Recent transactions** — Show last 5 transactions per strategy with date, ticker, action, shares, price, P&L.

## Output format

```
## Portfolio Report — {date} {time} CET

### Patient Strategy (conservative)
Cash: {cash_sek} SEK
Holdings:
  {TICKER}: {shares} sh @ {avg_cost} → {current_price} = {unrealized_pnl} SEK ({pct}%)
  ...
Total Value: {total} SEK ({return_pct}% from 500K)
Fees Paid: {fees} SEK
Trades: {count}

### Bold Strategy (breakout)
Cash: {cash_sek} SEK
Holdings:
  {TICKER}: {shares} sh @ {avg_cost} → {current_price} = {unrealized_pnl} SEK ({pct}%)
  ...
Total Value: {total} SEK ({return_pct}% from 500K)
Fees Paid: {fees} SEK
Trades: {count}

### Warrants
  {NAME}: {units} units @ {avg_cost} → {current_price} = {pnl} SEK ({pct}%)
  ...

### Recent Trades
Patient:
  {date} {action} {ticker} {shares}sh @ {price} — {reason}
Bold:
  {date} {action} {ticker} {shares}sh @ {price} — {reason}

### Performance Comparison
| Metric     | Patient | Bold  |
|------------|---------|-------|
| Total P&L  |         |       |
| Return %   |         |       |
| Win Rate   |         |       |
| # Trades   |         |       |
| Fees       |         |       |
```

Use exact numbers from the portfolio files. Do NOT approximate the math — follow the calculation formulas from CLAUDE.md exactly.
