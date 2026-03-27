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


## Avanza Trading API

When the user asks to place orders, check positions, or manage trades, use these functions from `portfolio.avanza_session`:

```python
from portfolio.avanza_session import (
    get_quote,           # get_quote("1069606") -> {buy, sell, last, changePercent}
    get_buying_power,    # get_buying_power() -> {buying_power, total_value, own_capital}
    get_positions,       # get_positions() -> [{name, volume, value, account_id, ...}]
    place_buy_order,     # place_buy_order("1069606", price=0.86, volume=5000) -> {orderRequestStatus, orderId}
    place_sell_order,    # place_sell_order("1069606", price=1.05, volume=5000) -> same
    cancel_order,        # cancel_order("865451335") -> {orderRequestStatus}
    api_get,             # api_get("/_api/trading/rest/orders") -> list open orders
)
# Stop-losses: api_get("/_api/trading/stoploss") to list
# Open orders: api_get("/_api/trading/rest/orders") to list
```

**Key rules:**
- Default account: `1625505` (ISK). Only use available cash.
- Sell + stop-loss volume must NOT exceed position size (Avanza blocks it as short-selling).
- Cancel orders uses POST not DELETE: `cancel_order(order_id)`.
- Stop-loss payload is nested — see `data/metals_avanza_helpers.py:place_stop_loss()` for format.
- Also works without Playwright via `portfolio.avanza_client` when TOTP credentials are configured.
