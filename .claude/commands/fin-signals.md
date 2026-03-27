Run a fresh signal report and display current consensus for all instruments. Read-only — do NOT trade.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"` to know which markets are active.

2. **Run signal report** — Execute: `.venv/Scripts/python.exe -u portfolio/main.py --report`
   This refreshes all 30 signals across all 19 instruments. Wait for completion (may take 1-2 min).

3. **Read compact summary** — Read `data/agent_summary_compact.json` for the full picture:
   - Per-ticker: consensus, confidence, weighted_confidence, vote breakdown, regime, RSI, MACD, volume ratio
   - Timeframe heatmaps (Now through 6mo)
   - Macro context: DXY, treasury yields, F&G, FOMC proximity
   - Focus probabilities (Mode B): directional probabilities at 3h/1d/3d for XAG-USD and BTC-USD
   - Signal accuracy stats

4. **Read prophecy** — Read `data/prophecy.json` for active beliefs and checkpoint status.

5. **Read notification config** — Read `config.json` → `notification.mode` and `focus_tickers` to know which format to use.

## Output format

Use the notification mode from config:

### Mode B (probability) — for focus tickers (XAG-USD, BTC-USD by default):
```
{TICKER}  ${price}  ↑/↓{pct}% 3h  ↑/↓{pct}% 1d  ↑/↓{pct}% 3d
  accuracy: {pct}% 1d ({samples} sam) | 7d: {cumulative_gain}%
  regime: {regime} | RSI: {rsi} | MACD: {macd}
  signals: {vote_breakdown} — {top BUY signals} vs {top SELL signals}
  heatmap: {7-char BHS heatmap Now→6mo}
```

### All tickers (Mode A grid):
```
{TICKER}  ${price}  {consensus}  {XB/YS/ZH}  {7-char heatmap}
```

### Summary sections:
```
## Signal Report — {date} {time} CET

### Focus Instruments (Mode B)
{rich format for focus tickers}

### All Instruments
{grid for remaining tickers, sorted: BUY first, then SELL, then HOLD}

### Macro Context
DXY: {value} ({5d_change}) | 10Y: {yield} | 2s10s: {spread} | F&G: {crypto}/{stock} | FOMC: {days}d

### Prophecy Check
{active beliefs with progress toward targets}

### Top Signals (by accuracy)
{top 5 most accurate signals with sample count}

### Actionable
{any tickers with BUY/SELL consensus — highlight these}
{if none: "No actionable signals. All HOLD."}
```

Be concise. Prices rounded aggressively ($68K, $426, $32, $1,949). Use the monospace ticker grid format from CLAUDE.md.


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
