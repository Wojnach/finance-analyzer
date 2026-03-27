Metals fishing plan -- compute optimal dip-buy AND peak-short levels for silver and gold certificates.

Supports BOTH directions:
- **BULL fishing**: buy BULL cert when underlying dips, sell on bounce (RSI < 45)
- **BEAR fishing**: buy BEAR cert when underlying spikes, sell on reversal (RSI > 65)

Uses AVA-issued daily leverage certificates (5x, zero commission, tight 0.5% spread).
No barrier/knockout risk. Intraday only (3-5h max hold).

## Steps

1. **Check the time** -- Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Avanza commodity cert hours: ~08:15-21:55 Swedish local time (market maker hours).
   **WARNING:** The API field `marketPlace.todayClosingTime` shows exchange hours (e.g. 17:30),
   NOT the market maker hours. AVA certs remain tradeable as long as the underlying trades.
   Verify by checking if bid/ask are present in `get_quote()`. No new entries after 18:55.

2. **Run the fishing script**:
   ```
   .venv/Scripts/python.exe scripts/fin_fish.py --metals silver,gold
   ```

   Optional flags:
   - `--hours N` -- override planning horizon (default: auto-compute from session)
   - `--budget N` -- SEK per fishing level (default: 20000)
   - `--max-levels N` -- max rows to show per direction (default: 8)

3. **Interpret the results**:

   The script shows TWO tables per metal -- BULL (fish dips) and BEAR (fish peaks):
   - **Level**: the underlying price trigger
   - **Move%**: how far the underlying must move from current price
   - **Fill%**: probability this level gets hit within the session
   - **Gross**: raw certificate gain (move% x leverage)
   - **Spread**: bid-ask spread cost (AVA ~0.5%, SG ~2.5%)
   - **Net%**: profit after spread = gross - spread
   - **EV/SEK**: expected value = fill_prob x net_gain_sek

4. **Direction logic** (automatic):
   - RSI < 45: BULL fishing favored (oversold, dips likely to bounce)
   - RSI > 65: BEAR fishing favored (overbought, peaks likely to fade)
   - RSI 45-65: neutral zone, both directions shown but lower conviction
   - Chronos 24h positive: favors BULL. Negative: favors BEAR.

5. **Instrument selection** (automatic, from `data/fin_fish_config.py`):
   - BULL SILVER X5 AVA 3 (id 1069606) -- fish silver dips, 5x, 0% commission
   - BEAR SILVER X5 AVA 12 (id 2286417) -- fish silver peaks, 5x, 0% commission
   - BULL GULD X5 AVA (id 738811) -- fish gold dips, 5x, 0% commission
   - BEAR GULD X5 VON4 (id 1047859) -- fish gold peaks, 5x, 2.2% spread (no AVA 5x bear gold exists)
   - BEAR GULD X2 AVA (id 738805) -- gold peaks fallback, 2x only, 0.5% spread

6. **Cascading exit plan** (shown for each level):
   Take-profit: TP1 +1.5% sell 40%, TP2 +2.5% sell 40%, TP3 +4.0% sell 20%
   Stop-loss: **-3% underlying minimum** (-15% cert for 5x leverage). Do NOT set tighter.
   Previous -1% underlying / -5% cert stops triggered on normal intraday wicks repeatedly.
   Time: 3h tighten stop, 5h force sell, 21:00 CET force sell everything

7. **If Avanza unavailable** -- still runs with Binance data, sends Telegram with plan.

## Critical rules
- **AVA certificates preferred** -- zero commission, 0.5% spread. Avoid SG (2.5% spread).
- **5x leverage default.** 3x available as safer option.
- **Max hold: 3-5h.** Daily certs reset overnight, MUST exit same day.
- **No new entries after 18:55 CET.** Cancel all at 21:00 CET.
- **Oil is the signal** -- Brent spike = metals dip = fish BULL. Brent drop = fish BEAR.
- **Daily range gate** -- if avg range < 3%, fishing EV is negative, don't fish.


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
