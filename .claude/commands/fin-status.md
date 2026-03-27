Perform a full system health check of the finance-analyzer trading agent. Read-only inspection — do NOT modify anything.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"` to establish current day/time (CET). Determine which markets are open right now:
   - Crypto/metals: 24/7
   - Avanza warrants: 08:15-21:55 CET
   - Swedish equities: 09:00-17:25 CET
   - US stocks: 15:30-22:00 CET

2. **Loop process** — Run `powershell.exe -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet64, StartTime, CommandLine | Format-Table -AutoSize"` to check if the data loop (`pf-loop.bat` / `main.py`) is running. Two python processes per loop is normal (venv launcher stub).

3. **Loop output** — Read the last 50 lines of `data/loop_out.txt` for recent activity and errors.

4. **Health state** — Read `data/health_state.json` for heartbeat, error counts, module failures, and agent silence detection.

5. **Trigger state** — Read `data/trigger_state.json` to see last trigger times and what fired recently.

6. **Layer 2 agent log** — Read the last 30 lines of `data/agent.log` to check if the Layer 2 agent is being invoked and completing successfully. Look for errors, timeouts, or "nested session" failures.

7. **Recent Telegram messages** — Read the last 5 lines of `data/telegram_messages.jsonl` to see what the user last received.

8. **Scheduled tasks** — Run `powershell.exe -NoProfile -Command "Get-ScheduledTask | Where-Object {$_.TaskName -like 'PF-*'} | Select-Object TaskName, State, LastRunTime, LastTaskResult | Format-Table -AutoSize"` to verify task scheduler state.

9. **Singleton locks** — Check if `data/metals_loop.singleton.lock` exists (stale lock = dead process).

## Output format

```
## System Status — {day} {date} {time} CET

### Markets
{which are open/closed right now}

### Loop
{running/stopped, PID, uptime, last heartbeat age}

### Layer 2 Agent
{last invocation time, last completion, success/fail, any errors}

### Scheduled Tasks
{table of PF-* tasks and their state}

### Recent Activity
{last 3 Telegram messages summary — timestamp + first line}

### Issues
{any problems found: stale heartbeat, errors, missing processes, failed tasks}
- If no issues: "All systems nominal."
```

Keep it scannable. Flag anything that needs attention.


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
