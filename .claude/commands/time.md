Get the current time, date, and day of the week. Also show which markets are currently open.

## Steps

1. Run: `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`

2. Based on the time (CET), report which markets are open RIGHT NOW:
   - **Crypto** (BTC, ETH): 24/7
   - **US Stocks** (NASDAQ/NYSE): 15:30-22:00 CET (note: DST-dependent)
   - **Avanza warrants**: 08:15-21:55 CET (check todayClosingTime via API if critical)
   - **WTI electronic**: nearly 24h (Sun 17:00 - Fri 16:00 CT)
   - **NYMEX pit**: 15:00-20:30 CET

3. Output concisely:
   ```
   {Day} {date} {time} CET
   Markets: Crypto OPEN | US Stocks {OPEN/CLOSED} ({time to close/open}) | Avanza {OPEN/CLOSED} | Oil {OPEN/CLOSED}
   ```
