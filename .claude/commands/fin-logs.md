Check recent logs for errors, crashes, and issues. Read-only diagnostic.

## Steps (run all inspections in parallel where possible)

1. **Loop output** — Read last 100 lines of `data/loop_out.txt`. Look for:
   - Python exceptions / tracebacks
   - "ERROR" or "WARNING" log lines
   - Crash restart markers ("Restarting in..." or "CRASH ALERT")
   - API failures (Binance, Alpaca, Avanza, Telegram)
   - Stale data warnings

2. **Agent log** — Read last 50 lines of `data/agent.log`. Look for:
   - Layer 2 invocation attempts and completions
   - Timeout failures (session died before finishing)
   - "nested session" errors (CLAUDECODE env var leak)
   - Tier classification (T1/T2/T3) distribution

3. **Metals loop output** — Read last 50 lines of `data/metals_loop_out.txt` (if exists). Look for:
   - Swing trader errors
   - Order placement failures
   - Stop-loss API errors (especially the dangerous regular-order-as-stoploss bug)

4. **Health state** — Read `data/health_state.json` for:
   - Error counts by type
   - Module failure tracking
   - Heartbeat staleness

5. **Crash history** — Search `data/telegram_messages.jsonl` for recent messages containing "CRASH" or "ERROR" (last 20 lines).

6. **Signal accuracy check** — Run `.venv/Scripts/python.exe -u portfolio/main.py --accuracy` to see if any signals have degraded below 40% (inversion candidates).

## Output format

```
## Log Report — {date} {time} CET

### Loop Health
Status: {running/stopped}
Last heartbeat: {timestamp} ({age} ago)
Errors in last 24h: {count}
{list any ERROR/WARNING lines with timestamps}

### Layer 2 Agent
Last invocation: {timestamp}
Last completion: {timestamp}
Success rate (last 10): {X}/10
{list any failures with reason}

### Metals Loop
{status and any errors}

### Signal Accuracy Alerts
{any signals below 40% accuracy with 20+ samples — these should be investigated}
{any signals recently inverted (sub-50%) that weren't before}

### Issues Found
{numbered list of problems, sorted by severity}
1. [CRITICAL] ...
2. [WARNING] ...
3. [INFO] ...

{if no issues: "No issues found in recent logs."}
```

Focus on actionable problems. Don't list every normal log line — only flag anomalies.
