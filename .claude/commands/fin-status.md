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
