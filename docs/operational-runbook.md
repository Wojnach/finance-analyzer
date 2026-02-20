# Portfolio Intelligence — Operational Runbook

> **Last updated:** 2026-02-20
> **System:** herc2 (Windows 11 Pro), Python 3.12, Claude Code Layer 2

## Table of Contents

1. [System Overview](#system-overview)
2. [Scheduled Tasks](#scheduled-tasks)
3. [Starting and Stopping](#starting-and-stopping)
4. [Restart Procedures](#restart-procedures)
5. [Layer 2 (Claude Code) Debugging](#layer-2-debugging)
6. [Silent Failure Investigation](#silent-failure-investigation)
7. [Signal Reports and Accuracy](#signal-reports-and-accuracy)
8. [ML Retraining](#ml-retraining)
9. [Manual Layer 2 Invocation](#manual-layer-2-invocation)
10. [Common Failure Modes](#common-failure-modes)
11. [Log File Locations](#log-file-locations)
12. [Dashboard](#dashboard)

---

## System Overview

The system runs as two layers:

- **Layer 1 (Python fast loop):** Runs continuously via `pf-loop.bat`. Collects data every
  60 seconds, computes 25 signals across 7 timeframes for 31 tickers, detects trigger
  conditions, and invokes Layer 2 when something meaningful changes.
- **Layer 2 (Claude Code agent):** Invoked by Layer 1 via subprocess (`claude -p`). Reads
  all signal data, makes independent trading decisions for Patient and Bold portfolios,
  sends Telegram notifications.

Both layers run on herc2 at `Q:\finance-analyzer\`.

---

## Scheduled Tasks

All tasks are managed via Windows Task Scheduler.

| Task name         | Status   | Schedule              | Action                                         |
| ----------------- | -------- | --------------------- | ---------------------------------------------- |
| **PF-DataLoop**   | ENABLED  | On logon (continuous) | `scripts\win\pf-loop.bat` — Layer 1 fast loop  |
| **PF-Dashboard**  | ENABLED  | On logon (continuous) | Flask dashboard on port 5055                    |
| **PF-OutcomeCheck** | ENABLED | Daily 18:00 local     | `--check-outcomes` — backfill signal accuracy   |
| **PF-MLRetrain**  | ENABLED  | Weekly                | `--retrain` — retrain ML classifier             |
| **PF-Loop**       | DISABLED | —                     | Redundant with PF-DataLoop, caused duplicates   |
| **Portfolio-Agent** | DISABLED | —                    | Bypassed trigger system, caused 15-min HOLD spam |

### Viewing task status

```cmd
schtasks /query /tn "PF-DataLoop" /v /fo LIST
schtasks /query /tn "PF-Dashboard" /v /fo LIST
schtasks /query /tn "PF-OutcomeCheck" /v /fo LIST
```

### Checking for duplicate processes

```powershell
powershell.exe -NoProfile -Command "Get-Process python* | Select-Object Id, ProcessName, StartTime, Path"
```

Two python processes per loop is normal (Windows venv launcher stub re-spawns as system Python).

---

## Starting and Stopping

### Start the data loop

```cmd
cd /d Q:\finance-analyzer
scripts\win\pf-loop.bat
```

Or via Task Scheduler:

```cmd
schtasks /run /tn "PF-DataLoop"
```

### Stop the data loop

Find and kill the Python process:

```powershell
powershell.exe -NoProfile -Command "Get-Process python* | Where-Object { $_.CommandLine -like '*main.py*--loop*' } | Stop-Process -Force"
```

Or kill the cmd window running `pf-loop.bat`.

### Start the dashboard

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe dashboard\app.py
```

### Stop the dashboard

```powershell
powershell.exe -NoProfile -Command "Get-Process python* | Where-Object { $_.CommandLine -like '*app.py*' } | Stop-Process -Force"
```

---

## Restart Procedures

### Full system restart

1. Stop all Python processes:
   ```powershell
   powershell.exe -NoProfile -Command "Get-Process python* | Stop-Process -Force"
   ```
2. Wait 5 seconds for clean shutdown.
3. Start the data loop:
   ```cmd
   schtasks /run /tn "PF-DataLoop"
   ```
4. Start the dashboard:
   ```cmd
   schtasks /run /tn "PF-Dashboard"
   ```
5. Verify Layer 1 is running by checking `data/loop_out.txt` for recent timestamps.
6. Verify dashboard at `http://localhost:5055`.

### Layer 1 restart (data loop only)

The `pf-loop.bat` script has auto-restart with a 30-second delay. If the Python process
crashes, it will restart automatically. To force a restart:

1. Kill the existing process.
2. Wait for `pf-loop.bat` to auto-restart (30s), OR manually run `schtasks /run /tn "PF-DataLoop"`.

### After a reboot

Both PF-DataLoop and PF-Dashboard are configured to start on logon. After logging into
herc2, verify both are running:

```powershell
powershell.exe -NoProfile -Command "Get-Process python* | Select-Object Id, ProcessName, StartTime"
```

---

## Layer 2 Debugging

### Check if Layer 2 is being invoked

```cmd
type Q:\finance-analyzer\data\invocations.jsonl | findstr /c:"2026-02-20"
```

Or view the last few entries:

```cmd
powershell.exe -NoProfile -Command "Get-Content Q:\finance-analyzer\data\invocations.jsonl -Tail 5"
```

### Check Layer 2 agent output

```cmd
type Q:\finance-analyzer\data\agent.log
```

Or the last 50 lines:

```cmd
powershell.exe -NoProfile -Command "Get-Content Q:\finance-analyzer\data\agent.log -Tail 50"
```

### Check if Layer 2 is sending Telegram messages

```cmd
powershell.exe -NoProfile -Command "Get-Content Q:\finance-analyzer\data\telegram_messages.jsonl -Tail 5"
```

### Common Layer 2 issues

1. **"Nested session" error:** The `CLAUDECODE` environment variable is set, preventing
   `claude -p` from starting. Fix: ensure `pf-loop.bat` clears it (`set CLAUDECODE=`) and
   `invoke_agent()` in `main.py` strips it from the environment.

2. **Agent timeout (600s):** If Claude Code hangs, the main loop kills it via `taskkill /F /T`
   after 600 seconds. Check `data/agent.log` for partial output.

3. **Agent not invoked:** Check trigger state in `data/trigger_state.json`. Verify that
   triggers are actually firing by checking `data/loop_out.txt` for trigger log lines.

---

## Silent Failure Investigation

When the system appears to be running but is not producing output:

### Step 1: Check if Layer 1 is running

```cmd
powershell.exe -NoProfile -Command "Get-Content Q:\finance-analyzer\data\loop_out.txt -Tail 20"
```

Look for recent timestamps. If timestamps are stale (>5 minutes old during market hours),
Layer 1 has crashed.

### Step 2: Check for errors in Layer 1

```cmd
powershell.exe -NoProfile -Command "Select-String -Path Q:\finance-analyzer\data\loop_out.txt -Pattern 'Error|Exception|Traceback' -Context 0,3 | Select-Object -Last 10"
```

### Step 3: Check if triggers are firing

```cmd
powershell.exe -NoProfile -Command "Get-Content Q:\finance-analyzer\data\trigger_state.json" | python -m json.tool
```

### Step 4: Check agent_summary.json freshness

```cmd
powershell.exe -NoProfile -Command "(Get-Item Q:\finance-analyzer\data\agent_summary.json).LastWriteTime"
```

If this timestamp is stale, Layer 1 is not writing data.

### Step 5: Check for CLAUDECODE env var issue

```cmd
powershell.exe -NoProfile -Command "Select-String -Path Q:\finance-analyzer\data\agent.log -Pattern 'nested' | Select-Object -Last 5"
```

This was the cause of a 34-hour Layer 2 outage on Feb 18-19, 2026.

### Step 6: Check disk space

```cmd
powershell.exe -NoProfile -Command "Get-PSDrive Q"
```

JSONL files grow unbounded. If disk is full, Layer 1 will fail to write data.

---

## Signal Reports and Accuracy

### Run a one-shot signal report

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\main.py --report
```

This runs Layer 1 once (no loop), computes all signals, and prints a summary. Does not
invoke Layer 2. Useful for verifying signal computation is working.

### Check signal accuracy

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\main.py --accuracy
```

Prints per-signal hit rates at 1d/3d/5d/10d horizons. Requires outcome data to have been
backfilled by PF-OutcomeCheck.

### Manually backfill outcomes

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\main.py --check-outcomes
```

Reads `data/signal_log.jsonl`, looks up historical prices at each horizon timestamp, and
writes actual price outcomes back into the log. This is normally run daily by PF-OutcomeCheck.

### Check Fear & Greed

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\fear_greed.py
```

### Check sentiment

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\sentiment.py
```

---

## ML Retraining

### Trigger manual retrain

```cmd
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\main.py --retrain
```

This retrains the HistGradientBoosting ML classifier on fresh 1h Binance candle data.
Normally runs weekly via PF-MLRetrain scheduled task.

### Check training data freshness

```cmd
powershell.exe -NoProfile -Command "(Get-Item Q:\finance-analyzer\data\ml_model_btc.pkl).LastWriteTime"
powershell.exe -NoProfile -Command "(Get-Item Q:\finance-analyzer\data\ml_model_eth.pkl).LastWriteTime"
```

### Verify model is loaded

Check `data/agent_summary.json` for the `ml` signal — if it shows "HOLD" with no indicators,
the model may not be loading correctly.

---

## Manual Layer 2 Invocation

To manually trigger a Layer 2 (Claude Code) analysis without waiting for the trigger system:

### Option 1: Use the agent script directly

```cmd
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
scripts\win\pf-agent.bat
```

### Option 2: Set a flag for the loop to pick up

The loop checks for a force-trigger mechanism. The simplest approach is to delete or modify
`data/trigger_state.json` to reset cooldowns:

```cmd
del Q:\finance-analyzer\data\trigger_state.json
```

The next loop cycle will treat all signals as fresh and likely fire a trigger.

---

## Common Failure Modes

### 1. Layer 2 silent for hours

**Symptom:** No new entries in `invocations.jsonl` or `telegram_messages.jsonl` during market hours.

**Causes:**
- `CLAUDECODE` env var inherited from Claude Code session (prevents `claude -p` from starting)
- Agent timeout (600s) killing every invocation before completion
- No triggers firing (all signals stable, prices flat)
- `pf-loop.bat` crashed and did not restart (rare, but check `loop_out.txt`)

**Fix:** Check `data/agent.log` for errors. Clear `CLAUDECODE` env var. Restart `pf-loop.bat`.

### 2. Duplicate processes

**Symptom:** Multiple instances of the same loop running, causing duplicate signals or API rate limits.

**Causes:**
- Both PF-DataLoop and PF-Loop scheduled tasks enabled (PF-Loop should be DISABLED)
- Manual start while scheduled task is already running

**Fix:** Disable PF-Loop task. Kill all Python processes and restart only PF-DataLoop.

### 3. 15-minute HOLD spam on Telegram

**Symptom:** Telegram messages arriving every 15 minutes with identical HOLD analysis.

**Causes:**
- Portfolio-Agent scheduled task is ENABLED (should be DISABLED)
- This task bypasses the trigger system and fires Layer 2 on a fixed schedule

**Fix:** Disable the Portfolio-Agent scheduled task.

### 4. Stale signal data

**Symptom:** `agent_summary.json` shows old timestamps. Dashboard shows stale prices.

**Causes:**
- Layer 1 loop crashed (check `loop_out.txt`)
- Binance/Alpaca API is down or rate-limited
- Network connectivity issues on herc2

**Fix:** Restart the data loop. Check API connectivity.

### 5. Accuracy data shows 0% or missing

**Symptom:** `--accuracy` shows no data or all zeros.

**Causes:**
- PF-OutcomeCheck has not run (check scheduled task)
- `signal_log.jsonl` is empty or corrupted
- Historical price lookup is failing

**Fix:** Manually run `--check-outcomes`. Verify `signal_log.jsonl` has entries.

### 6. Dashboard returns 401 Unauthorized

**Symptom:** All API endpoints return `{"error": "Unauthorized"}`.

**Cause:** `dashboard_token` is set in `config.json` but the request lacks the token.

**Fix:** Include `?token=YOUR_TOKEN` in the URL or use `Authorization: Bearer YOUR_TOKEN` header.

---

## Log File Locations

| File                                | Purpose                                | Growth rate       |
| ----------------------------------- | -------------------------------------- | ----------------- |
| `data/loop_out.txt`                 | Layer 1 stdout (silent failure debug)  | ~1MB/day          |
| `data/agent.log`                    | Layer 2 stdout/stderr                  | ~100KB/invocation |
| `data/signal_log.jsonl`             | All signal votes + prices (per cycle)  | ~5MB/day          |
| `data/invocations.jsonl`            | Layer 2 invocation timestamps + reasons| ~10KB/day         |
| `data/telegram_messages.jsonl`      | All sent Telegram messages             | ~10KB/day         |
| `data/layer2_journal.jsonl`         | Layer 2 decision journal               | ~5KB/invocation   |
| `data/layer2_context.md`            | Layer 2 memory (regenerated each cycle)| Constant (~10KB)  |
| `data/trigger_state.json`           | Current trigger state                  | Constant (~2KB)   |
| `data/agent_summary.json`           | Latest signal snapshot                 | Constant (~200KB) |
| `data/portfolio_state.json`         | Patient portfolio                      | Grows with trades |
| `data/portfolio_state_bold.json`    | Bold portfolio                         | Grows with trades |
| `data/portfolio_value_history.jsonl`| Portfolio value snapshots for charting  | ~1KB/cycle        |

**Warning:** JSONL files grow unbounded. Monitor disk space. Log rotation is available but
migration to SQLite is planned (see TODO.md Phase 6).
