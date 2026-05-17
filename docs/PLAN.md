# PLAN — Hide scheduled-task windows + dashboard duplicate detection

Date: 2026-05-17
Branch: `fix-hide-task-windows`
Worktree: `.worktrees/hide-windows`

## Problem

Every scheduled task on Windows pops a visible terminal window when it
launches (Task Scheduler invokes `cmd.exe /c …bat` or `powershell.exe -File …`
in an interactive session). The user works over Parsec and the windows
stack up across the desktop — annoying and obscures real work. The
visual cue *does* serve one purpose: spotting duplicate loops by eye
(right now: `PF-CryptoLoop`, `PF-MetalsLoop`, `PF-MstrLoop`,
`PF-LoopResume` are each registered twice; the user sees that as two
windows).

## Goal

1. Hide all popup windows for ~14 PF-* scheduled tasks that launch
   batch/PowerShell/python wrappers.
2. Restore the "I can tell at a glance if duplicates are running"
   property via a dashboard tile + endpoint instead of windows.

## Design

### 1. Universal hidden launcher — `scripts/win/run-hidden.vbs`

WSH (`wscript.exe`) can run a command with `windowStyle = 0` (hidden),
no console attached, no taskbar entry. Single small VBS shim:

```vbs
' run-hidden.vbs <command-line>
' Runs the command in a hidden window. Detaches; no console.
If WScript.Arguments.Count < 1 Then WScript.Quit 1
CreateObject("WScript.Shell").Run WScript.Arguments(0), 0, False
```

Scheduled-task action becomes:

```powershell
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe /c \`"$bat\`"`""
```

Why VBS not PowerShell `-WindowStyle Hidden`: PowerShell hidden still
flashes a console briefly on cold-start; WSH never opens one.

### 2. Replace `timeout` with `ping`-based sleep in batch wrappers

`timeout /t N /nobreak` errors when stdin is not attached to a console
(`ERROR: Input redirection is not supported, exiting the process
immediately.`). Hidden execution has no console → loop crashes after
first iteration.

Replace every `timeout /t N /nobreak >nul` with the classic headless
equivalent:

```bat
ping -n 31 127.0.0.1 >nul
```

(`ping -n N` sleeps roughly `N-1` seconds; 31 for the 30s waits, 16 for
the 15s waits.)

Affected files (10): `crypto-loop.bat`, `golddigger-loop.bat`,
`golddigger.bat`, `metals-loop.bat`, `oil-loop.bat`, `pf-loop.bat`,
`rc-server.bat`, `rc-server-2.bat`, `rc-server-3.bat`,
`silver-monitor.bat`.

### 3. Patch every install-*.ps1 to wrap its action via the VBS shim

Three patterns observed across the install scripts. Patches per pattern:

| Pattern (was)                      | Replace with                                    |
|------------------------------------|-------------------------------------------------|
| `Execute cmd.exe /c "..."`         | `Execute wscript.exe "<vbs>" "cmd.exe /c ..."` |
| `Execute powershell.exe -File ...` | add `-WindowStyle Hidden` AND wrap via VBS      |
| `Execute python.exe -u ...`        | swap to `pythonw.exe` (no console) OR wrap VBS  |

`pythonw.exe` exists at `.venv/Scripts/pythonw.exe`. For one-shot Python
tasks (`PF-LogRotate`, `PF-LoopHealthDaily`, `PF-FixAgent`,
`PF-HealthCheck-*`, `PF-MetaLearnerRetrain`, `PF-LocalLlmReport`) we
swap to `pythonw.exe`. Output is already redirected to log files in
each wrapper, so losing console stdout is harmless.

Scripts to patch (≈16):
- install-crypto-loop-task.ps1
- install-metals-loop-task.ps1
- install-mstr-loop-task.ps1
- install-oil-loop-task.ps1
- install-market-tasks.ps1 (PF-SilverMonitor + PF-GoldDigger)
- install-research-task.ps1
- install-signal-research-task.ps1
- install-shadow-review-task.ps1
- install-golddigger-task.ps1
- install-adversarial-review-task.ps1
- install-local-llm-report-task.ps1
- install-meta-learner-task.ps1
- install-fix-agent-task.ps1
- install-log-rotate-task.ps1
- install-loop-health-daily-task.ps1
- install-loop-health-report-task.ps1
- install-loop-health-watchdog-task.ps1
- install-loop-resume-task.ps1
- install-rc-keepalive-task.ps1
- install-rc-server-task.ps1
- install-rc-watchdog-task.ps1
- install-health-check-tasks.ps1

### 4. Dashboard — `/api/loop-processes` + tile

Endpoint enumerates Python processes via `psutil` (already used by
`portfolio/gpu_gate.py`), matches each against known loop signatures,
and returns:

```json
{
  "loops": [
    {"name": "main",        "matches": ["python -u portfolio/main.py --loop"],
     "pids": [12345],       "count": 1, "duplicate": false,
     "uptime_seconds": 4521},
    {"name": "metals",      "matches": ["python -u data/metals_loop.py"],
     "pids": [67890,67891], "count": 2, "duplicate": true,
     "uptime_seconds": 1023},
    ...
  ],
  "any_duplicate": true,
  "checked_at": "2026-05-17T18:42:11Z"
}
```

Known signatures (substring match on full cmdline):
- `main` → `portfolio\main.py --loop`
- `metals` → `data\metals_loop.py`
- `crypto` → `data\crypto_loop.py`
- `oil` → `data\oil_loop.py`
- `mstr` → `portfolio.mstr_loop`
- `golddigger` → `portfolio.golddigger`
- `silver_monitor` → `data\silver_monitor.py`
- `dashboard` → `dashboard\app.py`
- `hw_monitor` → `hw_monitor.py`

New module: `portfolio/loop_processes.py` — pure-Python, no I/O side
effects, returns dict; covered by unit tests with monkey-patched psutil.

Dashboard view: small tile on home page (per `feedback_dashboard_priorities`
— operational signals lead, not portfolio P&L). Red badge per duplicate,
green check when clean. Front-end module: `dashboard/static/js/views/loop_processes.js`.

### 5. Runbook — `docs/HIDDEN_TASKS.md`

How to re-install all tasks after pulling these changes:

```powershell
Get-ChildItem Q:\finance-analyzer\scripts\win\install-*.ps1 |
  ForEach-Object { & $_.FullName }
```

How to spot-check (one expected hit per loop):

```powershell
Get-Process python | Where-Object { $_.CommandLine -match "main\.py --loop" }
```

## Execution Order

1. Worktree + branch.
2. Replace `timeout` in 10 bat files. **Commit.**
3. Create `run-hidden.vbs`. **Commit.**
4. Patch install-*.ps1 in groups of 4–5. **Commit per group.**
5. Add `portfolio/loop_processes.py` + tests. **Commit.**
6. Wire `/api/loop-processes` into `dashboard/app.py`. **Commit.**
7. Add front-end tile. **Commit.**
8. Add `docs/HIDDEN_TASKS.md`. **Commit.**
9. Adversarial review (caveman:cavecrew-reviewer).
10. Fix P1/P2 findings. **Commit.**
11. Full pytest. **Commit any fixes.**
12. Merge into main, push, restart loops, kill duplicates.

## Risks (pre-premortem)

- `wscript.exe` not on every Windows install — actually shipped with
  every Windows since 2000; risk = nil on Win11 Pro.
- VBS shim mis-quotes the inner command line — explicit test: run
  `wscript run-hidden.vbs "cmd /c echo hi > Q:\tmp\test.txt"` and
  verify the file content.
- `pythonw.exe` swap breaks tasks that depend on console handle for
  llama-cpp / GPU init — none observed; LLM infra runs in separate
  `Q:/models/.venv-llm`, scheduled tasks here only do Python script
  work.
- Existing duplicate tasks (PF-CryptoLoop etc registered twice) — the
  re-install scripts use `Unregister-ScheduledTask -Confirm:$false`
  before re-registering, so duplicates get cleaned up as a side effect.

## Out of scope

- Consolidating into one Windows Terminal with tabs (would lose
  per-task auto-restart + independent execution-time-limit). Revisit
  if hidden-mode + dashboard tile turn out unsatisfying.
- Removing the 49 disabled signals or other unrelated cleanup.

## Premortem

(filled in after fresh agent runs failure-narratives pass — section
appended in step 3 of the protocol)
