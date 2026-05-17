# Hidden Scheduled Tasks — Runbook

Date: 2026-05-18
Owner: anyone touching `scripts/win/install-*.ps1` or `*.bat` wrappers.

## What this is

Every PF-* scheduled task launches its wrapper through a tiny VBS
shim (`scripts/win/run-hidden.vbs`) so the user does not see ~14
terminal windows pop up across the desktop on logon — particularly
annoying when working over Parsec. The visual "is there a duplicate
running" cue is replaced by a dashboard tile + the
`/api/loop-processes` endpoint.

## Architecture

```
Task Scheduler
   └─ wscript.exe run-hidden.vbs "cmd.exe" "/c" "<bat or python cmd>"
        └─ cmd.exe /c <bat>     (or python.exe / powershell.exe)
              └─ python.exe -u <loop>    (one process per loop)
                    └─ [Layer 2] subprocess.Popen(["claude", "-p", ...],
                                                  stdin=DEVNULL)
```

Key invariants:

- The Layer 2 chain (any process that may eventually spawn `claude
  -p`) MUST still go through `python.exe`, not `pythonw.exe`. The
  `claude` CLI used to silently exit 0 while printing "Not logged in"
  when no TTY was present (the March–April 2026 outage); the fix is
  on the inspection side (assert stderr empty, parse JSON), but
  changing the stdio chain unnecessarily would re-expose the hazard.
- `agent_invocation.py` line 1074 sets `stdin=subprocess.DEVNULL`
  explicitly. Do not remove. If you add new `subprocess.run/Popen`
  calls in `agent_invocation.py`, pass `stdin=DEVNULL` too.
- The wrapper `.bat` files use `ping -n N 127.0.0.1 >nul` for sleeps,
  NOT `timeout /t N /nobreak`. `timeout` errors out when stdin is not
  attached to a console; hidden execution gives no console.

## Installation flow

1. **Kill orphan python processes first.** `Unregister-ScheduledTask`
   removes the task definition but not its already-spawned child
   processes. If you skip this step the duplicate-detection tile
   will flash red on the first poll after re-install (correct — but
   easy to dismiss as "expected, just merged" and then ignore for
   days while two main loops race on `portfolio_state.json`).

   ```powershell
   Get-Process python,pythonw,wscript,cmd -ErrorAction SilentlyContinue |
     Where-Object { $_.CommandLine -match 'finance-analyzer|metals_loop|portfolio\.main|crypto_loop|oil_loop|golddigger|mstr_loop|silver_monitor' } |
     Stop-Process -Force
   ```

2. **Re-register every PF-* task.** From an Administrator PowerShell:

   ```powershell
   Get-ChildItem Q:\finance-analyzer\scripts\win\install-*.ps1 |
     ForEach-Object { & $_.FullName }
   ```

3. **Verify quoting survived the round-trip.** This is the cheap
   detection hook for the multi-layer quoting hazard (PowerShell →
   Task Scheduler XML → Win32 `CreateProcessW`). It also catches any
   install script that forgot to wrap via wscript.

   ```powershell
   powershell -ExecutionPolicy Bypass `
     -File Q:\finance-analyzer\scripts\win\verify-tasks.ps1 -Run
   ```

   The `-Run` flag actually starts each task and asserts the wrapper's
   expected log mtime advances within 90 s. Without `-Run` the
   script only inspects the registered Execute/Arguments.

4. **Spot-check the dashboard tile.** Browse to
   `https://<dashboard-url>/#loop-processes`. Expect one green row
   per loop you actually want running. Red rows are duplicates.

## Spot-checking by hand

```powershell
# Are there any python.exe processes I don't expect?
Get-Process python | ForEach-Object {
  "{0,-8} {1}" -f $_.Id, $_.CommandLine
}

# Which PF-* tasks are registered?
Get-ScheduledTask | Where-Object { $_.TaskName -like 'PF-*' } |
  Select-Object TaskName, State | Format-Table -AutoSize

# What does the duplicate-detection endpoint return?
curl https://<dashboard-url>/api/loop-processes
```

## Re-enabling popup windows for debugging

If a hidden task is misbehaving and you need to see its terminal:

```powershell
# Temporarily revert one task to visible. Pick any one install
# script as a template — diff against pre-hide-windows to recover
# the original cmd.exe-direct invocation.
git show HEAD~5:scripts/win/install-metals-loop-task.ps1 > /tmp/visible.ps1
# Edit + run /tmp/visible.ps1 against the task you want to debug.
```

Or simpler: in another terminal, run the wrapper bat directly so
you can see what it does:

```cmd
cd /d Q:\finance-analyzer
scripts\win\metals-loop.bat
```

## Files involved

| Path | What |
|------|------|
| `scripts/win/run-hidden.vbs` | Universal hidden launcher (multi-arg form) |
| `scripts/win/install-*.ps1` (22 files) | Patched to wrap actions via wscript+VBS |
| `scripts/win/*.bat` (10 files) | `timeout` swapped for `ping` |
| `scripts/win/verify-tasks.ps1` | Post-install quoting + smoke-test runner |
| `portfolio/loop_processes.py` | psutil scan + duplicate detection |
| `dashboard/app.py` `/api/loop-processes` | Endpoint |
| `dashboard/static/js/views/loop_processes.js` | Dashboard view |

## Related docs

- `docs/PLAN.md` — design + premortem
- `CLAUDE.md` startup-check section — the original Layer 2 outage
  context that informs why N1 from the premortem matters
- `memory/feedback_dashboard_priorities.md` — why this tile lives
  under More, not on the home page (operational signals first, but
  this one is "occasional check" not "every-glance")
