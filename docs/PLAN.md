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

Fresh agent ran 2026-05-17. Six failure narratives, ordered by severity.
Mitigations folded into the plan; revised execution rules at bottom.

### N1 (CRITICAL) — `pythonw.exe` swap causes silent Layer 2 auth outage
`pythonw.exe` has no console. Any Python process that later spawns
`claude -p` inherits no TTY. `claude` detects no TTY, prints "Not
logged in" to stderr, exits 0 — exact pattern as the March–April
3-week outage. `journal_written` count still ticks because the wrapper
logs HOLDs. Six days of zero trades unnoticed.
**Mitigation:** NEVER swap to `pythonw.exe` for any task whose process
tree can reach `agent_invocation.py`. Concrete blacklist: `PF-DataLoop`,
`PF-MetalsLoop`, `PF-MstrLoop`, `PF-GoldDigger`, `PF-CryptoLoop`,
`PF-OilLoop`, `PF-FixAgent`, `PF-AfterHoursResearch`,
`PF-AdversarialReview`, `PF-SignalResearch`, `PF-ShadowReview`,
`PF-LocalLlmReport` (anything that subprocesses `claude`). For ALL
tasks, route via `wscript run-hidden.vbs "cmd /c <bat or python cmd>"`.
Keep `python.exe` (not pythonw) inside the .bat. Output already
redirected to log files; losing console stdout is harmless. **Drop
the `pythonw.exe` option from §3 entirely.**
**Detection hook:** `agent_invocation.py` line 1074 already passes
`stdin=subprocess.DEVNULL`. Add assert in subprocess result handler:
if `result.stdout.strip() == ""` AND stderr contains `Not logged in`
or `not authenticated`, append to `data/critical_errors.jsonl` with
`category="layer2_auth"`. Dispatcher fires fix-agent automatically.

### N2 (HIGH) — `tqdm`/`isatty` in metals/LLM path crashes hidden run
`metals_loop.py` + signal modules + llama-cpp may use `tqdm` progress
bars or `sys.stdout.isatty()` for column wrapping. Currently parent is
`cmd → python -u …` with attached console; hidden mode strips the
console. `OSError: handle invalid` or tqdm `AttributeError` on import.
Crash, exponential backoff, silver fast-tick disabled.
**Mitigation:** add `tests/test_no_terminal_assumptions.py` — for each
loop module, run `python -c "import data.metals_loop"` etc. under
`subprocess.Popen(stdin=DEVNULL, stdout=PIPE, stderr=PIPE,
creationflags=CREATE_NO_WINDOW)` and assert clean import. Existing
`-u` flag (unbuffered) + redirected stdout is the same condition the
hidden run sees; if today's loops survive that they'll survive hidden.
Reality check: current bat files already use `>> data\*_out.txt 2>&1`
which redirects everything — tqdm with no tty already gets a pipe, not
a tty. Risk is low but the regression test is cheap.

### N3 (HIGH) — VBS double-quote mangling
`WScript.Shell.Run` re-tokenises its argument string via Windows
`CreateProcessW` quoting rules. PowerShell here-string + backtick-quote
+ scheduled-task XML round-trip = three layers of escaping. Smoke test
on a single-token command does not exercise the failure mode.
**Mitigation:** add `scripts/win/verify-tasks.ps1` — for each PF-* task
just registered, fetch `(Get-ScheduledTask).Actions[0].Arguments`,
print it verbatim, then `Start-ScheduledTask` and assert the wrapper's
expected log file mtime advanced within 90s. Run as the FINAL step of
every install-*.ps1 (added at the bottom). Failure aborts the install.
**Belt + braces:** instead of nesting quotes via PowerShell, pass the
bat path as `WScript.Arguments(1)` and have the VBS build the
command line itself. Cleaner shim:

```vbs
' run-hidden.vbs <exe> [arg1] [arg2] ...
If WScript.Arguments.Count < 1 Then WScript.Quit 1
Dim cmd, i
cmd = """" & WScript.Arguments(0) & """"
For i = 1 To WScript.Arguments.Count - 1
  cmd = cmd & " """ & WScript.Arguments(i) & """"
Next
CreateObject("WScript.Shell").Run cmd, 0, False
```

Then PS becomes:
```powershell
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$bat`""
```
One quoting layer instead of two. VBS handles re-quoting via the loop.

### N4 (REJECTED) — ping drift causes loop phase convergence
**Premise wrong.** The bat-wrapper sleep only fires on python EXIT
(crash recovery). When healthy, python runs continuously and the
wrapper sits in `START /WAIT` (or its absence). Loop cadence is
controlled by python's own `time.sleep` inside the cycle, not the bat
wrapper. ~400 ms drift on a 30s post-crash backoff cannot synchronise
two healthy loops. ACCEPT ping-based sleep; document why in the bat
files via a one-line comment.

### N5 (CRITICAL) — orphan python processes from old registrations
`Unregister-ScheduledTask` removes the task definition but does NOT
kill processes the task previously spawned. After re-install, two
`python.exe -u portfolio/main.py --loop` PIDs exist; new tile reports
duplicate (correct), user dismisses as "just merged", day 4 ignores
red badge. Actual problem: two main loops racing on `portfolio_state.json`.
**Mitigation 1:** runbook (`docs/HIDDEN_TASKS.md`) step 1 explicitly:
```powershell
Get-Process python,pythonw,wscript,cmd -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'finance-analyzer|metals_loop|portfolio\.main|crypto_loop|oil_loop|golddigger|mstr_loop|silver_monitor' } |
  Stop-Process -Force
```
**Mitigation 2:** `/api/loop-processes` payload includes
`process_started_at` (UTC ISO) per pid and `task_last_registered_at`
per task. Tile colours:
 - green = exactly 1 pid per loop
 - orange = duplicate AND any pid older than the task's last register
   time (i.e. orphan from old registration — kill it)
 - red = duplicate AND both pids younger than last register (genuine
   bug, race somewhere)

### N6 (ACCEPTED — already fixed) — Layer 2 stdin behavior change
Premortem worried `pythonw.exe`-parent grandchildren get empty stdin.
Code audit: `agent_invocation.py` line 1074 already passes
`stdin=subprocess.DEVNULL` to the claude subprocess. Stdin source is
explicit, not inherited. Risk only revives if N1 mitigation is
ignored AND someone adds a NEW subprocess invocation without DEVNULL.
ACCEPT current state; add a CI grep:

```bash
grep -n "subprocess.\(run\|Popen\|call\)" portfolio/agent_invocation.py |
  grep -v "stdin=" && echo "Layer 2 subprocess without explicit stdin" && exit 1
```

## Plan revisions from premortem

1. §3 table — drop the `pythonw.exe` option entirely. Universal wrapper
   is `wscript → cmd /c → original-command`. Reason: N1.
2. §1 VBS shim — switch to the multi-arg form (one quoting layer).
   Reason: N3.
3. §4 endpoint — add `process_started_at` per pid + colour logic for
   orphan vs genuine race. Reason: N5.
4. §5 runbook — add explicit kill-orphans step BEFORE re-running any
   install script. Reason: N5.
5. New file `scripts/win/verify-tasks.ps1` — assertion runner.
   Reason: N3.
6. New test `tests/test_no_terminal_assumptions.py`. Reason: N2.
7. New test `tests/test_layer2_subprocess_stdin.py` — grep-style guard.
   Reason: N6.

