' run-hidden.vbs [wait] <exe> [arg1] [arg2] ...
'
' Generic hidden launcher used by every PF-* scheduled task. WScript.Run
' with windowStyle=0 starts the child process with no console attached
' and no taskbar entry — eliminates the popup terminal windows that
' previously cluttered the desktop when many loops were scheduled.
'
' Two modes (2026-06-10):
'
'   default (detached) — Run(cmd, 0, False); wscript exits 0 immediately.
'     Required for the long-lived loop tasks (PF-DataLoop, PF-MetalsLoop,
'     ...): Task Scheduler must not wait on a child that runs for days,
'     and ExecutionTimeLimit must not kill it.
'
'   wait — when the FIRST script argument is the literal string "wait",
'     Run(cmd, 0, True) blocks until the child exits and the child's exit
'     code is propagated via WScript.Quit. Use this for one-shot tasks
'     (PF-PendingPickups etc.) so Task Scheduler "Last Result" reflects
'     the real outcome. Before this mode existed, PF-PendingPickups
'     exited 2 every day for ~20 days while Last Result showed 0
'     (audit docs/IMPROVEMENT_AUDIT_2026-06-10.md).
'
' Quoting: we receive the exe and each argument as separate WScript
' arguments (one quoting layer survives Task Scheduler XML round-trip
' and PowerShell escaping), then build the command line here. This is
' safer than nesting backtick-quotes in the install-*.ps1 callers — see
' docs/PLAN.md N3 for the failure mode the multi-arg form prevents.
'
' Usage from PowerShell (detached):
'   $action = New-ScheduledTaskAction -Execute "wscript.exe" `
'       -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$bat`""
'
' Usage from PowerShell (wait — exit code propagated):
'   $action = New-ScheduledTaskAction -Execute "wscript.exe" `
'       -Argument "`"$vbs`" `"wait`" `"cmd.exe`" `"/c`" `"$bat`""

Option Explicit

Dim startIdx, waitMode
waitMode = False
startIdx = 0

If WScript.Arguments.Count >= 1 Then
  If LCase(WScript.Arguments(0)) = "wait" Then
    waitMode = True
    startIdx = 1
  End If
End If

If WScript.Arguments.Count < startIdx + 1 Then
  WScript.Quit 1
End If

Dim cmd, i, arg
cmd = """" & WScript.Arguments(startIdx) & """"
For i = startIdx + 1 To WScript.Arguments.Count - 1
  arg = WScript.Arguments(i)
  cmd = cmd & " """ & arg & """"
Next

Dim rc
If waitMode Then
  ' Blocking launch: propagate the child's exit code so Task Scheduler
  ' "Last Result" is meaningful for one-shot tasks.
  rc = CreateObject("WScript.Shell").Run(cmd, 0, True)
  WScript.Quit rc
Else
  ' Detached launch: wscript exits immediately with 0. Task Scheduler
  ' treats that as success and the action completes; the actual loop
  ' process keeps running in the background.
  CreateObject("WScript.Shell").Run cmd, 0, False
End If
