' run-hidden.vbs <exe> [arg1] [arg2] ...
'
' Generic hidden launcher used by every PF-* scheduled task. WScript.Run
' with windowStyle=0 starts the child process with no console attached
' and no taskbar entry — eliminates the popup terminal windows that
' previously cluttered the desktop when many loops were scheduled.
'
' Quoting: we receive the exe and each argument as separate WScript
' arguments (one quoting layer survives Task Scheduler XML round-trip
' and PowerShell escaping), then build the command line here. This is
' safer than nesting backtick-quotes in the install-*.ps1 callers — see
' docs/PLAN.md N3 for the failure mode the multi-arg form prevents.
'
' Usage from PowerShell:
'   $action = New-ScheduledTaskAction -Execute "wscript.exe" `
'       -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$bat`""
'
' The child is detached (third arg = False), so wscript exits
' immediately. Task Scheduler treats that as success and the action
' completes; the actual loop process keeps running in the background.

Option Explicit

If WScript.Arguments.Count < 1 Then
  WScript.Quit 1
End If

Dim cmd, i, arg
cmd = """" & WScript.Arguments(0) & """"
For i = 1 To WScript.Arguments.Count - 1
  arg = WScript.Arguments(i)
  cmd = cmd & " """ & arg & """"
Next

CreateObject("WScript.Shell").Run cmd, 0, False
