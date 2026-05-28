@echo off
REM PF-ClaudeUpdate -- weekly self-update of the WINDOWS production Claude Code binary.
REM
REM Why this exists (2026-05-28): the WSL/interactive install auto-updates on launch,
REM but the Windows binary that runs Layer 2 (`claude -p` inside the loop .bat files)
REM is almost never launched interactively, so its native auto-updater never fires.
REM It silently froze at 2.1.144 (2026-05-19) and missed the opus-4-8 model until a
REM manual `claude update` on 2026-05-28. This task keeps it current set-and-forget.
REM Scheduled Sunday 17:30 CET (machine is OFF overnight; 17-18 is a known on-window;
REM US market closed Sunday so a new binary settles before Monday trading).
REM
REM Logging note: redirection is done PER-LINE inside this .bat on purpose. The shared
REM run-hidden.vbs shim quotes every arg, so a ">>" passed as a task arg arrives as a
REM literal quoted token (not a redirect) and silently drops the log. Doing it here,
REM in real shell context, is the only place ">>" actually redirects.

set CLAUDE=C:\Users\Herc2\.local\bin\claude.exe
set LOG=Q:\finance-analyzer\data\claude_update_task.log

echo ==== %DATE% %TIME% : claude update start ====>> "%LOG%" 2>&1
echo [before]>> "%LOG%" 2>&1
"%CLAUDE%" --version>> "%LOG%" 2>&1
"%CLAUDE%" update>> "%LOG%" 2>&1
echo [after]>> "%LOG%" 2>&1
"%CLAUDE%" --version>> "%LOG%" 2>&1
echo ==== %DATE% %TIME% : claude update done ====>> "%LOG%" 2>&1
