@echo off
REM Claude Code Remote Control — Server 2 (Development)
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

:restart
echo [%date% %time%] Starting RC server 2 (Development)... >> data\rc-server-2_out.txt 2>&1
claude remote-control --name "Development" --spawn worktree --capacity 4 >> data\rc-server-2_out.txt 2>&1
echo [%date% %time%] RC server 2 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-2_out.txt 2>&1
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 16 127.0.0.1 >nul
goto restart
