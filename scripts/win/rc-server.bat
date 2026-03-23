@echo off
REM Claude Code Remote Control — Server 1 (Trading)
REM Auto-restarts on disconnect/crash with 15s delay.
REM Use rc-server-ensure.ps1 to launch all 3 servers independently.
cd /d Q:\finance-analyzer

REM Clear Claude Code session markers (prevents nested-session errors)
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

:restart
echo [%date% %time%] Starting RC server 1 (Trading)... >> data\rc-server_out.txt 2>&1
claude remote-control --name "Trading" --spawn worktree --capacity 4 >> data\rc-server_out.txt 2>&1
echo [%date% %time%] RC server 1 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server_out.txt 2>&1
timeout /t 15 /nobreak >nul
goto restart
