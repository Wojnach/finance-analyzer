@echo off
REM Claude Code Remote Control — Server 2 (Development)
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

:restart
echo [%date% %time%] Starting RC server 2 (Development)... >> data\rc-server-2_out.txt 2>&1
claude remote-control --name "Development" --spawn worktree --capacity 4 >> data\rc-server-2_out.txt 2>&1
echo [%date% %time%] RC server 2 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-2_out.txt 2>&1
timeout /t 15 /nobreak >nul
goto restart
