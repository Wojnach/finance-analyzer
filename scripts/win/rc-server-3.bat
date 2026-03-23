@echo off
REM Claude Code Remote Control — Server 3 (Research)
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

:restart
echo [%date% %time%] Starting RC server 3 (Research)... >> data\rc-server-3_out.txt 2>&1
claude remote-control --name "Research" --spawn worktree --capacity 4 >> data\rc-server-3_out.txt 2>&1
echo [%date% %time%] RC server 3 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-3_out.txt 2>&1
timeout /t 15 /nobreak >nul
goto restart
