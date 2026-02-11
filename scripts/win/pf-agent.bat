@echo off
REM Portfolio Intelligence — Claude Code Trading Agent (Layer 2)
REM Invoked by Layer 1 (main.py) when a trigger fires, or manually.
REM Claude Code auto-loads CLAUDE.md from the project root for full instructions.

cd /d Q:\finance-analyzer

REM Collect fresh data (useful for manual runs; loop already wrote agent_summary.json)
echo Collecting market data...
.venv\Scripts\python.exe -u portfolio\collect.py
if %errorlevel% neq 0 (
    echo ERROR: Data collection failed
    exit /b 1
)

REM Invoke Claude Code as the trading decision-maker
echo Running trading agent...
claude -p "You are the Layer 2 trading agent. Read data/agent_summary.json (signals, trigger reasons, timeframes) and data/portfolio_state.json (portfolio). Follow the instructions in CLAUDE.md to analyze, decide, and act. If you trade, edit portfolio_state.json and send Telegram. If you hold, only send Telegram if something is noteworthy." --allowedTools "Edit,Read,Bash,Write" --max-turns 10
