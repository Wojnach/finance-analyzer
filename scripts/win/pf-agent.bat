@echo off
REM Portfolio Intelligence — Claude Code Trading Agent
REM Runs Claude Code to analyze market and make trading decisions
REM Schedule this via Windows Task Scheduler (every 15-30 min)

cd /d Q:\finance-analyzer

REM Step 1: Collect current market data and signals
echo Collecting market data...
.venv\Scripts\python.exe -u portfolio\collect.py
if %errorlevel% neq 0 (
    echo ERROR: Data collection failed
    exit /b 1
)

REM Step 2: Run Claude Code trading agent
echo Running trading agent...
claude -p "You are the portfolio trading agent for a simulated 500K SEK crypto portfolio. Read data/agent_summary.json for current signals and data/portfolio_state.json for portfolio state. Analyze all 7 signals (RSI, MACD, EMA, BB, Fear&Greed, News sentiment, ML model). If 5+ of 7 agree on BUY or SELL, execute the trade by editing portfolio_state.json (update cash_sek, holdings, append to transactions with timestamp and reason). Then send a Telegram message explaining your decision using: python -c 'import json,requests;c=json.load(open(\"config.json\"));requests.post(f\"https://api.telegram.org/bot{c[\"telegram\"][\"token\"]}/sendMessage\",json={\"chat_id\":c[\"telegram\"][\"chat_id\"],\"text\":\"YOUR_MESSAGE\",\"parse_mode\":\"Markdown\"})'. If signals are mixed (less than 5 agreeing), do nothing — no trade, no message. Be disciplined." --allowedTools "Edit,Read,Bash,Write" --max-turns 10
