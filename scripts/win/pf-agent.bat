@echo off
REM Portfolio Intelligence — Claude Code Trading Agent (Layer 2)
REM Invoked by Layer 1 (main.py) when a trigger fires, or manually.
REM Claude Code auto-loads CLAUDE.md from the project root for full instructions.

cd /d Q:\finance-analyzer

REM Invoke Claude Code as the trading decision-maker
REM Layer 1 already wrote fresh agent_summary.json — no need to re-collect
echo Running trading agent...
claude -p "You are the Layer 2 trading agent. Read data/agent_summary.json (signals, trigger reasons, timeframes), data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json (Bold portfolio). Follow the instructions in CLAUDE.md to analyze, decide, and act for BOTH strategies independently. Always send a Telegram message." --allowedTools "Edit,Read,Bash,Write" --max-turns 10
