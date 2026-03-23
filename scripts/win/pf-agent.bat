@echo off
REM Portfolio Intelligence — Claude Code Trading Agent (Layer 2)
REM Invoked by Layer 1 (main.py) when a trigger fires, or manually.
REM Claude Code auto-loads CLAUDE.md for project context. Trading playbook is in docs/TRADING_PLAYBOOK.md.

cd /d Q:\finance-analyzer

REM Clear Claude Code session markers — prevents "nested session" error when launched from
REM a process tree that already has Claude Code running (e.g. Task Scheduler inheriting env)
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

REM Invoke Claude Code as the trading decision-maker
REM Layer 1 already wrote fresh agent_summary.json — no need to re-collect
echo Running trading agent...
claude -p "You are the Layer 2 trading agent. FIRST read docs/TRADING_PLAYBOOK.md for trading rules. Then read data/layer2_context.md (your memory from previous invocations). Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json (Bold portfolio). Follow the playbook to analyze, decide, and act for BOTH strategies independently. Compare your previous theses and prices with current data — were you right? Always write a journal entry and send a Telegram message." --allowedTools "Edit,Read,Bash,Write" --max-turns 40
