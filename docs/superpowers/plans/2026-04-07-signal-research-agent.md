# Signal Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a daily autonomous Claude Code agent (18:30 CET) that searches academic papers, blogs, and quant forums for novel trading signal ideas, scores and ranks them, implements the best candidate as a signal module, backtests it, runs codex review, and merges to main.

**Architecture:** Single prompt file piped to `claude -p` via a Windows batch file, scheduled in Task Scheduler. 9 phases: baseline -> academic search -> web research -> signal extraction -> scoring -> context reset -> implement & backtest -> codex review -> verify & ship. Research backlog persists in JSONL + SQLite.

**Tech Stack:** Claude Code CLI (Opus 4.6), Python 3.11, pandas, SQLite, Windows Task Scheduler, Semantic Scholar API, arXiv API, WebSearch/WebFetch.

**Spec:** `docs/superpowers/specs/2026-04-07-signal-research-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `docs/signal-research-prompt.md` | Main prompt with all 9 phases (~400 lines) |
| `scripts/signal-research.bat` | Windows batch launcher with logging |
| `scripts/win/install-signal-research-task.ps1` | Task Scheduler installer (18:30 daily) |

No existing files are modified. This is purely additive. Runtime data files (`data/signal_research_*.json`, `data/signal_research_backlog.jsonl`, `data/signal_research_log.jsonl`) are created by the agent at first run.

---

### Task 1: Create the Windows batch launcher

**Files:**
- Create: `scripts/signal-research.bat`

This is the entry point for Task Scheduler. It mirrors the pattern in `scripts/after-hours-research.bat`.

- [ ] **Step 1: Write the batch file**

```bat
@echo off
REM ============================================================
REM  PF-SignalResearch — Daily AI signal discovery & implementation
REM  Scheduled: 18:30 CET daily via Task Scheduler
REM  Prompt: docs\signal-research-prompt.md
REM  Progress: data\signal-research-progress.json
REM  Logs: data\signal-research-log.jsonl
REM  Output: data\signal_research_out.txt
REM
REM  Produces:
REM    data\signal_research_papers.json    — academic paper findings
REM    data\signal_research_web.json       — web/blog research findings
REM    data\signal_research_ranked.json    — scored & ranked candidates
REM    data\signal_research_summary.json   — context-reset summary
REM    data\signal_research_backlog.jsonl  — append-only candidate backlog
REM ============================================================

set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
cd /d Q:\finance-analyzer

REM --- Timestamp ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

REM --- Reset progress file ---
powershell -NoProfile -Command ^
  "Set-Content -Path 'Q:\finance-analyzer\data\signal-research-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched, waiting for Claude to begin\"}')"

REM --- Log: session starting ---
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\"}')"

echo [%TS_START%] Signal Research session starting...

REM --- Run claude with the research prompt ---
type Q:\finance-analyzer\docs\signal-research-prompt.md | claude -p --verbose --model claude-opus-4-6 > Q:\finance-analyzer\data\signal_research_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%

REM --- Timestamp end ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_END=%%T

REM --- Calculate duration ---
for /f "tokens=*" %%D in ('powershell -NoProfile -Command ^
  "$s=[datetime]::Parse('%TS_START%'); $e=[datetime]::Parse('%TS_END%'); [int]($e - $s).TotalSeconds"') do set DURATION=%%D

REM --- Read last phase from progress file ---
set LAST_PHASE=unknown
if exist Q:\finance-analyzer\data\signal-research-progress.json (
    for /f "tokens=*" %%P in ('powershell -NoProfile -Command ^
      "$j = Get-Content 'Q:\finance-analyzer\data\signal-research-progress.json' -Raw | ConvertFrom-Json; $j.current_phase"') do set LAST_PHASE=%%P
)

REM --- Read phases completed count ---
set PHASES_DONE=0
if exist Q:\finance-analyzer\data\signal-research-progress.json (
    for /f "tokens=*" %%N in ('powershell -NoProfile -Command ^
      "$j = Get-Content 'Q:\finance-analyzer\data\signal-research-progress.json' -Raw | ConvertFrom-Json; $j.phases_completed.Count"') do set PHASES_DONE=%%N
)

REM --- Count previous runs from log ---
set SUCCESS_COUNT=0
set FAIL_COUNT=0
if exist Q:\finance-analyzer\data\signal-research-log.jsonl (
    for /f %%N in ('powershell -NoProfile -Command ^
      "$m = Select-String -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Pattern '\"event\":\"success\"' -SimpleMatch; if($m){$m.Count}else{0}"') do set SUCCESS_COUNT=%%N
    for /f %%N in ('powershell -NoProfile -Command ^
      "$m = Select-String -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Pattern '\"event\":\"failed\"' -SimpleMatch; if($m){$m.Count}else{0}"') do set FAIL_COUNT=%%N
)

REM --- Log result with phase info ---
if not %EXIT_CODE%==0 goto :log_failed

set /a NEW_SUCCESS=%SUCCESS_COUNT%+1
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"success\",\"exit_code\":0,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%,\"total_success\":%NEW_SUCCESS%,\"total_failed\":%FAIL_COUNT%}')"
echo [%TS_END%] SUCCESS in %DURATION%s ^| last_phase=%LAST_PHASE% ^| phases_done=%PHASES_DONE% ^| (total: %NEW_SUCCESS% ok, %FAIL_COUNT% failed)
goto :done

:log_failed
set /a NEW_FAIL=%FAIL_COUNT%+1
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"failed\",\"exit_code\":%EXIT_CODE%,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%,\"total_success\":%SUCCESS_COUNT%,\"total_failed\":%NEW_FAIL%}')"
echo [%TS_END%] FAILED exit=%EXIT_CODE% in %DURATION%s ^| last_phase=%LAST_PHASE% ^| phases_done=%PHASES_DONE% ^| (total: %SUCCESS_COUNT% ok, %NEW_FAIL% failed)

:done
exit /b %EXIT_CODE%
```

- [ ] **Step 2: Verify batch file is syntactically correct**

Run: `cat /mnt/q/finance-analyzer/scripts/signal-research.bat | head -5`
Expected: First 5 lines showing `@echo off` and the header comment.

- [ ] **Step 3: Commit**

```bash
git add scripts/signal-research.bat
git commit -m "feat(signal-research): add Windows batch launcher for daily signal research agent"
```

---

### Task 2: Create the Task Scheduler installer

**Files:**
- Create: `scripts/win/install-signal-research-task.ps1`

Mirrors `scripts/win/install-research-task.ps1` (the after-hours installer).

- [ ] **Step 1: Write the PowerShell installer**

```powershell
# Install PF-SignalResearch scheduled task
# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-signal-research-task.ps1
#
# Schedule:
#   Daily at 18:30 CET (after EU market close, before after-hours research at 22:30)
#   Runs Claude Code CLI with signal research prompt

$taskName = "PF-SignalResearch"
$scriptPath = "Q:\finance-analyzer\scripts\signal-research.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

# Create the action
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"

# Trigger: daily at 18:30 (after EU close, before after-hours at 22:30)
$trigger = New-ScheduledTaskTrigger -Daily -At "18:30"

# Settings: allow long runs, restart on failure, run whether logged in or not
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

# Register
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Daily AI signal research: academic papers, web search, scoring, implementation, backtest, codex review. Runs Claude Opus." `
    -RunLevel Highest

Write-Host ""
Write-Host "Installed: $taskName"
Write-Host "Schedule:  Daily at 18:30 (after EU close)"
Write-Host "Script:    $scriptPath"
Write-Host "Prompt:    Q:\finance-analyzer\docs\signal-research-prompt.md"
Write-Host "Output:    Q:\finance-analyzer\data\signal_research_out.txt"
Write-Host "Timeout:   2 hours max"
Write-Host ""
Write-Host "To run manually:  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To check status:  Get-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove:        Unregister-ScheduledTask -TaskName '$taskName'"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/win/install-signal-research-task.ps1
git commit -m "feat(signal-research): add Task Scheduler installer (daily 18:30 CET)"
```

---

### Task 3: Create the main prompt — Phase 0 (Baseline) and Phase 1 (Academic Search)

**Files:**
- Create: `docs/signal-research-prompt.md`

This is the core of the system — a ~400-line prompt that guides the Claude agent through all 9 phases. We build it in stages across Tasks 3-5 to keep each commit focused.

- [ ] **Step 1: Write prompt header + Phase 0 + Phase 1**

```markdown
ultrathink

# AI Signal Research Agent — Daily Signal Discovery & Implementation

## MANDATORY: Read Execution Protocol First

Before doing ANY work, read and internalize `docs/GUIDELINES.md` — the canonical execution
protocol. Follow it exactly: explore -> plan -> implement in batches -> verify -> ship.

Key rules from the protocol:
- **Use worktrees.** Isolate work: `git worktree add worktrees/signal-research-<date> -b signal-research/<date>`.
- **Implement in batches.** 5-10 files max per batch. Commit after each batch.
- **Test everything.** `.venv/Scripts/python.exe -m pytest tests/ -n auto` after each batch.
- **Codex adversarial review.** After implementation: `/codex:adversarial-review --wait --scope branch --effort xhigh`. Fix valid findings.
- **Fetch before merge.** ALWAYS `git fetch origin && git pull origin main` before merging. Resolve conflicts carefully — NEVER delete code added by parallel work on main.
- **Commit, merge, push.** Use Windows git for push: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
- **Clean up worktrees.** `git worktree remove <path> && git branch -d <branch>` after merging.
- **Do NOT ask for approval.** Make your best call, document reasoning in commits.
- **Spend your entire context.** Do not stop early.

---

You are an autonomous AI signal research agent for a quantitative trading system. Your job is
to discover novel trading signals from academic literature, quant blogs, and industry research,
then implement and backtest the most promising candidate. Each session adds to a persistent
backlog of scored signal ideas that compounds over time.

The system is at `Q:\finance-analyzer`. Read `CLAUDE.md` for full architecture.

## PROGRESS TRACKING (MANDATORY)

Update the progress file at every phase transition:

```python
import json, datetime, pathlib
progress_file = pathlib.Path("data/signal-research-progress.json")
progress = json.loads(progress_file.read_text()) if progress_file.exists() else {"phases_completed": []}
progress.update({
    "current_phase": "PHASE N: NAME",
    "phase_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "last_update": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "running",
    "notes": "Brief description"
})
progress_file.write_text(json.dumps(progress, indent=2))
```

Mark phases complete, set `"status": "done"` or `"status": "failed"` at end.

---

## PHASE 0: BASELINE — Determine what to research

Read the current system state to decide where to focus:

1. Read `data/accuracy_cache.json` — identify signals with lowest accuracy and assets where
   the system performs worst. These are the highest-priority research targets.
2. Read last 50 lines of `data/signal_research_backlog.jsonl` (if it exists) — check what has
   already been researched. Do NOT re-research topics already in the backlog unless they are
   marked "rejected" with a reason that no longer applies.
3. Read `portfolio/tickers.py` — check `SIGNAL_NAMES` and `DISABLED_SIGNALS` to know what
   signals already exist and which are disabled.
4. Read `data/daily_research_signal_audit.json` (if it exists, from after-hours agent) —
   leverage any signal performance insights from last night's audit.
5. Read `data/daily_research_quant.json` (if it exists) — check for quant research leads
   from the after-hours agent that could inform signal search.

**Determine today's focus:**
- **Always research:** cross-asset / intermarket signals (they benefit all assets)
- **Weakness-driven:** pick 2-3 assets where signal accuracy is lowest or deteriorating fastest
- **Avoid:** categories well-covered by existing signals unless you find evidence of a better approach

**Output:** Keep your research plan in memory. No file output for this phase.

---

## PHASE 1: ACADEMIC SEARCH — Papers and formal research

Search academic literature for trading signal ideas targeting your focus assets.

### Semantic Scholar API (free, no key, 100 req/5min)

For each focus asset + cross-asset, run searches like:

```
WebFetch("https://api.semanticscholar.org/graph/v1/paper/search?query=<QUERY>&year=2024-2026&limit=10&fields=title,abstract,year,citationCount,url,externalIds")
```

**Search queries to run (adapt per session's focus assets):**

Silver/Gold:
- "gold price prediction machine learning 2025"
- "silver trading signal indicator"
- "precious metals forecasting deep learning"
- "gold silver ratio mean reversion strategy"
- "real yield gold correlation trading"

Bitcoin/Ethereum:
- "bitcoin on-chain metrics trading signal"
- "ethereum DeFi activity price prediction"
- "cryptocurrency sentiment NLP trading strategy"
- "bitcoin futures funding rate alpha"
- "crypto regime detection trading"

US Stocks:
- "equity momentum signal machine learning"
- "post earnings announcement drift prediction"
- "cross-sectional stock return features"
- "VIX equity market signal strategy"

Cross-asset:
- "intermarket trading signals cross-asset"
- "macro factor investing regime switching"
- "cross-asset momentum correlation"
- "yield curve equity signal"

### arXiv API (free, no key)

```
WebFetch("http://export.arxiv.org/api/query?search_query=all:<QUERY>&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending")
```

Run 3-4 arXiv queries focused on quantitative finance (q-fin) and machine learning (cs.LG).

### SSRN (via WebSearch)

```
WebSearch("site:ssrn.com <asset> trading signal 2024 2025 2026")
```

Run 2-3 SSRN searches for your focus assets.

### For each paper found:
- Extract: title, abstract, year, citation count, URL
- Check if already in backlog (search JSONL for title substring match)
- Skip duplicates
- If abstract mentions specific indicators, formulas, or features — flag for deeper reading
- Use WebFetch to read full paper if freely accessible (arXiv PDFs, SSRN downloads)

**Deliverable:** Write `data/signal_research_papers.json`:
```json
{
  "date": "YYYY-MM-DD",
  "queries_run": [...],
  "papers_found": [
    {
      "title": "...",
      "authors": "...",
      "year": 2025,
      "source": "semantic_scholar|arxiv|ssrn",
      "url": "...",
      "citation_count": 42,
      "abstract_summary": "...",
      "signal_ideas": ["..."],
      "relevance": "high|medium|low",
      "assets": ["XAG-USD", "XAU-USD"]
    }
  ]
}
```

---
```

- [ ] **Step 2: Commit**

```bash
git add docs/signal-research-prompt.md
git commit -m "feat(signal-research): add prompt phases 0-1 (baseline + academic search)"
```

---

### Task 4: Add prompt Phases 2-5 (Web Research, Extraction, Scoring, Context Reset)

**Files:**
- Modify: `docs/signal-research-prompt.md` (append after Phase 1)

- [ ] **Step 1: Append Phases 2-5 to the prompt**

Append the following after the Phase 1 section:

```markdown

## PHASE 2: WEB RESEARCH — Blogs, forums, industry reports

Search non-academic sources for practical, backtested signal ideas.

### WebSearch queries (run 10-15 queries):

**General quant:**
- "new trading indicator 2025 2026 backtest results"
- "quantitative trading feature engineering <focus_asset>"
- "algorithmic trading signal discovery <focus_asset>"
- "site:quantifiedstrategies.com <focus_asset>"
- "site:quantocracy.com trading signal"
- "site:alphaarchitect.com factor investing"

**Asset-specific:**
- Silver/Gold: "gold silver ratio strategy backtest", "COT positioning gold signal",
  "central bank gold buying indicator", "site:cmegroup.com precious metals research"
- BTC/ETH: "bitcoin MVRV SOPR trading signal", "ethereum staking flow indicator",
  "site:nansen.ai on-chain <crypto>", "crypto funding rate strategy backtest"
- Stocks: "earnings drift strategy backtest", "sector rotation indicator",
  "cross-sectional momentum features ML"
- MSTR: "MicroStrategy BTC correlation indicator", "MSTR NAV premium signal"
- Cross-asset: "yield curve equity signal backtest", "DXY commodity correlation strategy",
  "VIX term structure trading"

**Community & open source:**
- "site:reddit.com/r/algotrading new indicator 2025 2026"
- "site:reddit.com/r/quant feature engineering"
- "github algorithmic-trading signal discovery trending"
- "QuantConnect community strategy backtest"

### WebFetch promising results
For the top 5-10 search results that look actionable, use WebFetch to read the full page
and extract:
- Signal computation details (formula, lookback, thresholds)
- Any reported backtest results (Sharpe, win rate, max drawdown)
- Data requirements
- Implementation complexity

### Industry/Exchange research:
- "CME Group research <metals or crypto> 2025 2026"
- "Binance research report trading signal"
- "Nansen on-chain indicator predictive"

**Deliverable:** Write `data/signal_research_web.json`:
```json
{
  "date": "YYYY-MM-DD",
  "queries_run": [...],
  "sources_found": [
    {
      "title": "...",
      "url": "...",
      "source_type": "blog|forum|industry|github",
      "signal_ideas": ["..."],
      "backtest_results": {"sharpe": 1.2, "win_rate": 0.58},
      "data_requirements": ["..."],
      "relevance": "high|medium|low",
      "assets": ["BTC-USD"]
    }
  ]
}
```

---

## PHASE 3: SIGNAL EXTRACTION — Structure all findings

Parse every promising signal idea from Phases 1-2 into a structured candidate.

For each candidate signal:
1. **Name:** concise snake_case (e.g., `eth_gas_momentum`, `gold_real_yield_divergence`)
2. **Category:** technical, macro, on-chain, intermarket, sentiment, microstructure, fundamental
3. **Target assets:** which tickers this signal applies to
4. **Description:** 2-3 sentences a developer could implement from
5. **Formula:** exact computation steps, lookback period, thresholds
6. **Data sources:** what APIs/data are needed (flag if we already have it)
7. **Parameters:** suggested lookback, thresholds, sensitivity
8. **Source:** full citation with URL
9. **Reported performance:** any Sharpe, win rate, alpha, information ratio from the source

**Deduplication — check each candidate against:**
- Existing 32 signals in `portfolio/tickers.py` SIGNAL_NAMES (by name + computation overlap)
- Previous entries in `data/signal_research_backlog.jsonl` (by title + description)
- If a candidate is just a parameter variant of an existing signal, SKIP it

**Deliverable:** Append all new candidates to `data/signal_research_backlog.jsonl`
(one JSON line per candidate):
```json
{"date":"YYYY-MM-DD","name":"eth_gas_momentum","asset_class":"crypto","target_assets":["ETH-USD"],"category":"on-chain","description":"...","formula":"...","data_sources":["etherscan_gas"],"parameters":{"lookback":7,"threshold":1.5},"novelty_score":7.5,"edge_evidence":6.0,"data_availability":8.0,"implementation_cost":7.0,"non_redundancy":8.5,"composite_score":7.2,"source_url":"...","source_type":"academic","citation":"...","status":"new","created_at":"ISO"}
```

Also insert each candidate into the SQLite database:
```python
import sqlite3, json, datetime
db = sqlite3.connect("data/signal_log.db")
db.execute("""CREATE TABLE IF NOT EXISTS signal_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    name TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    target_assets TEXT,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    formula TEXT,
    data_sources TEXT,
    parameters TEXT,
    novelty_score REAL,
    edge_evidence REAL,
    data_availability REAL,
    implementation_cost REAL,
    non_redundancy REAL,
    composite_score REAL,
    source_url TEXT,
    source_type TEXT,
    citation TEXT,
    status TEXT DEFAULT 'new',
    backtest_sharpe REAL,
    backtest_winrate REAL,
    backtest_notes TEXT,
    implemented_module TEXT,
    implemented_date TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""")
db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_status ON signal_candidates(status)")
db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_score ON signal_candidates(composite_score DESC)")
db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_asset ON signal_candidates(asset_class)")
db.commit()
# Then INSERT each candidate
```

---

## PHASE 4: SCORING & RANKING — Pick the best candidate

Score each new candidate on 5 dimensions (0-10 each):

| Dimension | Weight | 10 = Best | 0 = Worst |
|-----------|--------|-----------|-----------|
| **Novelty** | 0.25 | Completely new concept | Already exists in our system |
| **Edge Evidence** | 0.30 | Peer-reviewed, Sharpe > 1.0 | Just an untested idea |
| **Data Availability** | 0.15 | Data we already fetch | Expensive proprietary data |
| **Implementation Cost** | 0.15 | < 50 lines, standard interface | Needs new infrastructure |
| **Non-Redundancy** | 0.15 | Uncorrelated with all 32 signals | Duplicate of existing |

**Composite score** = 0.25 * novelty + 0.30 * edge + 0.15 * data + 0.15 * cost + 0.15 * non_redundancy

**Selection criteria for implementation (ALL must pass):**
1. Composite score >= 6.0
2. Data is already available OR free API
3. Can implement as standard signal module (`compute_xxx_signal(df, context)` -> dict)
4. At least one cited performance metric

Pick the **single highest-scoring candidate** that passes all criteria.

If no candidate scores >= 6.0, skip Phases 6-7. Go directly to Phase 8 and report
"no viable candidate this session" in the Telegram summary. This is fine — not every
session will produce an implementable signal. The backlog still gains value.

Also consider the backlog: if a previously-scored candidate from a past session has a
higher composite score than any new candidate AND is still status='new', prefer it instead.

**Deliverable:** Write `data/signal_research_ranked.json`:
```json
{
  "date": "YYYY-MM-DD",
  "new_candidates_scored": 8,
  "backlog_candidates_checked": 12,
  "top_candidate": {
    "name": "...",
    "composite_score": 7.8,
    "scores": {"novelty": 8, "edge_evidence": 7, "data_availability": 9, "implementation_cost": 7, "non_redundancy": 8},
    "description": "...",
    "formula": "...",
    "source": "..."
  },
  "all_ranked": [...],
  "implementation_decision": "implement|skip",
  "skip_reason": null
}
```

Update SQLite: set composite_score and individual scores for all candidates.

---

## PHASE 5: CONTEXT RESET — Prepare for implementation

This is critical for context window management. Write everything to disk, then work only
from the summary going forward.

1. Write `data/signal_research_summary.json`:
```json
{
  "date": "YYYY-MM-DD",
  "assets_researched": ["XAG-USD", "BTC-USD", "cross-asset"],
  "total_papers_found": 15,
  "total_web_sources_found": 22,
  "total_new_candidates": 8,
  "top_candidate": {
    "name": "...",
    "description": "...",
    "formula": "...",
    "data_sources": [...],
    "parameters": {...},
    "composite_score": 7.8,
    "source": "...",
    "target_assets": [...],
    "category": "...",
    "implementation_notes": "Use signal_utils.rsi() for RSI computation. Fetch gas data via WebFetch from etherscan. Register via register_enhanced() in signal_registry.py."
  },
  "backlog_additions": 8,
  "skipped_implementation_reason": null
}
```

2. **From this point forward:** work ONLY from the summary file. Do NOT reference the raw
   research data in Phases 1-2. This effectively resets your context for implementation.
3. If `implementation_decision` is "skip", jump to Phase 8.

---
```

- [ ] **Step 2: Commit**

```bash
git add docs/signal-research-prompt.md
git commit -m "feat(signal-research): add prompt phases 2-5 (web research, extraction, scoring, context reset)"
```

---

### Task 5: Add prompt Phases 6-8 (Implement, Codex Review, Verify & Ship)

**Files:**
- Modify: `docs/signal-research-prompt.md` (append after Phase 5)

- [ ] **Step 1: Append Phases 6-8 and reference sections**

Append the following after the Phase 5 section:

```markdown

## PHASE 6: IMPLEMENT & BACKTEST — Build the signal module

Follow the /fgl protocol. Read `data/signal_research_summary.json` for the top candidate.

### Step 1: Create worktree

```bash
cd /mnt/q/finance-analyzer
DATE=$(date +%Y-%m-%d)
git worktree add worktrees/signal-research-$DATE -b signal-research/$DATE
cd worktrees/signal-research-$DATE
```

### Step 2: Implement signal module

Create `portfolio/signals/<signal_name>.py` following the exact pattern of existing signals
(e.g., `portfolio/signals/mean_reversion.py`):

**Required interface:**
```python
def compute_<signal_name>_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute <signal_name> signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (minimum N rows)
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys:
            action: "BUY" | "SELL" | "HOLD"
            confidence: float 0.0-1.0
            sub_signals: dict of sub-indicator votes
            indicators: dict of raw indicator values
    """
```

**Implementation rules:**
- Use existing helpers from `portfolio.signal_utils`: sma, ema, rsi, true_range, majority_vote, safe_float, rma, wma, roc
- Use `portfolio.data_collector` for any additional data fetching
- Use `portfolio.indicators` for baseline TA (RSI, MACD, EMA, BB, ATR)
- Handle edge cases: empty DataFrame, NaN values, insufficient rows
- Return HOLD with 0.0 confidence on any error or insufficient data
- If the signal needs external data (e.g., on-chain metrics), implement the fetch function
  within the signal module itself. Use `portfolio.http_retry` for HTTP calls.
- Cap confidence at 0.7 for any signal using external/non-price data (match existing pattern)

### Step 3: Register the signal

Add the signal to `portfolio/signal_registry.py` in the `_register_defaults()` function.
**BUT** also add it to `DISABLED_SIGNALS` in `portfolio/tickers.py` with a comment:

```python
# In portfolio/tickers.py:
DISABLED_SIGNALS = {"ml", "funding", "crypto_macro", "<new_signal_name>"}  # New: pending live validation
```

This ensures:
- The signal computes and gets logged to signal_log.db (shadow mode)
- It does NOT vote in consensus (no impact on live trading)
- Accuracy tracking starts immediately
- The auto-improve or after-hours agent can enable it after validation

### Step 4: Write tests

Create `tests/test_signal_<signal_name>.py`:

```python
"""Tests for <signal_name> signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.<signal_name> import compute_<signal_name>_signal


def _make_df(n=100):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_<signal_name>_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_<signal_name>_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_<signal_name>_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_<signal_name>_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_<signal_name>_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_<signal_name>_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_<signal_name>_signal(df, context=ctx)
        assert isinstance(result, dict)
```

Replace `<signal_name>` with the actual signal name from the top candidate.

### Step 5: Run tests

```bash
.venv/Scripts/python.exe -m pytest tests/test_signal_<signal_name>.py -v
```

All tests must pass.

### Step 6: Quick backtest

Fetch 1 year of daily data for each target asset and compute signal accuracy:

```python
import json
from pathlib import Path
from portfolio.data_collector import get_klines
from portfolio.signals.<signal_name> import compute_<signal_name>_signal

results = {}
for ticker in target_assets:
    df = get_klines(ticker, "1d", limit=365)
    if df is None or len(df) < 50:
        continue

    correct_1d, correct_3d, correct_5d = 0, 0, 0
    total = 0

    for i in range(50, len(df) - 5):
        window = df.iloc[:i+1].copy()
        sig = compute_<signal_name>_signal(window)
        if sig["action"] == "HOLD":
            continue

        total += 1
        future_1d = df.iloc[i+1]["close"] / df.iloc[i]["close"] - 1
        future_3d = df.iloc[min(i+3, len(df)-1)]["close"] / df.iloc[i]["close"] - 1
        future_5d = df.iloc[min(i+5, len(df)-1)]["close"] / df.iloc[i]["close"] - 1

        if sig["action"] == "BUY":
            if future_1d > 0.0005: correct_1d += 1
            if future_3d > 0.0005: correct_3d += 1
            if future_5d > 0.0005: correct_5d += 1
        elif sig["action"] == "SELL":
            if future_1d < -0.0005: correct_1d += 1
            if future_3d < -0.0005: correct_3d += 1
            if future_5d < -0.0005: correct_5d += 1

    if total > 0:
        results[ticker] = {
            "total_signals": total,
            "accuracy_1d": correct_1d / total,
            "accuracy_3d": correct_3d / total,
            "accuracy_5d": correct_5d / total,
        }

# Update SQLite candidate record with backtest results
import sqlite3
db = sqlite3.connect("data/signal_log.db")
best_acc = max((r.get("accuracy_1d", 0) for r in results.values()), default=0)
db.execute(
    "UPDATE signal_candidates SET backtest_winrate=?, backtest_notes=?, updated_at=? WHERE name=? AND status='new' ORDER BY created_at DESC LIMIT 1",
    (best_acc, json.dumps(results), datetime.datetime.now(datetime.timezone.utc).isoformat(), signal_name)
)
db.commit()
```

Print backtest results. If 1d accuracy < 45% on all assets, add a warning to the commit
message but still proceed (the signal starts disabled anyway).

### Step 7: Commit

```bash
git add portfolio/signals/<signal_name>.py tests/test_signal_<signal_name>.py
git add portfolio/signal_registry.py portfolio/tickers.py
git commit -m "feat(signals): add <signal_name> signal (disabled, pending live validation)

Source: <citation>
Composite score: <score>/10
Backtest 1d accuracy: <accuracy>%
Status: disabled (shadow mode) — will be enabled after live validation"
```

---

## PHASE 7: CODEX REVIEW & FIX

Run codex adversarial review on the worktree branch:

```
/codex:adversarial-review --wait --scope branch --effort xhigh
```

For each finding:
- **Valid bug or logic error:** Fix it, commit with `fix: <description>`
- **Style or minor issue:** Fix if trivial (< 2 min), skip otherwise
- **False positive:** Document why in a brief commit message, proceed

Run tests again after fixes:
```bash
.venv/Scripts/python.exe -m pytest tests/test_signal_<signal_name>.py -v
```

---

## PHASE 8: VERIFY & SHIP

### If implementation was done (Phases 6-7 completed):

**Step 1: Fetch latest main (CRITICAL — handle parallel work)**
```bash
cd /mnt/q/finance-analyzer
cmd.exe /c "cd /d Q:\finance-analyzer && git fetch origin && git pull origin main"
```

**Step 2: Merge main into worktree branch**
```bash
cd worktrees/signal-research-<date>
git merge main
```

**Step 3: Resolve conflicts carefully**
- NEVER delete code that appeared in main since the branch was created
- If conflict is in a file you did NOT modify: take main's version entirely
- If conflict is in a file you DID modify (e.g., `tickers.py` DISABLED_SIGNALS,
  `signal_registry.py` _register_defaults): merge BOTH changes, keeping main's
  additions AND your additions
- When in doubt: preserve both sides, test, then clean up

**Step 4: Run codex review on merged result (if there were conflicts)**
```
/codex:adversarial-review --wait --scope branch --effort high
```

**Step 5: Run full test suite**
```bash
.venv/Scripts/python.exe -m pytest tests/ -n auto
```

**Step 6: Merge into main and push**
```bash
cd /mnt/q/finance-analyzer
git merge signal-research/<date>
cmd.exe /c "cd /d Q:\finance-analyzer && git push"
```

**Step 7: Clean up worktree**
```bash
git worktree remove worktrees/signal-research-<date>
git branch -d signal-research/<date>
```

**Step 8: Update SQLite**
```python
import sqlite3, datetime
db = sqlite3.connect("data/signal_log.db")
db.execute(
    "UPDATE signal_candidates SET status='implemented', implemented_module=?, implemented_date=?, updated_at=? WHERE name=? AND status='new' ORDER BY composite_score DESC LIMIT 1",
    (f"portfolio/signals/{signal_name}.py", datetime.date.today().isoformat(), datetime.datetime.now(datetime.timezone.utc).isoformat(), signal_name)
)
db.commit()
```

### Always (whether implementation happened or not):

**Step 9: Send Telegram summary**
```python
import json
from portfolio.telegram_notifications import send_telegram
config = json.loads(open("config.json").read())

# Build summary
summary_lines = ["SIGNAL RESEARCH REPORT", ""]
summary_lines.append(f"Date: {date}")
summary_lines.append(f"Assets researched: {', '.join(assets)}")
summary_lines.append(f"Papers found: {n_papers}")
summary_lines.append(f"Web sources: {n_web}")
summary_lines.append(f"New candidates: {n_candidates}")
summary_lines.append("")

if implemented:
    summary_lines.append(f"IMPLEMENTED: {signal_name}")
    summary_lines.append(f"Score: {score}/10")
    summary_lines.append(f"Backtest 1d: {accuracy:.0%}")
    summary_lines.append(f"Status: disabled (shadow mode)")
    summary_lines.append(f"Source: {citation}")
else:
    summary_lines.append("No viable candidate this session.")
    if skip_reason:
        summary_lines.append(f"Reason: {skip_reason}")

summary_lines.append("")
summary_lines.append(f"Backlog size: {backlog_size} candidates")

send_telegram(config, "\n".join(summary_lines))
```

**Step 10: Update progress**
```python
progress = json.loads(progress_file.read_text())
progress["status"] = "done"
progress["last_update"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
progress["notes"] = f"Session complete. {'Implemented ' + signal_name if implemented else 'No implementation (no viable candidate)'}."
progress_file.write_text(json.dumps(progress, indent=2))
```

---

## SIGNAL MODULE INTERFACE REFERENCE

All signal modules in `portfolio/signals/` follow this pattern:

```python
"""<Signal Name> signal module.

<Description of what this signal does and its sub-indicators.>

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least N rows of data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from portfolio.signal_utils import majority_vote, safe_float, sma, ema, rsi

MIN_ROWS = 20  # Minimum rows for reliable computation

def compute_<name>_signal(df: pd.DataFrame, context: dict = None) -> dict:
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    close = df["close"]
    # ... compute sub-indicators ...

    votes = [sub1_vote, sub2_vote, ...]
    action, confidence = majority_vote(votes, count_hold=False)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {"sub1": sub1_vote, "sub2": sub2_vote, ...},
        "indicators": {"ind1": float(val1), "ind2": float(val2), ...},
    }
```

## SCORING RUBRIC REFERENCE

| Score | Novelty | Edge Evidence | Data Availability | Impl Cost | Non-Redundancy |
|-------|---------|---------------|-------------------|-----------|----------------|
| 10 | Brand new concept | Peer-reviewed, Sharpe>1.5 | Already fetched | <20 lines | Zero correlation |
| 8 | New combination | Backtested, Sharpe>1.0 | Free API, easy | <50 lines | Low correlation |
| 6 | Novel application | Blog backtest, win>55% | Free API, moderate | New module | Some overlap |
| 4 | Minor variation | Theoretical only | Paid API needed | Needs infra | Moderate overlap |
| 2 | Parameter tweak | Anecdotal only | Expensive/rare | Major work | High overlap |
| 0 | Already exists | No evidence | Not available | Impractical | Duplicate |

## RESEARCH SOURCES REFERENCE

### Academic APIs (free, no key)
- Semantic Scholar: api.semanticscholar.org — 100 req/5min
- arXiv: export.arxiv.org/api — no rate limit

### Targeted sites
- quantifiedstrategies.com, quantocracy.com, alphaarchitect.com
- CME Group research, Nansen.ai, Glassnode insights
- reddit.com/r/algotrading, /r/quant
- QuantConnect community, GitHub trending
- SSRN (via WebSearch)

### Asset-specific themes to explore

**Silver:** Gold-silver ratio, industrial demand (solar/electronics), COT positioning,
ETF flows (SLV), mine supply, Chinese import data.

**Gold:** Real interest rate (TIPS), central bank buying, DXY inverse, VIX/fear premium,
inflation expectations (breakeven rates), GVZ option skew.

**Bitcoin:** MVRV, SOPR, exchange flows, active addresses, UTXO age, futures funding,
OI changes, basis, hashrate, halving cycle, stablecoin supply ratio.

**Ethereum:** ETH/BTC ratio, staking flows, DeFi TVL, gas fees, EIP-1559 burn,
L2 adoption, developer activity.

**MSTR:** BTC-MSTR correlation (~0.97), NAV premium/discount, debt maturity,
options flow, Chaikin MF vs OBV divergence, convertible note pricing.

**Cross-asset:** Equity-bond correlation, DXY vs commodity basket, VIX term structure,
yield curve slope, credit spreads, sector rotation.

## KEY PRINCIPLES

- **Evidence over opinion.** Every signal must cite data, research, or backtest results.
- **Implement, don't just suggest.** The goal is to ship code, not write reports.
- **Safety first.** New signals start disabled. No impact on live trading.
- **Compound knowledge.** Each session's backlog builds on previous sessions.
- **Quality over quantity.** One well-implemented signal beats five sketched ideas.
- **The system trades real money.** Test everything. Never deploy untested changes.

## EXECUTION GUIDELINES

- **Use subagents and agent teams** when parallel work helps.
- **Save progress** to `data/signal-research-progress.json` at every phase transition.
- **Do not ask for approval.** Make your best judgment calls.
- **If something is too risky**, skip it with a comment and document in the progress file.
- **Spend your entire context.** Do not stop early. This is a deep work session.
```

- [ ] **Step 2: Commit**

```bash
git add docs/signal-research-prompt.md
git commit -m "feat(signal-research): add prompt phases 6-8 (implement, codex review, verify & ship) + reference sections"
```

---

### Task 6: Final review and verification

- [ ] **Step 1: Verify all three files exist and are non-empty**

```bash
wc -l scripts/signal-research.bat scripts/win/install-signal-research-task.ps1 docs/signal-research-prompt.md
```

Expected: ~75 lines (.bat), ~45 lines (.ps1), ~400 lines (.md)

- [ ] **Step 2: Verify the batch file references the correct prompt path**

```bash
grep "signal-research-prompt.md" scripts/signal-research.bat
```

Expected: `type Q:\finance-analyzer\docs\signal-research-prompt.md | claude -p ...`

- [ ] **Step 3: Verify the installer references the correct batch path**

```bash
grep "signal-research.bat" scripts/win/install-signal-research-task.ps1
```

Expected: `$scriptPath = "Q:\finance-analyzer\scripts\signal-research.bat"`

- [ ] **Step 4: Verify the prompt references all 9 phases**

```bash
grep "^## PHASE" docs/signal-research-prompt.md
```

Expected: PHASE 0 through PHASE 8 (9 phases total)

- [ ] **Step 5: Verify the prompt includes the SQLite CREATE TABLE statement**

```bash
grep "CREATE TABLE" docs/signal-research-prompt.md
```

Expected: `CREATE TABLE IF NOT EXISTS signal_candidates`

- [ ] **Step 6: Verify the prompt includes codex review instruction**

```bash
grep "codex:adversarial-review" docs/signal-research-prompt.md
```

Expected: At least 2 matches (Phase 7 + Phase 8 conflict resolution)

- [ ] **Step 7: Verify the prompt includes the fetch-before-merge instruction**

```bash
grep "git fetch origin" docs/signal-research-prompt.md
```

Expected: At least 1 match in Phase 8

- [ ] **Step 8: Final commit with all files**

If any files were missed in earlier commits:
```bash
git add scripts/signal-research.bat scripts/win/install-signal-research-task.ps1 docs/signal-research-prompt.md docs/superpowers/specs/2026-04-07-signal-research-design.md
git commit -m "docs(signal-research): add design spec for daily AI signal research agent"
```

---

## Post-Implementation: Install the Task

After all code is committed and pushed, the user installs the scheduled task by running:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\win\install-signal-research-task.ps1
```

This is a manual step — the user runs it once on their Windows machine.
