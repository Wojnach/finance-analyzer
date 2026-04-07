# Design: AI-Driven Daily Signal Research Agent

**Date:** 2026-04-07
**Status:** Approved
**Schedule:** Daily 18:30 CET via Windows Task Scheduler (`PF-SignalResearch`)

## Summary

A new autonomous Claude Code agent that runs daily at 18:30 CET to discover, evaluate,
and implement novel trading signals. It searches academic papers, quant blogs, and industry
reports for signal ideas across all asset classes, scores them for novelty and edge evidence,
implements the highest-ranked candidate in a git worktree, backtests it, runs codex adversarial
review, and merges to main. Unimplemented candidates persist in a ranked backlog for future
sessions.

## Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Relationship to after-hours agent | Standalone (B) | Separate context window, no competition |
| Schedule | 18:30 CET daily | After EU close, before after-hours at 22:30 |
| Search breadth | All APIs + WebSearch + WebFetch | Maximum coverage: Semantic Scholar, arXiv, SSRN, blogs |
| Implementation scope | Research + implement + backtest (B) | Full autonomous loop per session |
| Execution protocol | /fgl (worktree + codex review) | Proven quality gate |
| Signals per session | Research many, implement best (C) | Wide net, high quality on the one that ships |
| Asset rotation | Hybrid (C) | Cross-asset always + weakness-driven 2-3 assets |
| Backlog persistence | JSONL + SQLite (C) | Append-only log + queryable index |
| Context management | Single agent, disk-based context reset (C) | Proven by after-hours agent (25+ runs) |
| Merge strategy | Fetch latest, resolve conflicts, preserve parallel work | Codex review after merge resolution |

## Architecture

```
[Windows Task Scheduler] --18:30 CET--> [signal-research.bat]
      |
      v
[claude -p < docs/signal-research-prompt.md]
      |
      v
Phase 0: Baseline (read accuracy, load backlog, pick assets)
Phase 1: Academic Search (Semantic Scholar + arXiv APIs)
Phase 2: Web Research (WebSearch + WebFetch for blogs/reports)
Phase 3: Signal Extraction (parse findings -> structured candidates)
Phase 4: Scoring & Ranking (score, pick top-1 for implementation)
Phase 5: Context Reset (write summary, clear research context)
Phase 6: Implement & Backtest (worktree, signal module, tests, backtest)
Phase 7: Codex Review & Fix (adversarial review on branch)
Phase 8: Verify & Ship (fetch latest, resolve conflicts, test, merge, push, Telegram)
      |
      v
[data/signal_research_backlog.jsonl]  -- all candidates, scored
[data/signal_log.db :: signal_candidates]  -- queryable index
[data/signal_research_progress.json]  -- phase tracking
[data/signal_research_out.txt]  -- stdout capture
```

## Files to Create

| File | Purpose |
|------|---------|
| `docs/signal-research-prompt.md` | Main prompt (all 9 phases, ~400 lines) |
| `scripts/signal-research.bat` | Windows batch launcher |
| `scripts/win/install-signal-research-task.ps1` | Task Scheduler installer |

## Files to Modify

None. This is purely additive.

## Data Files (created at runtime by the agent)

| File | Purpose |
|------|---------|
| `data/signal_research_progress.json` | Phase tracking (same pattern as after-hours) |
| `data/signal_research_papers.json` | Academic paper findings per session |
| `data/signal_research_web.json` | Web/blog research findings per session |
| `data/signal_research_ranked.json` | Scored & ranked candidates per session |
| `data/signal_research_summary.json` | Context-reset summary (<200 lines) |
| `data/signal_research_backlog.jsonl` | Append-only backlog of all candidates ever found |
| `data/signal_research_log.jsonl` | Session log (start/success/fail events) |
| `data/signal_research_out.txt` | Raw stdout/stderr capture |

## SQLite Schema Addition

Add `signal_candidates` table to existing `data/signal_log.db`:

```sql
CREATE TABLE IF NOT EXISTS signal_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,                    -- YYYY-MM-DD session date
    name TEXT NOT NULL,                    -- signal name (e.g. "eth_gas_momentum")
    asset_class TEXT NOT NULL,             -- crypto, metals, stocks, cross-asset
    target_assets TEXT,                    -- JSON array: ["ETH-USD", "BTC-USD"]
    category TEXT NOT NULL,               -- technical, macro, on-chain, intermarket, sentiment, microstructure
    description TEXT NOT NULL,             -- human-readable description
    formula TEXT,                         -- computation formula or pseudocode
    data_sources TEXT,                    -- JSON array of required data sources
    parameters TEXT,                      -- JSON dict of suggested parameters
    novelty_score REAL,                   -- 0-10: how novel vs existing signals
    edge_evidence REAL,                   -- 0-10: strength of backtested/cited evidence
    data_availability REAL,              -- 0-10: how easy to get required data
    implementation_cost REAL,            -- 0-10: ease of implementation (10=trivial)
    non_redundancy REAL,                 -- 0-10: independence from existing signals
    composite_score REAL,                -- weighted average of above scores
    source_url TEXT,                      -- paper URL, blog link, etc.
    source_type TEXT,                    -- academic, blog, industry, forum
    citation TEXT,                       -- formatted citation
    status TEXT DEFAULT 'new',           -- new, implementing, implemented, rejected, deferred
    backtest_sharpe REAL,               -- if backtested: Sharpe ratio
    backtest_winrate REAL,              -- if backtested: win rate
    backtest_notes TEXT,                 -- backtest summary
    implemented_module TEXT,             -- if implemented: portfolio/signals/xxx.py
    implemented_date TEXT,               -- when implemented
    rejection_reason TEXT,               -- if rejected: why
    created_at TEXT NOT NULL,            -- ISO timestamp
    updated_at TEXT NOT NULL             -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_candidates_status ON signal_candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_score ON signal_candidates(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_asset ON signal_candidates(asset_class);
```

## Phase Details

### Phase 0: Baseline

Read current system state to determine what to research:

1. Read `data/accuracy_cache.json` — identify weakest signals and assets
2. Read `data/signal_research_backlog.jsonl` (last 50 entries) — avoid re-researching
3. Query `signal_candidates` table for existing candidates by status
4. Read `portfolio/tickers.py` for `SIGNAL_NAMES` — know what's already implemented
5. Determine today's focus:
   - **Always:** cross-asset / intermarket signals
   - **Weakness-driven:** pick 2-3 assets where signal accuracy is lowest or deteriorating
6. Read `data/daily_research_signal_audit.json` (from after-hours agent) if available — leverage its findings

**Output:** In-memory research plan (which assets, which categories to prioritize, what to avoid).

### Phase 1: Academic Search

For each focus asset + cross-asset, search academic literature:

**Semantic Scholar API** (free, no key, 100 req/5min):
```
GET https://api.semanticscholar.org/graph/v1/paper/search
  ?query=<asset>+trading+signal+prediction
  &year=2024-2026
  &limit=10
  &fields=title,abstract,year,citationCount,url,externalIds
```

**arXiv API** (free, no key):
```
GET http://export.arxiv.org/api/query
  ?search_query=all:<asset>+AND+(trading+OR+signal+OR+prediction)
  &start=0&max_results=10
  &sortBy=submittedDate&sortOrder=descending
```

**SSRN** (via WebSearch):
```
WebSearch("site:ssrn.com <asset> trading signal 2024 2025 2026")
```

For each paper found:
- Extract title, abstract, year, citation count, URL
- Check if already in backlog (skip duplicates by title similarity)
- If abstract mentions specific indicators/features, flag for deeper reading
- Use WebFetch to read full paper if freely accessible

**Search queries per asset class:**
- **Silver/Gold:** "gold price prediction machine learning", "silver trading signal",
  "precious metals forecasting", "gold-silver ratio strategy", "real yield gold correlation"
- **BTC/ETH:** "bitcoin on-chain trading signal", "ethereum DeFi prediction",
  "crypto sentiment NLP trading", "bitcoin futures funding rate alpha"
- **Stocks:** "equity momentum signal ML", "earnings drift prediction",
  "cross-sectional stock prediction features", "VIX equity signal"
- **Cross-asset:** "intermarket trading signals", "cross-asset momentum",
  "macro factor investing", "regime switching trading"

**Output:** Write `data/signal_research_papers.json`.

### Phase 2: Web Research

Search non-academic sources for practical signal ideas:

**WebSearch queries:**
- "new trading indicator 2025 2026 backtest results"
- "quantitative trading signal <asset> blog"
- "algorithmic trading feature engineering <asset>"
- "<asset> prediction model open source github"
- "trading edge <asset> systematic strategy"
- Site-specific: "site:quantifiedstrategies.com", "site:quantocracy.com",
  "site:alphaarchitect.com", "site:medium.com quantitative trading"

**WebFetch** promising results to extract signal details.

**Check GitHub trending:**
- WebSearch("github algorithmic-trading trending 2026")
- WebSearch("github trading-agent LLM signal discovery")

**Industry/Exchange research:**
- WebSearch("CME Group research <metals/crypto>")
- WebSearch("Binance research report 2025 2026")
- WebSearch("Nansen on-chain <crypto> indicator")

**Forums/Communities:**
- WebSearch("site:reddit.com/r/algotrading new indicator")
- WebSearch("site:reddit.com/r/quant feature engineering")
- WebSearch("QuantConnect community strategy <asset>")

**Output:** Write `data/signal_research_web.json`.

### Phase 3: Signal Extraction

Parse all findings from Phases 1-2 into structured signal candidates:

For each promising idea found:
1. Extract the **signal name** (concise, snake_case)
2. Classify by **category**: technical, macro, on-chain, intermarket, sentiment, microstructure, fundamental
3. Identify **target assets**: which tickers benefit from this signal
4. Extract the **formula/computation**: how is it calculated? What data inputs?
5. Note **data sources** required (free API? paid? already available?)
6. Extract any **reported performance**: Sharpe, win rate, alpha, information ratio
7. Write a **description** a developer could implement from
8. Note the **source** with full citation

**Deduplication:** Compare each candidate against:
- Existing 32 signals (by name similarity and computation overlap)
- Previous backlog entries (by title and description similarity)
- Skip anything that's just a parameter variant of an existing signal

**Output:** Append all new candidates to `data/signal_research_backlog.jsonl` (one JSON line per candidate). Insert into `signal_candidates` SQLite table.

### Phase 4: Scoring & Ranking

Score each new candidate on 5 dimensions (0-10 each):

| Dimension | Weight | Scoring Guide |
|-----------|--------|---------------|
| **Novelty** | 0.25 | 10=completely new concept, 5=new combination of known ideas, 0=already exists |
| **Edge Evidence** | 0.30 | 10=peer-reviewed with >1.0 Sharpe, 5=backtested blog post, 0=just an idea |
| **Data Availability** | 0.15 | 10=data we already have, 5=free API, 0=expensive/proprietary |
| **Implementation Cost** | 0.15 | 10=<50 lines, 5=new module, 0=requires infrastructure |
| **Non-Redundancy** | 0.15 | 10=uncorrelated with all existing signals, 5=moderate overlap, 0=duplicate |

**Composite score** = weighted average.

**Selection criteria for implementation:**
1. Composite score >= 6.0
2. Data we already have OR free API
3. Can be implemented as a standard signal module (compute_xxx_signal interface)
4. At least one cited performance metric

Pick the **single highest-scoring candidate** that passes all criteria.
If no candidate scores >= 6.0, skip implementation phases and log "no viable candidate this session."

**Output:** Write `data/signal_research_ranked.json` with all scored candidates sorted by composite_score. Update `signal_candidates` SQLite with scores.

### Phase 5: Context Reset

This is the critical context management phase:

1. Write `data/signal_research_summary.json`:
   ```json
   {
     "date": "YYYY-MM-DD",
     "assets_researched": ["XAG-USD", "BTC-USD", "cross-asset"],
     "total_papers_found": 15,
     "total_web_sources_found": 22,
     "total_new_candidates": 8,
     "top_candidate": {
       "name": "eth_gas_momentum",
       "description": "...",
       "formula": "...",
       "data_sources": [...],
       "parameters": {...},
       "composite_score": 7.8,
       "source": "...",
       "implementation_plan": "..."
     },
     "backlog_additions": [...],
     "skipped_implementation_reason": null
   }
   ```
2. From this point forward, the agent works ONLY from the summary — not the raw research.
3. If no viable candidate, skip to Phase 8 (send Telegram summary of research, no implementation).

### Phase 6: Implement & Backtest

Follow /fgl protocol:

1. **Create worktree:** `git worktree add worktrees/signal-research-<date> -b signal-research/<date>`
2. **Implement signal module:** Create `portfolio/signals/<signal_name>.py` following existing patterns:
   - Function signature: `compute_<name>_signal(df: pd.DataFrame, context: dict = None) -> dict`
   - Return: `{"action": "BUY"/"SELL"/"HOLD", "confidence": 0.0-1.0, "sub_signals": {}, "indicators": {}}`
   - Register via `signal_registry.register_enhanced()`
3. **Write tests:** Create `tests/test_signal_<name>.py` with:
   - Unit tests for the computation
   - Edge cases (empty data, NaN, insufficient bars)
   - Verify return format matches interface
4. **Run tests:** `.venv/Scripts/python.exe -m pytest tests/test_signal_<name>.py -v`
5. **Backtest:** Run a quick backtest using the signal against historical data:
   - Fetch 1 year of daily data for target assets via data_collector
   - Compute signal for each bar
   - Calculate: hit rate (1d/3d/5d), Sharpe, max drawdown, win rate
   - Write backtest results to the candidate's SQLite entry
6. **Commit:** Conventional commit message describing the new signal.

**Implementation rules:**
- Use existing `signal_utils.py` helpers (sma, ema, rsi, true_range, etc.)
- Use existing `data_collector.py` for data fetching
- Use existing `indicators.py` for baseline TA
- If the signal needs a new external data source, implement it as a separate fetch function in the signal module
- Do NOT modify `signal_engine.py` to enable the signal — leave it disabled by default. Add to `DISABLED_SIGNALS` list with a comment: `# New: pending live validation`
- The signal starts disabled. The auto-improve or after-hours agent can enable it after live validation.

### Phase 7: Codex Review & Fix

Run codex adversarial review on the worktree branch:

```
/codex:adversarial-review --wait --scope branch --effort xhigh
```

For each finding:
- **Valid bug/issue:** Fix it, commit the fix.
- **Style/minor:** Fix if trivial, skip if not.
- **False positive:** Document why in commit message, proceed.

Run tests again after fixes.

### Phase 8: Verify & Ship

**Critical: Handle parallel work safely.**

1. **Fetch latest main:**
   ```bash
   cd /mnt/q/finance-analyzer
   cmd.exe /c "cd /d Q:\finance-analyzer && git fetch origin && git pull origin main"
   ```
2. **Merge main into worktree branch** (NOT the other way around first):
   ```bash
   cd worktrees/signal-research-<date>
   git merge main
   ```
3. **Resolve conflicts carefully:**
   - NEVER delete code that appeared in main since the branch was created
   - If conflict is in a file the agent didn't modify: take main's version
   - If conflict is in a file the agent DID modify: merge both changes, preserving main's additions
   - When in doubt, preserve both sides
4. **Run codex review on merged result** if there were conflicts
5. **Run full test suite:**
   `.venv/Scripts/python.exe -m pytest tests/ -n auto`
6. **If tests pass:** Merge branch into main, push:
   ```bash
   cd /mnt/q/finance-analyzer
   git merge signal-research/<date>
   cmd.exe /c "cd /d Q:\finance-analyzer && git push"
   ```
7. **Clean up worktree:**
   ```bash
   git worktree remove worktrees/signal-research-<date>
   git branch -d signal-research/<date>
   ```
8. **Update signal_candidates SQLite:** Set status='implemented', implemented_date, implemented_module
9. **Send Telegram notification:**
   ```python
   from portfolio.telegram_notifications import send_telegram
   import json
   config = json.loads(open("config.json").read())
   send_telegram(config, "🔬 SIGNAL RESEARCH\n\n" + summary_text)
   ```
   Include: papers scanned, candidates found, what was implemented, backtest results.
10. **Update progress:** Set status="done" in progress file.

If no implementation was done (no viable candidate), still send Telegram with research summary.

## Scoring Weights Rationale

- **Edge Evidence (0.30):** Highest weight because unproven signals waste implementation time. Academic citation or backtest results are the strongest filter.
- **Novelty (0.25):** The whole point is to find signals we don't already have. Rehashing existing signals has no value.
- **Data Availability (0.15):** Signals requiring expensive data are deferred, not rejected — they go to backlog.
- **Implementation Cost (0.15):** Simple signals get tested faster. Complex ones can be deferred.
- **Non-Redundancy (0.15):** Correlated signals add noise, not alpha. The meta-learner benefits from diverse inputs.

## Asset-Specific Research Themes

### Silver (XAG-USD)
- Gold-silver ratio mean reversion
- Industrial demand proxies (solar panel production, electronics PMI)
- COT positioning (commercial vs speculative)
- ETF flows (SLV, SIVR)
- Mine supply disruptions
- Chinese import data

### Gold (XAU-USD)
- Real interest rate (TIPS yield inverted)
- Central bank buying patterns
- DXY inverse correlation
- VIX/fear premium
- Inflation expectations (breakeven rates)
- Option skew / GVZ

### Bitcoin (BTC-USD)
- On-chain: MVRV, SOPR, exchange flows, active addresses, UTXO age
- Futures: funding rate trends, OI changes, basis
- Mining: hashrate, difficulty adjustments, miner revenue
- Sentiment: crypto fear & greed decomposition, social volume
- Halving cycle position
- Stablecoin supply ratio

### Ethereum (ETH-USD)
- ETH/BTC ratio momentum
- Staking flows (net validators, withdrawal queue)
- DeFi TVL changes, protocol revenue
- Gas fee trends, EIP-1559 burn rate
- L2 adoption metrics
- Developer activity (GitHub commits)

### MicroStrategy (MSTR)
- BTC-MSTR correlation (0.97 over 90d)
- NAV premium/discount to BTC holdings
- Debt maturity schedule
- Options flow (unusual activity)
- Chaikin MF vs OBV divergence
- Convertible note pricing

### Cross-Asset
- Equity-bond correlation regime
- DXY momentum vs commodity basket
- VIX term structure (contango/backwardation)
- Yield curve slope changes
- Credit spread widening/tightening
- Sector rotation indicators

## Research Sources

### Academic APIs (structured, free)
- **Semantic Scholar:** `api.semanticscholar.org` — 100 req/5min, no key
- **arXiv:** `export.arxiv.org/api` — no rate limit, no key

### Web Search (via Claude Code tools)
- **WebSearch:** Google/Bing for blogs, forums, industry reports
- **WebFetch:** Read full pages for signal extraction

### Targeted Sites
- quantifiedstrategies.com — backtested strategy ideas
- quantocracy.com — quant blog aggregator
- alphaarchitect.com — factor investing research
- CME Group research — commodity/futures insights
- Nansen.ai research — on-chain analytics
- reddit.com/r/algotrading, /r/quant — community ideas
- QuantConnect community — open-source strategies
- GitHub trending: algorithmic-trading, trading-agent topics
- SSRN (via WebSearch) — working papers

## Prompt Structure

The prompt (`docs/signal-research-prompt.md`) will follow the exact same structure as
`docs/after-hours-research-prompt.md`:

1. `ultrathink` header
2. Mandatory execution protocol reference (`docs/GUIDELINES.md`)
3. Key rules summary (worktrees, batches, codex review, fetch-before-merge)
4. Progress tracking boilerplate (same JSON pattern)
5. Phase-by-phase instructions with deliverables
6. Research source reference list
7. Signal module interface specification
8. Scoring rubric
9. Key principles

## Batch File Structure

`scripts/signal-research.bat` mirrors `scripts/after-hours-research.bat`:
- Reset progress JSON
- Log session start to JSONL
- Pipe prompt to `claude -p --verbose --model claude-opus-4-6`
- Capture stdout to `data/signal_research_out.txt`
- Log session end with exit code, duration, phases completed
- Track success/failure counts

## Task Scheduler

`scripts/win/install-signal-research-task.ps1` mirrors `scripts/win/install-research-task.ps1`:
- Task name: `PF-SignalResearch`
- Schedule: Daily at 18:30 (Swedish local time)
- Execution time limit: 2 hours
- Restart on failure: 1 retry after 5 minutes
- Start when available: yes (catch up if machine was off)

## Integration with Existing System

### Does NOT modify:
- `signal_engine.py` — new signals start disabled
- `signal_registry.py` — new signals self-register via existing mechanism
- `config.json` — no config changes
- After-hours agent — runs independently at 22:30
- Auto-improve agent — runs independently at 09:00

### Feeds into:
- **Signal backlog** — ranked candidates for future sessions or manual review
- **After-hours agent** — can read `signal_research_ranked.json` for context
- **Auto-improve agent** — can enable validated signals (move from DISABLED to active)
- **Meta-learner** — new signals add features once enabled
- **Morning briefing** — can include "new signal implemented" highlights

### Self-improving loop:
```
Day 1: Research → implement signal_X (disabled)
Day 2-7: Signal_X computes in shadow mode, accuracy tracked
Day 7+: After-hours agent sees signal_X has >55% accuracy → enables it
Day 14+: Meta-learner incorporates signal_X as a feature
```

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Bad signal degrades system | New signals start disabled, require live validation |
| Agent breaks existing code | Worktree isolation + full test suite + codex review |
| Merge conflicts from parallel work | Fetch latest + conservative merge (preserve main's code) |
| Context window exhaustion | Disk-based context reset at Phase 5 |
| API rate limits | Semantic Scholar: 100/5min. arXiv: no limit. WebSearch: ~50/session |
| Duplicate research | SQLite dedup check against existing candidates |
| Low-quality signals | Composite score >= 6.0 gate before implementation |
| Agent runs too long | 2-hour execution time limit in Task Scheduler |

## Success Criteria

After 30 days of operation:
- 20+ unique signal candidates researched and scored
- 5-10 signal modules implemented (disabled, backtested)
- 2-3 signals promoted to active (>55% accuracy in shadow mode)
- Research backlog with 50+ scored ideas
- Zero regressions to existing signals or system stability
