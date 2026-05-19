# Claude Code Plugins, Skills & MCP Servers

Last updated: 2026-03-31

## Context Budget

| Component | Est. tokens | Notes |
|---|---|---|
| Local skills (75) | ~166k | `.claude/skills/` |
| Superpowers plugin | ~125k | Heaviest single plugin |
| Other plugins (20) | ~82k | Official + community marketplaces |
| CLAUDE.md + memory | ~15k | Project instructions + auto-memory |
| **Total overhead** | **~388k** | |
| **Available for work** | **~612k** | Out of 1M context |

## Enabled Plugins (21 across 4 marketplaces)

### Official Marketplace (`claude-plugins-official`)

| Plugin | Tokens | Purpose |
|---|---|---|
| superpowers | ~125k | Core workflow: brainstorm, plan, TDD, review, verify |
| qodo-skills | ~27k | Auto-quality rules before code generation |
| pr-review-toolkit | ~11.5k | Multi-agent PR review |
| hookify | ~10k | Hook creation from conversation analysis |
| feature-dev | ~6k | 7-phase guided feature development |
| telegram | ~5k | MCP server for Telegram bot interaction |
| claude-md-management | ~4.5k | CLAUDE.md auditing and improvement |
| remember | ~2.7k | Session state persistence |
| commit-commands | ~2.4k | Git commit/push/PR workflows |
| code-simplifier | ~850 | Code clarity and consistency |
| explanatory-output-style | ~684 | Educational insight annotations |
| clangd-lsp | ~195 | C/C++ language server |
| pyright-lsp | ~141 | Python type checking |
| security-guidance | ~70 | Security best practices |
| github | ~67 | GitHub API access |
| context7 | ~65 | Library documentation lookup |

### Disabled

| Plugin | Tokens | Why disabled |
|---|---|---|
| microsoft-docs | ~11.6k | User provides docs as .md/.pdf directly; never used in sessions |

### Community Marketplaces

| Plugin | Marketplace | Purpose |
|---|---|---|
| finance-skills | himself65/finance-skills | Options payoff, stock correlation, yfinance, Twitter/Discord/Telegram reader, Hormuz strait |
| trading-ideas | quant-sentiment-ai/claude-equity-research | Goldman Sachs-style equity research via `/trading-ideas NVDA` |
| financial-analysis | anthropics/financial-services-plugins | DCF, comps, LBO, 3-statement models, deck QC |
| equity-research | anthropics/financial-services-plugins | Earnings analysis, initiating coverage, thesis tracking |
| wealth-management | anthropics/financial-services-plugins | Portfolio rebalancing, tax-loss harvesting, financial planning |

## Local Skills (75 in `.claude/skills/`)

### From tradermonty/claude-trading-skills (39 skills)

**Screening & Analysis:**
- `us-stock-analysis` — Comprehensive fundamental + technical analysis for US stocks
- `canslim-screener` — William O'Neil's CANSLIM growth stock methodology
- `vcp-screener` — Minervini's Volatility Contraction Pattern
- `finviz-screener` — Build FinViz screener URLs from natural language
- `pair-trade-screener` — Statistical arbitrage pair identification
- `value-dividend-screener` — Value + dividend screening (not installed, see Skipped section)
- `pead-screener` — Post-Earnings Announcement Drift patterns

**Market Regime & Breadth:**
- `macro-regime-detector` — Cross-asset regime shifts (RSP/SPY, yield curve, credit)
- `market-breadth-analyzer` — 0-100 composite breadth score
- `market-top-detector` — Distribution days + leading stock deterioration
- `us-market-bubble-detector` — Minsky/Kindleberger bubble framework
- `market-environment-analysis` — Global market assessment
- `uptrend-analyzer` — Monty's Uptrend Ratio Dashboard
- `downtrend-duration-analyzer` — Historical correction length analysis
- `ftd-detector` — Follow-Through Day market bottom signals
- `exposure-coach` — Net exposure ceiling and market posture summary

**Edge Discovery Pipeline (7 skills):**
- `edge-signal-aggregator` — Rank signals from multiple edge-finding skills
- `edge-candidate-agent` — Generate research tickets from EOD observations
- `edge-pipeline-orchestrator` — End-to-end edge research workflow
- `edge-strategy-designer` — Convert edge concepts into strategy drafts
- `edge-strategy-reviewer` — Quality gate for strategy drafts
- `edge-hint-extractor` — Extract edge hints from market observations
- `edge-concept-synthesizer` — Abstract hints into reusable concepts

**Strategy & Portfolio:**
- `technical-analyst` — Chart-based weekly price analysis
- `sector-analyst` — Sector rotation and cycle positioning
- `institutional-flow-tracker` — 13F smart money flows
- `portfolio-manager` — Portfolio analysis via Alpaca integration
- `position-sizer` — Risk-based sizing (ATR, Kelly, fixed fractional)
- `options-strategy-advisor` — Options strategy selection and simulation
- `scenario-analyzer` — 18-month multi-scenario analysis
- `backtest-expert` — Systematic backtesting methodology
- `stanley-druckenmiller-investment` — Macro trading synthesis
- `strategy-pivot-designer` — Detect stagnation, propose pivots

**Signals & Postmortem:**
- `signal-postmortem` — Analyze past signal outcomes and failures
- `trade-hypothesis-ideator` — Generate falsifiable trade hypotheses
- `trader-memory-core` — Thesis lifecycle tracking

**Data & Events:**
- `economic-calendar-fetcher` — FOMC/CPI/NFP upcoming events via FMP
- `market-news-analyst` — Recent market-moving news analysis
- `earnings-trade-analyzer` — Post-earnings 5-factor scoring
- `theme-detector` — Trending market themes and narratives

### From agiprolabs/claude-trading-skills (16 skills)

**Quantitative:**
- `kelly-criterion` — Optimal sizing with fractional variants
- `mean-reversion` — Hurst exponent, half-life, O-U modeling
- `correlation-analysis` — Rolling correlation, tail dependence, regime-dependent
- `cointegration-analysis` — Engle-Granger, Johansen, rolling stability
- `volatility-modeling` — GARCH, EWMA, realized vol, vol cones
- `regime-detection` — Volatility clustering, trend detection
- `signal-classification` — XGBoost/LightGBM with walk-forward validation
- `walk-forward-validation` — Time-series-aware splits, overfit detection

**Execution & Risk:**
- `exit-strategies` — Systematic exit rules, trailing stops
- `slippage-modeling` — Execution cost estimation, AMM depth
- `risk-management` — Portfolio-level drawdown management, circuit breakers
- `liquidity-analysis` — DEX liquidity depth and slippage

**Data & Tools:**
- `feature-engineering` — ML feature construction from market data
- `custom-indicators` — NVT, exchange flow, funding rate, smart money flow
- `ohlcv-processing` — Data quality, gap handling, anomaly detection
- `portfolio-analytics` — Return metrics, risk-adjusted ratios, rolling analysis
- `pandas-ta` — 130+ indicators via pandas-ta
- `ta-lib` — 150+ functions + 61 candlestick patterns
- `backtrader` — Event-driven backtesting framework
- `trading-visualization` — Candlesticks, equity curves, drawdowns
- `trade-journal` — Structured logging and behavioral pattern detection
- `strategy-framework` — Standardized strategy template
- `sentiment-analysis` — Social media, news, on-chain sentiment
- `whale-tracking` — Large wallet monitoring and smart money signals
- `coingecko-api` — Broad crypto market data (13k+ tokens)
- `market-microstructure-traditional` — Order book dynamics, market making theory

### From marketcalls/vectorbt-backtesting-skills (6 skills)

- `vectorbt-expert` — VectorBT backtesting expert
- `backtest` — Quick backtest with data fetch, signals, stats, plots
- `optimize` — Parameter optimization with heatmaps
- `strategy-compare` — Side-by-side strategy comparison
- `quick-stats` — Inline key stats for default EMA crossover
- `setup` — Environment setup for backtesting

### From staskh/trading_skills (4 skills)

- `greeks` — Options Greeks via Black-Scholes (needs `pip install trading-skills`)
- `risk-assessment` — VaR, beta, drawdown (needs `pip install trading-skills`)
- `earnings-calendar` — Upcoming earnings dates (needs `pip install trading-skills`)
- `news-sentiment` — Yahoo Finance news sentiment (needs `pip install trading-skills`)

## Subagents (`.claude/agents/`)

| Agent | Model | Purpose |
|---|---|---|
| `quant-analyst` | Opus | Quantitative trading strategies, financial models, risk analytics, derivatives pricing, Monte Carlo, GARCH |

## MCP Servers (`.mcp.json`)

| Server | Source | API Key | What it provides |
|---|---|---|---|
| `fred` | stefanoamorelli/fred-mcp-server (npm) | FRED key (configured) | 800k+ economic time series — yields, inflation, GDP, fed funds |
| `avanza-mcp` | AnteWall/avanza-mcp (PyPI) | None (public API) | Swedish stocks, funds, order books, charts, financial ratios |
| `financial-datasets` | @financial-datasets/mcp-server (npm) | None | Income statements, balance sheets, cash flow, prices, news |
| `tradingview` | tradingview-mcp-server (PyPI) | None | 30+ technical indicators, live screening, Bollinger squeeze, backtesting |

---

## NOT INSTALLED — Reference for Future Use

### Skills Skipped (tradermonty)

| Skill | Tokens | Why skipped |
|---|---|---|
| breadth-chart-analyst | 8,114 | Overlaps with market-breadth-analyzer |
| dividend-growth-pullback-screener | 2,974 | Dividend niche, not our focus |
| value-dividend-screener | 4,214 | Dividend niche |
| kanchi-dividend-sop | 1,760 | Very specific dividend accounting SOP |
| kanchi-dividend-review-monitor | 1,031 | Dividend monitoring niche |
| kanchi-dividend-us-tax-accounting | 1,181 | US dividend tax accounting |
| data-quality-checker | 1,485 | Meta/infrastructure skill |
| dual-axis-skill-reviewer | 1,098 | Meta skill for reviewing other skills |
| skill-designer | 722 | Meta skill for creating skills |
| skill-idea-miner | 803 | Meta skill |
| skill-integration-tester | 988 | Meta skill |
| earnings-calendar (tradermonty) | — | Already have staskh version |

### Skills Skipped (agiprolabs) — DeFi/Solana Specific

| Skill | Why skipped |
|---|---|
| birdeye-api, dexscreener-api, defillama-api, helius-api, solanatracker-api | Solana/DeFi APIs — not our exchanges |
| dex-pool-analysis, dex-execution, raptor-dex | DEX-specific execution |
| jito-bundles, mev-analysis, shredstream | MEV/Solana infrastructure |
| solana-tx-building, solana-rpc, yellowstone-grpc | Solana chain interaction |
| impermanent-loss, lp-math | Liquidity providing (not our model) |
| pumpfun-mechanics | Memecoin mechanics |
| sybil-detection | DeFi fraud detection |
| copy-trading | Copy trading automation |
| token-holder-analysis, token-economics | Token-specific analysis |
| wallet-profiling | DeFi wallet analysis |

### Skills Skipped (agiprolabs) — Tax/Compliance Niche

| Skill | Why skipped |
|---|---|
| crypto-tax-export, tax-loss-harvesting, tax-liability-tracking | Swedish tax system differs |
| wash-sale-detection | US-specific rule |
| regulatory-reporting | US compliance framework |
| cost-basis-engine, trade-accounting | Accounting niche |

### Skills Skipped (agiprolabs) — Other

| Skill | Why skipped |
|---|---|
| position-sizing | Overlap with position-sizer (tradermonty) |
| vectorbt | Overlap with vectorbt-expert |
| yield-analysis | Bond/DeFi yield niche |
| rl-execution | Reinforcement learning execution (experimental) |
| fixed-income | Stub only |
| options-pricing | Stub only |
| market-microstructure | Overlap with market-microstructure-traditional |

### MCP Servers NOT Configured

| Server | Repo | Why not installed |
|---|---|---|
| Alpaca MCP | alpacahq/alpaca-mcp-server | Redundant — already use Alpaca via `data_collector.py` |
| Binance MCP | ethancod1ng/binance-mcp-server | Redundant — already use Binance via `data_collector.py` |
| Alpha Vantage MCP | alphavantage/alpha_vantage_mcp | Redundant — already use via `alpha_vantage.py` (25/day limit) |
| NewsAPI MCP | berlinbra/news-api-mcp | Redundant — already use via `sentiment.py` |
| Yahoo Finance MCP | Alex2Yang97/yahoo-finance-mcp | Partial overlap with yfinance in Python |
| CoinGecko MCP (hosted) | mcp.api.coingecko.com | Free but adds another running server; coingecko-api skill covers it |
| Polygon.io MCP | polygon-io/mcp_polygon | Needs paid API key |
| Finnhub MCP | cfdude/mcp-finnhub | Needs API key; overlaps with existing data sources |
| Twelve Data MCP | twelvedata/mcp | Needs API key |
| EdgarTools MCP | edgartools.io | Free but needs pip install + Python MCP server setup |
| LSEG MCP | lseg.com | Enterprise subscription required |
| Daloopa MCP | docs.daloopa.com | Subscription ~$500+/mo |
| QuantConnect MCP | quantconnect.com | Expensive for live; free cloud backtest only |
| Interactive Brokers MCP | Multiple repos | Don't use IB |
| Kraken CLI | krakenfx/kraken-cli | Don't use Kraken |
| MetaTrader 5 MCP | ariadng/metatrader-mcp-server | Don't use MT5 |
| Schwab MCP | sudowealth/schwab-mcp | Don't use Schwab |

### Plugin Marketplaces NOT Added

| Marketplace | Repo | Why not added |
|---|---|---|
| JoelLewis/finance_skills | 84 skills | Wealth management/compliance focus, not algorithmic trading |
| alirezarezvani/claude-skills | 205 skills | Mostly non-finance (engineering, marketing, product) |
| wangfe/awesome-finance-skills | Auto-generated daily | Variable quality, auto-generated |
| jmanhype/claude-code-plugin-marketplace | 19 plugins | Most are manifest-only stubs |
| daymade/claude-code-skills | 43 skills | General tooling, not trading-specific |

### Frameworks NOT Installed

| Framework | Why not |
|---|---|
| GSD (Get Shit Done) | Greenfield project orchestration; our workflow is incremental |
| BMAD | Full agile framework with PM/Architect agents; overkill |
| Ralph Loop | High velocity, high risk; wrong for a trading system |
| SpecKit | Lightweight alternative to GSD; same reason — incremental workflow |
