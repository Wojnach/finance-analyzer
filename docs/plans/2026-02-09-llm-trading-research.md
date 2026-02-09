# LLM Trading Bots — Deep Research (2026-02-09)

## TL;DR

- **Best LLM-driven framework:** TradingAgents (29k stars, Apache 2.0, supports Claude)
- **Best LLM-trained model:** FinGPT (18.5k stars, MIT, fine-tunable for $300)
- **Best lightweight model:** FinBERT (2k stars, runs on CPU, sentiment analysis)
- **Best Claude integration:** Alpaca MCP Server + CCXT MCP Server + Freqtrade-MCP
- **Best hybrid approach:** FinRL + LLM sentiment signals (RL makes decisions, LLM provides context)
- **Cost reality:** $28/month for 1h candle analysis with Claude Sonnet + caching

---

## Part 1: LLM-DRIVEN Trading Bots (LLM makes decisions)

### Tier 1 — Production-Grade (10k+ stars)

#### TradingAgents

- **URL:** https://github.com/TauricResearch/TradingAgents
- **Stars:** 29.6k | **License:** Apache 2.0 | **Last update:** 2026-02-07
- **Architecture:** Multi-agent mirroring real trading firms
  - 7 agents: Fundamental Analyst, Sentiment Analyst, News Analyst, Technical Analyst, Bullish/Bearish Debaters, Trader, Risk Manager, Portfolio Manager
  - Built on LangGraph
- **LLM support:** GPT-5.x, Claude 4.x, Gemini 3.x, Grok 4.x, Ollama (local), OpenRouter
- **Claude support:** YES (full)
- **Markets:** Stocks (simulated exchange only — no live trading)
- **Backtesting:** YES | **Live:** NO
- **Verdict:** Best framework architecture but simulation only. Could be adapted for live with exchange integration.

#### FinGPT

- **URL:** https://github.com/AI4Finance-Foundation/FinGPT
- **Stars:** 18.5k | **License:** MIT | **Last update:** 2026-02-06
- **Architecture:** Data-centric financial LLM platform + FinGPT-Trading module
- **Models:** Fine-tuned LLaMA-2 (7B, 13B), ChatGLM2-6B, Falcon-7B
- **Claude support:** NO (uses fine-tuned open models)
- **Use case:** Sentiment analysis → trading signals
- **Verdict:** Best for training your OWN financial model. $300 to fine-tune.

#### FinRL

- **URL:** https://github.com/AI4Finance-Foundation/FinRL
- **Stars:** 13.9k | **License:** MIT
- **Architecture:** Reinforcement Learning for trading. LLM optional (FinRL-DeepSeek extension)
- **Markets:** Stocks, crypto, HFT, portfolio allocation. 100+ exchanges via CCXT.
- **Claude support:** Not natively, but extensible
- **Verdict:** Most mature trading framework. RL makes decisions, LLM provides sentiment.

### Tier 2 — Active Research (1k-11k stars)

#### AI-Trader

- **URL:** https://github.com/HKUDS/AI-Trader
- **Stars:** 11k | **License:** MIT
- **Architecture:** Multi-model competition arena. LLMs compete head-to-head.
- **Claude support:** YES
- **Markets:** NASDAQ 100, SSE 50, Crypto (BTC, ETH, XRP, SOL, ADA, etc.)
- **Live:** YES at https://ai4trade.ai
- **Verdict:** Research platform. Shows how different LLMs perform against each other.

#### FinRobot

- **URL:** https://github.com/AI4Finance-Foundation/FinRobot
- **Stars:** 6k | **License:** Apache 2.0
- **Architecture:** 4-layer AI agent platform for financial analysis
- **Agents:** Market Forecaster, Financial Analyst, Trade Strategist
- **Claude support:** Unclear (GPT-4 focused)
- **Verdict:** Analysis platform, not a trading bot.

### Tier 3 — Interesting Smaller Projects

#### MAHORAGA

- **URL:** https://github.com/ygwyg/MAHORAGA
- **Stars:** 480 | **License:** Other | **Last update:** 2026-02-04
- **Architecture:** Autonomous 24/7 agent on Cloudflare Workers
- **Claude support:** YES (full — Claude, GPT, Gemini, Grok, DeepSeek)
- **Markets:** Stocks + crypto via Alpaca (paper + live)
- **Two-tier LLM:** Cheap model for research (gpt-4o-mini), expensive for decisions (gpt-4o)
- **Data:** StockTwits, Reddit sentiment
- **Verdict:** Actually deployed and trading. Sentiment-driven. Small but practical.

#### CloddsBot

- **URL:** https://github.com/alsk1992/CloddsBot
- **Stars:** 37 | **Last update:** 2026-02-09
- **Architecture:** 700+ markets (prediction markets, perp futures, DeFi/DEX). 4 agents, 118 skills.
- **Claude support:** YES (primary reasoning engine)
- **Markets:** Polymarket, Kalshi, Betfair, Binance, Bybit, Hyperliquid, Jupiter, Uniswap V3
- **Verdict:** Most ambitious scope. Very complex setup. Built on Claude.

#### LLM-TradeBot

- **URL:** https://github.com/EthanAlgoX/LLM-TradeBot
- **Stars:** 164 | **License:** MIT | **Last update:** 2026-02-07
- **Architecture:** 12 specialized agents, dynamic symbol selection
- **Claude support:** YES (switchable between DeepSeek, OpenAI, Claude, Qwen, Gemini)
- **Verdict:** Active development, immature.

#### Trading-GPT

- **URL:** https://github.com/yubing744/trading-gpt
- **Stars:** 56
- **Architecture:** LangChain + bbgo (crypto trading framework in Go)
- **Claude support:** YES
- **Markets:** Crypto + stocks via bbgo (10+ exchanges)
- **Verdict:** Natural language strategy writing → executable code. Interesting concept.

---

## Part 2: LLM-TRAINED Models (fine-tuned for finance)

### Best for Trading Signal Generation

| Model                  | Stars | Size   | Runs on CPU?    | What it does                        | HuggingFace                      |
| ---------------------- | ----- | ------ | --------------- | ----------------------------------- | -------------------------------- |
| **FinBERT** (ProsusAI) | 2k    | 110M   | YES             | Sentiment (pos/neg/neutral)         | ProsusAI/finbert                 |
| **FLANG-ELECTRA**      | 56    | 110M   | YES             | Best BERT-class finance model       | SALT-NLP/FLANG-ELECTRA           |
| **Trading-Hero-LLM**   | —     | 110M   | YES             | Trading signals, 90.8% accuracy     | fuchenru/Trading-Hero-LLM        |
| **FinGPT v3.3**        | 18.5k | 13B    | NO (12GB+ VRAM) | Sentiment + forecasting             | FinGPT/fingpt-mt_llama2-13b_lora |
| **FinMA** (PIXIU)      | 829   | 7B/30B | NO (14GB+ VRAM) | Multi-task finance                  | TheFinAI/finma-7b-full           |
| **Fin-o1**             | —     | 8B     | NO (16GB VRAM)  | Reasoning (beats GPT-o1 on finance) | TheFinAI/Fin-o1-8B               |
| **AdaptLLM/finance**   | —     | 7B/13B | NO              | QA, analysis (rivals BloombergGPT)  | AdaptLLM/finance-LLM             |

### Key Insight: FinBERT is the Sweet Spot for Freqtrade

FinBERT runs on CPU, is tiny (110M params), and does one thing well: sentiment classification.
It can run INSIDE the Podman container alongside Freqtrade with zero GPU requirement.
Use it as a feature in FreqAI or directly in the strategy.

### For Heavier Analysis (run on Windows PC via SSH)

FinGPT or Fin-o1 require GPU. Can run on the Windows PC (herc2) and serve predictions via API.
The Steam Deck sends requests over Tailscale, gets sentiment/prediction back.

---

## Part 3: Claude-Specific Trading Integration

### MCP Servers (plug directly into Claude Code or Claude Desktop)

#### For Crypto (our use case — Binance futures)

| Server                  | What it does                     | URL                                                |
| ----------------------- | -------------------------------- | -------------------------------------------------- |
| **CCXT MCP**            | 100+ exchanges including Binance | https://github.com/Nayshins/mcp-server-ccxt        |
| **Crypto Exchange MCP** | Binance/Bybit/OKX real-time data | pulsemcp.com/servers/sydowma-crypto-exchange       |
| **CoinGecko MCP**       | Market data, prices              | docs.coingecko.com/docs/mcp-server                 |
| **LunarCrush MCP**      | Crypto social sentiment          | pulsemcp.com/servers/lunarcrush                    |
| **TradingView MCP**     | Technical indicators, patterns   | https://github.com/atilaahmettaner/tradingview-mcp |

#### For Freqtrade Integration

| Server            | What it does                 | URL                                      |
| ----------------- | ---------------------------- | ---------------------------------------- |
| **Freqtrade-MCP** | Control Freqtrade via Claude | https://github.com/kukapay/freqtrade-mcp |

This is huge — it lets Claude monitor bot status, view trades, check profit, reload config, even place trades via Freqtrade's REST API.

#### For Market Data

| Server                     | What it does                | URL                                              |
| -------------------------- | --------------------------- | ------------------------------------------------ |
| **Yahoo Finance MCP**      | Free, no rate limits        | https://github.com/Alex2Yang97/yahoo-finance-mcp |
| **Alpha Vantage MCP**      | Official, includes intraday | mcp.alphavantage.co                              |
| **Financial Datasets MCP** | Financials, prices, news    | https://github.com/financial-datasets/mcp-server |

#### Comprehensive Platform

| Server                            | What it does                       | URL                                                       |
| --------------------------------- | ---------------------------------- | --------------------------------------------------------- |
| **finance-trading-ai-agents-mcp** | Full department-based architecture | https://github.com/aitrados/finance-trading-ai-agents-mcp |

One-click local deployment. Mimics real financial company with department agents. Free. Combines technical indicators with price action analysis.

### Architecture: Claude as Trading Copilot

```
┌─────────────────────────────────────────────────────────┐
│                    Claude (Sonnet/Haiku)                  │
│  ┌───────────┐ ┌───────────┐ ┌────────────────────────┐ │
│  │ CCXT MCP  │ │ TradingView│ │ Freqtrade-MCP         │ │
│  │ (market   │ │ MCP        │ │ (bot control,          │ │
│  │  data)    │ │ (patterns) │ │  trade execution)      │ │
│  └───────────┘ └───────────┘ └────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
              ┌────────▼────────┐
              │   Freqtrade     │
              │   (Podman)      │
              │   TABaseStrategy│
              └─────────────────┘
```

### Cost Analysis: Claude for Trading

**Claude Sonnet 4.5:** $3 input / $15 output per 1M tokens
**With prompt caching:** 90% savings on cached content

| Frequency             | Without caching | With caching | Monthly |
| --------------------- | --------------- | ------------ | ------- |
| Every 5m (288/day)    | $10.80/day      | $7.69/day    | ~$231   |
| Every 1h (24/day)     | $1.44/day       | $0.92/day    | ~$28    |
| 3x/day (signals only) | $0.28/day       | $0.20/day    | ~$6     |

**Recommendation:** Start with 1h analysis ($28/month with caching). Only move to 5m if the alpha justifies the cost.

**Haiku 4.5** ($1/$5) would cut costs by ~3x: ~$10/month for hourly analysis.

---

## Part 4: Recommended Approach for This Project

### Short-term (Phase 2): Rule-based patterns

- Already planned. No LLM costs. Proven, fast, free.
- TA-Lib candlestick + TradingPatternScanner + multi-timeframe

### Medium-term (Phase 3): FreqAI + FinBERT

- Add FinBERT sentiment as a FreqAI feature (runs on CPU in container)
- CatBoost/LightGBM for ML predictions
- No API costs, runs locally on Steam Deck

### Long-term (Phase 4): Claude as Trading Copilot

- Install Freqtrade-MCP + CCXT MCP
- Claude monitors and analyzes trades
- 1h analysis cycle (~$28/month)
- Human reviews Claude's recommendations
- NOT autonomous — copilot mode

### Stretch: TradingAgents Experiment

- Fork TradingAgents, add exchange integration
- Run multi-agent debate on Claude + local models (Ollama)
- Expensive but fascinating research

---

## Key GitHub Repos to Watch

| Repo                          | Stars | Why                              |
| ----------------------------- | ----- | -------------------------------- |
| TradingAgents                 | 29.6k | Best multi-agent architecture    |
| FinGPT                        | 18.5k | Best financial LLM platform      |
| FinRL                         | 13.9k | Most mature RL trading framework |
| AI-Trader                     | 11k   | Multi-model competition          |
| Freqtrade-MCP                 | —     | Claude ↔ Freqtrade bridge        |
| CCXT MCP                      | —     | 100+ exchange data for Claude    |
| finance-trading-ai-agents-mcp | —     | Full AI trading department       |
| MAHORAGA                      | 480   | Actually deployed Claude trading |
