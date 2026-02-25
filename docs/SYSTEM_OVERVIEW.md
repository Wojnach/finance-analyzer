# Portfolio Intelligence System — System Overview

> **Generated:** 2026-02-25 (auto-improvement session)
> **Canonical architecture doc:** docs/architecture-plan.md
> **Layer 2 instructions:** CLAUDE.md

## Architecture Summary

Two-layer autonomous trading system managing 31 instruments (2 crypto, 2 metals, 27 US equities)
with 29 signals across 7 timeframes, dual simulated portfolios (Patient + Bold, 500K SEK each).

| Layer | Role | Runs | Decides |
|-------|------|------|---------|
| **Layer 1** (Python) | Data collection, signal computation, trigger detection | Every 60s (market), 5min (closed), 10min (weekend) | When to invoke Layer 2 |
| **Layer 2** (Claude Code) | Analysis, trade execution, Telegram notifications | On trigger (~20-50/day) | What to trade, what to say |

## Module Map

### Core Pipeline (Layer 1)



### Signal Modules (29 total: 11 core + 18 enhanced)

**Core signals** (in signal_engine.py):
1. RSI(14), 2. MACD(12,26,9), 3. EMA(9,21), 4. BB(20,2), 5. Fear & Greed,
6. Sentiment, 7. CryptoTrader-LM (Ministral-8B), 8. Volume Confirmation
- Disabled: ML Classifier (#8, 28.2%), Funding Rate (#9, 27.0%), Custom LoRA (#11, 20.9%)

**Enhanced composite signals** (portfolio/signals/, each 4-8 sub-indicators via majority vote):
12-trend, 13-momentum, 14-volume_flow, 15-volatility, 16-candlestick, 17-structure,
18-fibonacci, 19-smart_money, 20-oscillators, 21-heikin_ashi, 22-mean_reversion,
23-calendar, 24-macro_regime, 25-momentum_factors, 26-news_event, 27-econ_calendar,
28-forecast (Kronos/Chronos), 29-claude_fundamental (LLM cascade)

### Infrastructure



### External Dependencies

| Dependency | Purpose | Rate Limit |
|------------|---------|------------|
| Binance Spot API | Crypto OHLCV (BTC, ETH) | 600/min |
| Binance FAPI | Metals OHLCV (XAU, XAG) | 600/min |
| Alpaca IEX v2 | US stock OHLCV (27 tickers) | 150/min |
| yfinance | Stock fallback (extended hours) | 30/min |
| Alternative.me | Crypto Fear & Greed | 5min cache |
| Frankfurter API | USD/SEK exchange rate | 1h cache |
| Alpha Vantage | Stock fundamentals | 5/min, 25/day |
| Telegram Bot API | Notifications | Unbounded |
| Ministral-8B (local GPU) | CryptoTrader-LM signal | 15min cache |
| Claude Code CLI | Layer 2 agent | On trigger |
| Kronos/Chronos (local) | Forecast signal #28 | 5min cache |

### Data Files

| File | Written By | Read By | Purpose |
|------|-----------|---------|---------|
| agent_summary.json | L1 | L2 (T3), Dashboard | Full 29-signal snapshot (~64KB) |
| agent_summary_compact.json | L1 | L2 (T1/T2) | Tiered compaction (~15KB) |
| agent_context_t1.json | L1 | L2 (T1) | Held positions + macro (~200 lines) |
| agent_context_t2.json | L1 | L2 (T2) | Triggered tickers + detail (~600 lines) |
| portfolio_state.json | L2 | L1, Dashboard | Patient holdings + transactions |
| portfolio_state_bold.json | L2 | L1, Dashboard | Bold holdings + transactions |
| trigger_state.json | L1 | L1 | Consensus, sustained counts, sentiment |
| layer2_journal.jsonl | L2 | L1 (digest), Dashboard | Decisions, theses, reflections |
| layer2_context.md | L1 | L2 | Memory built from journal entries |
| signal_log.jsonl | L1 | Accuracy tracker | All signal votes per cycle |
| health_state.json | L1 | Dashboard, monitors | Heartbeat, error log |
| invocations.jsonl | L1 | Dashboard, health | Layer 2 invocation history |
| telegram_messages.jsonl | L2 | Dashboard | All sent Telegram messages |
| fundamentals_cache.json | L1 | L1 (enrichment) | Alpha Vantage stock data |

### Deployment

- **Machine:** Windows 11 (herc2), RTX 3080 10GB
- **Python:** 3.12, two venvs (.venv for main, .venv-llm for GPU inference)
- **Scheduler:** Windows Task Scheduler (PF-DataLoop, PF-Dashboard, PF-OutcomeCheck, PF-MLRetrain)
- **Auto-restart:** pf-loop.bat with :restart loop + 30s delay
- **Dashboard:** Flask on port 5055

## Discrepancies vs Architecture Doc

As of the Feb 25 improvement session, the architecture doc is current:

1. ~~Signal count drift~~ — **FIXED**: Doc now correctly shows 29 signals (11 core + 18 enhanced).
2. ~~Applicable counts wrong~~ — **FIXED**: crypto=26, stocks/metals=25.
3. ~~Missing files~~ — **FIXED**: forecast.py, claude_fundamental.py, alpha_vantage.py all documented.
4. **Scheduled tasks**: PF-ForceSleep/PF-WakeUp/PF-AutoImprove tasks not yet in arch doc (low priority).

## Known Issues Summary

### Fixed (Feb 25 improvement session)
- ~~atomic_append_jsonl missing fsync~~ — Added flush+fsync (BUG-1)
- ~~Sentiment state race with trigger.py~~ — Extracted to own state file (BUG-2/ARCH-1)
- ~~Agent log fd leak on timeout~~ — Ownership transfer pattern (BUG-3)
- ~~Stale ticker accumulation~~ — 24h pruning in reporting.py (BUG-4)
- ~~Disabled signals in accuracy reports~~ — Tagged (OFF), filtered from agent summary (REF-1)
- ~~requirements.txt stale~~ — Updated to actual dependencies (REF-2)
- ~~No linter/formatter config~~ — pyproject.toml with ruff config (FEAT-2)
- ~~conftest.py minimal~~ — Expanded with shared fixtures (ARCH-3)
- ~~Architecture doc drift~~ — Signal counts updated to 29 (FEAT-1)

### Remaining
- Post-trade detection reads portfolio JSON without coordination (race with Layer 2 writes)
- No cache hit/miss metrics
- Market timing has no US holiday support
- Tier 3 fallback hardcoded in pf-agent.bat (loses tiering)
