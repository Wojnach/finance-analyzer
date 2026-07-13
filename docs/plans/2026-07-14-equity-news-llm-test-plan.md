# Plan: LLM backtest on news/company-driven tickers

Goal: current backtests feed indicator-only context — fine for
crypto/metals (flow-driven), but news/company-driven equities move on
headlines, earnings, and narratives. Test whether our LLMs (phi4 first)
can call direction on such tickers, and whether ADDING historical
headlines to the prompt is what unlocks it. 60%-per-cell keep policy
applies unchanged.

## Ticker candidates (data availability checked)

| Ticker         | Why                                                            | Data                                |
| -------------- | -------------------------------------------------------------- | ----------------------------------- |
| MSTR           | already Tier-1; BTC-proxy + company drama — half news-driven   | Alpaca (has key) / yfinance 1h      |
| NVDA           | pure AI-news momentum name; formerly tracked                   | yfinance 1h (730d limit OK)         |
| PLTR           | retail/news-driven; formerly tracked                           | yfinance 1h                         |
| TSLA           | the canonical narrative stock                                  | yfinance 1h                         |
| SAAB-B / SEB-C | Swedish, Avanza-tradable — but NO usable historical candle API | skip for backtest; live-shadow only |

Start with MSTR + NVDA (one semi-crypto, one pure-news) — 2 tickers
keeps the matrix readable.

## Design: two arms per ticker (the actual experiment)

- **Arm A — indicators only** (existing harness): RSI/MACD/EMA/BB/vol.
  Baseline; expectation: weak on news-driven names.
- **Arm B — indicators + historical headlines**: same context + top 3-5
  headlines from the 48h before each timestamp. If B >> A on equities,
  the design lesson is "LLM voters on equities need the news feed" and
  live wiring must include headlines (production context already has a
  headlines block — sentiment.py / news feeds — so live parity exists).

## Historical headlines source (the hard part)

- **Finnhub** free tier: `/company-news?symbol=X&from=&to=` — 1 year
  back, per-ticker, clean JSON. Primary choice. Needs free API key
  (user action, 1 min signup).
- **GDELT 2.0 doc API**: free, no key, full archive — fallback; noisier,
  needs ticker→company-name query mapping.
- NewsAPI (existing key): only ~30 days back on free tier — useless for
  Feb-Jul window. Alpha Vantage NEWS_SENTIMENT: 25 req/day cap — too slow.
- Cache all fetched headlines to `data/backtest_headlines_<ticker>.json`
  so reruns don't refetch.

## Harness changes needed (small)

1. `fetch_klines_yf(ticker, interval)` — yfinance candles path
   (market-hours index; 1h interval, 730d limit fine for Feb-Jul).
2. Session-aware outcomes: +1d horizon = next trading day's same-time
   close (pandas index shift on trading calendar), NOT +24h wall clock.
   Weekend/holiday gaps must not silently score as outcomes.
3. `--headlines finnhub` flag: injects cached headlines into context;
   prompt builders already accept a headlines/sentiment block.
4. Skip timestamps outside regular trading hours.

## Runs (phi4 first, ~1-2h GPU each arm)

1. phi4 × MSTR+NVDA × 1d horizon × Arm A (indicators only).
2. phi4 × MSTR+NVDA × Arm B (with headlines).
3. Compare per cell; if B-A gap > ~8 pts → headlines matter → consider
   Arm B reruns for crypto/metals too (news matters there as well?).
4. Only then other models (qwen3's live edge allegedly came from richer
   context — Arm B is its fair retrial).

## Prerequisites / user actions

- [ ] Finnhub free API key → config.json `finnhub_key` (user signup).
- [ ] Confirm ticker list (MSTR+NVDA proposed; TSLA/PLTR optional).

## Sequencing

After phi4 interval sweep completes + verdicts land. Runs on herc2 via
the same orchestrator (args-file mechanism, resumable, monitored).
