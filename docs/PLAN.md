# Plan: Remove Unused Instruments

**Date:** Mar 1, 2026

## Goal
Remove 12 instruments the user no longer holds/tracks, add 1 new Nordic stock (Investor B).

## Tickers to REMOVE

| Ticker | Type | Reason |
|--------|------|--------|
| MSTR | Tier 1 Stock | Not in user's current portfolio |
| BABA | Tier 1 Stock | Not in user's current portfolio |
| GRRR | Tier 1 Stock | Not in user's current portfolio |
| IONQ | Tier 1 Stock | Not in user's current portfolio |
| TEM | Tier 1 Stock | Not in user's current portfolio |
| UPST | Tier 1 Stock | Not in user's current portfolio |
| VERI | Tier 1 Stock | Not in user's current portfolio |
| QQQ | Tier 1 ETF | Not in user's current portfolio |
| K33 | Tier 2 Nordic | Not in user's current portfolio |
| H100 | Tier 2 Nordic | Not in user's current portfolio |
| BTCAP-B | Tier 2 Nordic | Not in user's current portfolio |
| BULL-NDX3X | Tier 3 Warrant | Underlying QQQ removed, user didn't list it |

## Tickers to ADD

| Ticker | Type | Notes |
|--------|------|-------|
| INVE-B | Tier 2 Nordic | Investor AB class B, Stockholm Exchange |

## Tickers KEPT (no changes)

- **Tier 1 Stocks (15):** AMD, GOOGL, AMZN, AAPL, AVGO, LMT, META, MU, NVDA, PLTR, SOUN, SMCI, TSM, TTWO, VRT
- **Tier 1 Crypto (2):** BTC-USD, ETH-USD (underlyings for XBT-TRACKER, ETH-TRACKER)
- **Tier 1 Metals (2):** XAU-USD, XAG-USD (XAG underlying for MINI-SILVER; XAU for macro/correlation)
- **Tier 2 Nordic (2+1):** SAAB-B, SEB-C, INVE-B (new)
- **Tier 3 Warrants (4):** XBT-TRACKER, ETH-TRACKER, MINI-SILVER, MINI-TSMC

## What could break

- Tests hardcoding removed tickers (MSTR especially, used heavily in tests) — update test data
- Portfolio state files may reference removed tickers in transaction history — leave history intact
- Signal log / accuracy data has historical entries for removed tickers — leave intact
- BULL-NDX3X removal cascades: QQQ was its underlying, both go

## Execution Order (2 batches)

### Batch 1: Core config + source-of-truth (5 files)
1. `portfolio/tickers.py` — remove 8 tickers from SYMBOLS + STOCK_SYMBOLS
2. `config.json` — remove K33, H100, BTCAP-B, BULL-NDX3X; add INVE-B
3. `portfolio/sentiment.py` — remove from TICKER_CATEGORIES
4. `portfolio/social_sentiment.py` — remove MSTR entries
5. `portfolio/news_keywords.py` — remove from SECTOR_MAP

### Batch 2: Secondary refs + docs (5 files)
6. `portfolio/reporting.py` — remove MSTR from cross-asset followers
7. `portfolio/risk_management.py` — remove MSTR from CORRELATED_PAIRS
8. `CLAUDE.md` — update instrument tables (Tier 1/2/3)
9. `docs/architecture-plan.md` — update instrument tables
10. Test files — update hardcoded ticker references to use kept tickers
