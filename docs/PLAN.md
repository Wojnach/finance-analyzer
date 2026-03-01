# Plan: On-Chain Data + Per-Ticker Signal Accuracy

**Date:** Mar 1, 2026
**Branch:** `feat/onchain-accuracy`

## Goal
Implement two high-priority TODO items:
1. Per-ticker per-signal accuracy cross-tabulation
2. BGeometrics on-chain data integration for BTC

## Feature 1: Per-Ticker Per-Signal Accuracy

**Why:** We know "RSI is 57% overall" and "XAG is 76% overall" but NOT "RSI for XAG is 85%".
This cross-tabulation lets Layer 2 trust the right signals per ticker.

**Data source:** SQLite `data/signal_log.db` — ~45K ticker_signals rows + ~158K outcomes.

**What could break:** Nothing — purely additive.

### Files:
- `portfolio/accuracy_stats.py` — add `accuracy_by_ticker_signal()`
- `portfolio/signal_db.py` — add SQL-optimized `ticker_signal_accuracy()`
- `portfolio/reporting.py` — surface as `signal_reliability` in compact summary
- `tests/test_ticker_signal_accuracy.py` (NEW)

## Feature 2: BGeometrics On-Chain Data

**Why:** BTC/ETH decisions have zero on-chain context. MVRV, SOPR, realized price are what
distinguish "RSI oversold" from "below realized price — generational buy."

**API:** bitcoin-data.com — free tier 8 req/hr, 15/day. Token auth (`?token=XXX`).
**Budget:** 6 metrics × 2/day = 12 req (fits 15/day). Cache 12h.
**Endpoints:** `/v1/mvrv/{last}`, `/v1/sopr/{last}`, `/v1/nupl/{last}`,
`/v1/realized-price/{last}`, `/v1/exchange-netflow`, `/v1/btc-liquidations`

**What could break:** Nothing — additive. Missing token → graceful None.

### Files:
- `portfolio/onchain_data.py` (NEW) — fetcher with 12h cache
- `portfolio/shared_state.py` — add TTL + rate limiter
- `portfolio/reporting.py` — surface `onchain` section in compact summary
- `config.json` — add `bgeometrics` block (TODO: MANUAL REVIEW for token)
- `tests/test_onchain_data.py` (NEW)

## Execution Batches

### Batch 1: Per-ticker accuracy core + tests
Files: `accuracy_stats.py`, `signal_db.py`, `tests/test_ticker_signal_accuracy.py`

### Batch 2: Surface accuracy in reporting
Files: `reporting.py`

### Batch 3: On-chain data module + tests
Files: `onchain_data.py` (NEW), `shared_state.py`, `tests/test_onchain_data.py`

### Batch 4: Surface on-chain in reporting + config
Files: `reporting.py`, `config.json`

### Batch 5: Docs + cleanup
Files: `memory/todo.md`, `docs/SESSION_PROGRESS.md`
