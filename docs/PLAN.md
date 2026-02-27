# Focus Mode: Silver + BTC Probability System — Implementation Plan

**Date:** Feb 27, 2026

## What

Replace BUY/SELL labels with directional probabilities + accuracy context for focus instruments (XAG-USD, BTC-USD). The existing 31-ticker system stays running (Mode A). The new probability format is Mode B, switchable via config.

## Why

- XAG-USD signals are 71-83% accurate, BTC-USD is 44-54%
- Silver gained 19% spot / ~96% on 5x warrant over 9 days while the system sent 73 HOLDs
- Focus on instruments where we actually have edge, with accuracy-calibrated probabilities

## What Could Break

- `outcome_tracker.py` change (adding 3h horizon) — could slow backfill if not gated
- `reporting.py` changes — must not break existing compact summary format
- `main.py` changes — must not slow the main loop (hourly snapshots are lightweight)
- Config changes — must default to "signals" mode so Mode A is untouched

## Execution Order

### Batch 1: Phase 1 Core Modules (NEW files only — zero regression risk)
1. `portfolio/ticker_accuracy.py` — per-ticker accuracy + probability engine
2. `portfolio/cumulative_tracker.py` — hourly price snapshots + rolling changes
3. `portfolio/warrant_portfolio.py` — warrant state + leverage-aware P&L
4. `data/portfolio_state_warrants.json` — default empty state

### Batch 2: Phase 1 Tests
5. `tests/test_ticker_accuracy.py` — ~50 tests
6. `tests/test_cumulative_tracker.py` — ~30 tests
7. `tests/test_warrant_portfolio.py` — ~30 tests
→ Run tests, verify all pass

### Batch 3: Phase 1 Modifications (existing files, minimal changes)
8. `portfolio/outcome_tracker.py` — add "3h" horizon

### Batch 4: Phase 2 — Reporting Integration
9. `portfolio/reporting.py` — add focus_probabilities, cumulative_gains, warrant_portfolio to compact summary
10. `portfolio/journal.py` — warrant positions in context
11. `portfolio/main.py` — hourly snapshot call
→ Run tests, verify no regressions

### Batch 5: Phase 3 — Notifications
12. `portfolio/daily_digest.py` — morning daily digest (new, separate from existing 4h digest)
13. `portfolio/message_throttle.py` — analysis message cooldown
14. `tests/test_daily_digest.py`
15. `tests/test_message_throttle.py`
16. `portfolio/message_store.py` — add "daily_digest" to SEND_CATEGORIES
→ Run tests

### Batch 6: Phase 4 — Config & Docs
17. `config.json` — notification config section
18. `portfolio/telegram_poller.py` — /mode command
19. `CLAUDE.md` — Mode B format spec
→ Final test suite run

## Decisions

- ticker_accuracy.py uses existing load_entries() from accuracy_stats rather than duplicating SQLite access — keeps the single-source pattern
- 3h horizon added to HORIZONS dict in outcome_tracker — only fetches when entry is 3h+ old, no extra API calls for recent entries
- Warrant portfolio is separate from Patient/Bold portfolios — tracks real Avanza positions, not simulated
- Daily digest is a NEW module (daily_digest.py), not modifying existing digest.py — avoids breaking the 4h digest
