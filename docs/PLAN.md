# Plan — Fix MSTR & Multi-Ticker Outcome Tracking (2026-04-20)

## Problem

Outcome tracking is broken for several /fin-* commands and for Layer 2 journal entries on MSTR/BTC/ETH. Specifically:

### Confirmed broken paths
1. **`system_lessons.json.by_command` only tracks fin-silver and fin-gold.** `fin_evolve.py:796` is a hardcoded list: `for cmd in ("fin-silver", "fin-gold"):`. Commands fin-crypto, fin-mstr, fin-btc, fin-eth are logged to `fin_command_log.jsonl` but never aggregated into lessons.

2. **`_find_price_at()` only reads `price_snapshots_hourly.jsonl`.** When snapshots don't cover the target timestamp (common for stocks after-hours, weekends, or when MSTR coverage is 42% of hourly slots), the function returns None and the backfill silently skips the entry. Result:
   - `journal_outcomes.jsonl`: 130 MSTR entries queued, 0 scored with outcome fields
   - `fin_command_log.jsonl`: 88 MSTR entries, 0 scored

3. **`backfill_outcomes()` processes single-ticker rows only.** `fin-crypto` entries have `tickers: ["BTC-USD", "ETH-USD", "MSTR"]` as a list plus nested `btc`/`eth`/`mstr` blocks with prices, but the backfill iterates `entry.get("ticker")` and skips multi-ticker rows entirely.

### Confirmed NOT broken (ruled out during exploration)
- `ticker_signal_accuracy_cache.json` — schema is `{horizon: {ticker: {signal: {...}}}, "time": ...}`. Top-level horizon keys are correct; MSTR/BTC-USD/ETH-USD are nested under `1d`. A prior agent mis-diagnosed this.
- `layer2_decision_outcomes.jsonl` — works (10 MSTR entries scored via `decision_outcome_tracker.py` which calls `_fetch_historical_price()` directly).
- `price_snapshots_hourly.jsonl` — MSTR is captured when US market is open (~42% of hourly slots). The gaps are handled by the fallback in Batch 2.

## Why this matters

- **MSTR has 130 journal verdicts over 2 months with zero outcome scoring.** We can't calibrate MSTR signal accuracy.
- **fin-crypto and fin-mstr commands never get scored.** When the user asks "how accurate was your last MSTR call?" we literally have no data.
- **`system_lessons.json.by_ticker` for MSTR is missing** because both paths above are blocked.

## Batches

### Batch 1 — Dynamic `by_command` in `fin_evolve.py`
**Risk:** Low. No data path changes, just lesson generation.
**Files:**
- `portfolio/fin_evolve.py` (replace hardcoded list at line 796)
- `tests/test_fin_evolve.py` (add test — create if missing)

**Change:** Replace `for cmd in ("fin-silver", "fin-gold"):` with dynamic detection from the scored entries.

**Verification:** `system_lessons.json.by_command` contains all commands present in scored data. Existing fin-silver/fin-gold entries unchanged.

### Batch 2 — Live-price fallback in `_find_price_at()`
**Risk:** Medium. Adds network dependency to the backfill cycle.
**Files:**
- `portfolio/fin_evolve.py` (`_find_price_at` fallback to `outcome_tracker._fetch_historical_price`)
- `tests/test_fin_evolve.py` (mock-based test)

**Change:** When snapshot lookup returns None, call the live API fallback (existing, used by decision_outcome_tracker). Wrap in try/except; limit retries to 1; bail if ticker/timestamp out of supported range.

**Verification:** Unit test with mocked API. Integration: running backfill on 130 MSTR queued entries scores them.

### Batch 3 — Multi-ticker fin-crypto backfill
**Risk:** Medium. Changes how fin_command_log backfill iterates entries.
**Files:**
- `portfolio/fin_evolve.py` (`backfill_outcomes()` — handle `tickers` list with per-ticker nested blocks)
- `tests/test_fin_evolve.py` (test for multi-ticker entries)

**Change:** When `entry.get("ticker")` is None and `entry.get("tickers")` is a list, iterate per-ticker blocks (`entry["btc"]`, `entry["eth"]`, `entry["mstr"]`) to score each. Write outcomes back as `entry["btc"]["outcome_1d_pct"]` etc.

**Verification:** fin-crypto entries in `fin_command_log.jsonl` get scored. Test with sample multi-ticker entry.

### Batch 4 — Run backfill and verify
**Risk:** Low. One-shot data fixup.
**Steps:**
1. Run `.venv/Scripts/python.exe portfolio/fin_evolve.py` standalone
2. Verify `journal_outcomes.jsonl` MSTR entries have `outcome_1d_pct` / `outcome_3d_pct`
3. Verify `system_lessons.json.by_command` contains fin-crypto, fin-mstr
4. Verify `system_lessons.json.by_ticker` contains MSTR with >0 scored verdicts

### Batch 5 — Adversarial review
- Spawn `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` agents in parallel on the diff
- Run codex adversarial review on branch SHA
- Address P1/P2 findings, document P3 decisions

### Batch 6 — Tests + ship
- `.venv/Scripts/python.exe -m pytest tests/ -n auto --timeout=60`
- Fix regressions
- Merge to main, push via `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`

## What could break

- **Rate limits** if many entries hit the live API. Mitigate: reuse `_yfinance_limiter`, skip entries older than 30 days, cap retries to 1.
- **Double-scoring from fin-crypto + single-ticker commands.** Mitigate: treat fin-crypto as supplemental — only score if that (ticker, timestamp) pair isn't already scored from a single-ticker entry.
- **Mass write on first backfill** of 218 entries. Mitigate: acceptable — one-shot cost; subsequent runs are idempotent.

## Order of execution
1. Batch 1 (trivial)
2. Batch 2 (enables Batch 3)
3. Batch 3 (multi-ticker)
4. Batch 4 (run + verify data)
5. Batch 5 (adversarial review)
6. Batch 6 (tests + ship)
