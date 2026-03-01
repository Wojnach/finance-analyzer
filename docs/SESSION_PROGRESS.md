# Session Progress — 2026-03-01 (On-Chain + Accuracy)

## Status: Batch 4 of 5 complete

### Completed
- **Batch 1** (commit `7550f56`): Per-ticker per-signal accuracy — `accuracy_by_ticker_signal()` + `ticker_signal_accuracy()` SQL + 16 tests
- **Batch 2** (commit `3f5952d`): Surface `signal_reliability` in compact + tier2 summaries
- **Batch 3** (commit `619841e`): BGeometrics on-chain data module — `onchain_data.py` + 26 tests
- **Batch 4** (commit `27f1e5c`): Surface `onchain` in compact + tier2 + config.json

### Remaining
- **Batch 5**: Docs cleanup (todo.md, CLAUDE.md update), full test suite, merge + push

### Test Results
- 42 new tests (16 accuracy + 26 on-chain), all passing
- 116 related tests passing (0 regressions)
- Full suite not yet run (Batch 5)

### Files Changed
| File | Change |
|------|--------|
| `portfolio/accuracy_stats.py` | +`accuracy_by_ticker_signal()`, +`top_signals_for_ticker()` |
| `portfolio/signal_db.py` | +`ticker_signal_accuracy()` SQL method |
| `portfolio/onchain_data.py` | NEW — BGeometrics fetcher + cache + interpretation |
| `portfolio/shared_state.py` | +`ONCHAIN_TTL` constant |
| `portfolio/reporting.py` | +`signal_reliability` section, +`onchain` section, propagation |
| `config.json` | +`bgeometrics` config block (token empty — TODO: MANUAL REVIEW) |
| `tests/test_ticker_signal_accuracy.py` | NEW — 16 tests |
| `tests/test_onchain_data.py` | NEW — 26 tests |
