# Subsystem 1: Signals Core — Round 5 Findings

## CRITICAL (P1)

**SC-R5-1** — Utility boost can bypass accuracy gate by inflating sub-threshold signals above 47%.
`signal_engine.py:1996-2009`. A signal at 32% accuracy with positive avg_return gets boosted to 48% and clears the gate.
Fix: Apply utility boost only to weight calculation, not to gate decision.

**SC-R5-2** — SQL accuracy path in signal_db.py doesn't filter neutral outcomes (|change_pct| < 0.05%).
`signal_db.py:270-272`. JSONL path skips neutral outcomes; SQL counts them as correct. Inflates SQL accuracy.
Fix: Add _MIN_CHANGE_PCT filter to SQL accuracy methods.

**SC-R5-3** — Regime-accuracy overlay replaces blended accuracy, wiping directional fields (buy_accuracy, sell_accuracy).
`signal_engine.py:1944-1947`. When regime data replaces blended data and per-ticker data also lacks directional fields,
the directional gate falls back to overall accuracy, letting both BUY and SELL through when one should be gated.
Fix: Ensure ticker_accuracy.accuracy_by_ticker_signal() computes directional fields.

## HIGH (P2)

**SC-R5-4** — signal_history.py read-modify-write race. No lock on concurrent updates from ThreadPoolExecutor.
**SC-R5-5** — forecast_accuracy.py uses raw read_text() instead of file_utils.load_jsonl().
**SC-R5-6** — blend_accuracy_data uses max(at_samples, rc_samples) as total — fictional sample count.

## MEDIUM (P3)

**SC-R5-7** — _get_regime_gated returns _default for unrecognized horizons (12h falls to default gate).
**SC-R5-8** — outcome_tracker fear_greed derivation ignores sustained-fear gate.
