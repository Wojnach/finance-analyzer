# Research Implementation Plan — 2026-04-10 After-Hours Session

## Context

After-hours research session findings from Phase 0 signal audit:
- **Regime**: All 4 instruments in ranging. Low conviction environment.
- **System health**: Perfect — 0 errors, all 23 signal modules 100%.
- **Key finding**: Per-ticker accuracy reveals massive hidden variance.
  Global accuracy masks that ministral is 71.7% on MSTR but 20.4% on XAG.
  The per-ticker override (BUG-158) is working but missing directional fields.
- **12 of 32 active signals below 50%** at 1d global accuracy.
- **Directional gate too permissive**: 0.35 threshold misses macro_regime BUY (38.9%).

## Bugs & Problems Found

1. **Per-ticker accuracy override missing directional fields** (signal_engine.py:1870-1877)
   - Override copies `accuracy`, `total`, `correct` — but NOT `buy_accuracy`, `sell_accuracy`, `total_buy`, `total_sell`
   - Result: directional gate (line 826-837) falls back to overall per-ticker accuracy
   - Impact: directional asymmetries like ministral BUY 51.6% vs SELL 59.8% are invisible per-ticker

2. **`accuracy_by_ticker_signal()` lacks directional breakdown** (ticker_accuracy.py:16-74)
   - Only returns overall accuracy per ticker+signal
   - Cannot distinguish BUY vs SELL accuracy per ticker

3. **Module failures**: monte_carlo, price_targets, equity_curve (08:37 UTC) — non-critical, recurring.

## Improvements Prioritized (Impact × Ease)

### Tier 1: Implement NOW (high impact, easy)

**1. Per-Ticker Directional Accuracy** [MEDIUM EFFORT, HIGHEST IMPACT]
- Extend `accuracy_by_ticker_signal()` to return `buy_accuracy`, `sell_accuracy`, `total_buy`, `total_sell`
- Extend the BUG-158 per-ticker override (line 1870) to copy these directional fields
- This enables the directional gate to work per-ticker, preventing:
  - ministral voting BUY on XAG (20.4% overall → BUY likely worse)
  - macro_regime voting BUY on BTC (34.4% overall → BUY accuracy likely terrible)
- Files: `portfolio/ticker_accuracy.py`, `portfolio/signal_engine.py`

**2. Raise Directional Gate Threshold** [EASY, HIGH IMPACT]
- Current: `_DIRECTIONAL_GATE_THRESHOLD = 0.35`
- Proposed: `_DIRECTIONAL_GATE_THRESHOLD = 0.40`
- Additional signals caught: macro_regime BUY (38.9%), fibonacci SELL (35.9%), futures_flow (36-37%)
- macro_regime BUY currently passes overall gate (46.6%) but BUY direction is 38.9% — actively harmful
- Files: `portfolio/signal_engine.py` (1 line)

**3. Per-Ticker Directional Accuracy Cache** [MEDIUM EFFORT, HIGH IMPACT]
- The cached version (`accuracy_by_ticker_signal_cached()`) needs to include directional data
- This populates the data for improvement #1 to consume
- Files: `portfolio/accuracy_stats.py` (if cached there)

### Tier 2: Implement if time permits

**4. Signal Audit Deliverable** [EASY]
- Write `data/daily_research_signal_audit.json` with correlation analysis, regime performance
- Purely informational, no code changes

**5. Morning Briefing** [EASY]
- Synthesize all findings into `data/morning_briefing.json`
- Send Telegram summary

### Tier 3: Defer to backlog

- Walk-forward signal weight optimizer
- IC-based dynamic signal weighting (instead of accuracy-based)
- HMM regime blending (replace discrete regime labels with continuous probabilities)
- Transformer price prediction pipeline
- Multi-agent debate system

## Execution Order

### Batch 1: Per-Ticker Directional Accuracy (2 files + tests)
1. `portfolio/ticker_accuracy.py` — add directional fields to `accuracy_by_ticker_signal()`
2. `portfolio/signal_engine.py` — extend BUG-158 override to copy directional fields
3. `tests/test_ticker_accuracy.py` — test directional accuracy computation
4. `tests/test_signal_engine.py` — test directional gating with per-ticker data

### Batch 2: Directional Gate Threshold (1 file + tests)
1. `portfolio/signal_engine.py` — raise `_DIRECTIONAL_GATE_THRESHOLD` from 0.35 to 0.40
2. `tests/test_signal_engine.py` — verify macro_regime BUY gets gated at 38.9%

### Batch 3: Research Deliverables (data files only)
1. `data/daily_research_signal_audit.json` — Phase 3 deliverable
2. `data/daily_research_macro.json` — Phase 1 deliverable (from background agent)
3. `data/daily_research_quant.json` — Phase 2 deliverable (from background agent)
4. `data/daily_research_ticker_deep_dive.json` — Phase 2 deep dive
5. `data/morning_briefing.json` — Phase 8 deliverable

## Risk Assessment

- **Per-ticker directional accuracy**: Low risk — additive data, fail-closed (falls back to global accuracy if per-ticker data unavailable).
- **Directional gate 0.35→0.40**: Low risk — only affects 3-4 additional signal×direction pairs. The accuracy gate at 45% already catches most bad signals. This catches the ones that pass overall but have a terrible weak direction.
- **No config.json changes**: No API key or threshold changes to production config.
