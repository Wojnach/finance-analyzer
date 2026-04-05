# After-Hours Research Plan — 2026-04-05

## Context

Sunday session. All instruments in "ranging" regime. 11/30 signals below 45% recent
accuracy; 6 signals above 58%. System correctly staying flat (HOLD all strategies since
Apr 3). Per-ticker consensus accuracy reveals severe problems: AMD 24.8%, GOOGL 31.3%,
META 34.2%, AMZN 39.0%. Top signals (econ_calendar 66.9%, claude_fundamental 62.3%)
have zero recent votes.

## Bugs & Problems Found

1. **No per-ticker consensus accuracy gate.** AMD at 24.8% consensus accuracy means the
   system is wrong 3/4 of the time on AMD, yet it still generates non-HOLD consensus.
   File: `portfolio/signal_engine.py`, end of `generate_signal()`.

2. **Ranging regime winners under-boosted.** `ministral` (68.0% recent) has NO ranging-specific
   boost in `REGIME_WEIGHTS['ranging']`. `fibonacci` at 1.6x could go higher.
   File: `portfolio/signal_engine.py`, line 126-161.

3. **EWMA accuracy not used in main voting.** `signal_accuracy_ewma()` exists in
   `accuracy_stats.py` but the main `generate_signal()` uses 70/30 blend instead of EWMA.
   EWMA with halflife=5d would be smoother and avoid the cliff at exactly 7 days.
   File: `portfolio/accuracy_stats.py` line 137, `portfolio/signal_engine.py` line 1397-1419.

4. **Regime gating too aggressive for short horizons.** 13 signals gated at `_default` in
   ranging, but several work at 3h (news_event 58.5%, smart_money 53.1%). The horizon-aware
   gating exists but the 3h/4h allowlist is too narrow.
   File: `portfolio/signal_engine.py`, line 169-221.

## Improvements Prioritized (Impact × Ease)

### Batch 1: Per-ticker consensus gate (HIGH impact, EASY)
Add a per-ticker consensus accuracy gate at the end of `generate_signal()`. If a ticker's
historical consensus accuracy (1d, 50+ samples) is below 38%, force HOLD. This prevents
actively harmful recommendations on AMD, GOOGL, META.

**Files:** `portfolio/signal_engine.py`
**Tests:** `tests/test_signal_engine.py` — add test for per-ticker consensus gate
**Risk:** Low — only affects tickers where we're already wrong most of the time

### Batch 2: Boost ranging-regime signal weights (MEDIUM impact, EASY)
- Add `ministral: 1.4` to `REGIME_WEIGHTS['ranging']`
- Boost `fibonacci` from 1.6 to 1.8 in ranging
- Boost `mean_reversion` from 1.5 to 1.7 in ranging
- Add `macd: 1.3` to ranging (58.7% recent)
- Add `rsi: 1.3` to ranging (52.7% recent but historically stable)

**Files:** `portfolio/signal_engine.py`
**Tests:** existing regime weight tests
**Risk:** Low — only adjusts relative weights within ranging regime

### Batch 3: Use EWMA accuracy for blending (MEDIUM impact, MEDIUM)
Replace the 70/30 recent/alltime blend with EWMA (halflife=5d). The EWMA function already
exists and is tested. This gives smoother transitions as signals enter/exit accuracy.

**Files:** `portfolio/signal_engine.py`, `portfolio/accuracy_stats.py`
**Tests:** `tests/test_accuracy_stats.py` — verify EWMA blend produces same rankings
**Risk:** Medium — changes the accuracy data feeding into all weighting. Need to verify
that EWMA doesn't cause unexpected gate transitions.

### Batch 4: Expand 3h horizon allowlist for ranging (LOW-MEDIUM impact, EASY)
Add `news_event`, `smart_money`, `volatility_sig`, `momentum_factors`, `trend` to the
3h ranging allowlist (they have >50% accuracy at 3h even in ranging).

**Files:** `portfolio/signal_engine.py`, line 190
**Tests:** existing regime gating tests
**Risk:** Low — only affects 3h/4h horizon, which is already more permissive

## What to Defer

1. **Per-ticker signal selection** — Different signal subsets per ticker. Hard to implement
   and test correctly without per-ticker backtesting infrastructure.
2. **Meta-learner retraining** — Requires GPU and longer backtest. Defer to next session.
3. **Dynamic correlation penalty** — Low impact for implementation effort.

## Execution Order

1. Batch 1 (per-ticker consensus gate) — highest impact, prevents harm
2. Batch 2 (ranging weight boost) — easy, improves signal quality
3. Batch 3 (EWMA blend) — requires more testing
4. Batch 4 (3h allowlist) — small improvement, easy to verify
5. Run full test suite after all batches
6. Commit, merge, push
