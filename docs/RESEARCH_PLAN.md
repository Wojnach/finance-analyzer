# Research Session Plan — 2026-05-13

## Context

After-hours research. PPI surged 6% YoY (Iran war energy costs). Trump-Xi summit in
Beijing (May 14-15). Rate-cut expectations killed. XAG trending-up ($89), BTC trending-down
($79K), ETH trending-down ($2,250). All 28 Layer 2 invocations today = HOLD. Heavy signal
flip-flopping on all tickers.

## Key Findings

1. **Funding rate signal ENABLED at 30.8% accuracy** (743 sam, 1d). BUY-only, all wrong.
   0% activation on all tickers recently. Pure noise that escaped the 47% gate because
   it rarely votes — but when it does, it's harmful. MUST DISABLE.

2. **macro_regime at 47.0% ENABLED** (29,626 sam). BUY accuracy 41.3% = actively harmful
   on BUY side. 88% activation on XAG where it consistently votes BUY in a ranging market.
   Massive sample size confirms this is noise, not bad luck.

3. **claude_fundamental DISABLED at 57.9%** (11,581 sam). Disabled May 3 due to 19.8%
   recent crash from Opus BUY bias. But BUY accuracy 60.5% all-time. The bias was
   regime-specific (ranging market). Could re-enable with stricter bias detection.
   DECISION: Leave disabled — the 19.8% recent crash is too severe and root cause
   (LLM structural BUY bias) isn't fixed.

4. **Phantom voter problem**: On XAG-USD, 13 of 22 "enabled" signals output HOLD 100%
   of the time in last 50 entries. They never vote BUY/SELL but count toward MIN_VOTERS.
   Real consensus is driven by ~9 signals, not 22.

5. **Per-ticker 1d consensus still weak**: BTC 53%, XAG 49.7%, XAU 50%, ETH 49.4%,
   MSTR 46.5%. Only BTC has marginal edge. XAG improves at 3d (57.4%).

6. **Trigger sensitivity too high**: 28 invocations for 0 trades = wasted Claude compute.
   System correctly decided HOLD every time, but each invocation costs tokens/time.

## Implementation Plan

### Batch 1: Disable harmful signals (tickers.py) — LOW RISK
Add to DISABLED_SIGNALS:
- `funding`: 30.8% at 1d (743 sam). BUY-only, all wrong. 0% activation recently.
  Should have been caught by accuracy gate but escapes because it rarely votes.

### Batch 2: Disable macro_regime (tickers.py) — LOW RISK
- `macro_regime`: 47.0% at 1d (29,626 sam). BUY accuracy 41.3%. 88% activation on XAG.
  This is the single biggest noise source in XAG consensus.
  NOTE: This was NOT disabled yesterday because it was at 47% (right at gate threshold).
  But with 29K samples, this is statistically certain noise. The high-sample gate
  threshold is 50% for signals with 7000+ samples — macro_regime should be caught
  by this, but it's at 47% which is above the standard 47% gate but below the 50%
  high-sample gate. Verify whether the high-sample gate is actually catching it.
  If not, formal disable is needed.

### Batch 3: Fix news_event BUY-side (signal_engine.py or tickers.py) — LOW RISK
- news_event BUY accuracy: 20.5% (atrocious). SELL accuracy: 50.4% (marginal).
  The directional gate at 40% should catch this (20.5% < 40%), but verify it's
  actually being enforced. If not, add explicit directional blacklist.

### Batch 4: Tests — NO RISK
- Run test suite to verify changes don't break anything

## What NOT to change (deferred)
- claude_fundamental: disabled for good reason (LLM BUY bias not fixed)
- calendar: disabled for good reason (structural BUY bias, 29.3% recent)
- IC weighting: already implemented, working as designed
- Per-ticker accuracy gating: already at 45% threshold, working
- Regime-adaptive signal subsets: high effort, defer to dedicated session
- Phantom voter fix: needs careful design to not break MIN_VOTERS logic

## Execution Order
1. Create worktree `research/daily-2026-05-13`
2. Batch 1: Disable funding + commit
3. Batch 2: Disable/verify macro_regime + commit
4. Batch 3: Verify news_event directional gate + commit if fix needed
5. Run test suite
6. Merge, push, restart loops
7. Morning briefing + Telegram
