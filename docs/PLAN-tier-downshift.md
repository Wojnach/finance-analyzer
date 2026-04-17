# PLAN — Option P: Confidence-aware tier downshift (T2→T1 for low-conviction)

Date: 2026-04-17
Branch: `fix/tier-downshift-low-conviction`
Worktree: `/mnt/q/finance-analyzer-tiershift`
Base SHA: `7c9f4ad1` (origin/main)

## Problem

Layer 2 Claude Code invocations consume the Max subscription token window.
On 2026-04-16, 27 trigger-causes fired across ~21 T2/T3 invocations in ~5h.
10 of those were XAU alone, all returning HOLD — low-conviction signals that
don't benefit from T2's 40-turn budget.

See `docs/PLAN-trigger-noise.md` for the full option menu (A-H). User picked
**Option P** — downshift tier rather than skip invocation — because it
preserves every trigger AND every signal/accuracy datum, trading only
analysis depth on low-conviction cases.

## Design

### Where

`portfolio/trigger.py::classify_tier()` — the single gate that decides T1/T2/T3
before `invoke_agent()` spawns Claude.

### What

Add a post-classification downshift step: if tier is T2 AND all trigger reasons
are either (a) low-conviction consensus crossings (`<40%`) or (b) fade flips
(`*->HOLD sustained`), downshift to T1.

Parse existing trigger-reason strings:
- Consensus: `"XAU-USD consensus BUY (34%)"` → confidence 0.34
- Fade flip: `"XAU-USD flipped BUY->HOLD (sustained)"` → fade
- Direction flip: `"XAU-USD flipped BUY->SELL (sustained)"` → NOT fade, don't downshift
- Price move / F&G / post-trade: don't downshift

**New constant:** `TIER_DOWNSHIFT_CONFIDENCE = 0.40`

### What does NOT change

- Trigger firing — all triggers still fire
- Signal log — unchanged semantics
- Accuracy cache — unchanged
- Consensus math in `signal_engine.py` — untouched
- SwingTrader, autonomous, dashboard — unaffected (don't read tier)
- T3 logic (first-of-day, F&G extreme, periodic full review) — preserved

### Why not downshift T3

T3 fires on "first trigger of day" and F&G extremes and 4h periodic review.
These deserve full budget even if the triggering ticker is low-conviction;
they're orthogonal to the ticker's current conviction. Leave T3 alone.

## Impact (based on 2026-04-16 data)

| Today | Current | With P |
|---|---|---|
| XAU 3 noise consensus (30-34%) | T2 × 40 turns = 120 | T1 × 15 turns = 45 |
| XAU 5 fade flips (*→HOLD) | T2 × 40 turns = 200 | T1 × 15 turns = 75 |
| ETH/BTC/XAG high-conviction | T2/T3 (unchanged) | T2/T3 (unchanged) |
| MSTR 68% SELL | T2 | T2 (above threshold) |

**Net:** ~200 turns/day saved (~20-25% budget reduction).

## What could break

1. **Classify_tier tests** — existing tests assert T2 on specific consensus
   strings. Need new tests that use high-conviction reasons (≥40%) so they
   still return T2.
2. **Edge case: mixed-reason triggers.** `"XAU-USD consensus BUY (30%); BTC-USD flipped BUY->SELL (sustained)"` — should NOT downshift because BTC direction-flip is meaningful. Test this.
3. **Reason string drift.** If `check_triggers` ever changes its format (e.g.
   drops the `(NN%)` suffix), downshift silently fails open (all treated as
   high-conviction). That's the safe failure mode — we'd over-invoke, not
   under-invoke. Document this.
4. **Perception gate interaction.** Gate currently skips T1 if no non-HOLD
   signals. With P, more triggers will be T1, so more may be skipped by the
   gate. This compounds the savings — not a bug, but worth noting.
5. **Telegram format.** T1 prompt tells Claude "brief journal + short Telegram".
   Low-conviction triggers will produce terser messages. User goal (less noise)
   is aligned. Not a concern.

## Batches

### Batch 1: Helper + downshift in classify_tier
Files:
- `portfolio/trigger.py` — add `TIER_DOWNSHIFT_CONFIDENCE`, `_should_downshift_to_t1()`, apply in `classify_tier()`

### Batch 2: Tests
Files:
- `tests/test_trigger_core.py` — extend existing TestClassifyTier classes with downshift cases

### Batch 3: Docs
Files:
- `docs/PLAN-trigger-noise.md` — mark P shipped with SHA
- `docs/SESSION_PROGRESS.md` — new entry at top

## Rollback

Revert the single commit on `trigger.py`. Constant `TIER_DOWNSHIFT_CONFIDENCE`
is the only config-like parameter; to disable without revert, set it to 0.0
(no reason will ever be below 0% confidence).

## Parallel-session safety

Three sessions in flight today. Touching only `portfolio/trigger.py` and
`tests/test_trigger_core.py` — the accuracy-gating branch touches
`signal_engine.py`, the adversarial-review-2 branch touches signal-engine
circuit breaker. No file overlap. Merge order immaterial.

## Execution order

1. Plan commit (this file)
2. Batch 1 code commit
3. Batch 2 test commit
4. Run targeted tests: `pytest tests/test_trigger_core.py -xvs`
5. Run full test suite: `pytest tests/ -n auto --timeout=60`
6. Codex review (per GUIDELINES.md)
7. Address findings, commit
8. Merge to main (ff if possible), push via Windows git
9. Update SESSION_PROGRESS.md, commit
10. Restart PF-DataLoop so new classify_tier logic is live
11. Clean up worktree + branch
