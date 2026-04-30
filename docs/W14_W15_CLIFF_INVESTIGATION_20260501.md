# W14→W15 Signal-Accuracy Cliff Investigation

**Date:** 2026-05-01
**Investigator:** research/w14-w15-cliff-20260501 worktree
**Hypothesis under test:** The W14→W15 cliff drop on 2026-04-07 was a secondary effect of the documented Layer 2 auth outage (2026-04-06 to 2026-04-09).
**Verdict:** **HYPOTHESIS REFUTED — LOW confidence the auth outage caused the cliff. HIGH confidence the cliff was a code-change artifact (commit `06527b83`, 2026-04-06 23:50 CET).**

## Smoking-gun evidence

### 1. Auth outage IS confirmed for that window — but its impact is on Layer 2, not signal accuracy

`data/layer2_journal.jsonl` daily entry counts:

| Date | Entries |
|---|---|
| 2026-04-01 | 56 |
| 2026-04-02 | 58 |
| 2026-04-03 | 6 |
| 2026-04-04 | **0** |
| 2026-04-05 | **0** |
| 2026-04-06 | **0** |
| 2026-04-07 | **0** |
| 2026-04-08 | **0** |
| 2026-04-09 | **0** |
| 2026-04-10 | **0** |
| 2026-04-11 | **0** |
| 2026-04-12 | **0** |
| 2026-04-13 | 21 |

Layer 2 was completely silent **04-04 to 04-12** (9 days). `data/contract_violations.jsonl` shows 396 `cycle_duration` violations during 04-06 to 04-09 with mean cycle 406s vs 60s target (6.8x slowdown). 18 cycles exceeded 600s.

But: **signal accuracy is computed by `outcome_tracker` independently of Layer 2.** Layer 2's silence cannot directly cause signal-accuracy drops. The slow cycles (cycle_duration violations) could in theory cause stale-snapshot attribution drift in outcome backfill, but that would degrade ALL signals uniformly — not produce the structured cliff observed.

### 2. The actual cause: commit `06527b83` at 2026-04-06 23:50 CET

```
06527b83 feat(signals): add crisis mode detection and tighten accuracy gating
```

Three concurrent changes, all directly affecting consensus accuracy from the next cycle onward:

1. **Crisis mode**: 0.6x weight penalty on trend signals (ema, trend, heikin_ashi, volume_flow) and 1.3x boost on mean-reversion (mean_reversion, calendar) when ≥3 macro signals show recent accuracy <35%.
2. **Group-leader gate threshold lowered 0.47 → 0.46**, force-HOLDing previously-borderline signals like sentiment.
3. **min_recent_samples lowered 50 → 30**, engaging the recent-accuracy blend on more signals (e.g. smart_money switches from 53.6% lifetime to 39.6% recent).

This commit landed exactly at the W14/W15 boundary. Mechanically, it is sufficient to produce a cliff because: signals previously contributing wins to consensus get force-HOLDed, and the remaining active voters' wins/losses are re-attributed without those wins propping the average. There is no need to invoke the auth outage at all.

### 3. The cliff was also amplified by a cascade of subsequent gating changes (W15-W16)

| Date | Commit | Change |
|---|---|---|
| 04-09 23:42 | `bf6f03c6` | Update correlation groups + directional accuracy gating |
| 04-10 23:43 | `6ec4be9c` | Per-ticker directional accuracy + raise directional gate 30%→40% |
| 04-11 15:14 | `6c7e2899` | `ACCURACY_GATE_THRESHOLD` 0.45 → **0.47** |
| 04-12 23:46 | `70603577` | Disable forecast, gate econ_calendar in ranging |
| 04-12 23:48 | `c97fe2da` | Tiered accuracy gate — 50% for high-sample signals |
| 04-15 23:50 | `57cb903a` | Per-ticker blacklist (the commit blamed by the 2026-04-16 investigation) |
| 04-19 23:44 | `e706ce7c` | **Conditional crisis-mode trend penalty** (effectively a softening fix) |

The April 16 investigation (`memory/project_accuracy_degradation_20260416.md`) attributed the cliff to commit `57cb903` (per-ticker blacklist), but `57cb903` shipped on 2026-04-15 — a week AFTER the W15 cliff began. That memory's conclusion is partially correct (the blacklist worsened W16), but the W15 cliff's primary trigger was `06527b83`.

### 4. Vote-distribution analysis confirms vote DIRECTIONS were stable, only weighting/gating changed

Compared W14_pre_outage (03-31 to 04-05), outage window (04-06 to 04-09), W15_post_outage (04-10 to 04-13) for Tier-1 (BTC/ETH/XAU/XAG/MSTR) using `signal_log.jsonl`:

- Most signals' BUY/HOLD/SELL ratios are within ±5pp across the three windows.
- A few signals (rsi, momentum_factors, volume_flow, trend) show slight increases in BUY rate post-outage — consistent with the Apr 9-10 metals-rally regime, not a measurement bug.
- No signal flipped its directional bias.

This rules out "stale snapshot directional bias" as the cliff mechanism.

### 5. Recovery test: signals returned to W14_pre baseline after fixes shipped

Per `data/accuracy_snapshots.jsonl` (newest 13 snapshots, 04-19 to 04-30):

- Overall lifetime consensus: **0.485** (stable across whole post-cliff window)
- Recent-7d consensus: **0.442 to 0.516**, mean ~0.48 — back to W14_pre band

Signals **did** recover after the auth fix, but they also recovered after the **gating-revert fixes** (`fd504d44` 04-16 trim blacklist, `e706ce7c` 04-19 conditional crisis penalty). Recovery cannot distinguish between the two hypotheses by itself, but combined with (2) it points to gating, not auth.

## What the auth outage DID cause (separately)

- 9 days of zero Layer 2 trade decisions (04-04 to 04-12) — operationally invisible to portfolio
- 396 cycle_duration violations with 6.8x slowdown
- The 2026-04-16 fix (`877221a` claude_gate detect_auth_failure narrowing) shipped 9 days later
- The whole reason CLAUDE.md now starts with the critical-errors check

But it did NOT cause the cliff in signal accuracy.

## Recommendation

No code fix warranted. The cliff was **intended behavior of the 04-06 gating tightening**, possibly over-corrected during the regime transition, and was already addressed by the cascade of follow-up fixes through 04-19. The April 16 investigation memory is partially wrong about the timeline (blames a 04-15 commit for a 04-07 cliff) but its proposed fixes (un-blacklist where per-ticker accuracy is high, add per-ticker gates, revert recency weights) were correct directionally and have already been shipped.

**Action items for the historical record:**
- Consider amending `memory/project_accuracy_degradation_20260416.md` with a note that commit `06527b83` (2026-04-06) was the cliff's actual trigger; commit `57cb903a` (2026-04-15) was a separate W16 worsening event.
- The "auth outage caused the cliff" hypothesis can be retired with HIGH confidence in refutation.
