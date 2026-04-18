ultrathink

# Metals Swing Entry-Gate Hardening — 2026-04-18

Branch: `fix/entry-gates-20260418`
Worktree: `/mnt/q/finance-analyzer-gates`
Motivating incident: **2026-04-17 MINI L SILVER AVA 336 trade** (BUY 15.58,
SELL 14.79, −5.07% / −50.56 SEK, held 3.8h). Signal at entry: 9B/1S,
RSI 68.0, MACD +0.015, regime trending-up, confidence 0.66.

## Anatomy of the bad entry

Per the XAG signal timeline pulled from `data/metals_signal_log.jsonl`:

```
15:00 UTC  RSI 79.4  7B/3S  conf 0.40  MACD +0.22     SKIP (conf low)
15:11      RSI 75.3  6B/6S  conf 0.32              (mixed)
15:22      RSI 72.1  8B/2S  conf 0.37              (close but skip)
15:28      RSI 68.8  9B/2S  conf 0.74              (conf DOUBLED in 6min)
15:50      RSI 66.0  8B/2S  conf 0.80  MACD +0.001 SKIP (MACD not improving)
16:01      RSI 68.0  9B/1S  conf 0.66  MACD +0.015 SKIP (MACD not improving)
16:02      RSI 68.0  9B/1S  conf 0.66  MACD rising BUY @ 15.58  ← fires
16:12      RSI 66.8  8B/2S  conf 0.00              ← 10 min after buy
17:19      RSI 57.0  4B/2S  conf 0.21  REGIME→ranging  ← trend ends
19:50      sell @ 14.79 (-5.07%) — exit optimizer, session-end imminent
```

Four equation bugs are visible in this trace. Before coding, I debate each one
against its strongest counter-argument.

---

## Bug A — Signal persistence not required

**Claim.** Gate fires the instant all conditions pass ONCE. Yesterday, the 0.66
confidence was fleeting: 0.40 → 0.80 → 0.66 → **0.00 ten minutes after the buy**.
Adding a "conf ≥ threshold for N cycles" gate blocks entries into fleeting windows.

**Counter-argument.** Silver moves fast — in a real breakout the entry is worth
more the earlier it fires. Waiting 2-3 min means buying later in the same move,
which is often *worse*. The existing `MOMENTUM_ENTRY` override (relaxes to
conf 0.50 when a momentum candidate is fresh) exists precisely because fast
moves shouldn't be gated by persistence.

**Synthesis.** Both sides have merit. The key observation: the *conf jumping
to 0.66 and then collapsing to 0.00* is statistically fragile, regardless of
regime. A 2-cycle persistence check (≈2 min) costs minimal entry lag but
eliminates single-cycle phantom spikes. Keep the `MOMENTUM_ENTRY` override
(which already has its own hysteresis via `MOMENTUM_CANDIDATE_TTL_SEC`).

**Decision: SHIP**, 2-cycle persistence, applies to the standard-gate path.
Momentum-override path bypasses (momentum trades want to catch the move).

---

## Bug B — MACD improvement is direction-only, not level-aware

**Claim.** `MACD_IMPROVING_CHECKS=2` passes when MACD goes 0.001 → 0.015 (+14x
in %terms, but +0.014 absolute — noise). Meanwhile MACD had *decayed* from
+0.22 → +0.001 (lost 99%) in the hour before. The "improving" gate is fooled
by micro-recovery after major decline.

**Counter-argument.** MACD is a differential of EMAs — its scale depends on the
underlying price. A fixed threshold like "MACD > 0.05" works for XAG at 82
but is nonsense for BTC at 76000. Using "recent peak" introduces new fragility
(one-off spikes distort the baseline). The real signal we want is "momentum
is currently meaningful" — hard to define without asset-specific calibration.

**Synthesis.** Ratio gate works: **require MACD ≥ 30% of its recent 30-min
peak** (not a fixed threshold). This auto-scales to asset and regime. If
MACD peaked at +0.22 and current is +0.015, the ratio is 7% → fail. If MACD
peaked at +0.05 and current is +0.02, the ratio is 40% → pass.

**Decision: SHIP** as MACD decay protection. Use a ratio to sidestep the
asset-calibration problem. Keep existing `MACD_IMPROVING_CHECKS` (they're
complementary — "direction" + "meaningful level").

---

## Bug C — RSI_ENTRY_HIGH = 68 not regime-adjusted

**Claim.** Yesterday's RSI 68 in trending-up regime is textbook
buy-at-peak. Mean-reversion wisdom says wait for RSI<55 in established
uptrends.

**Counter-argument.** The Minervini/O'Neil school says the *opposite* — buy
breakouts above RSI 70, not pullbacks below RSI 50. In a strong
trending-up regime, RSI *stays* elevated; waiting for RSI<60 can mean
never entering. A hard-coded regime-based RSI cap can cost alpha in real
trends.

**Synthesis.** Both schools have statistical support, but the *specific
failure mode* yesterday wasn't "RSI too high" — it was "RSI was at 79 and
DECAYED to 68". Same RSI level, very different setups. Better gate: require
RSI to have been *rising* (or at least flat) over the last 5 cycles. If
RSI went 79 → 72 → 68, that's *declining*, reject. If RSI went 55 → 60 →
68, that's rising through, accept.

**Decision: SHIP** but as **RSI slope gate**, not regime-indexed level cap.
Require `RSI[current] >= RSI[5 cycles ago]` OR use a "`min(last 30 min RSI)`
< 55" check that proves we saw a dip. Keep existing `RSI_ENTRY_HIGH=68` cap
as a hard safety net.

---

## Bug D — No regime persistence / transition-risk check

**Claim.** Regime flipped from trending-up → ranging 77 min after entry.
Entering at the end of a regime phase is bad timing. Add a
"regime has been trending-up for ≥ N cycles" gate.

**Counter-argument.** If we require 30+ min of regime stability, we miss
the early-breakout entries where a regime just flipped. The whole point of
a regime detector is to recognize *new* regimes. Requiring age negates
that. Worse — "regime age" has no information about *future* regime
durability; many regimes are strongest right after they flip.

**Synthesis.** Regime persistence by itself is NOT a good gate — the
counter-argument is correct. The real signal is **transition-risk
convergence**: RSI decay + MACD decay + volume fading all at once. If all
three momentum indicators are *leaving*, the regime is about to flip
regardless of how long it's been active.

The MACD decay gate (Bug B) and RSI slope gate (Bug C) already capture
this indirectly. So Bug D is best addressed as a *side effect* of B+C,
not its own gate. I was wrong to frame it as "regime persistence".

**Decision: DROP as a standalone gate.** The MACD-decay and RSI-slope
gates already capture the "momentum is leaving" signal that was the real
issue. Adding a regime-age gate on top would create a third filter that
stacks without adding information (and would cost us early-breakout
entries).

---

## Summary after debate

| # | Original idea | Shipping? | Revised form |
|---|---------------|-----------|--------------|
| A | Signal persistence | ✅ YES | 2-cycle conf persistence, standard-gate path only (momentum override bypasses) |
| B | MACD absolute level | ✅ YES | MACD ≥ 30% of 30-min rolling peak (ratio, asset-agnostic) |
| C | Regime-adjusted RSI | ✅ YES (revised) | RSI slope ≥ 0 over last 5 cycles OR recent-30min RSI dipped below 55 |
| D | Regime persistence | ❌ DROP | Covered by A+B+C. Regime-age alone has no alpha. |

Net: 3 new gates, all in the standard-gate path in `_evaluate_entry`.
Momentum-override path unaffected (fast moves still fire on the relaxed
MOMENTUM_MIN_BUY_CONFIDENCE=0.50 gate).

## Counterfactual on yesterday's bad entry

Each new gate, evaluated against the 16:02 entry:

- **A** — Conf was 0.80 (15:50), 0.66 (16:01), 0.66 (16:02). 2-cycle
  persistence at threshold 0.60 = passes 16:02 (0.66, 0.66 both above). **No
  help on this specific incident.**
- **B** — MACD peaked at +0.22 at 15:00, entry MACD +0.015+. Ratio ≈ 7%.
  **Gate rejects (<30%). Would have blocked yesterday's entry.** ✓
- **C** — RSI timeline 15:00→16:02: 79, 77, 75, 72, 68, 66, 68, 68.
  Slope over 5 cycles (≈5 min) at entry: neutral (oscillating near 68).
  `min(30min)` = 66. 66 < 55? No. **Gate passes.** Partial help only
  (RSI had oscillated, not clearly declining at the exact moment).

So **B is the key gate** for yesterday's specific incident; A and C are
additional safety for other failure modes.

## Execution plan

### Batch 1 — config + state + persistence gate
- `data/metals_swing_config.py`: add `SIGNAL_PERSISTENCE_CHECKS`,
  `MACD_DECAY_PEAK_LOOKBACK`, `MACD_DECAY_MIN_RATIO`,
  `RSI_SLOPE_LOOKBACK_CHECKS`, `RSI_DIP_LOOKBACK_CHECKS`,
  `RSI_DIP_BELOW_LEVEL`. All with incident rationale inline.
- `data/metals_swing_trader.py`: add `confidence_history`,
  `rsi_history` to state; wire into `_evaluate_entry` as new gates.
  Use existing `macd_history` for decay-peak check.
- `tests/test_metals_swing_entry_gates.py`: 8+ tests covering each
  new gate (pass/skip pairs + the counterfactual on yesterday's entry).

### Batch 2 — verify + Codex review
- Full suite: `pytest tests/ -n auto --ignore=tests/integration`.
- `codex review --base main`. Address any P1/P2.

### Batch 3 — merge, push, restart
- Merge to main via no-ff.
- Push via cmd.exe (Windows git).
- Restart `PF-MetalsLoop` (DataLoop not touched).
- Clean up worktree + branch.

## Risk assessment

- **False positives going up** (blocking good entries): the most sensitive
  gate is B (MACD decay). Threshold 30% is conservative. Can loosen to 50%
  if it blocks too many legit breakouts. Monitor for 48h via logs.
- **False negatives going down** (letting bad entries through): most of the
  equation-noise entries should now fail at least one of A/B/C.
- **Backwards compatibility**: existing positions are not affected;
  only new-entry decisions consult the new gates. Momentum-override path
  unchanged so user's explicit fast-tick catches still work.
