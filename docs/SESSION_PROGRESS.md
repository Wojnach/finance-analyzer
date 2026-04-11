# Session Progress — Auto-Improve 2026-04-11

## Status: SHIPPING

Autonomous improvement session: direction-specific consensus weight + code quality.

### What shipped (3 commits on improve/auto-session-2026-04-11)

1. **`ea48a19`** docs(plan): improvement plan for auto-session 2026-04-11
2. **`11aaf27`** fix(signals): BUG-182 use directional accuracy as consensus weight
   - `signal_engine.py:842`: `_weighted_consensus()` now uses `buy_accuracy` as weight
     for BUY votes and `sell_accuracy` for SELL votes, falling back to overall accuracy
     when directional samples < 20.
   - 6 new tests in `test_weighted_consensus.py::TestDirectionalWeightScaling`
   - Expected impact: +2-5pp consensus accuracy by down-weighting signals in their weak direction
3. **`9a7325d`** chore: increase cache size to 512, fix volatility comment
   - `shared_state.py`: `_CACHE_MAX_SIZE` 256 → 512 (reduces cache thrashing)
   - `volatility.py`: fixed misleading comment about MIN_ROWS constraint
   - Updated 3 test assertions for new cache size

### Key findings during exploration

- **BUG-182 confirmed**: qwen3 BUY 30.4% vs SELL 74.3% — a 44pp asymmetry. Using overall
  accuracy (59.8%) as weight massively overvalued BUY votes. Now fixed.
- **Fear & Greed gate verified**: blended accuracy = 0.586 (alltime), not 0.357 as
  previously estimated. Recent data (7d) has 0 samples. fear_greed is correctly ungated.
- **Per-ticker gating already works**: ministral on XAG-USD (18.9%) is gated by the
  accuracy gate (< 0.45). BUG-158 per-ticker override ensures this.
- **Autonomous throttle is correct**: BUY/SELL signals always bypass the global throttle.
  The throttle only suppresses pure-noise HOLD messages. No per-ticker change needed.
- **Trade guards race condition is process-level**: Layer 2 runs as subprocess, not thread.
  `threading.Lock` wouldn't help; `atomic_write_json` is adequate.

### Agent findings verified (4 false positives identified)

| Reported Issue | Verdict |
|---|---|
| trigger.py first-of-day T3 broken | CORRECT code — `last_trigger_date` logic works |
| trigger.py signal flip reasons missing | By design — section #2 handles flips |
| signal_registry.py None load crash | Handled at signal_engine.py:1637-1638 |
| autonomous.py empty journal crash | Has `if prev_entries else None` guard |

### Production accuracy snapshot (active Tier 1, 1d horizon)

- Worst: metals_cross_asset XAU 17.9%, ministral XAG 18.9%, custom_lora BTC 23.5%
- All worst offenders are already gated by per-ticker accuracy gate (< 0.45)
- BUG-182 fix adds another layer: weak-direction votes now weighted by actual dir. accuracy

### Next priorities
1. Monitor BUG-182 impact over 48-72h (check if consensus accuracy improves)
2. Consider raising accuracy gate to 0.47 (would gate volatility_sig 0.453, trend 0.454)
3. IC-based weighting (Spearman correlation, not hit rate) — deferred to research
4. HMM regime detection (probabilistic, per-instrument) — deferred to research
