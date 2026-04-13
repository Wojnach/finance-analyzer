# Session Progress — Auto-Improvement (2026-04-13)

## Status: IN PROGRESS — Full test suite running

### Shipped Changes (3 commits on improve/auto-session-2026-04-13)

**Commit `b4e92a2`** — fix(signals): SC-I-001 + CROSS-001
1. **SC-I-001 fixed**: `_weighted_consensus()` was unconditionally re-applying regime gating,
   negating BUG-158 per-ticker exemptions. Added `regime_gated_override` parameter so the
   already-exempted set threads through without re-computation.
2. **CROSS-001 fixed**: `outcome_tracker` was reading `_votes` (post-gated) instead of
   `_raw_votes` (pre-gated), creating a dead-signal trap. Now uses `_raw_votes` with `_votes`
   fallback.
3. 6 new tests for both fixes.

**Commit `0036d3c`** — fix(main): OR-I-001 non-blocking executor shutdown
4. **OR-I-001 fixed**: `ThreadPoolExecutor` context manager's `__exit__()` called
   `shutdown(wait=True)`, blocking the main loop when threads hung past 180s timeout.
   Replaced with manual lifecycle: `shutdown(wait=False, cancel_futures=True)`.
5. 3 new tests verifying non-blocking shutdown behavior.

**Commit `711ff94`** — feat(signals): per-ticker signal gating
6. **news_event gated for ETH-USD**: 39.2% accuracy with 100% SELL bias. Added
   `_TICKER_DISABLED_SIGNALS` dict for static per-ticker signal disabling.
7. 4 new tests for per-ticker gating mechanism.

### What's Next
- Full test suite verification (running now)
- Merge to main, push, restart loops
- Monitor consensus accuracy improvement (check 2026-04-15)

### Remaining From Previous Sessions
- IC-weighted signal ensemble (AlphaForge, AAAI 2025) — highest-leverage next improvement
- BOCPD online regime detection — 80% unknown regime bottleneck
- Adaptive F&G thresholds — percentile-based
- Gate credit_spread_risk SELL for metals (auto-handled by directional gate at 40%)
