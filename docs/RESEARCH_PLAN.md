# After-Hours Research Plan — 2026-05-26

## Bugs & Problems Found

### P0 (Fixed This Session)
1. **Stale credit_spread_risk override** — `signal_engine.py:710-713`
   re-enabled a signal at 20% recent accuracy for BTC/ETH. FIXED: removed override.
2. **crypto_evrp gate oscillation** — Enabled at claimed 80.5% (77 sam),
   degraded to 43.4% recent. Blended 47.0% at gate boundary. FIXED: formally disabled.

### P0 (Known, Not Fixed)
3. **Layer 2 agent timeouts** — 15+ triggers this week with no journal entry.
   Genuine timeouts, not a race condition. Root cause: agent subprocess produces
   no useful output within tier timeout. Needs prompt efficiency investigation.
4. **Avanza session expired** since May 23. Needs manual BankID re-login.

### P1 (Monitor)
5. **crypto_macro degrading** — 46.5% recent (310 sam). Below 47% gate,
   auto-gated at runtime. If trend continues, formally disable.
6. **qwen3 recent degradation** — 46.8% recent (124 sam) vs 59.7% all-time.
   Recent sample count low, may recover. Monitor.

## Improvements Prioritized (from research)

### Implemented This Session
1. Remove credit_spread_risk override — DONE
2. Disable crypto_evrp — DONE

### Defer to Backlog
3. **Close walk-forward weight loop** — connect trained weights to _weighted_consensus.
   All infrastructure exists. Est. 2 days. → IMPROVEMENT_BACKLOG.md
4. **Rolling IC weighting** — replace accuracy-only with EWMA Spearman IC.
   ic_computation.py needs per-horizon extension. Est. 3 days.
5. **Fix dynamic correlation groups** — use agreement rate instead of Pearson.
   Known bug. Est. 2 days.
6. **Extract gs_ratio_velocity signal** — shadow deploy. Est. 1 day.
7. **HMM regime detection** — formalize drift_regime_gate. Est. 4 days.

## Research Deliverables
- `data/daily_research_review.json` — Phase 0 daily review
- `data/daily_research_macro.json` — Phase 1 market research
- `data/daily_research_quant.json` — Phase 2 quant research
- `data/daily_research_signal_audit.json` — Phase 3 signal audit
- `data/daily_research_ticker_deep_dive.json` — Per-ticker deep dives (XAG, BTC)
