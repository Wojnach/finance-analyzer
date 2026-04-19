# Agent Review: signals-core

## P1 Findings
1. **IC multiplier rewards contrarian signals** — signals with negative IC still vote at 60% weight (signal_engine.py:1312-1333)
2. **ticker_accuracy missing neutral filter** — per-ticker accuracy inflated vs global (ticker_accuracy.py:60-61). Confidence: 92
3. **Directional accuracy not blended** — recent degradation invisible to directional gate (accuracy_stats.py:818-828). Confidence: 88
4. **ic_computation relative Path("data")** — silently disables IC weighting (ic_computation.py:19). Confidence: 95

## P2 Findings
1. **outcome_tracker TOCTOU** — backfill replace loses recent signal_log entries (outcome_tracker.py:430-446). Confidence: 88
2. **ministral applicable-count wrong** for metals/stocks (signal_engine.py:426,807-831). Confidence: 85
3. **Circuit breaker raw_candidates mismatch** with extra_info["_voters"] when horizon blacklisting active
4. **signal_history read-modify-write not thread-safe** (signal_history.py:53-82). Latent.
5. **accuracy_stats unbounded file read** (accuracy_stats.py:1166-1175)
6. **Circuit breaker high_gate_val math** — investigated, found correct. Non-issue.

## P3 Findings
1. ic_buy/ic_sell naming mismatch (ic_computation.py:119)
2. RSI 3h threshold divergence in outcome_tracker
3. blend_accuracy_data defaults diverge from signal_engine constants
