# Adversarial Review: signals-core subsystem (2026-05-08)

## Critical Findings

P1: portfolio/signal_engine.py:2246
BUG: _confluence_score() tie-break uses >= instead of >, creating silent majority flip.
Problem: `majority_dir = "BUY" if buy_count >= sell_count else "SELL"` assigns BUY when counts are equal (tie). A 50-50 vote should not get a tie-break to BUY. When vote is 3 BUY / 3 SELL / 2 HOLD, this arbitrarily tags "majority_dir" as BUY.
Fix: Change to `>`. Line 2246: `majority_dir = "BUY" if buy_count > sell_count else "SELL"`.

P1: portfolio/signal_engine.py:2229-2232
BUG: Consensus decision gate requires `buy_conf >= 0.5`, but with equal weights (buy_weight == sell_weight) both can be exactly 0.5. Both conditions can be true simultaneously.
Problem: If buy_weight == sell_weight, then buy_conf == sell_conf == 0.5. Line 2229 evaluates `0.5 > 0.5 (false) and 0.5 >= 0.5 (true)` = false. Line 2231 evaluates `0.5 > 0.5 (false)` = false. Both fail, returns HOLD. But if floating-point rounding makes one side 0.50000001, the first true branch fires and emits trade on noise. The logic is order-dependent on rounding.
Fix: Require strict majority (>0.5), not equality. Line 2229: `if buy_conf > 0.5:` and line 2231: `if sell_conf > 0.5:`.

P2: portfolio/accuracy_stats.py:920
ISSUE: blend_accuracy_data() blends with fixed weights (0.70 recent, 0.30 all-time) but does not validate that at_acc and rc_acc are finite. A NaN in rc_acc poisons the blend.
Problem: Lines 918-920 compute `blended = w * rc_acc + (1 - w) * at_acc` without checking rc_acc and at_acc. If rc_acc is NaN (from corrupted cache), blended becomes NaN, and _safe_accuracy() catches it downstream but the damage is already in the cache. A signal at NaN will clear the 47% accuracy gate (since `None < 0.47` is False).
Fix: Validate rc_acc and at_acc with _safe_accuracy() before blend: `rc_acc = _safe_accuracy(rc_acc, 0.5); at_acc = _safe_accuracy(at_acc, 0.5)`.

P2: portfolio/signal_engine.py:2507-2509
BUG: Dynamic MIN_VOTERS uses stale `_voters` count that predates persistence filter.
Problem: `active_voters = extra_info.get("_voters_post_filter", extra_info.get("_voters", 0))` reads `_voters` which is counted BEFORE _apply_persistence_filter() removes non-persistent signals. If 4 signals persist, but 6 were initially active, `_voters=6` but only 4 votes go to consensus. Dynamic MIN_VOTERS at 3.5 (rounded to 4 in ranging) could accept the consensus even though only 4 true voters remain. The gate uses stale count.
Fix: Track `_voters_post_filter` before calling _weighted_consensus: count non-HOLD signals in the filtered_votes dict, store that.

P2: portfolio/outcome_tracker.py:469
EDGE: change_pct calculation doesn't guard against base_price == hist_price (division by non-zero but no change detection).
Problem: Lines 446-447 check `if not base_price or base_price <= 0` to skip, but allows `hist_price == base_price`. When hist_price == base_price, change_pct rounds to 0.00, which is fine. But no early-continue, so the outcome dict is written with change_pct=0 even though the signal had no predictive value (outcome was neutral). Accuracy tracking includes the 0-change cases. Not a crash, but inflates sample counts for neutral outcomes.
Fix: Add check before computing: `if abs(hist_price - base_price) < 0.01: continue` (skip outcomes with <1 cent change).

P2: portfolio/signal_engine.py:3329-3333
LOGIC: MIN_VOTERS selection for "Now" timeframe (stock/metals consensus) is independent of asset class.
Problem: Lines 3331-3333 use `MIN_VOTERS_STOCK` for both STOCK_SYMBOLS and METALS_SYMBOLS. Metals have 28 active signals vs crypto's 30. If MIN_VOTERS_STOCK=3 but metals consensus happens to have only correlated Fibonacci/Trend signals active (10 candidates pre-gate, 2 post-gate due to correlation), the consensus fails the quorum even though 2 is below spec. The gate is not asset-class aware.
Fix: Use `_dynamic_min_voters_for_regime(regime)` like the 1d path does, or define MIN_VOTERS_METALS = 3 explicitly and route to it.

P1: portfolio/signal_engine.py:264-310
RISK: _apply_persistence_filter() cold-start seeds state with all signals at >= _PERSISTENCE_MIN_CYCLES if vote != "HOLD", allowing all signals to pass on cycle 1.
Problem: Line 284: `"cycles": _PERSISTENCE_MIN_CYCLES if vote != "HOLD" else 0`. On the very first call for a ticker, if a signal votes BUY, it gets cycles=2 (and min is 2), so it passes the filter immediately. The intent of persistence is to require 2+ CONSECUTIVE votes, but cycle 1 doesn't have a prior vote — the signal was never persistent. The cold-start logic defeats the filter.
Fix: Seed with cycles=0 for all signals, regardless of vote. Filter only activates after 2+ cycles have history: `"cycles": 0`.

P2: portfolio/signal_engine.py:1421-1441
ISSUE: _count_active_voters_at_gate() applies directional gate but does NOT apply persistence filter.
Problem: The function is called to count voters before consensus, but it operates on raw unfiltered votes. If persistence filter would remove 5 signals, _count_active_voters_at_gate returns count of 8 (pre-filter), but consensus sees 3 (post-filter). The circuit breaker uses the pre-filter count to decide whether to relax the accuracy gate, creating a disconnect.
Fix: Pass filtered_votes (post-persistence) to _count_active_voters_at_gate, not raw votes.

P1: portfolio/accuracy_stats.py:934
BUG: Blended "correct" field computed as `int(round(blended * total))` assumes homogeneous mix, breaks on mode transitions.
Problem: Line 934 reconstructs "correct" from "accuracy * total", but blended accuracy is a weighted average that doesn't directly map to integer correct counts. If at_acc=0.70 over 100 samples (70 correct) and rc_acc=0.50 over 30 samples (15 correct), total=100 and blended≈0.68, then "correct" becomes round(0.68*100)=68, not the actual 70+15=85. The reconstructed count is inconsistent with the source data. Downstream code that uses this "correct" field (e.g., for sorting) gets stale counts.
Fix: Merge the actual "correct" counts from at/rc like total: `result["correct"] = (at.get("correct", 0) or 0) + (rc.get("correct", 0) or 0)` then recompute accuracy. Or don't emit "correct" from blend at all (it's derived).

## Summary

7 findings: 1 critical consensus logic flaw (P1:2246), 1 critical gate bypass (P1:2229), 1 cold-start persistence bypass (P1:264), 1 stale voter count (P2:2507), 1 NaN propagation (P2:920), 1 derived-field inconsistency (P2:934), 1 consensus quorum mismatch (P2:3333), 1 outcome inflation edge (P2:469).
