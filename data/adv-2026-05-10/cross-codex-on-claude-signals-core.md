OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-signals-core
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e127e-5e31-7d12-b953-5ed286124056
--------
user
You are an adversarial code reviewer cross-critiquing another reviewer's findings.

CONTEXT
- Codebase: Q:\finance-analyzer (autonomous trading system, Python).
- Subsystem: signals-core
- The other reviewer (Claude Opus 4.7) audited this subsystem and produced
  the findings below.
- Your job: critique each finding by inspecting the actual source files and
  deciding whether the finding is correct.

PROCEDURE
For each finding in the list below:
1. Open the cited file(s) at the cited line(s) in the working tree.
2. Decide one of:
   - CONFIRMED: bug/issue is real, severity is reasonable.
   - PARTIAL: there is something there, but the analysis is wrong in a
     specific way (e.g., wrong line, wrong cause, wrong severity).
   - FALSE-POSITIVE: the finding is wrong; the code does not have the
     described problem. State why concretely (cite the actual code).
3. If you find a NEW issue while inspecting, list it under "New findings".

OUTPUT FORMAT (Markdown, no preamble)
## Verdicts

- [<orig severity>] <one-line restatement> — file:line
  Verdict: CONFIRMED | PARTIAL | FALSE-POSITIVE
  Reason: <one sentence citing actual code>
  (Adjustment: <if PARTIAL, what's actually wrong>)

## New findings (you, not Claude)

- [P1|P2|P3] <one-line> — file:line
  <one paragraph>

## Summary
- Confirmed: N
- Partial: N
- False-positive: N
- New from you: N

CLAUDE'S FINDINGS TO CRITIQUE:
=== BEGIN ===
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
=== END ===

Use git/grep/read tools as needed. Be ruthless and cite code. Do not soften.

codex
I’m checking each cited finding against the working tree, starting by opening the referenced sections in `signal_engine.py`, `accuracy_stats.py`, and `outcome_tracker.py` so the verdicts are grounded in the actual code.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_confluence_score|_weighted_consensus|_apply_persistence_filter|_count_active_voters_at_gate|blend_accuracy_data|change_pct|MIN_VOTERS|_voters_post_filter|_voters" portfolio/signal_engine.py portfolio/accuracy_stats.py portfolio/outcome_tracker.py' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem portfolio | Select-Object Name' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 1211ms:
portfolio/outcome_tracker.py:135:        total_voters = buy_count + sell_count
portfolio/outcome_tracker.py:144:            "total_voters": total_voters,
portfolio/outcome_tracker.py:450:                continue  # BUG-220: skip — no base price to compute change_pct
portfolio/outcome_tracker.py:471:                change_pct = round(
portfolio/outcome_tracker.py:480:                    "change_pct": change_pct,
portfolio/outcome_tracker.py:490:                            round(hist_price, 2), change_pct, outcome_ts_str,
portfolio/accuracy_stats.py:170:def _vote_correct(vote, change_pct, min_change_pct=None):
portfolio/accuracy_stats.py:174:    Outcomes within ±min_change_pct are considered noise and should not count
portfolio/accuracy_stats.py:177:    threshold = min_change_pct if min_change_pct is not None else _MIN_CHANGE_PCT
portfolio/accuracy_stats.py:178:    # 2026-04-22: some outcome entries have change_pct=None (missing backfill
portfolio/accuracy_stats.py:182:    if change_pct is None or abs(change_pct) < threshold:
portfolio/accuracy_stats.py:184:    if vote == "BUY" and change_pct > 0:
portfolio/accuracy_stats.py:186:    return bool(vote == "SELL" and change_pct < 0)
portfolio/accuracy_stats.py:206:    # 2026-04-22 follow-up: count outcomes we skip because change_pct is None.
portfolio/accuracy_stats.py:210:    null_change_pct_skipped = 0
portfolio/accuracy_stats.py:225:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:226:            if change_pct is None:
portfolio/accuracy_stats.py:227:                null_change_pct_skipped += 1
portfolio/accuracy_stats.py:234:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:249:    if null_change_pct_skipped > 0:
portfolio/accuracy_stats.py:250:        pct = 100.0 * null_change_pct_skipped / total_outcomes_seen
portfolio/accuracy_stats.py:253:            "change_pct=None — check outcome_tracker / signal_db backfill",
portfolio/accuracy_stats.py:254:            horizon, null_change_pct_skipped, total_outcomes_seen, pct,
portfolio/accuracy_stats.py:344:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:351:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:423:            change_pct = outcome.get("change_pct")
portfolio/accuracy_stats.py:424:            if change_pct is None:
portfolio/accuracy_stats.py:434:                if abs(change_pct) < _MIN_CHANGE_PCT:
portfolio/accuracy_stats.py:440:                if (vote == "BUY" and change_pct > cost_pct) or (vote == "SELL" and change_pct < -cost_pct):
portfolio/accuracy_stats.py:501:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:502:            result_val = _vote_correct(consensus, change_pct)
portfolio/accuracy_stats.py:536:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:537:            result_val = _vote_correct(consensus, change_pct)
portfolio/accuracy_stats.py:601:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:602:            result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:627:      - BUY  → +change_pct   (positive = correct direction)
portfolio/accuracy_stats.py:628:      - SELL → -change_pct   (negative change = correct direction → positive return)
portfolio/accuracy_stats.py:630:    Neutral outcomes (|change_pct| < _MIN_CHANGE_PCT) are skipped.
portfolio/accuracy_stats.py:695:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:702:            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
portfolio/accuracy_stats.py:712:                    dir_return = change_pct
portfolio/accuracy_stats.py:714:                    dir_return = -change_pct
portfolio/accuracy_stats.py:863:def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
portfolio/accuracy_stats.py:942:        # _weighted_consensus's directional gate/weighting at its raw
portfolio/accuracy_stats.py:1344:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:1352:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1754:                change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:1760:                    result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1814:    gating in signal_engine._weighted_consensus().
portfolio/accuracy_stats.py:1836:            change_pct = outcome.get("change_pct", 0)
portfolio/accuracy_stats.py:1842:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:2025:            change_pct = outcome.get("change_pct")
portfolio/accuracy_stats.py:2026:            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
portfolio/accuracy_stats.py:2029:            correct = (consensus == "BUY" and change_pct > 0) or \
portfolio/accuracy_stats.py:2030:                      (consensus == "SELL" and change_pct < 0)
portfolio/signal_engine.py:263:def _apply_persistence_filter(votes: dict[str, str], ticker: str | None) -> dict[str, str]:
portfolio/signal_engine.py:603:MIN_VOTERS_CRYPTO = 3  # crypto has 30 signals (8 core + 22 enhanced; ml disabled) — need 3
portfolio/signal_engine.py:604:MIN_VOTERS_STOCK = 3  # stocks have 24-26 signals (7 core + 17-19 enhanced, GPU-dependent) — need 3
portfolio/signal_engine.py:607:# circuit-breaker precondition. Placing here (after MIN_VOTERS_*) keeps the
portfolio/signal_engine.py:608:# relationship explicit and prevents silent drift if the base MIN_VOTERS_*
portfolio/signal_engine.py:610:_MIN_VOTERS_BASE = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)
portfolio/signal_engine.py:613:_POST_EXCLUSION_MIN = _MIN_VOTERS_BASE
portfolio/signal_engine.py:614:# Lone-signal escape floor: raised from 2 to _MIN_VOTERS_BASE (3) because a
portfolio/signal_engine.py:618:_LONE_SIGNAL_FLOOR = _MIN_VOTERS_BASE
portfolio/signal_engine.py:623:assert MIN_VOTERS_CRYPTO > 0 and MIN_VOTERS_STOCK > 0, (
portfolio/signal_engine.py:624:    "MIN_VOTERS_* must be positive")
portfolio/signal_engine.py:1422:def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
portfolio/signal_engine.py:1440:    # the same logic in `_weighted_consensus` (line ~2068).
portfolio/signal_engine.py:1498:def _dynamic_min_voters_for_regime(regime):
portfolio/signal_engine.py:1544:    Uses `_count_active_voters_at_gate` which applies directional gating,
portfolio/signal_engine.py:1558:    #     `extra_info["_voters"]` against dynamic_min. `_voters` is raw
portfolio/signal_engine.py:1563:    #     Downstream's raw `_voters` doesn't account for top-N or
portfolio/signal_engine.py:1565:    #     below MIN_VOTERS_BASE (3) — the floor across all asset classes —
portfolio/signal_engine.py:1575:    min_regime_quorum = _dynamic_min_voters_for_regime(regime)
portfolio/signal_engine.py:1580:    # P2-F (2026-04-17): derived from MIN_VOTERS_CRYPTO/STOCK rather than
portfolio/signal_engine.py:1589:    baseline = _count_active_voters_at_gate(
portfolio/signal_engine.py:1595:    best_possible = _count_active_voters_at_gate(
portfolio/signal_engine.py:1602:    # P2-A (2026-04-17): raised from 2 to MIN_VOTERS_BASE (3). A 2-voter
portfolio/signal_engine.py:1607:    # MIN_VOTERS_* to avoid creating "relaxed" sub-quorum consensuses.
portfolio/signal_engine.py:1621:        active = _count_active_voters_at_gate(
portfolio/signal_engine.py:1738:def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
portfolio/signal_engine.py:2237:def _confluence_score(votes, indicators):
portfolio/signal_engine.py:2423:      4. Dynamic MIN_VOTERS — raises the bar in uncertain markets
portfolio/signal_engine.py:2504:    # --- Stage 4: Dynamic MIN_VOTERS ---
portfolio/signal_engine.py:2509:    active_voters = extra_info.get("_voters_post_filter",
portfolio/signal_engine.py:2510:                                    extra_info.get("_voters", 0))
portfolio/signal_engine.py:2511:    dynamic_min = _dynamic_min_voters_for_regime(regime)
portfolio/signal_engine.py:2513:    if action != "HOLD" and active_voters < dynamic_min:
portfolio/signal_engine.py:2515:            "stage": "dynamic_min_voters", "regime": regime,
portfolio/signal_engine.py:2516:            "required": dynamic_min, "actual": active_voters, "effect": "force_hold",
portfolio/signal_engine.py:2527:        total_voters = buy_count + sell_count
portfolio/signal_engine.py:2528:        if total_voters > 0:
portfolio/signal_engine.py:2529:            agreement_ratio = max(buy_count, sell_count) / total_voters
portfolio/signal_engine.py:3248:    # all downstream code (core gate, min_voters, unanimity penalty) sees
portfolio/signal_engine.py:3249:    # post-gated counts.  _weighted_consensus also applies this internally
portfolio/signal_engine.py:3310:    # HERE, before buy/sell counting, so `active_voters` reflects the post-
portfolio/signal_engine.py:3312:    # _weighted_consensus, leaving `extra_info["_voters"]` stale - Stage 4's
portfolio/signal_engine.py:3314:    # actually drove the consensus. _weighted_consensus still applies the
portfolio/signal_engine.py:3353:    active_voters = buy + sell
portfolio/signal_engine.py:3355:        min_voters = MIN_VOTERS_STOCK
portfolio/signal_engine.py:3357:        min_voters = MIN_VOTERS_STOCK  # metals use same threshold
portfolio/signal_engine.py:3359:        min_voters = MIN_VOTERS_CRYPTO
portfolio/signal_engine.py:3362:    if core_active == 0 or active_voters < min_voters:
portfolio/signal_engine.py:3366:        buy_conf = buy / active_voters
portfolio/signal_engine.py:3367:        sell_conf = sell / active_voters
portfolio/signal_engine.py:3387:            blend_accuracy_data,
portfolio/signal_engine.py:3416:        accuracy_data = blend_accuracy_data(
portfolio/signal_engine.py:3475:                # Without these, _weighted_consensus directional gate falls back
portfolio/signal_engine.py:3547:    consensus_votes = _apply_persistence_filter(votes, ticker)
portfolio/signal_engine.py:3560:    # `_weighted_consensus` runs its own pre-pass as defense-in-depth.
portfolio/signal_engine.py:3565:    post_persistence_voters = sum(
portfolio/signal_engine.py:3569:    weighted_action, weighted_conf = _weighted_consensus(
portfolio/signal_engine.py:3582:    # BUG-227: Apply core gate AND MIN_VOTERS gate to weighted consensus.
portfolio/signal_engine.py:3583:    # Use post_persistence_voters (not pre-filter active_voters) because the
portfolio/signal_engine.py:3585:    if core_active == 0 or post_persistence_voters < min_voters:
portfolio/signal_engine.py:3590:    confluence = _confluence_score(votes, extra_info)
portfolio/signal_engine.py:3599:    extra_info["_voters"] = active_voters  # pre-filter (compatibility)
portfolio/signal_engine.py:3600:    extra_info["_voters_post_filter"] = post_persistence_voters
portfolio/signal_engine.py:3614:    extra_info["_confluence_score"] = confluence
portfolio/signal_engine.py:3620:    # Apply confidence penalty cascade (regime, volume/ADX, trap, dynamic min_voters)

 succeeded in 1230ms:

[32;1mName[0m
[32;1m----[0m
accuracy_degradation.py
accuracy_stats.py
forecast_accuracy.py
outcome_tracker.py
signal_db.py
signal_engine.py
signal_history.py
signal_postmortem.py
signal_registry.py
signal_utils.py
signal_weight_optimizer.py
signal_weights.py
ticker_accuracy.py


codex
I have the anchor points. Next I’m reading the surrounding implementations, especially where the comments mention prior fixes, because several of these claims look like they may already have been addressed in this branch.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=2208; $end=2260; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=2488; $end=2520; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=3300; $end=3368; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 669ms:
 2488:             price_down = recent_close.iloc[-1] < recent_close.iloc[0]
 2489: 
 2490:             if recent_vol is not None and len(recent_vol) >= 5:
 2491:                 vol_declining = recent_vol.iloc[-1] < recent_vol.iloc[0] * 0.8
 2492: 
 2493:                 if action == "BUY" and price_up and vol_declining:
 2494:                     conf *= 0.5
 2495:                     penalty_log.append({"stage": "trap", "type": "bull_trap", "mult": 0.5})
 2496:                 elif action == "SELL" and price_down and vol_declining:
 2497:                     conf *= 0.5
 2498:                     penalty_log.append({"stage": "trap", "type": "bear_trap", "mult": 0.5})
 2499:         except Exception:
 2500:             logger.warning("Trap detection failed for %s", ticker, exc_info=True)
 2501:     # BUG-90: Clamp after Stage 3
 2502:     conf = min(1.0, conf)
 2503: 
 2504:     # --- Stage 4: Dynamic MIN_VOTERS ---
 2505:     # P2-C (2026-04-17): delegate to shared helper to avoid drift with the
 2506:     # circuit breaker's recovery-floor logic. Same semantic as before.
 2507:     # BUG-227: Use post-persistence voter count (not pre-filter) so the gate
 2508:     # reflects the actual participating voters after debounce filtering.
 2509:     active_voters = extra_info.get("_voters_post_filter",
 2510:                                     extra_info.get("_voters", 0))
 2511:     dynamic_min = _dynamic_min_voters_for_regime(regime)
 2512: 
 2513:     if action != "HOLD" and active_voters < dynamic_min:
 2514:         penalty_log.append({
 2515:             "stage": "dynamic_min_voters", "regime": regime,
 2516:             "required": dynamic_min, "actual": active_voters, "effect": "force_hold",
 2517:         })
 2518:         action = "HOLD"
 2519:         conf = 0.0
 2520: 

 succeeded in 648ms:
 3300:             logger.debug(
 3301:                 "RES-2026-04-21: %s exempt from %s regime gating — recent 7d "
 3302:                 "accuracy %.1f%% (%d sam) overrides stale gate",
 3303:                 sig_name, regime, r_acc * 100, r_samples,
 3304:             )
 3305:     for sig_name in regime_gated_effective:
 3306:         if sig_name in votes and votes[sig_name] != "HOLD":
 3307:             votes[sig_name] = "HOLD"
 3308: 
 3309:     # P1-B (2026-04-17 adversarial review): apply horizon-specific blacklist
 3310:     # HERE, before buy/sell counting, so `active_voters` reflects the post-
 3311:     # horizon-disable state. Previously this gating only happened inside
 3312:     # _weighted_consensus, leaving `extra_info["_voters"]` stale - Stage 4's
 3313:     # dynamic_min check could pass a 5-voter count even though only 2 voters
 3314:     # actually drove the consensus. _weighted_consensus still applies the
 3315:     # same gating internally (idempotent: HOLD->HOLD is a no-op) as defense
 3316:     # in depth for callers that bypass generate_signal.
 3317:     horizon_disabled_effective = _get_horizon_disabled_signals(ticker, horizon)
 3318:     for sig_name in horizon_disabled_effective:
 3319:         if sig_name in votes and votes[sig_name] != "HOLD":
 3320:             votes[sig_name] = "HOLD"
 3321: 
 3322:     # Codex round 2 P1 (2026-04-28): macro-window force-HOLD must mutate
 3323:     # `votes` BEFORE buy/sell/core_active are computed below — otherwise
 3324:     # those counts come from the pre-mutation state and the gate at
 3325:     # line ~3333 ("core_active == 0 ...") sees a stale 1 even when the
 3326:     # only core voter (e.g. claude_fundamental) was suppressed by macro.
 3327:     # Mirrors the existing regime_gate / horizon_disabled mutation
 3328:     # pattern above.
 3329:     macro_active_effective = _is_macro_window_cached()
 3330:     if macro_active_effective and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
 3331:         for sig_name in MACRO_WINDOW_FORCE_HOLD_SIGNALS:
 3332:             if sig_name in votes and votes[sig_name] != "HOLD":
 3333:                 votes[sig_name] = "HOLD"
 3334: 
 3335:     if ticker:
 3336:         _record_phase(ticker, "regime_gate", _phase_start)
 3337:         _phase_start = time.monotonic()
 3338: 
 3339:     # Derive buy/sell counts from named votes (post-gating)
 3340:     buy = sum(1 for v in votes.values() if v == "BUY")
 3341:     sell = sum(1 for v in votes.values() if v == "SELL")
 3342: 
 3343:     # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
 3344:     # Enhanced signals can strengthen/weaken a consensus but never create one alone.
 3345:     core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
 3346:     core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
 3347:     core_active = core_buy + core_sell
 3348: 
 3349:     # Total applicable signals: computed dynamically from SIGNAL_NAMES
 3350:     # minus DISABLED_SIGNALS minus per-asset-class exclusions.
 3351:     total_applicable = _compute_applicable_count(ticker, skip_gpu=skip_gpu)
 3352: 
 3353:     active_voters = buy + sell
 3354:     if ticker in STOCK_SYMBOLS:
 3355:         min_voters = MIN_VOTERS_STOCK
 3356:     elif ticker in METALS_SYMBOLS:
 3357:         min_voters = MIN_VOTERS_STOCK  # metals use same threshold
 3358:     else:
 3359:         min_voters = MIN_VOTERS_CRYPTO
 3360: 
 3361:     # Core gate: if no core signal is active, force HOLD regardless of enhanced votes
 3362:     if core_active == 0 or active_voters < min_voters:
 3363:         action = "HOLD"
 3364:         conf = 0.0
 3365:     else:
 3366:         buy_conf = buy / active_voters
 3367:         sell_conf = sell / active_voters
 3368:         if buy_conf > sell_conf and buy_conf >= 0.5:

 succeeded in 681ms:
 2208:         # SELL (if it ever emits one) keeps 1.0x because that's genuinely
 2209:         # informative. Previous version penalized ALL votes equally.
 2210:         signal_bias = act_data.get("bias", 0.0)
 2211:         signal_samples = act_data.get("samples", 0)
 2212:         if signal_samples >= _BIAS_MIN_ACTIVE and signal_bias > _BIAS_THRESHOLD:
 2213:             buy_rate = act_data.get("buy_rate", 0.0)
 2214:             sell_rate = act_data.get("sell_rate", 0.0)
 2215:             bias_direction = "BUY" if buy_rate >= sell_rate else "SELL"
 2216:             if vote == bias_direction:
 2217:                 penalty = _BIAS_EXTREME_PENALTY if signal_bias > _BIAS_EXTREME_THRESHOLD else _BIAS_PENALTY
 2218:                 weight *= penalty
 2219:         if vote == "BUY":
 2220:             buy_weight += weight
 2221:         elif vote == "SELL":
 2222:             sell_weight += weight
 2223:     if gated_signals:
 2224:         logger.debug("Accuracy-gated signals (<%s%%): %s", ACCURACY_GATE_THRESHOLD * 100, gated_signals)
 2225:     total_weight = buy_weight + sell_weight
 2226:     if total_weight == 0:
 2227:         return "HOLD", 0.0
 2228:     buy_conf = buy_weight / total_weight
 2229:     sell_conf = sell_weight / total_weight
 2230:     if buy_conf > sell_conf and buy_conf >= 0.5:
 2231:         return "BUY", round(buy_conf, 4)
 2232:     if sell_conf > buy_conf and sell_conf >= 0.5:
 2233:         return "SELL", round(sell_conf, 4)
 2234:     return "HOLD", round(max(buy_conf, sell_conf), 4)
 2235: 
 2236: 
 2237: def _confluence_score(votes, indicators):
 2238:     active = {k: v for k, v in votes.items() if v != "HOLD"}
 2239:     if not active:
 2240:         return 0.0
 2241:     buy_count = sum(1 for v in active.values() if v == "BUY")
 2242:     sell_count = sum(1 for v in active.values() if v == "SELL")
 2243:     majority = max(buy_count, sell_count)
 2244:     score = majority / len(active)
 2245:     if indicators.get("volume_action") in ("BUY", "SELL"):
 2246:         vol_dir = indicators.get("volume_action")
 2247:         majority_dir = "BUY" if buy_count >= sell_count else "SELL"
 2248:         if vol_dir == majority_dir:
 2249:             score += 0.1
 2250:     return min(round(score, 4), 1.0)
 2251: 
 2252: 
 2253: def _time_of_day_factor(horizon=None):
 2254:     hour = datetime.now(UTC).hour
 2255:     if horizon in ("3h", "4h"):
 2256:         from portfolio.short_horizon import time_of_day_scale_3h
 2257:         return time_of_day_scale_3h(hour)
 2258:     # Default 1d behavior
 2259:     if 2 <= hour <= 6:
 2260:         return 0.8

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=250; $end=320; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=1410; $end=1665; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/accuracy_stats.py'; "'$start=880; $end=960; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 801ms:
  880:         dict: Blended {signal_name: {accuracy, total, correct, pct}}.
  881:     """
  882:     if not alltime and not recent:
  883:         return {}
  884:     # Codex round-10 P2 (2026-04-17 follow-up): don't early-return recent
  885:     # without going through the blend loop - the min_recent_samples floor
  886:     # must apply to recent-only signals too (a 20-sample signal with recent
  887:     # accuracy=0.80 should default to neutral 0.5, not vote at 0.80).
  888:     # Treating empty alltime/recent as {} keeps the loop's per-signal
  889:     # sample-threshold logic authoritative.
  890:     if not alltime:
  891:         alltime = {}
  892:     if not recent:
  893:         recent = {}
  894: 
  895:     # P1-D (2026-04-17 adversarial review): iterate over the UNION of signal
  896:     # names, not just alltime. Previously a signal present only in `recent`
  897:     # was silently dropped, and directional keys (buy_accuracy, sell_accuracy,
  898:     # total_buy, total_sell) were copied only from `at` - so a signal with
  899:     # sell_accuracy=0.28 over 400 samples in recent but no alltime entry
  900:     # silently passed the directional gate because total_sell defaulted to 0.
  901:     accuracy_data = {}
  902:     all_signal_names = set(alltime) | set(recent)
  903:     for sig_name in all_signal_names:
  904:         at = alltime.get(sig_name) or {}
  905:         rc = recent.get(sig_name) or {}
  906:         at_acc = at.get("accuracy", 0.5) if at else 0.5
  907:         rc_acc = rc.get("accuracy", 0.5) if rc else 0.5
  908:         rc_samples = rc.get("total", 0) if rc else 0
  909:         at_samples = at.get("total", 0) if at else 0
  910: 
  911:         # Blend only when recent has enough samples AND alltime exists;
  912:         # otherwise fall back to whichever source has data.
  913:         # Codex round-10 P2 (2026-04-17 follow-up): previously a recent-only
  914:         # signal with <min_recent_samples samples fell through to rc_acc,
  915:         # letting an immature signal's raw recent accuracy drive consensus.
  916:         # Now we require min_recent_samples even for recent-only signals,
  917:         # falling back to a neutral 0.5 otherwise (matches pre-patch
  918:         # semantics for signals below the recent-sample floor).
  919:         if rc_samples >= min_recent_samples and at_samples > 0:
  920:             divergence = abs(rc_acc - at_acc)
  921:             w = fast_weight if divergence > divergence_threshold else normal_weight
  922:             blended = w * rc_acc + (1 - w) * at_acc
  923:         elif at_samples > 0:
  924:             blended = at_acc
  925:         elif rc_samples >= min_recent_samples:
  926:             blended = rc_acc  # recent-only signal with enough samples
  927:         else:
  928:             blended = 0.5  # immature signal: neutral default
  929: 
  930:         total = max(at_samples, rc_samples)
  931:         result = {
  932:             "accuracy": blended,
  933:             "total": total,
  934:             "samples": total,
  935:             "enabled": sig_name not in DISABLED_SIGNALS,
  936:             "correct": int(round(blended * total)),  # BUG-186
  937:             "pct": round(blended * 100, 1),
  938:         }
  939:         # Codex round 11 P2 (2026-04-17 follow-up): directional stats must
  940:         # follow the same sample-floor rule as `accuracy`. Without this,
  941:         # a recent-only signal with 20-29 one-sided votes still influenced
  942:         # _weighted_consensus's directional gate/weighting at its raw
  943:         # recent directional accuracy, even though the overall `accuracy`
  944:         # field had already been set back to neutral 0.5. Omit directional
  945:         # keys entirely for immature signals so downstream callers see the
  946:         # `.get('buy_accuracy', acc)` fallback.
  947:         _directionals_trustworthy = (
  948:             at_samples > 0 or rc_samples >= min_recent_samples
  949:         )
  950:         if _directionals_trustworthy:
  951:             # Merge directional keys from the larger-sample source per key.
  952:             # Prevents silent gate-bypass when a key exists only in `recent`.
  953:             for key in ("buy_accuracy", "sell_accuracy"):
  954:                 if key in at and key in rc:
  955:                     side_total = "total_buy" if key == "buy_accuracy" else "total_sell"
  956:                     at_side = at.get(side_total, 0) or 0
  957:                     rc_side = rc.get(side_total, 0) or 0
  958:                     result[key] = at[key] if at_side >= rc_side else rc[key]
  959:                 elif key in at:
  960:                     result[key] = at[key]

 succeeded in 822ms:
  250: _PERSISTENCE_ENABLED = True        # toggle for easy disable
  251: _PERSISTENCE_MAX_TICKERS = 32      # bound on tracked tickers (prod=5, cap guards tests/probes)
  252: _persistence_state: dict[str, dict[str, dict]] = {}  # {ticker: {signal: {"vote": str, "cycles": int}}}
  253: _persistence_lock = threading.Lock()
  254: 
  255: # Cross-ticker consensus cache: stores the most recent consensus action per
  256: # ticker so synthetic cross-asset signals can reference other tickers' results.
  257: # Stale reads (MSTR processing before BTC in the same cycle) are acceptable —
  258: # the 60s loop ensures data is at most one cycle old.
  259: _cross_ticker_consensus: dict[str, dict] = {}  # {ticker: {"action": str, "confidence": float}}
  260: _cross_ticker_lock = threading.Lock()
  261: 
  262: 
  263: def _apply_persistence_filter(votes: dict[str, str], ticker: str | None) -> dict[str, str]:
  264:     """Filter votes to only include signals that persisted for MIN_PERSISTENCE_CYCLES.
  265: 
  266:     Returns a new dict with non-persistent signals forced to HOLD.
  267:     The original votes dict is not modified (needed for accuracy tracking).
  268: 
  269:     Cold-start: on the first cycle for a ticker (no prior state), all signals
  270:     pass through unfiltered. Filtering only activates once we have history.
  271:     """
  272:     if not _PERSISTENCE_ENABLED or not ticker:
  273:         return votes
  274: 
  275:     with _persistence_lock:
  276:         # Cold start: if we have NO history for this ticker, seed state and
  277:         # pass all votes through. The filter only applies from cycle 2 onward.
  278:         if ticker not in _persistence_state:
  279:             if len(_persistence_state) >= _PERSISTENCE_MAX_TICKERS:
  280:                 evict_count = len(_persistence_state) // 2
  281:                 for old_key in list(_persistence_state)[:evict_count]:
  282:                     del _persistence_state[old_key]
  283:             _persistence_state[ticker] = {
  284:                 sig: {"vote": vote, "cycles": _PERSISTENCE_MIN_CYCLES if vote != "HOLD" else 0}
  285:                 for sig, vote in votes.items()
  286:             }
  287:             return votes  # first cycle — trust all signals
  288: 
  289:         ticker_state = _persistence_state[ticker]
  290:         filtered = {}
  291:         for sig, vote in votes.items():
  292:             prev = ticker_state.get(sig)
  293: 
  294:             if vote == "HOLD":
  295:                 # Signal went quiet — reset persistence
  296:                 ticker_state[sig] = {"vote": "HOLD", "cycles": 0}
  297:                 filtered[sig] = "HOLD"
  298:             elif prev is None or prev["vote"] != vote:
  299:                 # New direction or first appearance — start counting
  300:                 ticker_state[sig] = {"vote": vote, "cycles": 1}
  301:                 # Not yet persistent — force HOLD for consensus
  302:                 filtered[sig] = "HOLD"
  303:             else:
  304:                 # Same direction as previous cycle — increment
  305:                 prev["cycles"] += 1
  306:                 if prev["cycles"] >= _PERSISTENCE_MIN_CYCLES:
  307:                     filtered[sig] = vote  # persistent — let it vote
  308:                 else:
  309:                     filtered[sig] = "HOLD"  # still provisional
  310: 
  311:         return filtered
  312: 
  313: 
  314: # Disabled signal per-ticker rescue (2026-04-18): signals in DISABLED_SIGNALS
  315: # that have proven accuracy on specific tickers. These are re-enabled for
  316: # compute+consensus on the listed ticker only. The standard accuracy gate
  317: # (47%) still protects against degradation.
  318: # Format: {(signal_name, ticker)} — if (sig, ticker) is in this set, the
  319: # signal is computed and votes for that ticker despite being globally disabled.
  320: # Evidence: data/disabled_signal_rescue_2026-04-18.json

 succeeded in 847ms:
 1410:     import math
 1411:     if value is None:
 1412:         return 0
 1413:     try:
 1414:         f = float(value)
 1415:     except (TypeError, ValueError):
 1416:         return 0
 1417:     if math.isnan(f) or math.isinf(f) or f < 0:
 1418:         return 0
 1419:     return int(f)
 1420: 
 1421: 
 1422: def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
 1423:                                   base_gate, relaxation):
 1424:     """Count how many signals would pass gating at gate=(base_gate - relaxation).
 1425: 
 1426:     Counts only voters that survive the full gate cascade:
 1427:       1) excluded (top-N)
 1428:       2) group-gated (correlation leader below group-leader gate)
 1429:       3) accuracy gate at (base - relaxation), tiered for high-sample signals
 1430:       4) directional gate (unchanged by relaxation)
 1431: 
 1432:     Returns int — the number of signals still voting BUY/SELL.
 1433:     """
 1434:     gate_val = base_gate - relaxation
 1435:     # SC-P1-2 (2026-05-02 adversarial follow-ups): high-sample tier is NOT
 1436:     # relaxed. A signal with 10K+ samples at sub-50% accuracy has measurable
 1437:     # negative edge — circuit-breaker relaxation must not promote it back to
 1438:     # voting. Standard tier (under 10K samples) still relaxes so borderline
 1439:     # newer signals can be rescued during regime transitions. Must mirror
 1440:     # the same logic in `_weighted_consensus` (line ~2068).
 1441:     high_gate_val = _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD
 1442:     active = 0
 1443:     for signal_name, vote in votes.items():
 1444:         if vote == "HOLD":
 1445:             continue
 1446:         if signal_name in excluded:
 1447:             continue
 1448:         if signal_name in group_gated:
 1449:             continue
 1450:         stats = accuracy_data.get(signal_name) or {}
 1451:         # P1-C (2026-04-17 adversarial review): coerce None / NaN values to
 1452:         # safe defaults. The live path previously crashed with TypeError when
 1453:         # `accuracy_data[sig]` contained explicit None values (e.g., from a
 1454:         # half-written cache). Replay had `except TypeError` but live didn't.
 1455:         acc = _safe_accuracy(stats.get("accuracy"), default=0.5)
 1456:         samples = _safe_sample_count(stats.get("total"))
 1457:         effective_gate = gate_val
 1458:         if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
 1459:             effective_gate = max(gate_val, high_gate_val)
 1460:         if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
 1461:             continue
 1462:         # Directional gate is not relaxed by the circuit breaker — those gates
 1463:         # catch signals that are actively wrong in one direction.
 1464:         if vote == "BUY":
 1465:             dir_acc = _safe_accuracy(stats.get("buy_accuracy"), default=acc)
 1466:             dir_n = _safe_sample_count(stats.get("total_buy"))
 1467:         else:
 1468:             dir_acc = _safe_accuracy(stats.get("sell_accuracy"), default=acc)
 1469:             dir_n = _safe_sample_count(stats.get("total_sell"))
 1470:         if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
 1471:             continue
 1472:         active += 1
 1473:     return active
 1474: 
 1475: 
 1476: def _normalize_regime(regime):
 1477:     """P2-D (2026-04-17): normalize regime strings to a canonical lowercase form.
 1478: 
 1479:     Protects against case/typo variants ("TRENDING-UP", " trending-up ",
 1480:     "trending_up") that would otherwise silently fall through to the
 1481:     strictest-quorum default. Returns None unchanged.
 1482:     """
 1483:     if regime is None:
 1484:         return None
 1485:     if not isinstance(regime, str):
 1486:         return regime  # Let downstream default handle non-strings.
 1487:     normalized = regime.strip().lower().replace("_", "-")
 1488:     # Common alias fixups.
 1489:     if normalized in ("trendingup", "trending"):
 1490:         normalized = "trending-up"
 1491:     elif normalized == "trendingdown":
 1492:         normalized = "trending-down"
 1493:     elif normalized in ("highvol", "high-volatility", "high_vol"):
 1494:         normalized = "high-vol"
 1495:     return normalized
 1496: 
 1497: 
 1498: def _dynamic_min_voters_for_regime(regime):
 1499:     """Regime-dependent final quorum. Single source of truth - called by both
 1500:     the circuit breaker and apply_confidence_penalties.
 1501: 
 1502:     This is the minimum voter count the OUTER consensus path requires before
 1503:     emitting a non-HOLD action. The circuit breaker uses it to size its
 1504:     recovery floor so relaxation is only engaged when it could reach the
 1505:     regime's actual quorum.
 1506: 
 1507:     2026-04-17 (P2-C/P2-D): de-duplicated. apply_confidence_penalties
 1508:     previously had an inline copy at line ~1623 that had to stay in lockstep
 1509:     manually - now it calls this helper. Also accepts case/typo-variant
 1510:     regime strings via _normalize_regime.
 1511:     """
 1512:     canonical = _normalize_regime(regime)
 1513:     if canonical in ("trending-up", "trending-down"):
 1514:         return 3
 1515:     if canonical == "high-vol":
 1516:         return 4
 1517:     return 5  # ranging, unknown, None
 1518: 
 1519: 
 1520: def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
 1521:                               regime=None):
 1522:     """Compute circuit-breaker relaxation to preserve voter diversity.
 1523: 
 1524:     Progressively tests relaxation values 0, step, 2*step, ..., up to
 1525:     _GATE_RELAXATION_MAX. Returns the smallest relaxation that yields at
 1526:     least _MIN_ACTIVE_VOTERS_SOFT active voters.
 1527: 
 1528:     Decision tree:
 1529:       - baseline >= floor                    -> 0.0 (no relaxation needed)
 1530:       - best_possible <= baseline            -> 0.0 (relaxation doesn't help;
 1531:                                                 either a low-signal scenario
 1532:                                                 or a genuine regime break
 1533:                                                 where remaining signals are
 1534:                                                 below even the 41% relaxed
 1535:                                                 gate — letting them vote
 1536:                                                 would be wrong)
 1537:       - best_possible >= floor               -> smallest step that meets floor
 1538:       - baseline < best_possible < floor     -> _GATE_RELAXATION_MAX (partial
 1539:                                                 recovery: a single
 1540:                                                 irrecoverable outlier must
 1541:                                                 not veto relaxation for the
 1542:                                                 rest - Codex P2 fix)
 1543: 
 1544:     Uses `_count_active_voters_at_gate` which applies directional gating,
 1545:     so signals gated on BUY-accuracy=30% don't inflate the decision.
 1546: 
 1547:     Returns float - relaxation in absolute accuracy points (e.g., 0.02).
 1548:     """
 1549:     # Defensive: caller may pass None for either set (older paths or a future
 1550:     # refactor). Treat as empty to avoid `in None` TypeErrors in a hot path.
 1551:     excluded = excluded or set()
 1552:     group_gated = group_gated or set()
 1553: 
 1554:     # Three guards, in increasing strictness, all applied:
 1555:     #
 1556:     #   Guard A (raw vs regime quorum):
 1557:     #     Matches downstream's `apply_confidence_penalties` which checks
 1558:     #     `extra_info["_voters"]` against dynamic_min. `_voters` is raw
 1559:     #     non-HOLD count post-regime, pre top-N/group-gate, so this check
 1560:     #     must NOT subtract `excluded` or `group_gated`.
 1561:     #
 1562:     #   Guard B (post-exclusion slate viability):
 1563:     #     Downstream's raw `_voters` doesn't account for top-N or
 1564:     #     correlation-group exclusions. If the POST-exclusion slate is
 1565:     #     below MIN_VOTERS_BASE (3) — the floor across all asset classes —
 1566:     #     a relaxed consensus would be built from a too-thin slate even
 1567:     #     though downstream would accept the raw count. Codex round 9
 1568:     #     (2026-04-17) caught this with a 3-signal correlation cluster
 1569:     #     gated out, leaving only 2 voters to drive consensus.
 1570:     #
 1571:     #   Guard C (lone-signal escape):
 1572:     #     Even with a large post-exclusion slate, directional gating can
 1573:     #     leave a single accuracy-passing signal. `best_possible >= 2`
 1574:     #     catches this case.
 1575:     min_regime_quorum = _dynamic_min_voters_for_regime(regime)
 1576:     raw_candidates = sum(1 for v in votes.values() if v != "HOLD")
 1577:     if raw_candidates < min_regime_quorum:
 1578:         return 0.0
 1579: 
 1580:     # P2-F (2026-04-17): derived from MIN_VOTERS_CRYPTO/STOCK rather than
 1581:     # hardcoded. If the base quorum changes, this follows automatically.
 1582:     post_exclusion_candidates = sum(
 1583:         1 for sn, v in votes.items()
 1584:         if v != "HOLD" and sn not in excluded and sn not in group_gated
 1585:     )
 1586:     if post_exclusion_candidates < _POST_EXCLUSION_MIN:
 1587:         return 0.0
 1588: 
 1589:     baseline = _count_active_voters_at_gate(
 1590:         votes, accuracy_data, excluded, group_gated, base_gate, 0.0,
 1591:     )
 1592:     if baseline >= _MIN_ACTIVE_VOTERS_SOFT:
 1593:         return 0.0
 1594: 
 1595:     best_possible = _count_active_voters_at_gate(
 1596:         votes, accuracy_data, excluded, group_gated,
 1597:         base_gate, _GATE_RELAXATION_MAX,
 1598:     )
 1599: 
 1600:     # Lone-signal escape guard. Even when raw candidates meet the downstream
 1601:     # quorum, directional gating can leave a thin set of recoverable voters.
 1602:     # P2-A (2026-04-17): raised from 2 to MIN_VOTERS_BASE (3). A 2-voter
 1603:     # "consensus" is still exposure-worthy in trending markets where
 1604:     # dynamic_min=3 — but any relaxation that only recovers 2 voters from a
 1605:     # large slate is catching signals that the downstream quorum would
 1606:     # accept as a weak consensus. Require at least as many as the base
 1607:     # MIN_VOTERS_* to avoid creating "relaxed" sub-quorum consensuses.
 1608:     if best_possible < _LONE_SIGNAL_FLOOR:
 1609:         return 0.0
 1610: 
 1611:     # Regime break: relaxation recovers nothing beyond baseline. Keep the
 1612:     # strict gate so the event shows up in logs rather than silently opening
 1613:     # to sub-41% signals.
 1614:     if best_possible <= baseline:
 1615:         return 0.0
 1616: 
 1617:     # Integer steps up to and including max - use int steps to avoid float drift.
 1618:     n_steps = int(round(_GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP))
 1619:     for i in range(1, n_steps + 1):
 1620:         candidate_rel = round(i * _GATE_RELAXATION_STEP, 6)
 1621:         active = _count_active_voters_at_gate(
 1622:             votes, accuracy_data, excluded, group_gated, base_gate, candidate_rel,
 1623:         )
 1624:         if active >= _MIN_ACTIVE_VOTERS_SOFT:
 1625:             return candidate_rel
 1626:     # Partial-recovery case (Codex P2 fix): best_possible > baseline but
 1627:     # still < floor. A single irrecoverable outlier shouldn't veto recovery
 1628:     # of the recoverable majority - apply max relaxation to get as many
 1629:     # voters back as possible. Logs still carry the relaxation value so
 1630:     # operators can distinguish this from a clean relaxation-to-floor.
 1631:     return _GATE_RELAXATION_MAX
 1632: 
 1633: 
 1634: # ---------------------------------------------------------------------------
 1635: # IC-based weight multiplier (2026-04-18)
 1636: # ---------------------------------------------------------------------------
 1637: 
 1638: def _compute_ic_mult(ic: float, icir: float, samples: int) -> float:
 1639:     """Compute IC-based weight multiplier for a signal.
 1640: 
 1641:     Returns a multiplicative adjustment based on the signal's Information
 1642:     Coefficient:
 1643:     - IC > 0 with stable ICIR → boost (catches big moves)
 1644:     - IC ≈ 0 with many samples → slight penalty (phantom performer)
 1645:     - IC < 0 with stable ICIR → penalty (contrarian, accuracy gate handles)
 1646:     - Insufficient data or unstable → 1.0 (no adjustment)
 1647: 
 1648:     Clamped to [_IC_MULT_FLOOR, _IC_MULT_CAP].
 1649:     """
 1650:     if samples < _IC_MIN_SAMPLES:
 1651:         return 1.0
 1652:     # Zero-IC penalty for phantom performers: signals with many samples but
 1653:     # no return-magnitude predictive power (e.g., calendar, econ_calendar).
 1654:     if abs(ic) < 0.01 and samples >= _IC_ZERO_MIN_SAMPLES:
 1655:         return _IC_ZERO_PENALTY
 1656:     if abs(icir) < _IC_STABILITY_MIN:
 1657:         return 1.0
 1658:     raw = 1.0 + _IC_ALPHA * ic
 1659:     return max(_IC_MULT_FLOOR, min(_IC_MULT_CAP, raw))
 1660: 
 1661: 
 1662: # IC data cache: reuse ic_computation.py infrastructure with in-memory TTL.
 1663: _ic_data_cache: dict = {}
 1664: _ic_data_lock = threading.Lock()
 1665: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/outcome_tracker.py'; "'$start=430; $end=500; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=3528; $end=3610; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 600ms:
  430:             if ticker not in outcomes:
  431:                 outcomes[ticker] = {h: None for h in HORIZONS}
  432:             for h_key in HORIZONS:
  433:                 if outcomes[ticker].get(h_key) is None:
  434:                     all_filled = False
  435: 
  436:         if all_filled and all(
  437:             all(outcomes[t].get(h) is not None for h in HORIZONS) for t in tickers
  438:         ):
  439:             continue
  440: 
  441:         entry_updated = False
  442:         for ticker in tickers:
  443:             if ticker not in known_tickers:
  444:                 continue  # skip removed/unknown tickers (e.g. AI)
  445:             if ticker not in outcomes:
  446:                 outcomes[ticker] = {h: None for h in HORIZONS}
  447: 
  448:             base_price = tickers[ticker].get("price_usd")
  449:             if not base_price or base_price <= 0:
  450:                 continue  # BUG-220: skip — no base price to compute change_pct
  451:             for h_key, h_seconds in HORIZONS.items():
  452:                 if outcomes[ticker].get(h_key) is not None:
  453:                     continue
  454:                 target_ts = entry_ts + h_seconds
  455:                 if now_ts < target_ts:
  456:                     continue
  457: 
  458:                 cache_key = (ticker, int(target_ts // 3600))
  459:                 if cache_key not in price_cache:
  460:                     try:
  461:                         price_cache[cache_key] = _fetch_historical_price(
  462:                             ticker, target_ts
  463:                         )
  464:                     except Exception:
  465:                         price_cache[cache_key] = None
  466: 
  467:                 hist_price = price_cache[cache_key]
  468:                 if hist_price is None:
  469:                     continue
  470: 
  471:                 change_pct = round(
  472:                     ((hist_price - base_price) / base_price) * 100, 2
  473:                 )
  474: 
  475:                 outcome_ts_str = datetime.fromtimestamp(
  476:                     target_ts, tz=UTC
  477:                 ).isoformat()
  478:                 outcomes[ticker][h_key] = {
  479:                     "price_usd": round(hist_price, 2),
  480:                     "change_pct": change_pct,
  481:                     "ts": outcome_ts_str,
  482:                 }
  483:                 entry_updated = True
  484: 
  485:                 # Dual-write outcome to SQLite
  486:                 if _db is not None:
  487:                     try:
  488:                         _db.update_outcome(
  489:                             entry["ts"], ticker, h_key,
  490:                             round(hist_price, 2), change_pct, outcome_ts_str,
  491:                         )
  492:                     except Exception as e:
  493:                         logger.debug("SQLite outcome write failed: %s", e)
  494: 
  495:         entry["outcomes"] = outcomes
  496: 
  497:         # C6: MWU weight update removed — SignalWeightManager.batch_update()
  498:         # wrote to data/signal_weights.json but signal_engine.py never read it.
  499:         # The entire MWU adaptation path was dead code producing disk I/O and
  500:         # CPU burn for zero effect. See Adversarial Review C6 for details.

 succeeded in 661ms:
 3528:     sig_cfg = (config or {}).get("signals", {})
 3529:     if not _accuracy_failed and sig_cfg.get("use_best_horizon", False):
 3530:         try:
 3531:             from portfolio.accuracy_stats import signal_best_horizon_accuracy
 3532:             best_hz = signal_best_horizon_accuracy(min_samples=50)
 3533:             for sig_name, bh_data in best_hz.items():
 3534:                 if bh_data.get("total", 0) >= 30:
 3535:                     # Only override if best-horizon accuracy is meaningfully better
 3536:                     current = accuracy_data.get(sig_name, {}).get("accuracy", 0.5)
 3537:                     if bh_data["accuracy"] > current + 0.03:
 3538:                         accuracy_data[sig_name] = bh_data
 3539:         except Exception:
 3540:             logger.debug("Best-horizon accuracy unavailable", exc_info=True)
 3541:     accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
 3542:     max_signals = sig_cfg.get("max_active_signals")
 3543: 
 3544:     # Signal persistence filter: only let signals that maintained their vote
 3545:     # for 2+ consecutive cycles participate in consensus. Raw votes are kept
 3546:     # intact for accuracy tracking (signal_log records unfiltered votes).
 3547:     consensus_votes = _apply_persistence_filter(votes, ticker)
 3548:     # Track how many signals were filtered for debugging
 3549:     _filtered_count = sum(
 3550:         1 for s in votes
 3551:         if votes[s] != "HOLD" and consensus_votes.get(s) == "HOLD"
 3552:     )
 3553:     if _filtered_count > 0:
 3554:         extra_info["_persistence_filtered"] = _filtered_count
 3555: 
 3556:     # Macro-window force-HOLD has already been applied to `votes` above
 3557:     # (before buy/sell/core_active counting), so `consensus_votes` —
 3558:     # which derives from `votes` via persistence filter — already
 3559:     # carries the suppression. No additional mutation needed here.
 3560:     # `_weighted_consensus` runs its own pre-pass as defense-in-depth.
 3561: 
 3562:     # BUG-224: compute post-persistence voter count so downstream consumers
 3563:     # (accuracy tracking, Layer 2) see the actual participating voter count,
 3564:     # not the inflated pre-filter number.
 3565:     post_persistence_voters = sum(
 3566:         1 for v in consensus_votes.values() if v in ("BUY", "SELL")
 3567:     )
 3568: 
 3569:     weighted_action, weighted_conf = _weighted_consensus(
 3570:         consensus_votes, accuracy_data, regime, activation_rates,
 3571:         accuracy_gate=accuracy_gate,
 3572:         max_signals=max_signals,
 3573:         horizon=horizon,
 3574:         regime_gated_override=regime_gated_effective,
 3575:         ticker=ticker,
 3576:     )
 3577: 
 3578:     if ticker:
 3579:         _record_phase(ticker, "weighted_consensus", _phase_start)
 3580:         _phase_start = time.monotonic()
 3581: 
 3582:     # BUG-227: Apply core gate AND MIN_VOTERS gate to weighted consensus.
 3583:     # Use post_persistence_voters (not pre-filter active_voters) because the
 3584:     # persistence filter may have reduced voters below the threshold.
 3585:     if core_active == 0 or post_persistence_voters < min_voters:
 3586:         weighted_action = "HOLD"
 3587:         weighted_conf = 0.0
 3588: 
 3589:     # Confluence score
 3590:     confluence = _confluence_score(votes, extra_info)
 3591: 
 3592:     # Time-of-day confidence adjustment
 3593:     tod_factor = _time_of_day_factor(horizon=horizon)
 3594:     weighted_conf *= tod_factor
 3595: 
 3596:     # Store raw consensus in extra for debugging, then use weighted as primary
 3597:     extra_info["_raw_action"] = action
 3598:     extra_info["_raw_confidence"] = conf
 3599:     extra_info["_voters"] = active_voters  # pre-filter (compatibility)
 3600:     extra_info["_voters_post_filter"] = post_persistence_voters
 3601:     extra_info["_total_applicable"] = total_applicable
 3602:     extra_info["_buy_count"] = buy
 3603:     extra_info["_sell_count"] = sell
 3604:     extra_info["_core_buy"] = core_buy
 3605:     extra_info["_core_sell"] = core_sell
 3606:     extra_info["_core_active"] = core_active
 3607:     extra_info["_votes"] = votes
 3608:     extra_info["_raw_votes"] = raw_votes  # C10: pre-gate votes for accuracy recovery
 3609:     extra_info["_regime"] = regime
 3610:     if horizon:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_compute_gate_relaxation\\(\" portfolio/signal_engine.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 561ms:
1520:def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
2047:    relaxation = _compute_gate_relaxation(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=2028; $end=2065; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 690ms:
 2028:         if crisis_penalize_trend:
 2029:             logger.info(
 2030:                 "Crisis mode active (full): %d/%d macro signals broken, "
 2031:                 "trend avg %.1f%% < %.0f%% floor — penalizing trend, boosting MR",
 2032:                 broken_count, len(_MACRO_CRISIS_SIGNALS),
 2033:                 avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
 2034:             )
 2035:         else:
 2036:             logger.info(
 2037:                 "Crisis mode active (partial): %d/%d macro signals broken, but "
 2038:                 "trend avg %.1f%% >= %.0f%% floor — NOT penalizing trend signals",
 2039:                 broken_count, len(_MACRO_CRISIS_SIGNALS),
 2040:                 avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
 2041:             )
 2042: 
 2043:     # Voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy gating reconfig).
 2044:     # Only the overall accuracy gate is relaxable — directional and correlation
 2045:     # gates still fire. Prevents regime-transition over-gating that silenced
 2046:     # ~8 voters in W15/W16.
 2047:     relaxation = _compute_gate_relaxation(
 2048:         votes=votes,
 2049:         accuracy_data=accuracy_data,
 2050:         excluded=excluded,
 2051:         group_gated=group_gated_signals,
 2052:         base_gate=gate,
 2053:         regime=regime,
 2054:     )
 2055:     if relaxation > 0:
 2056:         logger.debug(
 2057:             "Circuit breaker: relaxing accuracy gate by %.0fpp "
 2058:             "(base=%.2f -> effective=%.2f) to preserve voter diversity",
 2059:             relaxation * 100, gate, gate - relaxation,
 2060:         )
 2061: 
 2062:     # IC-based weight multiplier (2026-04-18): load IC data once per consensus
 2063:     # call. Returns {"global": {sig: {ic, icir, samples}}, "per_ticker": {...}}
 2064:     # or None if IC computation is unavailable.
 2065:     ic_cache = _get_ic_data(horizon) if horizon else None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_weighted_consensus\\(\" -g\"*.py\"" in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-10T15:26:46.481073Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 578ms:
codex
I’m checking call sites now. One of Claude’s claims depends on whether these helpers are fed raw votes or already persistence-filtered votes; that needs the actual caller path, not just the helper body.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_weighted_consensus" portfolio' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_count_active_voters_at_gate|_compute_gate_relaxation" portfolio/signal_engine.py' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 602ms:
portfolio\signal_engine.py:1440:    # the same logic in `_weighted_consensus` (line ~2068).
portfolio\signal_engine.py:1738:def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
portfolio\signal_engine.py:3249:    # post-gated counts.  _weighted_consensus also applies this internally
portfolio\signal_engine.py:3312:    # _weighted_consensus, leaving `extra_info["_voters"]` stale - Stage 4's
portfolio\signal_engine.py:3314:    # actually drove the consensus. _weighted_consensus still applies the
portfolio\signal_engine.py:3475:                # Without these, _weighted_consensus directional gate falls back
portfolio\signal_engine.py:3560:    # `_weighted_consensus` runs its own pre-pass as defense-in-depth.
portfolio\signal_engine.py:3569:    weighted_action, weighted_conf = _weighted_consensus(
portfolio\accuracy_stats.py:942:        # _weighted_consensus's directional gate/weighting at its raw
portfolio\accuracy_stats.py:1814:    gating in signal_engine._weighted_consensus().

 succeeded in 574ms:
1422:def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
1520:def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
1544:    Uses `_count_active_voters_at_gate` which applies directional gating,
1589:    baseline = _count_active_voters_at_gate(
1595:    best_possible = _count_active_voters_at_gate(
1621:        active = _count_active_voters_at_gate(
2047:    relaxation = _compute_gate_relaxation(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=1738; $end=2236; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 678ms:
 1738: def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
 1739:                         accuracy_gate=None, max_signals=None, horizon=None,
 1740:                         regime_gated_override=None, ticker=None):
 1741:     """Compute weighted consensus using accuracy, IC, regime, and activation frequency.
 1742: 
 1743:     Weight per signal = accuracy_weight * ic_mult * regime_mult * normalized_weight
 1744:                         * horizon_mult * activity_cap
 1745:     where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
 1746:     Rare, balanced signals get more weight; noisy/biased signals get less.
 1747: 
 1748:     Signals below the accuracy gate (with sufficient samples) are force-skipped —
 1749:     they are noise, not useful contrarian indicators.
 1750: 
 1751:     Regime gating: signals in REGIME_GATED_SIGNALS for the current regime are
 1752:     forced to HOLD before vote processing — they produce negative alpha.
 1753: 
 1754:     Correlation deduplication: within defined correlation groups, only the
 1755:     highest-accuracy signal gets full weight. Others get 0.3x penalty.
 1756: 
 1757:     Horizon-specific weights: signals with divergent accuracy across horizons
 1758:     get boosted or penalized via HORIZON_SIGNAL_WEIGHTS.
 1759: 
 1760:     Activity rate cap: signals with >70% activation rate get 0.5x penalty
 1761:     to prevent a single high-activity signal from dominating consensus.
 1762: 
 1763:     Top-N gate: when max_signals is set, only the top max_signals non-HOLD
 1764:     signals (ranked by accuracy) participate in the consensus. This focuses
 1765:     the vote on the best performers and ignores marginal contributors.
 1766:     """
 1767:     gate = accuracy_gate if accuracy_gate is not None else ACCURACY_GATE_THRESHOLD
 1768:     buy_weight = 0.0
 1769:     sell_weight = 0.0
 1770:     gated_signals = []
 1771:     regime_mults = REGIME_WEIGHTS.get(regime, {})
 1772:     activation_rates = activation_rates or {}
 1773:     horizon_mults = _get_horizon_weights(horizon)
 1774: 
 1775:     # Codex round 10/11/12 (2026-04-17 follow-up): deep-sanitize accuracy_data
 1776:     # at function entry.
 1777:     #   Round 10: coerced non-dict container values to {}.
 1778:     #   Round 11: found dict values with poisoned numeric fields still
 1779:     #             crashed. Added per-field coercion.
 1780:     #   Round 12: coerce-with-0.5-default silently promoted partially-
 1781:     #             written cache rows ({"accuracy": null, "total": 200}) into
 1782:     #             mature 50% signals that cleared the min-samples gate.
 1783:     #             Now: if a numeric field is poisoned, DROP that field so
 1784:     #             downstream `.get(..., default)` falls back cleanly. A row
 1785:     #             whose overall accuracy is poisoned but total=200 becomes
 1786:     #             {"total": 200} - the gate sees no accuracy, the downstream
 1787:     #             code default to the safe fallback. The row no longer
 1788:     #             masquerades as a 50%-accurate mature signal.
 1789:     # Codex round 13 (2026-04-17): a poisoned accuracy must invalidate its
 1790:     # PAIRED sample count too. Otherwise `{"accuracy": None, "total": 200}`
 1791:     # becomes `{"total": 200}` which downstream still reads as a mature
 1792:     # 50% signal (accuracy defaults to 0.5, samples=200 clears the gate).
 1793:     # Drop-together semantics: overall acc poisoned -> drop (accuracy, total);
 1794:     # buy_accuracy poisoned -> drop (buy_accuracy, total_buy); likewise for
 1795:     # sell. Fields whose pair is clean but themselves clean pass through.
 1796:     import math as _math
 1797: 
 1798:     def _coerce_sample_count(val):
 1799:         """Return int >= 0, or None if val is missing/poisoned/invalid."""
 1800:         if val is None:
 1801:             return None
 1802:         try:
 1803:             f = float(val)
 1804:         except (TypeError, ValueError):
 1805:             return None
 1806:         if _math.isnan(f) or _math.isinf(f) or f < 0:
 1807:             return None
 1808:         return int(f)
 1809: 
 1810:     _PAIRED = (
 1811:         ("accuracy", "total"),
 1812:         ("buy_accuracy", "total_buy"),
 1813:         ("sell_accuracy", "total_sell"),
 1814:     )
 1815:     if accuracy_data:
 1816:         _sanitized = {}
 1817:         for _k, _v in accuracy_data.items():
 1818:             if not isinstance(_v, dict):
 1819:                 _sanitized[_k] = {}
 1820:                 continue
 1821:             _clean = dict(_v)  # start from a copy, then prune.
 1822:             for _acc_key, _cnt_key in _PAIRED:
 1823:                 _acc_has = _acc_key in _clean
 1824:                 _cnt_has = _cnt_key in _clean
 1825:                 if _acc_has:
 1826:                     _clean_acc = _safe_accuracy(_clean.get(_acc_key), default=None)
 1827:                 else:
 1828:                     _clean_acc = None
 1829:                 if _cnt_has:
 1830:                     _clean_cnt = _coerce_sample_count(_clean.get(_cnt_key))
 1831:                 else:
 1832:                     _clean_cnt = None
 1833:                 # Decide whether to keep each field:
 1834:                 #   Both clean      -> keep both.
 1835:                 #   Only acc clean  -> keep acc; drop cnt (if it was present-and-poisoned).
 1836:                 #   Only cnt clean  -> drop BOTH (count without trustworthy accuracy
 1837:                 #                      must not promote the row to a mature signal).
 1838:                 #   Neither clean   -> drop both.
 1839:                 if _clean_acc is not None and _clean_cnt is not None:
 1840:                     _clean[_acc_key] = _clean_acc
 1841:                     _clean[_cnt_key] = _clean_cnt
 1842:                 elif _clean_acc is not None and not _cnt_has:
 1843:                     # Accuracy present (clean), count field absent - keep acc.
 1844:                     _clean[_acc_key] = _clean_acc
 1845:                 else:
 1846:                     # Poisoned accuracy OR poisoned count: drop both so the
 1847:                     # row doesn't masquerade as a mature signal. Downstream
 1848:                     # .get() calls then use their safe defaults.
 1849:                     _clean.pop(_acc_key, None)
 1850:                     _clean.pop(_cnt_key, None)
 1851:             _sanitized[_k] = _clean
 1852:         accuracy_data = _sanitized
 1853:     else:
 1854:         accuracy_data = {}
 1855: 
 1856:     # Regime gating: force-HOLD signals that produce negative alpha in this regime.
 1857:     # BUG-149: now horizon-aware — e.g., trend works at 3h in ranging (61.6%)
 1858:     # SC-I-001: when caller provides regime_gated_override (with BUG-158 per-ticker
 1859:     # exemptions already applied), use it instead of recomputing from scratch.
 1860:     regime_gated = regime_gated_override if regime_gated_override is not None else _get_regime_gated(regime, horizon)
 1861:     votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}
 1862: 
 1863:     # Horizon-specific per-ticker blacklist (2026-04-16, Batch 4). Extends the
 1864:     # compute-time _default blacklist with horizon-specific entries. Compute time
 1865:     # can't see horizon (one vote reused across 3h/4h/12h/1d/3d/5d/10d consensus),
 1866:     # so per-horizon gating must happen here.
 1867:     horizon_disabled = _get_horizon_disabled_signals(ticker, horizon)
 1868:     if horizon_disabled:
 1869:         votes = {k: ("HOLD" if k in horizon_disabled else v) for k, v in votes.items()}
 1870: 
 1871:     # Macro-window force-HOLD pre-pass (2026-04-28). When a high-impact
 1872:     # event is within ~24h past or ~72h future, force-HOLD the signals
 1873:     # whose lag/bias makes them dominantly wrong in news-driven regimes.
 1874:     # The downweight branch for the other macro-fragile signals lives in
 1875:     # the weight loop below so it composes with regime/horizon multipliers.
 1876:     macro_active = _is_macro_window_cached()
 1877:     if macro_active and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
 1878:         votes = {
 1879:             k: ("HOLD" if k in MACRO_WINDOW_FORCE_HOLD_SIGNALS else v)
 1880:             for k, v in votes.items()
 1881:         }
 1882: 
 1883:     # Top-N gate: only let the top max_signals (by accuracy) participate.
 1884:     # Codex round 2 P2 (2026-04-28): rank with macro-adjusted accuracy so
 1885:     # downweighted signals lose Top-N slots to healthier peers during a
 1886:     # macro window. Without this, sentiment can keep its slot at full
 1887:     # raw accuracy and exclude a peer that would have voted more reliably.
 1888:     def _topn_accuracy_key(s: str) -> float:
 1889:         base = accuracy_data.get(s, {}).get("accuracy", 0.5)
 1890:         if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
 1891:             base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
 1892:         return base
 1893: 
 1894:     active_votes = {k: v for k, v in votes.items() if v != "HOLD"}
 1895:     if max_signals and len(active_votes) > max_signals:
 1896:         ranked = sorted(
 1897:             active_votes.keys(),
 1898:             key=_topn_accuracy_key,
 1899:             reverse=True,
 1900:         )
 1901:         excluded = set(ranked[max_signals:])
 1902:     else:
 1903:         excluded = set()
 1904: 
 1905:     # Pre-compute which signal is the "leader" (highest accuracy) in each
 1906:     # correlation group, considering only signals that are actively voting.
 1907:     # Prefer dynamic groups (from signal_log correlations) over static.
 1908:     active_non_hold = {s for s, v in votes.items() if v != "HOLD"}
 1909:     _active_corr_groups = _get_correlation_groups() or _STATIC_CORRELATION_GROUPS
 1910: 
 1911:     # Codex P2 (2026-04-28): apply the macro-window downweight to the
 1912:     # leader-selection key BEFORE picking the leader. Otherwise sentiment
 1913:     # (lifetime ~70% acc) stays leader of macro_external during a macro
 1914:     # window — and the 0.15x follower penalty pushes healthier peers
 1915:     # below sentiment's already-halved weight, making the overlay
 1916:     # actively reinforce the wrong signal.
 1917:     def _leader_accuracy_key(s: str) -> float:
 1918:         base = accuracy_data.get(s, {}).get("accuracy", 0.5)
 1919:         if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
 1920:             base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
 1921:         return base
 1922: 
 1923:     group_leaders = {}
 1924:     for group_name, group_sigs in _active_corr_groups.items():
 1925:         active_in_group = active_non_hold & group_sigs
 1926:         if len(active_in_group) <= 1:
 1927:             continue
 1928:         best_sig = max(active_in_group, key=_leader_accuracy_key)
 1929:         group_leaders[group_name] = best_sig
 1930: 
 1931:     # Correlation group leader gating: when the best signal in a group has
 1932:     # accuracy below threshold (with enough samples), gate the ENTIRE group.
 1933:     # Prevents the "least bad" broken signal from voting.
 1934:     # 2026-03-31: macro_external group (fear_greed 25.9%, sentiment 46.8%,
 1935:     # news_event 29.5%) — even the leader is near noise.
 1936:     # 2026-04-06: Lowered from 0.47 → 0.46 to catch borderline cases where
 1937:     # sentiment (blended ~46.4%) barely escapes as group leader.
 1938:     _GROUP_LEADER_GATE_THRESHOLD = 0.46
 1939:     group_gated_signals = set()
 1940:     for group_name, group_sigs in _active_corr_groups.items():
 1941:         leader = group_leaders.get(group_name)
 1942:         if leader:
 1943:             leader_stats = accuracy_data.get(leader, {})
 1944:             leader_acc = leader_stats.get("accuracy", 0.5)
 1945:             leader_samples = leader_stats.get("total", 0)
 1946:             if leader_samples >= ACCURACY_GATE_MIN_SAMPLES and leader_acc < _GROUP_LEADER_GATE_THRESHOLD:
 1947:                 group_gated_signals.update(group_sigs & active_non_hold)
 1948:                 logger.debug(
 1949:                     "Correlation group %s gated: leader %s at %.1f%% < %.0f%% threshold",
 1950:                     group_name, leader, leader_acc * 100, _GROUP_LEADER_GATE_THRESHOLD * 100,
 1951:                 )
 1952: 
 1953:     # Build a mapping of signal → correlation penalty (per-cluster override).
 1954:     # When a signal is in multiple groups, use the harshest (lowest) penalty.
 1955:     penalized_signals: dict[str, float] = {}
 1956:     for group_name, group_sigs in _active_corr_groups.items():
 1957:         leader = group_leaders.get(group_name)
 1958:         if leader:
 1959:             penalty = _CLUSTER_CORRELATION_PENALTIES.get(group_name, _CORRELATION_PENALTY)
 1960:             for s in group_sigs:
 1961:                 if s != leader and s in active_non_hold:
 1962:                     penalized_signals[s] = min(penalized_signals.get(s, 1.0), penalty)
 1963: 
 1964:     # Meta-cluster deduplication (2026-05-01): when leaders from related
 1965:     # sub-clusters agree on direction, apply penalty to redundant leaders.
 1966:     # Prevents the trend mega-view from getting 3.0x effective leader weight
 1967:     # when pure_trend/oscillator_trend/structural_flow leaders vote identically.
 1968:     for meta_name, sub_clusters in _META_CLUSTER_GROUPS.items():
 1969:         meta_leaders: dict[str, str] = {}
 1970:         for sc_name in sub_clusters:
 1971:             leader = group_leaders.get(sc_name)
 1972:             if leader and leader in active_non_hold:
 1973:                 meta_leaders[sc_name] = leader
 1974:         if len(meta_leaders) < 2:
 1975:             continue
 1976:         # Check if all leaders agree on direction
 1977:         leader_directions = {sc: votes.get(ldr, "HOLD")
 1978:                             for sc, ldr in meta_leaders.items()}
 1979:         active_dirs = set(leader_directions.values()) - {"HOLD"}
 1980:         if len(active_dirs) != 1:
 1981:             continue  # Leaders disagree — informative diversity, no penalty
 1982:         # All leaders agree: keep best-accuracy leader, penalize others
 1983:         best_sc = max(meta_leaders,
 1984:                       key=lambda sc: _leader_accuracy_key(meta_leaders[sc]))
 1985:         for sc_name, leader in meta_leaders.items():
 1986:             if sc_name != best_sc:
 1987:                 current = penalized_signals.get(leader, 1.0)
 1988:                 penalized_signals[leader] = min(current, _META_CLUSTER_PENALTY)
 1989:                 logger.debug(
 1990:                     "Meta-cluster %s: %s leader %s agrees with %s leader %s "
 1991:                     "— penalized to %.0f%%",
 1992:                     meta_name, sc_name, leader, best_sc,
 1993:                     meta_leaders[best_sc], _META_CLUSTER_PENALTY * 100,
 1994:                 )
 1995: 
 1996:     # Crisis mode detection: when multiple macro-external signals have degraded
 1997:     # accuracy, the market is in an abnormal regime (war, systemic crisis) where
 1998:     # trend-following breaks and mean-reversion becomes more reliable.
 1999:     #
 2000:     # 2026-04-19: Made crisis response conditional on trend signal performance.
 2001:     # When macro signals are broken but trend signals have >55% accuracy, the
 2002:     # crisis is in the macro indicators, not in the trend — penalizing trend
 2003:     # signals that are winning is actively harmful (observed: trend 61.6%,
 2004:     # EMA 62.9% being penalized 0.6x while crisis mode was active).
 2005:     _MACRO_CRISIS_SIGNALS = {"fear_greed", "macro_regime", "structure", "news_event", "sentiment"}
 2006:     broken_count = sum(
 2007:         1 for s in _MACRO_CRISIS_SIGNALS
 2008:         if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
 2009:         and accuracy_data.get(s, {}).get("accuracy", 0.5) < _CRISIS_THRESHOLD
 2010:     )
 2011:     crisis_mode = broken_count >= _CRISIS_MIN_BROKEN
 2012: 
 2013:     _TREND_SIGNALS = {"ema", "trend", "heikin_ashi", "volume_flow"}
 2014:     _MR_SIGNALS = {"mean_reversion", "calendar"}
 2015: 
 2016:     # Check if trend signals are actually underperforming before penalizing.
 2017:     # If avg trend accuracy > 55%, trend is capturing edge despite macro chaos.
 2018:     _CRISIS_TREND_ACCURACY_FLOOR = 0.55
 2019:     crisis_penalize_trend = False
 2020:     if crisis_mode:
 2021:         trend_accs = [
 2022:             accuracy_data.get(s, {}).get("accuracy", 0.5)
 2023:             for s in _TREND_SIGNALS
 2024:             if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
 2025:         ]
 2026:         avg_trend_acc = sum(trend_accs) / len(trend_accs) if trend_accs else 0.5
 2027:         crisis_penalize_trend = avg_trend_acc < _CRISIS_TREND_ACCURACY_FLOOR
 2028:         if crisis_penalize_trend:
 2029:             logger.info(
 2030:                 "Crisis mode active (full): %d/%d macro signals broken, "
 2031:                 "trend avg %.1f%% < %.0f%% floor — penalizing trend, boosting MR",
 2032:                 broken_count, len(_MACRO_CRISIS_SIGNALS),
 2033:                 avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
 2034:             )
 2035:         else:
 2036:             logger.info(
 2037:                 "Crisis mode active (partial): %d/%d macro signals broken, but "
 2038:                 "trend avg %.1f%% >= %.0f%% floor — NOT penalizing trend signals",
 2039:                 broken_count, len(_MACRO_CRISIS_SIGNALS),
 2040:                 avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
 2041:             )
 2042: 
 2043:     # Voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy gating reconfig).
 2044:     # Only the overall accuracy gate is relaxable — directional and correlation
 2045:     # gates still fire. Prevents regime-transition over-gating that silenced
 2046:     # ~8 voters in W15/W16.
 2047:     relaxation = _compute_gate_relaxation(
 2048:         votes=votes,
 2049:         accuracy_data=accuracy_data,
 2050:         excluded=excluded,
 2051:         group_gated=group_gated_signals,
 2052:         base_gate=gate,
 2053:         regime=regime,
 2054:     )
 2055:     if relaxation > 0:
 2056:         logger.debug(
 2057:             "Circuit breaker: relaxing accuracy gate by %.0fpp "
 2058:             "(base=%.2f -> effective=%.2f) to preserve voter diversity",
 2059:             relaxation * 100, gate, gate - relaxation,
 2060:         )
 2061: 
 2062:     # IC-based weight multiplier (2026-04-18): load IC data once per consensus
 2063:     # call. Returns {"global": {sig: {ic, icir, samples}}, "per_ticker": {...}}
 2064:     # or None if IC computation is unavailable.
 2065:     ic_cache = _get_ic_data(horizon) if horizon else None
 2066:     ic_global = ic_cache.get("global", {}) if ic_cache else {}
 2067:     ic_per_ticker = ic_cache.get("per_ticker", {}) if ic_cache else {}
 2068: 
 2069:     for signal_name, vote in votes.items():
 2070:         # P1-1 (2026-05-02 adversarial follow-ups): defensive — initialize
 2071:         # _rescued at the TOP of every iteration so a future contributor who
 2072:         # adds a third branch to the gate-check below cannot leak a stale
 2073:         # True from a prior iteration into line 2123 (`if _rescued: weight
 2074:         # *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY`). Today both arms of the
 2075:         # if/else at line 2072 set _rescued, so the bug doesn't manifest in
 2076:         # production — but the structural guarantee is now hardcoded.
 2077:         _rescued = False
 2078:         if vote == "HOLD":
 2079:             continue
 2080:         if signal_name in excluded:
 2081:             continue
 2082:         # Correlation group leader gating: entire group silenced
 2083:         if signal_name in group_gated_signals:
 2084:             gated_signals.append(signal_name)
 2085:             continue
 2086:         stats = accuracy_data.get(signal_name, {})
 2087:         acc = stats.get("accuracy", 0.5)
 2088:         samples = stats.get("total", 0)
 2089:         # Accuracy gate: skip signals that are below threshold with enough data.
 2090:         # Tiered: established signals (10000+ samples) use a tighter 50% gate;
 2091:         # newer signals use the standard 47% gate.
 2092:         # SC-P1-2 (2026-05-02 adversarial follow-ups): the high-sample tier
 2093:         # (10K+ samples, 0.50 gate) is NOT relaxed. A signal with 10K+ samples
 2094:         # at sub-50% accuracy has statistically demonstrated negative edge —
 2095:         # circuit-breaker relaxation must not let it back in. The standard
 2096:         # tier still relaxes uniformly so newer borderline signals can be
 2097:         # rescued during regime transitions.
 2098:         effective_gate = gate - relaxation
 2099:         if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
 2100:             effective_gate = max(
 2101:                 gate - relaxation,
 2102:                 _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD,
 2103:             )
 2104:         if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
 2105:             # Directional rescue: before gating, check if the vote direction
 2106:             # has strong enough accuracy to justify a reduced-weight vote.
 2107:             if vote == "BUY":
 2108:                 rescue_acc = stats.get("buy_accuracy", 0.0)
 2109:                 rescue_n = stats.get("total_buy", 0)
 2110:             else:
 2111:                 rescue_acc = stats.get("sell_accuracy", 0.0)
 2112:                 rescue_n = stats.get("total_sell", 0)
 2113:             if (rescue_n >= _DIRECTIONAL_RESCUE_MIN_SAMPLES
 2114:                     and rescue_acc >= _DIRECTIONAL_RESCUE_THRESHOLD):
 2115:                 logger.debug(
 2116:                     "Directional rescue: %s overall=%.1f%% (gated) but "
 2117:                     "%s=%.1f%% (%d sam) — rescued at %.0f%% weight",
 2118:                     signal_name, acc * 100, vote,
 2119:                     rescue_acc * 100, rescue_n,
 2120:                     _DIRECTIONAL_RESCUE_WEIGHT_PENALTY * 100,
 2121:                 )
 2122:                 # Fall through to weighting with rescue penalty applied later
 2123:                 _rescued = True
 2124:             else:
 2125:                 gated_signals.append(signal_name)
 2126:                 continue
 2127:         else:
 2128:             _rescued = False
 2129:         # Directional accuracy gate: gate individual BUY/SELL direction when
 2130:         # direction-specific accuracy is very poor, even if overall accuracy passes.
 2131:         # E.g., qwen3 overall=59.8% passes, but BUY=30.0% → gate BUY only.
 2132:         if vote == "BUY":
 2133:             dir_acc = stats.get("buy_accuracy", acc)
 2134:             dir_n = stats.get("total_buy", 0)
 2135:         else:
 2136:             dir_acc = stats.get("sell_accuracy", acc)
 2137:             dir_n = stats.get("total_sell", 0)
 2138:         if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
 2139:             gated_signals.append(f"{signal_name}_{vote}")
 2140:             continue
 2141:         # BUG-182: Use direction-specific accuracy as weight when available.
 2142:         # A signal with overall 60% accuracy may be 30% for BUY and 75% for SELL.
 2143:         # Using overall accuracy over-weights the weak direction.
 2144:         _DIR_WEIGHT_MIN_SAMPLES = 20
 2145:         if vote == "BUY" and stats.get("total_buy", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
 2146:             weight = stats.get("buy_accuracy", acc)  # BUG-185: .get() for cache safety
 2147:         elif vote == "SELL" and stats.get("total_sell", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
 2148:             weight = stats.get("sell_accuracy", acc)  # BUG-185: .get() for cache safety
 2149:         elif samples >= 20:
 2150:             weight = acc
 2151:         else:
 2152:             weight = 0.5
 2153:         # Apply directional rescue penalty: rescued signals contribute at
 2154:         # reduced weight since their overall accuracy failed the gate.
 2155:         if _rescued:
 2156:             weight *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY
 2157:         # IC-based weight adjustment: boost signals with high return-magnitude
 2158:         # predictive power, penalize phantom performers with zero IC.
 2159:         if ic_global:
 2160:             # Prefer per-ticker IC when available with enough samples
 2161:             _ic_info = None
 2162:             if ticker and ic_per_ticker:
 2163:                 _ic_info = ic_per_ticker.get(ticker, {}).get(signal_name)
 2164:                 if _ic_info and _ic_info.get("samples", 0) < _IC_MIN_SAMPLES:
 2165:                     _ic_info = None  # fall back to global
 2166:             if _ic_info is None:
 2167:                 _ic_info = ic_global.get(signal_name, {})
 2168:             _ic = _ic_info.get("ic", 0.0)
 2169:             _icir = _ic_info.get("icir", 0.0)
 2170:             _ic_n = _ic_info.get("samples", 0)
 2171:             ic_mult = _compute_ic_mult(_ic, _icir, _ic_n)
 2172:             weight *= ic_mult
 2173:         # Regime adjustment
 2174:         weight *= regime_mults.get(signal_name, 1.0)
 2175:         # Horizon-specific weight adjustment
 2176:         if signal_name in horizon_mults:
 2177:             weight *= horizon_mults[signal_name]
 2178:         # Macro-window downweight (2026-04-28). Composes with regime/
 2179:         # horizon multipliers — e.g., during a macro window in ranging
 2180:         # regime, sentiment hits 0.5 (macro) × 0.X (regime) × Y (horizon).
 2181:         # Only applies to MACRO_WINDOW_DOWNWEIGHT_SIGNALS — the
 2182:         # FORCE_HOLD signals were already mutated to HOLD above and won't
 2183:         # reach this branch.
 2184:         if macro_active and signal_name in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
 2185:             weight *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
 2186:         # Crisis mode adjustments: penalize trend signals (only if they're
 2187:         # underperforming), boost mean-reversion. See 2026-04-19 fix above.
 2188:         if crisis_mode:
 2189:             if signal_name in _TREND_SIGNALS and crisis_penalize_trend:
 2190:                 weight *= _CRISIS_TREND_PENALTY
 2191:             elif signal_name in _MR_SIGNALS:
 2192:                 weight *= _CRISIS_MR_BOOST
 2193:         # Activation frequency normalization (rarity * bias correction)
 2194:         act_data = activation_rates.get(signal_name, {})
 2195:         norm_weight = act_data.get("normalized_weight", 1.0)
 2196:         weight *= norm_weight
 2197:         # Activity rate cap: penalize signals with extremely high activation rates
 2198:         act_rate = act_data.get("activation_rate", 0.0)
 2199:         if act_rate > _ACTIVITY_RATE_CAP:
 2200:             weight *= _ACTIVITY_RATE_PENALTY
 2201:         # Correlation penalty: secondary signals in a group get reduced weight
 2202:         if signal_name in penalized_signals:
 2203:             weight *= penalized_signals[signal_name]
 2204:         # Directional bias penalty (2026-05-02 research): signals with extreme
 2205:         # BUY/SELL bias get penalized ONLY when voting in their bias direction.
 2206:         # Contrarian votes (rare, high-value) keep full weight.
 2207:         # E.g., calendar is 100% BUY — its BUY votes get 0.5x, but a rare
 2208:         # SELL (if it ever emits one) keeps 1.0x because that's genuinely
 2209:         # informative. Previous version penalized ALL votes equally.
 2210:         signal_bias = act_data.get("bias", 0.0)
 2211:         signal_samples = act_data.get("samples", 0)
 2212:         if signal_samples >= _BIAS_MIN_ACTIVE and signal_bias > _BIAS_THRESHOLD:
 2213:             buy_rate = act_data.get("buy_rate", 0.0)
 2214:             sell_rate = act_data.get("sell_rate", 0.0)
 2215:             bias_direction = "BUY" if buy_rate >= sell_rate else "SELL"
 2216:             if vote == bias_direction:
 2217:                 penalty = _BIAS_EXTREME_PENALTY if signal_bias > _BIAS_EXTREME_THRESHOLD else _BIAS_PENALTY
 2218:                 weight *= penalty
 2219:         if vote == "BUY":
 2220:             buy_weight += weight
 2221:         elif vote == "SELL":
 2222:             sell_weight += weight
 2223:     if gated_signals:
 2224:         logger.debug("Accuracy-gated signals (<%s%%): %s", ACCURACY_GATE_THRESHOLD * 100, gated_signals)
 2225:     total_weight = buy_weight + sell_weight
 2226:     if total_weight == 0:
 2227:         return "HOLD", 0.0
 2228:     buy_conf = buy_weight / total_weight
 2229:     sell_conf = sell_weight / total_weight
 2230:     if buy_conf > sell_conf and buy_conf >= 0.5:
 2231:         return "BUY", round(buy_conf, 4)
 2232:     if sell_conf > buy_conf and sell_conf >= 0.5:
 2233:         return "SELL", round(sell_conf, 4)
 2234:     return "HOLD", round(max(buy_conf, sell_conf), 4)
 2235: 
 2236: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"def _safe_accuracy|_safe_accuracy\\(\" portfolio/accuracy_stats.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-10T15:26:58.649964Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:

 exited 1 in 530ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "safe_accuracy" portfolio/accuracy_stats.py' in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-10T15:27:01.350051Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:

 exited 1 in 497ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "safe_accuracy" portfolio' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 508ms:
portfolio\signal_engine.py:1388:def _safe_accuracy(value, default):
portfolio\signal_engine.py:1455:        acc = _safe_accuracy(stats.get("accuracy"), default=0.5)
portfolio\signal_engine.py:1465:            dir_acc = _safe_accuracy(stats.get("buy_accuracy"), default=acc)
portfolio\signal_engine.py:1468:            dir_acc = _safe_accuracy(stats.get("sell_accuracy"), default=acc)
portfolio\signal_engine.py:1826:                    _clean_acc = _safe_accuracy(_clean.get(_acc_key), default=None)

codex
I’ve reached the first hard split: some findings describe current code accurately, while others are clearly stale against this branch because the code already carries explicit fixes and compatibility fields for the exact scenario. I’m verifying the remaining edge cases in `accuracy_stats.py` and `outcome_tracker.py` before I lock the counts.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"blend_accuracy_data\\(|accuracy_data = blend_accuracy_data|signal_best_horizon_accuracy|correct\\\"\\]\" portfolio" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_MIN_CHANGE_PCT|vote_correct|change_pct.*None|neutral" portfolio/accuracy_stats.py portfolio/outcome_tracker.py' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 760ms:
portfolio/accuracy_stats.py:167:_MIN_CHANGE_PCT = 0.05  # outcomes within ±0.05% are treated as neutral (skip)
portfolio/accuracy_stats.py:170:def _vote_correct(vote, change_pct, min_change_pct=None):
portfolio/accuracy_stats.py:173:    Returns True (correct), False (incorrect), or None (neutral — skip this outcome).
portfolio/accuracy_stats.py:177:    threshold = min_change_pct if min_change_pct is not None else _MIN_CHANGE_PCT
portfolio/accuracy_stats.py:178:    # 2026-04-22: some outcome entries have change_pct=None (missing backfill
portfolio/accuracy_stats.py:179:    # data for 4h+ horizons). Treat as neutral instead of TypeErroring — was
portfolio/accuracy_stats.py:182:    if change_pct is None or abs(change_pct) < threshold:
portfolio/accuracy_stats.py:183:        return None  # neutral — price didn't move enough to judge (or unknown)
portfolio/accuracy_stats.py:206:    # 2026-04-22 follow-up: count outcomes we skip because change_pct is None.
portfolio/accuracy_stats.py:226:            if change_pct is None:
portfolio/accuracy_stats.py:234:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:236:                    continue  # neutral outcome — don't count
portfolio/accuracy_stats.py:253:            "change_pct=None — check outcome_tracker / signal_db backfill",
portfolio/accuracy_stats.py:351:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:353:                    continue  # neutral outcome — skip
portfolio/accuracy_stats.py:398:                  Must exceed _MIN_CHANGE_PCT (5 bps) to filter beyond
portfolio/accuracy_stats.py:399:                  the neutral outcome threshold. Metals warrants ~10 bps,
portfolio/accuracy_stats.py:424:            if change_pct is None:
portfolio/accuracy_stats.py:433:                # Skip neutral outcomes (below minimum move)
portfolio/accuracy_stats.py:434:                if abs(change_pct) < _MIN_CHANGE_PCT:
portfolio/accuracy_stats.py:502:            result_val = _vote_correct(consensus, change_pct)
portfolio/accuracy_stats.py:537:            result_val = _vote_correct(consensus, change_pct)
portfolio/accuracy_stats.py:602:            result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:625:    For each non-HOLD signal vote with a non-neutral outcome, compute the
portfolio/accuracy_stats.py:630:    Neutral outcomes (|change_pct| < _MIN_CHANGE_PCT) are skipped.
portfolio/accuracy_stats.py:696:            # 2026-05-03: None-guard mirrors _vote_correct (line 112) and the
portfolio/accuracy_stats.py:702:            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
portfolio/accuracy_stats.py:703:                continue  # neutral / unknown outcome — skip
portfolio/accuracy_stats.py:887:    # accuracy=0.80 should default to neutral 0.5, not vote at 0.80).
portfolio/accuracy_stats.py:917:        # falling back to a neutral 0.5 otherwise (matches pre-patch
portfolio/accuracy_stats.py:928:            blended = 0.5  # immature signal: neutral default
portfolio/accuracy_stats.py:944:        # field had already been set back to neutral 0.5. Omit directional
portfolio/accuracy_stats.py:1352:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1354:                    continue  # neutral outcome — don't count
portfolio/accuracy_stats.py:1760:                    result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1842:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:2026:            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:

2026-05-10T15:27:10.646608Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:

 exited 1 in 809ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n blend_accuracy_data portfolio' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n '\"correct\"' portfolio/accuracy_stats.py portfolio/signal_engine.py portfolio/*.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-10T15:27:14.948312Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
portfolio/signal_engine.py:3471:                    "correct": t_stats.get("correct", 0),
portfolio/signal_engine.py:3492:                    "correct": int(per_ticker_acc * per_ticker_samples),
portfolio/accuracy_stats.py:203:    stats = {s: {"correct": 0, "total": 0,
portfolio/accuracy_stats.py:247:                    stats[sig_name]["correct"] += 1
portfolio/accuracy_stats.py:260:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:264:            "correct": s["correct"],
portfolio/accuracy_stats.py:382:            "correct": int(round(w_correct)),
portfolio/accuracy_stats.py:412:    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}
portfolio/accuracy_stats.py:441:                    stats[sig_name]["correct"] += 1
portfolio/accuracy_stats.py:446:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:448:            "correct": s["correct"],
portfolio/accuracy_stats.py:511:        "correct": correct,
portfolio/accuracy_stats.py:521:    stats = defaultdict(lambda: {"correct": 0, "total": 0})
portfolio/accuracy_stats.py:542:                stats[ticker]["correct"] += 1
portfolio/accuracy_stats.py:546:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:548:            "correct": s["correct"],
portfolio/accuracy_stats.py:571:        dict: {ticker: {"accuracy": float, "samples": int, "correct": int}}
portfolio/accuracy_stats.py:584:    stats = defaultdict(lambda: {"correct": 0, "total": 0})
portfolio/accuracy_stats.py:607:                stats[ticker]["correct"] += 1
portfolio/accuracy_stats.py:615:            "accuracy": data["correct"] / total,
portfolio/accuracy_stats.py:617:            "correct": data["correct"],
portfolio/accuracy_stats.py:936:            "correct": int(round(blended * total)),  # BUG-186
portfolio/accuracy_stats.py:1331:    regime_stats = defaultdict(lambda: {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES})
portfolio/accuracy_stats.py:1357:                    regime_stats[regime][sig_name]["correct"] += 1
portfolio/accuracy_stats.py:1365:            acc = s["correct"] / s["total"]
portfolio/accuracy_stats.py:1367:                "correct": s["correct"],
portfolio/accuracy_stats.py:1740:        s: {h: {"correct": 0, "total": 0} for h in HORIZONS}
portfolio/accuracy_stats.py:1765:                        stats[sig_name][horizon]["correct"] += 1
portfolio/accuracy_stats.py:1779:            acc = h_stats["correct"] / total
portfolio/accuracy_stats.py:1784:                best_correct = h_stats["correct"]
portfolio/accuracy_stats.py:1792:                "correct": best_correct,
portfolio/accuracy_stats.py:1823:        return {"correct": 0, "total": 0, "correct_buy": 0, "total_buy": 0,
portfolio/accuracy_stats.py:1850:                        s["correct"] += 1
portfolio/accuracy_stats.py:1855:                        s["correct"] += 1
portfolio/accuracy_stats.py:1864:            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:1868:                "correct": s["correct"],
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 701ms:
portfolio/signal_engine.py:3471:                    "correct": t_stats.get("correct", 0),
portfolio/signal_engine.py:3492:                    "correct": int(per_ticker_acc * per_ticker_samples),
portfolio/accuracy_stats.py:203:    stats = {s: {"correct": 0, "total": 0,
portfolio/accuracy_stats.py:247:                    stats[sig_name]["correct"] += 1
portfolio/accuracy_stats.py:260:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:264:            "correct": s["correct"],
portfolio/accuracy_stats.py:382:            "correct": int(round(w_correct)),
portfolio/accuracy_stats.py:412:    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}
portfolio/accuracy_stats.py:441:                    stats[sig_name]["correct"] += 1
portfolio/accuracy_stats.py:446:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:448:            "correct": s["correct"],
portfolio/accuracy_stats.py:511:        "correct": correct,
portfolio/accuracy_stats.py:521:    stats = defaultdict(lambda: {"correct": 0, "total": 0})
portfolio/accuracy_stats.py:542:                stats[ticker]["correct"] += 1
portfolio/accuracy_stats.py:546:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:548:            "correct": s["correct"],
portfolio/accuracy_stats.py:571:        dict: {ticker: {"accuracy": float, "samples": int, "correct": int}}
portfolio/accuracy_stats.py:584:    stats = defaultdict(lambda: {"correct": 0, "total": 0})
portfolio/accuracy_stats.py:607:                stats[ticker]["correct"] += 1
portfolio/accuracy_stats.py:615:            "accuracy": data["correct"] / total,
portfolio/accuracy_stats.py:617:            "correct": data["correct"],
portfolio/accuracy_stats.py:936:            "correct": int(round(blended * total)),  # BUG-186
portfolio/accuracy_stats.py:1331:    regime_stats = defaultdict(lambda: {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES})
portfolio/accuracy_stats.py:1357:                    regime_stats[regime][sig_name]["correct"] += 1
portfolio/accuracy_stats.py:1365:            acc = s["correct"] / s["total"]
portfolio/accuracy_stats.py:1367:                "correct": s["correct"],
portfolio/accuracy_stats.py:1740:        s: {h: {"correct": 0, "total": 0} for h in HORIZONS}
portfolio/accuracy_stats.py:1765:                        stats[sig_name][horizon]["correct"] += 1
portfolio/accuracy_stats.py:1779:            acc = h_stats["correct"] / total
portfolio/accuracy_stats.py:1784:                best_correct = h_stats["correct"]
portfolio/accuracy_stats.py:1792:                "correct": best_correct,
portfolio/accuracy_stats.py:1823:        return {"correct": 0, "total": 0, "correct_buy": 0, "total_buy": 0,
portfolio/accuracy_stats.py:1850:                        s["correct"] += 1
portfolio/accuracy_stats.py:1855:                        s["correct"] += 1
portfolio/accuracy_stats.py:1864:            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:1868:                "correct": s["correct"],
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 succeeded in 718ms:
portfolio\accuracy_stats.py:863:def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
portfolio\signal_engine.py:3387:            blend_accuracy_data,
portfolio\signal_engine.py:3416:        accuracy_data = blend_accuracy_data(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=3458; $end=3502; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 771ms:
 3458:             logger.debug("Regime-conditional accuracy unavailable", exc_info=True)
 3459: 
 3460:     # BUG-158: Override global accuracy with per-ticker accuracy for ALL signals.
 3461:     # Per-ticker variance is enormous: fear_greed is 93.8% on XAG-USD but 25.9%
 3462:     # globally. Using global accuracy throws away alpha on specific instruments.
 3463:     # H3: Skip when primary load failed to preserve fail-closed gate.
 3464:     _PER_TICKER_MIN_SAMPLES = 30
 3465:     if not _accuracy_failed and _ticker_acc_data:
 3466:         for sig_name, t_stats in _ticker_acc_data.items():
 3467:             if t_stats.get("total", 0) >= _PER_TICKER_MIN_SAMPLES:
 3468:                 override = {
 3469:                     "accuracy": t_stats["accuracy"],
 3470:                     "total": t_stats["total"],
 3471:                     "correct": t_stats.get("correct", 0),
 3472:                     "pct": t_stats.get("pct", round(t_stats["accuracy"] * 100, 1)),
 3473:                 }
 3474:                 # Copy directional fields for per-ticker directional gating.
 3475:                 # Without these, _weighted_consensus directional gate falls back
 3476:                 # to overall per-ticker accuracy, missing direction-specific
 3477:                 # weaknesses (e.g., ministral BUY 15% on XAG even if overall 20%).
 3478:                 for field in ("correct_buy", "total_buy", "buy_accuracy",
 3479:                               "correct_sell", "total_sell", "sell_accuracy"):
 3480:                     if field in t_stats:
 3481:                         override[field] = t_stats[field]
 3482:                 accuracy_data[sig_name] = override
 3483:     elif not _accuracy_failed:
 3484:         # Fallback: LLM-specific per-ticker data from extra_info
 3485:         for llm_sig in ("qwen3", "ministral"):
 3486:             per_ticker_acc = extra_info.get(f"{llm_sig}_accuracy")
 3487:             per_ticker_samples = extra_info.get(f"{llm_sig}_samples", 0)
 3488:             if per_ticker_acc is not None and per_ticker_samples >= 20:
 3489:                 accuracy_data[llm_sig] = {
 3490:                     "accuracy": per_ticker_acc,
 3491:                     "total": per_ticker_samples,
 3492:                     "correct": int(per_ticker_acc * per_ticker_samples),
 3493:                     "pct": round(per_ticker_acc * 100, 1),
 3494:                 }
 3495: 
 3496:     # Utility boost: scale accuracy weight by return-based utility score.
 3497:     # Utility boost and best-horizon overlay.
 3498:     # H3: Skip when primary load failed to preserve fail-closed gate.
 3499:     if not _accuracy_failed:
 3500:         try:
 3501:             from portfolio.accuracy_stats import signal_utility
 3502:             # BUG-135: Use acc_horizon (not hardcoded "1d") so utility boost

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "correct" portfolio/signal_engine.py portfolio/accuracy_stats.py | Select-Object -First 200' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 545ms:
portfolio/accuracy_stats.py:170:def _vote_correct(vote, change_pct, min_change_pct=None):
portfolio/accuracy_stats.py:173:    Returns True (correct), False (incorrect), or None (neutral — skip this outcome).
portfolio/accuracy_stats.py:199:        dict: {signal_name: {correct, total, accuracy, pct}} for each signal.
portfolio/accuracy_stats.py:203:    stats = {s: {"correct": 0, "total": 0,
portfolio/accuracy_stats.py:204:                 "correct_buy": 0, "total_buy": 0,
portfolio/accuracy_stats.py:205:                 "correct_sell": 0, "total_sell": 0} for s in SIGNAL_NAMES}
portfolio/accuracy_stats.py:234:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:241:                        stats[sig_name]["correct_buy"] += 1
portfolio/accuracy_stats.py:245:                        stats[sig_name]["correct_sell"] += 1
portfolio/accuracy_stats.py:247:                    stats[sig_name]["correct"] += 1
portfolio/accuracy_stats.py:260:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:261:        buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
portfolio/accuracy_stats.py:262:        sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
portfolio/accuracy_stats.py:264:            "correct": s["correct"],
portfolio/accuracy_stats.py:270:            "correct_buy": s["correct_buy"],
portfolio/accuracy_stats.py:273:            "correct_sell": s["correct_sell"],
portfolio/accuracy_stats.py:306:        dict: {signal_name: {accuracy, total_weight, effective_samples, total, correct, pct}}
portfolio/accuracy_stats.py:307:        where ``total`` and ``correct`` are int(round(...)) of the weighted sums
portfolio/accuracy_stats.py:318:    # Accumulate per-signal: weighted_total, weighted_correct, sum_of_sq_weights
portfolio/accuracy_stats.py:320:        s: {"w_total": 0.0, "w_correct": 0.0, "sum_w2": 0.0}
portfolio/accuracy_stats.py:351:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:358:                    stats[sig_name]["w_correct"] += weight
portfolio/accuracy_stats.py:364:        w_correct = s["w_correct"]
portfolio/accuracy_stats.py:368:            accuracy = w_correct / w_total
portfolio/accuracy_stats.py:382:            "correct": int(round(w_correct)),
portfolio/accuracy_stats.py:391:    A signal vote is only counted as correct if the price move exceeds
portfolio/accuracy_stats.py:393:    signals that are "technically correct" but unprofitable after execution.
portfolio/accuracy_stats.py:404:        dict: {signal_name: {correct, total, accuracy, pct, cost_bps}}
portfolio/accuracy_stats.py:409:    # Cost threshold: moves below this are unprofitable even if directionally correct
portfolio/accuracy_stats.py:412:    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}
portfolio/accuracy_stats.py:439:                # Cost-adjusted: correct only if move exceeds cost
portfolio/accuracy_stats.py:441:                    stats[sig_name]["correct"] += 1
portfolio/accuracy_stats.py:446:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:448:            "correct": s["correct"],
portfolio/accuracy_stats.py:485:    correct = 0
portfolio/accuracy_stats.py:502:            result_val = _vote_correct(consensus, change_pct)
portfolio/accuracy_stats.py:507:                correct += 1
portfolio/accuracy_stats.py:509:    acc = correct / total if total > 0 else 0.0
portfolio/accuracy_stats.py:511:        "correct": correct,
portfolio/accuracy_stats.py:521:    stats = defaultdict(lambda: {"correct": 0, "total": 0})
portfolio/accuracy_stats.py:537:            result_val = _vote_correct(consensus, change_pct)
portfolio/accuracy_stats.py:542:                stats[ticker]["correct"] += 1
portfolio/accuracy_stats.py:546:        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:548:            "correct": s["correct"],
portfolio/accuracy_stats.py:571:        dict: {ticker: {"accuracy": float, "samples": int, "correct": int}}
portfolio/accuracy_stats.py:584:    stats = defaultdict(lambda: {"correct": 0, "total": 0})
portfolio/accuracy_stats.py:602:            result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:607:                stats[ticker]["correct"] += 1
portfolio/accuracy_stats.py:615:            "accuracy": data["correct"] / total,
portfolio/accuracy_stats.py:617:            "correct": data["correct"],
portfolio/accuracy_stats.py:627:      - BUY  → +change_pct   (positive = correct direction)
portfolio/accuracy_stats.py:628:      - SELL → -change_pct   (negative change = correct direction → positive return)
portfolio/accuracy_stats.py:696:            # 2026-05-03: None-guard mirrors _vote_correct (line 112) and the
portfolio/accuracy_stats.py:710:                # Directional return: positive when signal was correct
portfolio/accuracy_stats.py:747:    across all processes is correct. The satellite loops do not call this
portfolio/accuracy_stats.py:872:        alltime: Dict of {signal_name: {accuracy, total, correct, pct}}.
portfolio/accuracy_stats.py:873:        recent: Dict of {signal_name: {accuracy, total, correct, pct}}.
portfolio/accuracy_stats.py:880:        dict: Blended {signal_name: {accuracy, total, correct, pct}}.
portfolio/accuracy_stats.py:936:            "correct": int(round(blended * total)),  # BUG-186
portfolio/accuracy_stats.py:1287:                f"{sig_name:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%{disabled_tag}"
portfolio/accuracy_stats.py:1294:                f"{'Consensus':<16}{cons['correct']:>7}  {cons['total']:>5}  {cons['accuracy']*100:>7.1f}%"
portfolio/accuracy_stats.py:1307:                    f"{ticker:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%"
portfolio/accuracy_stats.py:1324:        dict: {regime: {signal_name: {correct, total, accuracy, pct}}}
portfolio/accuracy_stats.py:1330:    # {regime: {signal_name: {correct, total}}}
portfolio/accuracy_stats.py:1331:    regime_stats = defaultdict(lambda: {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES})
portfolio/accuracy_stats.py:1352:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1357:                    regime_stats[regime][sig_name]["correct"] += 1
portfolio/accuracy_stats.py:1365:            acc = s["correct"] / s["total"]
portfolio/accuracy_stats.py:1367:                "correct": s["correct"],
portfolio/accuracy_stats.py:1498:    failure is logged but doesn't raise — a stale L2 self-corrects on its
portfolio/accuracy_stats.py:1722:        dict: {signal_name: {accuracy, total, correct, pct, best_horizon}}
portfolio/accuracy_stats.py:1738:    # {sig_name: {horizon: {correct, total}}}
portfolio/accuracy_stats.py:1740:        s: {h: {"correct": 0, "total": 0} for h in HORIZONS}
portfolio/accuracy_stats.py:1760:                    result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1765:                        stats[sig_name][horizon]["correct"] += 1
portfolio/accuracy_stats.py:1772:        best_correct = 0
portfolio/accuracy_stats.py:1779:            acc = h_stats["correct"] / total
portfolio/accuracy_stats.py:1784:                best_correct = h_stats["correct"]
portfolio/accuracy_stats.py:1792:                "correct": best_correct,
portfolio/accuracy_stats.py:1810:    Returns nested dict: {ticker: {signal_name: {correct, total, accuracy, pct,
portfolio/accuracy_stats.py:1811:        correct_buy, total_buy, buy_accuracy, correct_sell, total_sell, sell_accuracy}}}
portfolio/accuracy_stats.py:1821:    # {ticker: {signal: {correct, total, correct_buy, total_buy, correct_sell, total_sell}}}
portfolio/accuracy_stats.py:1823:        return {"correct": 0, "total": 0, "correct_buy": 0, "total_buy": 0,
portfolio/accuracy_stats.py:1824:                "correct_sell": 0, "total_sell": 0}
portfolio/accuracy_stats.py:1842:                result_val = _vote_correct(vote, change_pct)
portfolio/accuracy_stats.py:1850:                        s["correct"] += 1
portfolio/accuracy_stats.py:1851:                        s["correct_buy"] += 1
portfolio/accuracy_stats.py:1855:                        s["correct"] += 1
portfolio/accuracy_stats.py:1856:                        s["correct_sell"] += 1
portfolio/accuracy_stats.py:1864:            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
portfolio/accuracy_stats.py:1865:            buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
portfolio/accuracy_stats.py:1866:            sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
portfolio/accuracy_stats.py:1868:                "correct": s["correct"],
portfolio/accuracy_stats.py:1872:                "correct_buy": s["correct_buy"],
portfolio/accuracy_stats.py:1875:                "correct_sell": s["correct_sell"],
portfolio/accuracy_stats.py:1895:            signal, correct, total, accuracy, pct.
portfolio/accuracy_stats.py:1918:    Cache structure: {horizon: {ticker: {signal: {correct, total, accuracy, pct}}}, "time": ...}
portfolio/accuracy_stats.py:1998:        predicted_confidence, actual_accuracy, sample_count, correct_count.
portfolio/accuracy_stats.py:2004:    # Build list of (confidence, correct_bool) tuples
portfolio/accuracy_stats.py:2029:            correct = (consensus == "BUY" and change_pct > 0) or \
portfolio/accuracy_stats.py:2031:            samples.append((confidence, correct))
portfolio/accuracy_stats.py:2037:        bucket_samples = [(c, correct) for c, correct in samples if lo <= c < hi]
portfolio/accuracy_stats.py:2042:                "actual_accuracy": None, "sample_count": 0, "correct_count": 0,
portfolio/accuracy_stats.py:2045:        correct_count = sum(1 for _, c in bucket_samples if c)
portfolio/accuracy_stats.py:2051:            "actual_accuracy": round(correct_count / n, 4),
portfolio/accuracy_stats.py:2052:            "sample_count": n, "correct_count": correct_count,
portfolio/signal_engine.py:821:        # fast-blended accuracy is ~40% (correctly gated), but at 3h it escapes.
portfolio/signal_engine.py:2193:        # Activation frequency normalization (rarity * bias correction)
portfolio/signal_engine.py:3471:                    "correct": t_stats.get("correct", 0),
portfolio/signal_engine.py:3478:                for field in ("correct_buy", "total_buy", "buy_accuracy",
portfolio/signal_engine.py:3479:                              "correct_sell", "total_sell", "sell_accuracy"):
portfolio/signal_engine.py:3492:                    "correct": int(per_ticker_acc * per_ticker_samples),

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/accuracy_stats.py'; "'$start=860; $end=882; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 598ms:
  860: _BLEND_DEFAULT_MIN_RECENT_SAMPLES = 30
  861: 
  862: 
  863: def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
  864:                         normal_weight=0.70, fast_weight=0.90,
  865:                         min_recent_samples=_BLEND_DEFAULT_MIN_RECENT_SAMPLES):
  866:     """Blend all-time and recent accuracy using adaptive recency weighting.
  867: 
  868:     When recent accuracy diverges sharply from all-time (> divergence_threshold),
  869:     fast-track to higher recent weight for faster regime adaptation.
  870: 
  871:     Args:
  872:         alltime: Dict of {signal_name: {accuracy, total, correct, pct}}.
  873:         recent: Dict of {signal_name: {accuracy, total, correct, pct}}.
  874:         divergence_threshold: Absolute accuracy difference that triggers fast blend.
  875:         normal_weight: Recent weight when divergence is below threshold.
  876:         fast_weight: Recent weight when divergence exceeds threshold.
  877:         min_recent_samples: Minimum recent samples before blending (else use alltime).
  878: 
  879:     Returns:
  880:         dict: Blended {signal_name: {accuracy, total, correct, pct}}.
  881:     """
  882:     if not alltime and not recent:

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=1380; $end=1409; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 655ms:
 1380: # when pure_trend, oscillator_trend, and structural_flow leaders vote identically.
 1381: # When leaders DISAGREE, no penalty is applied — that's informative diversity.
 1382: _META_CLUSTER_GROUPS: dict[str, list[str]] = {
 1383:     "trend_mega": ["pure_trend", "oscillator_trend", "structural_flow"],
 1384: }
 1385: _META_CLUSTER_PENALTY = 0.35  # 2nd/3rd agreeing leaders get 35% weight
 1386: 
 1387: 
 1388: def _safe_accuracy(value, default):
 1389:     """Coerce an accuracy value to a clean float, mapping None/NaN/inf to `default`.
 1390: 
 1391:     2026-04-17 (P1-C): the live consensus path previously crashed with
 1392:     TypeError when `accuracy_data[sig]` held explicit None (e.g., from a
 1393:     half-written cache), and with a silent fall-through-as-valid when it
 1394:     held NaN (every comparison with NaN is False). This helper normalizes.
 1395:     """
 1396:     import math
 1397:     if value is None:
 1398:         return default
 1399:     try:
 1400:         f = float(value)
 1401:     except (TypeError, ValueError):
 1402:         return default
 1403:     if math.isnan(f) or math.isinf(f):
 1404:         return default
 1405:     return f
 1406: 
 1407: 
 1408: def _safe_sample_count(value):
 1409:     """Coerce a sample count to a non-negative int; None/NaN/negative -> 0."""

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=3378; $end=3430; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 644ms:
 3378:     # Weighted consensus using accuracy data, regime, and activation frequency
 3379:     # (regime already computed early in the function for F&G gating)
 3380:     accuracy_data = {}
 3381:     activation_rates = {}
 3382:     # H3: Define acc_horizon before the try/except so the except block and
 3383:     # subsequent code can reference it even if the import fails.
 3384:     acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
 3385:     try:
 3386:         from portfolio.accuracy_stats import (
 3387:             blend_accuracy_data,
 3388:             get_or_compute_accuracy,
 3389:             get_or_compute_per_ticker_accuracy,
 3390:             get_or_compute_recent_accuracy,
 3391:             load_cached_activation_rates,
 3392:         )
 3393: 
 3394:         # BUG-178 (2026-04-16): the get_or_compute_* helpers serialize the
 3395:         # cache-miss compute via _accuracy_compute_lock so 5 parallel ticker
 3396:         # threads don't each pay the 7s+ cost of loading 50,000 signal-log
 3397:         # entries when the 1h TTL expires (was 215s wall before the fix).
 3398:         # See accuracy_stats.py for the lock rationale.
 3399:         alltime = get_or_compute_accuracy(acc_horizon)
 3400:         recent = get_or_compute_recent_accuracy(acc_horizon, days=7)
 3401:         # BUG-164 lazy-populate per-ticker consensus accuracy — _ptc_key
 3402:         # convention preserved by get_or_compute_per_ticker_accuracy.
 3403:         _ptc_data = get_or_compute_per_ticker_accuracy(acc_horizon)
 3404:         # RES-2026-04-17: Pass per-ticker consensus accuracy into extra_info
 3405:         # so apply_confidence_penalties can penalize tickers where the consensus
 3406:         # system itself has below-coinflip accuracy (e.g. ETH-USD 47.7% at 3h,
 3407:         # MSTR 45.9%). The consensus is the aggregated output, not individual
 3408:         # signals — if it's below 50%, the ensemble is net-negative for this ticker.
 3409:         if _ptc_data and ticker and isinstance(_ptc_data, dict):
 3410:             _ticker_ptc = _ptc_data.get(ticker)
 3411:             if isinstance(_ticker_ptc, dict):
 3412:                 extra_info["_ptc_accuracy"] = _ticker_ptc.get("accuracy")
 3413:                 extra_info["_ptc_samples"] = _ticker_ptc.get("total", 0)
 3414: 
 3415:         # ARCH-23: Use shared blend function (replaces inline logic).
 3416:         accuracy_data = blend_accuracy_data(
 3417:             alltime, recent,
 3418:             divergence_threshold=_RECENCY_DIVERGENCE_THRESHOLD,
 3419:             normal_weight=_RECENCY_WEIGHT_NORMAL,
 3420:             fast_weight=_RECENCY_WEIGHT_FAST,
 3421:             min_recent_samples=_RECENCY_MIN_SAMPLES,
 3422:         )
 3423: 
 3424:         activation_rates = load_cached_activation_rates()
 3425:         _accuracy_failed = False
 3426:     except Exception:
 3427:         logger.error("Accuracy stats load failed", exc_info=True)
 3428:         # H3: Fail-closed: gate all signals (0% accuracy, 999 samples) rather than
 3429:         # leaving accuracy_data = {} which bypasses the accuracy gate entirely.
 3430:         accuracy_data = {sig: {"accuracy": 0.0, "total": 999} for sig in SIGNAL_NAMES}

codex
I’ve isolated one likely false-positive already: the neutral-outcome claim ignores `_vote_correct`, which explicitly skips `change_pct` values inside ±0.05%. I’m confirming the rest of the dataflow so each verdict cites the exact guard or missing guard.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/accuracy_stats.py'; "'$start=167; $end=186; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_persistence_filtered|_voters_post_filter|dynamic_min_voters|apply_confidence_penalties|_weighted_consensus\\(\" portfolio/signal_engine.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"sorted\\(|sort\\(|key=.*correct|correct.*sort|top.*correct\" portfolio/accuracy_stats.py portfolio/signal_engine.py" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 512ms:
1498:def _dynamic_min_voters_for_regime(regime):
1500:    the circuit breaker and apply_confidence_penalties.
1507:    2026-04-17 (P2-C/P2-D): de-duplicated. apply_confidence_penalties
1557:    #     Matches downstream's `apply_confidence_penalties` which checks
1575:    min_regime_quorum = _dynamic_min_voters_for_regime(regime)
1738:def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
2416:def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
2509:    active_voters = extra_info.get("_voters_post_filter",
2511:    dynamic_min = _dynamic_min_voters_for_regime(regime)
2515:            "stage": "dynamic_min_voters", "regime": regime,
3405:        # so apply_confidence_penalties can penalize tickers where the consensus
3554:        extra_info["_persistence_filtered"] = _filtered_count
3569:    weighted_action, weighted_conf = _weighted_consensus(
3600:    extra_info["_voters_post_filter"] = post_persistence_voters
3621:    action, conf, penalty_log = apply_confidence_penalties(

 succeeded in 528ms:
portfolio/accuracy_stats.py:1274:        sorted_sigs = sorted(
portfolio/accuracy_stats.py:1301:            sorted_tickers = sorted(
portfolio/accuracy_stats.py:1683:    alerts.sort(key=lambda a: abs(a["change"]), reverse=True)
portfolio/accuracy_stats.py:1903:    ranked.sort(key=lambda x: x["accuracy"], reverse=True)
portfolio/signal_engine.py:535:        f"valid keys: {sorted(_VALID_HORIZON_KEYS)}")
portfolio/signal_engine.py:1896:        ranked = sorted(

 succeeded in 583ms:
  167: _MIN_CHANGE_PCT = 0.05  # outcomes within ±0.05% are treated as neutral (skip)
  168: 
  169: 
  170: def _vote_correct(vote, change_pct, min_change_pct=None):
  171:     """Check if a signal vote matches the price outcome.
  172: 
  173:     Returns True (correct), False (incorrect), or None (neutral — skip this outcome).
  174:     Outcomes within ±min_change_pct are considered noise and should not count
  175:     for or against the signal's accuracy.
  176:     """
  177:     threshold = min_change_pct if min_change_pct is not None else _MIN_CHANGE_PCT
  178:     # 2026-04-22: some outcome entries have change_pct=None (missing backfill
  179:     # data for 4h+ horizons). Treat as neutral instead of TypeErroring — was
  180:     # killing --accuracy report mid-horizon. Matches the None-guard pattern
  181:     # at accuracy_stats.py:1617 and in ic_computation / train_signal_weights.
  182:     if change_pct is None or abs(change_pct) < threshold:
  183:         return None  # neutral — price didn't move enough to judge (or unknown)
  184:     if vote == "BUY" and change_pct > 0:
  185:         return True
  186:     return bool(vote == "SELL" and change_pct < 0)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/accuracy_stats.py'; "'$ranges=@(@(1268,1305),@(1896,1908)); foreach($r in $ranges){$start=$r[0];$end=$r[1];$i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }; '"'' }" in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 599ms:
 1268:         print()
 1269:         print(f"--- {h} Horizon ({horizon_counts[h]} entries with outcomes) ---")
 1270:         print()
 1271: 
 1272:         # ARCH-24: Pass pre-loaded entries to avoid re-reading 68MB file per call.
 1273:         sig_acc = signal_accuracy(h, entries=entries)
 1274:         sorted_sigs = sorted(
 1275:             SIGNAL_NAMES, key=lambda s: sig_acc[s]["accuracy"], reverse=True
 1276:         )
 1277: 
 1278:         print(f"{'Signal':<16}{'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
 1279:         print(f"{'------':<16}{'-------':>7}  {'-----':>5}  {'--------':>8}")
 1280: 
 1281:         for sig_name in sorted_sigs:
 1282:             s = sig_acc[sig_name]
 1283:             if s["total"] == 0:
 1284:                 continue
 1285:             disabled_tag = " (OFF)" if sig_name in DISABLED_SIGNALS else ""
 1286:             print(
 1287:                 f"{sig_name:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%{disabled_tag}"
 1288:             )
 1289: 
 1290:         cons = consensus_accuracy(h, entries=entries)
 1291:         print()
 1292:         if cons["total"] > 0:
 1293:             print(
 1294:                 f"{'Consensus':<16}{cons['correct']:>7}  {cons['total']:>5}  {cons['accuracy']*100:>7.1f}%"
 1295:             )
 1296: 
 1297:         ticker_acc = per_ticker_accuracy(h, entries=entries)
 1298:         if ticker_acc:
 1299:             print()
 1300:             print("Per-Ticker:")
 1301:             sorted_tickers = sorted(
 1302:                 ticker_acc.keys(), key=lambda t: ticker_acc[t]["accuracy"], reverse=True
 1303:             )
 1304:             for ticker in sorted_tickers:
 1305:                 s = ticker_acc[ticker]

 1896:     """
 1897:     all_data = accuracy_by_ticker_signal(horizon, min_samples=min_samples)
 1898:     ticker_data = all_data.get(ticker, {})
 1899:     ranked = [
 1900:         {"signal": sig, **data}
 1901:         for sig, data in ticker_data.items()
 1902:     ]
 1903:     ranked.sort(key=lambda x: x["accuracy"], reverse=True)
 1904:     return ranked
 1905: 
 1906: 
 1907: # ---------------------------------------------------------------------------
 1908: # Per-ticker accuracy cache


exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_confluence_score|confluence" portfolio/signal_engine.py portfolio/*.py' in Q:\finance-analyzer\.worktrees\adv-signals-core
2026-05-10T15:28:10.111582Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
portfolio/signal_engine.py:2237:def _confluence_score(votes, indicators):
portfolio/signal_engine.py:3590:    confluence = _confluence_score(votes, extra_info)
portfolio/signal_engine.py:3614:    extra_info["_confluence_score"] = confluence
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 513ms:
portfolio/signal_engine.py:2237:def _confluence_score(votes, indicators):
portfolio/signal_engine.py:3590:    confluence = _confluence_score(votes, extra_info)
portfolio/signal_engine.py:3614:    extra_info["_confluence_score"] = confluence
rg: portfolio/*.py: The filename, directory name, or volume label syntax is incorrect. (os error 123)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_confluence_score|confluence_score|_confluence" .' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 488ms:
.\portfolio\signal_engine.py:2237:def _confluence_score(votes, indicators):
.\portfolio\signal_engine.py:3590:    confluence = _confluence_score(votes, extra_info)
.\portfolio\signal_engine.py:3614:    extra_info["_confluence_score"] = confluence

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"apply_confidence_penalties\\(\" -g\"*.py\" ." in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 530ms:
.\portfolio\signal_engine.py:2416:def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
.\portfolio\signal_engine.py:3621:    action, conf, penalty_log = apply_confidence_penalties(

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/accuracy_stats.py'; "'$start=220; $end=247; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 565ms:
  220:             outcome = outcomes.get(ticker, {}).get(horizon)
  221:             if not outcome:
  222:                 continue
  223:             total_outcomes_seen += 1
  224: 
  225:             change_pct = outcome.get("change_pct", 0)
  226:             if change_pct is None:
  227:                 null_change_pct_skipped += 1
  228:             signals = tdata.get("signals", {})
  229: 
  230:             for sig_name in SIGNAL_NAMES:
  231:                 vote = signals.get(sig_name, "HOLD")
  232:                 if vote == "HOLD":
  233:                     continue
  234:                 result_val = _vote_correct(vote, change_pct)
  235:                 if result_val is None:
  236:                     continue  # neutral outcome — don't count
  237:                 stats[sig_name]["total"] += 1
  238:                 if vote == "BUY":
  239:                     stats[sig_name]["total_buy"] += 1
  240:                     if result_val:
  241:                         stats[sig_name]["correct_buy"] += 1
  242:                 else:
  243:                     stats[sig_name]["total_sell"] += 1
  244:                     if result_val:
  245:                         stats[sig_name]["correct_sell"] += 1
  246:                 if result_val:
  247:                     stats[sig_name]["correct"] += 1

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=3496; $end=3544; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 637ms:
 3496:     # Utility boost: scale accuracy weight by return-based utility score.
 3497:     # Utility boost and best-horizon overlay.
 3498:     # H3: Skip when primary load failed to preserve fail-closed gate.
 3499:     if not _accuracy_failed:
 3500:         try:
 3501:             from portfolio.accuracy_stats import signal_utility
 3502:             # BUG-135: Use acc_horizon (not hardcoded "1d") so utility boost
 3503:             # reflects the actual prediction horizon's return profile.
 3504:             utility_data = signal_utility(acc_horizon)
 3505:             for sig_name in list(accuracy_data.keys()):
 3506:                 u = utility_data.get(sig_name, {})
 3507:                 u_score = u.get("avg_return", 0.0)
 3508:                 samples = u.get("samples", 0)
 3509:                 if samples >= 30 and u_score > 0:
 3510:                     boost = min(1.0 + u_score, 1.5)
 3511:                     if sig_name in accuracy_data:
 3512:                         # BUG-136: Build a new dict instead of mutating in-place.
 3513:                         # The accuracy_data may be a reference to cached alltime data.
 3514:                         boosted_acc = min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)
 3515:                         accuracy_data[sig_name] = {
 3516:                             **accuracy_data[sig_name],
 3517:                             "accuracy": boosted_acc,
 3518:                         }
 3519:         except Exception:
 3520:             logger.debug("Utility weighting unavailable", exc_info=True)
 3521: 
 3522:     if ticker:
 3523:         _record_phase(ticker, "utility_overlay", _phase_start)
 3524:         _phase_start = time.monotonic()
 3525: 
 3526:     # Multi-horizon: optionally use each signal's best horizon accuracy.
 3527:     # H3: Skip when primary load failed to preserve fail-closed gate.
 3528:     sig_cfg = (config or {}).get("signals", {})
 3529:     if not _accuracy_failed and sig_cfg.get("use_best_horizon", False):
 3530:         try:
 3531:             from portfolio.accuracy_stats import signal_best_horizon_accuracy
 3532:             best_hz = signal_best_horizon_accuracy(min_samples=50)
 3533:             for sig_name, bh_data in best_hz.items():
 3534:                 if bh_data.get("total", 0) >= 30:
 3535:                     # Only override if best-horizon accuracy is meaningfully better
 3536:                     current = accuracy_data.get(sig_name, {}).get("accuracy", 0.5)
 3537:                     if bh_data["accuracy"] > current + 0.03:
 3538:                         accuracy_data[sig_name] = bh_data
 3539:         except Exception:
 3540:             logger.debug("Best-horizon accuracy unavailable", exc_info=True)
 3541:     accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
 3542:     max_signals = sig_cfg.get("max_active_signals")
 3543: 
 3544:     # Signal persistence filter: only let signals that maintained their vote

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "_PERSISTENCE_MIN_CYCLES|Cold-start|persistence filter" portfolio/signal_engine.py' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 509ms:
238:# Signal persistence filter (2026-04-20): require signals to maintain their
247:# Only signals with cycles >= _PERSISTENCE_MIN_CYCLES get their vote passed
249:_PERSISTENCE_MIN_CYCLES = 2        # require 2+ consecutive same-direction votes
269:    Cold-start: on the first cycle for a ticker (no prior state), all signals
284:                sig: {"vote": vote, "cycles": _PERSISTENCE_MIN_CYCLES if vote != "HOLD" else 0}
306:                if prev["cycles"] >= _PERSISTENCE_MIN_CYCLES:
3544:    # Signal persistence filter: only let signals that maintained their vote
3558:    # which derives from `votes` via persistence filter — already
3584:    # persistence filter may have reduced voters below the threshold.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'portfolio/signal_engine.py'; "'$start=236; $end=249; $i=0; Get-Content $path | ForEach-Object { $i++; if($i -ge $start -and $i -le $end){ '"'{0,5}: {1}' -f "'$i, $_ } }' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 673ms:
  236: _IC_DATA_TTL = 3600     # IC cache TTL (matches ic_computation.py)
  237: 
  238: # Signal persistence filter (2026-04-20): require signals to maintain their
  239: # vote for MIN_PERSISTENCE_CYCLES consecutive cycles before counting in
  240: # consensus. Eliminates documented "single-check MACD/RSI/volume improvements
  241: # are noise" pattern. Raw votes are still recorded for accuracy tracking —
  242: # only the consensus input is filtered.
  243: #
  244: # Design: in-memory dict tracks {ticker: {signal: {"vote": X, "cycles": N}}}.
  245: # When a signal's vote matches its previous non-HOLD vote, cycles increments.
  246: # When it flips direction or goes HOLD→non-HOLD for the first time, cycles=1.
  247: # Only signals with cycles >= _PERSISTENCE_MIN_CYCLES get their vote passed
  248: # to consensus; others are treated as HOLD for consensus purposes only.
  249: _PERSISTENCE_MIN_CYCLES = 2        # require 2+ consecutive same-direction votes

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "BUG-186" portfolio' in Q:\finance-analyzer\.worktrees\adv-signals-core
 succeeded in 545ms:
portfolio\accuracy_stats.py:936:            "correct": int(round(blended * total)),  # BUG-186

codex
## Verdicts

- [P1] `_confluence_score()` biases a BUY/SELL tie toward BUY when adding the volume bonus — `portfolio/signal_engine.py:2247`
  Verdict: PARTIAL
  Reason: `_confluence_score()` does use `buy_count >= sell_count` at line 2247, so a 50/50 split can get a BUY-only `+0.1`, but that score is only stored in `extra_info["_confluence_score"]` at line 3614 and never drives the emitted action or confidence.
  Adjustment: Real diagnostic bug, not a P1 trading-logic flaw.

- [P1] `_weighted_consensus()` can emit a trade on a 50/50 split because the winner only needs `>= 0.5` — `portfolio/signal_engine.py:2230-2233`
  Verdict: FALSE-POSITIVE
  Reason: with `total_weight = buy_weight + sell_weight` and `buy_conf = buy_weight / total_weight`, `sell_conf = sell_weight / total_weight` (lines 2225, 2228-2229), any branch satisfying `buy_conf > sell_conf` or `sell_conf > buy_conf` already implies the winner is strictly `> 0.5`; exact ties already return `HOLD`.

- [P2] `blend_accuracy_data()` can propagate NaN because it never validates `at_acc` / `rc_acc` — `portfolio/accuracy_stats.py:920-922`
  Verdict: PARTIAL
  Reason: the blend helper does use raw `rc_acc`/`at_acc` in `abs(...)` and `w * rc_acc + (1 - w) * at_acc` with no finite-value guard, so NaN can contaminate the in-memory blended row, but this function does not write cache and `_weighted_consensus()` later sanitizes poisoned `accuracy`/`total` pairs before gating (`portfolio/signal_engine.py:1822-1850`).
  Adjustment: Missing local sanitation is real; the cache-poisoning and gate-bypass explanation is wrong, and the severity is too high.

- [P2] Dynamic `MIN_VOTERS` still uses the stale pre-persistence voter count — `portfolio/signal_engine.py:2509-2510`
  Verdict: FALSE-POSITIVE
  Reason: the current code already prefers `_voters_post_filter` at lines 2509-2510, and `generate_signal()` sets that field from `consensus_votes` at lines 3565-3600 before `apply_confidence_penalties()` is called at line 3621.

- [P2] `outcome_tracker` inflates accuracy samples by recording zero-change outcomes — `portfolio/outcome_tracker.py:471-480`
  Verdict: FALSE-POSITIVE
  Reason: zero-change outcomes are written, but `accuracy_stats._vote_correct()` treats `abs(change_pct) < 0.05` as neutral at lines 182-183 and callers skip those outcomes instead of incrementing totals at lines 234-237.

- [P2] The "Now" quorum path is not asset-class aware because metals reuse stock `MIN_VOTERS` — `portfolio/signal_engine.py:3354-3359`
  Verdict: FALSE-POSITIVE
  Reason: Claude cited the wrong lines; the actual quorum branch is asset-class aware (`STOCK_SYMBOLS`, `METALS_SYMBOLS`, else crypto), and both current base constants are `3` anyway (`MIN_VOTERS_STOCK` at line 604, `MIN_VOTERS_CRYPTO` at line 603), so a separate `MIN_VOTERS_METALS = 3` would change nothing.

- [P1] `_apply_persistence_filter()` bypasses persistence on cold start — `portfolio/signal_engine.py:278-287`
  Verdict: PARTIAL
  Reason: first-cycle pass-through is real, but it is explicit in the function docstring/comments at lines 269-270 and implemented by `return votes` at line 287; changing only the seed value at line 284 would not stop it.
  Adjustment: The real issue is an intentional cold-start exception that contradicts the stricter contract described at lines 238-249, so Claude identified the effect but not the actual mechanism or fix.

- [P2] `_count_active_voters_at_gate()` ignores persistence filtering and overcounts voters — `portfolio/signal_engine.py:1422-1473`
  Verdict: FALSE-POSITIVE
  Reason: the helper itself is persistence-agnostic, but the live path passes `consensus_votes = _apply_persistence_filter(votes, ticker)` into `_weighted_consensus()` at lines 3547 and 3569, and `_compute_gate_relaxation()` is only called inside that post-filter path at line 2047.

- [P1] Blended `correct = int(round(blended * total))` is inconsistent with the source data — `portfolio/accuracy_stats.py:930-936`
  Verdict: FALSE-POSITIVE
  Reason: `blend_accuracy_data()` is intentionally constructing a synthetic blended row with synthetic `total = max(at_samples, rc_samples)` and synthetic `correct`, and Claude’s proposed `at.correct + rc.correct` is wrong because recent overlaps all-time; current consensus code reads `accuracy`, `total`, and directional fields, not `correct` (`portfolio/signal_engine.py:2086-2149`).

## New findings (you, not Claude)

- None.

## Summary
- Confirmed: 0
- Partial: 3
- False-positive: 6
- New from you: 0
tokens used
94,635
## Verdicts

- [P1] `_confluence_score()` biases a BUY/SELL tie toward BUY when adding the volume bonus — `portfolio/signal_engine.py:2247`
  Verdict: PARTIAL
  Reason: `_confluence_score()` does use `buy_count >= sell_count` at line 2247, so a 50/50 split can get a BUY-only `+0.1`, but that score is only stored in `extra_info["_confluence_score"]` at line 3614 and never drives the emitted action or confidence.
  Adjustment: Real diagnostic bug, not a P1 trading-logic flaw.

- [P1] `_weighted_consensus()` can emit a trade on a 50/50 split because the winner only needs `>= 0.5` — `portfolio/signal_engine.py:2230-2233`
  Verdict: FALSE-POSITIVE
  Reason: with `total_weight = buy_weight + sell_weight` and `buy_conf = buy_weight / total_weight`, `sell_conf = sell_weight / total_weight` (lines 2225, 2228-2229), any branch satisfying `buy_conf > sell_conf` or `sell_conf > buy_conf` already implies the winner is strictly `> 0.5`; exact ties already return `HOLD`.

- [P2] `blend_accuracy_data()` can propagate NaN because it never validates `at_acc` / `rc_acc` — `portfolio/accuracy_stats.py:920-922`
  Verdict: PARTIAL
  Reason: the blend helper does use raw `rc_acc`/`at_acc` in `abs(...)` and `w * rc_acc + (1 - w) * at_acc` with no finite-value guard, so NaN can contaminate the in-memory blended row, but this function does not write cache and `_weighted_consensus()` later sanitizes poisoned `accuracy`/`total` pairs before gating (`portfolio/signal_engine.py:1822-1850`).
  Adjustment: Missing local sanitation is real; the cache-poisoning and gate-bypass explanation is wrong, and the severity is too high.

- [P2] Dynamic `MIN_VOTERS` still uses the stale pre-persistence voter count — `portfolio/signal_engine.py:2509-2510`
  Verdict: FALSE-POSITIVE
  Reason: the current code already prefers `_voters_post_filter` at lines 2509-2510, and `generate_signal()` sets that field from `consensus_votes` at lines 3565-3600 before `apply_confidence_penalties()` is called at line 3621.

- [P2] `outcome_tracker` inflates accuracy samples by recording zero-change outcomes — `portfolio/outcome_tracker.py:471-480`
  Verdict: FALSE-POSITIVE
  Reason: zero-change outcomes are written, but `accuracy_stats._vote_correct()` treats `abs(change_pct) < 0.05` as neutral at lines 182-183 and callers skip those outcomes instead of incrementing totals at lines 234-237.

- [P2] The "Now" quorum path is not asset-class aware because metals reuse stock `MIN_VOTERS` — `portfolio/signal_engine.py:3354-3359`
  Verdict: FALSE-POSITIVE
  Reason: Claude cited the wrong lines; the actual quorum branch is asset-class aware (`STOCK_SYMBOLS`, `METALS_SYMBOLS`, else crypto), and both current base constants are `3` anyway (`MIN_VOTERS_STOCK` at line 604, `MIN_VOTERS_CRYPTO` at line 603), so a separate `MIN_VOTERS_METALS = 3` would change nothing.

- [P1] `_apply_persistence_filter()` bypasses persistence on cold start — `portfolio/signal_engine.py:278-287`
  Verdict: PARTIAL
  Reason: first-cycle pass-through is real, but it is explicit in the function docstring/comments at lines 269-270 and implemented by `return votes` at line 287; changing only the seed value at line 284 would not stop it.
  Adjustment: The real issue is an intentional cold-start exception that contradicts the stricter contract described at lines 238-249, so Claude identified the effect but not the actual mechanism or fix.

- [P2] `_count_active_voters_at_gate()` ignores persistence filtering and overcounts voters — `portfolio/signal_engine.py:1422-1473`
  Verdict: FALSE-POSITIVE
  Reason: the helper itself is persistence-agnostic, but the live path passes `consensus_votes = _apply_persistence_filter(votes, ticker)` into `_weighted_consensus()` at lines 3547 and 3569, and `_compute_gate_relaxation()` is only called inside that post-filter path at line 2047.

- [P1] Blended `correct = int(round(blended * total))` is inconsistent with the source data — `portfolio/accuracy_stats.py:930-936`
  Verdict: FALSE-POSITIVE
  Reason: `blend_accuracy_data()` is intentionally constructing a synthetic blended row with synthetic `total = max(at_samples, rc_samples)` and synthetic `correct`, and Claude’s proposed `at.correct + rc.correct` is wrong because recent overlaps all-time; current consensus code reads `accuracy`, `total`, and directional fields, not `correct` (`portfolio/signal_engine.py:2086-2149`).

## New findings (you, not Claude)

- None.

## Summary
- Confirmed: 0
- Partial: 3
- False-positive: 6
- New from you: 0
