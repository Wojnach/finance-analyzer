# Claude critique of codex findings — data-external

## Verdicts

- [P0] Missing internal portfolio modules in worktree — portfolio/data_collector.py:10-14
  Verdict: FALSE-POSITIVE
  Reason: All referenced modules (shared_state, api_utils, file_utils, price_source, circuit_breaker, http_retry, indicators) exist in the main repo. The codex error was about a worktree branch being incomplete, not the main codebase. Main branch has all imports satisfied.

- [P1] Avoid waiting on timed-out timeframe workers — portfolio/data_collector.py:333-338
  Verdict: CONFIRMED
  Reason: ThreadPoolExecutor.cancel() does not stop running futures. Once a task is executing, shutdown(wait=True) in the `with` block waits indefinitely. BUG-179 comment acknowledges the intent but the implementation doesn't prevent hangs—only logs them. The timeout triggers but doesn't actually unblock the collector.

- [P1] Persist FTD age in calendar terms, not window offsets — portfolio/market_health.py:446-450
  Verdict: CONFIRMED
  Reason: Line 450 persists ftd_day_offset (a window-relative index), but line 244 uses `n - 1 - ftd_day_offset` where n=60 on every fresh call. When the window is stable in size, elapsed_days calculation freezes once ftd_day_offset entered the previous window. The state machine never ages from ftd_confirmed to confirmed_uptrend.

- [P2] Reconstruct FTD state from history when no snapshot exists — portfolio/market_health.py:206-210
  Verdict: CONFIRMED
  Reason: When prev_state is None (cold start), state defaults to STATE_CORRECTING and only the final bar is processed. A sustained 30-day uptrend with no correction will not be recognized as STATE_RALLY_ATTEMPT or uptrend—it stays correcting indefinitely. The breadth_score is depressed until the next correction/rally cycle naturally occurs.

## New findings (mine)

- [P1] Incomplete timeframe fetch silently degrades LLM context — portfolio/signal_engine.py:2300
  When collect_timeframes() returns early due to timeout (line 333-338 in data_collector.py), incomplete results are passed to generate_signal(). The _build_llm_context() function iterates over the partial timeframes dict, silently omitting missing entries. Ministral/Qwen3 signals then see degraded context (missing 12h/2d/7d signals) without warning. While LLMs can tolerate missing context, silent quality degradation during timeout failures violates fail-loudly principle.

- [P1] Wrong timeframe selected when "Now" fetch times out — portfolio/main.py:486
  If the "Now" (15m) timeframe hits timeout and doesn't appear in raw_results, collect_timeframes() returns an incomplete list sorted by original order. Line 486 uses `tfs[0][1]` assuming it's the "Now" entry, but it may be "12h" or later if "Now" timed out. The fallback at line 492 saves this case, but the silent mismatch between assumption and actual data is fragile and could cause wrong signal inputs if the fallback path is skipped or doesn't catch all cases.

## Summary
- Confirmed: 3
- Partial: 0
- False-positive: 1
- New: 2
