# Codex Review — 2-orchestration

## Summary

The snapshot contains multiple logic bugs in active orchestration paths, including misidentifying stock triggers, firing stale sustained triggers after restarts, and treating Swedish market holidays as normal trading days. These issues can change when Layer 2 runs and what ticker context it sees, so the patch should not be considered correct.

Full review comments:

- [P2] Recognize stock tickers in consensus and price-move reasons — Q:/fa-review/portfolio/agent_invocation.py:147-149
  When `check_triggers()` emits stock reasons like `"MSTR consensus BUY (...)"` or `"NVDA moved 2.1% up"`, this regex never matches because it only accepts stock symbols followed by `flipped|crossed|broke`. `_extract_ticker()` then falls back to `XAG-USD`, which means the trade-guard gate, decision-feedback injection, and multi-agent specialist launch all use silver as the primary ticker for common stock-only triggers.

- [P2] Reset persisted debounce state in the startup-grace path — Q:/fa-review/portfolio/trigger.py:183-190
  This restart path rewrites `last` and `triggered_consensus` but leaves `sustained_counts`, `sustained_sentiment`, and `stable_sentiment` from the previous process intact. Because those structures persist `_mono_start`, the second cycle after a restart can immediately emit a `... sentiment bullish->bearish (sustained)` trigger from stale state even though the new process has only observed one sample, which defeats the stated goal of suppressing restart noise.

- [P2] Treat Swedish-only holidays as closed in `get_market_state()` — Q:/fa-review/portfolio/market_timing.py:334-336
  The module defines `is_swedish_market_holiday()` but this branch only checks US holidays. On Swedish-only closures such as 2026-05-14 (Ascension Day), `get_market_state()` returns `"open"` and includes all stock symbols, so `portfolio.main.run()` keeps stock/warrant processing and Layer 2 activity alive even though Avanza/Nasdaq Stockholm are closed.

- [P2] Use a Tier-3 timeout when falling back to `pf-agent.bat` — Q:/fa-review/portfolio/agent_invocation.py:724-731
  In the `claude`-missing fallback we switch to `pf-agent.bat` and explicitly log that it is "always Tier 3", but the later `_agent_timeout = timeout` still keeps the originally requested tier's budget. In environments where `claude` is not on `PATH`, a T1/T2 invocation will therefore kill the fallback full-review agent after 120s/600s instead of 900s, which makes the recovery path unreliable exactly when it is needed.
The snapshot contains multiple logic bugs in active orchestration paths, including misidentifying stock triggers, firing stale sustained triggers after restarts, and treating Swedish market holidays as normal trading days. These issues can change when Layer 2 runs and what ticker context it sees, so the patch should not be considered correct.

## Full review comments

- [P2] Recognize stock tickers in consensus and price-move reasons — Q:/fa-review/portfolio/agent_invocation.py:147-149
  When `check_triggers()` emits stock reasons like `"MSTR consensus BUY (...)"` or `"NVDA moved 2.1% up"`, this regex never matches because it only accepts stock symbols followed by `flipped|crossed|broke`. `_extract_ticker()` then falls back to `XAG-USD`, which means the trade-guard gate, decision-feedback injection, and multi-agent specialist launch all use silver as the primary ticker for common stock-only triggers.

- [P2] Reset persisted debounce state in the startup-grace path — Q:/fa-review/portfolio/trigger.py:183-190
  This restart path rewrites `last` and `triggered_consensus` but leaves `sustained_counts`, `sustained_sentiment`, and `stable_sentiment` from the previous process intact. Because those structures persist `_mono_start`, the second cycle after a restart can immediately emit a `... sentiment bullish->bearish (sustained)` trigger from stale state even though the new process has only observed one sample, which defeats the stated goal of suppressing restart noise.

- [P2] Treat Swedish-only holidays as closed in `get_market_state()` — Q:/fa-review/portfolio/market_timing.py:334-336
  The module defines `is_swedish_market_holiday()` but this branch only checks US holidays. On Swedish-only closures such as 2026-05-14 (Ascension Day), `get_market_state()` returns `"open"` and includes all stock symbols, so `portfolio.main.run()` keeps stock/warrant processing and Layer 2 activity alive even though Avanza/Nasdaq Stockholm are closed.

- [P2] Use a Tier-3 timeout when falling back to `pf-agent.bat` — Q:/fa-review/portfolio/agent_invocation.py:724-731
  In the `claude`-missing fallback we switch to `pf-agent.bat` and explicitly log that it is "always Tier 3", but the later `_agent_timeout = timeout` still keeps the originally requested tier's budget. In environments where `claude` is not on `PATH`, a T1/T2 invocation will therefore kill the fallback full-review agent after 120s/600s instead of 900s, which makes the recovery path unreliable exactly when it is needed.
