# Codex Adversarial Review — orchestration subsystem

Reviewer: codex / gpt-5.4 (xhigh reasoning)
Date: 2026-05-09
Branch: review/2026-05-08-orchestration (off empty-baseline)

Format: `[Pri] file.py:line — problem | FIX: repair`

---

[P0] portfolio/health.py:35 — `update_health()` stamps `last_invocation_ts` on every trigger, so `check_agent_silence()` treats skipped/off-hours/failed Layer 2 cycles as fresh invocations and can hide a real L2 outage from operators. | FIX: update `last_invocation_ts` only from the actual invocation/completion path, and keep trigger timestamps separate from invocation timestamps.
[P1] portfolio/agent_invocation.py:676 — If both portfolio drawdown checks throw, the 50% hard block disappears and Layer 2 still runs. | FIX: track whether any drawdown check succeeded and fail closed when none do.
[P1] portfolio/main.py:661 — When the ticker pool times out, `shutdown(wait=False)` leaves the timed-out worker threads alive, so they can keep mutating shared caches/state during the next cycle after this cycle already counted them as failed. | FIX: isolate ticker work in killable processes or block reuse of shared state until timed-out workers are definitively gone.
[P1] portfolio/agent_invocation.py:883 — On any transaction-baseline read error, both counts reset to `0`, so completion replays the entire transaction history into `record_trade()` and poisons overtrading guards. | FIX: use a sentinel for “baseline unavailable” and skip trade-recording bookkeeping for that cycle instead of assuming zero prior transactions.
[P1] portfolio/agent_invocation.py:240 — Untickered triggers default to `XAG-USD`, so post-trade/F&G invocations can point multi-agent focus, trade-guard blocking, and decision feedback at the wrong instrument. | FIX: return `None` when no ticker is present and make each downstream consumer handle the no-ticker case explicitly.
[P1] portfolio/agent_invocation.py:774 — Synthesis still proceeds after specialist failures and reads the usual report files, so stale `_specialist_*.md` output from a previous ticker can drive the current decision. | FIX: delete specialist report files before launch and only include reports that were freshly produced by successful specialists in the synthesis prompt.
[P1] portfolio/session_calendar.py:156 — `stock_us` session logic ignores NYSE holidays and reports the market as open on holiday weekdays. | FIX: gate the US-stock open check with `is_us_market_holiday()` before declaring the session open.
[P1] portfolio/session_calendar.py:184 — Warrant/`stock_se` session logic ignores Swedish market holidays and reports an open session on full-market closures. | FIX: gate the EU-session open check with `is_swedish_market_holiday()` before declaring the session open.
[P1] portfolio/trigger.py:239 — The ranging dampener still advances `triggered_consensus` to BUY/SELL on a suppressed low-confidence signal, so a later stronger signal never triggers Layer 2. | FIX: leave the baseline at HOLD when dampening suppresses the trigger and only advance it after a trigger-worthy consensus actually fires.
[P2] portfolio/main.py:849 — Every `False` from `invoke_agent()` is logged as `skipped_busy`, overwriting the real reason (drawdown block, gate skip, spawn failure, etc.) and breaking downstream monitoring/dedup logic. | FIX: have `invoke_agent()` return a structured outcome and log that exact status once instead of collapsing all failures into `skipped_busy`.
[P2] portfolio/multi_agent_layer2.py:207 — Specialist timeouts use `proc.kill()` on the direct Claude process only, which leaks child Node/MCP processes on Windows and lets timed-out specialists keep consuming RAM and handles. | FIX: kill the full process tree with the same taskkill/process-group logic used in `claude_gate.py`.
[P2] portfolio/autonomous.py:88 — Autonomous mode swallows every exception and returns control as if the fallback analysis ran, so main records an `autonomous` trigger even when no journal, decision log, or Telegram output was produced. | FIX: return an explicit success/failure result (or re-raise) and make main log a failure status when autonomous analysis did not complete.
[P3] portfolio/reflection.py:71 — SELL PnL uses all historical buys and never consumes sold shares, so partial exits are double-counted and the reflection win-rate/PnL statistics drift wrong. | FIX: track remaining lots or decrement share counts as sells are matched so each share is realized exactly once.
P0=1 P1=8 P2=3 P3=1
