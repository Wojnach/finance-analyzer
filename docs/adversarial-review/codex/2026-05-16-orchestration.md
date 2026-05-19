## [P1] Timed-out ticker workers keep running into the next cycle
**File:** portfolio/main.py:661  
**Bug:** `pool.shutdown(wait=False, cancel_futures=True)` abandons already-running ticker threads after a timeout. Python cannot cancel running threads, so they continue mutating shared caches/logs while the loop proceeds.  
**Why it matters:** A stuck `_process_ticker` can finish during the next cycle and update shared LLM/cache state using stale prices/signals, creating cross-cycle races and corrupting downstream decisions.  
**Fix:** Treat ticker timeout as a hard cycle failure: isolate worker side effects, use subprocesses/process pool for killability, or wait for running futures to finish before starting the next cycle.

## [P1] Layer 2 bypasses the central Claude kill switch
**File:** portfolio/agent_invocation.py:1056  
**Bug:** This spawns `subprocess.Popen` directly instead of routing through `claude_gate.invoke_claude`, bypassing the module-level `CLAUDE_ENABLED` kill switch, central rate limiting, and central invocation accounting.  
**Why it matters:** If `CLAUDE_ENABLED=False` is set during an outage or runaway-cost event, the main trading Layer 2 path can still spawn Claude and produce trade guidance.  
**Fix:** Route this path through `claude_gate` or make `agent_invocation` enforce the same kill switch, locking, accounting, and tree-kill contract.

## [P1] Disabled Layer 2 fails open on config read errors
**File:** portfolio/claude_gate.py:156  
**Bug:** `_load_config_layer2_enabled()` uses raw `open/json.load` and returns `True` on any exception.  
**Why it matters:** If `config.json` is temporarily unreadable, corrupt, or mid-write while `layer2.enabled=false`, Claude invocations are allowed anyway. That defeats the operational disable switch.  
**Fix:** Use `load_json(CONFIG_FILE, default={})`; fail closed for unreadable config when checking a disable/kill setting.

## [P1] Second `bought` command silently overwrites an open ISKBETS position
**File:** portfolio/iskbets.py:761  
**Bug:** `_handle_bought()` loads state and unconditionally replaces `state["active_position"]` without checking whether one already exists.  
**Why it matters:** A duplicate or mistaken Telegram command destroys the previous open position record without adding it to `trade_history`, losing entry price, size, stop, and P&L tracking for real money.  
**Fix:** Reject `bought` when `active_position` is already set unless an explicit replace/force command is used.

## [P2] Suppressed trigger consumes the consensus baseline
**File:** portfolio/trigger.py:318  
**Bug:** Low-confidence consensus triggers update `triggered_consensus[ticker] = action` before returning, so the later transition to a high-confidence BUY/SELL in the same direction is no longer “new.”  
**Why it matters:** Example: `MU` enters BUY at 30% and is suppressed by `consensus_min_pct=60`; later it strengthens to BUY 80%, but no Layer 2 trigger fires because the baseline is already BUY.  
**Fix:** Do not advance the consensus baseline for budget-suppressed candidates, or track suppressed state separately and trigger when it crosses the configured floor.

## [P2] Startup grace can skip post-trade reassessment
**File:** portfolio/trigger.py:228  
**Bug:** The startup grace block returns before `_check_recent_trade(state)` runs.  
**Why it matters:** If Layer 2 made a trade and the loop restarted before the next trigger check, the first cycle after restart only updates baselines and skips the required post-trade reassessment.  
**Fix:** Run recent-trade detection before startup grace, or exempt post-trade reassessment from grace suppression.

## [P2] Market state ignores Swedish market holidays
**File:** portfolio/market_timing.py:336  
**Bug:** `get_market_state()` checks only `is_us_market_holiday(now)` and never uses `is_swedish_market_holiday()`.  
**Why it matters:** On a Swedish-only closure such as Midsummer Eve, the loop can treat the market as open and include Avanza/Nordic instruments even though they cannot trade.  
**Fix:** Split US and Swedish instrument availability, and exclude Avanza/Swedish symbols on Swedish holidays.

## [P2] Loop uses stale market state after sleeping across boundaries
**File:** portfolio/main.py:1350  
**Bug:** `get_market_state()` is called before `_sleep_for_next_cycle()`, then the pre-sleep `active_symbols` are used after the sleep.  
**Why it matters:** A cycle that starts sleeping at 06:59 UTC can wake after EU open but still run with the pre-open symbol set, missing the first open-cycle scan.  
**Fix:** Recompute market state and active symbols immediately after sleeping, just before `run()`.

## [P2] Post-cycle tasks use stale config forever
**File:** portfolio/main.py:1314  
**Bug:** `config = _load_config()` is loaded once before the infinite loop and then reused for `_run_post_cycle(config, ...)` on every cycle.  
**Why it matters:** Runtime changes to notification routing, budgets, digest settings, or safety toggles are ignored by post-cycle work until process restart.  
**Fix:** Reload config inside each loop iteration before `_run_post_cycle()`.

## [P2] Failed Claude watch calls suppress retries
**File:** portfolio/analyze.py:814  
**Bug:** `last_claude_time` is updated after timeout, missing binary, nonzero exit, or parse failure.  
**Why it matters:** During `--watch`, a transient Claude failure prevents another Claude analysis for 15 minutes while leveraged positions remain open.  
**Fix:** Update `last_claude_time` only after a successful, parsed Claude response; retry failures on the next interval or with a short backoff.

## [P2] Watch parser turns “do not SELL” into SELL
**File:** portfolio/analyze.py:548  
**Bug:** `_parse_watch_response()` sets action to SELL if `"SELL"` appears anywhere in the ticker line.  
**Why it matters:** A response like `BTC: HOLD - do not sell yet` is parsed as SELL, generating a false exit recommendation.  
**Fix:** Parse the action token immediately after `TICKER:` with an anchored regex for `HOLD|SELL`.

## [P2] ISKBETS `sold` without price fabricates P&L from highest price
**File:** portfolio/iskbets.py:807  
**Bug:** `_handle_sold()` defaults `current_price` to `pos["highest_price"]` when the user does not provide a sale price.  
**Why it matters:** If a position hit a trailing stop at 95 after reaching 110 from a 100 entry, replying `sold` records +10% instead of -5%.  
**Fix:** Require an explicit sold price, or fetch a live price and mark it as estimated.

## [P2] ISKBETS accepts zero/negative trade inputs
**File:** portfolio/iskbets.py:730  
**Bug:** `_handle_bought()` parses `price_usd` and `amount_sek` but never validates they are positive finite values.  
**Why it matters:** `bought BTC 0 100000` crashes on division by zero; `bought BTC 65000 -100000` creates negative shares and inverted P&L.  
**Fix:** Reject non-finite, zero, or negative price/amount before computing shares.

## [P2] Emergency alerts consume normal alert budget
**File:** portfolio/alert_budget.py:41  
**Bug:** Emergency alerts append to `_sent_timestamps` even though the doc says they bypass the budget.  
**Why it matters:** Three emergency stop/crash alerts can exhaust `max_per_hour=3`, causing the next normal or important alert to be buffered for the rest of the hour.  
**Fix:** Do not record emergency sends in the normal budget bucket, or track them separately.

## [P2] Escalation gate leaks hung runner threads
**File:** portfolio/escalation_gate.py:212  
**Bug:** On runner timeout, `shutdown(wait=False, cancel_futures=True)` cannot kill the already-running thread.  
**Why it matters:** Repeated Ministral hangs leave one live thread per trigger, eventually consuming resources and allowing old classifier calls to keep running after the decision already failed open.  
**Fix:** Use a subprocess with a hard kill, or a shared long-lived worker with enforceable request timeouts.

## [P2] String `"false"` is parsed as escalate=true
**File:** portfolio/escalation_gate.py:136  
**Bug:** `bool(obj.get("escalate"))` treats any non-empty string as `True`.  
**Why it matters:** If the model returns `{"escalate": "false"}`, the gate escalates to Claude instead of downgrading to autonomous.  
**Fix:** Require a real boolean, or explicitly parse accepted string values `"true"` and `"false"`.

## [P2] Prophecy updates are lost under concurrent writers
**File:** portfolio/prophecy.py:210  
**Bug:** `evaluate_checkpoints()` performs load-modify-save on `prophecy.json` without any lock or compare-and-swap.  
**Why it matters:** If one process triggers a checkpoint while another adds or updates a belief, the later atomic write can overwrite the other update. Atomic writes prevent partial files, not lost updates.  
**Fix:** Serialize prophecy mutations with a file lock or use an append/event log with merge-on-read.

## [P2] Naive checkpoint deadlines never expire
**File:** portfolio/prophecy.py:232  
**Bug:** `datetime.fromisoformat(deadline)` can produce a naive datetime, and comparing it to aware UTC `now` raises `TypeError`, which is swallowed.  
**Why it matters:** A deadline like `"2026-05-16"` is ignored forever instead of expiring the checkpoint.  
**Fix:** Normalize parsed deadlines to UTC-aware datetimes; treat date-only deadlines explicitly.

## [P2] Tier portfolio snapshots drop unpriced holdings
**File:** portfolio/reporting.py:1064  
**Bug:** `_portfolio_snapshot()` only adds holding value when `ticker in prices_usd`; otherwise the holding is omitted from `total` and from `holdings_summary`.  
**Why it matters:** If a held stock lacks a fresh price during an off-hours or data-failure cycle, Tier 1/Tier 2 context can show a cash-only portfolio and understate exposure.  
**Fix:** Preserve holdings even without price and use last known/stale summary prices with a stale marker.

## [P2] Autonomous mode fabricates a healthy bold portfolio on load failure
**File:** portfolio/autonomous.py:837  
**Bug:** `_load_bold_state_safe()` catches every exception and returns a fake 500,000 SEK empty portfolio.  
**Why it matters:** A corrupt or locked bold state file is reported as no holdings and flat P&L, causing autonomous messages and reasoning to ignore real exposure.  
**Fix:** Surface the load failure in the decision log/message and avoid substituting a healthy default portfolio.

## SUMMARY P1=4 P2=16 P3=0