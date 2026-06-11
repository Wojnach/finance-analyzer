# Improvement Audit — 2026-06-10

Multi-agent audit: 13 read-only finders (one per subsystem/dimension) + adversarial skeptic verification of every P0/P1 batch. 25 agents, 660 tool calls, ~23 min. **138 findings: 0 P0, 36 P1 (3 refuted, most downgraded to P2 on verification), 68 P2, 34 P3.**

Severity legend: P0 real-money loss / corruption / exploit · P1 wrong decisions / silent failure / broken automation · P2 maintainability+latent risk · P3 nit. `Effective` = skeptic-adjusted severity where verified.

## Executive summary — confirmed highest-impact items

- **[signal-modules]** _expiry_proximity votes BUY almost every cycle because Deribit nearest expiry is always 0-1 days (daily expiries) — `/mnt/q/finance-analyzer/portfolio/signals/crypto_macro.py:187`
- **[metals-real-money]** Playwright context thread-affinity conflict silently disables grid fisher (and stop-management calls) for entire trading days — `/mnt/q/finance-analyzer/portfolio/avanza_session.py:147`
- **[prophecy]** All prophecy 'critical' alerts are invisible to the startup check (level mismatch) — `/mnt/q/finance-analyzer/prophecy/alerts.py:22`
- **[prophecy]** Headless web-researching agent runs with repo-wide Bash(*)/Write(*) allow — prompt-injection to RCE on the trading box — `/mnt/q/finance-analyzer/scripts/prophecy-daily.bat:56`
- **[ops-automation]** Quoted redirection tokens make every PF-PendingPickups run fail silently — root cause of LLM-CRYPTOTRADER-72H 20-day overdue — `scripts/win/install-pending-pickups-task.ps1:28`

## Live incident — accuracy degradation root cause

### [P1 → P2 · reliability · conf 0.90] BGeometrics exchange-netflow feed silently dead since ~Apr 10 — crypto_macro and onchain_btc sub-indicators degraded for 2 months with no alert
`/mnt/q/finance-analyzer/portfolio/onchain_data.py:173`

data/onchain_cache.json (refreshed Jun 10) contains keys [mvrv, mvrv_zscore, sopr, nupl, realized_price, long_liquidations, short_liquidations] — NO 'netflow' key — so _fetch_exchange_netflow (onchain_data.py:173-179) has been returning nothing. Consequences: (1) data/exchange_netflow_history.jsonl has exactly ONE entry (ts 2026-04-10) — crypto_macro_data.get_exchange_netflow_trend (crypto_macro_data.py:340-346) permanently returns trend='insufficient_data' so crypto_macro's exchange_netflow sub-vote has been a silent HOLD for 2 months; (2) the active On-Chain BTC signal's netflow_signal interpretation (onchain_data.py:337-343) is never set, silently shrinking that composite. CLAUDE.md still advertises 'Exchange Netflow' as a component of signal 13. No critical_errors entry, no log alert beyond debug-level — exactly the silent-failure pattern the loop-contract work was built to catch. Not the cause of the June degradation (predates it by 7 weeks, and absence pushes toward HOLD not wrong votes), but it removes the one sub-indicator that could have voted SELL on distribution flows during the crash.

**Suggested fix:** Verify the BGeometrics /v1/exchange-netflow endpoint status (plan tier may have dropped it). If recoverable, fix the fetch; if not, remove the sub-indicator or replace the source. Add a staleness contract: alert when exchange_netflow_history.jsonl latest ts exceeds N days, mirroring the metals-catalog staleness gates.

**Skeptic verdict (real):** Confirmed: data/onchain_cache.json (ts 1781071318 ≈ Jun 10) has no 'netflow' key; data/exchange_netflow_history.jsonl has exactly 1 line (ts 1775837235 ≈ Apr 10); get_exchange_netflow_trend requires >=3 history entries (crypto_macro_data.py:339-346) so it returns 'insufficient_data' which maps to HOLD in _exchange_netflow_signal; netflow_signal interp at onchain_data.py:337-343 never set; zero 'netflow'/'bgeometrics' entries in critical_errors.jsonl and no health.py coverage, so genuinely silent for 2 months. Severity P2 not P1: failure direction is HOLD (abstention), not wrong votes; netflow is 1 of 5 crypto_macro subs (BTC only — ETH never had it, crypto_macro_data.py:454-459) and 1 of ~7 onchain interp fields, and crypto_macro is itself a 0.2x-weighted extreme-bias voter — so trading impact is marginal, but the missing staleness contract is a real reliability gap worth fixing.

### [P1 → P2 · design · conf 0.85] VERDICT: degradation is a genuine June 1-6 regime crash hitting structurally long-biased signals; 2026-06-06 commits exonerated; alert working as designed
`/mnt/q/finance-analyzer/portfolio/signals/crypto_macro.py:65`

Root-cause verdict across the three hypotheses. H1 (2026-06-06 auto-improve commits) REJECTED: the first CRITICAL accuracy_degradation alert is 2026-06-03T06:23 (data/critical_errors.jsonl), three days BEFORE the session; B5 (9304ad15) only formalized a signal that was already per-ticker force-HOLD for XAU/XAG, its only applicable tickers, and its data module portfolio/metals_cross_assets.py has no other live consumers (copper_gold_ratio, ovx_metals_spillover, breakeven_inflation_momentum import it but are in DISABLED_SIGNALS and not in _SHADOW_SAFE_SIGNALS, so they never compute); B6 verified as dead-code removal (shadow path at signal_engine.py:3813-3816 only executes for DISABLED_SIGNALS members; drift_regime_gate/amihud are active); the HORIZON_SIGNAL_WEIGHTS edit (a5319fb9) only changes a fallback used when accuracy_cache.json is unavailable (signal_engine.py:1614-1669); c81fbcde (B2/B4) doesn't touch the accuracy pipeline. H2 (stale upstream data) NOT causal for the June window: williams_vix_fix, drift_regime_gate, statistical_jump_regime are OHLCV-only; gold_btc_ratio_history.jsonl is fresh through Jun 10; arxiv_*.xml rate-limit files belong to the research pipeline, not signals (but see separate netflow finding). H3 (real regime shift) CONFIRMED: signal_log.db shows BTC -21% (avg 77.5K May 21 -> 61K Jun 6), ETH -27% (2131 -> 1565), XAG whipsawing +/-5-20% daily. Every degraded signal is structurally long-biased: crypto_macro voted BUY 478 vs SELL 89 since May 26 (ETH acc 25%, four full days at 0%); williams_vix_fix is a vol-spike bottom-picker (BUY 318 vs SELL 29, 23-30% acc); drift_regime_gate rode trend-long into the crash (BTC 27%); econ_calendar relief-BUY had the identical failure and was already regime-gated (3b8e8136) after its 06-02 collapse. Outcome grading verified sound (outcome_tracker.backfill_outcomes computes change_pct from snapshot base price vs Binance/Alpaca historical close; _vote_correct symmetric with 0.05% deadband). The remaining design defect: crypto_macro's sub-indicators are contrarian-long in crashes — _options_gravity votes BUY whenever price is >3% below max pain (which is always true after a leg down, line 65-66) and _put_call_sentiment votes BUY on fear (PCR>1.2, line 95-96), so the majority vote is perma-BUY for the entire duration of any sustained downtrend; ETH lacks even the netflow sub. This is an ACTIVE consensus voter for BTC/ETH.

**Suggested fix:** Apply the same trending-down regime gate that fixed econ_calendar (commit 3b8e8136) to crypto_macro: suppress or down-weight BUY when context regime == trending-down. Consider the same for the williams_vix_fix XAU/XAG per-ticker overrides. No revert of the 2026-06-06 commits is needed. Resolve the open accuracy_degradation rows in critical_errors.jsonl with a 'genuine regime shift, gates applied' resolution.

**Skeptic verdict (real):** Core verdict confirmed independently: accuracy_degradation CRITICALs actually start 2026-06-01T06:20 (not 06-03 as claimed — even earlier, strengthening exoneration of the 06-06 commits); 9304ad15 commit message matches B5/B6 exactly; shadow path gated on DISABLED_SIGNALS at signal_engine.py:3813; signal_log.db confirms BTC 77,461→61,195 (-21%) and ETH 2,130→1,597 May 21→Jun 6 with crypto_macro voting BUY 2491 vs SELL 218 (BTC) and 2680 vs 69 (ETH) since May 26; the BUY-on-fear logic at crypto_macro.py:65-66/95-96 is real and majority_vote treats HOLDs as abstentions (signal_utils.py:93-96) so one BUY sub flips the composite. Severity reduced to P2 because mitigations the finding omits bound the harm: crypto_macro's 93% BUY bias triggers the extreme-bias 0.2x vote weight (signal_engine.py:559-560, acknowledged at tickers.py:315), the 47% accuracy gate auto-force-HOLDs it as recent accuracy collapses, options_gravity only votes within 7 days of expiry (crypto_macro.py:61-63), and the williams_vix_fix part of the fix is stale — its XAU/XAG per-ticker overrides were already removed 2026-05-31 (signal_engine.py:727-730), it is shadow-only now.

### [P2 · doc-drift · conf 0.92] CLAUDE.md active-signal roster stale: Metals Cross-Asset and Crypto EVRP listed active but both disabled; Crypto Macro description doesn't match the module
`/mnt/q/finance-analyzer/CLAUDE.md:215`

CLAUDE.md lines 213-219 list '11. Metals Cross-Asset' and '15. Crypto EVRP' among the 21 active signals. metals_cross_asset was moved to DISABLED_SIGNALS on 2026-06-06 (commit 9304ad15, tickers.py:305) and crypto_evrp was re-disabled 2026-05-26 (tickers.py:94) — CLAUDE.md even self-contradicts by also listing Crypto EVRP under 'Pending validation' disabled signals at line 245. Line 214 describes Crypto Macro as 'DeFi TVL, staking yields, protocol revenue' but the actual module (portfolio/signals/crypto_macro.py) computes options gravity/put-call ratio/gold-BTC rotation/exchange netflow/expiry proximity — nothing matching the description. Additionally .claude/rules/signals.md states applicable signal counts crypto=29/stocks=25/metals=27 while CLAUDE.md line 257 says crypto=19/stocks=15/metals=17. Every Layer 2 trading session and headless subagent loads CLAUDE.md as ground truth, so a stale roster misinforms the component that makes trade decisions and any session investigating this exact degradation (e.g. trusting '68.1% recent' for drift_regime_gate at line 220 when it is currently 38-43%).

**Suggested fix:** Update the active list (19 active after the two disables, not 21), remove Crypto EVRP/Metals Cross-Asset from the active roster, rewrite the Crypto Macro description to match the module's actual sub-indicators, and reconcile the applicable-counts between CLAUDE.md and .claude/rules/signals.md.

### [P2 · design · conf 0.85] signal_accuracy conflates shadow votes with consensus votes — williams_vix_fix global degradation alert is ~70% driven by tickers where it never votes in consensus
`/mnt/q/finance-analyzer/portfolio/accuracy_stats.py:231`

signal_accuracy (accuracy_stats.py:230-247) counts every non-HOLD action in the per-ticker signals dict identically. But for globally-disabled signals in _SHADOW_SAFE_SIGNALS, the logged action is the SHADOW vote (real action recorded for outcome tracking while consensus sees force-HOLD; signal_engine.py:3816-3837). williams_vix_fix is active ONLY via XAU/XAG per-ticker overrides, yet since May 26 the DB shows shadow votes on BTC (83 graded, 25%), ETH (77, 26%), MSTR (48, 23%) vs active-scope XAG (57, 53%) and XAU (77, 30%). The 'williams_vix_fix 50.9%->31.2%' scope=signal alert firing daily since 06-08 is therefore dominated by instruments where the signal contributes nothing to trading. Worse, the per-ticker alert scope that DOES reflect traded impact (XAU::williams_vix_fix) can never fire because override tickers accumulate ~77 graded samples per 14d window, below MIN_SAMPLES_CURRENT=200 (accuracy_degradation.py:67-68). Net effect: operators get alerted on shadow noise and are blind to active-scope decay; weight-tuning sessions consume the same conflated numbers.

**Suggested fix:** Tag shadow votes at write time (the feedback_log_everything memory already prescribes tag-at-write/filter-at-read) — e.g. store shadow votes under a separate key or a per-row shadow flag — and have signal_accuracy/accuracy_degradation report active-scope and shadow-scope accuracy separately. Lower the per-ticker min-sample floor for signals whose only active scope is a per-ticker override.

### [P2 · design · conf 0.80] SE significance gate assumes independent samples but 60s-cadence snapshots of persistent votes are massively autocorrelated — alert confidence and weight-tuning stats are inflated
`/mnt/q/finance-analyzer/portfolio/accuracy_degradation.py:646`

_binomial_diff_se_pp (accuracy_degradation.py:599-614) and the 2-SE gate in _maybe_alert (line 646-647) treat each signal-log row as an independent Bernoulli trial. But votes persist across 60s cycles: econ_calendar emitted 200+ IDENTICAL votes per day for 3 straight weeks (all-BUY May 14-Jun 2, then all-SELL Jun 4-05 — verified from signal_log.db), and crypto_macro/williams_vix_fix show the same day-block structure. Effective independent sample count is roughly #days x #tickers (~14-28 per 14d window), not the 200-1100 'samples' the gate uses, so se_pp is understated by roughly sqrt(10-30)x and the 'roughly 1 in 1000' noise-floor claim in the comment at line 61-66 doesn't hold. Same inflation feeds accuracy_cache numbers used by auto-improve sessions: commit a5319fb9 boosted crypto_evrp to weight 1.5 citing '99.1% at 1d_recent (233 sam)' — verified to be one autocorrelated all-SELL burst during the crash (since May 26: BTC 0 BUY/266 SELL, ETH 0/244), i.e. a single directional bet repeated 233 times, not 233 independent wins.

**Suggested fix:** Compute SE on de-duplicated effective samples — e.g. collapse consecutive identical (ticker, signal, vote) runs into one trial, or aggregate to per-day-per-ticker accuracy before differencing. Apply the same effective-n discipline to any session that edits weights from accuracy_cache (a directional_skew>0.9 or runs-test flag next to each accuracy number would prevent the crypto_evrp 99.1% class of mistake).

### [P2 · quality · conf 0.70] 2026-06-06 fallback weights bake in crash-window burst statistics — crypto_evrp 1.5x from a 233-vote all-SELL streak, sentiment 1.3x while force-HOLD-disabled
`/mnt/q/finance-analyzer/portfolio/signal_engine.py:1551`

Commit a5319fb9 set HORIZON_SIGNAL_WEIGHTS['1d']['crypto_evrp']=1.5 citing '99.1% at 1d_recent (233 sam) — massive jump from 40.2%'. Verified from signal_log.db: those samples are a single uninterrupted SELL streak during the Jun 1-6 crash (BTC 0 BUY/266 SELL, ETH 0/244 since May 26) — the same small-sample/burst illusion that got crypto_evrp re-disabled on 2026-05-26 (tickers.py:94 comment: 'the 80.5% was 77 samples'), now repeated at higher weight. sentiment got 1.3 ('recovered to 62.8%') while sentiment remains in DISABLED_SIGNALS. These are fallback weights only (used when accuracy_cache.json is missing/empty, signal_engine.py:1614-1669), so live impact today is nil — but the failure window is exactly a cache-loss event during or after a crash, when the fallback would boost a contrarian-bear signal 1.5x into a recovery rally, and they also act as anchor documentation for future sessions ('massive jump' framing invites premature re-enable).

**Suggested fix:** Re-derive the 1d fallback weights excluding signals currently in DISABLED_SIGNALS (or cap disabled-signal fallback weights at 1.0), and annotate burst-driven accuracy with directional_skew (the metric added in 8df43907 exists precisely for this) before using it to set weights.

### [P3 · design · conf 0.75] econ_calendar event_free_window emits unconditional BUY for entire multi-day quiet-calendar stretches — regime gate only covers explicitly classified trending-down
`/mnt/q/finance-analyzer/portfolio/signals/econ_calendar.py:142`

_post_event_relief returns BUY whenever the next event is >72h away (econ_calendar.py:141-145), with no price-trend input at all. signal_log.db shows the consequence: ~100-215 identical BUY votes per day for 19 consecutive days (May 14-Jun 2), producing daily accuracies that are coin-flips of the market's direction (92% May 23, 0% May 26, 0% Jun 2) and the BTC-USD::econ_calendar 50.2%->27.3% CRITICAL alerts of Jun 3. The 3b8e8136 regime gate (line 268-269) now suppresses BUY when regime == 'trending-down', which fixed the June instance (econ_calendar dropped out of alerts after Jun 4 and the Jun 4-5 all-SELL days scored 85-100%). Residual risk: the gate depends on the regime classifier flagging 'trending-down' — in a slow grind-down classified as 'ranging' the perma-BUY behavior returns; a calm calendar is not bullish evidence in any regime, it is absence of evidence.

**Suggested fix:** Make event_free_window a HOLD (or require price confirmation, e.g. close > EMA) instead of an unconditional BUY; keep post-event relief (4-24h window) as the only BUY path since that is the actual documented anomaly.

## Ops automation (pickups, fix-agent, scheduled tasks)

### [P1 → P1 · bug · conf 0.85] Quoted redirection tokens make every PF-PendingPickups run fail silently — root cause of LLM-CRYPTOTRADER-72H 20-day overdue
`scripts/win/install-pending-pickups-task.ps1:28`

The task action passes ">>" and "2>&1" as individually quoted arguments through run-hidden.vbs, which re-quotes each argument (run-hidden.vbs:29-33). cmd.exe only honors redirection operators when UNQUOTED, so python receives literal argv ['>>', 'Q:\\...\\pending_pickups_task.log', '2>&1']. argparse in process_pending_pickups.py rejects them ('unrecognized arguments') and exits 2 before processing any pickup. Verified: schtasks shows the task installed and running daily (Last Run 2026-06-10 08:09:01, Last Result 0 because wscript detaches and exits 0), the configured 'Task To Run' contains the quoted '>>' tokens, data/pending_pickups_task.log has never been created, and data/pending_pickups.json still shows LLM-CRYPTOTRADER-72H status=pending with due_ts 2026-05-21 and empty history — 20 days overdue despite ~20 daily task runs. The failure is doubly invisible: the redirection that would have captured the argparse error is itself the broken part, and the vbs detach makes Task Scheduler report success.

**Suggested fix:** Wrap the python invocation in a .bat file (the pattern used by pf-outcome-check.bat etc.) that performs redirection inside the batch context, and point the task action at the .bat via run-hidden.vbs. Then force-run the pickup: .venv/Scripts/python.exe scripts/process_pending_pickups.py --force LLM-CRYPTOTRADER-72H. Also delete the dead $cmd/$args variables at lines 24-25 that suggest a working redirection that never existed.

**Skeptic verdict (real):** Reproduced empirically: ran the exact wscript→run-hidden.vbs→cmd /c chain with an argv-dump script; python received literal ['>>', '<logfile>', '2>&1'] and no log file was created. process_pending_pickups.py:170 uses strict argparse.parse_args (only --dry-run/--force) so every run exits 2 before processing; live task shows quoted '>>' in Task To Run, Last Run 2026-06-10 Result 0, data/pending_pickups_task.log never created, and the sole pickup (due 2026-05-21) is still pending with empty history. P1 stands because CLAUDE.md explicitly tells sessions to 'let the cron path run on its own schedule' — a permanently dead path.

### [P1 → P2 · reliability · conf 0.80] PF-FixAgentDispatcher scheduled task does not exist — fix-agent automation documented in CLAUDE.md is entirely dead
`scripts/win/install-fix-agent-task.ps1:7`

A full schtasks query lists no PF-FixAgentDispatcher (or any fix-agent-like) task, while CLAUDE.md documents it as live ('every 10 min ... spawns a Claude fix agent'). Corroborating evidence: data/critical_errors.jsonl contains exactly 2 fix_attempt_started rows ever (both 2026-05-28 19:14, 6ms apart — the dry-run code path at fix_agent_dispatcher.py:412-414 which writes 'started' rows then continues without 'completed' rows), and zero fix_attempt_skipped/completed/fix_agent_failed rows; data/fix_agent_state.json has empty by_category since May 28. Meanwhile 16 unresolved accuracy_degradation criticals accumulated Jun 6-10 with no dispatch. The designed disable mechanism (data/fix_agent.disabled kill switch, fix_agent_dispatcher.py:41,256-257) is NOT present, so this is not a deliberate, discoverable disable — the task was either never installed in production or unregistered out-of-band (possibly during the 2026-06-06 Claude token freeze). Any session or runbook trusting CLAUDE.md's description of auto-spawn behavior is operating on false assumptions.

**Suggested fix:** Decide intent: if the freeze should suppress fix agents, touch data/fix_agent.disabled (the documented kill switch) AND install the task so state is self-describing; update CLAUDE.md to note current status. If the task was simply never installed, run install-fix-agent-task.ps1 as admin. Add a check to scripts/check_critical_errors.py or the loop-health watchdog that alerts when unresolved criticals exist but no fix_attempt_* rows appear within N hours.

**Skeptic verdict (real):** Confirmed: full schtasks query lists ~50 PF-* tasks (Disabled ones included) with no fix-agent task; data/fix_agent.disabled absent; fix_agent_state.json by_category empty; only 2 fix_attempt_started rows ever (2026-05-28 dry-run path, fix_agent_dispatcher.py:412-414); 16 unresolved accuracy_degradation criticals Jun 6-9 undispatched. Downgraded to P2: dispatcher spawns via portfolio/claude_gate.invoke_claude (fix_agent_dispatcher.py:420) and claude_gate.py:62 has CLAUDE_ENABLED=False since the 2026-06-06 token freeze, so it would currently no-op even if installed — but the absence predates the freeze, isn't the documented kill-switch mechanism, and the freeze re-enable recipe doesn't cover it, so the doc/reality mismatch is real.

### [P1 → P2 · design · conf 0.75] run-hidden.vbs detached launch makes Task Scheduler health signals meaningless for every PF-* task
`scripts/win/run-hidden.vbs:35`

run-hidden.vbs launches the child with WScript.Shell.Run(cmd, 0, False) — detached, wscript exits immediately with 0. Consequences for ALL tasks using this shim: (1) 'Last Result' is always 0 regardless of child outcome — confirmed live by PF-PendingPickups showing Last Result 0 while its python child exits 2 every day; (2) -ExecutionTimeLimit settings (install-pending-pickups-task.ps1:39 '15 minutes', install-fix-agent-task.ps1:38 '20 minutes ... agent timeout is 15 min + buffer') only bound the milliseconds-long wscript process, not the real child — the comments claiming runtime caps are dead; (3) -MultipleInstances IgnoreNew (install-fix-agent-task.ps1:39, install-log-rotate-task.ps1:39) cannot prevent overlap because the task instance 'completes' instantly, so a long-running child coexists with the next trigger's child. The process_pending_pickups exit-code contract ('Exit code 1 if a handler returned verdict=error so the cron logs surface it', process_pending_pickups.py:21-22) is unfulfillable under this launcher.

**Suggested fix:** For non-loop tasks (pickups, log-rotate, dispatcher) drop the detach: either have run-hidden.vbs accept a wait flag (Run cmd, 0, True) and propagate WScript.Quit with the child exit code, or schedule python directly with a hidden-window principal. Reserve fire-and-forget detach for the long-lived loop tasks only.

**Skeptic verdict (real):** Code confirmed (run-hidden.vbs:35 Run cmd,0,False) and the detach is intentional and documented (vbs:18-20: 'Task Scheduler treats that as success'), so partly by-design for long-lived loops; but the cited consequences for one-shot tasks are real and verified live: PF-PendingPickups Last Result 0 while its child exits 2 daily, ExecutionTimeLimit comments (install-fix-agent-task.ps1:31, install-pending-pickups-task.ps1:39) and MultipleInstances IgnoreNew (install-fix-agent-task.ps1:39, install-log-rotate-task.ps1:39) are moot since the task instance completes in milliseconds, and the exit-code contracts in process_pending_pickups.py:21-22 and fix_agent_dispatcher.py's docstring are unfulfillable. P2: observability/design gap, not itself a functional failure.

### [P2 · bug · conf 0.85] Telegram notification path imports nonexistent portfolio.config — pickup verdict alerts can never send, permanently and silently
`scripts/process_pending_pickups.py:85`

_send_telegram does 'from portfolio import config as cfg_mod' (line 85). No portfolio/config.py, portfolio/config/ package, or config attribute in portfolio/__init__.py exists (verified: only config_validator.py, grid_fisher_config.py, logging_config.py; the real loader is portfolio.api_utils.load_config, see portfolio/main.py:126). The ImportError is swallowed by the bare 'except Exception: pass' (lines 90-92), so every pickup verdict's Telegram alert silently no-ops — the design's primary human-notification channel for unattended verdicts has never worked. The '# type: ignore[attr-defined]' comment shows the type checker flagged this and was suppressed instead of fixed. (Same broken import pattern exists in portfolio/mstr_loop/telegram_report.py:141, outside this audit's scope.)

**Suggested fix:** Replace with 'from portfolio.api_utils import load_config' and pass load_config() to send_telegram(msg, config). Narrow the except clause to log the exception (logger.warning) instead of pass, so a future regression is at least visible in the task log once logging works.

### [P2 · reliability · conf 0.85] A single handler error permanently kills a pickup: status='error' is never retried and is invisible to the session-start bottle
`scripts/process_pending_pickups.py:211`

When a handler returns verdict='error', the pickup gets status='error' (line 211). On every subsequent run the dispatcher skips it ('if status != "pending": continue', line 189-190), so even a transient failure (e.g. data file briefly missing, outcome backfill behind) permanently removes the pickup from automation with no retry and no escalation — exit code 1 (line 218) is swallowed by the detached vbs launcher. Worse, scripts/session_start_bottle.py only surfaces status=='pending' (line 87) and status=='completed' (line 101) pickups, so an errored pickup also vanishes from the session-start bottle: the exact 'work scheduled for a future session' the system was built to never forget becomes silently forgotten.

**Suggested fix:** Either keep status='pending' on error and record the failure in history with a retry counter (give up after N attempts), or add an explicit error branch in session_start_bottle.py that surfaces '[ERRORED]' pickups verbatim, mirroring the OVERDUE treatment.

### [P2 · design · conf 0.80] accuracy_degradation dedup keys on exact alert-set membership — boundary churn produced 16 duplicate unresolved critical rows in 4 days
`portfolio/loop_contract.py:2032`

violation_identity_payload (lines 2032-2040) keys accuracy_degradation identity on the sorted (scope::key) alert set, deliberately ignoring drifting percentages. But the SET itself churns: signals near the '>15pp drop AND <50% absolute' threshold enter and leave hourly (journal shows '10 signal(s)' → '9' → '8' → '7' → '8' ... Jun 6-9), and every distinct membership combination is a new identity hash, so the dedup at loop_contract.py:1548 never matches. Result: 16 unresolved accuracy_degradation rows since Jun 6, each requiring its own resolves_ts resolution line (the Jun 5 cleanup wrote 16 near-identical resolution rows), and scripts/check_critical_errors.py prints every row individually at session start (line 177-178) — classic alert fatigue that trains humans and agents to bulk-acknowledge without reading.

**Suggested fix:** Add a per-invariant journal-row cooldown for accuracy_degradation (e.g. at most one new unresolved row per 24h while ANY prior row for the invariant is unresolved), or use Jaccard-similarity instead of exact set equality for identity. Independently, make check_critical_errors.py group unresolved entries by category with a count and latest message, and support a category-level resolves_category field so one resolution line clears a flood.

### [P2 · bug · conf 0.70] Dispatcher persists cooldown state only after ALL categories finish — overlapping runs can spawn duplicate concurrent Opus fix agents
`scripts/fix_agent_dispatcher.py:464`

_save_state(state) is called once at run() end (line 464), after sequentially invoking up to N categories at 900s timeout each (line 429-436). A dispatcher run with 2+ categories takes >15 minutes, exceeding the 10-minute trigger interval. Because run-hidden.vbs detaches the child, MultipleInstances=IgnoreNew does not prevent a second dispatcher process from starting; it loads the state file before the first run has saved (no cooldown recorded yet) and check_gates (line 269-272) allows a duplicate spawn for the same category. Result: two concurrent Opus agents with Edit access working the same error — wasted quota and potential conflicting edits to the same files. A crash or power loss mid-run likewise discards all cooldown/backoff bumps accumulated in memory, so failed attempts don't count toward the 30m→2h→12h backoff.

**Suggested fix:** Persist state immediately after each update_state_after_attempt call (move _save_state inside the loop), and add a coarse run-lock (e.g. O_EXCL lockfile with PID + stale-after timestamp in data/) so overlapping dispatcher processes exit early.

### [P3 · doc-drift · conf 0.90] Docstring claims 'recurring' pickups are flipped back to pending after completion — not implemented; recurring pickups never run at all
`scripts/process_pending_pickups.py:12`

Lines 12-14 state: 'Pickups created with status="recurring" are flipped back to status="pending" immediately after completion'. No such code exists anywhere in main(): the status filter at line 189 ('if status != "pending": continue') means a status='recurring' pickup is never dispatched in the first place, and the completion path (line 211) only writes 'completed'/'error'. Anyone creating a recurring pickup per the docstring gets a permanently inert entry with zero feedback.

**Suggested fix:** Either implement it (accept status in {'pending','recurring'} at dispatch, and after a non-error run set status back to 'recurring' with a recomputed due_ts) or delete the docstring claim.

### [P3 · quality · conf 0.85] --force with a typo'd or unknown pickup ID prints '[pickup] no due pickups' and exits 0
`scripts/process_pending_pickups.py:220`

With --force <ID>, non-matching pickups are skipped (lines 184-186) and if no pickup matches, any_due stays False, the script prints the routine 'no due pickups' message (line 221) and exits 0. An operator force-running a misspelled ID — which CLAUDE.md and the bottle hook explicitly instruct humans to do — gets a success-looking no-op instead of an error. Given this is the documented manual recovery path for the currently broken cron path, a silent no-op here directly extends incidents like LLM-CRYPTOTRADER-72H.

**Suggested fix:** After the loop, if args.force and no pickup matched, print 'pickup id not found: <id> (known: ...)' and return a non-zero exit code.

### [P3 · quality · conf 0.85] Dead $cmd/$args variables (with $args being a PowerShell automatic variable) mask the broken action line
`scripts/win/install-pending-pickups-task.ps1:24`

Lines 24-25 build $cmd='cmd.exe' and $args='/c "python" -u "script" >> "log" 2>&1' — a form whose redirection WOULD have worked if passed as a single argument — but both variables are unused; the actual action at lines 27-29 was rewritten into the per-token quoted form that broke redirection (see P1 finding). Assigning to $args also shadows PowerShell's automatic $args variable, a lint-level hazard. The leftover code makes the script read as if log capture works, which delayed diagnosis of the 20-day silent failure. -RunLevel Highest (line 47) is also unnecessary privilege for a script that only reads/writes repo data files.

**Suggested fix:** Delete the dead variables when fixing the action line, rename any retained variable away from $args, and drop -RunLevel Highest.

### [P3 · bug · conf 0.70] _auto_resolve_stale_categories keys fix-entries by original category, but resolution lines use category='resolution' — auto-resolve is dead code in practice
`scripts/check_critical_errors.py:89`

The stale-category auto-resolve requires 'at least one resolution or info entry for that category' (lines 89-92: latest_fix_by_cat keyed on e['category']). But the documented resolution format (CLAUDE.md) and all observed resolution rows in data/critical_errors.jsonl carry category='resolution' with resolves_ts pointing at the original — they populate latest_fix_by_cat['resolution'], never the failing category. Info-level rows with the original category essentially don't exist (emitters write critical-level only; dispatcher meta-rows use their own categories). So the fix_ts condition at line 97 almost never holds and stale categories linger the full 7-day window, adding noise the feature was built to remove.

**Suggested fix:** When an entry has resolves_ts, resolve it to the referenced entry's category (build a ts→category index first) and credit latest_fix_by_cat under THAT category, falling back to the entry's own category field.

## Metals subsystem (REAL MONEY)

### [P1 → P1 · bug · conf 0.90] Playwright context thread-affinity conflict silently disables grid fisher (and stop-management calls) for entire trading days
`/mnt/q/finance-analyzer/portfolio/avanza_session.py:147`

avanza_session caches one sync-Playwright context in module globals (_pw_context, created at avanza_session.py:147). Sync Playwright objects are bound to the creating thread; calls from any other thread raise greenlet 'cannot switch to a different thread (which happens to have exited)'. Three competing initializers exist: (a) avanza_account_check._api_get_categorized_accounts (avanza_account_check.py:321-324) runs api_get on a TRANSIENT ThreadPoolExecutor thread at metals_loop startup — if it wins, the context is bound to a thread that immediately exits; (b) GridFisher._safe_session_call's persistent 'grid-fisher-session' worker (grid_fisher.py:1066); (c) main-thread stop-management paths (_capture_stop_snapshot metals_loop.py:3989, cancel_all_stop_losses_for metals_loop.py:4101, spike rollback metals_loop.py:5476/5497). The recovery classifier is_browser_dead_error (avanza_resilient_page.py:55-67) does NOT match the greenlet error, so _with_browser_recovery never relaunches; GridFisher._safe_session_call just logs and returns None. Main-thread paths self-heal by calling close_playwright() and rebinding (metals_loop.py:4001-4003), which then re-breaks the grid worker. Empirical: data/grid_fisher_decisions.jsonl contains 16,719 'cannot switch to a different thread' session_call_error entries between 2026-05-11 and 2026-06-09, with ~1,754/day (i.e. every tick, all day) on at least 8 full days — the grid market-maker was completely blind (no reconcile, no stop rearm, no EOD flat) those days with only journal-level logging, no Telegram/critical_errors escalation. While inventory is held, this is naked-position risk; it also intermittently blocks the L3 emergency-sell stop-snapshot path.

**Suggested fix:** Pin all avanza_session traffic to one dedicated long-lived internal worker thread inside avanza_session itself (module-level single-thread executor wrapping api_get/api_post/api_delete), OR add the greenlet 'cannot switch to a different thread' marker to is_browser_dead_error so _with_browser_recovery tears down and relaunches on the calling thread. Also escalate: N consecutive tick_fetch_degraded cycles should write data/critical_errors.jsonl + Telegram.

**Skeptic verdict (real):** Confirmed empirically: 16,719 'cannot switch to a different thread' session_call_error entries in data/grid_fisher_decisions.jsonl (2026-05-11 → 2026-06-09), with ~1,754/day = every tick on 9 full days; is_browser_dead_error (avanza_resilient_page.py:55-67) has no marker matching the greenlet message so _with_browser_recovery (avanza_session.py:224) never relaunches, and avanza_account_check.py:321-324 does bind the singleton context (avanza_session.py:147) to a transient thread. Only journal-level logging (grid_fisher.py:1080-1084), no critical_errors/Telegram. One caveat: severity narrative is slightly overstated — the grid never had a fill (0 fill_buy in the log, inventory 0), so no naked position actually occurred; severity rests on the subsystem being silently dead 30%+ of trading days for a month.

### [P1 → P2 · bug · conf 0.80] Global halt returns before the EOD sweep and never cancels armed buys; halted flag is never cleared
`/mnt/q/finance-analyzer/portfolio/grid_fisher.py:1759`

In GridFisher.tick, the global-halt check (grid_fisher.py:1759-1765) returns early BEFORE the EOD block (1768-1778). On a session that breaches the global loss limit, every subsequent tick re-derives the halt from session P&L and returns — so (a) eod_market_flat/eod_cancel_buys never run that day: leveraged warrant inventory is carried overnight precisely on the worst-loss day, and (b) armed buy limit orders are never cancelled at halt time and can keep filling at the broker while the system is 'halted'. Additionally, roll_session_if_new_day (grid_fisher.py:491-531) never resets state.halted/halt_reason, so the flag stays True forever in state and on the dashboard (summarise, grid_fisher.py:2047) even though trading silently resumes next session when P&L resets — the flag gates nothing.

**Suggested fix:** On halt: cancel_armed_buys for all instruments, and still execute the EOD sweep path (or run eod handling before the halt early-return). Reset halted/halt_reason in roll_session_if_new_day, and make tick() check state.halted explicitly rather than re-deriving from P&L.

**Skeptic verdict (real):** Confirmed in code: tick() returns at grid_fisher.py:1765 before the EOD block at 1768-1778 with no cancel_armed_buys on halt, roll_session_if_new_day (grid_fisher.py:491-531) resets P&L but never resets halted/halt_reason, and the placement loop never checks state.halted (halt is re-derived from P&L each tick). Downgraded to P2: halt has never fired (0 halt_global entries in grid_fisher_decisions.jsonl, current state halted=false), carried inventory would retain its broker stop (valid_days=8, avanza_session.py:731), and global notional is capped at 6,500 SEK.

### [P1 → P2 · bug · conf 0.70] get_open_orders swallows RuntimeError and returns [] — grid fisher reconcile then mass-misclassifies live orders as filled/cancelled
`/mnt/q/finance-analyzer/portfolio/avanza_session.py:661`

get_open_orders (avanza_session.py:653-669) catches RuntimeError from api_get (any non-401 HTTP error, e.g. Avanza 5xx) on BOTH the primary and fallback endpoints and returns []. GridFisher.tick explicitly relies on 'On failure we get None (NOT []), which is distinguishable from empty book' (grid_fisher.py:1657-1677) — that contract only holds for AvanzaSessionError/timeout, not for HTTP errors. On a transient 5xx, reconcile_against_live receives an empty open_order_ids set and marks every ARMED buy/sell tier filled or cancelled (grid_fisher.py:721-770): still-resting buys are dropped from state (next tick re-places a fresh ladder on top of the live zombie orders, doubling exposure outside the planned-notional cap accounting), and tracked sells are lost so later real fills never decrement inventory (phantom inventory → oversized EOD sell). The same [] fallback also misleads _rollback_spike_order_and_restore (metals_loop.py:5497-5502), which treats 'order not in open list' as terminal and restores full stops on top of a possibly-live spike sell.

**Suggested fix:** Make get_open_orders raise (or return None) on read failure and fix the two fallback `except RuntimeError: return []` branches; alternatively add get_open_orders_strict and use it from GridFisher.tick and _rollback_spike_order_and_restore. Treat 'cannot read open orders' as a degraded cycle, never as an empty book.

**Skeptic verdict (real):** Confirmed: api_get raises RuntimeError on any non-401 HTTP error (avanza_session.py:265) and get_open_orders catches it on both endpoints returning [] (avanza_session.py:661-669), violating the explicit None-vs-[] contract documented in grid_fisher.py:1657-1663; reconcile_against_live (grid_fisher.py:721-747) then marks every ARMED tier cancelled/filled, and _rollback_spike_order_and_restore (metals_loop.py:5497-5502) treats [] as spike-terminal and restores full stops. Downgraded to P2: trigger requires BOTH order endpoints to HTTP-error while get_positions succeeds in the same tick (otherwise the degraded-tick path catches it), and zero occurrences in the decisions log to date.

### [P1 → P2 · bug · conf 0.65] 1000-SEK minimum order guard also blocks SELLs — sub-1000 SEK grid inventory can never be exited (EOD flat retries forever)
`/mnt/q/finance-analyzer/portfolio/avanza_session.py:601`

_place_order's H8 guard (avanza_session.py:599-602) raises ValueError for ANY order under 1000 SEK, including risk-reducing SELLs. Grid fisher legitimately produces sub-1000 lots: reconcile partial-fill handling shrinks tier qty to the actually-filled delta (grid_fisher.py:740-744), and a 1200-SEK leg that has dropped in price can fall under 1000 at the bid. eod_market_flat then calls place_sell_order at bid*0.99 (grid_fisher.py:1969-1973); the ValueError is swallowed by _safe_session_call into a None → 'eod_market_sell_failed' → stop_needs_rearm=True → retry next tick → fails identically forever. The position cannot be exited by the system at all (rotation take-profit sells hit the same guard); the only exits are the broker stop (place_stop_loss deliberately only WARNS under 1000, avanza_session.py:769-780) or manual intervention. Note the rules file says >=1000 SEK 'per leg' to avoid minimum courtage — a fee-efficiency rule, not a reason to forbid closing a position.

**Suggested fix:** Exempt SELL orders (or add an allow_sub_minimum flag used by exit paths) from the H8 minimum in _place_order, mirroring place_stop_loss's warn-don't-raise behaviour; alternatively have eod_market_flat detect the sub-minimum case and escalate to Telegram instead of silently retrying.

**Skeptic verdict (real):** Confirmed: H8 guard (avanza_session.py:599-602) raises for any side, place_sell_order routes through _place_order (avanza_session.py:572), and eod_market_flat's failed sell (grid_fisher.py:1969-1987) would retry identically each tick after cancelling the stop — partial-fill qty shrink (grid_fisher.py:740-744) can legitimately produce sub-1000 lots; the contrast with place_stop_loss's deliberate warn-don't-raise (avanza_session.py:769-780, dated 2026-04-17) supports this being an oversight for exits. Downgraded to P2: zero 'below minimum 1000' or eod_market_sell_failed entries in the decisions log (grid never had a fill), and worst-case unprotected exposure is bounded under ~1000 SEK per lot.

### [P1 → P2 · design · conf 0.65] Two getUpdates consumers on the same bot token — telegram_poller eats CONFIRM replies so confirmed orders silently expire
`/mnt/q/finance-analyzer/portfolio/avanza_orders.py:274`

avanza_orders._check_telegram_confirm polls Telegram getUpdates with its own offset file (avanza_orders.py:260-294, called once per main-loop cycle from portfolio/main.py:1044), while telegram_poller polls the SAME bot token every 5 seconds with a separate offset (telegram_poller.py:124-141). Telegram getUpdates delivers each update to whichever consumer fetches it first and discards updates below the highest confirmed offset. The 5s poller wins essentially every race; a user's 'CONFIRM <token>' reply is unrecognized by the poller (settled drop, offset advanced) and is gone from the server before avanza_orders ever polls. Result: the human-in-the-loop confirmation flow for real-money Tier-2 orders is broken whenever the poller is running — pending orders expire after 5 minutes despite a timely CONFIRM, and the failure looks like the user never replied.

**Suggested fix:** Single consumer: have telegram_poller recognize CONFIRM messages and hand them to avanza_orders (e.g. write matched tokens to a small JSONL/state file that check_pending_orders reads), and remove the direct getUpdates call from avanza_orders.

**Skeptic verdict (real):** Confirmed architecturally: both consumers poll getUpdates on config['telegram']['token'] with independent offsets (avanza_orders.py:260-294 once per 60s cycle vs telegram_poller.py:113-141 every 5s, started in main.py:1425-1427 in the same process), the poller has no CONFIRM handling and settles unrecognized messages with offset ack (telegram_poller.py:151-155) — a CONFIRM reply would be consumed and lost. Downgraded to P2: the flow is dormant — request_order has no production callers, and neither data/avanza_pending_orders.json nor data/avanza_telegram_offset.txt has ever been created, so no real order has been harmed; it breaks deterministically only if/when the confirmation flow is actually used.

### [P1 → REFUTED · bug · conf 0.60] minutes_until_eod rolls over to tomorrow after the 21:55 cutoff — placement re-arms in the 21:55-22:00 window right after the EOD flat
`/mnt/q/finance-analyzer/portfolio/grid_fisher.py:314`

minutes_until_eod returns minutes-until-TOMORROW's cutoff once 21:55 Stockholm has passed (grid_fisher.py:314-317). The tick's EOD gate (1768-1778) therefore stops firing at 21:55:01, and the normal placement path resumes while FNSE warrants still trade until ~22:00. Gate A staleness (30-min threshold, GRID_QUOTE_STALENESS_THRESHOLD_S=1800) does not block because the orderbook traded seconds ago. Net effect: the system can market-flat inventory at ~21:50 and then place a fresh 2-tier buy ladder at 21:56 that can fill in the closing minutes — recreating the overnight leveraged exposure the EOD flat exists to prevent (rotation sell is a day order that expires at close; only the stop survives). The 22:00-22:25 tail can also re-trigger the documented ghost-cancel loop until Gate A's 30-min staleness kicks in.

**Suggested fix:** Return 0 (or a negative sentinel / separate 'post_close' flag) from minutes_until_eod between the cutoff and local midnight, or have tick() skip placement whenever local Stockholm time is past the cutoff for the current session.

**Skeptic verdict (refuted):** Refuted: metals_loop gates the entire cycle on is_market_hours() (metals_loop.py:7245 'continue' fires before the grid tick at 7542), which returns False after 21.92h ≈ 21:55:12 Stockholm local (metals_loop.py:1583; get_cet_time in data/metals_shared.py:62 is DST-safe Europe/Stockholm) — so the tick cannot run 21:56-22:00 or in the 22:00-22:25 'ghost-cancel' window. The rollover itself is documented as intentional (grid_fisher.py:295-299). Residual exposure is a ~12-second sliver (21:55:00-21:55:12), at most one tick on some days, not the claimed 5-minute window.

### [P2 · doc-drift · conf 0.75] Hardcoded 21:55 close in grid fisher and metals_loop violates the 'use todayClosingTime' rule — EOD flat never runs on Swedish half-days
`/mnt/q/finance-analyzer/portfolio/grid_fisher.py:287`

Project rules (.claude/rules/metals-avanza.md 'Trading Hours' and memory reference_avanza_trading_hours) mandate reading todayClosingTime from the Avanza API and explicitly say do NOT hardcode 21:55. grid_fisher.py:287-288 hardcodes _EOD_LOCAL_HOUR=21/_EOD_LOCAL_MINUTE=55, and metals_loop.is_market_hours (data/metals_loop.py:1571-1583) hardcodes 08:15-21:55. On Swedish early-close days (half-days before holidays, ~13:00 close), the grid's EOD sweep window (21:45-21:55) occurs hours after the market closed: buy tiers are never cancelled (they expire as day orders, benign) but inventory is never market-flattened — held overnight on 5x certs, with the stop as the only protection across the gap. zoneinfo handles DST, but not the exchange calendar.

**Suggested fix:** Fetch todayClosingTime per instrument (Avanza market-guide payload already carries it) once per session and compute the EOD cutoff from it, falling back to 21:55 only when the fetch fails; apply the same to is_market_hours.

### [P2 · bug · conf 0.70] eod_market_flat marks armed sell tiers CANCELLED without verifying the cancel succeeded — combined sell volume can exceed the position
`/mnt/q/finance-analyzer/portfolio/grid_fisher.py:1940`

eod_market_flat cancels armed sell tiers via _safe_session_call(cancel_order) and unconditionally sets tier.status=ORDER_CANCELLED (grid_fisher.py:1940-1946), ignoring a None/'rejected' result — asymmetric with cancel_armed_buys (1304-1321), which only marks CANCELLED after orderRequestStatus=='SUCCESS'. If the cancel fails, the old sell still rests at the broker but is dropped from state, and the full-inventory EOD sell is then placed on top: total sell volume > position, which Avanza rejects as short.sell.not.allowed (so the EOD flat fails for the day and retries against the same zombie) or, worse, both fill. Two further weaknesses in the same function: when the pre-sell quote fetch fails, the 'aggressive' limit falls back to avg_entry_price*0.99 (1962-1968), which can sit far ABOVE the market and never fill; and once eod_sell_order_id is set, the in-flight guard (1931-1938) prevents any repricing of a non-filling EOD sell for the rest of the session — with the stop already nulled at 1948-1956, the lot is unprotected until the next-day rearm.

**Suggested fix:** Mirror cancel_armed_buys: only flip a sell tier to CANCELLED on confirmed SUCCESS, and skip (retry next tick) otherwise. Skip the EOD sell when the quote is unavailable rather than pricing off avg entry, and add a reprice path: if the EOD sell hasn't filled after N ticks, cancel-and-replace at the fresh bid.

### [P2 · bug · conf 0.55] _handle_buy_fill add-to-existing places a second full-volume trailing stop without cancelling the previous one
`/mnt/q/finance-analyzer/data/metals_loop.py:4793`

When a trade-queue BUY adds to an existing position (metals_loop.py:4760-4772), the hardware-trailing branch (4793-4810) places a NEW trailing stop sized to the TOTAL units and overwrites POSITIONS[pos_key]['hw_trailing_stop_id'] — the previous stop (covering the old units) is never cancelled and its ID is forgotten by local state. The broker then holds stops totalling old_units + total_units > position. Consequences: stacked encumbrance that relies entirely on the server-side enumeration in cancel_all_stop_losses_for to unwind before any sell, and on a trigger event both stops fire sells whose combined volume exceeds the position (second one rejected as short.sell.not.allowed at best, racing fills at worst). Violates the rules-file invariant 'Sell + stop-loss volume must NOT exceed position size. Check existing orders before placing.'

**Suggested fix:** Before placing the new trailing stop in the add-to-existing path, cancel the recorded hw_trailing_stop_id via cancel_stop_loss (and verify), or call cancel_all_stop_losses_for(ob_id) and re-arm a single stop on the new total volume.

### [P2 · bug · conf 0.50] reconcile_against_live volume-delta inference is confounded when a buy and a sell both disappear in the same tick
`/mnt/q/finance-analyzer/portfolio/grid_fisher.py:730`

Reconcile classifies missing orders purely from the live-volume delta (grid_fisher.py:721-770). If a buy tier and a sell tier both fill within one 60s window with offsetting volumes (e.g. inventory 74, buy 74 fills and rotated sell 74 fills → live_vol 74), the buy loop computes delta=0 and marks the filled buy CANCELLED, then the sell loop computes inventory_drop=0 and marks the filled sell CANCELLED. Net inventory happens to be right, but: realised P&L on the sell is never booked (session_pnl_sek, consecutive_losses, the per-session loss breaker and the global halt all see nothing), avg_entry_price is wrong for the new lot, no rotation (take-profit sell + stop rearm) occurs for the newly filled buy — the fresh inventory sits with only the stale stop from the previous lot. Plausible in a fast tape on a 5x cert, and more likely the longer a degraded-session gap (finding on thread affinity) delays reconciliation.

**Suggested fix:** Use per-order fill information instead of pure volume deltas: query deals/executions (Avanza deals-and-orders endpoint) or compare order-level state for each missing order_id, falling back to the volume heuristic only when the per-order lookup fails.

### [P3 · design · conf 0.70] Global session-loss halt threshold scales with instrument count — 3000 SEK (~46% of the 6500 SEK budget) before the breaker fires
`/mnt/q/finance-analyzer/portfolio/grid_fisher.py:657`

should_halt_global sets the halt threshold to GRID_PER_SESSION_LOSS_LIMIT_SEK (500) × len(state.by_instrument) (grid_fisher.py:656-661). by_instrument is seeded with every ticker×direction pair from GRID_ACTIVE_INSTRUMENTS — 6 instruments (grid_fisher_config.py:140-144) — so the 'global' breaker only trips at -3000 SEK on a 6500 SEK budget. Since at most one direction per ticker is ever armed, the realistic concurrent exposure is 3 instruments, and the per-instrument -500 freeze makes a -3000 aggregate nearly unreachable before everything is already frozen individually: the global circuit breaker is effectively decorative. Adding instruments to the catalog silently loosens it further.

**Suggested fix:** Make the global limit an independent config constant (e.g. GRID_GLOBAL_SESSION_LOSS_LIMIT_SEK = 1000-1500) instead of deriving it from instrument count, or scale by active tickers (one direction each) rather than all seeded instruments.

## Prophecy subsystem

### [P1 → P1 · bug · conf 0.85] All prophecy 'critical' alerts are invisible to the startup check (level mismatch)
`/mnt/q/finance-analyzer/prophecy/alerts.py:22`

alerts.log_critical defaults to level="error" (and cost.py over-budget uses level="warning"), but scripts/check_critical_errors.py:122 find_unresolved() only surfaces entries with level == "critical" (other producers, e.g. portfolio/claude_gate.py:282, portfolio/portfolio_mgr.py:117, all write level: "critical"). Every prophecy alert — stale publish, torn raw, zero records, claude run error, cost over soft cap — is appended to critical_errors.jsonl but is never surfaced by the session-start check or the fix-agent dispatcher. This silently defeats the entire anti-stale/alerting design documented in publish.py and alerts.py docstrings (the exact 'silent failure' class the journal exists to prevent).

**Suggested fix:** Change log_critical's default level to "critical" (keep "warning" for the soft-cap row only if intentionally non-surfacing, and document that), or extend find_unresolved to include level in {"critical","error"}. Add a test asserting a publish stale-mark is picked up by check_critical_errors.find_unresolved.

**Skeptic verdict (real):** Confirmed: prophecy/alerts.py:22 defaults level="error" and every call site uses the default (publish.py:58, cost.py:86/128) or "warning" (cost.py:135), while both scripts/check_critical_errors.py:122 and scripts/fix_agent_dispatcher.py:186 skip anything where level != "critical". The alerts.py docstring (lines 3-6) explicitly states the intent is to be read by check_critical_errors.py + the fix-agent dispatcher, so this is a bug, not a choice; tests (tests/test_prophecy_pipeline.py:135/178/186) only assert the file exists, never that entries surface.

### [P1 → P2 · design · conf 0.85] 8 of 13 enabled instruments can never be outcome-scored — accuracy loop structurally absent
`/mnt/q/finance-analyzer/prophecy/outcomes.py:109`

outcomes.py fetches realized prices only via portfolio.outcome_tracker._fetch_historical_price, whose maps (portfolio/tickers.py:13-56) cover only BTC-USD, ETH-USD, XAU-USD, XAG-USD, MSTR. CL=F, BZ=F, XBT-TRACKER, ETH-TRACKER, MINI-SILVER, SAAB-B, SEB-C and INVE-B always return None → counted as 'unscorable' and skipped, every day, with no alert. prep.py:75 has a price_source fallback for *current* prices (so these instruments get predictions with a spot), but outcomes has no equivalent historical fallback. The result: accuracy.json — the feedback loop that is the system's stated purpose — will never contain a majority of the configured instruments, and nobody is told.

**Suggested fix:** Add a historical-price fallback (portfolio.price_source klines with start/end, Avanza chart API for warrants/Swedish equities) or disable unscorable instruments in prophecy_config.json by default; log a critical error when an instrument stays unscorable past its first matured horizon.

**Skeptic verdict (real):** Confirmed: portfolio/tickers.py:13-56 maps only BTC-USD/ETH-USD/XAU-USD/XAG-USD/MSTR, and portfolio/outcome_tracker.py:211 _fetch_historical_price returns None for everything else, so the 8 other enabled instruments in data/prophecy_runs/prophecy_config.json (CL=F, BZ=F, 3 warrants, SAAB-B, SEB-C, INVE-B) are either counted 'unscorable' forever (oil, which gets a spot via prep.py:75 price_source fallback) or silently skipped at outcomes.py:90 (warrants/Tier-2 with spot=None), with no critical alert — only a stdout counter (outcomes.py:144) that Task Scheduler discards. Downgraded to P2: the 5 core instruments are scored, prep.py:62-64 comments show the no-feed limitation was partially known, and it is a design gap in a frozen non-trading subsystem rather than an active failure.

### [P1 → P2 · reliability · conf 0.70] Unvalidated Claude-supplied spot_at_prediction can permanently crash or poison outcome scoring
`/mnt/q/finance-analyzer/prophecy/outcomes.py:114`

publish.py:107-109 stamps spot_at_prediction from the raw Claude record (raw_rec.get("spot_at_prediction")) whenever prep's live_price is None, with no type/positivity validation (validate_record never checks it). outcomes.py:114 then calls float(spot) unguarded: a non-numeric string or dict raises ValueError/TypeError, aborting the entire score() loop mid-journal. Because the poisoned row stays in prediction_journal.jsonl and rows are iterated in order, every subsequent daily run crashes at the same row — all later predictions are never scored, forever. The .bat (line 62) ignores outcomes' exit code and score() has no try/except or critical-error log around the loop, so the outage is fully silent. Additionally spot == 0 passes the `spot is None` check and makes _realized_direction (line 47, `if spot else 0.0`) return "flat" for every horizon, silently corrupting dir_hit_rate.

**Suggested fix:** In publish.py validate the fallback spot with the same _safe_float + > 0 rule as targets (else leave it None); in outcomes.py wrap per-record scoring in try/except that logs a critical error and continues, and skip records where spot fails float() or is <= 0.

**Skeptic verdict (real):** Code defect confirmed: publish.py:107-109 falls back to raw_rec.get("spot_at_prediction") with no validation (schema.py:239 comment confirms validate_record never checks it), outcomes.py:90 only filters `spot is None` (0 passes -> _realized_direction line 47 returns flat), and line 114 float(spot) is unguarded in a loop with no per-record try/except, so one poisoned journal row crashes every future score() run while prophecy-daily.bat:62/82 ignores outcomes' exit code. Downgraded to P2 because docs/prophecy-prompt.md never asks Claude to emit spot_at_prediction, so the fallback only fires when prep's live_price is None AND Claude spontaneously emits the field with a non-numeric value — low likelihood, severe-but-latent failure mode.

### [P1 → P1 · security · conf 0.60] Headless web-researching agent runs with repo-wide Bash(*)/Write(*) allow — prompt-injection to RCE on the trading box
`/mnt/q/finance-analyzer/scripts/prophecy-daily.bat:56`

The claude -p invocation passes no permission restrictions, and the repo's .claude/settings.json permissions.allow includes Bash(*), Write(*), Edit(*), WebFetch(*), WebSearch(*) — inherited by headless runs from this cwd. The prophecy prompt explicitly instructs the agent to fetch and read untrusted forum/social content (r/Silverbugs, WallStreetSilver, Flashback, X/TradingView — docs/prophecy-prompt.md:57-62) and to fan out subagents. A prompt-injection payload in any fetched page can therefore execute arbitrary shell commands with full user privileges on the machine that holds the live Avanza session, config.json API keys and real-money loops. The prompt's 'Do not commit, push, or modify any other file' line is advisory, not enforced.

**Suggested fix:** Constrain the headless run: pass --allowedTools (Read, WebSearch, WebFetch, Write limited to data/prophecy_runs/raw_*.json) or --permission-mode with a dedicated settings profile / --strict-mcp-config, instead of inheriting the interactive allow-all project settings. Same hardening applies to the other research .bat files.

**Skeptic verdict (real):** Confirmed: scripts/prophecy-daily.bat:56 invokes claude -p with no --allowedTools/--permission-mode after cd to the repo (line 20), and .claude/settings.json permissions.allow includes Bash(*), Edit(*), Write(*), WebFetch(*), WebSearch(*), which headless runs inherit from project settings; docs/prophecy-prompt.md (steps 3-4) and prophecy/strategies.py:114/211/228 explicitly direct the agent into untrusted forum content (r/Silverbugs, WallStreetSilver, Flashback, Avanza forum, X/TradingView), making injected-content-to-arbitrary-Bash a real path on the box holding the Avanza session and config.json keys. Currently mitigated only by the temporary SYSTEM_DISABLED sentinel (present in data/prophecy_runs/, checked at .bat line 32) — the freeze is explicitly temporary, so this needs hardening before re-enable.

### [P2 · bug · conf 0.85] cumulative_30d_usd is last-30-entries not 30 days, and same-day re-runs double-count
`/mnt/q/finance-analyzer/prophecy/cost.py:109`

record_cost computes `sum(... for r in rows[-30:])` over the last 30 JSONL *entries* and publishes it as cost_summary.cumulative_30d_usd. Entries are appended on every invocation with no per-date dedup: a manual re-run, the scheduled task's RestartCount=1 retry (install-prophecy-task.ps1:34-35), or re-running the bat after a partial failure appends duplicate rows for the same date, inflating the '30d' figure and shrinking the real time window covered. Conversely the window silently covers >30 days when runs are skipped. Same root issue exists in publish.py: re-running publish for the same date appends a full duplicate set of journal rows (outcomes dedup by (date,inst,horizon) hides it from scoring, but journal and cost both lack idempotency).

**Suggested fix:** Filter rows by ts >= now-30d when summing (entries carry ts already), and dedup by date — either skip append if a row for `date` exists or keep last-per-date when aggregating. Consider the same date-idempotency guard in publish.py.

### [P2 · bug · conf 0.80] Publish never reconciles against enabled instruments — hallucinated instruments accepted, missing ones unflagged
`/mnt/q/finance-analyzer/prophecy/publish.py:97`

The publish loop journals every record that passes validate_record, whose only instrument check is non-empty string (schema.py:198-200). An instrument Claude hallucinates (e.g. 'DOGE-USD') is journaled, lands in latest.json, and generates perpetual unscorable noise in outcomes. The inverse is worse: if Claude emits only 3 of 13 instruments (e.g. --max-turns 600 backstop hit mid-run), publish succeeds with stale=false and no alert — _mark_stale fires only on zero records (line 85/133). The prompt demands an entry for every instrument (docs/prophecy-prompt.md:131) but nothing downstream enforces it, so partial silent coverage looks healthy on the dashboard.

**Suggested fix:** Compare published keys against pcfg.enabled_instruments(): quarantine records for unknown instruments, and log_critical (plus a latest.json 'missing_instruments' field) when any enabled instrument is absent from the run.

### [P2 · reliability · conf 0.75] Kill switch is fail-open and the git-tracked sentinel makes go-live state unstable in both directions
`/mnt/q/finance-analyzer/scripts/prophecy-daily.bat:32`

The only token-spend guard (this path bypasses claude_gate) is `if exist %PDIR%\SYSTEM_DISABLED` — absence of a file authorizes spend. If data\prophecy_runs\ is deleted or renamed (data cleanup, disk restore) the freeze silently lifts while config.json guard 1 still passes. Conversely, SYSTEM_DISABLED is git-TRACKED (verified via git ls-files), so the documented go-live step (`del ...SYSTEM_DISABLED`, install-prophecy-task.ps1:60) leaves the repo permanently dirty and any `git checkout .` / stash / worktree merge resurrects the sentinel, silently re-freezing: the bat then exits 0 BEFORE the prophecy-log.jsonl 'started' append (line 44), so there is no record at all that runs stopped. No frozen-skip event is ever logged.

**Suggested fix:** Invert to fail-closed: require an ENABLED sentinel (e.g. data/prophecy_runs/LIVE, gitignored) to spend; log a jsonl event on frozen-skip so silence is detectable; untrack SYSTEM_DISABLED (git rm --cached + .gitignore) so go-live doesn't fight git.

### [P2 · reliability · conf 0.70] prep exit code unchecked; missing context disables publish's anti-stale guard
`/mnt/q/finance-analyzer/scripts/prophecy-daily.bat:49`

Step 1 (`%PY% -m prophecy.prep`) result is never checked — if prep crashes (import error, torn data file), the bat proceeds straight to the token-spending claude run. The prompt's self-heal ('run prophecy.prep yourself', docs/prophecy-prompt.md:41) fails for the same root cause, after which publish.py:75 skips the raw-vs-context mtime stale check entirely when ctx_path doesn't exist, and spot_at_prediction falls back to Claude's self-claimed value (publish.py:107-109). Net effect: a broken prep yields a full-cost run published with unverified spots and no coverage seeding, instead of an aborted run.

**Suggested fix:** After the prep call: `if errorlevel 1 ( log + exit /b )` before the claude step; in publish.py treat a missing context file for the run date as a stale condition (_mark_stale) rather than skipping the guard.

### [P2 · bug · conf 0.70] Weekend/holiday horizons for YF-mapped instruments score a ~4h same-day window as a 1-2 day move
`/mnt/q/finance-analyzer/prophecy/outcomes.py:105`

Maturity uses calendar deltas (HORIZON_DELTAS), and for YF_MAP instruments (MSTR) _fetch_historical_price (portfolio/outcome_tracker.py:286-289) returns the last daily close with date <= target date. A Friday ~12:00 UTC prediction's 1d horizon targets Saturday; the resolver returns Friday's close — i.e. 'realized' is measured ~4 hours after prediction, not 1 day. The 2d (Sunday) horizon resolves to the same Friday close. With the 0.3% flat band this systematically converts weekend-spanning 1d/2d MSTR predictions into near-flat outcomes (2 of 7 short horizons every week), structurally depressing dir_hit_rate and corrupting the per-horizon accuracy rollup.

**Suggested fix:** For non-24/7 instruments, advance target_dt to the next trading session close (or skip scoring when the resolved bar's date <= prediction date), and record the actually-used bar timestamp in the accuracy row for auditability.

### [P2 · quality · conf 0.65] Scoring semantics mismatch: flat band undefined for the model, direction/probability consistency unchecked
`/mnt/q/finance-analyzer/prophecy/outcomes.py:36`

Outcome scoring defines 'flat' as |move| <= 0.3% (_FLAT_BAND), but docs/prophecy-prompt.md never tells the model what 'flat' means, so prob_flat and direction='flat' predictions are scored against a band the producer doesn't know — for 1mo/2mo/6mo horizons a ±0.3% outcome is essentially impossible, making any 'flat' call an automatic miss while Brier treats flat as not-up. Separately, schema._validate_horizon (schema.py:119-174) never enforces that direction matches the dominant probability (the prompt requires it), so a record with direction='up' and prob_down=0.7 passes validation; dir_hit then scores the direction string while Brier scores prob_up, producing internally contradictory accuracy stats with no validation note.

**Suggested fix:** State the flat band (and per-horizon band, if widened for long horizons) in the prompt and HORIZON metadata; in _validate_horizon, repair direction to argmax(prob_up, prob_down, prob_flat) and append a validation note when they disagree.

### [P3 · design · conf 0.85] Model hardcoded in .bat; prophecy_config.json 'model' is dead config and cost rows can mislabel the actual model
`/mnt/q/finance-analyzer/scripts/prophecy-daily.bat:56`

config.py declares itself 'single source of truth' for which model to use (config.py:1-6) and exposes model()/DEFAULT_MODEL='claude-opus-4-8' (config.py:32,113), prep embeds it in the context, and cost.py:96 stamps each cost row with pcfg.model(). But the bat hardcodes `--model claude-opus-4-8`, so editing prophecy_config.json's model changes what cost_log.jsonl and the journal *claim* was used without changing the actual run — silently mislabeling cost/accuracy attribution the first time anyone swaps models via config.

**Suggested fix:** Have the bat read the model from config (e.g. `for /f %%M in ('%PY% -c "from prophecy import config; print(config.model())"')`) or, simpler, have cost.py prefer the model reported in the claude result object over pcfg.model().

## Orchestration (main loop, triggers, Layer 2)

### [P2 · design · conf 0.90] agent_invocation bypasses the claude_gate kill switch — freeze/re-enable requires flipping 3+ independent switches
`/mnt/q/finance-analyzer/portfolio/agent_invocation.py:1156`

claude_gate.py declares itself 'the ONLY approved way to invoke Claude Code' and 'direct subprocess.Popen([claude_cmd, -p, ...]) calls are FORBIDDEN' (claude_gate.py:3-9), yet agent_invocation.invoke_agent builds the claude command at line 1156 and Popens it directly at 1224 — skipping CLAUDE_ENABLED, the daily rate-limit warning, claude_invocations.jsonl cost tracking, and the tree-kill-capable spawn kwargs. The claude_gate.py:54-61 comment explicitly documents the consequence: the token freeze requires CLAUDE_ENABLED, config.layer2.enabled, AND data/metals_loop.py CLAUDE_ENABLED to be flipped together, and 'flipping only one is a silent half-on state'. The current taskkill-based kill path in _kill_overrun_agent also duplicates (less robustly, no CREATE_NEW_PROCESS_GROUP) what _run_with_tree_kill/_kill_process_tree already do.

**Suggested fix:** Either route the spawn through a non-blocking claude_gate API (add a gate-checked Popen helper that honors CLAUDE_ENABLED and logs to claude_invocations.jsonl), or at minimum call claude_gate.check_claude_gates('layer2') at the top of invoke_agent so one master switch governs all Claude paths.

### [P2 · bug · conf 0.85] _detect_regime reads extra['regime'] but signal_engine publishes extra['_regime'] — regime always 'range-bound'
`/mnt/q/finance-analyzer/portfolio/autonomous.py:421`

autonomous.py:_detect_regime does `sig.get('extra', {}).get('regime')`, but signal_engine.py:4543 stores the regime as `extra_info['_regime']` (trigger.py:303 correctly reads '_regime'). No code path puts a bare 'regime' key in extra, so regimes list is always empty and _detect_regime returns the fallback 'range-bound' for every autonomous decision. Since 2026-06-06 the autonomous path IS the production decision path (layer2 frozen), so every journal entry, decision log row, and Telegram reasoning line ('Regime: range-bound. Monitoring.') carries a fabricated regime, and any downstream consumer of journal_entry['regime'] is fed constant wrong data.

**Suggested fix:** Change to `sig.get('extra', {}).get('_regime')` (keep 'regime' as a fallback for old data).

### [P2 · bug · conf 0.75] invoke_agent silently discards an exited-but-unobserved agent's completion (no auth scan, no completion row, no record_trade)
`/mnt/q/finance-analyzer/portfolio/agent_invocation.py:871`

Inside the _completion_lock block, `if _agent_proc and _agent_proc.poll() is None:` only handles the still-running case. If the previous agent has EXITED but its completion was not yet observed (poll() returns an exit code), the condition is False and invoke_agent falls through: it closes _agent_log (line 896), then overwrites _agent_proc, _journal_count_before, _agent_tier etc. at spawn (lines 1209-1231). The finished invocation gets NO completion row in invocations.jsonl, NO journal_written/telegram_sent verification, NO _scan_agent_log_for_auth_failure (the exact silent 'Not logged in' exit-0 class the Mar-Apr 2026 outage exposed), no _record_new_trades, and no stack-overflow counter update. Window is bounded by the 30s watchdog tick, but trigger-storm cycles that invoke back-to-back can land inside it.

**Suggested fix:** In the locked block, when `_agent_proc is not None and _agent_proc.poll() is not None`, call `_check_agent_completion_locked()` before proceeding to spawn, so the finished run is fully accounted for.

### [P2 · reliability · conf 0.70] Layer 2 subprocess is orphaned across loop restarts — no PID persistence, possible concurrent duplicate agent
`/mnt/q/finance-analyzer/portfolio/agent_invocation.py:1224`

_agent_proc and all completion baselines are in-process module globals. The standard workflow restarts PF-DataLoop after every merge (taskkill on the python PID), and T2/T3 agents run 600-900s — so a restart mid-invocation leaves the claude CLI child running as an orphan (taskkill on the parent without /T does not kill the child tree). The new loop process knows nothing about it: the orphan keeps editing portfolio_state.json, appending journal/telegram entries, while the new loop can spawn a SECOND concurrent agent on the next trigger. Two agents doing read-modify-write on portfolio_state.json gives lost-update corruption of simulated holdings, plus duplicate Telegram and journal entries that confuse the count-delta completion heuristics.

**Suggested fix:** Persist {pid, start_ts, tier} to a small JSON at spawn; on loop startup, check the file and either taskkill /T the stale PID or refuse to spawn until it exits. Clear the file in the completion/kill paths.

### [P2 · reliability · conf 0.60] Singleton lock is not cross-environment: Windows msvcrt byte-range lock and WSL fcntl flock cannot see each other
`/mnt/q/finance-analyzer/portfolio/main.py:79`

_acquire_singleton_lock uses msvcrt.locking on Windows-Python and fcntl.flock on WSL-Python against the same data/main_loop.singleton.lock file. These two mechanisms do not interoperate (Windows byte-range locks vs POSIX advisory flock over drvfs/9p), so a `--loop` started from WSL (the documented interactive environment, with a second Claude Code install) is NOT blocked by the production Windows loop, and vice versa. Two loops then double-write trigger_state.json, signal_log, portfolio state, and double-invoke Layer 2 — the exact storm the C5 fix claims to prevent. The C5 comment ('Now supports both Windows and Linux/WSL') implies protection that only holds within one OS.

**Suggested fix:** Add an OS-agnostic secondary check: write PID+hostname+boot-id into the lock file and at startup probe whether that PID is alive on the Windows side (e.g. via tasklist through cmd.exe when running under WSL), or simply refuse `--loop` when running under WSL (platform check) since production is Windows-only.

### [P2 · bug · conf 0.60] telegram_sent detection counts ANY system Telegram during the agent's run as the agent's — masks incomplete runs
`/mnt/q/finance-analyzer/portfolio/agent_invocation.py:1549`

_detect_append on TELEGRAM_FILE compares whole-file line counts/last-ts before vs after the subprocess. telegram_messages.jsonl is shared by every sender in the system (4h digest, daily digest, LOOP ERRORS alerts, autonomous messages, message_throttle flush, metals subsystem). A T2/T3 agent runs 600-900s; any concurrent send in that window — guaranteed at digest boundaries and on any cycle error — makes telegram_sent=True for an agent that sent nothing, flipping a genuinely 'incomplete' run to 'success' and suppressing the *L2 INCOMPLETE* alert. This is the same silent-failure class the 2026-05-15/05-17 count-delta fix targeted, just via a different aliasing source. journal detection is less exposed (only L2/autonomous write it and they're mutually exclusive per-config) but the timeout/failed stubs written by this module itself also inflate it.

**Suggested fix:** Tag rows at write time (per the project's own 'tag rows at write-time, filter at read-time' lesson): have the agent write a per-invocation marker (e.g. message category='layer2' with the invocation ts), and have _detect_append count only rows whose source/category matches the agent, not raw file growth.

### [P3 · doc-drift · conf 0.85] CLAUDE.md and in-code comments still describe a 60s loop; actual cadence has been 600s since 2026-04-09
`/mnt/q/finance-analyzer/portfolio/market_timing.py:20`

INTERVAL_MARKET_OPEN/CLOSED/WEEKEND are all 600s (market_timing.py:20-22), but CLAUDE.md says 'Layer 1 (Python, 60s loop)' and '60s cycle: fetch OHLCV → ...', and main.py:1070's safeguard comment says 'every 100 cycles ≈ 100 min' when 100 cycles is now ≈16.7h — meaning the outcome-staleness and dead-signal safeguards run ~10x less often than the comment (and probably the original intent) implies. Headless Layer 2 agents read CLAUDE.md as ground truth every invocation, so the stale cadence claim feeds wrong assumptions into trade-decision prompts.

**Suggested fix:** Update CLAUDE.md's Layer 1 description to 600s, and either fix main.py's '≈ 100 min' comment or change the safeguard modulus to ~10 cycles to restore the intended ~100-minute interval.

### [P3 · quality · conf 0.80] Every False return from invoke_agent is re-logged as 'skipped_busy', double-logging wrong statuses into invocations.jsonl
`/mnt/q/finance-analyzer/portfolio/main.py:997`

main.py logs `'invoked' if result else 'skipped_busy'` (lines 972, 997), but invoke_agent returns False for at least nine distinct reasons (skipped_stack_overflow, skipped_auth_cooldown, layer2-disabled, perception-gate skip, blocked_drawdown_*, blocked_trade_guards, skipped_no_position, kill-failure, Popen error) — most of which ALREADY wrote their own accurate _log_trigger row. The result is two rows per skip, one mislabeled 'skipped_busy', polluting trigger-storm/no-invocation diagnosis. Related: the multi-agent quorum-fail path uses a bare `return` (agent_invocation.py:1104) returning None instead of False, inconsistent with the documented bool contract.

**Suggested fix:** Have invoke_agent own all trigger logging and return a status string (or enum) that main.py logs verbatim once; change line 1104 to `return False`.

### [P3 · bug · conf 0.70] Flip cooldown is armed before the claude_budget floor gate, so a suppressed flip blocks the next genuine flip for 30 min
`/mnt/q/finance-analyzer/portfolio/trigger.py:393`

Section 2 sets `flip_cooldowns[ticker] = _flip_now_ts` (line 393) and only THEN appends the flip to _floor_candidates (line 406). The min_weighted_confidence / min_atr_multiple floor (lines 477-492) can later drop that candidate, but the cooldown stays armed — so a higher-conviction genuine flip on the same ticker within FLIP_COOLDOWN_S (1800s) is silently suppressed at line 386 even though no Layer 2 invocation ever happened. Additionally, because suppression means triggered=False, prev['signals'] is not refreshed, so the same flip re-detects next cycle and burns repeated log lines while in cooldown. Only manifests when config.claude_budget floors are non-zero, which is exactly the token-conservation configuration this system is moving toward.

**Suggested fix:** Defer `flip_cooldowns[ticker] = _flip_now_ts` until after the floor gate accepts the reason (e.g. carry ticker through _floor_candidates and arm cooldown in the emit loop).

### [P3 · design · conf 0.70] Plain Layer-2 path has no autonomous fallback outside the agent window, unlike the autonomous-first path
`/mnt/q/finance-analyzer/portfolio/main.py:999`

In the autonomous-first branch (lines 898-987), a trigger outside _is_agent_window() falls back to autonomous_decision(), producing a journal entry and Telegram. In the plain layer2-enabled branch (lines 993-1000), the same off-window trigger is just logged 'skipped_offhours' — no journal, no decision, no notification. So with the default config (autonomous_first_enabled=False), every overnight/weekend/Swedish-holiday trigger on 24/7 instruments (BTC, ETH, XAU, XAG) produces no decision record at all, while flipping one budget flag changes that behavior as a side effect. Note _is_agent_window() also blocks on Swedish market holidays (market_timing.py:258) even though the simulated portfolios trade US/crypto instruments that are open.

**Suggested fix:** Add the same autonomous_decision() fallback to the skipped_offhours branch (it is recommendation-only and cheap), and reconsider whether is_swedish_market_holiday belongs in _is_agent_window for non-Avanza instruments.

### [P3 · reliability · conf 0.65] Layer 2 enable gate fails OPEN on config read error — token freeze can silently un-freeze
`/mnt/q/finance-analyzer/portfolio/agent_invocation.py:850`

invoke_agent wraps `config = _load_config()` in try/except with `config = {}` fallback, then `l2_cfg.get('enabled', True)` (line 854) — so any transient config.json read failure (Windows file lock during external edit, symlink target briefly missing, mtime-cache re-read race) makes the kill flag evaluate True and the claude subprocess spawns despite the 2026-06-06 freeze. claude_gate._load_config_layer2_enabled (claude_gate.py:171-177) is also fail-open but is explicitly backstopped by the CLAUDE_ENABLED hard gate; agent_invocation BYPASSES claude_gate entirely (see claude_gate.py:54-58), so this fail-open default is its ONLY gate. A kill switch should fail closed.

**Suggested fix:** On config load failure in invoke_agent, log and return False (fail-closed), or at minimum default `enabled` to the last successfully-read value rather than True.

### [P3 · bug · conf 0.60] _update_sustained persists time.monotonic() into trigger_state.json — docstring's restart claim is wrong and post-restart flips skip the debounce
`/mnt/q/finance-analyzer/portfolio/trigger.py:150`

The sustained-debounce entries (value/count/_mono_start) are persisted via state['sustained_counts'] (line 521). The docstring claims 'On process restart, monotonic origin resets and the duration gate conservatively starts fresh' — but on Windows time.monotonic() is QPC-since-boot, shared across processes, so within the same boot a persisted _mono_start stays comparable and the elapsed INCLUDES loop downtime. Concretely: after a restart, the startup grace cycle returns early (line 269) without touching sustained_counts; on the next cycle any ticker whose action differs from the grace baseline gets duration_ok=True immediately from the stale _mono_start (>=900s old), firing a sustained-flip trigger with zero actual debounce. After a reboot the persisted values are garbage in the other direction (negative elapsed, gate dead until the value changes).

**Suggested fix:** Don't persist _mono_start: strip it in _save_state (like _current_tickers) and re-seed on first sight after load, or store a wall-clock first-seen ts and compare with time.time() plus a sanity clamp.

## Signal engine core

### [P1 → P2 · bug · conf 0.85] Disabled btc_proxy signal still votes live on MSTR — bypasses DISABLED_SIGNALS force-HOLD
`portfolio/signal_engine.py:4020`

btc_proxy was formally disabled 2026-05-24 (tickers.py:251, '44.6% 1d (139 sam), BUY 31.1% ... Formal disable') and added to _SHADOW_SAFE_SIGNALS (signal_engine.py:810). But the DISABLED_SIGNALS force-HOLD only happens inside the enhanced-signal dispatch loop (line 3813), and btc_proxy is NOT a registered enhanced signal — it is injected inline at lines 4020-4029 (`votes["btc_proxy"] = btc_action`) for MSTR with no DISABLED_SIGNALS check. Consequences: (1) it counts toward buy/sell/active_voters/post_persistence_voters, so a formally-disabled voter can satisfy MIN_VOTERS and the Stage-4 dynamic quorum; (2) the directional-rescue path (lines 2767-2777) can let its SELL votes through at 0.70-0.95x weight despite the formal disable (BUY 31.1% implies high SELL accuracy); (3) its _SHADOW_SAFE_SIGNALS membership is dead code because the shadow branch is only reachable for registry-dispatched signals, so the intended shadow-only tracking never engages. This is exactly the 'disabled signals leak into live votes' failure mode.

**Suggested fix:** In the MSTR injection block, route the vote to shadow_votes (not votes) when 'btc_proxy' in DISABLED_SIGNALS and not promoted: `shadow_votes['btc_proxy'] = btc_action` + extra_info only. Remove the dead _SHADOW_SAFE_SIGNALS entry or keep it documented as outcome-tracking-only.

**Skeptic verdict (real):** Confirmed: signal_engine.py:4020-4029 injects votes['btc_proxy'] after the dispatch loop, and the only DISABLED_SIGNALS gate (line 3813) applies solely to registry-dispatched signals — btc_proxy has no signals/ module, so its _SHADOW_SAFE_SIGNALS entry (line 810) is unreachable dead code; disable commit 14573681 never touched the injection block, and live data/agent_context_t2.json:38 shows btc_proxy voting SELL on MSTR with accuracy_cache samples growing 139→228 post-disable. Downgraded P1→P2: impact is bounded — ranging-regime gate (line 1395), the 47% accuracy gate in _weighted_consensus (44.6% blended fails it; SELL survives only via directional rescue at reduced weight, which sell_accuracy 67.6%/111-sam arguably justifies), and the directional gate blocks its 26.5%-accurate BUYs — leaving quorum inflation (lines 4241-4254/4492-4515) on a single ticker as the main unmitigated effect.

### [P1 → P2 · design · conf 0.75] Entire per-horizon machinery (HORIZON_SIGNAL_WEIGHTS, IC multiplier, horizon blacklists, 3h caps) is dead on the production path — horizon is always None
`portfolio/signal_engine.py:2718`

The only production caller of generate_signal is main.py:512, which never passes horizon (verified: no other non-test callers in repo; `git log -S 'horizon=' -- portfolio/main.py` is empty, so it never did). With horizon=None: (a) `ic_cache = _get_ic_data(horizon) if horizon else None` (line 2718) — the whole IC-based weight multiplier subsystem (2026-04-18, lines 2276-2329) never runs; (b) `_get_horizon_weights(None)` returns {} (lines 1677-1678) — HORIZON_SIGNAL_WEIGHTS including the 2026-06-06 1d retune (lines 1550-1578, e.g. crypto_evrp 1.5 from a suspicious 99.1%/233-sam stat) has zero production effect; (c) horizon-specific entries in _TICKER_DISABLED_BY_HORIZON (3h/4h/12h blocks, lines 929-1093) never apply — only '_default' does, despite the comment at 2516-2519 claiming 'one vote reused across 3h/4h/12h/1d consensus'; (d) short_horizon slow-signal gating (4137-4141), 3h RSI thresholds (3346-3348) and CONFIDENCE_CAP_3H (4667-4669) are unreachable. Substantial tuning work (including the 2026-06-06 edit) is being applied to dead configuration while operators believe it shapes live consensus.

**Suggested fix:** Either pass an explicit horizon from main.py (e.g. '1d' for the primary loop signal) so weights/IC engage, or document that horizon-specific config is backtest/replay-only and stop tuning it as if live. Add a startup log line stating whether horizon weighting/IC is active.

**Skeptic verdict (real):** Confirmed mechanically: main.py:512 is the only production generate_signal caller (repo-wide grep) and passes no horizon (never has, per git log -S); with horizon=None the IC multiplier (signal_engine.py:2718→2811 'if ic_global'), _get_horizon_weights (1677-78 returns {}), horizon-specific blacklist entries (1122-23 returns _default only), 3h RSI thresholds (3346), 3h slow-signal gate (4137), and CONFIDENCE_CAP_3H (4667) are all inert — only backtester.py:143 and scripts/replay_consensus.py:147 exercise them. The 2026-06-06 retune (commit a5319fb9) tuned the 1d fallback dict production never reads, and outcome_tracker.py:587 invalidates horizon-weight caches the live path never consults. P2 not P1: this is wasted tuning + misleading comments (e.g. 2516-2519), not a trade-safety defect — the dead 3h/4h config targets per-horizon consensus paths that have never existed live, and the live 1d path still gets correct 1d accuracy gating via acc_horizon fallback (4291).

### [P2 · doc-drift · conf 0.90] CLAUDE.md 'Active 21 signals' list is stale — at least 5 listed-active signals are disabled, and 2 listed per-ticker overrides were removed
`CLAUDE.md:1`

CLAUDE.md lists as active: Crypto EVRP (#15), ADX Regime Switch (#17), Choppiness Regime Gate (#19), BOCPD Regime Switch (#20), Vol Ratio Regime (#21). All five are in tickers.py DISABLED_SIGNALS: crypto_evrp re-disabled 2026-05-26 (tickers.py:94), vol_ratio_regime 2026-06-01 (:102), adx_regime_switch 2026-06-01 (:230), choppiness_regime_gate 2026-06-01 (:234), bocpd_regime_switch 2026-06-01 (:241). The per-ticker overrides section also lists Williams VIX Fix → XAU/XAG (removed 2026-05-31, signal_engine.py:727-729) and Credit Spread Risk → BTC/ETH (removed 2026-05-26, signal_engine.py:733-735); only ('ml','ETH-USD') and ('realized_skewness','XAU-USD') remain in _DISABLED_SIGNAL_OVERRIDES (lines 723-736). This matters more than usual doc rot: Layer 2 trading sessions read CLAUDE.md as project context, so the trading agent reasons from a wrong picture of which signals vote. Amihud accuracy is also quoted as 68.0%/225 sam vs the 1d weight comment's 69.6%/618 — multiple snapshots of different dates.

**Suggested fix:** Regenerate the active-signal and per-ticker-override sections of CLAUDE.md from tickers.py DISABLED_SIGNALS + signal_engine._DISABLED_SIGNAL_OVERRIDES; ideally add a script that diffs CLAUDE.md's list against code and alerts on drift.

### [P2 · doc-drift · conf 0.85] Circuit-breaker comments and .claude/rules claim 6pp relaxation/41% floor and 10K-sample tier; constants say 2pp/45% and 7000
`portfolio/signal_engine.py:898`

_GATE_RELAXATION_MAX = 0.02 (line 903, '0.47 -> 0.45') was tightened from 0.06 in commit f4da38b3, but: the module comment directly above (line 898) still says 'relaxing the gate by up to 6pp (to 41% floor)'; the _compute_gate_relaxation docstring (lines 2170-2174) still references 'the 41% relaxed gate'; and .claude/rules/signals.md states 'max relaxation (6pp) ... effective floor 41%'. Similarly _ACCURACY_GATE_HIGH_SAMPLE_MIN = 7000 (line 448) while the comment block at lines 441-447 says 'raised high-sample min 5000 -> 10000', the in-function comments at 2074 and 2744-2747 say '10K+/10000+ samples', and .claude/rules/signals.md says '10,000+ samples gate at 50%'. Anyone (human or Layer 2 agent) reasoning from comments/rules will mis-predict gate behavior by 4pp and 3000 samples in a system where gate boundaries decide which signals vote.

**Suggested fix:** Update line 898, the _compute_gate_relaxation docstring, lines 2074/2744, and .claude/rules/signals.md to the current constants (2pp max, 0.45 floor; 7000-sample tier) — or derive the prose from the constants in one place.

### [P2 · bug · conf 0.80] Metals seasonal BUY multiplier applied AFTER the global 0.80 confidence cap — can emit confidence above the documented cap
`portfolio/signal_engine.py:4679`

Line 4664 enforces `conf = min(conf, 0.80)` with the rationale that >80% confidence is anti-correlated with accuracy; lines 4667-4669 add the 3h cap. The seasonal modifier (lines 4679-4691, added 2026-05-29) then multiplies XAG/XAU BUY confidence by up to 1.15 (Jan-Mar) AFTER both caps, so final confidence can reach ~0.92 — violating the calibration-motivated cap. With realistic post-compression values (calibration Stage 7 maps 1.0 to ~0.685, linear-factor boost x1.10 → ~0.75), seasonal x1.15 yields ~0.87 > 0.80. These confidences feed Layer 2 context and the metals swing-trader threshold logic (data/metals_swing_config.py MIN_BUY_CONFIDENCE), so the inflation can change real-money entry decisions in Jan-Apr.

**Suggested fix:** Move the seasonal multiplier block above the `conf = min(conf, 0.80)` cap (and the 3h cap), or re-clamp after applying it.

### [P2 · bug · conf 0.80] _compute_applicable_count wrongly excludes ministral for non-crypto tickers although ministral votes on all tickers
`portfolio/signal_engine.py:1723`

Lines 1722-1724 skip 'ministral' for non-crypto with the comment 'ministral (CryptoTrader-LM) only runs for crypto', but the actual ministral block (lines 3643-3691, comment 'all tickers — crypto, stocks, metals') runs for every ticker when GPU is available. So _total_applicable undercounts by 1 for XAU/XAG/MSTR. _total_applicable feeds the Stage-5b ensemble-entropy guard (lines 3229-3241): the HOLD bucket `_ent_hold = total - buy - sell` is too small, biasing normalized entropy and occasionally flipping the 0.8/0.9 penalty thresholds on metals/MSTR — i.e., wrong confidence penalties on real instruments. It also makes the dashboard/Layer-2 'X of Y signals' denominators wrong. Related doc drift: .claude/rules/signals.md says applicable counts crypto=29/stocks=25/metals=27 while CLAUDE.md says 19/15/17 — neither matches the dynamic computation.

**Suggested fix:** Delete the ministral special-case in _compute_applicable_count (or gate it on the same condition the vote uses). Reconcile the applicable-count numbers in CLAUDE.md and .claude/rules/signals.md with the dynamic count.

### [P2 · bug · conf 0.70] MIN_VOTERS_METALS=2 is nullified by Stage-4 dynamic MIN_VOTERS (floor 3) — 2-voter metals consensus can never trade
`portfolio/signal_engine.py:3197`

MIN_VOTERS_METALS was lowered to 2 on 2026-05-11 (line 1180, rationale at 4258-4261: 'persistence filter leaves only 2 voters in steady-state on XAG; the old 3-voter floor produced 0 trades in 20 days'). But apply_confidence_penalties Stage 4 (lines 3193-3203) force-HOLDs any non-HOLD action when post-persistence voters < _dynamic_min_voters_for_regime(regime), and that helper (lines 2150-2155) returns minimum 3 (trending), 4 (high-vol), 5 (ranging) with no asset-class awareness. apply_confidence_penalties is called unconditionally at line 4555. So the exact target case of the 2026-05-11 change — a 2-voter metals slate — is always forced to HOLD at Stage 4; metals trades since then flow only because soft dead-zone votes (Stage 2) push counts to >=3. The lowered floor is effectively dead code, and the documented intent is silently unmet.

**Suggested fix:** Make _dynamic_min_voters_for_regime asset-aware (accept ticker, floor at MIN_VOTERS_METALS for metals) or remove MIN_VOTERS_METALS and document that the effective metals floor is 3-5 by regime.

### [P2 · design · conf 0.65] Soft-vote dampening is scale-invariant — an all-soft same-direction slate still produces full directional confidence, contradicting the Fix B contract
`portfolio/signal_engine.py:2876`

The 2026-05-11 'Codex Fix B' docstring (lines 2406-2418) claims soft dead-zone votes are scaled so 'an all-soft slate could [no longer] produce full directional confidence' (3 x 0.18 ≈ 0.54 < 1.0). But the final confidence is `buy_weight / (buy_weight + sell_weight)` (lines 2894-2903), which is invariant to uniform scaling: three soft BUY votes with weights 0.18 each yield buy_conf = 1.0, identical to three strong votes. ema/macd/bb soft votes are CORE_SIGNAL_NAMES members, so an all-soft slate passes the core gate and MIN_VOTERS, producing weighted_conf=1.0 from pure dead-zone drift; only the unanimity penalty and calibration compression pull it back (to ~0.56 — above the metals trade floor). The dampening only changes outcomes when soft and strong votes oppose each other. The 0.30 per-vote cap (line 2881) does not fix this either.

**Suggested fix:** Make soft votes reduce absolute confidence, e.g. carry total_weight into the confidence calc (conf = winner_weight / max(total_weight, K) for some strong-vote-scale K), or cap weighted_conf when all participating voters are soft.

### [P2 · reliability · conf 0.50] backfill_outcomes rewrite can corrupt signal_log.jsonl if log rotation runs during the unlocked processing phase
`portfolio/outcome_tracker.py:533`

backfill_outcomes uses a 3-phase pattern: snapshot under the sidecar lock (lines 383-415), slow HTTP backfill with the lock RELEASED, then rewrite under the lock (lines 533-572). Phase 3 assumes the only possible interleaved mutation is an append: it copies the first head_end_offset bytes of the CURRENT file verbatim and treats bytes past snapshot_size as concurrent appends. But signal_log.jsonl is also subject to rotate_jsonl (log_rotation.py:46 policy, invoked hourly from main.py:407 rotate_all), which holds the same lock only for its own duration and REPLACES the file with a shorter tail-keep version. If rotation lands during phase 2 (backfill's HTTP loop can run minutes), phase 3 then: copies head bytes from the rotated file at a stale offset (can cut mid-line → malformed JSONL), re-appends the stale parsed entries (resurrecting rotated-away data and duplicating kept entries), and computes concurrent_tail_bytes against a stale snapshot_size (silently dropping post-rotation appends). The 2026-05-11 rotation-race fix covered append-vs-rotate, not backfill-vs-rotate.

**Suggested fix:** Record an identity marker at snapshot time (e.g. inode/file-id or first-line hash + size) and in phase 3 abort/retry the backfill if it changed; or take a cross-process advisory 'rewrite in progress' flag that rotate_jsonl checks before rotating signal_log.

### [P3 · doc-drift · conf 0.85] Persistence-filter comments stale after 2026-05-27 crypto change (says 'returns 1 for metals+crypto'; crypto is 2)
`portfolio/signal_engine.py:666`

_PERSISTENCE_CYCLES_BY_ASSET (lines 629-633) sets CRYPTO: 2 (raised 2026-05-27 per the comment at 626-627), but the in-function comment at lines 666-668 inside _apply_persistence_filter still says '_persistence_cycles_for() returns 1 for metals+crypto, 2 for stocks'. Anyone debugging crypto vote debounce (e.g. the ETH BUY→HOLD flip noise the change targeted) from the call-site comment will reason with the wrong threshold. Pure comment drift; behavior is correct.

**Suggested fix:** Update the comment at lines 666-668 to 'returns 1 for metals, 2 for crypto and stocks'.

### [P3 · quality · conf 0.80] CRITICAL-2 (ticker="" dispatch) assessment: effectively mitigated — guard logs a warning, sole production caller always passes a ticker; residual degradation is cosmetic
`portfolio/signal_engine.py:3301`

Assessment of the known pre-existing bug: the 2026-04-17 guard (lines 3294-3306) warns on empty/None ticker, and the only production call site is main.py:512 which passes ticker=name from the ticker list (never empty). Remaining empty-ticker behavior is graceful but quietly different: phase tracking and persistence filtering are skipped (lines 3332-3334, 663), skip_gpu is forced False (3338), fear_greed falls back to a shared cache key (3425) and the BTC-keyed fear-streak update can run for ticker=None (3433), per-ticker blacklists/overrides no-op, and _shadow_llm_runs_now returns False. None of these can fire in the current production wiring. Risk is regression-only: a future caller (e.g. prophecy/ or a probe script) passing '' would silently lose persistence + per-ticker gating while still emitting a tradeable-looking consensus, with only a log warning.

**Suggested fix:** Promote the warning to a hard normalization: `if not ticker: ticker = None` plus raise ValueError when config indicates production mode (or assert in main loop wiring), and add a regression test asserting generate_signal('' ) routes identically to None.

### [P3 · bug · conf 0.60] blend_accuracy_data double-counts the recent window in directional totals (recent is a subset of all-time)
`portfolio/accuracy_stats.py:985`

For directional stats, blend_accuracy_data sums `total_buy = at_v + rc_v` and sample-weight-averages buy/sell accuracy across the all-time and recent dicts (lines 970-989). But `recent` is computed from the same signal log with only a time cutoff (signal_accuracy_recent → signal_accuracy(since=...)), so every recent sample is already contained in all-time: the summed totals overstate directional sample counts by the size of the recent window, and the 'blended' directional accuracy double-weights the last 7 days relative to the documented 70/30 scheme. Practical effect: the directional gate (_DIRECTIONAL_GATE_MIN_SAMPLES=30) and directional-rescue threshold (30) trigger earlier than the real sample count justifies, and BUG-182 direction-specific weights use a skewed estimate. The comment ('Merge directional keys from the larger-sample source per key') doesn't match the implementation either.

**Suggested fix:** Blend directional accuracy with the same 70/30 (or fast 90/10) weights used for overall accuracy, and report total_buy/total_sell as max(at, rc) like the overall `total` field — not the sum.

## Signal modules

### [P1 → P1 · bug · conf 0.90] _expiry_proximity votes BUY almost every cycle because Deribit nearest expiry is always 0-1 days (daily expiries)
`/mnt/q/finance-analyzer/portfolio/signals/crypto_macro.py:187`

The expiry-proximity sub-indicator returns BUY whenever days_to_expiry <= 1 (both the quarterly and non-quarterly branches at lines 187-192 return BUY, making the is_quarterly check dead logic). Deribit lists DAILY BTC/ETH option expiries, so the nearest expiry chosen by crypto_macro_data._fetch_deribit_options is virtually always 0-1 days away. Verified live: data/agent_summary.json shows expiry_days_to_expiry=1, expiry_expiry_date=11JUN26 (a daily, tomorrow). This gives the composite a permanent BUY vote, producing massive structural long bias: signal_log.db tallies over recent snapshots show ETH-USD crypto_macro = 1459 BUY / 297 HOLD / 69 SELL and BTC-USD = 1236 BUY / 817 HOLD / 122 SELL. In the current downtrend this is the primary driver of the accuracy collapse logged in data/critical_errors.jsonl (crypto_macro 41.2%->23.8%, ETH-USD::crypto_macro 38.3%->19.5%). The module docstring (line 11, 'votes HOLD but flags the risk') also contradicts the code, which votes BUY.

**Suggested fix:** Only emit the post-expiry-relief BUY for genuine quarterly expiries with meaningful OI, or restrict the sub-indicator to monthly/quarterly expiries (filter expiry list before selecting nearest). At minimum make days<=1 on a non-quarterly daily expiry return HOLD as the docstring describes.

**Skeptic verdict (real):** Confirmed. crypto_macro.py:187-192 returns BUY for days<=1 regardless of is_quarterly (dead for the vote), and signal_utils.majority_vote (signal_utils.py:93-96) treats HOLD as abstention, so one permanent BUY sub-vote wins whenever the other subs abstain — live agent_summary.json shows expiry_days_to_expiry=1/11JUN26 (a daily, also misclassified is_quarterly=true via month-substring match), signal_log.db tallies BTC 2491 BUY/218 SELL and ETH 2680 BUY/69 SELL, and accuracy_cache.json 1d shows 4997 BUY vs 274 SELL samples (skew 0.896) with recent accuracy 38.1% matching the critical_errors.jsonl collapse entries (lines 1000-1001). Two caveats: the function docstring (crypto_macro.py:166) explicitly documents the 0-1d BUY bias so it is intentional-but-wrong design (built for monthly/quarterly structure), not a docstring/code contradiction; and 'primary driver of the accuracy collapse' is overstated since rsi/bb/drift_regime_gate collapsed in the same alerts.

### [P1 → P2 · reliability · conf 0.85] Exchange netflow sub-signal silently dead since 2026-04-10 — BGeometrics netflow feed broken, no alert
`/mnt/q/finance-analyzer/portfolio/crypto_macro_data.py:320`

data/exchange_netflow_history.jsonl contains exactly ONE entry (ts 2026-04-10, file mtime Apr 10), and data/onchain_cache.json (fresh, ts Jun 10) has NO 'netflow' key at all — _fetch_exchange_netflow in portfolio/onchain_data.py:173-179 has been returning None for two months (the /v1/exchange-netflow endpoint returns empty/error; its failure is only a warning inside _fetch_all_onchain, and the merged cache simply omits the key). get_exchange_netflow_trend therefore never appends history (line 336 guard), len(history) < 3 stays true forever, and the netflow sub-indicator returns trend='insufficient_data' -> permanent HOLD. One of crypto_macro's five sub-indicators (and the only on-chain one) has been dead for two months with zero surfacing in critical_errors — exactly the silent-failure pattern the project's loop-contract work targets. mvrv_zscore and liquidations are also null in the same cache, suggesting broader BGeometrics endpoint rot.

**Suggested fix:** Add a staleness invariant: if NETFLOW_HISTORY_FILE's newest entry is older than N days while the signal is active, log a critical_errors entry (category data_feed_stale). Verify the BGeometrics /v1/exchange-netflow endpoint/response shape and repair or replace the feed; until then mark the sub-indicator disabled rather than 'insufficient_data'.

**Skeptic verdict (real):** Confirmed: data/exchange_netflow_history.jsonl has exactly one entry (ts 1775837235 = 2026-04-10, mtime Apr 10), fresh data/onchain_cache.json (ts Jun 10) has no 'netflow' key while mvrv_zscore/long/short_liquidations are present-but-null, _fetch_exchange_netflow (onchain_data.py:173-179) returns None silently with only a warning in _fetch_all_onchain (onchain_data.py:235-236), and len(history)<3 keeps get_exchange_netflow_trend at 'insufficient_data' -> permanent HOLD; grep found zero netflow/bgeometrics entries in critical_errors.jsonl and no staleness monitoring anywhere. Downgraded to P2 because a permanent-HOLD sub-vote is an abstention under majority_vote (no directional bias), affects only the BTC leg of one already-degraded signal, and the impact is silent feature loss rather than wrong trades.

### [P1 → P2 · design · conf 0.80] Max pain, nearest PCR and options gravity all computed on the thin 0-1 DTE daily expiry chain
`/mnt/q/finance-analyzer/portfolio/crypto_macro_data.py:108`

_fetch_deribit_options selects the single nearest unexpired expiry (lines 108-123) and computes max_pain, nearest_call/put OI and nearest_pcr from that chain only. Because Deribit has daily expiries, this is a 0-1 DTE chain with thin, unrepresentative open interest, so max pain and PCR are noisy day-to-day. Downstream, crypto_macro._options_gravity's 'gravity weakens further from expiry' gate (crypto_macro.py:61, days > 7 -> HOLD) can never trip, so the gravity sub-indicator is always active on this noisy data, and _put_call_sentiment prefers nearest_pcr (crypto_macro.py:86) over the much more stable total_pcr. Three of five sub-indicators are therefore driven by a data slice the design (written for monthly/quarterly expiry structure) never intended.

**Suggested fix:** Compute max pain and PCR on the nearest MONTHLY/quarterly expiry (or OI-weighted across expiries with a minimum-OI threshold), and let _put_call_sentiment use total_pcr as primary. Keep days_to_expiry of the chosen chain so the >7d gravity gate becomes meaningful again.

**Skeptic verdict (real):** Confirmed in code: crypto_macro_data.py:108-127 selects the single nearest unexpired expiry and computes max_pain/nearest_pcr/nearest OI from that chain only; with Deribit dailies (live days_to_expiry=1) the days>7 gravity gate at crypto_macro.py:60-63 is unreachable and _put_call_sentiment prefers nearest_pcr over total_pcr (crypto_macro.py:86). Downgraded to P2 because, unlike finding 1, this produces noise rather than structural directional bias (gravity/PCR can vote either way), the module's confidence is capped at 0.7 and it is one of 21 voters behind the 47% accuracy gate; the dominant harm is already captured by the expiry_proximity finding.

### [P2 · doc-drift · conf 0.80] ADX module documented as transition-edge detector but 2 of 3 sub-indicators vote directionally in every stable trend
`/mnt/q/finance-analyzer/portfolio/signals/adx_regime_switch.py:63`

Module docstring (lines 12-14) says: 'When ADX crosses above 25 (entering trend): follow +DI/-DI... Stable regime = HOLD (no transition edge).' The code does not implement crossings: _adx_regime (lines 63-67) votes BUY/SELL whenever ADX > 25 regardless of transition, and _di_spread (lines 109-113) votes direction whenever |+DI - -DI| > 10 in any regime. Only _adx_momentum actually detects transitions. So in any established trend the composite persistently emits direction — it is a trend-follower, not the transition meta-signal it documents (and that was presumably validated when it was activated as signal #17 at 67% on 182 samples). Future tuning based on the docstring's mental model will misfire.

**Suggested fix:** Either implement actual cross-detection (ADX crossing the 25 threshold within the last N bars) for _adx_regime, or update the docstring to describe the persistent trend-following behavior so the design intent matches the code.

### [P2 · bug · conf 0.75] consecutive_negative counts 6-hour samples but crypto_macro treats them as days — BUY threshold fires ~4x too easily
`/mnt/q/finance-analyzer/portfolio/crypto_macro_data.py:414`

_append_netflow_history appends at most once per 6 hours (line 420, 21600s), so consecutive entries are 6h apart. But crypto_macro.py:34 defines _NETFLOW_ACCUM_DAYS = 5 documented as '5+ consecutive negative netflow = strong signal' interpreted as days, and _exchange_netflow_signal (crypto_macro.py:150-155) votes BUY at consecutive_neg >= 5 — which is only ~30 hours of (likely identical, since BGeometrics netflow is a daily metric sampled 4x/day) negative readings, and the 'accumulation + consecutive_neg >= 3' branch needs only ~18h. When the netflow feed is repaired (see related finding) this unit mismatch will reintroduce structural BUY bias into an already BUY-skewed composite.

**Suggested fix:** Either append once per day (cadence matching the daily metric) or deduplicate to calendar days when counting consecutive_negative; align _NETFLOW_ACCUM_DAYS semantics with the actual sample cadence.

### [P2 · design · conf 0.75] Systemic: module-level unlocked TTL caches in 8-thread signal loop — dogpile duplicate yfinance/FRED fetches, bypassing shared_state._cached
`/mnt/q/finance-analyzer/portfolio/signals/gold_platinum_ratio_risk.py:40`

Main loop computes tickers in a ThreadPoolExecutor with up to 8 workers (portfolio/main.py:572,620). At least 7 signal modules roll their own module-level dict caches with no lock: gold_platinum_ratio_risk.py:40 (_CACHE), copper_gold_ratio.py:43, btc_etf_flow.py:47, breakeven_inflation_momentum.py:50, credit_spread.py:53, hash_ribbons.py:51, crypto_evrp.py:53. On simultaneous cache miss, multiple worker threads each fire the expensive fetch (gold_platinum downloads 2x 2-year daily series per miss; copper_gold similar) — duplicated yfinance/FRED traffic and added cycle latency, the exact dogpile problem shared_state._cached already solves with _cache_lock and stale-fallback (shared_state.py:37-51, BUG-166). gold_platinum_ratio_risk additionally has no negative-result caching: when the fetch fails it returns None without recording the failure, so every compute call across 5 tickers x 7 timeframes re-attempts the full download during an outage. Other modules (metals_cross_asset, metals_vrp, stablecoin_supply_ratio, crypto_overnight_sentiment) correctly use locks or _cached — the codebase has two divergent caching idioms.

**Suggested fix:** Migrate the unlocked module caches to shared_state._cached (it already provides locking, dogpile prevention, and bounded staleness fallback), or at minimum add a module lock plus a short negative-cache TTL on fetch failure.

### [P2 · design · conf 0.70] 'Volume Confirm' sub-indicator votes BUY on any high relative volume, including crash bars
`/mnt/q/finance-analyzer/portfolio/signals/amihud_illiquidity_regime.py:107`

Sub-indicator 3 (lines 107-112) votes BUY when rvol > 1.3 and SELL when rvol < 0.6, with no reference to price direction at all. High relative volume occurs equally in panic selloffs, where this contributes a BUY vote, and in count_hold=False majority voting (line 115) a single direction-less vote can fully determine the composite action when the other two subs are HOLD (yielding action=BUY at confidence 1.0 capped to 0.7). The name 'Volume Confirm' implies confirmation of a directional move; the implementation is an unconditional volume->long mapping. The signal's good aggregate accuracy (68%, 225 samples) may mask this sub-indicator's contribution in down-moves.

**Suggested fix:** Condition the volume vote on bar direction (e.g., rvol > 1.3 AND close > open => BUY; rvol > 1.3 AND close < open => SELL), or demote it to a confidence multiplier on the other two sub-indicators instead of an independent directional vote.

### [P2 · design · conf 0.55] Drift regime gate fades persistent drifts with no exhaustion/trend-strength guard — catching falling knives in the current regime
`/mnt/q/finance-analyzer/portfolio/signals/drift_regime_gate.py:55`

_drift_fraction votes contrarian: BUY whenever <40% of the trailing 63 closes were positive (lines 55-58), with _drift_velocity BUY on accelerating down-drift. In a persistent multi-week downtrend the module emits BUY bar after bar with no confirmation the decline is exhausting (the _price_vs_sma sub only requires price 1.5 ATR below SMA — also true throughout a sustained decline). critical_errors.jsonl 2026-06-08/09 shows exactly this failure: drift_regime_gate 58.6%->43.5%, BTC-USD:: 50.2%->26.2%, ETH-USD:: 66.6%->35.0% as the market trended down. This is design behavior, not a coding bug, but the module has no defense for the regime where its premise (drift mean-reverts) breaks, and unlike the cited paper (which gates OTHER signals during drift regimes) it trades the fade directly.

**Suggested fix:** Add an exhaustion condition before contrarian BUY (e.g., require down-drift fraction to stop falling / velocity to turn positive, or RSI divergence), or use the drift classification as a gate on other voters per the cited research instead of a standalone contrarian voter.

### [P3 · quality · conf 0.90] OPTIONS_TTL defined at end of module with false 'imported from data module' comment, duplicating crypto_macro_data.OPTIONS_TTL
`/mnt/q/finance-analyzer/portfolio/signals/crypto_macro.py:281`

Line 280-281: '# Cache TTL imported from data module' / 'OPTIONS_TTL = 900'. It is not imported — it is an independent local constant that happens to equal crypto_macro_data.OPTIONS_TTL (crypto_macro_data.py:31), and it sits 53 lines below its only use site (line 228 inside compute_crypto_macro_signal). Works at runtime because the function executes after module load, but the misleading comment plus duplication means changing the TTL in crypto_macro_data.py silently leaves the signal-side _cached TTL at 900, and a future refactor moving the constant reference to module top could NameError.

**Suggested fix:** Replace with 'from portfolio.crypto_macro_data import OPTIONS_TTL' at the top of the module and delete the trailing definition.

### [P3 · quality · conf 0.70] NaN choppiness reported as regime 'choppy'; TREND/NEUTRAL branches duplicated
`/mnt/q/finance-analyzer/portfolio/signals/choppiness_regime_gate.py:102`

When _compute_choppiness yields NaN (short/degenerate data), _choppy_gate returns 'HOLD' (line 41-42), and the early-return block at lines 102-115 then labels indicators['regime'] = 'choppy' — downstream consumers and accuracy analysis cannot distinguish 'genuinely choppy market' from 'indicator not computable'. Additionally lines 120-126: the TREND and NEUTRAL branches are byte-identical except for the 0.7 confidence multiplier, inviting divergence bugs on future edits.

**Suggested fix:** Return regime='unknown' (or 'insufficient_data') when chop_val is NaN, and collapse the duplicated branches to a single majority_vote call with a conditional multiplier.

## Portfolio & risk

### [P1 → P2 · design · conf 0.65] Per-ticker cooldown blocks SELL/exits, and loss escalation makes exits hardest exactly after losses
`/mnt/q/finance-analyzer/portfolio/trade_guards.py:147`

Guard 1 (trade_guards.py:130-167) emits severity='block' for ANY action within the cooldown window — it never distinguishes BUY from SELL. record_trade() stamps the ticker on both BUY and SELL, so a fresh BUY immediately puts the position's own exit into a 30-min block; with LOSS_ESCALATION (line 29) the multiplier reaches 8x = 240 min after 4 consecutive losses. agent_invocation.py:1016-1038 skips the whole Layer 2 invocation when the trigger ticker is blocked for both strategies, and the [BLOCK] message is injected into the Layer 2 prompt otherwise. Net effect: after a BUY that goes wrong, the system cannot decide to cut the position for up to 4 hours — the risk-control inverts (escalating losses lengthen the time you are locked into a losing position). This contradicts the docstring's intent ('overtrading prevention') by also preventing loss-cutting.

**Suggested fix:** In check_overtrading_guards Guard 1, downgrade severity to 'warning' (or skip entirely) when action == 'SELL' and the strategy currently holds shares of the ticker — cooldowns should gate position-opening, not position-closing. Keep block semantics for BUY.

**Skeptic verdict (real):** Confirmed: Guard 1 (trade_guards.py:130-167) has no action check (Guard 3 at :190 is explicitly BUY-only, so the omission is real, not intentional — no comment/doc claims SELL-blocking is by design), record_trade stamps BUY at :273 via agent_invocation.py:1440-1446, loss multiplier scales the same cooldown (:134-135, up to 8x=240min), and agent_invocation.py:1031-1038 skips the whole invocation when the trigger ticker is blocked for both strategies — which happens exactly when the signal flips to SELL (reporting.py:502 computes warnings from the live signal action). Severity downgraded to P2: only the simulated Patient/Bold portfolios use trade_guards (metals/real-money paths don't import it), the block has no execution-time enforcement so Layer 2 invocations triggered by OTHER tickers can still exit the position (guard text is advisory-only in the prompt, agent_invocation.py:1124-1125), and the common window is 30 min — the 4h lockout needs 4 consecutive losses per strategy.

### [P2 · bug · conf 0.80] _DEFAULT_STATE holdings/transactions are shared mutable references across all default-state loads
`/mnt/q/finance-analyzer/portfolio/portfolio_mgr.py:75`

_validated_state does `result = {**_DEFAULT_STATE, **loaded}` and the missing-file path does `{**_DEFAULT_STATE, "start_date": ...}` (portfolio_mgr.py:74-75, 180). These are shallow copies: when the loaded state lacks 'holdings'/'transactions' (fresh bootstrap, corrupt file falling through to defaults, hand-edited file missing a key), the returned dict aliases the module-level _DEFAULT_STATE["holdings"] dict and _DEFAULT_STATE["transactions"] list. Any in-place mutation (update_state's mutate_fn, or a caller appending a transaction to a load_state() result) permanently contaminates the module global for the process lifetime AND leaks across portfolios: if Patient and Bold are both in the default path, they literally share the same transactions list, so a Patient trade appears in the Bold file on its next save. This bites precisely on the corruption-recovery path that the 2026-06-01 quarantine work hardened.

**Suggested fix:** Deep-copy on every use: `result = {**copy.deepcopy(_DEFAULT_STATE), **loaded}` or build the default dict fresh inside a _default_state() factory function instead of a module constant.

### [P2 · bug · conf 0.75] compute_probabilistic_stops: dead 'metals' branch (metals get stock annualization) and invalid 'stock' session key (MSTR gets warrant session)
`/mnt/q/finance-analyzer/portfolio/risk_management.py:466`

inst_type is assigned only the values 'crypto', 'warrant', or 'stock' (risk_management.py:452-457), but line 466 checks `inst_type in ("crypto", "metals")` — 'metals' can never occur. XAU/XAG positions therefore annualize with 252 trading days instead of 365, underestimating volatility by ~20% (sqrt(365/252)) and so underestimating stop-hit probability on the real-money-adjacent metals book. Separately, 'stock' is not a valid session_calendar key (SESSIONS has 'warrant', 'stock_se', 'stock_us', 'crypto'; session_calendar.py:176 falls back to SESSIONS['warrant']), so MSTR's remaining-session-minutes uses the Avanza warrant window (08:15-21:55 CET) instead of US market hours — stop-hit probabilities for MSTR are simulated over the wrong remaining horizon.

**Suggested fix:** Line 466: check `inst_type in ("crypto", "warrant")` (or map XAU/XAG to a 365-day class explicitly). Map US stocks to 'stock_us' before calling remaining_session_minutes.

### [P2 · reliability · conf 0.55] Drawdown peak has no sanity bound — one glitched history row permanently inflates peak and can hard-block all Layer 2 invocations
`/mnt/q/finance-analyzer/portfolio/risk_management.py:283`

check_drawdown's peak comes from _streaming_max over append-only portfolio_value_history.jsonl with zero outlier rejection (risk_management.py:283, 96-98). History rows are computed from agent_summary price_usd × fx_rate; a single bad print (exchange glitch, fx briefly wrong-but-in-band [7,15], manual edit) writes an inflated value that becomes the peak forever — it can never be exceeded by real values and the byte-offset cache plus full-file streaming re-find it every cycle. A 2x glitch makes current_drawdown_pct read ~50%+, which crosses agent_invocation's _DRAWDOWN_BLOCK_PCT=50 and silently blocks every Layer 2 invocation for that portfolio until someone surgically edits the history file. The NaN/Inf guard (line 292) catches non-finite values but not finite-but-wrong ones.

**Suggested fix:** Bound peak plausibility, e.g. reject history values > N× the max of (initial_value, rolling median of recent entries) when computing peak, or log a critical_errors entry when current_drawdown_pct > block threshold persists so the condition is surfaced rather than just skipping invocations.

### [P2 · bug · conf 0.55] compute_portfolio_var takes agent_summary fx_rate at face value — P1-15 sanity-band fix was not applied here
`/mnt/q/finance-analyzer/portfolio/monte_carlo_risk.py:408`

monte_carlo_risk.py:408 does `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)`. The fallback only covers a missing key; a present-but-bogus value (the stale fx_rate=1.0 case that adversarial review 05-01 P1-15 fixed in risk_management._resolve_fx_rate, which explicitly rejects out-of-band values and notes 'even if a stale agent_summary still embeds 1.0 itself') passes straight through. All *_sek VaR/CVaR/exposure outputs (lines 485-490) would be ~10x understated, feeding wrong risk numbers into the Layer 2 context built by reporting.py:536. The codebase already has the validated resolver; this module just doesn't use it.

**Suggested fix:** Replace the raw .get() with `from portfolio.risk_management import _resolve_fx_rate; fx_rate = _resolve_fx_rate(agent_summary)` (or move _resolve_fx_rate to fx_rates.py and use it in both places).

### [P3 · quality · conf 0.85] check_overtrading_guards ignores its 'portfolio' argument; position rate limit counts adds to existing positions as 'new positions'
`/mnt/q/finance-analyzer/portfolio/trade_guards.py:103`

The `portfolio` parameter of check_overtrading_guards (trade_guards.py:103) is never referenced in the body — get_all_guard_warnings threads patient_pf/bold_pf through for nothing, implying the guard checks position context when it doesn't. Consequently Guard 3 ('Max N NEW positions per window', line 189-229) counts every BUY recorded by record_trade, including scale-ins to an already-held ticker, because neither record_trade nor the guard consults holdings. With patient_position_limit=1 per 8h, a single add-on buy exhausts the window for genuinely new positions, and the misleading parameter hides that the data needed to distinguish the cases is already being passed in.

**Suggested fix:** Either use the portfolio arg (skip Guard 3 counting when portfolio['holdings'][ticker]['shares'] > 0 at BUY time, recorded via a flag in record_trade) or remove the parameter and rename the guard message to 'BUY rate limit' to match actual semantics.

### [P3 · design · conf 0.75] Zero-correlation fallback is claimed 'conservative for VaR' but understates joint tail risk; mixed-data path skips priors entirely
`/mnt/q/finance-analyzer/portfolio/monte_carlo_risk.py:137`

estimate_correlation_matrix's docstring (monte_carlo_risk.py:53-56) claims defaulting to zero correlation 'is conservative for VaR'. For a long-only book (holdings are long-only here) the opposite is true: independence shrinks the loss tail vs. positively-correlated assets, so VaR/CVaR are UNDERstated — e.g. BTC+ETH treated as independent halves the diversified tail loss vs. their ~0.8 prior. Compounding this, build_correlation_matrix (lines 137-143) uses the empirical path for ALL pairs as soon as any 2 tickers have >=30 observations, so pairs with <20 shared observations get corr=0 even when CORRELATION_PRIORS has a strong prior. Production currently calls compute_portfolio_var without historical_returns (reporting.py:536) so the priors path is used, which limits live impact — but the doc claim is directionally wrong and the mixed path is a trap. Also, alignment uses ri[:min_len] (head of series) rather than the most recent tail.

**Suggested fix:** Fix the docstring; in estimate_correlation_matrix fall back to _get_prior_correlation (not 0) for pairs with insufficient data; align series by tail (ri[-min_len:]).

### [P3 · doc-drift · conf 0.70] Doc-drift: 'stream the FULL history to find the true historical peak' is no longer true — 90-day rotation caps the peak window and cache resets to floor on shrink
`/mnt/q/finance-analyzer/portfolio/risk_management.py:280`

The A-PR-2 comment (risk_management.py:280-283) says streaming the full history fixes the breaker being 'blind to multi-day peaks'. Since commit 67fa12ce, log_rotation.py:170 rotates portfolio_value_history.jsonl at max_age_days=90 / 20MB, and the _peak_cache shrink branch (lines 77-79) deliberately resets peak to floor and rescans, discarding any cached all-time peak the moment rotation truncates the file. The effective behavior is now a trailing ~90-day-peak drawdown breaker, not an all-time-peak one. That may be a fine policy, but the comment and the check_drawdown docstring should say so — a future reviewer reasoning from the comment will assume peaks persist indefinitely.

**Suggested fix:** Update the A-PR-2 comment to state the peak window is bounded by the rotation policy (90d), or persist the all-time peak in a small sidecar JSON if all-time semantics are actually wanted.

### [P3 · reliability · conf 0.70] Corrupt trade_guard_state.json silently resets cooldowns and loss-escalation to zero
`/mnt/q/finance-analyzer/portfolio/trade_guards.py:37`

_load_state (trade_guards.py:35-42) uses load_json with a fresh default; on a corrupt state file every guard check sees zero consecutive losses and no cooldowns, and the next record_trade() _save_state overwrites the corrupt file with the reset state — the escalation memory (up to 8x cooldown after a loss streak) is silently wiped with only file_utils' generic 'corrupt JSON' WARNING. portfolio_mgr got a quarantine + critical_errors journal entry for exactly this failure mode on 2026-06-01 (commit dfae83de); the guard state, which exists specifically to throttle a strategy that is on a losing streak, has no equivalent protection or backup.

**Suggested fix:** Minimum: log at ERROR/append a critical_errors row when STATE_FILE exists but parses to default. Better: reuse the portfolio_mgr backup-rotation pattern (.bak) for trade_guard_state.json so escalation state survives one bad write.

### [P3 · reliability · conf 0.55] _streaming_max raises TypeError on non-numeric history values, permanently fail-opening the drawdown gate for that portfolio
`/mnt/q/finance-analyzer/portfolio/risk_management.py:96`

`val = entry.get(value_key, 0)` then `if val > peak` (risk_management.py:96-97) — a row where patient_value_sek/bold_value_sek is null or a string raises TypeError, which is not caught (only json.JSONDecodeError is, line 94) and propagates out of check_drawdown. agent_invocation.py:967-977 catches it per-portfolio and proceeds fail-OPEN, so a single malformed row in the append-only history (legacy writer bug, manual edit, partial line that happens to parse) disables the circuit breaker for that portfolio on every cycle forever, with only an ERROR log line. The function's docstring promises corruption tolerance but only covers JSON-level corruption, not type-level.

**Suggested fix:** Wrap the value handling: `if isinstance(val, (int, float)) and math.isfinite(val) and val > peak: peak = val` (mirrors the NaN guard already added at check_drawdown:292).

## Swing loops (crypto / oil / MSTR)

### [P1 → P2 · bug · conf 0.88] Live SHORT entries priced and sized off the BULL cert quote
`/mnt/q/finance-analyzer/portfolio/mstr_loop/execution.py:469`

_live_fetch_cert_ask() ignores its `direction` argument and always quotes config.BULL_MSTR_OB_ID (line 469, comment 'v1 LONG-only'), but _handle_buy() passes that price to _live_place_buy(decision.cert_ob_id, price=cert_ask, ...) (line 166, 499) and sizes units = int(notional / cert_ask) (line 147). When mean_reversion SHORT is activated (set MSTR_LOOP_BEAR_OB_ID + flip STRATEGY_TOGGLES['mean_reversion'], exactly the documented activation path in config.py:28-33,62), a live SHORT places a limit order on the BEAR cert at the BULL cert's ask with units computed from the wrong instrument's price — wrong notional and wrong limit. _compute_shadow_cert_price (line 74-87) is already direction-aware, so shadow/paper results will not surface this; it only manifests in live mode. Module docstring (lines 6-8) promises 'migration between phases is a config flag flip, not a code change' — false here.

**Suggested fix:** _live_fetch_cert_ask should take the decision's cert_ob_id (or direction-resolved ob_id: BULL for LONG, BEAR for SHORT) and quote that instrument, mirroring _compute_shadow_cert_price.

**Skeptic verdict (real):** Confirmed: execution.py:463-475 ignores `direction` and always quotes BULL_MSTR_OB_ID ('v1 LONG-only'), while mean_reversion.py:74-78 opens SHORTs on BEAR_MSTR_OB_ID and _handle_buy (execution.py:141-166) sizes units and the live limit price from that BULL ask; shadow path (execution.py:74-87) is direction-aware so shadow stats won't catch it. Downgraded from P1: path is triply dormant today — PHASE defaults shadow (config.py:19), mean_reversion toggle False (config.py:62), BEAR_MSTR_OB_ID None (config.py:33) — and live flip requires 90d shadow + human approval (docs/MSTR_LOOP_NOTES.md:33-40); still a must-fix before SHORT live since the documented activation path would trip it.

### [P1 → REFUTED · reliability · conf 0.85] Sell paths have no cert_bid > 0 guard — live limit-sell at price 0.0 on quote failure
`/mnt/q/finance-analyzer/portfolio/mstr_loop/execution.py:218`

_handle_buy guards `if cert_ask <= 0: return False` (lines 142-145), but _handle_sell (line 218) and _handle_partial_sell (line 288) do not. In live phase _live_fetch_cert_bid returns 0.0 on any quote failure (lines 478-489), after which _live_place_sell(decision, price=0.0, units) is called (lines 235, 312). Best case Avanza rejects the 0-price order and the exit silently fails every cycle while the position keeps losing (stop/trail/EOD exits all route through here); worst case a near-zero limit sell fills instantly at bid during a quote glitch. Proceeds/pnl bookkeeping is also computed from the 0 bid (lines 222-223), so paper-mode pnl during a quote outage in the shadow/paper fallback path can record an open winner as a 100% loss.

**Suggested fix:** Add `if cert_bid <= 0: log + return False` at the top of _handle_sell and _handle_partial_sell, plus a Telegram/critical-error escalation when a live exit is refused repeatedly.

**Skeptic verdict (refuted):** Refuted downstream: avanza_session._place_order raises ValueError for price <= 0 and for order total < 1000 SEK (avanza_session.py ~lines 590-600), so a 0.0-price sell never reaches Avanza; _live_place_sell catches it, logs via logger.exception, returns False, and _handle_sell returns False WITHOUT booking proceeds/pnl (execution.py:234-240), with the exit retried every 60s cycle. The paper/shadow '100% loss' claim is also wrong — the synthetic fallback floors at 0.01 (execution.py:61) and entry cert price is guarded > 0 at buy; only residual is the lack of escalation on repeatedly refused live exits, a minor hardening nit.

### [P1 → P2 · reliability · conf 0.82] Live cash sync claimed in comments but never implemented — live phase starts permanently cash-starved
`/mnt/q/finance-analyzer/portfolio/mstr_loop/state.py:109`

state.py:109-111 says 'Live cash is synced from Avanza at loop startup' and execution.py:169 says 'live cash will re-sync next cycle', but no sync code exists anywhere in portfolio/mstr_loop/ (grep for sync/avanza across loop.py/__main__.py/state.py finds only the comment). default_state() sets cash_sek=0.0 for live, so on phase flip to live _notional_for_entry() returns 0 (execution.py:105-113) and every BUY is refused forever — the bot silently trades nothing. Even with a pre-seeded state file, live cash deducts the estimated total_cost and never reconciles against real fills/courtage, so cash drifts from the broker truth indefinitely.

**Suggested fix:** Implement the promised startup + periodic Avanza cash sync for PHASE=live (or fail loudly at startup if PHASE=live and no sync is available), and correct both comments.

**Skeptic verdict (real):** Confirmed: grep across portfolio/mstr_loop/ finds the sync only in comments (state.py:109-111, execution.py:169); loop.run_forever (loop.py:154-164) and __main__.main contain no Avanza cash sync, so PHASE=live starts at cash_sek=0.0 and _notional_for_entry (execution.py:105-113) returns 0, skipping every BUY with an info log. Downgraded from P1: it fails safe (no orders, no money lost), is live-only, and the phase flip is a documented manual human-approved step (docs/MSTR_LOOP_NOTES.md:33-40) where the dead bot would be noticed; the stale comments and broken 'config flag flip' contract are the real defect.

### [P1 → P2 · bug · conf 0.75] No loop-level EOD-flatten backstop; strategy EOD exit is gated behind bundle.is_usable()
`/mnt/q/finance-analyzer/portfolio/mstr_loop/loop.py:103`

momentum_rider.py:144 says 'the loop also enforces it [EOD flatten] as a backstop', but loop.py contains no EOD flatten — only the kill-switch flatten (lines 76-98). The only EOD exit lives inside strategy._evaluate_exit, which is unreachable when bundle.is_usable() is False (momentum_rider.step returns None on stale data; is_usable requires source_stale_seconds <= 300 and not sig.stale, data_provider.py:62-70). So if the main data loop lags >5 min or marks MSTR stale during the 21:45-22:00 CET window, a 5x-leveraged position is silently carried overnight — violating the intraday-only design, contaminating shadow-phase go/no-go stats now, and creating real overnight gap risk in live later. The kill-switch flatten has the same data dependency: if build_bundle() returns None, positions are not flattened at all (lines 81-82).

**Suggested fix:** Add an explicit EOD backstop in run_cycle that force-exits all open positions inside in_eod_flatten_window() regardless of bundle usability (use last-known or synthetic price for journaling), and make kill-switch flatten work without a fresh bundle.

**Skeptic verdict (real):** Confirmed: momentum_rider.py:143-144 claims 'the loop also enforces it as a backstop' but loop.py:71-151 has no EOD flatten; the only EOD exit lives in _evaluate_exit, unreachable when is_usable() is False (momentum_rider.py:39, mean_reversion.py:33; 300s staleness gate at data_provider.py:62-70), and the kill-switch flatten also skips when build_bundle() returns None (loop.py:81-82). Calibrated P2 not P1: PHASE is shadow today so the failure mode is overnight-carry contamination of shadow/paper go-live stats requiring a >5-min data-loop lag exactly inside 21:45-22:00 CET; becomes a genuine P1 prerequisite before any live flip.

### [P2 · doc-drift · conf 0.95] Doc drift: 'CL=F → Binance FAPI real-time' claimed in CLAUDE.md and oil_loop header; actual route is yfinance-only
`/mnt/q/finance-analyzer/data/oil_loop.py:6`

oil_loop.py module docstring (lines 6-7) and CLAUDE.md:196 both state oil prices route 'CL=F → Binance FAPI real-time with yfinance fallback'. price_source.py routes CL=F/BZ=F to the _YFINANCE_LAST_RESORT set (lines 95-97) with an explicit comment 'Binance has no oil perpetual', and oil_loop's own fetch_live_prices docstring (lines 163-169) says 'NOT Binance FAPI — there is no oil perpetual'. The module header and the project doc both advertise a real-time feed that does not exist; anyone reasoning about oil price freshness from CLAUDE.md (e.g. when sizing intraday plays) gets a 10-15-minute-lagged feed they believe is real-time.

**Suggested fix:** Fix oil_loop.py lines 6-7 and CLAUDE.md:196 to say 'CL=F/BZ=F → yfinance (10-15 min lag, no free real-time source)'.

### [P2 · design · conf 0.90] crypto_loop.py and oil_loop.py are ~90% copy-pasted scaffolding; four divergent singleton-lock implementations across loops
`/mnt/q/finance-analyzer/data/oil_loop.py:63`

Singleton lock (crypto_loop.py:72-126 vs oil_loop.py:74-133), fast_tick_check (175-221 vs 226-272), write_heartbeat shim (250-281 vs 301-324), run_loop (284-352 vs 327-393), and the full CLI/main/telegram wiring (358-428 vs 399-469) are line-for-line duplicates with only cfg/trader names differing. Fixes already drift between copies: oil_loop._pid_alive carries a documented codex NameError fix (lines 75-81) applied differently in crypto_loop (module-level subprocess import, line 32, plus a redundant local import at line 76). mstr_loop/__main__.py implements a third, mechanically different lock (msvcrt/fcntl byte-lock, lines 36-93) and metals_loop a fourth. The next bug fixed in one copy will silently persist in the others — these processes guard real state files.

**Suggested fix:** Extract a shared portfolio/loop_runtime.py (singleton lock, fast-tick scheduler, heartbeat shim, signal handling, CLI) parameterized by cfg + trader factory; crypto/oil loops become ~30-line declarations.

### [P2 · design · conf 0.80] Executable trading subsystems live in data/ alongside runtime state dumps
`/mnt/q/finance-analyzer/data/crypto_loop.py:42`

data/crypto_loop.py, data/oil_loop.py, data/metals_loop.py plus their configs/traders (`from data import crypto_swing_config`, line 42) make data/ both a Python package of production trading code and the runtime artifact directory (hundreds of *.json/*.jsonl/*.lock/*.heartbeat files, per git status also scratch XML files). Consequences: cleanup/backup/rotation tooling targeting data/ risks touching code; code review and grep over 'source' must wade through state; the package name shadows the conventional data directory meaning; and the newer mstr_loop demonstrates the team already considers portfolio/ the right home (portfolio/mstr_loop/ with file paths pointing INTO data/). All loop code also uses CWD-relative paths ('data/agent_summary_compact.json', 'config.json'), so correctness depends on every scheduled task starting in repo root.

**Suggested fix:** Migrate the swing loops to portfolio/ (e.g. portfolio/crypto_loop/, portfolio/oil_loop/) keeping data/ for state only; anchor data paths on a repo-root constant rather than CWD.

### [P2 · reliability · conf 0.72] Oil fast-tick hammers yfinance every 10s for a feed that lags 10-15 minutes
`/mnt/q/finance-analyzer/data/oil_loop.py:240`

fast_tick_check (line 240) calls fetch_live_prices every FAST_TICK_INTERVAL_SEC=10s (oil_swing_config.py:225), and each call does a full yf.download of 1m bars — plus a second 1d download whenever the 1m feed is gapped (lines 185-189), i.e. two downloads per tick all weekend. That is up to ~360-720 Yahoo requests/hour around the clock, which trips yfinance rate limiting (empty frames → the main cycle loses its price too) for zero benefit: the dip/flush alert is computed on a feed the module's own comments say lags 10-15 min (lines 139-145), so 'velocity flush in <180s' detection (FAST_TICK_FLUSH_WINDOW_SEC) is structurally meaningless on this source.

**Suggested fix:** Skip the fast-tick sub-poll entirely for yfinance-routed instruments (sleep the remainder of the cycle), or raise the oil fast-tick interval to >= 60s and cache the last fetch.

### [P2 · bug · conf 0.70] PHASE env var unvalidated; _handle_partial_sell silently mutates position state on unknown phase
`/mnt/q/finance-analyzer/portfolio/mstr_loop/config.py:19`

PHASE is read from MSTR_LOOP_PHASE with no validation against {shadow,paper,live} (config.py:19). _handle_buy and _handle_sell fail safe on unknown phase (execution.py:171-173, 241-243), but _handle_partial_sell has no else-branch: with a typo'd phase (e.g. 'Paper', trailing whitespace beyond strip, 'prod') it skips all three execution branches yet still falls through to `pos.units -= units_to_sell; pos.units_sold += ...; state.total_pnl_sek += pnl_sek` (execution.py:319-322) — position and P&L state mutated with no order placed and no journal record. Reachable when positions persist in mstr_loop_state.json across a restart that introduces the bad env var.

**Suggested fix:** Validate PHASE at import (raise ValueError on unknown value) and add an explicit else returning False in _handle_partial_sell.

### [P2 · bug · conf 0.65] crypto fetch_live_prices accepts 0.0 prices; oil counterpart guards last > 0
`/mnt/q/finance-analyzer/data/crypto_loop.py:149`

fetch_live_prices stores `float(r.json().get('lastPrice', 0))` with no positivity check (line 149), unlike oil_loop.py:200-202 which requires `last > 0`. A missing/zero lastPrice in a 200 response feeds price 0.0 into CryptoSwingTrader.evaluate_and_execute (stop/TP/P&L math sees a -100% move on open positions) and into fast_tick_check where it becomes the reference price, causing ZeroDivisionError at line 198 on the next tick (caught, but the warn-spam masks the root cause and alerts go dead).

**Suggested fix:** Mirror the oil guard: only store the price if > 0; optionally also guard ref_price > 0 in fast_tick_check before dividing.

### [P2 · reliability · conf 0.65] Oil grid signal stamps ts=now over possibly stale bars — real-money grid can't detect data age
`/mnt/q/finance-analyzer/portfolio/oil_grid_signal.py:141`

compute_signal() builds direction/confidence (up to 0.8) from BZ=F bars and stamps the output with ts=_utcnow_iso() (lines 141-148), but never checks the age of the last bar and does not include the bar timestamp in meta. The consumer is grid_fisher, which places real Avanza orders. If yfinance serves stale data (delayed feed, partial outage returning cached bars, holiday) the signal still publishes a fresh-looking, confident direction computed from old prices. oil_loop.py:139-205 already established the precedent (_PRICE_MAX_AGE_SEC freshness guard, 'a daily-fallback bar must NOT be passed off as a live price') — this module, which actually feeds real money, lacks the same guard.

**Suggested fix:** Record hist.index[-1] in meta as last_bar_ts and return direction=None when the last bar is older than a threshold (e.g. 2-3x the bar interval during market hours).

### [P3 · quality · conf 0.70] Partial-exit ladder can strand units=0 positions and miscount them as losses
`/mnt/q/finance-analyzer/portfolio/mstr_loop/execution.py:408`

With entry_units=2 and PARTIAL_EXIT_TRANCHES [(2.0, 1/3), (4.0, 1/3)], round(2*1/3)=1 per tranche sells both units, leaving the position on-book with units=0 ('final third rides' — config.py:154 — but no third remains). The position then lingers until a strategy exit fires a full SELL of 0 units: proceeds=0, pnl_sek=0, which increments state.losses (execution.py:248-251, `pnl_sek > 0` is False) and total_trades — skewing the win-rate that the shadow→paper→live promotion decision reads from the scorecard. Small-unit entries are exactly what MIN_TRADE_SEK/low-cash mode produces.

**Suggested fix:** In _handle_partial_sell, close the position (remove + count stats) when pos.units hits 0; or floor tranche sizing so at least 1 unit always remains for the trail.

## Dashboard & security

### [P1 → P2 · security · conf 0.55] Stored XSS: scraped-derived markdown rendered with raw HTML passthrough
`dashboard/house_blueprint.py:308`

_render_markdown() calls md_lib.markdown(text, extensions=['tables','fenced_code','sane_lists']) with NO HTML sanitizer. Python-Markdown preserves raw HTML in the source by default (safe_mode was removed years ago; the library explicitly tells callers to post-sanitize with Bleach/nh3). The rendered output is concatenated into the page shell and served same-origin on the authenticated dashboard at candidate_detail() (line 776), run_detail() (line 746-749), and k10() (line 824). The .md files come from the findapartments pipeline, which writes reports built from externally scraped Hemnet/Booli listing data (addresses, descriptions) passed through an LLM. A listing field containing <script>...</script> that survives into a report renders as live JS in the dashboard origin. The CSP only sets frame-ancestors (no script-src), so inline scripts execute. The auth cookie is HttpOnly so it can't be read by JS, but the injected script runs in-origin and can drive every authenticated API (e.g. read live brokerage state from /api/avanza_account and exfiltrate it, or POST /api/validate-portfolio).

**Suggested fix:** Sanitize rendered HTML before returning it (e.g. nh3/bleach allowlist of tags), or HTML-escape the markdown source for any content derived from scraped/LLM output. Add a script-src 'self' CSP to the house pages as defense-in-depth.

**Skeptic verdict (real):** Confirmed sink: python-markdown 3.10.2 passes raw HTML/<script> through verbatim (verified by test with the exact extensions at house_blueprint.py:308-313); _render_markdown output is concatenated into _shell and returned as a raw Flask str (no Jinja autoescape) at candidate_detail():778, run_detail():747, k10():824. No bleach/nh3 sanitizer exists, and CSP at app.py:67-72 only sets frame-ancestors (no script-src) so inline JS executes. Source is attacker-influenced: analysis/findapartments_report.py render_candidate_md writes scraped/LLM fields unescaped (address :214/:219, bathroom_detail :228, pl['rationale'], methodology). Real but downgraded to P2: single-user auth-gated dashboard (token+CF Access) and exploitation requires a malicious public listing to survive scraping into a top-N deep-dive report and then be viewed.

### [P2 · security · conf 0.70] Magic-link slug and ?token= leak into Werkzeug/Cloudflare access logs; docstring claims otherwise
`dashboard/app.py:835`

The /go/<slug> docstring asserts the secret 'never appears in the URL or the response body, so it can't be lifted from browser history / referrer / server logs.' That is false for server logs: the slug IS the URL path, and Werkzeug's WSGIRequestHandler (ThreadedWSGIServer in _serve_dual_stack) logs the full request line ('GET /go/<slug> HTTP/1.1') to the werkzeug logger by default. The slug is a 1-year credential, so it lands in the dashboard log and any Cloudflare request log. The first-visit ?token=XXX path (auth.py:213) is worse: the raw dashboard_token appears in the query string and is logged verbatim before index()'s redirect strips it from the address bar. No request-logging suppression is configured anywhere in dashboard/.

**Suggested fix:** Disable or scrub Werkzeug request-line logging for /go and query strings, or move the slug/token into a POST body / fragment. At minimum correct the docstring claim — the slug does reach server logs.

### [P2 · reliability · conf 0.60] secure=True auth cookie is dropped over the plain-HTTP LAN bind, forcing ?token= on every request
`dashboard/auth.py:137`

_refresh_cookie() sets the pf_dashboard_token cookie with secure=True. The server (_serve_dual_stack, app.py:2444) binds a plain-HTTP socket on all interfaces ([::] with IPV6_V6ONLY=0), so it is directly reachable over http:// from LAN / WSL bridge / Tailscale — the very fallback path the dual-stack bind was added to support. Browsers refuse to store/send Secure cookies over plain HTTP, so on direct-HTTP LAN access the cookie is silently never persisted and the user must re-supply ?token= on every navigation. Each such request then writes the real token into the access log (see the logging finding). HSTS is also emitted (app.py:73) but is ignored over HTTP. The cookie path only works through the HTTPS Cloudflare tunnel.

**Suggested fix:** Either front all access through HTTPS (drop the plain-HTTP all-interfaces bind, bind loopback only behind the tunnel), or conditionally set Secure based on request.is_secure / X-Forwarded-Proto so the LAN HTTP fallback can actually keep a cookie.

### [P2 · design · conf 0.60] 2460-line app.py with 45 routes is a monolith mixing routing, normalization, caching and a Playwright worker
`dashboard/app.py:40`

dashboard/app.py is 2460 lines and registers 45 routes (55 with house_blueprint), interleaving HTTP routing, ~15 _normalize_*/_build_* data-shaping helpers (lines 253-748), three independent TTL caches, and a dedicated Avanza Playwright worker-thread subsystem (2002-2156). Most normalization helpers (golddigger/metals reshaping) are pure functions with no Flask dependency and could live in a module unit-tested without the app. The single file makes the auth surface and route inventory hard to audit (the CLAUDE.md route list is already maintained by hand and flagged as drift-prone). Auth was already split into auth.py to break a circular import; the same factoring should continue for the per-subsystem route groups and the avanza worker.

**Suggested fix:** Extract normalization helpers into a dashboard/serializers module and the Avanza worker into its own module; group routes into blueprints (metals, crypto, system) mirroring house_blueprint.

### [P2 · reliability · conf 0.55] /api/avanza_account: single-worker queue has no in-flight dedup or bound, can pile up under concurrency
`dashboard/app.py:2060`

_avanza_account_snapshot() enqueues a future onto an unbounded queue.Queue serviced by one Playwright worker thread; each snapshot can take up to 25s. Unlike /api/system_status and /api/trading_status (which hold a lock and re-check the cache so concurrent misses coalesce), the avanza route's 30s TTL cache check (api_avanza_account, ~line 2214) is not held across the snapshot build, and force=1 bypasses the cache entirely. So N concurrent cache-miss or force=1 requests enqueue N jobs that serialize on the worker; the k-th caller's own 25s wait expires and returns a timeout error while the worker keeps draining stale queued jobs, and the queue can grow without bound (e.g. a stuck/expired BankID session making every snapshot hit the full timeout).

**Suggested fix:** Coalesce concurrent requests onto a single in-flight future (collapse duplicate enqueues), bound the queue, and/or drop queued jobs older than the cache TTL.

### [P3 · doc-drift · conf 0.90] _in_session docstring still says 15:30-21:55; code uses 08:30-21:30
`dashboard/trading_status.py:279`

The _in_session() docstring states the warrant session is '15:30-21:55 inclusive of open, exclusive of close', but SESSION_OPEN=dtime(8,30) and SESSION_CLOSE=dtime(21,30) (lines 55-56), and the module header (lines 45-56) documents the corrected 08:30-21:30 window unified on 2026-05-11. The docstring is the pre-fix value and contradicts the code it documents, which is exactly the kind of stale comment that misleads a future session reasoning about why a bot renders OUTSIDE_HOURS.

**Suggested fix:** Update the _in_session docstring to 08:30-21:30 to match SESSION_OPEN/SESSION_CLOSE.

### [P3 · reliability · conf 0.60] JWKS fetch has no negative caching: blocks 5s per request during a Cloudflare JWKS outage
`dashboard/cf_access.py:83`

_get_jwks_client() only caches on success; on a fetch failure it returns None without recording the failure, so every subsequent CF-Access request (cf_email + cf_jwt headers present) re-attempts requests.get(certs_url) with a 5s timeout on the request thread before falling through to cookie/token auth. During a transient CF JWKS outage, each CF-fronted request pays the full 5s latency, and a burst of dashboard polling all eats 5s apiece. Successful clients are cached for an hour, so the asymmetry only bites the failure path.

**Suggested fix:** Add short negative caching (e.g. cache the None result for ~30-60s) so repeated failures don't each block for the full 5s timeout.

### [P3 · reliability · conf 0.50] POST /api/validate-portfolio accepts unbounded JSON body (no MAX_CONTENT_LENGTH)
`dashboard/app.py:1121`

api_validate_portfolio reads request.get_json(silent=True) with no size limit, and the Flask app sets no MAX_CONTENT_LENGTH anywhere. An authenticated client (or a compromised same-origin script via the markdown-XSS path) can POST an arbitrarily large body that Flask buffers and json-parses in memory on a worker thread. Low impact because the route is auth-gated and single-user, but combined with the all-interfaces bind it is a cheap memory-pressure lever.

**Suggested fix:** Set app.config['MAX_CONTENT_LENGTH'] to a sane ceiling (e.g. 1MB) so oversized bodies are rejected before parsing.

## Infrastructure (file I/O, HTTP, health, telegram)

### [P1 → P2 · bug · conf 0.80] check_agent_silence is blind to silent Layer 2 failure: trigger timestamp cached as last_invocation_ts
`/mnt/q/finance-analyzer/portfolio/health.py:35`

update_health() (health.py:33-35) writes state['last_invocation_ts'] = last_trigger_time whenever a trigger fires. main.py:1059-1063 passes trigger_reason on every triggered/force_report cycle regardless of whether the Layer 2 invocation actually ran or succeeded. check_agent_silence() (health.py:180-188) prefers this cached value and only falls back to the invocations.jsonl ground truth when the cache field is absent — which it never is once one trigger has fired. Consequence: if Layer 2 fails silently (the exact March-April 2026 'claude -p exits 0 printing Not logged in' outage class), is gated, or is disabled (the current 2026-06-06 token freeze), triggers keep firing, last_invocation_ts keeps advancing, and /api/health reports agent_silent=false forever. The monitor masks precisely the failure mode it was built to detect.

**Suggested fix:** Set last_invocation_ts only from actual invocation completion (e.g. in agent_invocation.py after appending to invocations.jsonl), or have check_agent_silence always read last_jsonl_entry(invocations.jsonl, field='ts') and use the health-state cache only as a staleness optimization keyed to the same source.

**Skeptic verdict (real):** Confirmed: main.py:1058-1064 passes trigger_reason on every triggered/force_report cycle regardless of invocation outcome (including autonomous/skipped/disabled paths), health.py:30-35 caches it as last_invocation_ts, and check_agent_silence (health.py:180-188) prefers the cache so the invocations.jsonl fallback never runs once a trigger has fired. Severity downgraded because the silent-L2 failure mode has dedicated detection in loop_contract.py's layer2_journal_activity CRITICAL invariant (loop_contract.py:516-524, trigger-without-journal check), so the blindness affects only the dashboard agent_silent metric and scripts/health_check.py:280.

### [P1 → P2 · bug · conf 0.75] Offset-holdback retry design is illusory: next getUpdates poll permanently acks a failed command server-side
`/mnt/q/finance-analyzer/portfolio/telegram_poller.py:161`

_handle_update advances self.offset in memory before dispatch (line 161) and on a raised dispatch deliberately skips persisting it so 'restart re-fetches and retries' (lines 162-167, 256-265, Codex P1 round-7). But Telegram confirms updates as soon as getUpdates is called with offset > update_id: 5 seconds after the failed dispatch, _poll_loop calls _get_updates with the advanced in-memory offset (lines 126-127), permanently deleting the failed update from Telegram's queue. The persisted-offset replay only works if the process dies within that single 5 s poll window. Additionally, any later settled update in the session persists offset past the failed one anyway (line 264). Net effect: a transient dispatch failure (Avanza session hiccup, network) silently consumes real-money bookkeeping commands like 'bought MSTR ...', leaving iskbets state diverged from the user's actual position.

**Suggested fix:** On dispatch failure, either (a) do not advance self.offset past the failed update_id (accept re-processing it next poll with a bounded retry count), or (b) persist the failed command to a local retry queue (e.g. telegram_inbound.jsonl already records it — add a replay of 'raised:*' rows at startup) before allowing the offset to advance.

**Skeptic verdict (real):** Confirmed: telegram_poller.py:161 advances self.offset in memory before dispatch; a raised dispatch propagates to _poll_loop (lines 113-121) which 5s later calls _get_updates with the advanced offset (lines 126-127), and per Telegram getUpdates semantics that permanently confirms the failed update — the persisted-offset replay (lines 254-265) only works if the process dies within that single 5s poll window, and any later settled update persists the global offset past the failed one anyway. Downgraded to P2: the failure is not fully silent (raised:* row logged to telegram_inbound.jsonl at line 252/289, Poller error logged, user gets no confirmation reply) and transient dispatch failures are rare, but the Codex-P1-round-7 retry design genuinely does not deliver its documented guarantee.

### [P1 → P2 · reliability · conf 0.70] msvcrt.locking(LK_LOCK) is not blocking — raises OSError after 10 retries; long rotations make Windows appends fail
`/mnt/q/finance-analyzer/portfolio/file_utils.py:295`

jsonl_sidecar_lock's comment (file_utils.py:268-269) claims 'msvcrt.locking blocks on contention on Windows'. Per CPython docs, LK_LOCK retries once per second and raises OSError after 10 failed attempts — it is bounded, not blocking. log_rotation.rotate_jsonl holds the same sidecar lock for the entire read + gzip-decompress/recompress-of-monthly-archive + rewrite sequence (log_rotation.py:379-504); for signal_log.jsonl (currently ~13 MB live, archives much larger decompressed) this can exceed 10 s. Any atomic_append_jsonl arriving during that window on Windows (production OS) raises OSError. Several call sites are unwrapped (e.g. agent_invocation.py:434 invocations log, accuracy_stats.py:1586, autonomous.py:172), so the entry is lost and the exception propagates into the calling cycle step. POSIX fcntl.flock blocks indefinitely, so tests on WSL never see this.

**Suggested fix:** Wrap msvcrt acquisition in a retry loop honoring a real deadline (e.g. loop on OSError until timeout), or use LK_NBLCK with explicit sleep/retry. Also fix the comment, and consider moving the gzip archive-merge work outside the lock (only the live-file read and os.replace need it).

**Skeptic verdict (real):** Confirmed: file_utils.py:268-269/:295 claims 'blocking' but CPython msvcrt LK_LOCK retries 10x1s then raises OSError; rotate_jsonl holds the same sidecar lock across the full read+gzip-archive-merge+rewrite (log_rotation.py:379-504) and runs hourly (main.py:404), with cross-process appenders (e.g. golddigger process appends golddigger_log.jsonl whose monthly archive is 7.8MB gz) and unwrapped call sites (_log_trigger, agent_invocation.py:427-434). Downgraded to P2: only holds >10s with a concurrent cross-process append fail, most rotated files have sub-second holds, and the blast radius is one lost log entry plus an exception usually absorbed by cycle-level try/except.

### [P1 → P2 · reliability · conf 0.70] Uncapped 429 retry_after sleep (doubled by jitter) can stall the 60s main loop for minutes to hours
`/mnt/q/finance-analyzer/portfolio/http_retry.py:58`

On HTTP 429, fetch_with_retry takes retry_after from the Telegram-style JSON body (resp.json()['parameters']['retry_after']) with no upper bound, then adds full jitter on top (wait += random.uniform(0, wait)), so a single attempt can sleep up to 2x retry_after, and with retries=3 the total worst case is ~6x retry_after. Telegram flood-waits can be hundreds to thousands of seconds. fetch_with_retry is called synchronously from message_store._do_send_telegram (invoked from send_or_store inside Layer 1 cycle paths like telegram_notifications._maybe_send_alert) and from telegram_poller. A flood-wait therefore blocks the loop thread far past the 300 s heartbeat staleness threshold — a Telegram send failure does block the loop. Secondary: the standard Retry-After HTTP header (used by Binance, which calls this via data_collector) is never consulted, only the Telegram JSON shape.

**Suggested fix:** Clamp wait to a hard cap (e.g. min(wait, 30s)); when retry_after exceeds the cap, return None immediately and let the caller log the message as unsent (it is already persisted to telegram_messages.jsonl). Use decorrelated jitter (random between base and retry_after, not additive doubling) and also parse the Retry-After header.

**Skeptic verdict (real):** Confirmed in http_retry.py:51-62: retry_after taken from the Telegram JSON body with no cap, then wait += random.uniform(0, wait) doubles it (worst ~6x retry_after over 3 retries), and _do_send_telegram (message_store.py:135) is called synchronously from main-loop cycle paths (main.py:782/1084/1181) with no keepalive wrapper; the Retry-After header is indeed never parsed. Downgraded to P2: requires Telegram to return a large flood-wait (rare given the system's send cooldowns/mute gates), and the effect is delayed cycles plus a false-stale heartbeat, fully recoverable.

### [P1 → REFUTED · reliability · conf 0.70] Cross-process GPU exclusion broken: peer Q:/models/gpu_lock.py uses non-atomic acquire and pid-blind stale break on the same lock file
`/mnt/q/finance-analyzer/portfolio/gpu_gate.py:264`

Two implementations share Q:/models/.gpu_lock. portfolio/gpu_gate.py acquires via atomic O_CREAT|O_EXCL with pid-alive stale checks, but Q:/models/gpu_lock.py (used by the LLM venv side) acquires via exists()-check-then-write_text (gpu_lock.py:88-92, classic TOCTOU — it can overwrite a live lock created in the race window) and its stale break (gpu_lock.py:102-111) checks only mtime>300s with no pid-alive check. Since gpu_gate holders never refresh the lock file's mtime during a hold, any legitimate hold longer than 300 s (large LLM batch, model swap under contention) gets its lock stolen by the peer while the owner is alive — two models then load on the 10 GB RTX 3080 simultaneously (VRAM OOM, inference crash, signal failures). Compounding: gpu_gate's _release_lock at line 264 unlinks unconditionally without verifying it still owns the file, so after a steal it deletes the thief's lock, cascading the breakage; and _try_break_stale_lock (lines 119-129) has its own read-then-unlink race against a concurrent break+recreate.

**Suggested fix:** Unify on one implementation (make gpu_lock.py import/replicate gpu_gate's O_EXCL acquire + pid-alive stale predicate). Have long holders touch the lock mtime periodically (the existing sweeper thread could refresh its own process's lock). Verify ownership (pid match) before unlinking in gpu_gate's release, as gpu_lock.py already does.

**Skeptic verdict (refuted):** The cited defects exist in Q:/models/gpu_lock.py (TOCTOU acquire lines 88-92, pid-blind mtime-only stale break lines 102-111), but the module is dormant in production: its only importer is fingpt_infer.py's direct-inference path (fingpt_infer.py:215/291), retired in the 2026-04-09 llm_batch migration — llm_batch.py:369 imports fingpt_infer solely for prompt templates/parsers, and every live GPU user (main loop signals, data/metals_llm.py:223, scripts/chronos_bolt_benchmark.py:124) goes through portfolio.gpu_gate, whose stale break requires the owner pid to be dead (gpu_gate.py:103-134), so no live-holder lock steal occurs between production processes. The gpu_gate ownership-blind unlink at gpu_gate.py:264 is latent hardening debt only reachable after a steal that the production predicate prevents.

### [P2 · bug · conf 0.60] prune_jsonl permanently drops the concatenated-object lines the 2026-06-06 recovery decoder was added to preserve
`/mnt/q/finance-analyzer/portfolio/file_utils.py:457`

_decode_jsonl_line (added 2026-06-06) recovers legacy lines carrying two JSON objects concatenated without a newline, specifically because dropping them lost real data (resolution rows in critical_errors.jsonl). But prune_jsonl validates lines with bare json.loads (line 457) and discards anything that fails as 'malformed', so any such legacy line is permanently deleted on the next prune. main.py:357-361 prunes invocations.jsonl, layer2_journal.jsonl, telegram_messages.jsonl and claude_invocations.jsonl to 5000 entries every cycle batch — all files written by the same historical append race. rotate_jsonl correctly keeps unparseable lines ('keep the entry to be safe'); prune_jsonl contradicts both that and the recovery decoder's intent.

**Suggested fix:** In prune_jsonl, validate with _decode_jsonl_line and keep lines where it returns non-empty (optionally splitting recovered multi-object lines into separate output lines, which also heals the file).

### [P2 · design · conf 0.60] atomic_write_jsonl bypasses jsonl_sidecar_lock — read-modify-rewrite callers can discard concurrent appends
`/mnt/q/finance-analyzer/portfolio/file_utils.py:352`

atomic_write_jsonl rewrites a JSONL file via tempfile + os.replace without taking the sidecar lock that atomic_append_jsonl and rotate_jsonl share. Any caller doing load_jsonl -> mutate -> atomic_write_jsonl races against concurrent appenders exactly like the rotation bug fixed 2026-05-11 (commit 3b623129): an append landing between the read and the replace is silently discarded. Concrete instance: fin_evolve.py:205-251 loads fin_command_log.jsonl, scores outcomes (can take seconds), then rewrites it — while /fin-* commands append to the same file via atomic_append_jsonl from other processes. signal_history.py:48 and forecast_accuracy.py:375 follow the same pattern. The 2026-05-11 fix is complete for rotation but the lock contract is not enforced or documented on this primitive.

**Suggested fix:** Take jsonl_sidecar_lock(path) inside atomic_write_jsonl (it is reentrant-safe here since no caller holds it), or at minimum document that callers performing read-modify-rewrite must hold the lock around the whole sequence and fix fin_evolve/signal_history/forecast_accuracy to do so.

### [P2 · bug · conf 0.55] last_jsonl_entry recovery decoder can return a non-dict scalar from a truncated >4KB final line
`/mnt/q/finance-analyzer/portfolio/file_utils.py:404`

last_jsonl_entry reads only the last 4096 bytes. If the file's final line exceeds 4 KB (layer2_journal.jsonl currently has lines up to 4730 bytes; 2 lines exceed 4 KB), the window contains only a truncated middle of that line. _decode_jsonl_line's recovery loop uses raw_decode, which accepts any JSON value — a truncation landing before a quoted key (e.g. ..."context": {...}) decodes the bare string "context" and returns it as the 'entry'. With field set, the isinstance(dict) guard returns None (graceful), but field=None callers get a str/number where a dict is expected: loop_contract.py:334/435 feed it to contract checks, agent_invocation.py:1613 feeds it to _write_fishing_context (exception suppressed, so it degrades silently). Pre-2026-06-06 behavior returned None for such lines.

**Suggested fix:** In _decode_jsonl_line's recovery loop, only accept objects (require line[idx] == '{' or skip non-dict results), or have last_jsonl_entry filter objs to dicts before selecting the last one. Optionally grow the tail window when the newest complete line cannot be decoded.

### [P3 · doc-drift · conf 0.85] Module docstring category routing table drifted from SEND_CATEGORIES
`/mnt/q/finance-analyzer/portfolio/message_store.py:52`

The module docstring (lines 1-19) enumerates which categories are sent to Telegram, but SEND_CATEGORIES (line 52) now also includes daily_digest, elongir and crypto_report, none of which appear in the docstring. Since this docstring is the routing reference agents read when deciding how a new message category will behave (send vs save-only), the drift can lead to wrong category choices for new notification paths.

**Suggested fix:** Regenerate the docstring list from SEND_CATEGORIES, or replace the prose lists with a single pointer to the SEND_CATEGORIES constant.

### [P3 · quality · conf 0.75] fetch_json silently swallows **kwargs — passing json_body produces a body-less request with no error
`/mnt/q/finance-analyzer/portfolio/http_retry.py:85`

fetch_json accepts **kwargs but never forwards them to fetch_with_retry (lines 90-91). A caller writing fetch_json(url, method='POST', json_body={...}) gets a POST with no body and no warning — a silent-failure trap in a codebase whose stated #1 failure class is silent failures. No current caller passes json_body (verified by grep), so this is latent, but the signature actively invites the mistake.

**Suggested fix:** Either forward kwargs (fetch_with_retry(..., **kwargs)) or remove **kwargs and add json_body as an explicit parameter.

## Documentation drift

### [P1 → P2 · doc-drift · conf 0.97] Active-signal list claims 21 active; reality is 15 — 6 listed signals are disabled
`/mnt/q/finance-analyzer/CLAUDE.md:202`

CLAUDE.md 'Signal System (80 Modules · 21 Active · 59 Disabled)' lists 21 active signals, but portfolio/tickers.py DISABLED_SIGNALS now disables 6 of them: metals_cross_asset (#11, disabled 2026-06-06, tickers.py:305), crypto_evrp (#15, re-disabled 2026-05-26, tickers.py:94), adx_regime_switch (#17, re-disabled 2026-06-01, tickers.py:230), choppiness_regime_gate (#19, tickers.py:234), bocpd_regime_switch (#20, tickers.py:241), vol_ratio_regime (#21, tickers.py:102). Actual active set is exactly 15: rsi, bb, fear_greed, ministral, qwen3, momentum, mean_reversion, news_event, econ_calendar, crypto_macro, cot_positioning, onchain, statistical_jump_regime, drift_regime_gate, amihud_illiquidity_regime. The stale entries also carry pre-collapse accuracies (e.g. ADX '67.0%, 182 sam' degraded to 49.0% on 492 sam per tickers.py:230). Crypto EVRP even appears in BOTH the active list (line 219) and the disabled 'Pending validation' list (line 245). Layer 2 reads CLAUDE.md on every invocation, so trade reasoning is anchored to dead signals with inflated accuracy claims. SESSION_PROGRESS.md 2026-06-01 documents the disablement but CLAUDE.md was never updated.

**Suggested fix:** Rewrite the Active section to the 15 actually-active signals (verify with: python -c "from portfolio.tickers import SIGNAL_NAMES, DISABLED_SIGNALS; ..."), move the 6 disabled ones to the Disabled section with their current accuracies, and update the header counts.

**Skeptic verdict (real):** Confirmed by executing `from portfolio.tickers import SIGNAL_NAMES, DISABLED_SIGNALS`: active set is exactly the 15 signals the finding lists; all 6 cited entries verified in DISABLED_SIGNALS with dated rationale comments (crypto_evrp tickers.py:94, vol_ratio_regime :102, adx_regime_switch :230, choppiness_regime_gate :234, bocpd_regime_switch :241, metals_cross_asset :305), and CLAUDE.md:202-225 still lists all 21 active with stale pre-collapse accuracies (Crypto EVRP duplicated in both Active #15 and Pending-validation). Downgraded P1→P2: disabled signals are force-HOLD and absent from agent_summary_compact/agent_context files Layer 2 trades from, so stale doc misleads context/reasoning but cannot directly drive trades on dead signals (and Layer 2 is currently frozen per 2026-06-06 token freeze).

### [P2 · doc-drift · conf 0.95] Per-ticker override list stale: williams_vix_fix and credit_spread_risk overrides were removed
`/mnt/q/finance-analyzer/CLAUDE.md:228`

CLAUDE.md claims 4 per-ticker overrides: Williams VIX Fix → XAU/XAG (line 228), Realized Skewness → XAU, Credit Spread Risk → BTC/ETH (line 230), ML Classifier → ETH. Code reality in portfolio/signal_engine.py:723-736 (_DISABLED_SIGNAL_OVERRIDES): only ('ml','ETH-USD') and ('realized_skewness','XAU-USD') remain. williams_vix_fix was REMOVED 2026-05-31 ('recent accuracy collapsed to 30.5% (131 sam). Actively harmful', signal_engine.py:727-729) and credit_spread_risk was REMOVED 2026-05-26 ('Override was re-enabling a broken signal', signal_engine.py:733-735). The doc still advertises 76.5%/60.9%/57.4% accuracies for overrides that no longer exist.

**Suggested fix:** Delete the Williams VIX Fix and Credit Spread Risk override lines from CLAUDE.md; keep only Realized Skewness → XAU and ML Classifier → ETH.

### [P2 · doc-drift · conf 0.95] README frozen at 2026-03-02 — describes 19 Tier-1 instruments (15 since-removed stocks), 30 signals, ~720 tests
`/mnt/q/finance-analyzer/README.md:27`

README.md last committed 2026-03-02 (c01b88a7) and predates both instrument purges. It claims '19 Tier 1' instruments including 15 US stocks (NVDA, AMD, AAPL, GOOGL, META, TSM, PLTR, AMZN, AVGO, MU, SOUN, SMCI, TTWO, VRT, LMT — all removed Mar 15 / Apr 09 per portfolio/tickers.py:22-23, which now contains exactly 5 SYMBOLS), 'MINI-TSMC' warrant (gone), '30 Signals (8 Core + 19 Enhanced + 3 AI)' (vs 89 tracked / 15 active), '~720 tests' (440 test files, ~10,400 test functions), and 'Layer 1 ... NEVER sends Telegram' (false — main.py:782/1084/1193 send error alerts, digest.py:269 sends 4h digests). Anyone onboarding from the README gets a picture of a system that hasn't existed for 3 months.

**Suggested fix:** Regenerate README instrument list, signal counts, test counts, and the Layer-1-never-sends-Telegram claim from current code; or replace the stale sections with pointers to CLAUDE.md/SYSTEM_OVERVIEW.md.

### [P2 · doc-drift · conf 0.92] 'MIN_VOTERS = 3 (all asset classes)' is wrong — metals run MIN_VOTERS = 2 since 2026-05-11
`/mnt/q/finance-analyzer/CLAUDE.md:251`

CLAUDE.md line 251 states 'MIN_VOTERS = 3 (all asset classes)' and docs/TRADING_PLAYBOOK.md:64 tells Bold 'Floor: MIN_VOTERS=3. Never trade when fewer agree.' But portfolio/signal_engine.py:1180 sets MIN_VOTERS_METALS = 2 (comment: '2026-05-11: metals run at noisier intraday horizon ... MIN_VOTERS=3 produced 0 trades in 20 days'). Layer 1 can therefore emit a metals consensus with 2 voters that the Layer 2 playbook instructs the agent to reject — the agent either refuses valid triggers or is confused about why the trigger fired.

**Suggested fix:** Update both docs: 'MIN_VOTERS = 3 (crypto/stocks), 2 (metals, since 2026-05-11)'.

### [P2 · doc-drift · conf 0.92] Test-surface claims stale: '~5,994 tests across 242 files' vs actual 440 files / ~10,400 test functions; TESTING.md says 7,730
`/mnt/q/finance-analyzer/CLAUDE.md:374`

CLAUDE.md:374 claims '~5,994 tests across 242 files, ~16 min sequential' and line 385 claims '26 pre-existing failures'. docs/TESTING.md:23-28 claims '~7,730 tests', '~242 files', '24 pre-existing failures' (snapshot dated 2026-04-19). Measured now: 440 test_*.py files in tests/ (ls tests/test_*.py | wc -l) and 10,376 'def test_' functions — roughly 1.8x the doc figures. The two docs also disagree with each other (5,994 vs 7,730; 26 vs 24 failures). Runtime estimates derived from the stale counts ('~5.5 min parallel') are likely also wrong, which matters for the pre-merge full-suite workflow.

**Suggested fix:** Run a one-shot pytest --collect-only -q count and update both CLAUDE.md and TESTING.md (test count, file count, failure count, runtimes) with a fresh as-of date.

### [P2 · doc-drift · conf 0.92] Playbook references disabled/retired systems: 'Forecast signal (#28)' with Kronos, 'all 30 signals', '20+ instruments'
`/mnt/q/finance-analyzer/docs/TRADING_PLAYBOOK.md:360`

Three stale claims the Layer 2 agent reads as operational truth: (1) line 360 'Forecast Health: Forecast signal (#28) uses health-weighted voting. Kronos mostly dead ... Chronos is primary. XAG-USD 24h ~76% accurate' — forecast was fully disabled 2026-05-12 at 25.6% recent accuracy (tickers.py:199-206) and Kronos was retired entirely 2026-04-21 (tickers.py:165-177); the '#28' numbering belongs to the long-dead 30-signal scheme. (2) lines 89/99 instruct 'Review all 30 signals across all timeframes' — there are 15 active signals. (3) line 36 'Full cross-asset analysis of all 20+ instruments' — there are 5 Tier-1 + 3 Tier-2 + 3 Tier-3 = 11 instruments since the Apr 09 purge. The agent is told to weight a signal that never votes and to expect instruments that no longer exist.

**Suggested fix:** Delete or rewrite the Forecast Health section (forecast disabled 2026-05-12), replace '30 signals' with 'active signals (see agent_summary)', and fix the instrument count.

### [P2 · doc-drift · conf 0.90] Signal-count claims internally inconsistent and all wrong: 80 registered / 65-voting / 44 disabled / 38 enhanced vs reality 70 / 89 / 76 / 80
`/mnt/q/finance-analyzer/CLAUDE.md:120`

CLAUDE.md gives four mutually inconsistent counts: line 120 '21 active signals (80 modules registered, 59 disabled)', line 284 'signal_engine.py (65-signal voting, 21 active)', line 233 'Disabled (44 — force-HOLD via DISABLED_SIGNALS)', line 285 'signals/*.py (38 enhanced modules)'. Measured reality: 70 register_enhanced() entries in portfolio/signal_registry.py, 89 names in tickers.py SIGNAL_NAMES, 76 entries in DISABLED_SIGNALS (→ 15 active), and 80 files in portfolio/signals/. None of the four doc numbers matches any code number, and they don't match each other.

**Suggested fix:** Reconcile to one set of numbers sourced from signal_registry/tickers (currently: ~70 registered enhanced, 89 tracked names, 15 active, 76 disabled, 80 files in signals/) and remove the duplicated stale counts.

### [P2 · doc-drift · conf 0.88] 'You are the ONLY Telegram sender — Layer 1 does NOT send messages' is false
`/mnt/q/finance-analyzer/docs/TRADING_PLAYBOOK.md:240`

docs/TRADING_PLAYBOOK.md:240 tells the Layer 2 agent it is the sole Telegram sender and that Layer 1 never sends messages. Code: portfolio/main.py sends error/crash alerts via send_or_store (main.py:782, 1084, 1096, 1181, 1193, 1401), portfolio/digest.py:269 sends the 4-hour digest, and CLAUDE.md's own module map lists digest.py (4h periodic), daily_digest.py (morning) and telegram_poller.py as Layer-1-side senders. README.md:14 repeats the same false claim. An agent that believes it is the only sender may duplicate digest content or mis-attribute messages it sees in data/telegram_messages.jsonl when reasoning about prior notifications.

**Suggested fix:** Reword to: 'You are the sole authority on TRADE notifications; Layer 1 independently sends error alerts, 4h digests, and the morning digest.'

### [P3 · doc-drift · conf 0.93] SYSTEM_OVERVIEW header contradicts its own body: '80 modules (21 active, 59 disabled)' vs '69 modules: 15 active + 54 disabled'
`/mnt/q/finance-analyzer/docs/SYSTEM_OVERVIEW.md:8`

docs/SYSTEM_OVERVIEW.md line 8 (Architecture Summary) says '80 signal modules (21 active, 59 disabled)' while line 35 of the same document says 'Signal System (69 modules: 69 enhanced, 15 active + 54 disabled)' and line 141 repeats '69 total: 15 active, 54 disabled'. The body (updated 2026-06-01) is close to current reality (15 active confirmed); the header was never updated. Additionally CLAUDE.md:321 points here claiming 'Full module map (142 modules)' while SYSTEM_OVERVIEW.md:26 claims '283 portfolio modules' — neither matches the other (167 top-level portfolio/*.py files exist).

**Suggested fix:** Fix line 8 to match the body's 15-active figure; reconcile the 142 (CLAUDE.md) vs 283 (overview) module-map count.

### [P3 · doc-drift · conf 0.90] Dashboard route count stale: claims 45+10=55, actual 50+11=61
`/mnt/q/finance-analyzer/CLAUDE.md:165`

CLAUDE.md:165 claims '45 routes in app.py + 10 in house_blueprint.py (55 total). Last reconciled 2026-06-01'. Actual: 50 unique @app.route paths in dashboard/app.py (none commented out) and 11 @bp.route paths in dashboard/house_blueprint.py = 61 total. The doc even predicts its own staleness ('re-grep ... if this list looks stale') — it is stale; 6 routes have been added in ~9 days without reconciliation, and the enumerated route list below it is correspondingly incomplete.

**Suggested fix:** Re-grep and update to '50 routes in app.py + 11 in house_blueprint.py (61 total), reconciled 2026-06-10'; add the missing routes to the enumerated list.

### [P3 · doc-drift · conf 0.85] 'Applicable signals: crypto=19, stocks=15, metals=17' arithmetically impossible with 15 active signals
`/mnt/q/finance-analyzer/CLAUDE.md:257`

CLAUDE.md:257 claims applicable signal counts of crypto=19, stocks=15, metals=17. portfolio/signal_engine.py _compute_applicable_count (lines 1697-1729) counts only non-disabled signals from SIGNAL_NAMES, and only 15 signals are currently active — so crypto=19 cannot be produced by the code; the numbers date from the 21-active era. (The internal rules file .claude/rules/signals.md gives yet another set: 29/25/27.) Code comments are also self-inconsistent: signal_engine.py:1178-1179 say 'crypto has 30 signals', line 147 says metals _total_applicable=20.

**Suggested fix:** Recompute via _compute_applicable_count for one ticker per class and update CLAUDE.md (and .claude/rules/signals.md) with the actual values and an as-of date.

### [P3 · quality · conf 0.75] Playbook code sample instructs raw json.load(open("config.json")) — contradicts CLAUDE.md Critical Rule 4 and secret-hygiene practice
`/mnt/q/finance-analyzer/docs/TRADING_PLAYBOOK.md:327`

docs/TRADING_PLAYBOOK.md:327 tells the Layer 2 agent to run `config = json.load(open("config.json"))` to route Telegram messages. CLAUDE.md Critical Rule 4 mandates 'Atomic I/O only. Use file_utils ... Never raw json.loads(open(...).read())', and the established practice (memory: 'Grep config.json, never Read it') is to avoid pulling the full secrets file around because config.json contains live API keys (Mar 15 exposure incident). The snippet also leaks a file handle (no context manager) and encourages agents to treat raw config reads as the sanctioned pattern — agents copying the playbook verbatim may equally choose to Read config.json into LLM context to check notification.mode (line 243 'Check config.json → notification.mode').

**Suggested fix:** Change the snippet to `from portfolio.file_utils import load_json; config = load_json("config.json")` and add a note: never Read/cat config.json into context — grep specific keys.

## Architecture & repo structure

### [P1 → P2 · design · conf 0.95] Real-money production code lives in the runtime-state directory data/ (36 tracked .py files incl. 7,904-line metals_loop.py)
`/mnt/q/finance-analyzer/data/metals_loop.py:201`

data/ is simultaneously the mutable runtime-state directory (dozens of gitignored JSON/JSONL files, with .gitignore carrying ~40 hand-maintained per-file rules) and the home of 36 tracked executables, including the real-money metals loop (7,904 lines), crypto_loop.py, oil_loop.py, and the three swing traders. The loops do sys.path.insert(0, DATA_DIR) (metals_loop.py:201-208, crypto_loop.py:40, oil_loop.py:41) so they use bare-module sibling imports (metals_shared, metals_llm) — any .py file dropped into data/ can shadow a stdlib or site-packages module inside the real-money process, and data/ is not a package so nothing namespaces it. Code/state mixing also means state-cleanup or backup tooling aimed at data/ can touch production code, and gitignore maintenance stays a permanent error-prone chore (the comment at .gitignore:69-78 shows this is already a known pain point).

**Suggested fix:** Incremental migration, one module at a time: create portfolio/loops/ (or a top-level loops/ package) and move the small loops first (crypto_loop, oil_loop, the *_swing_config / *_warrant_refresh files), leaving 2-line import shims at the old data/ paths so scheduled-task .bat entrypoints keep working. Move metals_loop.py last, after the already-pending ARCH-18 metals_loop split. Update the .bat entrypoints in scripts/win/ in the same commit as each move. Blast radius per step: one scheduled task + its tests.

**Skeptic verdict (real):** Core facts confirmed: 36 tracked .py in data/, metals_loop.py (7,904 lines) inserts data/ at sys.path[0] (data/metals_loop.py:206-207) with bare sibling imports, so stdlib/site-packages shadowing inside the real-money process is possible. But the finding overstates: crypto_loop.py:40 and oil_loop.py:41 insert the repo ROOT (_HERE.parent), not DATA_DIR, and use `from data import ...` namespace imports — no shadow risk there. Long-standing documented design debt (2026-04-09 ARCH-12 comment, ARCH-18 split pending), no active malfunction; P2 not P1.

### [P1 → P2 · reliability · conf 0.90] RC kill-switch guard is an untracked markdown file while the installers it warns about remain live and runnable
`/mnt/q/finance-analyzer/scripts/win/RC_DISABLED_DO_NOT_REENABLE.md:1`

scripts/win/RC_DISABLED_DO_NOT_REENABLE.md documents that the RC scheduled tasks were disabled on purpose, that they were already accidentally re-enabled once before, and that install-rc-keepalive-task.ps1 / install-research-task.ps1 / install-signal-research-task.ps1 / rc-server-ensure.ps1 will silently re-create and enable them if run. But the guard file itself is untracked (visible in git status untracked list), so it does not exist in clones, worktrees, or for any agent that consults git-tracked docs — while all the dangerous installers remain tracked and fully functional. The protection is exactly as durable as one local file on one machine; the documented failure mode ('any aggregate setup that invokes the above') is still open.

**Suggested fix:** Commit the guard file, and make the guard mechanical instead of documentary: add a sentinel check at the top of each RC installer (e.g. `if (Test-Path "$PSScriptRoot/RC_DISABLED_DO_NOT_REENABLE.md") { Write-Host 'RC disabled on purpose — see guard file'; exit 1 }`) or move the four RC scripts into scripts/win/disabled/. Blast radius: ops scripts only, zero runtime impact.

**Skeptic verdict (real):** Confirmed: scripts/win/RC_DISABLED_DO_NOT_REENABLE.md is untracked (?? in git status) and no installer has a sentinel check (grep RC_DISABLED in *.ps1 = zero hits); install-rc-keepalive-task.ps1 and rc-server-ensure.ps1 would silently recreate/relaunch. Mitigation the finding missed: the two highest-cost paths have TRACKED guards — after-hours-research.bat:2-8 and signal-research.bat:2-8 carry DISABLED 2026-06-05 headers blocking the claude-spend tasks even in clones/worktrees. Remaining exposure (RC session churn/sidebar clutter) is an ops annoyance, not money or data loss: P2.

### [P1 → P2 · doc-drift · conf 0.90] CLAUDE.md operating facts have drifted from code and are fed directly into the Layer 2 trading agent
`/mnt/q/finance-analyzer/CLAUDE.md:251`

CLAUDE.md:251 states 'MIN_VOTERS = 3 (all asset classes)', but signal_engine.py:1180 sets MIN_VOTERS_METALS = 2 (deliberately lowered 2026-05-11). CLAUDE.md explicitly says Layer 2 sessions read this file for context, so the trading agent reasons with a wrong quorum rule for metals — the asset class trading real money. Additional drift in the same file: CLAUDE.md:374 claims '~5,994 tests across 242 files' vs 441 actual test files in tests/; the signal-count story is internally inconsistent (header says '80 modules registered, 59 disabled', the Architecture section says '21 active signals (65 modules)', signal_engine.py's own comment at line 1176 says '32-signal'); and pyproject.toml:4 still describes the system as '32-signal analysis'.

**Suggested fix:** Reconcile the numbers now (one doc commit), then make the volatile facts non-driftable: either generate a small 'live facts' block (MIN_VOTERS per asset class, active/disabled signal counts, test-file count) from code into CLAUDE.md via a script run by the existing PF-LogRotate or health-check task, or add a check script that greps code constants vs CLAUDE.md and appends to critical_errors.jsonl on mismatch — mirroring the route-count self-audit note already present at the dashboard section. Blast radius: docs + one cron script.

**Skeptic verdict (real):** All drift facts verified: CLAUDE.md:251 says MIN_VOTERS=3 all asset classes vs portfolio/signal_engine.py:1180 MIN_VOTERS_METALS=2 (deliberate, dated 2026-05-11); CLAUDE.md:374 claims 242 test files vs 440 actual; header '80 modules' vs line 132 '65 modules'; pyproject.toml:4 still says '32-signal'. Severity downgraded because the quorum is applied by Layer 1 in signal_engine.py:4256-4258 — Layer 2 consumes precomputed consensus and never recomputes voter floors, so the wrong fact degrades agent context rather than trade math; Layer 2 is also currently frozen (2026-06-06 token freeze). Doc-hygiene fix warranted: P2.

### [P1 → P2 · reliability · conf 0.85] Actively maintained test file for the metals swing trader sits outside pytest testpaths and never runs in the default suite
`/mnt/q/finance-analyzer/data/test_metals_swing_trader.py:1`

pyproject.toml:28 sets testpaths = ["tests"], and CLAUDE.md's testing commands all invoke `pytest tests/`. data/test_metals_swing_trader.py ("entry/exit logic, warrant selection, state management" for the real-money-adjacent swing trader) is therefore never collected by the default suite, yet it is still being actively maintained (touched in commits cca4fd50, a2f51d88, 0a1f53fe), so contributors plausibly believe it runs. tests/ does contain other metals_swing test files (entry gates, sizing, momentum), but the core trader test file is the one parked in data/. This is the classic silent-coverage-loss failure mode: regressions in entry/exit logic pass CI.

**Suggested fix:** Move the file to tests/test_metals_swing_trader.py; its existing sys.path.insert(0, dirname(__file__)) hack must become a data/-path insert (or better, fixtures in tests/conftest.py that put data/ on sys.path, which other swing tests presumably already need). Then run it once to triage any rot accumulated while uncollected. Blast radius: tests only.

**Skeptic verdict (real):** Confirmed: pyproject.toml:28 testpaths=["tests"], data/test_metals_swing_trader.py (55 tests, collects cleanly, no rot) is referenced by no runner/CI/script — only a SESSION_PROGRESS.md changelog line — yet was touched in cca4fd50 (2026-05-11). Real silent-coverage gap, but downgraded to P2: 9 sibling metals/crypto/oil swing test files in tests/ (entry gates, sizing, momentum exits, TP/SL, low-cash, persistence, notifications) DO run by default, and the file still passes collection, so the loss is partial overlap, not total blindness.

### [P2 · quality · conf 0.90] Untracked-and-unignored debris accumulates at repo root and in data/ (zz* files, _livecheck/ 12.7MB, doubled-prefix arxiv_arxiv_*.xml, '0' and 'nul' redirect artifacts, 8 phone-*.png)
`/mnt/q/finance-analyzer/zzproc.txt:1`

git status permanently shows untracked noise that git check-ignore confirms is not ignored: zzproc.txt, zztl.txt, _livecheck/ (incl. a 12.7MB heatmap.html), data/arxiv_*.xml. The arxiv files additionally show a producer bug signature — both arxiv_macro.xml (14 bytes, empty/error response) and arxiv_arxiv_macro.xml (doubled prefix, written a day later) exist, indicating a fetcher that prepends 'arxiv_' to an already-prefixed name and once persisted rate-limit/error bodies. Root also carries '0' and 'nul' (classic Windows/bash redirection artifacts), _tmp_check.ps1, _check_tasks.ps1, home_phone_full.png + 7 phone-*.png screenshots. This noise degrades every `git status` read — significant in a repo operated largely by agents that parse status output — and invites accidental staging.

**Suggested fix:** Delete the debris; add .gitignore entries for the patterns that recur (zz*.txt, _livecheck/, data/arxiv_*.xml, /0, /nul, /_tmp_*.ps1, /phone-*.png) with a dated comment; if the arxiv fetcher still exists in an agent prompt or skill, fix the double-prefix and have it write under an ignored scratch path. Blast radius: none at runtime.

### [P2 · design · conf 0.85] Swing-trader subsystem is triplicated (metals/crypto/oil) with near-identical helper layers — every fix must land three times
`/mnt/q/finance-analyzer/data/oil_swing_trader.py:51`

crypto_swing_trader.py and oil_swing_trader.py are structural clones: the same 13 module-level helpers (_now_utc, _load_state, _save_state, _log_decision, _log_trade, _signal_age_seconds, _extract_action/_confidence/_voters/_indicator/_regime ...) appear with identical signatures at nearly identical line offsets (crypto:49-181 vs oil:51-183), followed by a parallel Trader class. metals_swing_trader.py (3,487 lines) shares the same skeleton. The triplication extends to *_swing_config.py and *_warrant_refresh.py (359/363/443 lines). Recent commit cca4fd50 ('low-cash mode + TP/SL on warrant + persistence dedup (metals/crypto/oil)') demonstrates the cost: one feature, three implementations. Divergence here is a trading-correctness risk over time, not just maintenance overhead.

**Suggested fix:** Extract the shared helper layer first (lowest risk): a swing_common.py with the state-I/O and signal-extraction helpers, imported by all three traders — pure moves, no behavior change, covered by existing swing tests. Then, optionally, a SwingTraderBase class for the shared entry/exit scaffolding with per-asset subclasses. Do not unify the configs; they encode deliberate per-asset differences. Blast radius per step: the three swing traders + their tests.

### [P2 · quality · conf 0.85] Committed one-off scripts with live-trading side effects (hardcoded account + orderbook IDs) linger as permanent executables
`/mnt/q/finance-analyzer/data/gold_sell_debug.py:17`

data/gold_sell_debug.py / gold_sell_final.py / gold_sell_retry.py hardcode ACCOUNT_ID 1625505 and gold orderbook 2308943 and place/cancel real Avanza orders when executed — they are February session debris that has since been actively groomed (commit 4adeec2d fixed their json.load patterns, lending them false legitimacy). data/layer2_invoke.py is similar: a one-shot that appends a hardcoded Feb-2026 journal entry referencing tickers (META, SMCI, MU, NVDA) removed from the system in March. scripts/ has the same pattern: _layer2_eth_sell_2026_05_13.py, resolve_critical_errors_20260517.py, _resolve_critical_errors_20260518.py. In a repo where autonomous agents grep for existing functionality before writing code (a CLAUDE.md rule), stale runnable order-placing scripts are discoverable footguns.

**Suggested fix:** Delete the dated one-offs (git history preserves them — that is the archive). Going forward, adopt a convention in CLAUDE.md: one-off operational scripts live in an untracked scratch/ directory, or are deleted in the same session that used them. Blast radius: zero runtime; verify nothing imports them first (grep shows no importers for the gold_sell_* / layer2_invoke trio).

### [P2 · quality · conf 0.85] 30 orphaned worktree directories (326MB) not registered with git, plus empty finance-analyzer-improve/ leftover
`/mnt/q/finance-analyzer/.worktrees:?`

`git worktree list` reports only the main checkout, yet .worktrees/ contains 30 directories (auto-session-2026-03-05 through research-2026-05-21, 326MB) — all orphaned copies left behind after merges. The repo's own session notes record that worktrees missing the config.json symlink cause 30-50 spurious test failures, so stale trees are a known confusion source when an agent or human lands in the wrong directory; they also bloat backups and grep/glob scans (ripgrep respects .gitignore only if these are ignored — and several greps in this audit had to dodge them). finance-analyzer-improve/ at repo root is a now-empty leftover directory whose name suggests a past nested clone; `git -C` into it silently resolves to the parent repo, which is exactly the kind of ambiguity that misleads automation.

**Suggested fix:** Run `git worktree prune`, delete the 30 stale directories and finance-analyzer-improve/, and add a cleanup step to an existing maintenance task (PF-LogRotate is the natural host): remove .worktrees/* older than N days that are not in `git worktree list --porcelain`. Blast radius: none — verify no scheduled task points into a worktree path first (grep scripts/win/ for '.worktrees').

### [P2 · design · conf 0.80] signal_engine.py (4,698 lines) couples policy constants, regime overlays, consensus math, and dispatch in one module
`/mnt/q/finance-analyzer/portfolio/signal_engine.py:3293`

The module mixes at least four distinct concerns: tuning policy as module constants with invariants (MIN_VOTERS_* block at 1176-1204, DISABLED_SIGNALS, exclusion floors), a macro-window regime overlay (line 1134), per-signal dispatch, and the consensus/weighting pipeline culminating in generate_signal at line 3293 — a single function spanning roughly 1,000 lines through the MIN_VOTERS gating at 4251-4512. Nearly every signal-system change (and the file's dense archaeology of dated BUG-/P2- comments) lands in this one file, maximizing merge-conflict surface across the parallel worktree-based sessions this project uses, and making targeted testing of the consensus math depend on importing the whole engine.

**Suggested fix:** Incremental, behavior-preserving extraction in dependency order: (1) signal_policy.py — pure constants + invariant asserts (MIN_VOTERS_*, DISABLED_SIGNALS, floors), re-exported from signal_engine for compatibility; (2) consensus.py — the weighting/quorum math as pure functions taking votes in, verdict out, unit-testable in isolation; (3) leave dispatch + orchestration in signal_engine.py as the facade. Each step is a pure move with existing tests as the safety net. Blast radius: imports within portfolio/ + test patch targets (grep for 'signal_engine.MIN_VOTERS' style patches in tests first).

### [P2 · design · conf 0.75] 28 copy-paste scheduled-task installers and ~35 PF-* task names with no machine-readable inventory
`/mnt/q/finance-analyzer/scripts/win/install-rc-keepalive-task.ps1:1`

scripts/win/ holds 69 files including 28 install-*-task.ps1 scripts, one per scheduled task; docs reference ~35 distinct PF-* task names, including case-variant duplicates for the same task (PF-MSTRLoop vs PF-MstrLoop in different docs). There is no single manifest stating which tasks should exist, which are deliberately disabled (the RC group), and what schedule/command each runs — the inventory lives spread across CLAUDE.md prose, 28 installers, and the actual Windows task scheduler state, which already diverged once (RC re-enable incident). For a system whose stated #1 principle is 'the loop must run 100% of the time', task-state drift is invisible until something stops firing.

**Suggested fix:** Add a declarative manifest (scripts/win/tasks.psd1 or tasks.json: name, schedule, command, enabled, owner-doc) plus one driver script with install/verify modes; `verify` compares Get-ScheduledTask reality against the manifest and writes mismatches to data/critical_errors.jsonl, plugging into the existing fix-agent dispatcher. Migrate installers to thin manifest entries incrementally; keep old scripts as wrappers until each is migrated. Blast radius: ops tooling only.

### [P2 · design · conf 0.70] Two unrelated 'prophecy' modules in one repo: top-level prophecy/ package vs portfolio/prophecy.py
`/mnt/q/finance-analyzer/prophecy/__init__.py:3`

The new daily price-prediction package is named prophecy/ at repo root while the unrelated macro-beliefs store is portfolio/prophecy.py — the collision is acknowledged in the package docstring itself ('NOT to be confused with portfolio.prophecy'). With pytest pythonpath=["."] and the repo root on sys.path in all entrypoints, `import prophecy` and `from portfolio import prophecy` resolve to two different trading subsystems whose names differ only by qualification. The hazard is concrete for this codebase specifically: Layer 2 and fix agents generate code from natural-language context where 'the prophecy file/module' is ambiguous, and the data split (data/prophecy.json = beliefs vs data/prophecy_runs/ = predictions) repeats the same near-miss naming. A docstring warning is the weakest possible guard.

**Suggested fix:** Rename while it is cheapest — the new package is frozen and unpushed (per session notes). Either rename the package (e.g. prophecy_daily/ or predictions/) or, less invasively, rename the legacy module portfolio/prophecy.py -> portfolio/beliefs.py with a deprecation re-export shim, since it has only ~4 import sites (main.py:1574, reporting.py:770, signals/news_event.py:455). Blast radius: 4 imports + the prophecy scheduled task + grep for string references.

### [P3 · quality · conf 0.80] 35 adversarial-review session artifacts committed under data/ (adv-2026-05-08/, adv-2026-05-10/)
`/mnt/q/finance-analyzer/data/adv-2026-05-08/findings.json:?`

data/adv-2026-05-08/ and data/adv-2026-05-10/ contain 35 git-tracked markdown/JSON dumps from past adversarial-review runs (claude-*/codex-* per-area reviews, critique prompt templates, findings.json/txt). These are point-in-time review outputs, not runtime data, yet they live in the runtime-state directory and in version control — compounding the data/-as-junk-drawer problem (finding on data/ code mixing) and going stale the moment the reviewed code changes (the reviews predate, e.g., the 2026-06-06 P0 fixes).

**Suggested fix:** Move durable conclusions into docs/reviews/ (or just rely on the commits that fixed the findings) and `git rm` the directories; future review runs should write to an untracked path (the _livecheck/ or a scratch/ convention). Blast radius: zero — grep confirms nothing imports from these paths.
